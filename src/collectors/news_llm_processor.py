"""LLM summarization and topic classification for WeChat research messages."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from src.agents.image_insight_models import ImageInsight
from src.agents.image_insight_utils import group_indices_into_batches, merge_image_insight_payloads
from src.agents.news_agent_orchestrator import CLASSIFICATION_SYSTEM_PROMPT, SUMMARY_SYSTEM_PROMPT
from src.agents.news_agent_provider import OpenAICompatibleAgentProvider
from src.agents.vision_batch_size import VISION_API_BATCH_SIZE
from src.domain.news_taxonomy import DEFAULT_TOPIC, normalize_event_name, normalize_topic
from src.utils.news_text import coerce_bool, coerce_list, coerce_str

logger = logging.getLogger(__name__)

DEFAULT_SUMMARY_CONCURRENCY = 32


@dataclass
class TopicClassification:
    item_id: str
    topic: str = ""
    event_name: str = ""
    is_noise: bool = False
    reasoning: str = ""
    confidence: float = 0.0
    risk_or_caveat: str = ""
    article_summary: str = ""
    image_insights: list[ImageInsight] = field(default_factory=list)
    llm_parsed: bool = False

    @property
    def industry(self) -> str:
        return ""

    @property
    def relevance(self) -> str:
        return ""

    @property
    def why_matters(self) -> str:
        return self.reasoning


IndustryClassification = TopicClassification


@dataclass
class SummarizationResult:
    summary: str = ""
    event_name: str = ""
    is_noise: bool = False
    image_insights: list[ImageInsight] = field(default_factory=list)
    llm_parsed: bool = False


class TopicClassificationProcessor:
    def __init__(
        self,
        orchestrator,
        batch_size: int = 12,
        summary_concurrency: int = DEFAULT_SUMMARY_CONCURRENCY,
        model: str = "deepseek-v4-pro",
        vision_model: Optional[str] = None,
        vision_enabled: bool = True,
        vision_batch_size: int = VISION_API_BATCH_SIZE,
        vision_images_first: bool = True,
        vision_disable_thinking: bool = True,
    ):
        self.orchestrator = orchestrator
        self.batch_size = batch_size
        self.summary_concurrency = max(int(summary_concurrency), 1)
        self.model = model
        self.vision_model = vision_model or model
        self.vision_enabled = vision_enabled
        self.vision_batch_size = max(1, int(vision_batch_size))
        self.vision_images_first = vision_images_first
        self.vision_disable_thinking = vision_disable_thinking

    def process_batch(self, items: list[dict]) -> list[dict]:
        if not self.orchestrator.can_classify_inline:
            raise RuntimeError(
                "LLM processing is required but the provider is unavailable. "
                "Set CLOSEAI_API_KEY or OPENAI_API_KEY and ensure llm.provider is openai-compatible."
            )
        return asyncio.run(self._process_batch_async(items))

    async def _process_batch_async(self, items: list[dict]) -> list[dict]:
        provider = self.orchestrator.provider
        if not isinstance(provider, OpenAICompatibleAgentProvider):
            raise RuntimeError("LLM processing requires an OpenAI-compatible provider")

        summarization_map = await self._run_summarization_async(provider, items)
        to_classify = [
            item for item in items
            if item["event_id"] in summarization_map
            and summarization_map[item["event_id"]].llm_parsed
            and not summarization_map[item["event_id"]].is_noise
        ]
        classified_map = await self._run_classification_async(
            provider,
            to_classify,
            summarization_map,
        )

        result = []
        for item in items:
            event_id = item["event_id"]
            enriched = dict(item)
            enriched.setdefault("agent_summary", "")
            enriched["agent_classified"] = False
            enriched["agent_summarized"] = False
            enriched["agent_scored"] = False
            enriched.setdefault("agent_image_insights", "[]")
            enriched.setdefault("selected_image_paths", "[]")

            summary_result = summarization_map.get(event_id)
            if summary_result and summary_result.llm_parsed:
                enriched["is_noise"] = summary_result.is_noise
                if summary_result.event_name:
                    enriched["event_name"] = normalize_event_name(
                        summary_result.event_name,
                        enriched.get("title", ""),
                    )
                if summary_result.summary:
                    enriched["agent_summary"] = summary_result.summary
                    enriched["agent_summarized"] = True
                image_paths = coerce_list(item.get("article_image_paths"))
                if summary_result.image_insights:
                    insights_payload = [
                        {
                            "image_index": insight.image_index,
                            "keep": insight.keep,
                            "explanation": insight.explanation,
                        }
                        for insight in summary_result.image_insights
                    ]
                    enriched["agent_image_insights"] = json.dumps(insights_payload, ensure_ascii=False)
                    enriched["selected_image_paths"] = json.dumps(
                        _select_image_paths(image_paths, summary_result.image_insights),
                        ensure_ascii=False,
                    )
            else:
                enriched["is_noise"] = False

            classification = classified_map.get(event_id)
            if classification and classification.llm_parsed:
                enriched = self._apply_classification(enriched, classification)
            elif summary_result and summary_result.is_noise:
                enriched["topic"] = DEFAULT_TOPIC
                enriched["event_name"] = normalize_event_name("", enriched.get("title", ""))
                enriched["industry"] = ""
                enriched["is_noise"] = True

            enriched["agent_scored"] = bool(
                enriched.get("agent_summarized") and enriched.get("agent_classified")
            )
            result.append(enriched)
        return result

    async def _run_summarization_async(
        self,
        provider: OpenAICompatibleAgentProvider,
        items: list[dict],
    ) -> dict[str, SummarizationResult]:
        if not items:
            return {}

        semaphore = asyncio.Semaphore(self.summary_concurrency)

        async def summarize_one(item: dict) -> tuple[str, SummarizationResult]:
            event_id = item["event_id"]
            async with semaphore:
                image_paths = coerce_list(item.get("article_image_paths"))
                if self.vision_enabled and image_paths:
                    result = await self._summarize_item_with_vision(provider, item, image_paths)
                else:
                    prompt = self._build_summary_single_item_prompt(item)
                    raw_text = await asyncio.to_thread(
                        provider._generate_text,
                        prompt,
                        SUMMARY_SYSTEM_PROMPT,
                        self.vision_disable_thinking,
                    )
                    result = self._parse_summary_item_response(
                        raw_text,
                        event_id,
                        expect_summary=True,
                    )
            return event_id, result

        pairs = await asyncio.gather(*(summarize_one(item) for item in items))
        summary_map: dict[str, SummarizationResult] = {}
        for event_id, result in pairs:
            if result.llm_parsed:
                summary_map[event_id] = result
        return summary_map

    async def _run_classification_async(
        self,
        provider: OpenAICompatibleAgentProvider,
        items: list[dict],
        summarization_map: dict[str, SummarizationResult],
    ) -> dict[str, TopicClassification]:
        classified_map: dict[str, TopicClassification] = {}
        if not items:
            return classified_map

        prompt = self._build_classification_batch_prompt(items, summarization_map)
        raw_responses = await provider.generate_many_texts_async(
            [prompt],
            system=CLASSIFICATION_SYSTEM_PROMPT,
            disable_thinking=self.vision_disable_thinking,
        )
        raw_text = raw_responses[0] if raw_responses else ""
        analyses = self._parse_batch_response(
            raw_text,
            [item["event_id"] for item in items],
            expect_summary=False,
        )
        for analysis in analyses:
            if analysis.llm_parsed:
                classified_map[analysis.item_id] = analysis
        return classified_map

    async def _summarize_item_with_vision(
        self,
        provider: OpenAICompatibleAgentProvider,
        item: dict,
        image_paths: list[str],
    ) -> SummarizationResult:
        indices = list(range(len(image_paths)))
        batches = group_indices_into_batches(indices, self.vision_batch_size)
        summary = ""
        event_name = ""
        is_noise = False
        merged_payload: list[dict[str, Any]] = []
        parsed_any = False

        for batch_index, batch_indices in enumerate(batches):
            batch_paths = [image_paths[index] for index in batch_indices]
            include_summary = batch_index == 0
            prompt = self._build_summary_single_prompt(
                item,
                batch_indices,
                include_summary=include_summary,
            )
            try:
                responses = await provider.generate_many_vision_texts_async(
                    [prompt],
                    [batch_paths],
                    system=SUMMARY_SYSTEM_PROMPT,
                    model=self.vision_model,
                    max_images=0,
                    images_first=self.vision_images_first,
                    disable_thinking=self.vision_disable_thinking,
                )
                raw_text = responses[0] if responses else ""
            except Exception as exc:
                logger.warning(
                    "Vision summarization failed for %s batch %s: %s",
                    item.get("event_id"),
                    batch_index,
                    exc,
                )
                continue

            parsed_result = self._parse_summary_item_response(
                raw_text,
                item["event_id"],
                expect_summary=include_summary,
            )
            if not parsed_result.llm_parsed:
                continue
            parsed_any = True
            if include_summary:
                summary = parsed_result.summary
                event_name = parsed_result.event_name
                is_noise = parsed_result.is_noise
            if parsed_result.image_insights:
                merged_payload = merge_image_insight_payloads(merged_payload, [
                    {
                        "image_index": insight.image_index,
                        "keep": insight.keep,
                        "explanation": insight.explanation,
                    }
                    for insight in parsed_result.image_insights
                ])

        if not parsed_any:
            return SummarizationResult()

        return SummarizationResult(
            summary=summary,
            event_name=event_name,
            is_noise=is_noise,
            image_insights=[
                ImageInsight(
                    image_index=int(entry["image_index"]),
                    keep=bool(entry.get("keep", False)),
                    explanation=str(entry.get("explanation", "") or ""),
                )
                for entry in merged_payload
            ],
            llm_parsed=True,
        )

    def _build_summary_single_item_prompt(self, item: dict) -> str:
        lines = [
            "请为以下 1 条公众号行业调研消息判断是否为噪声，生成事件名（event_name），并撰写内容摘要（article_summary）。",
            "只返回包含 1 个对象的 JSON 数组。",
            "对于明显不是行业调研分析的消息，例如：广告、登录提示、无正文的噪声，将 is_noise 设为 true。",
            "非噪声消息的 event_name 应保留主体、动作和最关键变化。",
            "无配图时 image_insights 返回 []。",
            "本步骤输出 item_id / is_noise / event_name / article_summary / image_insights，不要输出 topic。",
            "",
        ]
        lines.extend(self._format_item_lines(item, 0))
        return "\n".join(lines)

    def _build_classification_batch_prompt(
        self,
        items: list[dict],
        summarization_map: dict[str, SummarizationResult],
    ) -> str:
        lines = [
            f"请根据以下全部 {len(items)} 条已完成内容摘要与事件名，一次性把非噪声公众号行业调研消息归入你自行命名的大话题板块。",
            "输入中只有条目 ID、内容摘要和 event_name，不含标题、正文或其他原文。",
            "请通盘比较全部条目后归纳 topic，使相近消息共享可复用的大话题板块名。",
            "按消息的共同叙事、产业链矛盾、价格/订单/政策/技术主线归纳 topic。",
            "每条消息都要输出一个 topic。",
            "将所有结果放入一个 JSON 数组返回。只返回 JSON 数组，不要有 JSON 外的任何文字。",
            "若使用推理模型：JSON 数组必须出现在最终输出（content）中，不要只写在内部推理里。",
            "本步骤仅输出 item_id / topic / confidence / reasoning / risk_or_caveat，不读图，不要输出 image_insights / article_summary / event_name / is_noise。",
            "confidence 使用 0 到 1 的小数。",
            "",
        ]
        for index, item in enumerate(items):
            summary_result = summarization_map.get(item["event_id"])
            lines.extend(self._format_classification_item_lines(item, index, summary_result))
            lines.append("")
        return "\n".join(lines)

    def _build_summary_single_prompt(
        self,
        item: dict,
        image_indices: list[int],
        *,
        include_summary: bool,
    ) -> str:
        lines = [
            "请为以下 1 条公众号行业调研消息判断是否为噪声，生成事件名（event_name），并撰写内容摘要，同时解读列出的配图。",
            "只返回包含 1 个对象的 JSON 数组。",
            "对于明显不是行业调研分析的消息，例如：广告、登录提示、无正文的噪声，将 is_noise 设为 true。",
            "非噪声消息的 event_name 应保留主体、动作和最关键变化。",
        ]
        if include_summary:
            lines.append("本批次需同时输出 is_noise、event_name、article_summary 与 image_insights。")
        else:
            lines.append("本批次仅输出 image_insights，is_noise / event_name / article_summary 沿用首批结果或留空字符串。")
        lines.append("")
        lines.extend(self._format_item_lines(item, 0))
        lines.append("本批次需解读的配图编号：")
        for index in image_indices:
            lines.append(f"- 图片{index}")
        lines.append("image_insights 中的 image_index 必须使用上述全局编号。")
        return "\n".join(lines)

    def _format_classification_item_lines(
        self,
        item: dict,
        index: int,
        summary_result: Optional[SummarizationResult],
    ) -> list[str]:
        summary_text = ""
        event_name = ""
        if summary_result:
            summary_text = summary_result.summary
            event_name = summary_result.event_name
        if not summary_text and item.get("agent_summary"):
            summary_text = coerce_str(item.get("agent_summary"))
        if not event_name and item.get("event_name"):
            event_name = coerce_str(item.get("event_name"))
        return [
            f"--- 条目 {index} (ID: {item['event_id']}) ---",
            f"内容摘要：{summary_text or '（无摘要）'}",
            f"事件名：{event_name or '（无事件名）'}",
        ]

    def _format_item_lines(
        self,
        item: dict,
        index: int,
        *,
        include_summary_result: Optional[SummarizationResult] = None,
        include_classification: Optional[TopicClassification] = None,
    ) -> list[str]:
        lines = [
            f"--- 条目 {index} (ID: {item['event_id']}) ---",
            f"标题：{item.get('title', '')}",
            f"digest：{item.get('summary') or ''}",
            f"来源：{item.get('source_name', '')}",
            f"层级：Tier {item.get('source_tier', '')}",
        ]
        if include_summary_result and include_summary_result.summary:
            lines.append(f"内容摘要：{include_summary_result.summary}")
        if include_classification:
            lines.append(f"大话题板块：{include_classification.topic or DEFAULT_TOPIC}")
            lines.append(f"事件名：{include_classification.event_name}")
            lines.append(f"置信度：{include_classification.confidence:.2f}")
            lines.append(f"分类理由：{include_classification.reasoning}")
            if include_classification.risk_or_caveat:
                lines.append(f"证据限制：{include_classification.risk_or_caveat}")
        if item.get("source_url"):
            lines.append(f"链接：{item['source_url']}")
        content = coerce_str(item.get("article_content"))
        if content:
            lines.append(f"正文：{content}")
        else:
            lines.append("正文：（未抓取到，请结合标题与 digest 概括）")

        image_paths = coerce_list(item.get("article_image_paths"))
        if image_paths:
            lines.append(
                f"全文配图数量：{len(image_paths)}（全局编号为图片0..图片{len(image_paths) - 1}）"
            )
        return lines

    def _parse_batch_response(
        self,
        raw_text: str,
        item_ids: list[str],
        *,
        expect_summary: bool = False,
    ) -> list[TopicClassification]:
        data = self._extract_json_array(raw_text)
        if not isinstance(data, list):
            logger.warning("Failed to parse LLM classification response as JSON array:\n%s", raw_text)
            return [self._empty_analysis(item_id) for item_id in item_ids]

        parsed = []
        for entry in data:
            if isinstance(entry, dict):
                parsed.append(self._dict_to_analysis(entry, expect_summary=expect_summary))

        parsed_map = {analysis.item_id: analysis for analysis in parsed if analysis.item_id}
        return [parsed_map.get(item_id, self._empty_analysis(item_id)) for item_id in item_ids]

    def _parse_summary_response(
        self,
        raw_text: str,
        item_ids: list[str],
    ) -> dict[str, SummarizationResult]:
        data = self._extract_json_array(raw_text)
        if not isinstance(data, list):
            logger.warning("Failed to parse LLM summary response as JSON array:\n%s", raw_text)
            return {}

        result: dict[str, SummarizationResult] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            item_id = str(entry.get("item_id", "") or "")
            if not item_id:
                continue
            result[item_id] = self._entry_to_summary_result(entry)
        return result

    def _parse_summary_item_response(
        self,
        raw_text: str,
        item_id: str,
        *,
        expect_summary: bool,
    ) -> SummarizationResult:
        data = self._extract_json_array(raw_text)
        if not isinstance(data, list):
            logger.warning("Failed to parse LLM summary response as JSON array:\n%s", raw_text)
            return SummarizationResult()

        for entry in data:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("item_id", "") or "") not in ("", item_id):
                continue
            result = self._entry_to_summary_result(entry)
            if not expect_summary:
                result.summary = ""
            return result
        return SummarizationResult()

    def _entry_to_summary_result(self, entry: dict) -> SummarizationResult:
        insights = []
        for insight_entry in entry.get("image_insights") or []:
            if not isinstance(insight_entry, dict):
                continue
            try:
                image_index = int(insight_entry.get("image_index", 0))
            except (TypeError, ValueError):
                image_index = 0
            insights.append(ImageInsight(
                image_index=image_index,
                keep=bool(insight_entry.get("keep", False)),
                explanation=str(insight_entry.get("explanation", "") or ""),
            ))
        return SummarizationResult(
            summary=str(entry.get("article_summary", "") or "").strip(),
            event_name=normalize_event_name(
                entry.get("event_name") or entry.get("event") or "",
                entry.get("title", ""),
            ),
            is_noise=coerce_bool(entry.get("is_noise"), default=False),
            image_insights=insights,
            llm_parsed=True,
        )

    @staticmethod
    def _extract_json_array(raw_text: str):
        text = raw_text.strip()
        fence = chr(96) * 3
        if text.startswith(fence):
            text = re.sub(r"^" + fence + r"(?:json)?\s*", "", text, count=1)
            text = re.sub(r"\s*" + fence + r"\s*$", "", text, count=1)

        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError:
            pass

        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
        return None

    def _apply_classification(self, event: dict, analysis: TopicClassification) -> dict:
        enriched = dict(event)
        enriched["topic"] = analysis.topic or DEFAULT_TOPIC
        if not enriched.get("event_name"):
            enriched["event_name"] = analysis.event_name or normalize_event_name("", enriched.get("title", ""))
        enriched["industry"] = ""
        enriched["is_noise"] = False
        enriched["agent_classified"] = True
        enriched["agent_reasoning"] = analysis.reasoning
        enriched["agent_confidence"] = analysis.confidence
        enriched["agent_risk_or_caveat"] = analysis.risk_or_caveat
        enriched.setdefault("agent_image_insights", "[]")
        enriched.setdefault("selected_image_paths", "[]")
        return enriched

    def _dict_to_analysis(self, payload: dict, *, expect_summary: bool = False) -> TopicClassification:
        insights = []
        for entry in payload.get("image_insights") or []:
            if not isinstance(entry, dict):
                continue
            try:
                image_index = int(entry.get("image_index", 0))
            except (TypeError, ValueError):
                image_index = 0
            insights.append(ImageInsight(
                image_index=image_index,
                keep=bool(entry.get("keep", False)),
                explanation=str(entry.get("explanation", "") or ""),
            ))
        item_id = str(payload.get("item_id", "") or "")
        topic = normalize_topic(
            payload.get("topic")
            or payload.get("section")
            or payload.get("block")
            or payload.get("industry", "")
        )
        event_name = normalize_event_name(
            payload.get("event_name")
            or payload.get("event")
            or payload.get("title", "")
        )
        is_noise = coerce_bool(payload.get("is_noise"), default=False)
        return TopicClassification(
            item_id=item_id,
            topic=topic or DEFAULT_TOPIC,
            event_name=event_name,
            is_noise=is_noise,
            reasoning=str(payload.get("reasoning", "") or ""),
            confidence=_coerce_confidence(payload.get("confidence", 0.0)),
            risk_or_caveat=str(payload.get("risk_or_caveat", "") or ""),
            article_summary=str(payload.get("article_summary", "") or "") if expect_summary else "",
            image_insights=insights,
            llm_parsed=bool(item_id),
        )

    def _empty_analysis(self, item_id: str = "") -> TopicClassification:
        return TopicClassification(item_id=item_id, llm_parsed=False)

    @staticmethod
    def _chunk(items: list[dict], size: int):
        for start in range(0, len(items), size):
            yield items[start:start + size]


IndustryClassificationProcessor = TopicClassificationProcessor


def _coerce_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, confidence))


def _select_image_paths(image_paths: list[str], insights: list[ImageInsight]) -> list[str]:
    if not image_paths:
        return []
    if not insights:
        return list(image_paths)

    keep_indices = {insight.image_index for insight in insights if insight.keep}
    if not keep_indices:
        return []

    selected = []
    for index, path in enumerate(image_paths):
        if index in keep_indices:
            selected.append(path)
    return selected
