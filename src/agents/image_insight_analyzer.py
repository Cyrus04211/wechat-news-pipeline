"""Batched vision calls for per-image insight analysis."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from src.agents.image_insight_utils import (
    merge_image_insight_payloads,
    parse_image_insight_response,
)
from src.agents.news_agent_provider import OpenAICompatibleAgentProvider
from src.agents.image_insight_models import ImageInsight
from src.agents.vision_batch_size import VISION_API_BATCH_SIZE
from src.utils.news_text import coerce_str

logger = logging.getLogger(__name__)

IMAGE_INSIGHT_SYSTEM_PROMPT = """# 公众号调研图片解读

请只分析用户给出的微信公众号配图，判断每张图是否与对应消息的大话题、事件和摘要实质相关。

## 输出要求

1. 只返回 JSON 数组，包含 1 个对象
2. 必须输出字段：item_id、image_insights
3. image_insights 中每项包含 image_index、keep、explanation
4. image_index 必须使用提示中给出的全局图片编号
5. explanation 用一句中文说明图片内容及其与事件的关联（或为何无关）
6. 不要输出 topic / reasoning / article_summary"""


class ImageInsightAnalyzer:
    def __init__(
        self,
        *,
        provider: Optional[OpenAICompatibleAgentProvider] = None,
        llm_config: Optional[dict[str, Any]] = None,
        vision_model: Optional[str] = None,
        vision_batch_size: int = VISION_API_BATCH_SIZE,
        vision_images_first: bool = True,
        vision_disable_thinking: bool = True,
    ):
        llm_config = llm_config or {}
        self.vision_batch_size = max(1, int(vision_batch_size))
        self.vision_images_first = bool(vision_images_first)
        self.vision_disable_thinking = bool(vision_disable_thinking)
        self.vision_model = vision_model or llm_config.get("vision_model") or llm_config.get("model", "deepseek-v4-pro")
        if provider is not None:
            self.provider = provider
        else:
            self.provider = OpenAICompatibleAgentProvider(
                api_key=llm_config.get("api_key"),
                model=llm_config.get("model", "deepseek-v4-pro"),
                base_url=llm_config.get("api_base"),
                request_timeout_seconds=int(llm_config.get("request_timeout_seconds", 600)),
                concurrency=int(llm_config.get("concurrency", 1)),
            )

    @property
    def enabled(self) -> bool:
        return isinstance(self.provider, OpenAICompatibleAgentProvider) and self.provider.is_available

    async def analyze_indices(
        self,
        *,
        item_id: str,
        title: str,
        source_name: str = "",
        content: str = "",
        industry: str = "",
        reasoning: str = "",
        image_paths: list[str],
        indices: list[int],
    ) -> list[ImageInsight]:
        if not indices or not image_paths:
            return []
        if not isinstance(self.provider, OpenAICompatibleAgentProvider):
            return []

        batches = self._group_indices(image_paths, indices)
        collected: list[ImageInsight] = []
        for batch_indices in batches:
            batch_paths = [image_paths[index] for index in batch_indices if 0 <= index < len(image_paths)]
            if not batch_paths:
                continue
            prompt = self._build_prompt(
                item_id=item_id,
                title=title,
                source_name=source_name,
                content=content,
                industry=industry,
                reasoning=reasoning,
                indices=batch_indices,
            )
            try:
                responses = await self.provider.generate_many_vision_texts_async(
                    [prompt],
                    [batch_paths],
                    system=IMAGE_INSIGHT_SYSTEM_PROMPT,
                    model=self.vision_model,
                    max_images=0,
                    images_first=self.vision_images_first,
                    disable_thinking=self.vision_disable_thinking,
                )
                raw_text = responses[0] if responses else ""
            except Exception as exc:
                logger.warning("Image insight batch failed for %s: %s", item_id, exc)
                continue

            parsed_entries = parse_image_insight_response(raw_text, item_id)
            for entry in parsed_entries:
                collected.append(ImageInsight(
                    image_index=int(entry["image_index"]),
                    keep=bool(entry.get("keep", False)),
                    explanation=str(entry.get("explanation", "") or ""),
                ))
        return collected

    async def analyze_missing(
        self,
        *,
        item_id: str,
        title: str,
        source_name: str = "",
        content: str = "",
        industry: str = "",
        reasoning: str = "",
        image_paths: list[str],
        existing_raw: Any,
    ) -> list[dict[str, Any]]:
        from src.agents.image_insight_utils import missing_image_insight_indices

        paths = list(image_paths)
        missing = missing_image_insight_indices(paths, existing_raw)
        if not missing:
            return parse_payload(existing_raw)

        new_insights = await self.analyze_indices(
            item_id=item_id,
            title=title,
            source_name=source_name,
            content=content,
            industry=industry,
            reasoning=reasoning,
            image_paths=paths,
            indices=missing,
        )
        payload = [
            {
                "image_index": insight.image_index,
                "keep": insight.keep,
                "explanation": insight.explanation,
            }
            for insight in new_insights
        ]
        return merge_image_insight_payloads(existing_raw, payload)

    def analyze_missing_sync(self, **kwargs: Any) -> list[dict[str, Any]]:
        return asyncio.run(self.analyze_missing(**kwargs))

    def _group_indices(self, image_paths: list[str], indices: list[int]) -> list[list[int]]:
        from src.agents.image_insight_utils import group_indices_into_batches

        valid = [index for index in sorted(set(indices)) if 0 <= index < len(image_paths)]
        return group_indices_into_batches(valid, batch_size=self.vision_batch_size)

    @staticmethod
    def _build_prompt(
        *,
        item_id: str,
        title: str,
        source_name: str,
        content: str,
        industry: str,
        reasoning: str,
        indices: list[int],
    ) -> str:
        lines = [
            "请仅分析下列配图，并返回 1 个 JSON 对象组成的数组。",
            f"item_id: {item_id}",
            f"标题：{title}",
        ]
        if source_name:
            lines.append(f"来源：{source_name}")
        if industry:
            lines.append(f"已分类主题：{industry}")
        if reasoning:
            lines.append(f"分类理由：{reasoning}")
        if content:
            lines.append(f"正文摘要：{coerce_str(content)[:1200]}")
        lines.append("需要解读的图片编号与顺序如下：")
        for index in indices:
            lines.append(f"- 图片{index}")
        lines.append(
            "image_insights 中的 image_index 必须使用上述全局编号。"
        )
        return "\n".join(lines)


def parse_payload(existing_raw: Any) -> list[dict[str, Any]]:
    from src.research.article_summary import parse_image_insights

    return parse_image_insights(existing_raw)
