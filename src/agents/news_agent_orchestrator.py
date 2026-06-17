"""Orchestrator for LLM summarization and topic classification."""

import logging
from typing import Optional

from src.agents.news_agent_base import AgentProvider
from src.agents.news_agent_provider import create_agent_provider

logger = logging.getLogger(__name__)

SUMMARY_SYSTEM_PROMPT = """# 公众号行业调研消息摘要 Agent

你负责阅读微信公众号行业调研分析消息，判断其是否为有效调研内容，并撰写可进入日报/周报的内容摘要。

## 输入说明

每条消息包含：
- 标题、digest、正文（若有）
- 配图（若有）：按图片0、图片1…顺序提供，必须结合列出的配图撰写摘要与逐图解读

## 噪声判断

对于明显不是行业调研分析的消息，例如：广告、登录提示、无正文的噪声，将 is_noise 设为 true。
噪声消息可不写摘要，article_summary 留空字符串。

## 摘要原则

1. 只写输入中能支持的事实、数据、主体、时间和结论，不得补充外部知识或编造。
2. 区分事实、研报观点、预测、传闻和管理层表述；不要把预测写成已经发生。
3. 数字、百分比、目标价、产能、价格和年份必须来自输入；不确定则省略或标注为“文中称/报告预计”。
4. 摘要要说明“发生了什么 → 影响哪条产业链/业务线 → 为什么值得关注”。
5. 若证据来自单一公众号、卖方研报、交流纪要或传闻，应在摘要末尾保留一句风险或证据限制。
6. 语言要通顺，避免堆砌标题党措辞；不要复述无关段落。

## 你的任务

为每条消息输出：

1. is_noise：是否为噪声
2. event_name：非噪声时输出中文事件名，应保留主体、动作和最关键变化；噪声时留空字符串
3. article_summary：非噪声时输出 120-220 字中文摘要；噪声时留空字符串
4. image_insights：对本次提示中列出的每一张配图输出解读
   - image_index：全局图片编号（与提示一致）
   - keep：是否与该消息实质相关、值得保留
   - explanation：一句中文说明图片内容及其与摘要的关联；若不保留，说明它为何只是广告、二维码、装饰、封面或无关图

本步骤不输出 topic。

## 输出格式

只返回 JSON 数组：

{"item_id": "<输入中的item_id>", "is_noise": false, "event_name": "<中文事件名>", "article_summary": "<中文内容摘要>", "image_insights": [{"image_index": 0, "keep": true, "explanation": "<中文>"}]}"""


CLASSIFICATION_SYSTEM_PROMPT = """# 公众号行业调研大话题分类 Agent

你负责一次性阅读本批次全部已完成内容摘要与事件名的非噪声微信公众号行业调研消息，通盘比较后根据每条消息的内容摘要和 event_name，自行归纳其所属的大话题板块。

## 分类原则

1. topic 是可复用的大话题板块名称，应该能承载多条相近消息，例如“存储涨价与产能紧张”“人形机器人供应链”等。
2. 对研报观点、预测、传闻、单一信源要降低置信度，并在风险字段说明证据限制。

## 输出字段

1. topic：中文大话题板块名
2. confidence：0 到 1 的小数
3. reasoning：一句中文，说明为什么归入该大话题
4. risk_or_caveat：一句中文，说明证据限制、预测/传闻属性或主要反例；没有则写空字符串

本步骤不读图，不要输出 image_insights / article_summary / event_name / is_noise。

## 输出格式

只返回 JSON 数组。推理模型须把 JSON 放在最终 content 中。

{"item_id": "<输入中的item_id>", "topic": "<中文大话题板块>", "confidence": 0.0, "reasoning": "<一句中文>", "risk_or_caveat": "<一句中文或空字符串>"}"""


class AgentOrchestrator:
    """Coordinates LLM summarization and classification for the news pipeline."""

    def __init__(
        self,
        provider: str = "auto",
        api_key: Optional[str] = None,
        model: str = "deepseek-v4-pro",
        base_url: Optional[str] = None,
        request_timeout_seconds: int = 180,
        concurrency: int = 4,
        enabled: bool = True,
    ):
        self.enabled = enabled
        self._provider = create_agent_provider(
            provider=provider,
            api_key=api_key,
            model=model,
            base_url=base_url,
            request_timeout_seconds=request_timeout_seconds,
            concurrency=concurrency,
        )
        self.stats: dict[str, int] = {}

    @property
    def provider(self) -> AgentProvider:
        return self._provider

    @property
    def can_classify_inline(self) -> bool:
        return self.enabled and self._provider.is_available

    def classify_items(
        self,
        items: list[dict],
        batch_size: int = 20,
        summary_concurrency: int = 32,
        vision_model: Optional[str] = None,
        vision_enabled: bool = True,
        vision_batch_size: int = 4,
        vision_images_first: bool = True,
        vision_disable_thinking: bool = True,
    ) -> list[dict]:
        from src.agents.vision_batch_size import VISION_API_BATCH_SIZE
        from src.collectors.news_llm_processor import TopicClassificationProcessor

        processor = TopicClassificationProcessor(
            self,
            batch_size=batch_size,
            summary_concurrency=summary_concurrency,
            model=self._provider.model if hasattr(self._provider, "model") else "deepseek-v4-pro",
            vision_model=vision_model,
            vision_enabled=vision_enabled,
            vision_batch_size=vision_batch_size or VISION_API_BATCH_SIZE,
            vision_images_first=vision_images_first,
            vision_disable_thinking=vision_disable_thinking,
        )
        return processor.process_batch(items)

    def get_summary(self) -> str:
        return (
            f"AgentOrchestrator(provider={self._provider.provider_name}, "
            f"enabled={self.enabled})"
        )
