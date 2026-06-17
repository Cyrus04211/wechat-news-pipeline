"""Concrete agent providers for LLM-enhanced news collection."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from src.agents.news_agent_base import AgentDecision, AgentProvider

logger = logging.getLogger(__name__)

DEFAULT_CLOSEAI_BASE_URL = "https://api.openai-proxy.org/v1"
DEEPSEEK_V4_PRO_MAX_VISION_IMAGES = 4


@dataclass(frozen=True)
class LLMChatCompletion:
    """Parsed chat completion. Only ``content`` is used downstream."""

    content: str
    reasoning_trace: str = ""
    finish_reason: str = ""


class OpenAICompatibleAgentProvider(AgentProvider):
    """OpenAI-compatible chat-completions provider.

    CloseAI documents an OpenAI-compatible API surface, so the pipeline only
    needs a configurable base URL + model name.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "deepseek-v4-pro",
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        request_timeout_seconds: int = 180,
        concurrency: int = 4,
        max_retries: int = 4,
    ):
        self._api_key = (
            api_key
            or os.environ.get("CLOSEAI_API_KEY", "")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        self.model = model
        self.base_url = (
            base_url
            or os.environ.get("CLOSEAI_API_BASE", "")
            or os.environ.get("OPENAI_BASE_URL", "")
            or DEFAULT_CLOSEAI_BASE_URL
        ).rstrip("/")
        self.temperature = temperature
        self.request_timeout_seconds = request_timeout_seconds
        self.concurrency = max(int(concurrency), 1)
        self.max_retries = max(int(max_retries), 1)

    @property
    def provider_name(self) -> str:
        return f"openai-compatible:{self.model}"

    @property
    def is_available(self) -> bool:
        return bool(self._api_key)

    def analyze(self, prompt: str, system: str = "", context: Optional[dict] = None) -> AgentDecision:
        try:
            text = self._generate_text(prompt=prompt, system=system)
            return AgentDecision(decision=text, confidence=1.0, reasoning=text)
        except Exception as exc:
            logger.warning(f"OpenAI-compatible API call failed: {exc}")
            return AgentDecision(decision="error", confidence=0.0, reasoning=str(exc))

    def analyze_batch(
        self,
        prompts: list[str],
        system: str = "",
        context: Optional[dict] = None,
    ) -> list[AgentDecision]:
        return [self.analyze(prompt, system=system, context=context) for prompt in prompts]

    async def generate_many_texts_async(
        self,
        prompts: list[str],
        system: str = "",
        *,
        disable_thinking: bool = False,
        concurrency: Optional[int] = None,
    ) -> list[str]:
        semaphore = asyncio.Semaphore(max(int(concurrency or self.concurrency), 1))

        async def run_one(index: int, prompt: str) -> tuple[int, str]:
            async with semaphore:
                text = await asyncio.to_thread(
                    self._generate_text,
                    prompt,
                    system,
                    disable_thinking,
                )
                return index, text

        tasks = [run_one(index, prompt) for index, prompt in enumerate(prompts)]
        results = await asyncio.gather(*tasks)
        ordered = [""] * len(prompts)
        for index, text in results:
            ordered[index] = text
        return ordered

    async def generate_many_vision_texts_async(
        self,
        prompts: list[str],
        image_paths_list: list[list[str]],
        system: str = "",
        *,
        model: Optional[str] = None,
        max_images: int = DEEPSEEK_V4_PRO_MAX_VISION_IMAGES,
        images_first: bool = True,
        disable_thinking: bool = True,
    ) -> list[str]:
        if len(prompts) != len(image_paths_list):
            raise ValueError("prompts and image_paths_list must have the same length")
        semaphore = asyncio.Semaphore(self.concurrency)

        async def run_one(index: int, prompt: str, image_paths: list[str]) -> tuple[int, str]:
            async with semaphore:
                text = await asyncio.to_thread(
                    self._generate_vision_text,
                    prompt,
                    system,
                    image_paths,
                    model,
                    max_images,
                    images_first,
                    disable_thinking,
                )
                return index, text

        tasks = [
            run_one(index, prompt, image_paths)
            for index, (prompt, image_paths) in enumerate(zip(prompts, image_paths_list))
        ]
        results = await asyncio.gather(*tasks)
        ordered = [""] * len(prompts)
        for index, text in results:
            ordered[index] = text
        return ordered

    def _generate_vision_text(
        self,
        prompt: str,
        system: str = "",
        image_paths: Optional[list[str]] = None,
        model: Optional[str] = None,
        max_images: int = DEEPSEEK_V4_PRO_MAX_VISION_IMAGES,
        images_first: bool = True,
        disable_thinking: bool = True,
    ) -> str:
        model_name = model or self.model
        payload = {
            "model": model_name,
            "messages": self._build_multimodal_messages(
                prompt=prompt,
                system=system,
                image_paths=image_paths or [],
                max_images=max_images,
                images_first=images_first,
            ),
        }
        self._apply_deepseek_payload_options(payload, disable_thinking=disable_thinking)
        return self._post_chat_completion(payload)

    def _generate_text(
        self,
        prompt: str,
        system: str = "",
        disable_thinking: bool = False,
    ) -> str:
        payload = {
            "model": self.model,
            "messages": self._build_messages(prompt=prompt, system=system),
        }
        if disable_thinking:
            self._apply_deepseek_payload_options(payload, disable_thinking=True)
        return self._post_chat_completion(payload)

    def _post_chat_completion(self, payload: dict) -> str:
        model_name = str(payload.get("model", self.model))
        if self._supports_temperature(model_name):
            payload["temperature"] = self.temperature
        last_error = None
        completion: Optional[LLMChatCompletion] = None
        for attempt in range(self.max_retries):
            try:
                session = requests.Session()
                session.trust_env = False
                response = session.post(
                    f"{self.base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                        "Connection": "close",
                    },
                    json=payload,
                    timeout=self.request_timeout_seconds,
                )
                response.raise_for_status()
                data = response.json()
            except requests.RequestException as exc:
                last_error = exc
                response = getattr(exc, "response", None)
                if response is not None:
                    body = (response.text or "")[:800]
                    logger.warning(
                        "LLM HTTP %s on attempt %s/%s: %s body=%s",
                        response.status_code,
                        attempt + 1,
                        self.max_retries,
                        exc,
                        body,
                    )
                elif attempt == 0:
                    logger.warning(
                        "LLM request failed on attempt %s/%s: %s",
                        attempt + 1,
                        self.max_retries,
                        exc,
                    )
                if attempt >= self.max_retries - 1:
                    raise
                wait_seconds = min(12, 2 * (attempt + 1))
                if response is None:
                    logger.warning("Retrying in %ss.", wait_seconds)
                time.sleep(wait_seconds)
                continue

            if data.get("error"):
                raise RuntimeError(data["error"])

            choices = data.get("choices") or []
            if not choices:
                raise RuntimeError(f"Missing choices in response: {json.dumps(data, ensure_ascii=False)}")

            completion = self._parse_completion_from_choice(choices[0])
            if completion.content:
                self._log_reasoning_trace(completion)
                return completion.content

            last_error = RuntimeError(
                "LLM returned empty content"
                + (f" (finish_reason={completion.finish_reason})" if completion.finish_reason else "")
            )
            if attempt >= self.max_retries - 1:
                raise last_error
            wait_seconds = min(12, 2 * (attempt + 1))
            logger.warning(
                "LLM returned empty content on attempt %s/%s "
                "(finish_reason=%s, reasoning_trace_chars=%s). Retrying in %ss.",
                attempt + 1,
                self.max_retries,
                completion.finish_reason or "unknown",
                len(completion.reasoning_trace),
                wait_seconds,
            )
            time.sleep(wait_seconds)
        else:
            raise last_error or RuntimeError("LLM request failed without an exception")

    def _build_messages(self, prompt: str, system: str = "") -> list[dict[str, str]]:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_multimodal_messages(
        self,
        prompt: str,
        system: str = "",
        image_paths: Optional[list[str]] = None,
        *,
        max_images: int = DEEPSEEK_V4_PRO_MAX_VISION_IMAGES,
        images_first: bool = True,
    ) -> list[dict]:
        """Build OpenAI-compatible multimodal messages for deepseek-v4-pro vision.

        Community practice (CloudBase / reAPI): use base64 Data URLs, put images
        before the text prompt, and cap image count (pro series: <= 4).
        """
        messages: list[dict] = []
        if system:
            messages.append({"role": "system", "content": system})

        image_parts: list[dict] = []
        paths = image_paths or []
        limit = len(paths) if int(max_images) <= 0 else min(len(paths), max(0, int(max_images)))
        for image_path in paths[:limit]:
            encoded = self._encode_image_path(image_path)
            if encoded:
                image_parts.append({
                    "type": "image_url",
                    "image_url": {"url": encoded},
                })

        text_part = {"type": "text", "text": prompt}
        content_parts = (image_parts + [text_part]) if images_first else ([text_part] + image_parts)
        messages.append({"role": "user", "content": content_parts})
        return messages

    @staticmethod
    def _is_deepseek_model(model_name: str) -> bool:
        return "deepseek" in (model_name or "").lower()

    @staticmethod
    def _apply_deepseek_payload_options(payload: dict, *, disable_thinking: bool = True) -> None:
        """Apply DeepSeek V4 thinking controls for raw HTTP requests.

        DeepSeek only accepts ``reasoning_effort`` values ``high``/``max``.
        ``reasoning_effort: none`` triggers 400. The OpenAI SDK merges
        ``extra_body`` into the JSON body; with ``requests`` we must place
        ``thinking`` at the top level instead of nesting it under extra_body.
        """
        model_name = str(payload.get("model", ""))
        if not OpenAICompatibleAgentProvider._is_deepseek_model(model_name):
            return

        extra_body = payload.pop("extra_body", None)
        if isinstance(extra_body, dict):
            for key, value in extra_body.items():
                payload.setdefault(key, value)

        payload.pop("reasoning_effort", None)

        if disable_thinking:
            payload["thinking"] = {"type": "disabled"}
        else:
            payload.setdefault("thinking", {"type": "enabled"})
            payload.setdefault("reasoning_effort", "high")

    @staticmethod
    def _apply_deepseek_vision_payload_options(payload: dict, *, disable_thinking: bool = True) -> None:
        """Backward-compatible alias for tests and older call sites."""
        OpenAICompatibleAgentProvider._apply_deepseek_payload_options(
            payload,
            disable_thinking=disable_thinking,
        )

    @staticmethod
    def _encode_image_path(image_path: str) -> str:
        path = Path(image_path)
        if not path.exists() or not path.is_file():
            return ""
        mime, _ = mimetypes.guess_type(path.name)
        if not mime:
            mime = "image/jpeg"
        try:
            data = base64.b64encode(path.read_bytes()).decode("ascii")
        except OSError:
            return ""
        return f"data:{mime};base64,{data}"

    def _supports_temperature(self, model: Optional[str] = None) -> bool:
        if self.temperature is None:
            return False
        model_name = (model or self.model).lower()
        return not model_name.startswith("deepseek")

    @staticmethod
    def _parse_completion_from_choice(choice: dict) -> LLMChatCompletion:
        message = choice.get("message", {})
        content = OpenAICompatibleAgentProvider._flatten_content(
            message.get("content", choice.get("text", ""))
        )
        reasoning_trace = OpenAICompatibleAgentProvider._flatten_content(
            message.get("reasoning_content", "")
        )
        return LLMChatCompletion(
            content=content,
            reasoning_trace=reasoning_trace,
            finish_reason=str(choice.get("finish_reason", "") or ""),
        )

    def _log_reasoning_trace(self, completion: LLMChatCompletion) -> None:
        if not completion.reasoning_trace:
            return
        logger.debug(
            "LLM reasoning trace captured (%s chars, finish_reason=%s): %s",
            len(completion.reasoning_trace),
            completion.finish_reason or "unknown",
            completion.reasoning_trace[:240],
        )

    @staticmethod
    def _flatten_content(content) -> str:
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if item.get("type") == "text":
                        parts.append(str(item.get("text", "")))
                    elif "content" in item:
                        parts.append(str(item.get("content", "")))
                elif item:
                    parts.append(str(item))
            return "".join(parts).strip()
        return str(content or "").strip()


class ClaudeCodeAgentProvider(AgentProvider):
    """Batch export/import mode for Codex-side review."""

    def __init__(self, default_confidence: float = 0.6):
        self.default_confidence = default_confidence
        self._pending_batch: list[dict] = []

    @property
    def provider_name(self) -> str:
        return "claude-code"

    @property
    def is_available(self) -> bool:
        return True

    def analyze(self, prompt: str, system: str = "", context: Optional[dict] = None) -> AgentDecision:
        item = {
            "prompt": prompt,
            "system": system,
            "context": context or {},
            "queued_at": datetime.now().isoformat(),
        }
        self._pending_batch.append(item)
        return AgentDecision(
            decision="pending_claude_code",
            confidence=self.default_confidence,
            reasoning=f"[Pending Claude Code scoring] batch item #{len(self._pending_batch)}",
            metadata={"batch_index": len(self._pending_batch) - 1},
        )

    def analyze_batch(
        self,
        prompts: list[str],
        system: str = "",
        context: Optional[dict] = None,
    ) -> list[AgentDecision]:
        return [self.analyze(prompt, system, context) for prompt in prompts]

    def write_scoring_batch(self, output_path: str = "data/news/scoring_batch.json") -> int:
        if not self._pending_batch:
            logger.info("No pending items to export.")
            return 0

        count = len(self._pending_batch)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump({
                "exported_at": datetime.now().isoformat(),
                "item_count": count,
                "items": self._pending_batch,
            }, handle, ensure_ascii=False, indent=2)

        logger.info(f"Exported {count} items to {output_path}")
        self._pending_batch = []
        return count

    def read_scoring_results(self, path: str = "data/news/scoring_results.json") -> list[dict]:
        if not os.path.exists(path):
            logger.warning(f"Results file not found: {path}")
            return []
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data.get("results", data if isinstance(data, list) else [])


def create_agent_provider(
    provider: str = "auto",
    api_key: Optional[str] = None,
    model: str = "deepseek-v4-pro",
    base_url: Optional[str] = None,
    request_timeout_seconds: int = 180,
    concurrency: int = 4,
) -> AgentProvider:
    normalized = (provider or "auto").strip().lower()
    if normalized == "anthropic":
        logger.warning("Anthropic provider has been removed from the main pipeline. Using openai-compatible mode.")
        normalized = "closeai"

    if normalized == "claude-code":
        logger.info("Using Claude Code agent provider (batch review mode)")
        return ClaudeCodeAgentProvider()

    inline = OpenAICompatibleAgentProvider(
        api_key=api_key,
        model=model,
        base_url=base_url,
        request_timeout_seconds=request_timeout_seconds,
        concurrency=concurrency,
    )
    if normalized in {"closeai", "openai", "openai-compatible"}:
        if inline.is_available:
            logger.info(f"Using OpenAI-compatible provider: {inline.model} @ {inline.base_url}")
            return inline
        logger.warning("Inline provider unavailable. Falling back to Claude Code.")
        return ClaudeCodeAgentProvider()

    if inline.is_available:
        logger.info(f"Auto-detected OpenAI-compatible provider: {inline.model} @ {inline.base_url}")
        return inline

    logger.info("No inline API key detected. Using Claude Code agent provider.")
    return ClaudeCodeAgentProvider()
