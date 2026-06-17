"""
统一的大模型调用接口
支持 GPT、Gemini、Claude、Qwen、DeepSeek 等多个模型
提供结构化输出、网络搜索、错误重试等功能
"""

# TODO
# 1. gemini没有加timeout参数
# 2. qwen、deepseek没有加web_search

import os
import json
import asyncio
import time
from enum import Enum
from typing import Any, Dict, List, Optional, Union, Type, TypeVar
from dataclasses import dataclass, field
from collections import deque
from pydantic import BaseModel, ValidationError
from utils.envs import (
    OPENAI_API_KEY,
    QWEN_API_KEY,
    DEEPSEEK_API_KEY,
    GEMINI_API_KEY,
    CLOSEAI_API_KEY,
    DMXAPI_API_KEY,
)
from utils.envs import (
    QWEN_BASE_URL,
    DEEPSEEK_BASE_URL,
    CLOSEAI_OPENAI_BASE_URL,
    CLOSEAI_GEMINI_BASE_URL,
    CLOSEAI_ANTHROPIC_BASE_URL,
    CLOSEAI_DEEPSEEK_BASE_URL,
    DMXAPI_BASE_URL,
)

try:
    from openai import OpenAI, AsyncOpenAI
    from openai import omit

    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    omit = object()

try:
    from google import genai
    from google.genai import types

    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from anthropic import Anthropic

    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


# ==================== 类型定义 ====================


class ModelProvider(str, Enum):
    """支持的模型提供商"""

    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    GROK = "grok"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"


class ApiProvider(str, Enum):
    """API协议提供商"""

    OPENAI = "openai"
    GEMINI = "gemini"
    CLAUDE = "claude"
    GROK = "grok"
    QWEN = "qwen"
    DEEPSEEK = "deepseek"
    CLOSEAI = "closeai"
    DMXAPI = "dmxapi"


@dataclass
class LLMConfig:
    """
    LLM配置类

    参数说明:
        model_name (str): 要使用的模型名称, 例如 "gpt-5.1", "gemini-3-flash-preview", "text-embedding-3-large", "text-embedding-v4" 等.
        apiprovider (ApiProvider): API 协议提供商, 指定如何与 LLM 接口对接(如 openai, qwen, deepseek, gemini, closeai).
        modelprovider (Optional[ModelProvider]): 实际的模型厂商(如 openai, qwen, deepseek, gemini). 可选, 若apiprovider不是closeai则可省略.
        api_key (Optional[str]): 用于访问模型API的密钥. 如果不提供则需保证utils.envs中的密钥已导入.
        timeout (Optional[int]): 单次请求的超时时间(秒), 默认不限制.
        max_retries (int): 请求失败时的最大重试次数, 默认为 1(不重试).
        retry_delay (float): 每次重试之间的等待时间(秒), 默认为 1.0.
        max_requests_per_minute (int): 每分钟允许的最大请求数, 用于全局速率控制(默认 1000).
        max_concurrent_requests (int): 最大并发请求数, 用于限制同一时刻的请求数量(默认 50).
        extra_params (Dict[str, Any]): 其他自定义参数, 可用于扩展配置, 自动传递给API客户端.
    """

    model_name: str
    apiprovider: ApiProvider
    modelprovider: Optional[ModelProvider] = None
    api_key: Optional[str] = None
    timeout: Optional[int] = None
    max_retries: int = 1
    retry_delay: float = 1.0
    # 速率限制配置
    max_requests_per_minute: int = 1000
    time_window: int = 60
    max_concurrent_requests: int = 50
    # 其他自定义参数
    extra_params: Dict[str, Any] = field(default_factory=dict)


# ==================== 速率限制器 ====================


class RateLimiter:
    """速率限制器，使用滑动窗口算法"""

    def __init__(
        self, max_requests: int = 50, time_window: int = 60, max_concurrent: int = 10
    ):
        self.max_requests = max_requests
        self.time_window = time_window
        self.max_concurrent = max_concurrent
        self.request_times = deque()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.lock = asyncio.Lock()

    async def acquire(self):
        """获取请求许可"""
        await self.semaphore.acquire()
        async with self.lock:
            now = time.time()
            # 清理过期请求
            while self.request_times and self.request_times[0] < now - self.time_window:
                self.request_times.popleft()

            # 检查是否超过限制
            if len(self.request_times) >= self.max_requests:
                oldest_time = self.request_times[0]
                wait_time = self.time_window - (now - oldest_time) + 0.1
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    # 重新清理
                    while (
                        self.request_times
                        and self.request_times[0] < now - self.time_window
                    ):
                        self.request_times.popleft()

            self.request_times.append(time.time())

    def release(self):
        """释放请求许可"""
        self.semaphore.release()


# ==================== 工具函数 ====================


def _extract_object_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """从Pydantic模型的JSON Schema中提取对象schema"""
    return {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }


class UnifiedLLMClient:

    def __init__(self, config: LLMConfig):
        """
        参数:
            config (LLMConfig): LLM 客户端的配置对象,包含模型名称、API密钥、API提供商、实际的模型厂商、
                                最大请求数、最大并发数、超时、重试策略等参数。
        """
        self.config = config
        self.rate_limiter = RateLimiter(
            max_requests=config.max_requests_per_minute,
            time_window=config.time_window,
            max_concurrent=config.max_concurrent_requests,
        )
        self._client = None
        self._async_client = None
        self._init_clients()

    def _init_clients(self):
        """初始化客户端"""
        api_key = self.config.api_key or self._get_api_key_env_name()
        if not api_key:
            raise ValueError(
                f"API key not provided for {self.config.apiprovider.value}. "
                f"Set it in config or environment variable {self._get_api_key_env_name()}"
            )

        if self.config.apiprovider == ApiProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package is required for OpenAI provider")
            self.config.modelprovider = ModelProvider.OPENAI
            self._client = OpenAI(api_key=api_key, timeout=self.config.timeout)
            self._async_client = AsyncOpenAI(
                api_key=api_key, timeout=self.config.timeout
            )

        elif self.config.apiprovider == ApiProvider.QWEN:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package is required for Qwen provider")
            base_url = QWEN_BASE_URL
            self.config.modelprovider = ModelProvider.QWEN
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout,
                **self.config.extra_params,
            )
            self._async_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout,
                **self.config.extra_params,
            )

        elif self.config.apiprovider == ApiProvider.DEEPSEEK:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package is required for DeepSeek provider")
            base_url = DEEPSEEK_BASE_URL
            self.config.modelprovider = ModelProvider.DEEPSEEK
            self._client = OpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout,
                **self.config.extra_params,
            )
            self._async_client = AsyncOpenAI(
                api_key=api_key,
                base_url=base_url,
                timeout=self.config.timeout,
                **self.config.extra_params,
            )

        elif self.config.apiprovider == ApiProvider.CLOSEAI:
            if not self.config.modelprovider:
                raise ValueError(
                    "modelprovider parameter is required when apiprovider is closeai"
                )
            if self.config.modelprovider == ModelProvider.OPENAI:
                if not OPENAI_AVAILABLE:
                    raise ImportError("openai package is required for CloseAI provider")
                base_url = CLOSEAI_OPENAI_BASE_URL
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    **self.config.extra_params,
                )
                self._async_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    **self.config.extra_params,
                )

            elif self.config.modelprovider == ModelProvider.DEEPSEEK:
                if not OPENAI_AVAILABLE:
                    raise ImportError("openai package is required for CloseAI provider")
                base_url = CLOSEAI_DEEPSEEK_BASE_URL
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    **self.config.extra_params,
                )
                self._async_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    **self.config.extra_params,
                )

            elif self.config.modelprovider == ModelProvider.GEMINI:
                if not GEMINI_AVAILABLE:
                    raise ImportError(
                        "google-genai package is required for Gemini provider"
                    )
                base_url = CLOSEAI_GEMINI_BASE_URL
                self._client = genai.Client(
                    api_key=api_key,
                    vertexai=True,
                    http_options={"base_url": base_url},
                    **self.config.extra_params,
                )
                self._async_client = genai.Client(
                    api_key=api_key,
                    vertexai=True,
                    http_options={"base_url": base_url},
                    **self.config.extra_params,
                )

            elif self.config.modelprovider == ModelProvider.CLAUDE:
                if not CLAUDE_AVAILABLE:
                    raise ImportError(
                        "anthropic package is required for Claude provider"
                    )
                base_url = CLOSEAI_ANTHROPIC_BASE_URL
                self._client = Anthropic(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    **self.config.extra_params,
                )
                # TODO
                self._async_client = None

            else:
                raise ValueError(
                    f"Unsupported model provider: {self.config.modelprovider} in CloseAI api"
                )

        elif self.config.apiprovider == ApiProvider.GEMINI:
            if not GEMINI_AVAILABLE:
                raise ImportError(
                    "google-genai package is required for Gemini provider"
                )
            self.config.modelprovider = ModelProvider.GEMINI
            self._client = genai.Client(api_key=api_key, **self.config.extra_params)
            self._async_client = genai.Client(
                api_key=api_key, **self.config.extra_params
            )

        elif self.config.apiprovider == ApiProvider.DMXAPI:
            if not self.config.modelprovider:
                raise ValueError(
                    "modelprovider parameter is required when apiprovider is closeai"
                )
            if self.config.modelprovider == ModelProvider.OPENAI:
                if not OPENAI_AVAILABLE:
                    raise ImportError("openai package is required for CloseAI provider")
                base_url = DMXAPI_BASE_URL
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    **self.config.extra_params,
                )
                self._async_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    **self.config.extra_params,
                )

            elif self.config.modelprovider == ModelProvider.DEEPSEEK:
                if not OPENAI_AVAILABLE:
                    raise ImportError("openai package is required for CloseAI provider")
                base_url = DMXAPI_BASE_URL
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    **self.config.extra_params,
                )
                self._async_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    **self.config.extra_params,
                )

            elif self.config.modelprovider == ModelProvider.GEMINI:
                if not GEMINI_AVAILABLE:
                    raise ImportError(
                        "google-genai package is required for Gemini provider"
                    )
                base_url = DMXAPI_BASE_URL
                self._client = genai.Client(
                    api_key=api_key,
                    vertexai=True,
                    http_options={"base_url": base_url},
                    **self.config.extra_params,
                )
                self._async_client = genai.Client(
                    api_key=api_key,
                    vertexai=True,
                    http_options={"base_url": base_url},
                    **self.config.extra_params,
                )

            elif self.config.modelprovider == ModelProvider.CLAUDE:
                if not CLAUDE_AVAILABLE:
                    raise ImportError(
                        "anthropic package is required for Claude provider"
                    )
                base_url = DMXAPI_BASE_URL
                self._client = Anthropic(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    **self.config.extra_params,
                )
                # TODO
                self._async_client = None

            elif self.config.modelprovider == ModelProvider.GROK:
                if not OPENAI_AVAILABLE:
                    raise ImportError("openai package is required for Grok provider")
                base_url = DMXAPI_BASE_URL
                self._client = OpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    **self.config.extra_params,
                )
                self._async_client = AsyncOpenAI(
                    api_key=api_key,
                    base_url=base_url,
                    timeout=self.config.timeout,
                    **self.config.extra_params,
                )
            else:
                raise ValueError(
                    f"Unsupported model provider: {self.config.modelprovider} in DMX api"
                )

    def _get_api_key_env_name(self) -> str:
        """获取API key的环境变量名"""
        env_map = {
            ApiProvider.OPENAI: OPENAI_API_KEY,
            ApiProvider.GEMINI: GEMINI_API_KEY,
            ApiProvider.QWEN: QWEN_API_KEY,
            ApiProvider.DEEPSEEK: DEEPSEEK_API_KEY,
            ApiProvider.CLOSEAI: CLOSEAI_API_KEY,
            ApiProvider.DMXAPI: DMXAPI_API_KEY,
        }
        return env_map.get(self.config.apiprovider, None)

    def __prepare_response_format(self, response_format: Any) -> Any:
        """准备响应格式"""
        if self.config.modelprovider == ModelProvider.QWEN:
            schema = response_format.model_json_schema()
            return {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.get("title", "Response"),
                    "schema": _extract_object_schema(schema),
                },
                "strict": True,
            }

        elif self.config.modelprovider == ModelProvider.GEMINI:
            return {
                "response_mime_type": "application/json",
                "response_schema": response_format,
            }

        else:
            return response_format

    def _call_llm(
        self,
        messages: Union[str, List[Dict[str, str]]],
        response_format: Optional[Type[BaseModel]] = omit,
        web_search: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        """调用OpenAI Client"""
        if self.config.modelprovider == ModelProvider.OPENAI:
            params = {
                "model": self.config.model_name,
                "input": messages,
            }
            if response_format != omit:
                params["text_format"] = response_format
            if web_search:
                params["tools"] = [{"type": "web_search"}]
            if max_tokens:
                params["max_tokens"] = max_tokens
            if temperature:
                params["temperature"] = temperature
            if reasoning:
                if isinstance(reasoning, str):
                    params["reasoning"] = {"effort": reasoning}
                else:
                    params["reasoning"] = reasoning
            if kwargs:
                params.update(kwargs)
            response = self._client.responses.parse(**params)
            return response.output_text

        elif self.config.modelprovider == ModelProvider.QWEN:
            params = {
                "model": self.config.model_name,
                "messages": messages,
            }
            if response_format != omit:
                params["response_format"] = self.__prepare_response_format(
                    response_format
                )
            if max_tokens:
                params["max_tokens"] = max_tokens
            if temperature:
                params["temperature"] = temperature
            if web_search:
                params.setdefault("extra_body", {})
                params["extra_body"]["enable_search"] = True
            if reasoning is not None:
                params.setdefault("extra_body", {})
                params["extra_body"]["enable_thinking"] = reasoning
            if kwargs:
                params.update(kwargs)
            response = self._client.chat.completions.parse(**params)
            return response.choices[0].message.content

        elif self.config.modelprovider == ModelProvider.DEEPSEEK:
            params = {
                "model": self.config.model_name,
                "messages": messages,
            }
            if response_format != omit:
                params["response_format"] = self.__prepare_response_format(
                    response_format
                )
            if max_tokens:
                params["max_tokens"] = max_tokens
            if temperature:
                params["temperature"] = temperature
            if web_search:
                params["tools"] = [{"type": "web_search"}]
            if kwargs:
                params.update(kwargs)
            response = self._client.chat.completions.parse(**params)
            return response.choices[0].message.content

        elif self.config.modelprovider == ModelProvider.GEMINI:
            params = {
                "model": self.config.model_name,
                "contents": messages,
                "config": types.GenerateContentConfig(),
            }
            if response_format != omit:
                params["config"].response_mime_type = "application/json"
                params["config"].response_json_schema = (
                    response_format.model_json_schema()
                )
            if max_tokens:
                params["config"].max_output_tokens = max_tokens
            if temperature:
                params["config"].temperature = temperature
            if web_search:
                grounding_tool = types.Tool(google_search=types.GoogleSearch())
                params["config"].tools = [grounding_tool]
            if reasoning is not None:
                if isinstance(reasoning, str):
                    params["config"].thinking_config = types.ThinkingConfig(
                        thinking_level=reasoning
                    )
                else:
                    params["config"].thinking_config = types.ThinkingConfig(**reasoning)
            if kwargs:
                params.update(kwargs)
            response = self._client.models.generate_content(**params)
            return response.text

        elif self.config.modelprovider == ModelProvider.GROK:
            msgs = (
                messages
                if isinstance(messages, list)
                else [{"role": "user", "content": messages}]
            )

            params = {
                "model": self.config.model_name,
                "messages": msgs,
            }

            if max_tokens is not None:
                params["max_tokens"] = max_tokens
            if temperature is not None:
                params["temperature"] = temperature

            if response_format != omit:
                params["response_format"] = self.__prepare_response_format(
                    response_format
                )

            if kwargs:
                params.update(kwargs)

            params = json.loads(json.dumps(params, ensure_ascii=False))

            response = self._client.chat.completions.create(**params)
            return response.choices[0].message.content or ""

        elif self.config.modelprovider == ModelProvider.CLAUDE:
            msgs = (
                messages
                if isinstance(messages, list)
                else [{"role": "user", "content": messages}]
            )
            params = {
                "model": self.config.model_name,
                "messages": msgs,
                "max_tokens": max_tokens or 8096,
            }
            tools_list = []
            schema_name = None

            if web_search:
                tools_list.append({"type": "web_search_20250305", "name": "web_search"})

            if response_format != omit:
                schema = response_format.model_json_schema()
                schema_name = schema.get("title", "Response")
                tools_list.append(
                    {
                        "name": schema_name,
                        "description": "Output the structured response in this exact format.",
                        "input_schema": _extract_object_schema(schema),
                    }
                )

            if tools_list:
                params["tools"] = tools_list
                if response_format != omit and not web_search:
                    params["tool_choice"] = {"type": "tool", "name": schema_name}

            if reasoning is not None:
                params["thinking"] = (
                    {"type": "enabled", "budget_tokens": 10000}
                    if isinstance(reasoning, str)
                    else reasoning
                )

            if temperature is not None:
                params["temperature"] = temperature

            if kwargs:
                params.update(kwargs)

            response = self._client.messages.create(**params)

            if response_format != omit:
                for block in response.content:
                    if block.type == "tool_use" and block.name == schema_name:
                        return json.dumps(block.input, ensure_ascii=False)
            for block in response.content:
                if block.type == "text":
                    return block.text
            return ""

        else:
            raise ValueError(
                f"Unsupported model provider: {self.config.modelprovider} in OpenAI api"
            )

    def generate(
        self,
        messages: Union[str, List[Dict[str, str]]],
        response_format: Optional[Type[BaseModel]] = omit,
        web_search: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        """
        调用大模型进行文本生成.

        参数:
            messages (Union[str, List[Dict[str, str]]]): 输入的消息内容, 可以是字符串或消息字典列表.
            response_format (Optional[Type[BaseModel]]): 返回结果的Pydantic模型结构体(默认为omit, 直接文本输出).
            web_search (bool): 是否启用网络搜索增强(默认为False).
            max_tokens (Optional[int]): 最大生成的token数, None表示不限.
            temperature (Optional[float]): 采样温度, 决定输出多样性, 默认None使用模型默认值.
            **kwargs: 其他传递给底层模型API的自定义参数.

        返回:
            str: 生成的文本内容.
        """

        for attempt in range(self.config.max_retries):
            try:
                content = self._call_llm(
                    messages,
                    response_format,
                    web_search,
                    max_tokens,
                    temperature,
                    reasoning,
                    **kwargs,
                )
                return content
            except Exception as e:
                if attempt == self.config.max_retries - 1:
                    raise Exception(
                        f"API调用失败(已尝试{self.config.max_retries}次): {str(e)}"
                    )
                wait_time = self.config.retry_delay * (attempt + 1)
                time.sleep(wait_time)
        raise Exception("所有重试均失败")

    async def _call_llm_async(
        self,
        messages: Union[str, List[Dict[str, str]]],
        response_format: Optional[Type[BaseModel]] = omit,
        web_search: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:

        await self.rate_limiter.acquire()

        try:
            for attempt in range(self.config.max_retries):
                try:
                    if self.config.modelprovider == ModelProvider.OPENAI:
                        params = {
                            "model": self.config.model_name,
                            "input": messages,
                        }
                        if response_format != omit:
                            params["text_format"] = response_format
                        if web_search:
                            params["tools"] = [{"type": "web_search"}]
                        if max_tokens:
                            params["max_tokens"] = max_tokens
                        if reasoning:
                            params["reasoning"] = (
                                {"effort": reasoning}
                                if isinstance(reasoning, str)
                                else reasoning
                            )
                        if temperature:
                            params["temperature"] = temperature
                        if kwargs:
                            params.update(kwargs)
                        response = await self._async_client.responses.parse(**params)
                        return response.output_text

                    elif self.config.modelprovider == ModelProvider.QWEN:
                        params = {
                            "model": self.config.model_name,
                            "messages": messages,
                        }
                        if response_format != omit:
                            params["response_format"] = self.__prepare_response_format(
                                response_format
                            )
                        if max_tokens:
                            params["max_tokens"] = max_tokens
                        if temperature:
                            params["temperature"] = temperature
                        if reasoning is not None:
                            params.setdefault("extra_body", {})
                            params["extra_body"]["enable_thinking"] = reasoning
                        if web_search:
                            params.setdefault("extra_body", {})
                            params["extra_body"]["enable_search"] = True
                        if kwargs:
                            params.update(kwargs)
                        response = await self._async_client.chat.completions.parse(
                            **params
                        )
                        return response.choices[0].message.content

                    elif self.config.modelprovider == ModelProvider.DEEPSEEK:
                        params = {
                            "model": self.config.model_name,
                            "messages": messages,
                        }
                        if response_format != omit:
                            params["response_format"] = self.__prepare_response_format(
                                response_format
                            )
                        if max_tokens:
                            params["max_tokens"] = max_tokens
                        if temperature:
                            params["temperature"] = temperature
                        # TODO
                        # if web_search:
                        #     params["tools"] = [{"type": "web_search"}]
                        if kwargs:
                            params.update(kwargs)
                        response = await self._async_client.chat.completions.parse(
                            **params
                        )
                        return response.choices[0].message.content

                    elif self.config.modelprovider == ModelProvider.GEMINI:
                        params = {
                            "model": self.config.model_name,
                            "contents": messages,
                            "config": types.GenerateContentConfig(),
                        }
                        if response_format != omit:
                            params["config"].response_mime_type = "application/json"
                            params["config"].response_json_schema = (
                                response_format.model_json_schema()
                            )
                        if max_tokens:
                            params["config"].max_output_tokens = max_tokens
                        if temperature:
                            params["config"].temperature = temperature
                        if web_search:
                            grounding_tool = types.Tool(
                                google_search=types.GoogleSearch()
                            )
                            params["config"].tools = [grounding_tool]
                        if reasoning is not None:
                            if isinstance(reasoning, str):
                                params["config"].thinking_config = types.ThinkingConfig(
                                    thinking_level=reasoning
                                )
                            else:
                                params["config"].thinking_config = types.ThinkingConfig(
                                    **reasoning
                                )
                        if kwargs:
                            params.update(kwargs)
                        response = await self._async_client.aio.models.generate_content(
                            **params
                        )
                        return response.text

                    elif self.config.modelprovider == ModelProvider.GROK:
                        msgs = (
                            messages
                            if isinstance(messages, list)
                            else [{"role": "user", "content": messages}]
                        )

                        params = {
                            "model": self.config.model_name,
                            "messages": msgs,
                        }

                        if max_tokens is not None:
                            params["max_tokens"] = max_tokens
                        if temperature is not None:
                            params["temperature"] = temperature

                        if response_format != omit:
                            params["response_format"] = self.__prepare_response_format(
                                response_format
                            )

                        if kwargs:
                            params.update(kwargs)

                        response = await self._async_client.chat.completions.create(
                            **params
                        )
                        return response.choices[0].message.content or ""

                    elif self.config.modelprovider == ModelProvider.CLAUDE:
                        raise NotImplementedError("Claude async is not yet supported")

                    else:
                        raise ValueError(
                            f"Unsupported model provider: {self.config.modelprovider} in OpenAI api"
                        )

                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise Exception(
                            f"API调用失败(已尝试{self.config.max_retries}次): {str(e)}"
                        )
                    wait_time = self.config.retry_delay * (attempt + 1)
                    await asyncio.sleep(wait_time)

            raise Exception("所有重试均失败")

        finally:
            self.rate_limiter.release()

    async def generate_async(
        self,
        messages: Union[str, List[Dict[str, str]]],
        response_format: Optional[Type[BaseModel]] = omit,
        web_search: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        reasoning: Optional[Union[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> str:
        """
        调用大模型进行文本生成.为异步客户端.

        参数:
            messages (Union[str, List[Dict[str, str]]]): 输入的消息内容, 可以是字符串或消息字典列表.
            response_format (Optional[Type[BaseModel]]): 返回结果的Pydantic模型结构体(默认为omit, 直接文本输出).
            web_search (bool): 是否启用网络搜索增强(默认为False).
            max_tokens (Optional[int]): 最大生成的token数, None表示不限.
            temperature (Optional[float]): 采样温度, 决定输出多样性, 默认None使用模型默认值.
            **kwargs: 其他传递给底层模型API的自定义参数.

        返回:
            str: 生成的文本内容.
        """

        content = await self._call_llm_async(
            messages,
            response_format,
            web_search,
            max_tokens,
            temperature,
            reasoning,
            **kwargs,
        )
        return content

    async def close(self):
        """关闭客户端(如底层有异步close方法则await,否则直接调用)"""
        if self._client:
            close_method = getattr(self._client, "close", None)
            if callable(close_method):
                result = close_method()
                if asyncio.iscoroutine(result):
                    await result
        if self._async_client:
            close_method = getattr(self._async_client, "close", None)
            if callable(close_method):
                result = close_method()
                if asyncio.iscoroutine(result):
                    await result

    def generate_embedding(self, textlist: List[str]) -> List[List[float]]:
        """
        调用大模型进行文本嵌入.
        参数:
            textlist (List[str]): 输入的文本列表.
        返回:
            List[List[float]]: 文本嵌入列表.
        """
        if self.config.modelprovider == ModelProvider.OPENAI:
            response = self._client.embeddings.create(
                input=textlist, model=self.config.model_name
            )
            return [item.embedding for item in response.data]

        elif self.config.modelprovider == ModelProvider.QWEN:
            """qwen模型存在一次最大输入size为10的限制, 因此需要分批处理"""
            embeddings = []
            batch_size = 10
            for i in range(0, len(textlist), batch_size):
                batch = textlist[i : i + batch_size]
                response = self._client.embeddings.create(
                    input=batch, model=self.config.model_name
                )
                res = json.loads(response.model_dump_json())
                batch_embeddings = [item["embedding"] for item in res["data"]]
                embeddings.extend(batch_embeddings)
            return embeddings

        else:
            raise ValueError(
                f"Unsupported model provider: {self.config.modelprovider} for embedding generation"
            )

    async def generate_embedding_async(self, textlist: List[str]) -> List[List[float]]:
        """
        调用大模型进行文本嵌入.为异步客户端.
        """
        await self.rate_limiter.acquire()
        try:
            for attempt in range(self.config.max_retries):
                try:
                    if self.config.modelprovider == ModelProvider.OPENAI:
                        response = await self._async_client.embeddings.create(
                            input=textlist, model=self.config.model_name
                        )
                        return [item.embedding for item in response.data]

                    elif self.config.modelprovider == ModelProvider.QWEN:
                        """qwen模型存在一次最大输入size为10的限制, 因此需要分批处理"""
                        embeddings = []
                        batch_size = 10
                        for i in range(0, len(textlist), batch_size):
                            batch = textlist[i : i + batch_size]
                            response = await self._async_client.embeddings.create(
                                input=batch, model=self.config.model_name
                            )
                            res = json.loads(response.model_dump_json())
                            batch_embeddings = [
                                item["embedding"] for item in res["data"]
                            ]
                            embeddings.extend(batch_embeddings)
                        return embeddings
                    else:
                        raise ValueError(
                            f"Unsupported model provider: {self.config.modelprovider} for embedding generation"
                        )

                except Exception as e:
                    if attempt == self.config.max_retries - 1:
                        raise Exception(
                            f"API调用失败(已尝试{self.config.max_retries}次): {str(e)}"
                        )
                    wait_time = self.config.retry_delay * (attempt + 1)
                    await asyncio.sleep(wait_time)

            raise Exception("所有重试均失败")

        finally:
            self.rate_limiter.release()


if __name__ == "__main__":

    # ==================== 测试代码 ====================
    # ==================== 1.文本生成 ==================
    # class Answer(BaseModel):
    #     title: str
    #     content: str

    # config = LLMConfig(
    #     model_name="gpt-5.2",
    #     apiprovider="openai",
    #     modelprovider="openai",
    # )
    # llmclient = UnifiedLLMClient(config=config)
    # messages = "what's the biggest news today in finance?"
    # res = llmclient.generate(messages, Answer, web_search=True)
    # print(res)

    # # ==================== 2.文本嵌入 ==================
    # config = LLMConfig(
    #     model_name="text-embedding-v4",
    #     apiprovider="qwen",
    #     modelprovider="qwen",
    # )
    # llmclient = UnifiedLLMClient(config=config)
    # textlist = ["hello", "world", "this is a test"]
    # embeddings = llmclient.generate_embedding(textlist)
    # print(len(embeddings), len(embeddings[0]))

    config = LLMConfig(
        model_name="grok-4.1",
        apiprovider="dmxapi",
        modelprovider="grok",
    )
    llmclient = UnifiedLLMClient(config=config)
    messages = "今天上海天气怎么样？"
    res = llmclient.generate(messages)
    print(res)
