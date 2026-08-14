"""LLM client封装，集成vLLM推理."""

from typing import List, Optional, Dict, Any, Union
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI

from app.config import get_settings


class LLMClient:
    """LLM客户端基类."""

    def __init__(self, model: Optional[str] = None):
        self.settings = get_settings()
        self.model = model or self.settings.VLLM_MODEL

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        聊天接口.

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            **kwargs: 额外参数

        Returns:
            助手回复文本
        """
        raise NotImplementedError


class vLLMClient(LLMClient):
    """vLLM推理客户端."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        初始化vLLM客户端.

        Args:
            model: 模型名称
            api_base: API base URL
            temperature: 温度参数
            max_tokens: 最大token数
        """
        super().__init__(model)
        settings = get_settings()
        self.api_base = api_base or f"http://{settings.VLLM_HOST}:{settings.VLLM_PORT}/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 使用LangChain的OpenAI兼容接口
        self._llm = ChatOpenAI(
            model=self.model,
            api_key="EMPTY",  # vLLM不需要API key
            base_url=self.api_base,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs
    ) -> str:
        """
        聊天接口.

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            **kwargs: 额外参数

        Returns:
            助手回复文本
        """
        from langchain_openai import ChatOpenAI

        # Create a new client for each request to avoid state issues
        llm = ChatOpenAI(
            model=self.model,
            api_key="EMPTY",
            base_url=self.api_base,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg["content"]

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        response = llm.invoke(langchain_messages)
        return response.content

    def chat_with_json(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        带JSON结构化输出的聊天接口.

        Args:
            messages: 消息列表
            json_schema: 期望的JSON schema

        Returns:
            解析后的JSON响应
        """
        from openai import OpenAI

        client = OpenAI(api_key="EMPTY", base_url=self.api_base)

        # 转换消息格式
        chat_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                chat_messages.append({"role": "system", "content": msg["content"]})
            elif role == "user":
                chat_messages.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                chat_messages.append({"role": "assistant", "content": msg["content"]})

        response = client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            response_format={"type": "json_object"} if json_schema else {"type": "text"},
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        import json
        content = response.choices[0].message.content
        if json_schema:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw": content}
        return {"text": content}


class VLMClient(vLLMClient):
    """视觉语言模型客户端，继承vLLMClient."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_base: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        初始化VLM客户端.

        Args:
            model: 模型名称（默认为qwen-vl-plus）
            api_base: API base URL
            temperature: 温度参数
            max_tokens: 最大token数
        """
        super().__init__(model, api_base, temperature, max_tokens)

    def chat_with_image(
        self,
        messages: List[Dict[str, Any]],
        **kwargs
    ) -> str:
        """
        带图像的聊天接口.

        Args:
            messages: 消息列表，支持图片内容
                    [{"role": "user", "content": [
                        {"type": "text", "text": "..."},
                        {"type": "image_url", "image_url": {"url": "..."}}
                    ]}]

        Returns:
            助手回复文本
        """
        from openai import OpenAI

        client = OpenAI(api_key="EMPTY", base_url=self.api_base)

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=kwargs.get("temperature", self.temperature),
            max_tokens=kwargs.get("max_tokens", self.max_tokens),
        )

        return response.choices[0].message.content


# 全局实例
_llm_client: LLMClient | None = None


def get_llm_client() -> LLMClient:
    """获取LLM客户端单例."""
    global _llm_client
    if _llm_client is None:
        _llm_client = vLLMClient()
    return _llm_client


_vlm_client: VLMClient | None = None


def get_vlm_client() -> VLMClient:
    """获取VLM客户端单例（避免每次调用都新建客户端）。"""
    global _vlm_client
    if _vlm_client is None:
        _vlm_client = VLMClient()
    return _vlm_client


class MiniMaxClient(LLMClient):
    """MiniMax API 客户端，用于SFT数据生成."""

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ):
        """
        初始化MiniMax客户端.

        Args:
            model: 模型名称，默认 abab6.5s-chat
            api_key: MiniMax API Key
            temperature: 温度参数
            max_tokens: 最大token数
        """
        settings = get_settings()
        super().__init__(model or settings.MINIMAX_MODEL)
        self.api_key = api_key or settings.MINIMAX_API_KEY
        self.base_url = "https://api.minimax.chat/v1"
        self.temperature = temperature
        self.max_tokens = max_tokens

        self._llm = ChatOpenAI(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def chat(self, messages: List[Dict[str, str]], **kwargs) -> str:
        """
        聊天接口.

        Args:
            messages: 消息列表 [{"role": "user", "content": "..."}]
            **kwargs: 额外参数

        Returns:
            助手回复文本
        """
        langchain_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg["content"]

            if role == "system":
                langchain_messages.append(SystemMessage(content=content))
            elif role == "user":
                langchain_messages.append(HumanMessage(content=content))
            elif role == "assistant":
                langchain_messages.append(AIMessage(content=content))

        response = self._llm.invoke(langchain_messages)
        return response.content

    def chat_with_json(
        self,
        messages: List[Dict[str, str]],
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        带JSON结构化输出的聊天接口.

        Args:
            messages: 消息列表
            json_schema: 期望的JSON schema

        Returns:
            解析后的JSON响应
        """
        from openai import OpenAI

        client = OpenAI(api_key=self.api_key, base_url=self.base_url)

        chat_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            if role == "system":
                chat_messages.append({"role": "system", "content": msg["content"]})
            elif role == "user":
                chat_messages.append({"role": "user", "content": msg["content"]})
            elif role == "assistant":
                chat_messages.append({"role": "assistant", "content": msg["content"]})

        response = client.chat.completions.create(
            model=self.model,
            messages=chat_messages,
            response_format={"type": "json_object"} if json_schema else {"type": "text"},
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        content = response.choices[0].message.content
        if json_schema:
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw": content}
        return {"text": content}


_minimax_client: MiniMaxClient | None = None


def get_minimax_client() -> MiniMaxClient:
    """获取MiniMax客户端单例."""
    global _minimax_client
    if _minimax_client is None:
        _minimax_client = MiniMaxClient()
    return _minimax_client
