from typing import Any, List, Literal, Optional, Type, Union
from typing_extensions import NotRequired, TypedDict

from .llms import BaseLLM

LLMType = Union[Literal["openai", "groq", "g4f", "hf"], BaseLLM, Type[BaseLLM]]


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class BasicLLMPayload(TypedDict):
    model: NotRequired[Any]
    temperature: float
    max_tokens: int
    top_p: float
    stream: bool
    stop: Optional[str]
    messages: List[Message]
    seed: Optional[int]


class BasicLLMResponseChoice(TypedDict):
    index: int
    # logprobs
    finish_reason: Optional[Literal["stop", "length"]]
    message: Message


class BasicLLMResponse(TypedDict):
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: int
    model: str
    system_fingerprint: str
    choices: List[BasicLLMResponseChoice]
