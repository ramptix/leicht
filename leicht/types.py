from typing import Any, List, Literal, Optional, Type, Union
from typing_extensions import NotRequired, TypedDict

from .llms import BaseLLM

LLMType = Union[Literal["openai", "groq"], Type[BaseLLM]]

class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str


class BasicLLMPayload(TypedDict):
    model: Any
    temperature: float
    max_tokens: int
    top_p: float
    stream: bool
    stop: Optional[str]
    messages: List[Message]


class BasicLLMResponseDelta(TypedDict):
    content: NotRequired[str]


class BasicLLMResponseChoice(TypedDict):
    index: int
    delta: BasicLLMResponseDelta
    # logprobs
    finish_reason: Optional[Literal["stop", "length"]]


class BasicLLMResponse(TypedDict):
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    created: int
    model: str
    system_fingerprint: str
    choices: List[BasicLLMResponseChoice]
