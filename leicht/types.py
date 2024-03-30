from typing import List, Literal, Optional
from typing_extensions import TypedDict


class Message(TypedDict):
    role: Literal["user", "assistant", "system"]
    content: str

class BasicLLMPayload(TypedDict):
    model: str
    temperature: int
    max_tokens: int
    top_p: int
    stream: bool
    stop: Optional[str]
    messages: List[Message]
