import os
from types import ModuleType
from typing import TYPE_CHECKING, Iterable, Literal, Optional, Union
from typing_extensions import Mapping, overload

import httpx

from .base import BaseLLM
from ..types import BasicLLMPayload

try:
    import orjson as json
except:
    import json

if TYPE_CHECKING:
    json: ModuleType

Headers = Mapping[str, str]
StreamingDict = Iterable[dict]
RunResult = Union[StreamingDict, dict]

class GroqPayload(BasicLLMPayload):
    model: Union[Literal["mistral-8x7b-32768", "gemma-7b-it"], str]

class Groq(BaseLLM):
    """Represents the Groq LLM."""
    __slots__ = (
        "_headers",
        "api_key",
    )
    _headers: Headers
    api_key: str

    def __init__(self, *, api_key: Optional[str] = None):
        # if `api_key` is not provided, use the env
        self.api_key = api_key or os.environ['GROQ_API_KEY']
        self._headers = {
            "Authorization": "Bearer %s" % api_key,
            "Content-Type": "application/json"
        }

    # We need to do it again for some reason.
    @overload
    def run(self, payload: GroqPayload, *, stream: Literal[True]) -> StreamingDict:
        ...

    @overload
    def run(self, payload: GroqPayload, *, stream: Literal[False]) -> dict:
        ...
    
    def run(self, payload: GroqPayload, *, stream: Optional[bool] = None) -> RunResult:
        should_stream = payload['stream'] if stream is None else stream
        client = httpx.Client()

        if should_stream:
            pipe = client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload
            )

            with pipe as r:
                for line in r.iter_lines():
                    yield json.loads(line)
        
        else:
            r = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload
            )
            return r.json()

    def __repr__(self):
        return "Groq(api_key='gsk_***')"
