import os
from types import ModuleType
from typing import Any, TYPE_CHECKING, Iterable, Literal, Optional, Union
from typing_extensions import Mapping, TypedDict

import httpx

from .base import BaseLLM, BaseResponse
from ..types import BasicLLMPayload, BasicLLMResponse

try:
    import orjson as json
except ImportError:
    import json

if TYPE_CHECKING:
    json: ModuleType

Model = Literal["mixtral-8x7b-32768", "gemma-7b-it"]
Headers = Mapping[str, str]


class GroqPayload(BasicLLMPayload):
    model: Model


class GroqResponseUsage(TypedDict):
    queue_time: float
    prompt_tokens: int
    prompt_time: float
    completion_tokens: int
    completion_time: float
    total_tokens: int
    total_times: float


class XGroq(TypedDict):
    id: str
    usage: GroqResponseUsage


class GroqResponseEnd(BasicLLMResponse):
    x_groq: XGroq

Response = Union[BasicLLMResponse, GroqResponseEnd]
StreamingDict = Iterable[Response]
RunResult = Union[StreamingDict, Response]

class GroqResponse(BaseResponse):
    def __init__(self, data: dict, *, stream: bool = False, pipe: Any):
        self._stream = stream
        self._data = data
        self._pipe = pipe

    def __iter__(self):
        def iterator():
            if not self._stream:
                raise TypeError("This is not a stream or streaming is completed.")

            pipe = self._pipe

            # We're adding this in case of out of bound errors
            last_d = {} # type: ignore
            text = ""

            with pipe as r:
                for line in r.iter_lines():
                    # len("data: ") = 6
                    # we'll remove the first 6 characters
                    raw = line[6:]

                    if raw == "[DONE]":
                        break
                    elif not raw:
                        continue

                    d = json.loads(raw)
                    yield d

                    if d.get('x_groq'):
                        last_d = d

                    text += d['choices'][0].get('content', '')

            last_d: GroqResponseEnd
            self._stream = False
            self._data = {
                **last_d,
                "choices": [{
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": text
                    }
                }]
            }
        return iterator()
    
    def __next__(self):
        return self.__iter__()


class Groq(BaseLLM):
    """Represents the Groq LLM."""

    __slots__ = (
        "_headers",
        "_api_key",
    )
    _headers: Headers
    _api_key: str

    def __init__(self, *, api_key: Optional[str] = None):
        # if `api_key` is not provided, use the env
        self._api_key = api_key or os.environ["GROQ_API_KEY"]
        self._headers = {
            "Authorization": "Bearer %s" % self._api_key,
            "Content-Type": "application/json",
        }

    def run(self, payload: GroqPayload, *, stream: Optional[bool] = None) -> GroqResponse:
        should_stream = payload["stream"] if stream is None else stream
        client = httpx.Client()

        if should_stream:
            pipe = client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=self._headers,
            )
            return GroqResponse({}, stream=True, pipe=pipe)
            
        else:
            r = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=payload,
                headers=self._headers,
            )
            return GroqResponse(r.json(), stream=False, pipe=None)

    def __repr__(self):
        return "Groq(api_key='gsk_***')"
