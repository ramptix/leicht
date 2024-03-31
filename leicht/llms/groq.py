from __future__ import annotations

import os
import re
from types import ModuleType
from typing import Any, TYPE_CHECKING, Iterable, List, Literal, Optional, Tuple, Union
from typing_extensions import Mapping, TypedDict

import httpx

from .base import BaseLLM, BaseResponse
from ..prompts import get_prompt
from ..types import BasicLLMPayload, BasicLLMResponse

try:
    import orjson as json  # type: ignore
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
    def __init__(
        self, 
        data: dict, 
        *, 
        stream: bool, 
        pipe: Any,
        json_mode: bool
    ):
        self._stream = stream
        self._data = data
        self._pipe = pipe
        self._json_mode = json_mode

        if json_mode:
            sc = self._data['choices'][0]['message'] # shortcut
            sc['json'] = json.loads(sc['content'])

    def __iter__(self):
        def iterator():
            if not self._stream or self._json_mode:
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
        "_payload",
        "_json_mode",
        "_tools"
    )
    _headers: Headers
    _api_key: str
    _payload: dict # extra payload to append
    _json_mode: bool
    _tools: List[str]
    _tool_self: Optional[Groq]

    def __init__(
        self, 
        *, 
        api_key: Optional[str] = None, 
        json_mode: bool = False,
        tools: Optional[List[str]] = None,
        **extra_payload
    ):
        # if `api_key` is not provided, use the env
        self._api_key = api_key or os.environ["GROQ_API_KEY"]
        self._headers = {
            "Authorization": "Bearer %s" % self._api_key,
            "Content-Type": "application/json",
        }

        self._payload = extra_payload
        self._tools = tools or []

        if tools:
            self._tool_self = Groq(api_key=self._api_key, json_mode=False)

        self._json_mode = json_mode
        if json_mode:
            self._payload["response_format"] = {
                "type": "json_object"
            }

    def run(self, payload: GroqPayload, *, stream: Optional[bool] = None) -> GroqResponse:
        should_stream = payload["stream"] if stream is None else stream
        client = httpx.Client()
        json_payload = self._payload | payload

        if (self._json_mode or "response_format" in payload) and stream:
            raise TypeError(
                "This instance of Groq is in JSON mode, which doesn't support streaming."
            )

        if should_stream:
            pipe = client.stream(
                "POST",
                "https://api.groq.com/openai/v1/chat/completions",
                json=json_payload,
                headers=self._headers,
            )
            return GroqResponse({}, stream=True, pipe=pipe, json_mode=self._json_mode)
            
        else:
            r = client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                json=json_payload,
                headers=self._headers,
                timeout=None
            )
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as err:
                raise RuntimeError(f"\n\nResponse:\n{r.json()}") from err
            return GroqResponse(r.json(), stream=False, pipe=None, json_mode=self._json_mode)
    
    def get_function_call(self, text: str, payload: GroqPayload) -> Optional[List[Tuple[str, str]]]:
        # Assert if _tool_self is available
        # This also prevents the following code block from getting a type warning
        assert self._tool_self, "'tools' are not available for this Groq session."

        result = self._tool_self.run({
            **payload,
            "messages": [{
                "role": "user",
                "content": get_prompt(
                    "functions-groq",
                    tools="\n\n".join(self._tools),
                    most_commonly_used=self._tools[0],
                    text=text
                )
            }]
        })

        content: str = result['choices'][0]['message'].get('content', '')
        run_tools = not content.lstrip().lstrip('"\'').lower().startswith("null")

        return Groq.parse_fn_call(content) if run_tools else None
    
    @staticmethod
    def parse_fn_call(text: str) -> List[Tuple[str, str]]:
        calls = []

        for line in text.splitlines():
            r = re.findall(r'^((?!\d)[a-zA-Z0-9_]+)\((.*)\)(?:.*)$', line)

            if not r:
                continue

            calls.append(r[0])
        
        return calls

    def __repr__(self):
        return "Groq(api_key='gsk_***')"
