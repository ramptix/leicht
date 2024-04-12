from __future__ import annotations

import json
import os
from types import ModuleType
from typing import (
    Any,
    TYPE_CHECKING,
    Iterable,
    List,
    Literal,
    NotRequired,
    Optional,
    Tuple,
    Union,
)
from typing_extensions import Mapping, TypedDict

import httpx

from .base import BaseLLM, BaseResponse
from ._fc import get_function_call
from ..types import BasicLLMPayload, BasicLLMResponse

if TYPE_CHECKING:
    json: ModuleType

Model = Literal["mixtral-8x7b-32768", "gemma-7b-it", "llama2-70b-4096"]
Headers = Mapping[str, str]


class GroqPayload(BasicLLMPayload):
    model: NotRequired[Model]


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


class FunctionCallResponse(TypedDict):
    functions: List[Tuple[str, str]]


Response = Union[BasicLLMResponse, GroqResponseEnd]
StreamingDict = Iterable[Response]
RunResult = Union[StreamingDict, Response]


class GroqResponse(BaseResponse):
    def __init__(self, data: dict, *, stream: bool, pipe: Any, json_mode: bool):
        self._stream = stream
        self._data = data
        self._pipe = pipe
        self._json_mode = json_mode

        if json_mode:
            sc = self._data["choices"][0]["message"]  # shortcut
            sc["json"] = json.loads(sc["content"])

    def __iter__(self):
        def iterator():
            if not self._stream or self._json_mode:
                if self._data:
                    raise TypeError("Streaming is completed.")

                raise TypeError("This is not a stream or streaming is completed.")

            pipe = self._pipe

            # We're adding this in case of out of bound errors
            last_d = {}  # type: ignore
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

                    if d.get("x_groq"):
                        last_d = d

                    text += d["choices"][0].get("content", "")

            last_d: GroqResponseEnd
            self._stream = False
            self._data = {
                **last_d,
                "choices": [
                    {"index": 0, "message": {"role": "assistant", "content": text}}
                ],
            }

        return iterator()

    def dict(self):
        list(self.__iter__())
        return self._data

    def __next__(self):
        return self.__iter__()

    def __repr__(self) -> str:
        return "GroqResponse(" + json.dumps(self._data) + ")"


class Groq(BaseLLM):
    """Represents the Groq LLM.

    Args:
        model (Model): The model name.
        api_key (str, optional): API key. If not provided, uses env ``GROQ_API_KEY``
            instead.
        json_mode (bool): JSON mode? **BETA**
        tools (list[str], optional): List of tools in ``str``.
        **extra_payload: Extra payload.
    """

    __slots__ = (
        "_headers",
        "_api_key",
        "_payload",
        "_json_mode",
        "_tools",
    )
    _headers: Headers
    _api_key: str
    _payload: dict  # extra payload to append
    _json_mode: bool
    _tools: List[str]
    _api_base = "https://api.groq.com/openai/v1"

    def __init__(
        self,
        model: Model = "mixtral-8x7b-32768",
        *,
        api_key: Optional[str] = None,
        json_mode: bool = False,
        tools: Optional[List[str]] = None,
        **extra_payload,
    ):
        # if `api_key` is not provided, use the env
        self._api_key = api_key or os.environ["GROQ_API_KEY"]
        self._headers = {
            "Authorization": "Bearer %s" % self._api_key,
            "Content-Type": "application/json",
        }
        self._payload = {"model": model, **extra_payload}

        self._json_mode = json_mode
        if json_mode:
            self._payload["response_format"] = {"type": "json_object"}

    def run(
        self, payload: GroqPayload, *, stream: Optional[bool] = None
    ) -> GroqResponse:
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
                self._api_base + "/chat/completions",
                json=json_payload,
                headers=self._headers,
            )
            return GroqResponse({}, stream=True, pipe=pipe, json_mode=self._json_mode)

        else:
            r = client.post(
                self._api_base + "/chat/completions",
                json=json_payload,
                headers=self._headers,
                timeout=None,
            )
            try:
                r.raise_for_status()
            except httpx.HTTPStatusError as err:
                raise RuntimeError(f"\n\nResponse:\n{r.json()}") from err
            return GroqResponse(
                r.json(), stream=False, pipe=None, json_mode=self._json_mode
            )

    def __call__(
        self, payload: GroqPayload
    ) -> Union[GroqResponse, FunctionCallResponse]:
        """Runs a call.

        Returns `FunctionCallResponse` if applicable for a function call.

        Args:
            payload (GroqPayload): The payload.
            stream (bool): Stream?
        """
        functions = get_function_call(payload["messages"], tools=self._tools)

        if functions:
            return {"functions": functions}

        return self.run(payload, stream=payload["stream"])

    def __repr__(self):
        return "Groq(api_key='gsk_***')"
