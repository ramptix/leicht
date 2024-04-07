from __future__ import annotations

import json
import os
import re
from contextlib import contextmanager
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
from ..prompts import get_prompt
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

    def __next__(self):
        return self.__iter__()

    def __repr__(self) -> str:
        return "GroqResponse(" + json.dumps(self._data) + ")"


class Groq(BaseLLM):
    """Represents the Groq LLM."""

    __slots__ = (
        "_headers",
        "_api_key",
        "_payload",
        "_json_mode",
        "_tools",
        "is_tool_self",
    )
    _headers: Headers
    _api_key: str
    _payload: dict  # extra payload to append
    _json_mode: bool
    _tools: List[str]
    _tool_self: Optional[Groq]
    _api_base = "https://api.groq.com/openai/v1"
    is_tool_self: bool

    def __init__(
        self,
        model: Model = "mixtral-8x7b-32768",
        *,
        api_key: Optional[str] = None,
        json_mode: bool = False,
        tools: Optional[List[str]] = None,
        tool_self: bool = False,
        **extra_payload,
    ):
        # if `api_key` is not provided, use the env
        self._api_key = api_key or os.environ["GROQ_API_KEY"]
        self._headers = {
            "Authorization": "Bearer %s" % self._api_key,
            "Content-Type": "application/json",
        }
        self._payload = {"model": model, **extra_payload}

        self.set(tools=tools or [])

        self._json_mode = json_mode
        if json_mode:
            self._payload["response_format"] = {"type": "json_object"}

        self.is_tool_self = tool_self

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

    def get_function_call(
        self, text: str, payload: GroqPayload
    ) -> Optional[List[Tuple[str, str]]]:
        # Assert if _tool_self is available
        # This also prevents the following code block from getting a type warning
        assert self._tool_self, "'tools' are not available for this Groq session."

        result = self._tool_self.run(
            {
                **payload,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Messages:\n"
                            + "\n".join(
                                [
                                    f"{m['role']}: {m['content']}"
                                    for m in payload["messages"]
                                ]
                            )
                            + get_prompt(
                                "functions-groq",
                                tools="\n\n".join(self._tools),
                                most_commonly_used=self._tools[0],
                                text=text,
                            )
                        ),
                    }
                ],
            }
        )

        content: str = result["choices"][0]["message"].get("content", "")
        run_tools = not content.lstrip().lstrip("\"'").lower().startswith("null")

        return Groq.parse_fn_call(content) if run_tools else None

    @staticmethod
    def parse_fn_call(text: str) -> List[Tuple[str, str]]:
        calls = []

        for line in text.splitlines():
            r = re.findall(r"^((?!\d)[a-zA-Z0-9_\\]+)\((.*)\)(?:.*)$", line)

            if not r:
                continue

            calls.append((
                r[0][0].replace('\\_', '_'), 
                r[0][1]
            ))

        return calls

    def __call__(
        self, payload: GroqPayload
    ) -> Union[GroqResponse, FunctionCallResponse]:
        """Runs a call.

        Args:
            payload (GroqPayload): The payload.
            stream (bool): Stream?
        """
        messages = payload["messages"]

        if self._tool_self:
            # tools are available
            fn = self.get_function_call(messages[-1]["content"], payload)
            if fn:
                return {"functions": fn}

        return self.run(payload, stream=payload["stream"])

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if k == "tools":
                self._tools = v
                self._tool_self = (
                    Groq(
                        self._payload["model"], 
                        api_key=self._api_key, 
                        json_mode=False,
                        tool_self=True
                    ) if v
                    else None
                )
            elif k == "fill_tools":
                self._tools = v
            elif k == "fill_tool_self":
                self._tool_self = v

        return self

    @contextmanager
    def notools(self, _m: bool = True):
        if _m:
            print("NOTOOLS")
            assert self._tool_self, "Tools are not available"

            tools, tool_self = self._tools, self._tool_self
            self.set(tools=[])
            yield
            self.set(fill_tools=tools, fill_tool_self=tool_self)
            print("TOOLS")
        else:
            yield

    def __repr__(self):
        return (
            "Groq(api_key='gsk_***'"
            + (", tool_self=True" if self.is_tool_self else "")
            + ")"
        )
