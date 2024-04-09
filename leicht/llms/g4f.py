from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Dict, List, Literal, Optional, Tuple, Union
from typing_extensions import Mapping, NotRequired

import httpx

from .base import BaseLLM
from .groq import Groq
from ..types import BasicLLMPayload
from ..prompts import get_prompt

# types
Model = Union[
    Literal["gpt-3.5-turbo", "gpt-3.5-turbo-0613", "gpt-3.5-turbo-16k"], str
]  # more models need to be tested


class Payload(BasicLLMPayload):
    model: NotRequired[Model]
    provider: NotRequired[str]


class GPT4Free(BaseLLM):
    __slots__ = (
        "base_url",
        "_payload",
        "_headers",
        "_tool_self",
        "_tools",
        "is_tool_self",
    )
    base_url: str
    _payload: Dict[str, Any]
    _headers: Mapping[str, str]
    _tool_self: Optional[GPT4Free]
    _tools: List[str]
    is_tool_self: bool

    def __init__(
        self,
        model: Model = "gpt-3.5-turbo",
        *,
        provider: str = "FlowGpt",
        base_url: str = "https://aweirddev-g4f.hf.space/v1",
        tools: Optional[List[str]] = None,
        tool_self: bool = False,
        **extra_payload,
    ):
        self.base_url = base_url
        self._payload = {"model": model, "provider": provider, **extra_payload}
        self._headers = {"Authorization": "Bearer xxx"}

        tools = tools or []
        self.set(tools=tools)

        self.is_tool_self = tool_self

    def run(self, payload: Payload, *, stream: Optional[bool] = None):
        client = httpx.Client()
        r = client.post(
            self.base_url + "/chat/completions",
            json={**payload, **self._payload},
            params={"id": time.time_ns()},
            headers=self._headers,
        )
        r.raise_for_status()
        return r.json()

    def get_function_call(
        self, text: str, payload: Payload
    ) -> Optional[List[Tuple[str, str]]]:
        # Check groq.py
        assert self._tool_self, "'tools' are not avaiable for this GPT74Free session."

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

    def __call__(self, payload: Payload):
        messages = payload["messages"]

        if self._tool_self:
            # tools are available
            fn = self.get_function_call(messages[-1]["content"], payload)
            if fn:
                return {"functions": fn}

        return self.run(payload)

    def set(self, **kwargs):
        for k, v in kwargs.items():
            if k == "tools":
                self._tools = v
                self._tool_self = (
                    GPT4Free(
                        model=self._payload["model"],
                        provider=self._payload["provider"],
                        base_url=self.base_url,
                        tool_self=True,
                    )
                    if v
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
            tools, tool_self = self._tools, self._tool_self
            self.set(tools=[])
            yield
            self.set(fill_tools=tools, fill_tool_self=tool_self)
        else:
            yield

    # @staticmethod
    # def from_hf(channel: Literal["g4f", "stable"], **kwargs) -> GPT4Free:
    #    if channel == "g4f":
    #        return GPT4Free()
    #    else:
    #        g4f = GPT4Free(
    #            model="mistral-7b-instruct-v0.2",
    #            base_url="https://aweirddev-mistral-7b-instruct-v0-2-leicht.hf.space",
    #            provider="",
    #            **kwargs,
    #        )
    #        return g4f
