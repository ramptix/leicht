import os
from typing import Literal, Optional, overload

import httpx

from .base import BaseLLM, RunResult, StreamingDict


class Groq(BaseLLM):
    """Represents the Groq LLM."""
    __slots__ = (
        "api_key",
    )
    api_key: str

    def __init__(self, *, api_key: Optional[str] = None):
        # if `api_key` is not provided, use the env
        self.api_key = api_key or os.environ['GROQ_API_KEY']

    # We need to do it again for some reason.
    @overload
    def run(self, payload: dict, *, stream: Literal[True]) -> StreamingDict:
        ...

    @overload
    def run(self, payload: dict, *, stream: Literal[False]) -> dict:
        ...
    
    def run(self, payload: dict, *, stream: bool = False) -> RunResult:
        client = httpx.Client()
        respondse
