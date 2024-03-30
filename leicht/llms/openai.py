"""The OpenAI LLM."""

import os
from typing import Optional

from .base import BaseLLM


class OpenAI(BaseLLM):
    """Represents an OpenAI.

    Args:
        api_key (str, optional): The API key.
    """

    __slots__ = ("api_key",)
    api_key: str

    def __init__(self, *, api_key: Optional[str] = None):
        # if `api_key` is not provided, use the env
        self.api_key = api_key or os.environ["OPENAI_API_KEY"]
