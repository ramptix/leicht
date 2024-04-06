"""The OpenAI LLM."""

import os
from typing import Optional, TYPE_CHECKING

from .base import BaseLLM

try:
    from openai import OpenAI as _OpenAI
except ImportError:
    _OpenAI = None

if TYPE_CHECKING:
    _OpenAI: Optional[type]


class OpenAILike(BaseLLM):
    """Represents an OpenAI-like LLM.

    Args:
        api_key (str, optional): The API key.
    """

    __slots__ = ("api_key",)
    api_key: str

    def __init__(
        self,
        *,
        base_url: str,
        api_key: Optional[str] = None,
        api_key_path: Optional[str] = None,
        **kwargs,
    ):
        # if `api_key` is not provided, use the env
        if not _OpenAI:
            raise ImportError(
                "\n\nPlease install the `openai` package to use "
                "\x1b[38;2;229;192;123mOpenAI\x1b[0m(\x1b[38;2;97;175;239mBaseLLM\x1b[0m).\n"
                "  \x1b[38;2;97;175;239m$ \x1b[38;2;229;192;123mpip\x1b[0m install openai\n"
            )

        self.api_key = api_key or os.environ[api_key_path] if api_key_path else ""
        self.openai = _OpenAI(base_url=base_url, api_key=self.api_key, **kwargs)

    def chat_completions(self):
        response = self.openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "write a poem about a tree"}]
        )
        return response

    def __repr__(self) -> str:
        return "OpenAI(api_key='sk-***')"


class OpenAI(OpenAILike):
    def __init__(self):
        super().__init__(
            base_url="https://api.openai.com/v1/",
            api_key_path="OPENAI_API_KEY"
        )
