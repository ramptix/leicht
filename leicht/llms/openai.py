"""The OpenAI LLM."""

from typing import Optional

from .base import BaseLLM


class OpenAI(BaseLLM):
    """Represents an OpenAI.
    
    Args:
        api_key: 
    """
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
