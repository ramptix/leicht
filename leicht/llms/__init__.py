"""LLMs."""

from .base import BaseLLM
from .groq import Groq
from .openai import OpenAI

try:
    import dotenv

    dotenv.load_dotenv()
except ImportError:
    dotenv = None  # unused. I just don't want to use 'pass'


__all__ = ("BaseLLM", "Groq", "OpenAI")
