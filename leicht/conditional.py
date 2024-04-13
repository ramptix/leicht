from typing import Optional
from .utils import clamp
from .llms._conditional import get_conditional


class Conditional:
    """Represents an LLM-based self-check conditional.

    Args:
        prompt (str): Prompt name or prompt content.
        __fillto (str): Where should the text be filled?
        __note (str, optional): Describe what this conditional statement does.
            Be brief.
        **kwargs: Keyword-only arguments for the prompt.
    """

    def __init__(
        self, prompt: str, __fillto: str, __note: Optional[str] = None, /, **kwargs
    ):
        self._note = __note
        self._prompt = prompt
        self._kwargs = kwargs
        self._fillto = __fillto.strip("{}")

    def check(self, text: str, /) -> bool:
        """Checks the conditional.

        Args:
            text (str): The text.
        """
        return get_conditional(self._prompt, **{**self._kwargs, self._fillto: text})

    def __repr__(self):
        return (
            f"Conditional({clamp(self._prompt)!r}" + f", {clamp(self._note, 81)!r})"
            if self._note
            else ")"
        )


class ConditionalCheckError(Exception):
    """Conditional check error."""
