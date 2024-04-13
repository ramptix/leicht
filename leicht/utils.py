import re
from typing import Union, List

from .types import Message


def clamp(t: str, m: int = 21):
    """Clamp text.

    Args:
        t (str): The text.
        m (int): Max length.
    """
    return t[: min(len(t), m)] + ("…" if len(t) > m else "")


def prompt_alike(t: str):
    return re.match(r"^(?:[a-zA-Z\d\.-]+\/)?[a-zA-Z\d\.-]+$", t)


def msgs_to_text(msgs: Union[List[Message], str]) -> str:
    if isinstance(msgs, str):
        return msgs

    return "\n".join([f"{msg['role']}: {msg['content']}" for msg in msgs])
