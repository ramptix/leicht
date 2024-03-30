from abc import ABC
from typing import Iterable

class BaseLLM(ABC):
    """Represents a base LLM (as an abstract class)."""
    __slots__: Iterable[str]

    def __repr__(self) -> str:
        ...
