from abc import ABC, abstractclassmethod as abc_method


class BaseLLM(ABC):
    """Represents a base LLM (as an abstract class)."""

    @abc_method
    def run(self, payload: dict) -> dict:
        ...
