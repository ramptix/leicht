from abc import ABC
from typing import Literal, Tuple, Iterable, Union, overload

StreamingDict = Iterable[dict]
RunResult = Union[dict, StreamingDict]

class BaseLLM(ABC):
    """Represents a base LLM (as an abstract class)."""
    __slots__: Tuple[str]

    @overload
    def run(self, payload: dict, *, stream: Literal[True]) -> StreamingDict:
        ...

    @overload
    def run(self, payload: dict, *, stream: Literal[False]) -> dict:
        ...

    def run(self, payload: dict, *, stream: bool = False) -> RunResult:
        ...
