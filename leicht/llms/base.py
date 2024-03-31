from types import ModuleType
from typing import Any, Iterable, TypeVar, Union, TYPE_CHECKING

try:
    import orjson as json
except ImportError:
    import json

if TYPE_CHECKING:
    json: ModuleType

T = TypeVar('T')

class BaseLLM:
    """Represents a base LLM."""

    __slots__: Iterable[str]

    def __init__(self): ...

    def __repr__(self) -> str: ...

class BaseResponse:
    __slots__ = ("_stream", "_data", "_pipe")
    _stream: bool
    _data: dict
    _pipe: Any

    def __init__(self, data: dict, *, stream: bool = False, pipe: Any):
        ...

    def copy(self) -> dict:
        return self._data.copy()

    def get(self, k: str, default: T, /) -> Union[Any, T]:
        return self._data.get(k, default)

    def items(self) -> Iterable:
        return self._data.items()

    def keys(self) -> Iterable:
        return self._data.keys()

    def values(self) -> Iterable:
        return self._data.values()

    def __getitem__(self, k: str) -> Any:
        return self._data[k]

    def __repr__(self) -> str:
        # using json.__name__ can suppress AttributeError's
        if json.__name__ == "json":
            return "GroqResponse(" + json.dumps(self._data) + ")"
        else:
            return (
                "GroqResponse(" +
                json.dumps(self._data, option=json.OPT_INDENT_2).decode('utf-8') + 
                ")"
            )
