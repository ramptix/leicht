from typing import Any, Iterable, TypeVar, Union

from git import Optional

T = TypeVar("T")

class BaseLLM:
    """Represents a base LLM."""

    __slots__: Iterable[str]

    def __init__(self): ...

    def __repr__(self) -> str: ...

    def __call__(self, *args, **kwrags) -> Any: ...


class BaseResponse:
    __slots__ = ("_stream", "_data", "_pipe")
    _stream: bool
    _data: dict
    _pipe: Any

    def __init__(self, data: dict, *, stream: bool = False, pipe: Any): ...

    def copy(self) -> dict:
        return self._data.copy()

    def get(self, k: str, default: Optional[T] = None, /) -> Union[Any, T]:
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
        ...
