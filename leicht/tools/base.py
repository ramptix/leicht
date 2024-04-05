import inspect
import re
from typing import Callable, List, Union

# regex patterns
REGEX_name = r"^((?!\d)[a-zA-Z0-9_]+)(.*)$" # /g
REGEX_ds_content = r"(?:\n\s*)?(.+)" # /gm
REGEX_args = r"(\32*(?!\d)[a-zA-Z0-9_]+\s*(.+): .+)+" # /gm

# types
Handler = Callable[..., Union[list, dict]]

class BaseTool:
    """Represents a base tool.

    Args:
        name (str): Name of the tool. Must match the function calling format in 
            Regex (``/^((?!\\d)[a-zA-Z0-9_]+)(.*)$/g``).
        description (str): Tool description.
    """

    __slots__ = ("name", "description", "handler")
    name: str
    description: str
    handler: Handler

    def __init__(
        self,
        name: str,
        description: str,
        handler: Handler
    ):
        if not re.match(REGEX_name, name):
            raise ValueError(f"Invalid function name {name!r}. (Not matching r{REGEX_name!r})")

        self.name = name
        self.description = description
        self.handler = handler

    @staticmethod
    def parse_args(fn: Handler) -> List[inspect.Parameter]:
        params = []

        for param in inspect.signature(fn).parameters.values():
            if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY):
                raise TypeError(
                    f"Unsupported parameter kind for param {str(param)!r}: {param.kind}"
                )

            params.append(param)

        return params

    @staticmethod
    def parse_docstring(docstring: str):
        ...
