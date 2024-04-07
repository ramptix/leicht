import ast
import inspect
import re
from types import EllipsisType
from typing import (
    Callable,
    Generic,
    List,
    Literal,
    Mapping,
    ParamSpec,
    Tuple,
    TypeVar,
    Union,
)
from typing_extensions import TypedDict

# regex patterns
REGEX_name = r"^((?!\d)[a-zA-Z0-9_]+)(.*)$"  # /g
REGEX_ds_content = r"(?:\n\s*)?(?!\s)(.+)"  # /gm
REGEX_param_desc = r"^(?!\d)[a-zA-Z0-9_]+\s*(?:\(.+\))?:\s*(.+)$"  # /g
REGEX_parse_args = r"^(?:((?!\d)[a-zA-Z0-9_]+)=)?(.*)$"  # /g

# constants
builtin_types = [bool, bytes, dict, int, float, str]
invalid_types = [list, set, dict, tuple, EllipsisType]

# types
BuiltinTypes = Union[bool, bytes, dict, float, str]
Handler = Callable[..., Union[list, dict, TypedDict]]

P = ParamSpec("P")
T = TypeVar("T", list, dict, TypedDict)


class DocstringResult(TypedDict):
    description: str
    args: List[str]


class BaseTool(Generic[P, T]):
    """Represents a base tool.

    Args:
        name (str): Name of the tool. Must match the function calling format in
            Regex (``/^((?!\\d)[a-zA-Z0-9_]+)(.*)$/g``).
        description (str): Tool description.
    """

    __slots__ = ("name", "handler", "description", "params", "docstring")
    name: str
    description: str
    params: List[inspect.Parameter]
    docstring: DocstringResult
    handler: Callable[P, T]

    def __init__(self, name: str, handler: Callable[P, T]):
        if not re.match(REGEX_name, name):
            raise ValueError(
                f"Invalid function name {name!r}. (Not matching r{REGEX_name!r})"
            )

        self.name = name
        self.handler = handler
        self.params = BaseTool.get_args(self.handler)
        self.docstring = BaseTool.parse_docstrings(self.handler.__doc__ or "")
        self.description = self.docstring["description"]

    @property
    def prompt(self) -> str:
        return BaseTool.mix_make_prompt(self.name, self.params, self.docstring)

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T: ...

    # Static Methods
    @staticmethod
    def get_args(fn: Handler) -> List[inspect.Parameter]:
        params = []
        values = inspect.signature(fn).parameters.values()

        for param in list(values):
            if param.kind not in (param.POSITIONAL_OR_KEYWORD, param.POSITIONAL_ONLY):
                raise TypeError(
                    f"Unsupported parameter kind for param {str(param)!r}: {param.kind}"
                )

            if param.annotation in invalid_types:
                raise TypeError(
                    f"{param.annotation!r} is an invalid type because it's one of {invalid_types!r}.\n"
                    f"Try custom ones or built-in's like {builtin_types!r} instead."
                )

            params.append(param)

        return params

    @staticmethod
    def parse_docstrings(docstring: str) -> DocstringResult:
        lines: List[str] = re.findall(REGEX_ds_content, docstring)
        state: Literal["null", "args"] = "null"

        desc = ""
        args = []

        for line in lines:
            if line.rstrip().lower() == "args:":
                state = "args"
                continue

            if state == "null":
                desc += line
            elif state == "args":
                args.append(re.findall(REGEX_param_desc, line)[0])

        return {"description": desc, "args": args}

    @staticmethod
    def mix_make_prompt(
        name: str, params: List[inspect.Parameter], docstring: DocstringResult
    ) -> str:
        """Mix ``params`` and ``docstring`` to make a prompt."""
        assert docstring["description"], "Description unprovided."
        assert len(params) == len(
            docstring["args"]
        ), "Descriptions do not match for args."

        args = []
        for p in params:
            args.append(
                f"{p.name}: {p.annotation.__name__}"
                + ((f"={p.default!r}") if p.default != inspect._empty else "")
            )

        prompt = f"{name}({', '.join(args)})\n" + docstring["description"] + "\n"
        for i in range(len(params)):
            param = params[i]
            prompt += f"{param.name} - {docstring['args'][i]}\n"

        if not params:
            prompt += "No args."
        elif prompt.endswith("\n"):
            prompt = prompt[:-1]

        return prompt

    @staticmethod
    def parse_args_from_text(
        text: str,
    ) -> Tuple[List[BuiltinTypes], Mapping[str, BuiltinTypes]]:
        args = []
        kwargs = {}

        module = ast.parse(f"_({text})")
        call: ast.Call = module.body[0].value  # type: ignore

        for arg in call.args:
            if not isinstance(arg, ast.Constant):
                raise ValueError("Call arguments can only be constants (ast.Constant)")

            if type(arg.value) in invalid_types:
                raise TypeError(
                    f"Invalid argument type. (got {type(arg.value)!r}, caused by LLM)"
                )

            args.append(arg.value)

        for kwarg in call.keywords:
            if not isinstance(kwarg.value, ast.Constant):
                raise ValueError(
                    "Call keyword-arguments can only be constants (ast.Constant)"
                )

            if type(kwarg.value.value) in invalid_types:
                raise TypeError(
                    f"Invalid keyword-argument type. (got {type(kwarg.value.value)!r}, caused by LLM)"
                )

            kwargs[kwarg.arg] = kwarg.value.value

        return args, kwargs


def tool(fn: Callable[P, T]) -> BaseTool[P, T]:
    """A decorator that makes wraps a function into a tool.

    Example:
        ```python
        @tool
        def weather(location: str):
            \"\"\"Gets the location.

            Args:
                location (str): The location.
            \"\"\"
            return { "weather": "nice" }
        ```

    Args:
        fn (Handler): Function.
    """

    class _ToolFactory(BaseTool):
        def __call__(self, *args: P.args, **kwargs: P.kwargs) -> T:
            return fn(*args, **kwargs)

    return _ToolFactory(fn.__name__, handler=fn)
