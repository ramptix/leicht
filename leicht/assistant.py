import re
from typing import List, Mapping, Optional, Union, overload

from .llms import BaseLLM, Groq, OpenAI
from .prompts import get_prompt
from .types import Message, LLMType
from .utils import clamp
from .tools.base import BaseTool

AnyLLM = Union[BaseLLM, Groq, OpenAI]

llm_mapping = {"openai": OpenAI, "groq": Groq}


def get_llm(llm: LLMType, **kwargs) -> AnyLLM:
    if isinstance(llm, str):
        return llm_mapping[llm](**kwargs)
    else:
        return llm.set(**kwargs)


class Assistant:
    """Represents an assistant.

    Args:
        description (str): Description of the assistant. The content will
            be set as system prompt. If given a prompt specification from
            `preprompted-data`, loads the prompt.
        llm (LLMType): The LLM. Defaults to OpenAI (shortcut: ``"openai"``).
    """

    __slots__ = ("llm", "messages", "tools")
    llm: AnyLLM
    messages: List[Message]
    tools: Mapping[str, BaseTool]

    def __init__(
        self, description: str, *, llm: LLMType, tools: Optional[List[BaseTool]] = None
    ):
        if re.match(r"^(?:[a-zA-Z\d\.-]+\/)?[a-zA-Z\d\.-]+$", description):
            # Is a prompt name specification
            try:
                description = get_prompt(description)
            except:  # noqa: E722
                ...  # Use the description

        self.tools = {tool.name: tool for tool in (tools or [])}
        self.llm = get_llm(llm, tools=[tool.prompt for tool in (tools or [])])
        self.messages = [{"role": "system", "content": description}]

    @overload
    def run(
        self,
        inquiry: List[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stream: bool = False,
        stop: Optional[str] = None,
        seed: Optional[int] = None,
    ): ...

    @overload
    def run(
        self,
        inquiry: str,
        *,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stream: bool = False,
        stop: Optional[str] = None,
        seed: Optional[int] = None,
    ): ...

    def run(
        self,
        inquiry: Union[List[Message], str],
        *,
        max_tokens: int = 4096,
        temperature: float = 1.0,
        top_p: float = 1.0,
        stream: bool = False,
        stop: Optional[str] = None,
        seed: Optional[int] = None,
    ):
        """Run the assistant instance."""
        if isinstance(inquiry, list):
            if not inquiry:
                raise ValueError(
                    "\n\nMessages cannot be empty. From:\n"
                    "  run(\x1b[38;2;171;178;191m[]\x1b[0m)\n"
                    "      \x1b[1;31m^^  cannot be empty\x1b[0m\n"
                )
            elif inquiry[-1]["role"] != "user":
                lmsg = inquiry[-1]  # last message
                raise ValueError(
                    "\n\nThe last message is sent by '%s', " % lmsg["role"]
                    + "which is not 'user'. From:\n"
                    + "  run([..., { 'role': \x1b[38;2;152;195;121m'%s'\x1b[0m, 'content': '%s' }])"
                    % (lmsg["role"], "...")
                    + "\n"
                    + " " * 22
                    + "\x1b[1;31m^" * (len(lmsg["role"]) + 2)
                    + "  should be 'user'.\x1b[0m\n"
                )

            self.messages += inquiry
        else:
            # Message(role="user", content=inquiry)
            self.messages.append({"role": "user", "content": inquiry})

        payload = {
            "max_tokens": max_tokens,
            "seed": seed,
            "stop": stop,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
        }
        res = self._request(payload, notools=False)

        functions = res.get("functions")
        if functions:
            for func in functions:
                # func[0] = name (str)
                # func[1] = arguments (unparsed, str)
                if func[0] in self.tools:
                    args, kwargs = BaseTool.parse_args_from_text(func[1])
                    res = self.tools[func[0]].__call__(*args, **kwargs)
                    self.messages.append(
                        {
                            "role": "assistant",
                            "content": f"I executed {func[0]}({func[1]}), results:\n{res}",
                        }
                    )
                    return self._request(payload, notools=True)

        else:
            return res

    def _request(self, p: dict, *, notools: bool = False):
        with self.llm.notools(notools):
            return self.llm(
                {  # type: ignore
                    **p,
                    "messages": self.messages,
                }
            )

    def __repr__(self) -> str:
        description = self.messages[0]["content"]
        return f"Assistant(description={clamp(description)!r}, len(tools)={len(self.tools)})"


def pipeline(description: str, llm: LLMType = "openai"): ...
