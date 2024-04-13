from typing import List, Mapping, Optional, Union, overload

from .llms.base import BaseLLM as AnyLLM
from .llms._pipeline import get_llm
from .prompts import get_prompt
from .types import Message, LLMType
from .utils import clamp, msgs_to_text
from .tools.base import BaseTool
from .logger import logger
from .utils import prompt_alike
from .conditional import Conditional, ConditionalCheckError


class Assistant:
    """Represents an assistant.

    Args:
        description (str): Description of the assistant. The content will
            be set as system prompt. If given a prompt specification from
            `preprompted-data`, loads the prompt.
        llm (LLMType): The LLM. Defaults to OpenAI (shortcut: ``"openai"``).
    """

    __slots__ = ("llm", "messages", "tools", "conditionals")
    llm: AnyLLM
    messages: List[Message]
    tools: Mapping[str, BaseTool]
    conditionals: List[Conditional]

    def __init__(
        self,
        description: str,
        *,
        llm: LLMType,
        tools: Optional[List[BaseTool]] = None,
        conditionals: Optional[List[Conditional]] = None,
        **llm_kwargs,
    ):
        if prompt_alike(description):
            # Is a prompt name specification
            try:
                description = get_prompt(description)
            except:  # noqa: E722
                ...  # Use the description

        tools = tools or []
        self.tools = {tool.name: tool for tool in tools}
        self.llm = get_llm(llm, tools=[tool.prompt for tool in tools], **llm_kwargs)
        self.messages = [
            {
                "role": "system",
                "content": description
                + (
                    (
                        "You can:\n"
                        + "\n".join(((tool.caps or tool.description) for tool in tools))
                        + "\n(all return real-time info)"
                    )
                    if tools
                    else ""
                ),
            }
        ]
        self.conditionals = conditionals or []

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
        logger.info("Assistant(): running")

        if self.conditionals:
            logger.info("Assistant(): checking conditionals...")
            for con in self.conditionals:
                logger.info(f"Assistant(): check - {con!r}")

                if not con.check(msgs_to_text(inquiry)):
                    raise ConditionalCheckError(
                        f"Rejected due to conditional check: {con!r}"
                    )

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

        logger.info("Assistant(): inferring if functions are needed")
        payload = {
            "max_tokens": max_tokens,
            "seed": seed,
            "stop": stop,
            "stream": stream,
            "temperature": temperature,
            "top_p": top_p,
        }
        res = self.llm(
            {
                **payload,
                "messages": self.messages,
            }
        )

        functions = res.copy().get("functions")
        if functions:
            logger.info(f"Assistant(): detected function calling from {self.llm!r}")

            for func in functions:
                # func[0] = name (str)
                # func[1] = arguments (unparsed, str)
                if func[0] in self.tools:
                    logger.info(f"Assistant(): running function {func[0]}({func[1]})")

                    # parse arguments and run
                    args, kwargs = BaseTool.parse_args_from_text(func[1])
                    res = self.tools[func[0]].__call__(*args, **kwargs)

                    self.messages.append(
                        {
                            "role": "system",
                            "content": f"I executed {func[0]}({func[1]}), results:\n{res}.\nReply the user.",
                        }
                    )

            logger.info("Assistant(): successfully ran all functions!")
            logger.info("Assistant(): asking for general response...")
            r = self.llm({**payload, "messages": self.messages}, notools=True)
            logger.info("Assistant(): `run` instance complete")
            return r

        else:
            logger.info("Assistant(): `run` instance complete (no funcs)")
            return res

    def __repr__(self) -> str:
        description = self.messages[0]["content"]
        return f"Assistant(description={clamp(description)!r}, tools={self.tools}, conditionals={self.conditionals})"


def pipeline(description: str, llm: LLMType = "openai"): ...
