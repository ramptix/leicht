from typing import List, Union, overload

from .llms import BaseLLM, Groq, OpenAI
from .types import Message, LLMType
from .utils import clamp

llm_mapping = {"openai": OpenAI, "groq": Groq}


def get_llm(llm: LLMType) -> BaseLLM:
    return llm_mapping[llm]() if isinstance(llm, str) else llm


class Assistant:
    """Represents an assistant.

    Args:
        description (str): Description of the assistant. The content will
            be set as system prompt.
        llm (LLMType): The LLM. Defaults to OpenAI (shortcut: ``"openai"``).
    """

    __slots__ = ("llm", "messages")
    llm: BaseLLM
    messages: List[Message]

    def __init__(self, description: str, *, llm: LLMType):
        self.llm = get_llm(llm)
        self.messages = [{"role": "system", "content": description}]

    @overload
    def run(self, inquiry: List[Message]): ...

    @overload
    def run(self, inquiry: str): ...

    def run(self, inquiry: Union[List[Message], str]):
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

    def function_call(self, inquiry: str):
        """Run the assistant with function call."""

    def __repr__(self) -> str:
        description = self.messages[0]["content"]
        return f"Assistant(description={clamp(description)!r}, tools=[])"


def pipeline(description: str, llm: LLMType = "openai"): ...
