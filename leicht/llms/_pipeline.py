"""Pipeline."""

from importlib.machinery import ExtensionFileLoader
from typing import Literal, Union

from .base import BaseLLM
from .hf import mistral_7b_instruct_v0_2_api
from ..types import LLMType, BasicLLMResponse

llm_mapping = {"openai": "OpenAI", "groq": "Groq"}


def get_llm(llm: Union[Literal["hf"], LLMType], **kwargs) -> BaseLLM:
    if isinstance(llm, str):
        mod = ExtensionFileLoader("leicht", "leicht.llms").load_module("leicht.llms")
        return getattr(mod, llm_mapping[llm])(**kwargs)
    elif isinstance(llm, type):
        return llm(**kwargs)
    else:
        return llm.set(**kwargs)


def pipeline(__name: LLMType, **kwargs) -> BasicLLMResponse:
    if __name == "hf":
        return mistral_7b_instruct_v0_2_api(**kwargs)

    model = get_llm(__name)
    return model(kwargs)  # type: ignore
