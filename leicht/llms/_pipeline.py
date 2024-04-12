"""Pipeline."""

from typing import Literal, Union

from .base import BaseLLM
from .groq import Groq
from .hf import mistral_7b_instruct_v0_2_api
from .openai import OpenAI
from ..types import LLMType, BasicLLMResponse

AnyLLM = Union[BaseLLM, Groq, OpenAI]

llm_mapping = {"openai": OpenAI, "groq": Groq}


def get_llm(
    llm: Union[Literal["hf"], LLMType], **kwargs
) -> Union[AnyLLM, BasicLLMResponse]:
    if isinstance(llm, str):
        return llm_mapping[llm](**kwargs)
    elif isinstance(llm, type):
        return llm(**kwargs)
    else:
        return llm.set(**kwargs)


def pipeline(__name: LLMType, **kwargs) -> BasicLLMResponse:
    if __name == "hf":
        return mistral_7b_instruct_v0_2_api(**kwargs)

    model = get_llm(__name)
    return model(kwargs)  # type: ignore
