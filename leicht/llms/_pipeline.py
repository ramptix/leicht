"""Pipeline."""

import os
from importlib.machinery import SourceFileLoader
from typing import Literal, Union

from .base import BaseLLM
from .hf import mistral_7b_instruct_v0_2_api
from ..types import LLMType, BasicLLMResponse

llm_mapping = {"openai": "OpenAI", "groq": "Groq"}


def get_llm(llm: LLMType, **kwargs) -> BaseLLM:
    """Gets an LLM.

    ```python
    get_llm("openai")  # literal string ref ("openai")
    get_llm(OpenAI, api_key="sk-xxx")  # un-initialized
    get_llm(OpenAI(), api_key="sk-xxx")  # initialized; sets extra keys
    ```
    
    Args:
        llm (LLMType): LLM type. Could be an initialized, uninitalized or 
            literal string reference of an LLM.
        **kwargs: Extra keyword-only arguments to pass to the LLM.
    """
    if isinstance(llm, str):
        mod = SourceFileLoader(
            "$importllms", os.path.join(os.path.dirname(__file__), "$importllms.py")
        ).load_module("$importllms")
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
