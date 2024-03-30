from .llms import BaseLLM, OpenAI


class Assistant:
    """Represents an assistant."""

    def __init__(self, description: str, llm: BaseLLM = OpenAI): ...


def pipeline(description: str, llm: BaseLLM = OpenAI): ...
