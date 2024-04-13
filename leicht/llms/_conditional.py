from ._pipeline import pipeline
from ..utils import prompt_alike
from ..prompts import get_prompt


def get_conditional(prompt: str, **kwargs: str) -> bool:
    """Checks a conditional statement from a piece of text.

    Args:
        prompt (str): Prompt name or prompt content.
        **kwargs: Keyword-only arguments for the prompt.
    """
    if prompt_alike(prompt):
        prompt = get_prompt(prompt)
    else:
        for k, v in kwargs.items():
            prompt = prompt.replace(k, v)

    res = pipeline("hf", messages=[{"role": "user", "content": prompt}])

    content: str = (
        res.copy()["choices"][0]["message"]["content"]
        .splitlines()[0]
        .split(".")[0]
        .strip()
        .lower()
    )

    assert content in {
        "true",
        "false",
        "null",
    }, "LLM did not reply with 'true', 'false' or 'null'"

    return {"true": True, "false": False, "null": False}[content]
