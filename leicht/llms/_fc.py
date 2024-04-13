"""Function call control."""

import re
from typing import List, Optional, Tuple
from typing_extensions import TypedDict

from ._pipeline import pipeline
from ..types import Message, BasicLLMResponse
from ..prompts import get_prompt
from ..logger import logger

FunctionCalls = List[Tuple[str, str]]


class FunctionCallResponse(TypedDict):
    functions: FunctionCalls


def get_function_call(
    messages: List[Message], tools: List[str]
) -> Optional[FunctionCalls]:
    logger.info("_fc: getting function call...")
    messages_text = "Given messages:\n" + "\n".join(
        (f"{m['role']}: {m['content']}" for m in messages)
    )
    res: BasicLLMResponse = pipeline(
        "hf",
        messages=[
            {
                "role": "user",
                "content": (
                    get_prompt(
                        "functions-v2",
                        tools="\n\n".join(tools),
                        most_commonly_used=tools[0].splitlines()[0],
                        messages=messages_text,
                    )
                ),
            }
        ],
    )

    content: str = res["choices"][0]["message"].get("content", "").strip()

    logger.info(f"_fc: function call (res): {content}")

    return (
        parse_function_call(content)
        if check_if_applicable_for_fn_call(content)
        else None
    )


def check_if_applicable_for_fn_call(content: str) -> bool:
    appl = (
        not content.lstrip()  # Clear spaces/indents
        .lstrip("\"'")  # Clear string quotes
        .lower()  # Convert to lower case
        .startswith("null")  # Startswith "null"? (AKA. no function call?)
    )
    logger.info(f"_fc: function call applicable? {appl}")

    return appl


def parse_function_call(text: str) -> FunctionCalls:
    logger.info("_fc: parsing function call...")

    calls = []

    for line in text.splitlines():
        r = re.findall(r"^((?!\d)[a-zA-Z0-9_\\]+)\((.*)\)(?:.*)$", line)

        if not r:
            continue

        r = r[0]  # first case

        calls.append((r[0].replace("\\_", "_"), r[1]))

    logger.info(f"_fc: function calls - {calls}")
    return calls
