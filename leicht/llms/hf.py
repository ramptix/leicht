import time
from typing import List

import httpx

from ..types import Message


def mistral_7b_instruct_v0_2(
    *,
    messages: List[Message],
    temperature: float = 0.9,
    frequency_penalty: float = 1.2,
    top_p: float = 0.9
):
    """Mistral 7b Instruct v0.2

    See issue: https://github.com/ramptix/leicht/issues/1
    
    ```python
    mistral_7b_instruct_v0_2(messages=[
        { "role": "user", "content": "Hello!" }
    ])
    ```
    """
    base_url = "https://aweirddev-mistral-7b-instruct-v0-2-leicht.hf.space"
    client = httpx.Client()
    r = client.post(
        base_url + "/chat/completions",
        params={
            "id": time.time_ns() # prevents "server unavailable" errors
        },
        json={
            "model": "mistral-7b-instruct-v0.2",
            "messages": messages,
            "temperature": temperature,
            "frequency_penalty": frequency_penalty,
            "top_p": top_p,
            "stream": False
        }
    )
    return r.json()
