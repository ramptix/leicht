"""Simple preprompt implementation.

https://github.com/ramptix/preprompted-data
"""

import gzip
import os

import httpx


def make_directory() -> None:
    os.makedirs(".preprompt/", exist_ok=True)

    if not os.path.exists(".preprompt/.gitignore"):
        # Add .gitignore so the cache doesn't go to the
        # user's git repository. We're responsible!
        # Damn!
        with open(".preprompt/.gitignore", "wb") as f:
            f.write(b"*")  # ignore all contents of this directory


def fetch_prompt(name: str) -> bytes:
    client = httpx.Client()
    r = client.get(
        f"https://raw.githubusercontent.com/ramptix/preprompted-data/main/src/{name}.md"
    )
    r.raise_for_status()

    return r.content.strip()


def save_prompt(path_name: str, data: bytes):
    with gzip.open(path_name, "wb") as file:
        file.write(data)


def read_prompt(path_name: str):
    with gzip.open(path_name, "rb") as file:
        return file.read().decode("utf-8")


def get_cached_prompt_or_fetch(name: str, no_cache: bool = False) -> str:
    if no_cache:
        return fetch_prompt(name).decode("utf-8")

    make_directory()
    path_name = ".preprompt/%s.prompt" % name

    if not os.path.exists(path_name):
        prompt_b: bytes = fetch_prompt(name)
        save_prompt(path_name, prompt_b)

        return prompt_b.decode("utf-8")

    return read_prompt(path_name)


def get_prompt(name: str, *, no_cache: bool = False, **kwargs: str) -> str:
    """Get a prompt from preprompted-data.

    Args:
        name (str): Name of the prompt.
        no_cache (bool): Do not fetch and save to cache.
        **kwargs: Kwargs to fill the prompt if needed.

    Returns:
        str: The prompt.

    Raises:
        httpx.HTTPStatusError: If fetching failed, this will be raised.
    """
    p = get_cached_prompt_or_fetch(name, no_cache=no_cache)

    if kwargs:
        for k, v in kwargs.items():
            p = p.replace("{%s}" % k, v)

    return p


def clear_cache():
    """Clears all prompts in ``.preprompt/*``."""
    import shutil  # noqa: F401

    shutil.rmtree(".preprompt", ignore_errors=True)


def update_all():
    """Update all prompts in ``.preprompt/*``."""
    make_directory()

    for file in os.listdir(".preprompt"):
        if file.endswith(".prompt"):
            # len(".prompt") = 7
            name = file[:-7]
            path_name = os.path.join(".preprompt", file)
            save_prompt(path_name, fetch_prompt(name))
