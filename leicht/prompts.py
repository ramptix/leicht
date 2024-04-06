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

    if name.startswith("community"):
        r = client.get(
            "https://github-discussions-api.vercel.app/body",
            params={
                "url": (
                    "https://github.com/ramptix/preprompted-data/discussions/%s"
                    % name.split("/")[1]
                )
            },
            timeout=None,
        )
        r.raise_for_status()
        return r.json()["body"].encode("utf8")

    r = client.get(
        f"https://raw.githubusercontent.com/ramptix/preprompted-data/main/src/{name}.md"
    )
    r.raise_for_status()

    return r.content.strip()


def save_prompt(path_name: str, data: bytes):
    if "/" in path_name:
        os.makedirs("/".join(path_name.split("/")[:-1]), exist_ok=True)

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


def update_all(_dir: str = ".preprompt"):
    """Update all prompts."""
    make_directory()
    path = lambda p: p.replace("\\", "/")  # noqa: E731

    for file in os.listdir(_dir):
        if file.endswith(".prompt"):
            path_name = path(os.path.join(_dir, file))
            name = path_name[11:-7]  # len(".preprompt/") = 11, len(".prompt") = 7

            try:
                save_prompt(path_name, fetch_prompt(name))
            except httpx.HTTPStatusError as err:
                if err.response.status_code == 404:
                    print(
                        f"\x1b[1;31m[404] Prompt {name!r} is not available, skipping.\x1b[0m"
                    )
                    os.remove(path_name)

        elif os.path.isdir(os.path.join(_dir, file)):
            update_all(".preprompt/%s" % file)
