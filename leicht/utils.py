def clamp(t: str, m: int = 21):
    """Clamp text.

    Args:
        t (str): The text.
        m (int): Max length.
    """
    return t[: min(len(t), m)] + ("…" if len(t) > m else "")
