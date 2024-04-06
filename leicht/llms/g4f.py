from .openai import OpenAILike


class G4F(OpenAILike):
    def __init__(self):
        super().__init__(
            base_url="https://aweirddev-g4f.hf.space/v1",
            api_key="xxx",
            api_key_path="..."
        )
