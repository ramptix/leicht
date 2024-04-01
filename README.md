# leicht

Leicht (/leícht/, or "light" in English) is a toolkit for building assistants, built to be lightweight.

```python
from leicht import Assistant
from leicht.llms import Groq

# Initialize an LLM
groq = Groq()

# Create an assistant
assistant = Assistant(
  "You're a helpful assistant.",
  llm=groq
)

# Run the assistant
assistant.run("knock knock")
```

Messages are stored for continuous chat.

## What We're Working On

1. **Pipeline.** Basically an `Assistant` but designed to be used one-time.

```python
pipeline(groq, "Who's this?") # Use the pipeline() API
assistant.pipeline("Who's this?") # Use from assistant, no messages saved to cache
```

2. **LLM: OpenAI.** The OpenAI LLM, inherited from `BaseLLM`.

...and many more!

[🔥 contribute](https://github.com/ramptix/leicht/fork)
