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

## Our Tasks

There are so many features I thought of but I'm literally on my phone (on the 🚽, forgive me, I'm at work) typing right now, so maybe I'll just leave it until I have time.

- [ ] **Pipeline.** Basically an `Assistant` but designed to be used one-time.

```python
pipeline(groq, "Who's this?") # Use the pipeline() API
assistant.pipeline("Who's this?") # Use from assistant, no messages saved to cache
```

- [ ] **LLM: OpenAI.** The OpenAI LLM, inherited from `BaseLLM`.

- [ ] **Tools API.** Tools are basically function calls. Including:
  - JSON API (openai-compatible).
  - Steps (might consider, basically a mixture of tools)
  - Custom

- [ ] **Tool: `Playwright`.** The LLM will be able to control a web browser! No worries, we'll use an online version — https://try.playwright.tech

...and many more!

[🔥 contribute](https://github.com/ramptix/leicht/fork)
