# leicht/llms

The LLM classes are separated so that it would be easier to install external LLM support (extras).

For example, one could publish an extra and the extra could be installed like so:

```shell
$ pip install leicht.llms.my_llm
```

It will be accessable in code:

```python
from leicht.llms.my_llm import MyLLM
```

This provides a cleaner API usage.
