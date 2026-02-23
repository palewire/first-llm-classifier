# Prompting with Python

Now that you've got your Python environment set up, it's time to start writing prompts and sending them off to Hugging Face.

First, you need to install the libraries we need. The [`huggingface_hub`](https://pypi.org/project/huggingface-hub/) package is the official client for Hugging Face's API. You should also install [`rich`](https://pypi.org/project/rich/), a helper library that will improve how your outputs look in Jupyter notebooks.

A common way to install packages inside your notebook's virtual environment is to run `uv add` in a cell. The `!` is a shortcut that allows you to run terminal commands from inside a Jupyter notebook. You can put the two together like:

```text
!uv add huggingface_hub rich
```

Drop that into the first cell of a new notebook and hit the play button in the top toolbar. You should see something like this:

![Installation output in Jupyter notebook](_static/uv-add.png)

Now lets import them in the cell that appears below the installation output. Hit play again.

```python
from rich import print
from huggingface_hub import InferenceClient
```

Remember saving your API key? Good. You'll need it now. Copy it from that text file and paste it inside the quotemarks as variable in a third cell. You should continue adding new cells as you need throughout the rest of the class.

```python
api_key = "Paste your key here"
```

Next we need to create a client that will allow us to send requests to Hugging Face's API. We do that by calling the `InferenceClient` class and passing it our API key.

```python
client = InferenceClient(token=api_key)
```

Let's make our first prompt. To do that, we submit a dictionary to Hugging Face's `chat.completions.create` method. The dictionary has a `messages` key that contains a list of dictionaries. Each dictionary in the list represents a message in the conversation. When the `role` is "user" it is roughly the same as asking a question to a chatbot.

We also need to pick a model from [among the choices Hugging Face gives us](https://huggingface.co/models). We're picking Llama 4, the latest from Meta.

```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence",
        }
    ],
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
)
```

Our client saves the response as a variable. Print that Python object to see what it contains.

```python
print(response)
```

You should see something like:

```python
ChatCompletionOutput(
    choices=[
        ChatCompletionOutputComplete(
            finish_reason="stop",
            index=0,
            message=ChatCompletionOutputMessage(
                role="assistant",
                content="Data journalism is crucial as it enables journalists to uncover insights, identify trends, and hold those in power accountable by analyzing and interpreting complex data, leading to more informed reporting and storytelling.",
                reasoning=None,
                tool_call_id=None,
                tool_calls=[],
            ),
            logprobs=None,
            seed=None,
        )
    ],
    created=1771798979,
    id="oYSotgN-zqrih-9d21e29c7eec059f",
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    system_fingerprint=None,
    usage=ChatCompletionOutputUsage(
        completion_tokens=37,
        prompt_tokens=21,
        total_tokens=58,
        reasoning_tokens=0,
    ),
    object="chat.completion",
    metadata={"weight_version": "default"},
    prompt=[],
)
```

There's a lot here, but the `message` has the actual response from the LLM. Let's just print the content from that message. Note that your response probably varies from this guide. That's because LLMs mostly are probablistic prediction machines. Every response can be a little different.

```python
print(response.choices[0].message.content)
```

```text
Data journalism is crucial as it enables journalists to uncover insights, identify trends, and hold those in power
accountable by analyzing and interpreting complex data, leading to more informed reporting and storytelling.
```

Let's pick a different model from among [the choices that Hugging Face offers](https://huggingface.co/models?pipeline_tag=text-generation&inference_provider=all&sort=trending). One we could try is Gemma3, an open model from Google. Rather than add a new cell, lets revise the code we already have and rerun it.

{emphasize-lines="8"}

```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence",
        }
    ],
    model="google/gemma-3-27b-it",
)
```

Again, your response might vary from what's here. Let's find out.

```python
print(response.choices[0].message.content)
```

```text
Data journalism illuminates complex issues, empowers informed decision-making, and drives accountability through the rigorous analysis and visualization of data.
```

:::{admonition} Sidenote
Hugging Face's Python library is very similar to the ones offered by OpenAI, Anthropic and other LLM providers. If you prefer to use those tools, the techniques you learn here should be easily transferable.

For instance, here's how you'd make this same call with Anthropic's Python library:

```python
from anthropic import Anthropic

client = Anthropic(api_key=api_key)

response = client.messages.create(
    messages=[
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence"
        }
    ],
    model="claude-sonnet-4-6",
)

print(response.content[0].text)
```

:::

A well-structured prompt helps the LLM provide more accurate and useful responses.

One common technique for improving results is to open with a "system" prompt to establish the model's tone and role. Let's switch back to Llama 4 and provide a `system` message that provides a specific motivation for the LLM's responses.

{emphasize-lines="3-6,12"}

```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "you are an enthusiastic nerd who believes data journalism is the future."
        },
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence",
        }
    ],
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
)
```

Check out the results.

```python
print(response.choices[0].message.content)
```

```text
Data journalism is revolutionizing the way we tell stories and uncover truths by harnessing the power of data
analysis and visualization to provide in-depth insights and hold those in power accountable, making it an
indispensable tool for a more informed and transparent society.
```

Want to see how tone affects the response? Change the system prompt to something old-school.

{emphasize-lines="5"}

```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": "you are a crusty, ill-tempered editor who hates math and thinks data journalism is a waste of time and resources."
        },
        {
            "role": "user",
            "content": "Explain the importance of data journalism in a concise sentence",
        }
    ],
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
)
```

Then re-run the code and summon J. Jonah Jameson.

```python
print(response.choices[0].message.content)
```

```text
*scoff* Fine. If I must, I'll grudgingly admit that data journalism can occasionally be useful in uncovering a
story that wouldn't have been possible through traditional reporting, but I still think it's a bunch of
number-crunching nonsense that's overhyped and underdelivers most of the time.
```
