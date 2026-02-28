# Structured responses

Here's a public service announcement. There's no law that says you have to ask LLMs for essays, poems or relationship advice.

Yes, they're great at drumming up long blocks of text. An LLM can spit out a long answer to almost any question. It's how they've been tuned and marketed by companies selling chatbots and more conversational forms of search.

But they're also great at answering simple questions with simple answers, a skill that has been overlooked in much of the hoopla that followed the introduction of ChatGPT.

Here's an example that simply prompts the LLM to answer a straightforward question. Since this prompt is a little longer than the one we used in the previous notebook, we'll assign it to a variable and use [Python's triple quote syntax](https://docs.python.org/3/tutorial/introduction.html#strings) to break across multiple lines and make it easier to read.

```python
prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.
"""
```

Run that cell and then lace the variable into our request.

{emphasize-lines="5"}

```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": prompt
        },
    ],
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
)
```

And now add a user message that provides the name of a professional sports team.

{emphasize-lines="7-10"}

```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": "Chicago Cubs",
        }
    ],
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
)
```

Check the response.

```python
print(response.choices[0].message.content)
```

And we'll bet you get the right answer.

```
Major League Baseball (MLB)
```

Try another one.

{emphasize-lines="9"}

```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": prompt
        },
        {
            "role": "user",
            "content": "Chicago Bears",
        }
    ],
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
)
```

```python
print(response.choices[0].message.content)
```

See what we mean?

```
National Football League (NFL)
```

This approach can be used to classify large datasets, adding a new column of data that categorizes text in a way that makes it easier to analyze.

Let's try it by making a function that will classify whatever team you provide. We'll include everything we've done so far in a single, reusable chunk of code.

```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": name,
            }
        ],
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
    )

    return response.choices[0].message.content
```

Try it with a single team to see how it works.

```python
classify_team("Chicago Bulls")
```

To show the power of reusability, let's make a list of teams.

```python
team_list = ["Chicago Cubs", "Chicago Bears", "Chicago Bulls"]
```

Loop through the list and ask the LLM to classify them one by one.

```python
for team in team_list:
    league = classify_team(team)
    print([team, league])
```

```python
['Chicago Cubs', 'Major League Baseball (MLB)']
['Chicago Bears', 'National Football League (NFL)']
['Chicago Bulls', 'National Basketball Association (NBA)']
```

### Validating responses with JSON schema

Due to its probabilistic nature, the LLM can sometimes return slight variations on the same answer. For instance, in one case it might say the Cubs are are in "MLB" and in another it might say "Major League Baseball". This can make it difficult to analyze the data later on, since you have to account for all the different ways the same answer might be phrased.

You can prevent this by adding a validation system that will only accept responses from a pre-defined list.

Most LLM providers, including Hugging Face, accept JSON schema as a way of enforcing the shape of the output. JSON is a JavaScript data format that is easy to work with in Python, and [JSON schema](https://json-schema.org) is a standard that predates modern LLMs and is used to describe an expected JSON output. There are a number of ways to make a JSON schema, from using libraries like [Pydantic](https://docs.pydantic.dev/latest/concepts/json_schema/) to [asking an LLM to write one for you](https://chatgpt.com/g/g-uPUxVmHC8-structured-output-json-schema-generator) to learning how to write it yourself.

To use a schema, most of the LLM libraries will use a `response_format` which tells the code to respond in JSON instead of text. For Hugging Face, that looks like this

```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "schema": schema,
        "strict": True, # You almost always want this to be true
    },
}
```

JSON schema can handle some fairly complex data structures, but for our purposes we'll stick to one common pattern: an allowlist of options called an [`enum`](https://json-schema.org/understanding-json-schema/reference/enum). Here's a handy utility function you can use to generate a response format with an allowlist of options like a list of leagues.

```python
def gen_allowlist_response_format(options):
    schema = {
      "type": "object",
      "properties": {
        "answer": {
          "type": "string",
          "enum": options
        }
      },
      "required": [
          "answer"
      ],
      "additionalProperties": False
    }

    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "AllowlistSchema",
            "schema": schema,
            "strict": True,
        },
    }

    return response_format
```

First, we'll need the `json` library, so you can import that at the top of your notebook.

{emphasize-lines="1"}

```python
import json
from rich import print
from huggingface_hub import InferenceClient
```

Then, you can integrate the utility by making a few changes to your classify function:

- Create a list of acceptable answers, and pass that list into the response format utility.
- Update our classification function to use the `json` python library and parse the response into a Python dictionary.
- Reach in and pull out just the answer. We know the `answer` key will always exist because the JSON schema demands it.

{emphasize-lines="12-16,30,33-34"}

```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.

If the team doesn't belong in the provided sports league options, reply with "Other".
"""

    acceptable_answers = [
        "MLB",
        "NFL",
        "NBA",
    ]

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": name,
            }
        ],
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        response_format=gen_allowlist_response_format(acceptable_answers),
    )

    response_dict = json.loads(response.choices[0].message.content)
    return response_dict["answer"]
```

:::{admonition} Sidenote
You might be wondering,

> Wait a second, I thought `response_format` was already telling it to return JSON, why do I need to parse the response like it's text?

Most LLMs only return text. Response format ensures the text response is valid JSON (so you'll never get an error running `json.loads`), but `content` will always be a string. That's why it needs to be parsed before you can access the data embedded in it.
:::

With the new structured output in place, run the loop again.

```python
for team in team_list:
    league = classify_team(team)
    print([team, league])
```

You'll notice it's only using the acronym as our allowlist instructs.

```
['Chicago Cubs', 'MLB']
['Chicago Bears', 'NFL']
['Chicago Bulls', 'NBA']
```

But what if you ask it for a team that's not in one of those leagues. Well you've commanded that it only return one of those answers, so it will follow those instructions.

```python
classify_team("Chicago Blackhawks")
```

```
'NFL'
```

You've essentially _forced_ it to hallucinate. That's why it's always vital to give any LLM task an out. A way to classify things it doesn't think fits nicely into the categories you've given it. It also helps to reinforce that out in the system prompt.

{emphasize-lines="9,16"}

```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.

If the team doesn't belong in the provided sports league options, reply with "Other".
"""

    acceptable_answers = [
        "MLB",
        "NFL",
        "NBA",
        "Other",
    ]

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": name,
            }
        ],
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        response_format=gen_allowlist_response_format(acceptable_answers),
    )

    response_dict = json.loads(response.choices[0].message.content)
    return response_dict["answer"]
```

Now try the Chicago Blackhawks again.

```python
classify_team("Chicago Blackhawks")
```

And you’ll get the answer you expect.

```
'Other'
```

Most LLMs are pre-programmed to be creative and generate a range of responses to same prompt. For structured responses like this, we don't want that. We want consistency. So it's a good idea to ask the LLM to be more straightforward by reducing a creativity setting known as `temperature` to zero.

{emphasize-lines="32"}

```python
def classify_team(name):
    prompt = """
    You are an AI model trained to classify text.

    I will provide the name of a professional sports team.

    You will reply with the sports league in which they compete.

    If the team doesn't belong in the provided sports league options, reply with "Other".
    """

    acceptable_answers = [
        "MLB",
        "NFL",
        "NBA",
        "Other"
    ]

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": name,
            }
        ],
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        response_format=gen_allowlist_response_format(acceptable_answers),
        temperature=0,
    )

    response_dict = json.loads(response.choices[0].message.content)
    return response_dict["answer"]
```

You can also increase reliability by priming the LLM with examples of the type of response you want. This technique is called ["few shot prompting"](https://www.ibm.com/think/topics/few-shot-prompting). In this style of prompting, which can feel like a strange form of roleplaying, you provide both the "user" input as well as the "assistant" response you want the LLM to mimic.

Here's how it's done:

{emphasize-lines="25-56"}

```python
def classify_team(name):
    prompt = """
    You are an AI model trained to classify text.

    I will provide the name of a professional sports team.

    You will reply with the sports league in which they compete.

    If the team doesn't belong in the provided sports league options, reply with "Other".
    """

    acceptable_answers = [
        "MLB",
        "NFL",
        "NBA",
        "Other"
    ]

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "Los Angeles Rams",
            },
            {
                "role": "assistant",
                "content": "NFL",
            },
            {
                "role": "user",
                "content": "Los Angeles Dodgers",
            },
            {
                "role": "assistant",
                "content": "MLB",
            },
            {
                "role": "user",
                "content": "Los Angeles Lakers",
            },
            {
                "role": "assistant",
                "content": "NBA",
            },
            {
                "role": "user",
                "content": "Los Angeles Kings",
            },
            {
                "role": "assistant",
                "content": "Other",
            },
            {
                "role": "user",
                "content": name,
            }
        ],
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        response_format=gen_allowlist_response_format(acceptable_answers),
        temperature=0,
    )

    response_dict = json.loads(response.choices[0].message.content)
    return response_dict["answer"]
```
