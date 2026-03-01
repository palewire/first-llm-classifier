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

## Validating responses with Pydantic

Due to its probabilistic nature, the LLM can sometimes return slight variations on the same answer. For instance, in one case it might say the Cubs are in "MLB" and in another it might say "Major League Baseball." This can make it difficult to analyze the data later on, since you have to account for all the different ways the same answer might be phrased.

In other circumstances, you might want to restrict the LLM to a specific set of answers, like a multiple choice question. For instance, when classifying email, you might want to restrict the LLM to only return "Spam" or "Not Spam."

You can handle these situations by adding a validation system that will only accept responses from a pre-defined list.

Most LLM providers, including Hugging Face, support a `response_format` parameter that enforces the shape of the output using [JSON schema](https://json-schema.org), an arcane specification system used to describe the structure of JSON data.

Rather than write JSON schema by hand, we'll use [Pydantic](https://docs.pydantic.dev/) — a popular Python library for data validation — to generate it for us.

Like Hugging Face's Python library, Pydantic will need to be installed using `uv`. Run the following command in a new cell.

```
!uv add pydantic
```

With Pydantic installed, you define a Python class that describes what the response should look like. The `Literal` type from Python's [typing](https://docs.python.org/3/library/typing.html) library restricts a field to specific values — exactly what we need for classification.

Return to our top cell and import these two new libraries.

```python
from pydantic import BaseModel
from typing import Literal
```

Create a new cell and define a Pydantic model that describes the shape of the response you want from the LLM. In this case, we want a single field called `answer` that can only be one of three values: "MLB", "NFL" or "NBA".

```python
class SportsLeague(BaseModel):
    answer: Literal["MLB", "NFL", "NBA"]
```

Pydantic can then generate the JSON schema automatically with its method `.model_json_schema()`, which we pass to the API's `response_format` parameter. We have to be careful to nest it in the way the API expects, which is a dictionary with a `name` and `schema` key.

We'll also need to parse the response using Pydantic's `model_validate_json` method, which takes the raw JSON string and converts it into a Python object that we can work with. This method also validates the response against our schema, so if the LLM returns something that doesn't fit our defined structure, we'll get an error instead of bad data.

{emphasize-lines="22-28,31-32"}

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
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SportsLeague",
                "schema": SportsLeague.model_json_schema()
            }
        },
    )

    result = SportsLeague.model_validate_json(response.choices[0].message.content)
    return result.answer
```

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

But what if you ask it for a team that's not in one of those leagues. Try it with the Chicago Blackhawks, a hockey team in the NHL.

```python
classify_team("Chicago Blackhawks")
```

```
'NFL'
```

You've forced it to hallucinate. To avoid these circumstances, it's vital to give any LLM task a way to classify things that don't fit nicely into the categories you've given it.

In this case, we can do that by adding an "Other" option to our `Literal` type and instructing the LLM to use it when a team doesn't fit into one of the three leagues.

{emphasize-lines="2"}

```python
class SportsLeague(BaseModel):
    answer: Literal["MLB", "NFL", "NBA", "Other"]
```

It also helps to reinforce the out in the system prompt.

{emphasize-lines="9"}

```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.

If the team's league is not in the provided options, reply with "Other".
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
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SportsLeague",
                "schema": SportsLeague.model_json_schema()
            }
        },
    )

    result = SportsLeague.model_validate_json(response.choices[0].message.content)
    return result.answer
```

Now try the Chicago Blackhawks again.

```python
classify_team("Chicago Blackhawks")
```

You’ll get the answer you expect.

```
'Other'
```

## Reducing creativity with temperature

Most LLMs are pre-programmed to be creative and generate a range of responses to the same prompt. For structured responses like this, we don't want that. We want consistency. So it's a good idea to ask the LLM to be more straightforward by reducing a creativity setting known as `temperature` to zero.

{emphasize-lines="31"}

```python
def classify_team(name):
    prompt = """
    You are an AI model trained to classify text.

    I will provide the name of a professional sports team.

    You will reply with the sports league in which they compete.

    If the team doesn't belong in the provided sports league options, reply with "Other".
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
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SportsLeague",
                "schema": SportsLeague.model_json_schema()
            }
        },
        temperature=0,
    )

    result = SportsLeague.model_validate_json(response.choices[0].message.content)
    return result.answer
```

## Few shot prompting

You can also increase reliability by priming the LLM with examples of the type of response you want. This technique is called ["few shot prompting."](https://www.ibm.com/think/topics/few-shot-prompting) This approach, which can feel like a strange form of roleplaying, calls on you to provide examples of the "user" input and "assistant" response you want the LLM to mimic.

Here's how it's done, using the Los Angeles teams as examples.

{emphasize-lines="19-50"}

```python
def classify_team(name):
    prompt = """
    You are an AI model trained to classify text.

    I will provide the name of a professional sports team.

    You will reply with the sports league in which they compete.

    If the team doesn't belong in the provided sports league options, reply with "Other".
    """

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
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "SportsLeague",
                "schema": SportsLeague.model_json_schema()
            }
        },
        temperature=0,
    )

    result = SportsLeague.model_validate_json(response.choices[0].message.content)
    return result.answer
```
