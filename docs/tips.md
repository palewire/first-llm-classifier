# Improving reliability

With a working LLM classifier in hand, there are a few tips and tricks that can improve its reliability and accuracy.

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
                "content": '{"answer": "NFL"}',
            },
            {
                "role": "user",
                "content": "Los Angeles Dodgers",
            },
            {
                "role": "assistant",
                "content": '{"answer": "MLB"}',
            },
            {
                "role": "user",
                "content": "Los Angeles Lakers",
            },
            {
                "role": "assistant",
                "content": '{"answer": "NBA"}',
            },
            {
                "role": "user",
                "content": "Los Angeles Kings",
            },
            {
                "role": "assistant",
                "content": '{"answer": "Other"}',
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

## Retrying failed requests

When making many API calls, temporary network errors or rate limits can cause failures. The [`tenacity`](https://tenacity.readthedocs.io/) library provides a `retry` decorator that will automatically retry a function if it raises an exception.

Import it in your top cell.

```python
import tenacity
```

Add the `@tenacity.retry` decorator to your classify function.

{emphasize-lines="1"}

```python
@tenacity.retry(stop=tenacity.stop_after_attempt(3))
def classify_team(name):
    ...
```

Any failed request will be retried automatically. This makes your classifier much more resilient to temporary network failures or API hiccups.

```{warning}
If you use `@tenacity.retry` with no arguments, tenacity will retry **forever** by default — with no wait between attempts and no limit on the number of tries. Always set a `stop` condition like `stop_after_attempt` so your script doesn't get stuck in an infinite loop.
```
