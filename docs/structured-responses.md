# Structured responses

Here's a public service announcement. There's no law that says you have to ask LLMs for essays, poems or relationship advice.

Yes, they're great at drumming up long blocks of text. An LLM can spit out a long answer to almost any question. It's how they've been tuned and marketed by companies selling chatbots and more conversational forms of search.

But they're also great at answering simple questions, a skill that has been overlooked in much of the hoopla that followed the introduction of ChatGPT.

Here's a example that simply prompts the LLM to answer a straightforward question.

```python
prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.
"""
```

Lace that into our request.

{emphasize-lines="5"}
```python
response = client.chat.completions.create(
    messages=[
        {
            "role": "system",
            "content": prompt
        },
    ],
    model="llama-3.3-70b-versatile",
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
            "content": "Minnesota Twins",
        }
    ],
    model="llama-3.3-70b-versatile",
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
            "content": "Minnesota Vikings",
        }
    ],
    model="llama-3.3-70b-versatile",
)
```

```python
print(response.choices[0].message.content)
```

See what we mean?

```
National Football League (NFL)
```

This approach can be use to classify large datasets, adding a new column of data that categories text in a way that makes it easier to analyze.

Let's try it by making a function that will classify whatever team you provide.

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
        model="llama-3.3-70b-versatile",
    )

    return response.choices[0].message.content
```

A list of teams.

```python
team_list = ["Minnesota Twins", "Minnesota Vikings", "Minnesota Timberwolves"]
```

Now, loop through the list and ask the LLM to code them one by one.

```python
for team in team_list:
    league = classify_team(team)
    print([team, league])
```

```python
['Minnesota Twins', 'Major League Baseball (MLB)']
['Minnesota Vikings', 'National Football League (NFL)']
['Minnesota Timberwolves', 'National Basketball Association (NBA)']
```

Due its probabilistic nature, the LLM can sometimes return slight variations on the same answer. You can prevent this by adding a validation system that will only accept responses from a pre-defined list.

{emphasize-lines="9-12,31-37"}
```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.

Your responses must come from the following list:
- Major League Baseball (MLB)
- National Football League (NFL)
- National Basketball Association (NBA)
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
        model="llama-3.3-70b-versatile",
    )

    answer = response.choices[0].message.content

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
    ]
    if answer not in acceptable_answers:
        raise ValueError(f"{answer} not in list of acceptable answers")

    return answer
```

Now, ask it for a team that's not in one of those leagues. You should get an error.

```python
classify_team("Minnesota Wild")
```

```python
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
Cell In[47], line 1
----> 1 classify_team("Minnesota Wild")

Cell In[45], line 36, in classify_team(name)
     30 acceptable_answers = [
     31     "Major League Baseball (MLB)",
     32     "National Football League (NFL)",
     33     "National Basketball Association (NBA)",
     34 ]
     35 if answer not in acceptable_answers:
---> 36     raise ValueError(f"{answer} not in list of acceptable answers")
     38 return answer

ValueError: National Hockey League (NHL)

However, since NHL is not in the provided list, I must inform you that the Minnesota Wild does not belong to any of the leagues mentioned (MLB, NFL, NBA). not in list of acceptable answers
```

For cases when there isn't an accurate answer in your validation list, you can choose to allow an "other" category.

{emphasize-lines="14,37"}
```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.

Your responses must come from the following list:
- Major League Baseball (MLB)
- National Football League (NFL)
- National Basketball Association (NBA)

If the team's league is not on the list, you should label them as "Other".
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
        model="llama-3.3-70b-versatile",
    )

    answer = response.choices[0].message.content

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
        "Other",
    ]
    if answer not in acceptable_answers:
        raise ValueError(f"{answer} not in list of acceptable answers")

    return answer
```

Now try the Minnesota Wild again.

```python
classify_team("Minnesota Wild")
```

And you'll get the answer you expect.

```python
'Other'
```

Most LLMs are pre-programmed to be creative and generate a range of responses to same prompt. For structured responses like this, we don't want that. We want consistency. So it's a good idea to ask the LLM to be more straightforward by reducing a creativity setting known as `temperature` to zero.

{emphasize-lines="29"}
```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.

Your responses must come from the following list:
- Major League Baseball (MLB)
- National Football League (NFL)
- National Basketball Association (NBA)

If the team's league is not on the list, you should label them as "Other".
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
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer = response.choices[0].message.content

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
        "Other",
    ]
    if answer not in acceptable_answers:
        raise ValueError(f"{answer} not in list of acceptable answers")

    return answer
```

You can also increase reliability by priming the LLM with examples of the type of response you want. This technique is called ["few shot prompting"](https://www.ibm.com/think/topics/few-shot-prompting). In this style of prompting, which can feel like a strange form of roleplaying, you provide both the "user" input as well as the "assistant" response you want the LLM to mimic.

Here's how it's done:

{emphasize-lines="23-54"}
```python
def classify_team(name):
    prompt = """
You are an AI model trained to classify text.

I will provide the name of a professional sports team.

You will reply with the sports league in which they compete.

Your responses must come from the following list:
- Major League Baseball (MLB)
- National Football League (NFL)
- National Basketball Association (NBA)

If the team's league is not on the list, you should label them as "Other".
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
                "content": "National Football League (NFL)",
            },
            {
                "role": "user",
                "content": "Los Angeles Dodgers",
            },
            {
                "role": "assistant",
                "content": " Major League Baseball (MLB)",
            },
            {
                "role": "user",
                "content": "Los Angeles Lakers",
            },
            {
                "role": "assistant",
                "content": "National Basketball Association (NBA)",
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
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer = response.choices[0].message.content

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
        "Other",
    ]
    if answer not in acceptable_answers:
        raise ValueError(f"{answer} not in list of acceptable answers")

    return answer
```
