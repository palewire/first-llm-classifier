# Bulk prompts

Our reusable prompting function is pretty cool. But requesting answers one by one across a big dataset could take forever. And it could cost us money to hit the Groq API so many times.

One solution is to submit your requests in batches and then ask the LLM to return its responses in bulk.

A common way to do that is to prompt the LLM to return its responses in JSON, a JavaScript data format that is easy to work with in Python.

To try that, we start by adding the built-in `json` library to our imports.

{emphasize-lines="1"}
```python
import json
from rich import print
from groq import Groq
```

Next, we make a series of changes to our function to adapt it to work with a batch of inputs. Get ready. It's a lot.

* We tweak the name of the function.
* We change our input argument to a list.
* We expand our prompt to explain that we will provide a list of team names.
* We ask the LLM to classify them individually, returning its answers in a JSON list.
* We insist on getting one answer for each input.
* We tweak our few-shot training to reflect this new approach.
* We submit our input as a single string with new lines separating each team name.
* We convert the LLM's response from a string to a list using the `json.loads` function.
* We check that the LLM's answers are in our list of acceptable answers with a loop through the list.
* We merge the team names and the LLM's answers into a dictionary returned by the function.

{emphasize-lines="2,17-27,36-43,46,53-54,62-66"}
```python
def classify_teams(name_list):
    prompt = """
You are an AI model trained to classify text.

I will provide list of professional sports team names separated by new lines

You will reply with the sports league in which they compete.

Your responses must come from the following list:
- Major League Baseball (MLB)
- National Football League (NFL)
- National Basketball Association (NBA)

If the team's league is not on the list, you should label them as "Other".

Your answers should be returned as a flat JSON list.

It is very important that the length of JSON list you return is exactly the same as the number of names you receive.

If I were to submit:

"Los Angeles Rams\nLos Angeles Dodgers\nLos Angeles Lakers\nLos Angeles Kings"

You should return the following:

["National Football League (NFL)", "Major League Baseball (MLB)", "National Basketball Association (NBA)", "Other"]
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "Chicago Bears\nChicago Cubs\nChicago Bulls\nChicago Blackhawks",
            },
            {
                "role": "assistant",
                "content": '["National Football League (NFL)", "Major League Baseball (MLB)", "National Basketball Association (NBA)", "Other"]',
            },
            {
                "role": "user",
                "content": "\n".join(name_list),
            },
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer_str = response.choices[0].message.content
    answer_list = json.loads(answer_str)

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
        "Other",
    ]
    for answer in answer_list:
        if answer not in acceptable_answers:
            raise ValueError(f"{answer} not in list of acceptable answers")

    return dict(zip(name_list, answer_list))
```

Try that with our team list.

```python
classify_teams(team_list)
```

And you'll see that it works with only a single API call. The same technique will work for a batch of any size.

```python
{
    "Minnesota Twins": "Major League Baseball (MLB)",
    "Minnesota Vikings": "National Football League (NFL)",
    "Minnesota Timberwolves": "National Basketball Association (NBA)",
}
```

Though, as you batches get bigger, one common problem is that the number of outputs from the LLM can fail to match the number of inputs you provide. This problem may lessen as LLMs improve, but for now it's a good idea to limit to batches to a few dozen inputs and to verify that you're getting the right number back.

{emphasize-lines="66-69"}
```python
def classify_teams(name_list):
    prompt = """
You are an AI model trained to classify text.

I will provide list of professional sports team names separated by new lines

You will reply with the sports league in which they compete.

Your responses must come from the following list:
- Major League Baseball (MLB)
- National Football League (NFL)
- National Basketball Association (NBA)

If the team's league is not on the list, you should label them as "Other".

Your answers should be returned as a flat JSON list.

It is very important that the length of JSON list you return is exactly the same as the number of names you receive.

If I were to submit:

"Los Angeles Rams\nLos Angeles Dodgers\nLos Angeles Lakers\nLos Angeles Kings"

You should return the following:

["National Football League (NFL)", "Major League Baseball (MLB)", "National Basketball Association (NBA)", "Other"]
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "Chicago Bears,Chicago Cubs,Chicago Bulls,Chicago Blackhawks",
            },
            {
                "role": "assistant",
                "content": '["National Football League (NFL)", "Major League Baseball (MLB)", "National Basketball Association (NBA)", "Other"]',
            },
            {
                "role": "user",
                "content": "\n".join(name_list),
            },
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer_str = response.choices[0].message.content
    answer_list = json.loads(answer_str)

    acceptable_answers = [
        "Major League Baseball (MLB)",
        "National Football League (NFL)",
        "National Basketball Association (NBA)",
        "Other",
    ]
    for answer in answer_list:
        if answer not in acceptable_answers:
            raise ValueError(f"{answer} not in list of acceptable answers")

    try:
        assert len(name_list) == len(answer_list)
    except AssertionError:
        raise ValueError(
            f"Number of outputs ({len(name_list)}) does not equal the number of inputs ({len(answer_list)})"
        )

    return dict(zip(name_list, answer_list))
```

Okay. Naming sports teams is a cute trick, but what about something hard? And whatever happened to that George Santos idea?

We'll tackle that by pulling in our example dataset using `pandas`, a popular data manipulation library in Python.

First, we need to install it. Back to our installation cell.

```text
%pip install groq rich ipywidgets pandas
```

Then import it.

{emphasize-lines="5"}
```python
import json
from rich import print
from groq import Groq
import pandas as pd
```

Now we're ready to load the California expenditures data prepared for the class. It contains the distinct list of all vendors listed as payees in itemized receipts attached to disclosure filings.

```python
df = pd.read_csv(
    "https://raw.githubusercontent.com/palewire/first-llm-classifier/refs/heads/main/_notebooks/Form460ScheduleESubItem.csv"
)
```

Have a look at a random sample to get a taste of what's in there.

```python
df.sample(10)
```

```
payee
10292	NICK & STEF'S
13392	SPIKE TV
4705	EQUALITY FOR ALL - NO ON 8
1037	ATLANTA BOOK SELLERS
11716	QUICK MESSENGER SERVICE
12365	RUBIOS MEXICAN GRILL
7761	KIMBERLY A. AGUIRRE
223	ACCUZIP, INC.
19	2000 COPIER & FAX, INC.
7989	KOZT
```

Now let's adapt what we have to fit. Instead of asking for a sports league back, we will ask the LLM to classify each payee as a restaurant, bar, hotel or other establishment.

{emphasize-lines="2-26,33-48,61-66"}
```python
def classify_payees(name_list):
    prompt = """You are an AI model trained to categorize businesses based on their names.

You will be given a list of business names, each separated by a new line.

Your task is to analyze each name and classify it into one of the following categories: Restaurant, Bar, Hotel, or Other.

It is extremely critical that there is a corresponding category output for each business name provided as an input.

If a business does not clearly fall into Restaurant, Bar, or Hotel categories, you should classify it as "Other".

Even if the type of business is not immediately clear from the name, it is essential that you provide your best guess based on the information available to you. If you can't make a good guess, classify it as Other.

For example, if given the following input:

"Intercontinental Hotel\nPizza Hut\nCheers\nWelsh's Family Restaurant\nKTLA\nDirect Mailing"

Your output should be a JSON list in the following format:

["Hotel", "Restaurant", "Bar", "Restaurant", "Other", "Other"]

This means that you have classified "Intercontinental Hotel" as a Hotel, "Pizza Hut" as a Restaurant, "Cheers" as a Bar, "Welsh's Family Restaurant" as a Restaurant, and both "KTLA" and "Direct Mailing" as Other.

Ensure that the number of classifications in your output matches the number of business names in the input. It is very important that the length of JSON list you return is exactly the same as the number of business names youyou receive.
"""
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "Intercontinental Hotel\nPizza Hut\nCheers\nWelsh's Family Restaurant\nKTLA\nDirect Mailing",
            },
            {
                "role": "assistant",
                "content": '["Hotel", "Restaurant", "Bar", "Restaurant", "Other", "Other"]',
            },
            {
                "role": "user",
                "content": "Subway Sandwiches\nRuth Chris Steakhouse\nPolitical Consulting Co\nThe Lamb's Club",
            },
            {
                "role": "assistant",
                "content": '["Restaurant", "Restaurant", "Other", "Bar"]',
            },
            {
                "role": "user",
                "content": "\n".join(name_list),
            },
        ],
        model="llama-3.3-70b-versatile",
        temperature=0,
    )

    answer_str = response.choices[0].message.content
    answer_list = json.loads(answer_str)

    acceptable_answers = [
        "Restaurant",
        "Bar",
        "Hotel",
        "Other",
    ]
    for answer in answer_list:
        if answer not in acceptable_answers:
            raise ValueError(f"{answer} not in list of acceptable answers")

    try:
        assert len(name_list) == len(answer_list)
    except AssertionError:
        raise ValueError(
            f"Number of outputs ({len(name_list)}) does not equal the number of inputs ({len(answer_list)})"
        )

    return dict(zip(name_list, answer_list))
```

Now pull out a random sample of payees as a list.

```python
sample_list = list(df.sample(10).payee)
```

And see how it does.

```python
classify_payees(sample_list)
```

```python
{
    "ARCLIGHT CINEMAS": "Other",
    "99 CENTS ONLY": "Other",
    "COMMONWEALTH COMMUNICATIONS": "Other",
    "CHILBO MYUNOK": "Other",
    "ADAM SCHIFF FOR SENATE": "Other",
    "CENTER FOR CREATIVE FUNDING": "Other",
    "JOE SHAW FOR HUNTINGTON BEACH CITY COUNCIL 2014": "Other",
    "MULVANEY'S BUILDING & LOAN": "Other",
    "ATV VIDEO CENTER": "Other",
    "HYATT REGENCY SAN FRANCISCO": "Hotel",
}
```

That's nice for a sample. But how do you loop through the entire dataset and code them.

One way to start is to write a function that will split up a list into batches of a certain size.

```python
def get_batch_list(li, n=10):
    """Split the provided list into batches of size `n`."""
    batch_list = []
    for i in range(0, len(li), n):
        batch_list.append(li[i : i + n])
    return batch_list
```

Before we loop through our payees, let's add a couple libraries that will let us avoid hammering Groq and keep tabs on our progress.

{emphasize-lines="1,4"}
```python
import time
import json
from rich import print
from rich.progress import track
from groq import Groq
import pandas as pd
```

That batching trick can then be fit into a new function that will accept a big list of payees and classify them batch by batch.

```python
def classify_batches(name_list, batch_size=10, wait=2):
    """Split the provided list of names into batches and classify with our LLM them one by one."""
    # Create a place to store the results
    all_results = {}

    # Batch up the list
    batch_list = get_batch_list(name_list, n=batch_size)

    # Loop through the list in batches
    for batch in track(batch_list):
        # Classify it with the LLM
        batch_results = classify_payees(batch)

        # Add what we get back to the results
        all_results.update(batch_results)

        # Tap the brakes to avoid overloading groq's API
        time.sleep(wait)

    # Return the results
    return all_results
```

Now, let's take out a bigger sample.

```python
bigger_sample = list(df.sample(100).payee)
```

And let it rip.

```python
classify_batches(bigger_sample)
```

Printing out to the console is interesting, but eventually you'll want to be able to work with the results in a more structured way. So let's convert the results into a `pandas` DataFrame by modifying our function.

{emphasize-lines="20-23"}
```python
def classify_batches(name_list, batch_size=10, wait=2):
    # Store the results
    all_results = {}

    # Batch up the list
    batch_list = get_batch_list(name_list, n=batch_size)

    # Loop through the list in batches
    for batch in track(batch_list):
        # Classify it
        batch_results = classify_payees(batch)

        # Add it to the results
        all_results.update(batch_results)

        # Tap the brakes
        time.sleep(wait)

    # Return the results
    return pd.DataFrame(all_results.items(), columns=["payee", "category"])
```

Results can now be stored as a DataFrame.

```python
results_df = classify_batches(bigger_sample)
```

And inspected using the standard `pandas` tools. Like a peek at the first records:

```python
results_df.head()
```

```
payee	category
0	CARROW'S RESTAURANT	Restaurant
1	TUSCAN STEAK	Restaurant
2	ORANGE POST OFFICE	Other
3	RCC, INC.	Other
4	THE MISSION INN HOTEL & SPA	Hotel
```

Or a sum of all the categories.

```python
results_df.category.value_counts()
```

```
category
Other         65
Restaurant    19
Hotel         13
Bar            3
Name: count, dtype: int64
```
