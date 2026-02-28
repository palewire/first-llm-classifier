# Bulk prompts

Our reusable prompting function is pretty cool. But requesting answers one by one across a big dataset could take forever. And it could cost us money to hit the Hugging Face API so many times.

One solution is to submit your requests in batches and then ask the LLM to return its responses in bulk.

We'll need to make a series of changes to our function to adapt it to work with a batch of inputs. Get ready. It's a lot.

- We define a new Pydantic model for bulk responses that expects a list of answers.
- We tweak the name of the classify function.
- We change our input argument to a list.
- We expand our prompt to explain that we will provide a list of team names.
- We ask the LLM to classify them individually, returning its answers in a JSON object.
- We tweak our few-shot training to reflect this new approach.
- We submit our input as a single string with new lines separating each team name.
- We merge the team names and the LLM's answers into a dictionary returned by the function.

```python
class BulkLeagueSchema(BaseModel):
    answers: list[Literal["MLB", "NFL", "NBA", "Other"]]
```

{emphasize-lines="3-9,25-36,39,44-45"}

```python
def classify_teams(name_list):
    prompt = """
You are an AI model trained to classify text.

I will provide list of professional sports team names separated by new lines

You will reply with the sports league in which they compete.

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
                "content": "Chicago Bears\nChicago Cubs\nChicago Bulls\nChicago Blackhawks",
            },
            {
                "role": "assistant",
                "content": '{"answers": ["NFL", "MLB", "NBA", "Other"]}',
            },
            {
                "role": "user",
                "content": "\n".join(name_list),
            },
        ],
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "BulkLeagueSchema",
                "schema": BulkLeagueSchema.model_json_schema()
            }
        },
        temperature=0,
    )

    result = BulkLeagueSchema.model_validate_json(response.choices[0].message.content)
    return dict(zip(name_list, result.answers))
```

Try that with our team list.

```python
classify_teams(team_list)
```

And you'll see that it works with only a single API call. The same technique will work for a batch of any size. Due to LLMs' tendency to lose attention and engage in strange loops as answers get longer, it's generally recommended to stick with batches of 10 or so.

```python
{
    "Chicago Cubs": "MLB",
    "Chicago Bears": "NFL",
    "Chicago Bulls": "NBA",
}
```

Okay. Naming sports teams is a cute trick, but what about something hard? And whatever happened to that George Santos idea?

We'll tackle that by pulling in our example dataset using `pandas`, a popular data manipulation library in Python.

First, we need to install it. Back to our installation cell.

```text
!uv add huggingface_hub rich ipywidgets pandas
```

Then import it.

{emphasize-lines="5"}

```python
from pydantic import BaseModel
from typing import Literal
from rich import print
from huggingface_hub import InferenceClient
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

{emphasize-lines="1-2,4-22,26-29,38-53"}

```python
class BulkPayeeSchema(BaseModel):
    answers: list[Literal["Restaurant", "Bar", "Hotel", "Other"]]

def classify_payees(name_list):
    prompt = """
You are an AI model trained to categorize businesses based on their names.

You will be given a list of business names, each separated by a new line.

Your task is to analyze each name and classify it into one of the following categories: Restaurant, Bar, Hotel, or Other.

If a business does not clearly fall into Restaurant, Bar, or Hotel categories, you should classify it as "Other".

Even if the type of business is not immediately clear from the name, it is essential that you provide your best guess based on the information available to you. If you can't make a good guess, classify it as Other.

For example, if given the following input:

"Intercontinental Hotel\nPizza Hut\nCheers\nWelsh's Family Restaurant\nKTLA\nDirect Mailing"

Your output should be a JSON object in the following format:

{"answers": ["Hotel", "Restaurant", "Bar", "Restaurant", "Other", "Other"]}

This means that you have classified "Intercontinental Hotel" as a Hotel, "Pizza Hut" as a Restaurant, "Cheers" as a Bar, "Welsh's Family Restaurant" as a Restaurant, and both "KTLA" and "Direct Mailing" as Other.
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
                "content": '{"answers": ["Hotel", "Restaurant", "Bar", "Restaurant", "Other", "Other"]}',
            },
            {
                "role": "user",
                "content": "Subway Sandwiches\nRuth Chris Steakhouse\nPolitical Consulting Co\nThe Lamb's Club",
            },
            {
                "role": "assistant",
                "content": '{"answers": ["Restaurant", "Restaurant", "Other", "Bar"]}',
            },
            {
                "role": "user",
                "content": "\n".join(name_list),
            },
        ],
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "BulkPayeeSchema",
                "schema": BulkPayeeSchema.model_json_schema()
            }
        },
        temperature=0,
    )

    result = BulkPayeeSchema.model_validate_json(response.choices[0].message.content)
    return dict(zip(name_list, result.answers))
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

That's nice for a sample. But how do you loop through the entire dataset and code it?

Let's add a couple libraries that will let us avoid hammering Hugging Face and keep tabs on our progress.

{emphasize-lines="3,5,8"}

```python
from pydantic import BaseModel
from typing import Literal
import time
from rich import print
from rich.progress import track
from huggingface_hub import InferenceClient
import pandas as pd
from itertools import batched
```

That batching trick can then be fit into a new function that will accept a big list of payees and classify them batch by batch.

```python
def classify_batches(name_list, batch_size=10, wait=2):
    """Split the provided list of names into batches and classify with our LLM them one by one."""
    # Create a place to store the results
    all_results = {}

    # Loop through the list in batches
    for batch in track(batched(name_list, batch_size)):
        # Classify it with the LLM
        batch_results = classify_payees(list(batch))

        # Add what we get back to the results
        all_results.update(batch_results)

        # Tap the brakes to avoid overloading Hugging Face's API
        time.sleep(wait)

    # Return the results
    return all_results
```

Printing out to the console is interesting, but with a bigger sample you'll want to be able to work with the results in a more structured way. First, let's update our model to `gpt-oss-20b` from OpenAI which is a bit faster for basic classification tasks like this one.

{emphasize-lines="52"}

```python
def classify_payees(name_list):
    prompt = """
You are an AI model trained to categorize businesses based on their names.

You will be given a list of business names, each separated by a new line.

Your task is to analyze each name and classify it into one of the following categories: Restaurant, Bar, Hotel, or Other.

If a business does not clearly fall into Restaurant, Bar, or Hotel categories, you should classify it as "Other".

Even if the type of business is not immediately clear from the name, it is essential that you provide your best guess based on the information available to you. If you can't make a good guess, classify it as Other.

For example, if given the following input:

"Intercontinental Hotel\nPizza Hut\nCheers\nWelsh's Family Restaurant\nKTLA\nDirect Mailing"

Your output should be a JSON object in the following format:

{"answers": ["Hotel", "Restaurant", "Bar", "Restaurant", "Other", "Other"]}

This means that you have classified "Intercontinental Hotel" as a Hotel, "Pizza Hut" as a Restaurant, "Cheers" as a Bar, "Welsh's Family Restaurant" as a Restaurant, and both "KTLA" and "Direct Mailing" as Other.
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
                "content": '{"answers": ["Hotel", "Restaurant", "Bar", "Restaurant", "Other", "Other"]}',
            },
            {
                "role": "user",
                "content": "Subway Sandwiches\nRuth Chris Steakhouse\nPolitical Consulting Co\nThe Lamb's Club",
            },
            {
                "role": "assistant",
                "content": '{"answers": ["Restaurant", "Restaurant", "Other", "Bar"]}',
            },
            {
                "role": "user",
                "content": "\n".join(name_list),
            },
        ],
        model="openai/gpt-oss-20b",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "BulkPayeeSchema",
                "schema": BulkPayeeSchema.model_json_schema()
            }
        },
        temperature=0,
    )

    result = BulkPayeeSchema.model_validate_json(response.choices[0].message.content)
    return dict(zip(name_list, result.answers))
```

Next, let's convert the results into a `pandas` DataFrame by modifying our batching function.

{emphasize-lines="16-17"}

```python
def classify_batches(name_list, batch_size=10, wait=2):
    # Store the results
    all_results = {}

    # Loop through the list in batches
    for batch in track(batched(name_list, batch_size)):
        # Classify it
        batch_results = classify_payees(list(batch))

        # Add it to the results
        all_results.update(batch_results)

        # Tap the brakes
        time.sleep(wait)

    # Return the results
    return pd.DataFrame(all_results.items(), columns=["payee", "category"])
```

Results can now be stored as a DataFrame.

```python
bigger_sample = list(df.sample(100).payee)
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

The results could be written out to a CSV file for later use.

```python
results_df.to_csv("payee_classifications.csv", index=False)
```
