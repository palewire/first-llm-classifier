# Bulk prompts

Our reusable prompting function is pretty cool. But requesting answers one by one across a big dataset could take forever. And it could cost us money to hit the Hugging Face API so many times.

One solution is to submit your requests in batches and ask the LLM to return its responses all at once.

We'll need to make a series of changes to our function to adapt it to work with a batch of inputs. Get ready. It's a lot.

We define a new Pydantic model for bulk responses that expects a list of answers.

```python
class SportsLeagueList(BaseModel):
    answers: list[Literal["MLB", "NFL", "NBA", "Other"]]
```

Then, in our function, we make the following changes:

- Tweak the name of the classify function to `classify_teams` to reflect the fact that it will now accept a list of team names instead of just one.
- Change our input argument from a single value to a list of values.
- Expand our prompt to explain that we will provide a list of team names.
- Tweak our few-shot training to reflect this new approach.
- Submit our input as a single string with new lines separating each team name.
- Merge the team names and the LLM's answers into a dictionary returned by the function.

Put all that together and here's where we land.

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
                "content": "Los Angeles Rams\nLos Angeles Dodgers\nLos Angeles Lakers\nLos Angeles Kings",
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
                "name": "SportsLeagueList",
                "schema": SportsLeagueList.model_json_schema()
            }
        },
        temperature=0,
    )

    result = SportsLeagueList.model_validate_json(response.choices[0].message.content)
    return dict(zip(name_list, result.answers))
```

Try that with our team list.

```python
classify_teams(team_list)
```

You'll see that it works with only a single API call. The same technique will work for a batch of any size. Due to LLMs' tendency to lose attention and engage in strange loops as answers get longer, start off with smaller batches of around 10 to 20 items, but you can experiment with what works best for your use case and see if you can push it further.

```python
{
    "Chicago Cubs": "MLB",
    "Chicago Bears": "NFL",
    "Chicago Bulls": "NBA",
}
```

## Introducing campaign finance data

Okay. Naming sports teams is a cute trick, but what about something a bit harder? And whatever happened to that George Santos idea?

We'll tackle that by pulling in our example dataset using [`pandas`](https://pandas.pydata.org/), a popular data manipulation library in Python.

First, we need to install it. Run another cell like this:

```text
!uv add pandas
```

Import it in your top cell and rerun.

{emphasize-lines="5"}

```python
from pydantic import BaseModel
from typing import Literal
from rich import print
from huggingface_hub import InferenceClient
import pandas as pd
```

We're ready to load the California expenditures data prepared for the class. It contains the distinct list of all vendors listed as payees in itemized receipts attached to disclosure filings.

```python
df = pd.read_csv(
    "https://raw.githubusercontent.com/palewire/first-llm-classifier/refs/heads/main/_notebooks/Form460ScheduleESubItem.csv"
)
```

Have a look at a random sample to get a taste of what's in there.

```python
df.sample(10)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">|    |   index | payee                               |
|---:|--------:|:------------------------------------|
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> |   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">14901</span> | THE STATIONERY STUDIO               |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> |    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1389</span> | BELL WINE AND SPIRITS               |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span> |   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">10472</span> | NEWSOM FOR CALIFORNIA GOVERNOR <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2022</span> |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span> |   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11301</span> | PASADENA JOURNAL NEWS               |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span> |    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3133</span> | CLEARMAN'S STEAK AND STEIN          |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span> |    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4606</span> | EL SAUZ TACOS                       |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span> |    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5491</span> | FRIENDS OF MARK TWAIN MIDDLE SCHOOL |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span> |    <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5050</span> | FAT CITY                            |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span> |   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11294</span> | PARVINDER KANG - PETTY CASHIER      |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span> |   <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">11410</span> | PEARL'S CAFE                        |
</pre>

Let's adapt what we've learned so far to fit this data.

Instead of asking for a sports league, we will ask the LLM to classify each payee as a restaurant, bar, hotel or other establishment. The kind of places where George Santos, and other politicians like him, might enjoy spending campaign funds.

Start by creating a new Pydantic model for the payee classifications.

```python
class PayeeList(BaseModel):
    answers: list[Literal["Restaurant", "Bar", "Hotel", "Other"]]
```

Then we will:

- Rename our function to `classify_payees`.
- Rewrite our prompt to explain the new task and categories.
- Update our few-shot training examples to reflect the new task.
- Swap in the new model for the response format and validation.

{emphasize-lines="1-22,30-45,55-56,62"}

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
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "PayeeList",
                "schema": PayeeList.model_json_schema()
            }
        },
        temperature=0,
    )

    result = PayeeList.model_validate_json(response.choices[0].message.content)
    return dict(zip(name_list, result.answers))
```

Pull out a random sample of payees as a list.

```python
sample_list = list(df.sample(10).payee)
```

See how it does.

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

## Submitting in batches

That's nice for a sample. But how do you loop through the entire dataset and code it?

Let's add a couple libraries that will let us avoid hammering Hugging Face and keep tabs on our progress.

{emphasize-lines="3,4,6"}

```python
from pydantic import BaseModel
from typing import Literal
import time
from itertools import batched
from rich import print
from rich.progress import track
from huggingface_hub import InferenceClient
import pandas as pd
```

That batching trick can then be fit into a new function that will accept a big list of payees and classify them batch by batch.

```python
def classify_batches(name_list, batch_size=20, wait=1):
    """Split a list of names into batches and classify them one by one."""
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

    # Return the results as a pandas DataFrame
    return pd.DataFrame(all_results.items(), columns=["payee", "category"])
```

Look closely at that last line and you'll see that the function is now returning a DataFrame instead of a dictionary. This will make what we back easier to work with and analyze.

Select a bigger sample.

```python
bigger_sample = list(df.sample(100).payee)
```

Fire away.

```python
results_df = classify_batches(bigger_sample)
```

Once it finishes, the results will be returned as a DataFrame that you can inspect using the standard `pandas` tools. Like a peek at the first ten records:

```python
results_df.head(10)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">|    | payee                               | category   |
|---:|:------------------------------------|:-----------|
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> | PETE'S CAFE                         | Restaurant |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> | ALAMO RENT-A-CAR SANTA ANA          | Other      |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span> | COMMITTEE TO ELECT ELENA            | Other      |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span> | FAIRMONT SAN JOSE                   | Hotel      |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">4</span> | BEST WESTERN HOTELS SAN SIMEON      | Hotel      |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span> | VAROGA RICEW &amp; SHALETT, INC.        | Other      |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">6</span> | EL SEGUNDO HERALD                   | Other      |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">7</span> | MCI                                 | Other      |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">8</span> | TIKAL RESTAURANT                    | Restaurant |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">9</span> | PEREA FOR SCCCD TRUSTEE AREA <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">5</span> <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2018</span> | Other      |
</pre>

Or do a bit of analysis a sum of all the categories.

```python
results_df.category.value_counts()
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">|    | category   |   count |
|---:|:-----------|--------:|
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">0</span> | Other      |      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">61</span> |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">1</span> | Restaurant |      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">19</span> |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">2</span> | Hotel      |      <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">17</span> |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span> | Bar        |       <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">3</span> |
</pre>
