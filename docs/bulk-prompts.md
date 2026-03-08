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

{emphasize-lines="1,5,18-25,28,32-38,42-43"}

```python
def classify_teams(name_list):
    prompt = """
You are an AI model trained to classify text.

I will provide a list of professional sports team names separated by newlines.

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

You'll see that it works with only a single API call. The same technique will work for a batch of any size. Due to LLMs' tendency to lose attention and engage in strange loops as answers get longer, start off with smaller batches, but you can experiment with what works best for your use case and see if you can push it further.

```python
{
    "Chicago Cubs": "MLB",
    "Chicago Bears": "NFL",
    "Chicago Bulls": "NBA",
}
```

## Introducing campaign finance data

Okay. Naming sports teams is a cute trick, but what about something a bit harder? And whatever happened to that George Santos idea?

We'll tackle that by pulling in our example dataset using [`pandas`](https://pandas.pydata.org/), a popular data manipulation library in Python that we installed at the start.

Import it in your top cell and rerun.

```python
import pandas as pd
```

We're ready to load the California expenditures data prepared for the class. It contains the distinct list of all vendors listed as payees in itemized receipts attached to disclosure filings.

```python
df = pd.read_csv(
    "https://palewi.re/docs/first-llm-classifier/_static/Form460ScheduleESubItem.csv"
)
```

Have a look at a random sample to get a taste of what's in there.

```python
df.sample(10)
```

```{code-block} text
|    |   index | payee                               |
|---:|--------:|:------------------------------------|
|  0 |   14901 | THE STATIONERY STUDIO               |
|  1 |    1389 | BELL WINE AND SPIRITS               |
|  2 |   10472 | NEWSOM FOR CALIFORNIA GOVERNOR 2022 |
|  3 |   11301 | PASADENA JOURNAL NEWS               |
|  4 |    3133 | CLEARMAN'S STEAK AND STEIN          |
|  5 |    4606 | EL SAUZ TACOS                       |
|  6 |    5491 | FRIENDS OF MARK TWAIN MIDDLE SCHOOL |
|  7 |    5050 | FAT CITY                            |
|  8 |   11294 | PARVINDER KANG - PETTY CASHIER      |
|  9 |   11410 | PEARL'S CAFE                        |
```

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
- Add a check to make sure the LLM returned the right number of answers.
- Return a DataFrame instead of a dictionary to make it easier to work with downstream.

Here's where that ends up

{emphasize-lines="2-13,21-36,43-49,53-56"}

```python
@stamina.retry(on=Exception, attempts=3)
def classify_payees(name_list):
    prompt = """
You are an AI model trained to categorize businesses based on their names.

You will be given a list of business names, each separated by a new line.

Your task is to analyze each name and classify it into one of the following categories: Restaurant, Bar, Hotel, or Other.

If a business does not clearly fall into Restaurant, Bar, or Hotel categories, you should classify it as "Other".

Even if the type of business is not immediately clear from the name, it is essential that you provide your best guess based on the information available to you. If you can't make a good guess, classify it as Other.
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
    assert len(result.answers) == len(name_list), \
        f"Expected {len(name_list)} answers but got {len(result.answers)}"
    return pd.DataFrame({"payee": name_list, "category": result.answers})
```

Pull out a random sample of payees as a list.

```python
sample_list = df.sample(10).payee
```

See how it does.

```python
classify_payees(sample_list)
```

```{code-block} text
|    | payee                                           | category   |
|---:|:------------------------------------------------|:-----------|
|  0 | ARCLIGHT CINEMAS                                | Other      |
|  1 | 99 CENTS ONLY                                   | Other      |
|  2 | COMMONWEALTH COMMUNICATIONS                     | Other      |
|  3 | CHILBO MYUNOK                                   | Other      |
|  4 | ADAM SCHIFF FOR SENATE                          | Other      |
|  5 | CENTER FOR CREATIVE FUNDING                     | Other      |
|  6 | JOE SHAW FOR HUNTINGTON BEACH CITY COUNCIL 2014 | Other      |
|  7 | MULVANEY'S BUILDING & LOAN                      | Other      |
|  8 | ATV VIDEO CENTER                                | Other      |
|  9 | HYATT REGENCY SAN FRANCISCO                     | Hotel      |
```

## Submitting in batches

That's nice for a sample. But how do you loop through the entire dataset and code it?

Let's add a couple libraries to our imports cell that will let us avoid hammering Hugging Face and keep tabs on our progress.

```python
import time
from itertools import batched
from tqdm.auto import tqdm
```

That batching trick can then be fit into a new function that will accept a big list of payees and classify them batch by batch.

```python
def classify_batches(name_list, batch_size=5, wait=1):
    """Split the provided list of names into batches and classify with our LLM one by one."""
    # Create a place to store the results
    all_results = []

    # Create a list that will split the name_list into batches
    batch_list = list(batched(list(name_list), batch_size))

    # Loop through the list in batches
    for batch in tqdm(batch_list, desc="Classifying batches..."):
        # Classify it with the LLM
        batch_df = classify_payees(list(batch))

        # Add what we get back to the results
        all_results.append(batch_df)

        # Tap the brakes to avoid overloading Hugging Face's API
        time.sleep(wait)

    # Combine the batch results into a single DataFrame
    return pd.concat(all_results, ignore_index=True)
```

Since `classify_payees` now returns a DataFrame, `classify_batches` collects them into a list and concatenates them at the end. This makes the results easy to work with.

Select a bigger sample.

```python
bigger_sample = df.sample(100).payee
```

Fire away.

```python
results_df = classify_batches(bigger_sample)
```

Once it finishes, the results will be returned as a DataFrame that you can inspect using the standard `pandas` tools. Like a peek at the first ten records:

```python
results_df.head(10)
```

```{code-block} text
|    | payee                               | category   |
|---:|:------------------------------------|:-----------|
|  0 | PETE'S CAFE                         | Restaurant |
|  1 | ALAMO RENT-A-CAR SANTA ANA          | Other      |
|  2 | COMMITTEE TO ELECT ELENA            | Other      |
|  3 | FAIRMONT SAN JOSE                   | Hotel      |
|  4 | BEST WESTERN HOTELS SAN SIMEON      | Hotel      |
|  5 | VAROGA RICEW & SHALETT, INC.        | Other      |
|  6 | EL SEGUNDO HERALD                   | Other      |
|  7 | MCI                                 | Other      |
|  8 | TIKAL RESTAURANT                    | Restaurant |
|  9 | PEREA FOR SCCCD TRUSTEE AREA 5 2018 | Other      |
```

Or do a bit of analysis of the categories.

```python
results_df.category.value_counts()
```

```{code-block} text
|    | category   |   count |
|---:|:-----------|--------:|
|  0 | Other      |      61 |
|  1 | Restaurant |      19 |
|  2 | Hotel      |      17 |
|  3 | Bar        |       3 |
```

Now you can start to dig into the details and see which payees were classified in which categories, and maybe even spot some interesting patterns in the data.

## Submitting in parallel

Our batching routine is a big improvement, but it still processes one batch at a time. Each batch has to finish before the next one starts. When you're working through thousands of payees, that wait adds up.

Python's standard library offers a simple way to speed things up. The [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) module can send multiple batches to the API at the same time.

Add it to your imports cell.

```python
from concurrent.futures import ThreadPoolExecutor
```

Now we can write a new version of our batching function that submits requests in parallel instead of waiting in line.

```python
def classify_batches_parallel(name_list, batch_size=5, max_workers=4):
    """Split the provided list of names into batches and classify with our LLM in parallel."""
    # Create a list that will split the name_list into batches
    batch_list = list(batched(list(name_list), batch_size))

    # Submit all the batches in parallel and collect results in order
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        all_results = list(
            tqdm(
                executor.map(classify_payees, [list(b) for b in batch_list]),
                total=len(batch_list),
                desc="Classifying batches...",
            )
        )

    # Combine the batch results into a single DataFrame
    return pd.concat(all_results, ignore_index=True)
```

The key change is small but powerful. Instead of a `for` loop that processes one batch at a time, we use a `ThreadPoolExecutor` to fire off all our batches at once. The `max_workers` argument controls how many can run simultaneously. The executor's `map` method collects results in the same order as the input, keeping our data lined up correctly. That doesn't cost us any speed — all the batches still run concurrently. The only difference is that `map` hands them back in the order they were submitted, rather than whichever finished first. Wrapping it in `tqdm` keeps our progress bar ticking. And since `classify_payees` already has the `@stamina.retry` decorator, any failed requests will be retried automatically — even when running in parallel.

Try it with the same sample.

```python
results_df = classify_batches_parallel(bigger_sample)
```

You should see the same results, but faster. With four workers running at once, you're making the most of the time you'd otherwise spend waiting for the API.

If you're working with a particularly large dataset, you can experiment with the `max_workers` parameter. Start low and increase it gradually. Push it too high and you'll risk overwhelming the API, but a modest number like four or five can cut your processing time dramatically.

If you ran this routine across the full list of payees, you could merge the results with the line-item spending reported by candidates and quickly calculate which candidates spent the most on hotels, or bars, or restaurants. If someone had a system like this in place during the 2022 election cycle, they might have been able to flag George Santos' suspicious spending patterns months before The New York Times broke the story.
