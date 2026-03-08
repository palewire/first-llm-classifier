# Bulk prompts

Our classifier works well for a sample. But how do you loop through the entire dataset and code it? There are thousands of payees to classify, and sending them all in a single request could overwhelm the LLM.

The solution is to split the full list into smaller batches and process them one by one.

Let's add a couple libraries to our imports cell that will let us avoid hammering Hugging Face and keep tabs on our progress.

```python
import time
from itertools import batched
from tqdm.auto import tqdm
```

We can fit this approach into a new function that will accept a big list of payees and classify them batch by batch.

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

Since `classify_payees` returns a DataFrame, `classify_batches` collects them into a list and concatenates them at the end. This makes the results easy to work with.

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

The key change is small but powerful.

Instead of a `for` loop that processes one batch at a time, we use a `ThreadPoolExecutor` to fire off all our batches at once. The `max_workers` argument controls how many can run simultaneously. The executor's `map` method collects results in the same order as the input, keeping our data lined up correctly. That doesn't cost us any speed — all the batches still run concurrently. The only difference is that `map` hands them back in the order they were submitted, rather than whichever finished first.

Wrapping it in `tqdm` keeps our progress bar ticking. And since `classify_payees` already has the `@stamina.retry` decorator, any failed requests will be retried automatically — even when running in parallel.

Try it with the same sample.

```python
results_df = classify_batches_parallel(bigger_sample)
```

You should see the same results, but faster. With four workers running at once, you're making the most of the time you'd otherwise spend waiting for the API.

If you're working with a particularly large dataset, you can experiment with the `max_workers` parameter. Start low and increase it gradually. Push it too high and you'll risk overwhelming the API, but a modest number like four or five can cut your processing time dramatically.

If you ran this routine across the full list of payees, you could merge the results with the line-item spending reported by candidates and quickly calculate which candidates spent the most on hotels, or bars, or restaurants. If someone had a system like this in place during the 2022 election cycle, they might have been able to flag George Santos' suspicious spending patterns months before The New York Times broke the story.
