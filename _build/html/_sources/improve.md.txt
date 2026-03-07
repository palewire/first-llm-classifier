# Improvement

With our LLM prompt showing such strong results, you might be content to leave it as it is. But there are always ways to improve, and you might come across a circumstance where the model's performance is less than ideal.

## Learn from your model's mistakes

One common tactic is to examine your model's misclassifications and tweak your prompt to address any patterns they reveal.

One simple way to do this is to merge the LLM's predictions with the human-labeled data and take a look at the discrepancies with your own eyes.

First, merge the LLM's predictions with the human-labeled data.

```python
comparison_df = llm_df.merge(
    sample_df, on="payee", how="inner", suffixes=["_llm", "_human"]
)
```

Then filter to cases where the LLM and human labels don't match.

```python
mistakes_df = comparison_df[
    comparison_df.category_llm != comparison_df.category_human
]
```

Looking at the misclassifications, you might notice that the LLM is struggling with a particular type of business name. You can then adjust your prompt to address that specific issue.

```python
mistakes_df.head(10)
```

<pre style="white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace">|     | payee                            | category_llm   | category_human   |
|----:|:---------------------------------|:---------------|:-----------------|
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">16</span> | SOTTOVOCE MADERO                 | Restaurant     | Other            |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">43</span> | SIBIA CAB                        | Bar            | Other            |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">56</span> | THE OVAL ROOM                    | Bar            | Restaurant       |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">85</span> | ELLA DINNING ROOM                | Restaurant     | Other            |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">87</span> | LAKELAND  VILLAGE                | Hotel          | Other            |
|  <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">95</span> | THE PALMS                        | Bar            | Restaurant       |
| <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">104</span> | GRUBHUB, INC.                    | Other          | Restaurant       |
| <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">136</span> | NORTHERN CALIFORNIA WINE COUNTRY | Bar            | Other            |
| <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">144</span> | MAYAHUEL                         | Bar            | Restaurant       |
| <span style="color: #008080; text-decoration-color: #008080; font-weight: bold">146</span> | TWENTY EIGHT                     | Bar            | Other            |
</pre>

You could then output the mistakes to a spreadsheet and inspect them more closely.

```python
mistakes_df.to_csv("mistakes.csv", index=False)
```

As I scanned the full list, I observed that the LLM was struggling with businesses that had both the word bar and the word restaurant in their name. A simple fix would be to add a new line to your prompt that instructs the LLM what to do in that case.

```
If a business name contains both the word "Restaurant" and
the word "Bar", you should classify it as a Restaurant.
```

## Be humble, human

Look closely at the misclassifications above and you'll see another interesting example. The first entry, "SOTTOVOCE MADERO," was classified as a restaurant by the LLM but labeled as "Other" by the human. According to our evaluation routine, the LLM got it wrong.

But a quick Google search will reveal that [Sottovoce](https://guide.michelin.com/us/en/ciudad-autonoma-de-buenos-aires/buenos-aires_777009/restaurant/sottovoce-1208879) is indeed a restaurant, found in the Madero Center neighborhood of Buenos Aires.

[![Michelin Guide entry for Sottovoce Madero](_static/sottovoce.png)](https://guide.michelin.com/us/en/ciudad-autonoma-de-buenos-aires/buenos-aires_777009/restaurant/sottovoce-1208879)

So, in this case, the LLM was actually correct and the human label was wrong, a truly humbling moment for the creators of this class. And reminder of how powerful LLMs can be at understanding and classifying data, even when the information is incomplete or ambiguous.

## Use training data as few-shot prompts

Earlier in the lesson, we showed how you can feed the LLM examples of inputs and output prior to your request as part of a "few shot" prompt. An added benefit of coding a supervised sample for testing is that you can also use the training slice of the set to prime the LLM with this technique. If you've already done the work of labeling your data, you might as well use it to improve your model as well.

Converting the training set you held to the side into a few-shot prompt is a simple matter of formatting it to fit your LLM's expected input. Here's how you might do it in our case.

```python
def get_fewshots(training_input, training_output, batch_size=10):
    """Convert the training data into a few-shot prompt"""
    # Batch up the training input into groups of `batch_size`
    input_batches = list(batched(training_input.payee, batch_size))

    # Do the same for the output
    output_batches = list(batched(training_output, batch_size))

    # Create a list to hold the formatted few-shot examples
    fewshot_list = []

    # Loop through the batches
    for input_list, output_list in zip(input_batches, output_batches):
        # Create a "user" message for the LLM
        prompt = "\n".join(input_list)

        # Serialize the expected "assistant" response using the Pydantic model
        response = PayeeList(answers=list(output_list)).model_dump_json()

        # Add both to the fewshot list in the format expected by our LLM
        fewshot_list.extend([
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ])

    # Return the list of few-shot examples, one for each batch
    return fewshot_list
```

Pass in your training data.

```python
fewshot_list = get_fewshots(training_input, training_output)
```

Take a peek at the first pair to see if it's what we expect.

```python
fewshot_list[:2]
```

```python
[
    {
        "role": "user",
        "content": "UFW OF AMERICA - AFL-CIO\nRE-ELECT FIONA MA\nELLA DINNING ROOM\nMICHAEL EMERY PHOTOGRAPHY\nLAKELAND  VILLAGE\nTHE IVY RESTAURANT\nMOORLACH FOR SENATE 2016\nBROWN PALACE HOTEL\nAPPLE STORE FARMERS MARKET\nCABLETIME TV",
    },
    {
        "role": "assistant",
        "content": '{"answers": ["Other", "Other", "Other", "Other", "Other", "Restaurant", "Other", "Hotel", "Other", "Other"]}',
    },
]
```

Return to `classify_payees` and replacing the previous hardcoded prompt with the generated few-shot list.

{emphasize-lines="7"}

```python
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            *get_fewshots(training_input, training_output),
            {
                "role": "user",
                "content": "\n".join(name_list),
            },
        ],
        model=model,
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "PayeeList",
                "schema": PayeeList.model_json_schema()
            }
        },
        temperature=0,
    )
```

And all you need to do is run it again.

```python
llm_df = classify_batches(
    test_input.payee,
    model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
)
```

How can you tell if your results have improved? You run that same classification report and see if the numbers have gone up.

```python
print(classification_report(test_output, llm_df.category))
```

Repeating this disciplined, scientific process of prompt refinement, testing and review can, after a few careful cycles, gradually improve your prompt to return even better results.
