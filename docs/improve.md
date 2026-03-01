# Improvement

With our LLM prompt showing such strong results, you might be content to leave it as it is. But there are always ways to improve, and you might come across a circumstance where the model's performance is less than ideal.

## Learn from your model's mistakes

One common tactic is to examine your model's misclassifications and tweak your prompt to address any patterns they reveal.

One simple way to do this is to merge the LLM's predictions with the human-labeled data and filter for discrepancies.

```python
comparison_df = llm_df.merge(
    sample_df, on="payee", how="inner", suffixes=["_llm", "_human"]
)
```

And filter to cases where the LLM and human labels don't match.

```python
comparison_df[comparison_df.category_llm != comparison_df.category_human]
```

Looking at the misclassifications, you might notice that the LLM is struggling with a particular type of business name. You can then adjust your prompt to address that specific issue.

```python
comparison_df.head()
```

In this case, I observed that the LLM was struggling with businesses that had both the word bar and the word restaurant in their name. A simple fix would be to add a new line to your prompt that instructs the LLM what to do in that case.

{emphasize-lines="23"}

```python
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

If a business name contains both the word "Restaurant" and the word "Bar", you should classify it as a Restaurant.

Ensure that the number of classifications in your output matches the number of business names in the input. It is very important that the length of JSON list you return is exactly the same as the number of business names you receive.
"""
```

Repeating this disciplined, scientific process of prompt refinement, testing and review can, after a few careful cycles, gradually improve your prompt to return even better results.

## Use training data as few shot prompts

Earlier in the lesson, we showed how you can feed the LLM examples of inputs and output prior to your request as part of a "few shot" prompt. An added benefit of coding a supervised sample for testing is that you can also use the training slice of the set to prime the LLM with this technique. If you've already done the work of labeling your data, you might as well use it to improve your model as well.

Converting the training set you held to the side into a few-shot prompt is a simple matter of formatting it to fit your LLM's expected input. Here's how you might do it in our case.

```python
def get_fewshots(training_input, training_output, batch_size=10):
    """Convert the training input and output from sklearn's train_test_split into a few-shot prompt"""
    # Batch up the training input into groups of `batch_size`
    input_batches = list(batched(training_input.payee, batch_size))

    # Do the same for the output
    output_batches = list(batched(training_output, batch_size))

    # Create a list to hold the formatted few-shot examples
    fewshot_list = []

    # Loop through the batches
    for i, input_list in enumerate(input_batches):
        fewshot_list.extend(
            [
                # Create a "user" message for the LLM formatted the same was a our prompt with newlines
                {
                    "role": "user",
                    "content": "\n".join(input_list),
                },
                # Create the expected "assistant" response as the JSON formatted output we expect
                {"role": "assistant", "content": json.dumps(list(output_batches[i]))},
            ]
        )

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
        "content": '["Other", "Other", "Other", "Other", "Other", "Restaurant", "Other", "Hotel", "Other", "Other"]',
    },
]
```

Now, we can add those examples to our prompt's `messages`.

{emphasize-lines="29-37"}

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

Ensure that the number of classifications in your output matches the number of business names in the input. It is very important that the length of JSON list you return is exactly the same as the number of business names you receive.
"""
    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            *fewshot_list,
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
    except:
        raise ValueError(
            f"Number of outputs ({len(name_list)}) does not equal the number of inputs ({len(answer_list)})"
        )

    return dict(zip(name_list, answer_list))
```

And all you need to do is run it again.

```python
llm_df = classify_batches(test_input.payee)
```

How can you tell if your results have improved? You run that same classification report and see if the numbers have gone up.

```python
print(
    classification_report(
        test_output,
        llm_df.category,
    )
)
```

Get used to this process of iterating on your prompt, testing, and reviewing the results. It's the key to getting the best performance out of your LLM.
