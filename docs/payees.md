# Campaign spending

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

Rather than submitting names one at a time, we can send the LLM a list of payees and ask it to classify them all at once. This saves time and API calls.

Start by creating a new Pydantic model for the payee classifications.

```python
class PayeeList(BaseModel):
    answers: list[Literal["Restaurant", "Bar", "Hotel", "Other"]]
```

Then we will:

- Write a function called `classify_payees` that accepts a list of payee names.
- Write a prompt that explains the new task and categories.
- Include few-shot training examples to guide the LLM.
- Use the new model for the response format and validation.
- Add a check to make sure the LLM returned the right number of answers.
- Return a DataFrame to make it easier to work with downstream.

Here's where that ends up.

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

You'll see that our classifier works with only a single API call. The same technique will work for a batch of any size. Due to LLMs' tendency to lose attention and engage in strange loops as answers get longer, start off with smaller batches, but you can experiment with what works best for your use case and see if you can push it further.
