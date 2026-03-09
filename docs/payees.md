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

Start by creating a new Pydantic model for the payee classification.

```python
class Payee(BaseModel):
    answer: Literal["Restaurant", "Bar", "Hotel", "Other"]
```

Then we will:

- Write a function called `classify_payee` that accepts a single payee name.
- Write a prompt that explains the new task and categories.
- Include few-shot training examples to guide the LLM.
- Use the new model for the response format and validation.

Here's where that ends up.

```python
@stamina.retry(on=Exception, attempts=3)
def classify_payee(name):
    prompt = """
You are an AI model trained to categorize businesses based on their names.

Your task is to analyze a business name and classify it into one of the following categories: Restaurant, Bar, Hotel, or Other.

If a business does not clearly fall into Restaurant, Bar, or Hotel categories, classify it as "Other".

Even if the type of business is not immediately clear from the name, provide your best guess based on the information available. If you can't make a good guess, classify it as Other.
"""

    response = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": prompt,
            },
            {
                "role": "user",
                "content": "Intercontinental Hotel",
            },
            {
                "role": "assistant",
                "content": '{"answer": "Hotel"}',
            },
            {
                "role": "user",
                "content": "Pizza Hut",
            },
            {
                "role": "assistant",
                "content": '{"answer": "Restaurant"}',
            },
            {
                "role": "user",
                "content": name,
            },
        ],
        model="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8",
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "Payee",
                "schema": Payee.model_json_schema()
            }
        },
        temperature=0,
    )

    result = Payee.model_validate_json(response.choices[0].message.content)
    return result.answer
```

Try it with a single payee.

```python
classify_payee("HYATT REGENCY SAN FRANCISCO")
```

```{code-block} text
'Hotel'
```
