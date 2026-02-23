# Evaluating prompts

Before the advent of large-language models, machine-learning systems were trained using a technique called [supervised learning](https://en.wikipedia.org/wiki/Supervised_learning). This approach required users to provide carefully prepared training data that showed the computer what was expected.

For instance, if you were developing a model to distinguish spam emails from legitimate ones, you would need to provide the model with a set of spam emails and another set of legitimate emails. The model would then use that data to learn the relationships between the inputs and outputs, which it could then apply to new emails.

In addition to training the model, the curated input would be used to evaluate the model's performance. This process typically involved splitting the supervised data into two sets: one for training and one for testing. The model could then be evaluated using a separate set of supervised data to ensure it could generalize beyond the examples it had been fed during training.

Large-language models operate differently. They are trained on vast amounts of text and can generate responses based on the relationships they derive from various machine-learning approaches. The result is that they can be used to perform a wide range of tasks without requiring supervised data to be prepared beforehand.

This is a significant advantage. However, it also raises questions about evaluating an LLM prompt. If we don’t have a supervised sample to test its results, how do we know if it’s doing a good job? How can we improve its performance if we can’t see where it gets things wrong?

In the final chapters, we will show how traditional supervision can still play a vital role in evaluating and improving an LLM prompt.

Start by outputting a random sample from the dataset to a file of comma-separated values. It will serve as our supervised sample. In general, the larger the sample the better the evaluation. But at a certain point the returns diminish. For this exercise, we will use a sample of 250 records.

```python
df.sample(250).to_csv("./sample.csv", index=False)
```

You can open the file in a spreadsheet program like Excel or Google Sheets. For each payee in the sample, you would provide the correct category in a companion column. This gradually becomes the supervised sample.

![Sample](_static/sample.png)

To speed the class along, we've already prepared a sample for you in [the class repository](https://github.com/palewire/first-llm-classifier). Our next step is to read it back into a DataFrame.

```python
sample_df = pd.read_csv(
    "https://raw.githubusercontent.com/palewire/first-llm-classifier/refs/heads/main/_notebooks/sample.csv"
)
```

We'll install the Python packages `scikit-learn`, `matplotlib`, and `seaborn`. Prior to LLMs, these libraries were the go-to tools for training and evaluating machine-learning models. We'll primarily be using them for testing.

Return to the Jupyter notebook and install the packages alongside our other dependencies.

```
!uv add huggingface_hub rich ipywidgets pandas scikit-learn matplotlib seaborn
```

Add the `train_test_split` function from `scikit-learn` to the import statement.

{emphasize-lines="8"}

```python
import json
import time
from rich import print
from rich.progress import track
from huggingface_hub import InferenceClient
import pandas as pd
from itertools import batched
from sklearn.model_selection import train_test_split
```

This tool is used to split a supervised sample into separate sets for training and testing.

The first input is the DataFrame column containing our supervised payees. The second input is the DataFrame column containing the correct categories.

The `test_size` parameter determines the proportion of the sample that will be used for testing. The `random_state` parameter ensures that the split is reproducible by setting a seed for the random number generator that draws the samples.

```python
training_input, test_input, training_output, test_output = train_test_split(
    sample_df[["payee"]],
    sample_df["category"],
    test_size=0.33,
    random_state=42,  # Remember Jackie Robinson. Remember Douglas Adams.
)
```

In a traditional training setup, the next step would be to train a machine-learning model in `sklearn` using the `training_input` and `training_output` sets. The model would then be evaluated using the `test_input` and `test_output` sets.

With the LLM we skip ahead to the testing phase. We pass the `test_input` set to our LLM prompt and compare the results to the right answers found in `test_output` set.

All that requires is that we pass the `payee` column from our `test_input` DataFrame to the function we created in the previous chapters.

```python
llm_df = classify_batches(list(test_input.payee))
```

Next, we import the `classification_report` and `confusion_matrix` functions from `sklearn`, which are used to evaluate a model's performance. We'll also pull in `seaborn` and `matplotlib` to visualize the results.

{emphasize-lines="9-11"}

```python
import json
import time
from rich import print
from rich.progress import track
from huggingface_hub import InferenceClient
import pandas as pd
from itertools import batched
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
```

The `classification_report` function generates a report card on a model's performance. You provide it with the correct answers in the `test_output` set and the model's predictions in your prompt's DataFrame. In this case, our LLM's predictions are stored in the `llm_df` DataFrame's `category` column.

```python
print(classification_report(test_output, llm_df.category))
```

That will output a report that looks something like this:

```
              precision    recall  f1-score   support

         Bar       0.25      0.50      0.33         2
       Hotel       1.00      0.78      0.88         9
       Other       0.95      0.93      0.94        57
  Restaurant       0.88      0.93      0.90        15

    accuracy                           0.90        83
   macro avg       0.77      0.79      0.76        83
weighted avg       0.92      0.90      0.91        83
```

At first, the report can be a bit overwhelming. What are all these technical terms?

Precision measures what statistics nerds call "positive predictive value." It's how often the model made the correct decision when it applied a category. For instance, in the "Bar" category here, the LLM has a precision of 0.25, which means only one out of four "Bar" predictions was correct. An analogy here is a baseball player's contact rate. Precision is a measure of how often the model connects with the ball when it swings its bat.

Recall measures how many of the supervised instances were identified by the model. In this case, it shows that the LLM correctly spotted about 78% of the hotels in our manual sample.

The f1-score is a combination of precision and recall. It's a way to measure a model's overall performance by balancing the two.

The support column shows how many instances of each category were in the supervised sample.

The averages at the bottom combine the results for all categories. The macro row is a simple average of all the scores in that column. The weighted row is a weighted average based on the number of instances in each category.

In the example result above, the overall accuracy is about 90%, but the lower macro average (0.76) shows the model is less consistent on rarer categories.

Another technique for evaluating classifiers is to visualize the results using a chart known as a confusion matrix. This chart shows how often the model correctly predicted each category and where it got things wrong.

Drawing one up requires the `confusion_matrix` function from `sklearn` and an embarrassing tangle of code from `seaborn` and `matplotlib` libraries. Most of it is boilerplate, but you need to punch your test variables, as well as the proper labels for the categories, in a few picky places.

```python
conf_mat = confusion_matrix(
    test_output, llm_df.category, labels=llm_df.category.unique()
)
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    xticklabels=llm_df.category.unique(),
    yticklabels=llm_df.category.unique(),
)
plt.ylabel("Actual")
plt.xlabel("Predicted")
```

![confusion matrix](_static/matrix-llm.png)

The diagonal line of cells running from the upper left to the lower right shows where the model correctly predicted the category. The off-diagonal cells show where it got things wrong. The color of the cells indicates how often the model made that prediction. For instance, we can see that one miscategorized hotel in the sample was predicted to be a restaurant and the second was predicted to be "Other."

Due to the inherent randomness in the LLM's predictions, it's a good idea to test your sample and run these reports multiple times to get a sense of the model's performance.

Before we look at how you might improve the LLM's performance, let's take a moment to compare the results of this evaluation against the old school approach where the supervised sample is used to train a machine-learning model that doesn't have access to the ocean of knowledge poured into an LLM.

This will require importing a mess of `sklearn` functions and classes. We'll use `TfidfVectorizer` to convert the payee text into a numerical representation that can be used by a `LinearSVC` classifier. We'll then use a `Pipeline` to chain the two together. If you have no idea what any of that means, don't worry. Now that we have LLMs in this world, you might never need to know.

{emphasize-lines="12-15"}

```python
import json
import time
from rich import print
from rich.progress import track
from huggingface_hub import InferenceClient
import pandas as pd
from itertools import batched
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
```

Here's a simple example of how you might train and evaluate a traditional machine-learning model using the supervised sample.

First you setup all the machinery.

```python
vectorizer = TfidfVectorizer(
    sublinear_tf=True,
    min_df=5,
    norm="l2",
    encoding="latin-1",
    ngram_range=(1, 3),
)
preprocessor = ColumnTransformer(
    transformers=[("payee", vectorizer, "payee")], sparse_threshold=0, remainder="drop"
)
pipeline = Pipeline(
    [("preprocessor", preprocessor), ("classifier", LinearSVC(dual="auto"))]
)
```

Then you train the model using those training sets we split out at the start.

```python
model = pipeline.fit(training_input, training_output)
```

And you ask the model to use its training to predict the right answers for the test set.

```python
predictions = model.predict(test_input)
```

Now, you can run the same evaluation code as before to see how the traditional model performed.

```python
print(classification_report(test_output, predictions))
```

```
              precision    recall  f1-score   support

         Bar       0.00      0.00      0.00         2
       Hotel       1.00      0.33      0.50         9
       Other       0.76      1.00      0.86        57
  Restaurant       1.00      0.33      0.50        15

    accuracy                           0.78        83
   macro avg       0.69      0.42      0.47        83
weighted avg       0.81      0.78      0.74        83
```

```python
labels = sorted(set(test_output) | set(predictions))

conf_mat = confusion_matrix(test_output, predictions, labels=labels)
fig, ax = plt.subplots(figsize=(5, 5))
sns.heatmap(
    conf_mat,
    annot=True,
    fmt="d",
    xticklabels=labels,
    yticklabels=labels,
)
plt.ylabel("Actual")
plt.xlabel("Predicted")
```

![confusion matrix](_static/matrix-ml.png)

Not great. The traditional model is guessing correctly about 75% of the time, but it's missing most cases of our "Bar", "Hotel" and "Restaurant" categories as almost everything is getting filed as "Other." The LLM, on the other hand, is guessing correctly more than 90% of the time and flagging many of the rare categories that we're seeking to find in the haystack of data.
