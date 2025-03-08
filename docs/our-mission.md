# What we'll do

Journalists frequently encounter the mountains of messy data generated by our periphrastic society. This vast and verbose corpus boasts everything from long-hand entries in police reports to the legalese of legislative bills.

Understanding and analyzing this data is critical to the job but can be time-consuming and inefficient. Computers can help by automating sorting through large blocks of text, extracting key details, and flagging unusual patterns.

A common goal in this work is to classify text into categories. For example, you might want to sort a collection of emails as “spam” and “not spam” or identify corporate filings that suggest a company is about to go bankrupt.

Traditional techniques for classifying text, like keyword searches or regular expressions, can be brittle and error-prone. Machine learning models can be more flexible, but they require large amounts of labeled training data and a significant amount of computer programming expertise and too often yield unimpressive results.

This class teaches you how to use a large-language model for a better payoff. We will demonstrate how, with a few well-crafted prompts, you can get structured, meaningful responses more efficiently and accurately than other methods.

## Our example case

To demonstrate the power of this approach, we’ll focus on a specific data set: campaign expenditures.

Political campaigns must disclose the money they spend on everything from pizza to private jets. Tracking that spending can reveal patterns that go unnoticed and lead to important stories.

But it’s no easy task. Each election cycle, hundreds of candidates log thousands of transactions into the public databases where spending is disclosed. That’s so much data that no one can examine it all. To make matters worse, campaigns often use vague or misleading descriptions of their spending, making it difficult to parse and understand.

It wasn’t until after his 2022 election to Congress that [journalists discovered](https://www.nytimes.com/2022/12/29/nyregion/george-santos-campaign-finance.html) that Rep. George Santos of New York had spent thousands of campaign dollars on questionable and potentially illegal expenses. While many of his expenses were publicly disclosed, they were buried in other transactions and largely overlooked before election day.

[![Santos story](_static/santos.png)](https://www.nytimes.com/2022/12/29/nyregion/george-santos-campaign-finance.html)

Inspired by this story, we will create a classifier that can scan the expenditures logged in campaign finance reports and identify those that may be newsworthy.

[![CCDC](_static/ccdc.png)](https://californiacivicdata.org/)

We will draw our data from California, where the California Civic Data Coalition developed a clean, structured version of the state's campaign finance database containing hundreds of thousands of campaign expenses.