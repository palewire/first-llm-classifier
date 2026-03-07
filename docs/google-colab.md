# Google Colab

If you want the fastest possible way to start writing code, [Google Colab](https://colab.research.google.com/) is the easiest option. It is a free, browser-based coding environment that requires no installation. You can be running your first notebook in under two minutes.

All you need is a Google account.

[![Google Colab homepage showing the tagline "Colab is a hosted Jupyter Notebook service" and example notebooks](_static/colab-homepage.png)](https://colab.research.google.com/)

## Create a new notebook

Go to [colab.research.google.com](https://colab.research.google.com/) and sign in with your Google account.

Click "New notebook" at the bottom of the welcome dialog.

![Google Colab welcome dialog with the "New notebook" option highlighted](_static/colab-new-notebook.png)

A fresh Jupyter notebook will open in your browser. You're already in a fully configured Python environment — there's nothing else to install.

## Verify Python is working

Click on the first cell, type the following and press Shift+Enter to run it:

```python
2+2
```

You should see `4` appear below the cell.

![A Jupyter notebook in Google Colab with a cell containing 2+2 and its output of 4](_static/colab-first-cell.png)

If so, you're all set up and ready to move on.

## Store your Hugging Face token

Google Colab has a built-in secrets manager that lets you store your Hugging Face API token securely so you don't have to paste it directly into your notebook.

Click the key icon in the left sidebar to open the Secrets panel. Click "Add new secret," name it `HF_TOKEN`, paste in your Hugging Face token, and enable "Notebook access."

![Google Colab Secrets panel showing a new secret named HF_TOKEN](_static/colab-secret.png)

When you're writing code later in this class, you can retrieve the token with:

```python
from google.colab import userdata

token = userdata.get("HF_TOKEN")
```

This keeps your key out of the notebook itself, which is safer if you ever share the file.

```{note}
If you are using Visual Studio Code instead of Google Colab, skip this step. The next chapter covers how to set your token there.
```
