# Installing Visual Studio Code

This class will show you how to interact with a large-language model using the Python computer programming language.

You can write Python code in your terminal, in a text file and any number of other places. If you're a skilled programmer who already has a preferred venue for coding, feel free to use it as you work through this class.

If you're not, the tool we recommend for beginners is [Visual Studio Code](https://code.visualstudio.com/), a free code editor made by Microsoft. It has built-in support for running Python notebooks — the same interactive, cell-by-cell coding environment used by [scientists](http://nbviewer.jupyter.org/github/robertodealmeida/notebooks/blob/master/earth_day_data_challenge/Analyzing%20whale%20tracks.ipynb), [scholars](http://nbviewer.jupyter.org/github/nealcaren/workshop_2014/blob/master/notebooks/5_Times_API.ipynb), [investors](https://github.com/rsvp/fecon235/blob/master/nb/fred-debt-pop.ipynb) and corporations to create and share their research. It is also used by journalists to develop stories and show their work.

VS Code can be installed on any operating system with a simple point-and-click interface.

## Install Python

Before setting up VS Code, you'll need to install the Python programming language on your computer. We'll use a tool called [uv](https://docs.astral.sh/uv/) that makes it easy to install and manage Python.

### MacOS

Open the Terminal application by searching for "Terminal" in your operating system's application finder.

Install uv by pasting in the following command and hitting enter:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close your terminal and open a new one for the changes to take effect.

Now install Python:

```bash
uv python install 3.13
```

Verify it worked by running:

```bash
uv python list
```

You should see Python 3.13 in the list. If so, you're all set.

### Windows

Open PowerShell by searching for "PowerShell" in your operating system's application finder.

Install uv by pasting in the following command and hitting enter:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Close your PowerShell window and open a new one for the changes to take effect.

Now install Python:

```powershell
uv python install 3.13
```

Verify it worked by running:

```powershell
uv python list
```

You should see Python 3.13 in the list. If so, you're all set.

## Install Visual Studio Code

The next step is to visit the [Visual Studio Code homepage](https://code.visualstudio.com/) in your web browser.

![Visual Studio Code homepage](/_static/vs-code-homepage.png)

Click the download button for your operating system. Find the file in your downloads directory and double click it to begin the installation process. Follow the instructions presented by the pop-up windows, sticking to the default options.

```{warning}
Your computer's operating system might flag the Visual Studio Code installer as an unverified or insecure application. Don't worry. The tool is developed by Microsoft and it's safe to use.

If your system is blocking you from installing the tool, you'll likely need to work around its barriers. For instance, on MacOS, this might require [visiting your system's security settings](https://www.wikihow.com/Install-Software-from-Unsigned-Developers-on-a-Mac) to allow the installation.
```

## Install the Python extension

Once Visual Studio Code is installed, open it by searching for "Visual Studio Code" in your operating system's application finder. That will open up a new window that looks something like this:

![Visual Studio Code splash screen](/_static/vs-code-splash.png)

Now you need to install the Python extension, which gives Visual Studio Code the ability to run Python code and notebooks. Click the Extensions icon in the left sidebar — it looks like four small squares.

![Visual Studio Code extensions icon](/_static/vs-code-extensions-icon.png)

Type "Python" into the search bar. The top result should be the Python extension published by Microsoft. Click the blue "Install" button.

![Python extension](/_static/vs-code-python-extension.png)

That's all you need. The Python extension automatically includes support for Jupyter notebooks.

## Open your first notebook

Click "File" in the menu bar and select "New File..." from the dropdown. When prompted to choose a file type, select "Jupyter Notebook."

![Visual Studio Code new notebook](/_static/vs-code-new-notebook.png)

Visual Studio Code will open a fresh notebook. You may see a prompt in the upper right corner asking you to select a Python kernel. Click it and choose the Python installation on your system.

![Visual Studio Code select kernel](/_static/vs-code-select-kernel.png)

```{warning}
If Visual Studio Code says it needs to install `ipykernel`, click "Install" when prompted. This is a small, free package that allows Visual Studio Code to run notebook cells. It only needs to happen once.
```

Welcome to your first notebook in Visual Studio Code. Let's make sure everything is working. Click on the first cell, type the following and hit the play button to the left of the cell, or press Shift+Enter:

```python
2+2
```

You should see the number `4` appear below the cell.

![Visual Studio Code first cell](/_static/vs-code-first-cell.png)

If so, congratulations. You're all set up and ready to move on to writing code.

:::{admonition} Note
If you're struggling to make Visual Studio Code work and need help with the basics, we recommend you check out ["First Python Notebook"](https://palewi.re/docs/first-python-notebook/), where you can get up to speed.
:::
