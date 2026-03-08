#!/bin/bash
set -e

# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync

# Register the Jupyter kernel
uv run python -m ipykernel install --user --name=first-llm-classifier --display-name='First LLM Classifier'
