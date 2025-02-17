![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg) ![Poetry 1.8.3](https://img.shields.io/badge/poetry-1.8.3-blue.svg?logo=poetry) ![Pymilvus 2.5.4](https://img.shields.io/badge/PyMilvus-2.5.4-green) ![Unsloth 2025.2.12](https://img.shields.io/badge/Unsloth-2025.2.12-yellow)

# pymilvus-deep-research

This repository provides a codebase for prototyping an application that can download and read Wikipedia pages, perform retrieval-augmented generation (RAG) queries, and compare the performance of various open-source reasoning models. These models are quantized to 4 bits for efficient local execution. The codebase is built based on the information and techniques provided in this blog post [[1]](https://milvus.io/blog/i-built-a-deep-research-with-open-source-so-can-you.md) by [milvus.io](https://milvus.io/).

# Install dependencies

To install the dependencies, the user may use `poetry==1.8.3` run the following commands to run the installation:

```bash
$ pip install --upgrade pip setuptools wheel poetry==1.8.3
$ poetry config virtualenvs.create false && poetry install --no-root --verbose
```


# Assumptions
The following assumptions are made to simplify the scope of the project:

- Only 4-bit quantized models are used to ensure the pipeline runs locally

- No additional fine-tuning on the reasoning models are performed

- The agent only has access to only Wikipedia (not the entire knowledge base on the web) and perform RAG queries

- Only text data is processed 

- The agent will not backtrack or consider pivots



# Supported Models 

| **Models**[[2]](https://unsloth.ai/blog/dynamic-4bit)                                                                         | **BERTScore** | **ROGUE** |
|-------------------------------------------------------------------------------------------------------------------------------|---------------|-----------|
| [DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit) |               |           |
| [DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit) |               |           |
| [DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit)   |               |           |
| [QwQ-32B-Preview-unsloth-bnb-4bit](https://huggingface.co/unsloth/QwQ-32B-Preview-unsloth-bnb-4bit)                           |               |           |
| [Qwen2.5-3B-Instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit)                   |               |           |

# References
[1] [I Built a Deep Research with Open Source—and So Can You!](https://milvus.io/blog/i-built-a-deep-research-with-open-source-so-can-you.md)

[2] [Unsloth - Dynamic 4-bit Quantization](https://unsloth.ai/blog/dynamic-4bit)

[3] [LangChain](https://python.langchain.com/docs/introduction/)

[4] [Qwen](https://huggingface.co/Qwen)

[5] [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)

