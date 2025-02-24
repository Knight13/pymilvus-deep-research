![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg) ![Poetry 1.8.3](https://img.shields.io/badge/poetry-1.8.3-blue.svg?logo=poetry) ![Pymilvus 2.5.4](https://img.shields.io/badge/PyMilvus-2.5.4-green) ![Unsloth 2025.2.12](https://img.shields.io/badge/Unsloth-2025.2.12-yellow)

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

All supported models, listed in the table below are evaluated on the BERTScore [[7]](https://arxiv.org/abs/1904.09675) metric on the HotpotQA [[6]](https://hotpotqa.github.io/) test set.

| **Models** [[2]](https://unsloth.ai/blog/dynamic-4bit)                                                                        | **BERTScore** [[7]](https://arxiv.org/abs/1904.09675)|
|-------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------|
| [DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit) |          ToDo                                        |
| [DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit](https://huggingface.co/unsloth/DeepSeek-R1-Distill-Qwen-7B-unsloth-bnb-4bit)   |          ToDo                                        |
| [Qwen2.5-3B-Instruct-unsloth-bnb-4bit](https://huggingface.co/unsloth/Qwen2.5-3B-Instruct-unsloth-bnb-4bit)                   |          ToDo                                        |


# Usage

To run inference using the 4-bit models, run the following command:

```bash
$ python main.py
```
The choice of the model, query, page title and topic can be set in `./configs/main.yaml` file.


# References
[1] [I Built a Deep Research with Open Source—and So Can You!](https://milvus.io/blog/i-built-a-deep-research-with-open-source-so-can-you.md)

[2] [Unsloth - Dynamic 4-bit Quantization](https://unsloth.ai/blog/dynamic-4bit)

[3] [LangChain](https://python.langchain.com/docs/introduction/)

[4] [Qwen](https://huggingface.co/Qwen)

[5] [DeepSeek R1](https://huggingface.co/deepseek-ai/DeepSeek-R1)

[6] [HotpotQA - A Dataset for Diverse, Explainable Multi-hop Question Answering](https://hotpotqa.github.io/)

[7] [BERTScore: Evaluating Text Generation with BERT](https://arxiv.org/abs/1904.09675)

# ToDo:

* Add evaluation script
* Evaluate DeepSeek-R1 and Qwen models on HotpotQA test set