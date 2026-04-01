# Baseline Framework Selection: GPTCache

## Overview

We selected **GPTCache** (v0.1.44, MIT License) by Zilliz as the baseline open-source LLM caching
library for this project. GPTCache is a semantic cache for LLM applications that intercepts API
calls, converts queries to embeddings, and serves cached responses for semantically similar queries.

## Main Features

| Feature | Detail |
|---------|--------|
| **Caching Model** | Semantic (embedding-based approximate matching) + exact match |
| **Embedding Support** | Pluggable: OpenAI, ONNX, Hugging Face Sentence-Transformers, etc. |
| **Scalar Storage** | SQLite (default), MySQL, PostgreSQL, Oracle |
| **Vector Storage** | FAISS (default), Milvus, Chromadb, Hnswlib, PGVector |
| **Similarity Evaluation** | Cosine distance, ONNX cross-encoder, configurable threshold |
| **Framework Integration** | LangChain, LlamaIndex, OpenAI API, Replicate, Cohere |
| **Language** | Python (100%) |
| **GitHub Stars** | ~8,000 |
| **License** | MIT |

## Default Eviction Policy

GPTCache ships with **LRU** (Least Recently Used) as the default eviction policy, implemented via
Python's `cachetools.LRUCache`. It also supports FIFO, LFU, and Random Replacement (RR), all
delegating to `cachetools` data structures. Eviction decisions are based **solely on entry count**
— the maintainers acknowledge this "can result in inaccurate resource evaluation and may cause
OOM errors" and list more sophisticated policies as a development goal.

## Why GPTCache

1. **Clean eviction abstraction.** The `EvictionBase` abstract class requires only three methods
   (`put`, `get`, `policy`), isolated in `gptcache/manager/eviction/`. Adding a new policy means
   creating one Python file and registering it in the factory — no changes across the codebase.

2. **Explicit gap in eviction sophistication.** The current LRU/FIFO policies ignore access
   frequency, entry cost, and entry size. The project roadmap explicitly calls for time-based and
   cost-aware eviction — a direct opportunity for contribution.

3. **Built-in semantic matching pipeline.** Embedding generation, vector search, and similarity
   evaluation are already implemented, allowing us to focus entirely on the eviction policy layer.

4. **Reproducibility.** Comprehensive test suite (`pytest`), documentation, and stable codebase
   (last commit August 2024 — mature, not a moving target).

5. **Contribution path.** MIT license with modular architecture makes upstream PR submission
   straightforward. The professor awards maximum grade for accepted open-source contributions.

## Alternatives Considered

| Library | Why Not |
|---------|---------|
| **LangChain Cache** | Eviction delegated to backend (Redis, etc.), not a first-class concept |
| **vLLM** | KV cache at inference level; enormous C++/CUDA codebase, no pluggable eviction |
| **SGLang** | RadixAttention in Rust; wrong abstraction level for policy research |
| **NVIDIA kvpress** | KV cache compression, not application-level response caching |
| **LMCache** | Tied to inference engines (vLLM/SGLang); requires GPU programming |
| **ModelCache** | Smaller community, partial Chinese docs, less clean eviction abstraction |
