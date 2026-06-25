# Knowledge Index

## Learning Loop

```text
Understand → Practice → Benchmark → Debug → Explain
```

Mục tiêu của vault là học theo main spine, tránh nhảy sớm sang công nghệ rời rạc khi nền chưa đủ chắc.

## 00 Base nền tảng

| Note | Core concepts | Proof of learning |
|---|---|---|
| `Knowledge/Main Spine/00 Base nền tảng/Python engineering.md` | project structure, package/import, function/class, typing, config, logging, CLI, testing | Refactor notebook thành project chạy bằng terminal |
| `Knowledge/Main Spine/00 Base nền tảng/Linux - Git - Docker.md` | terminal, SSH, env, Git branch, Dockerfile, Compose | Chạy app AI/RAG bằng Docker Compose |
| `Knowledge/Main Spine/00 Base nền tảng/API - Backend.md` | HTTP, REST, FastAPI, Pydantic, streaming, logging, error handling | Expose `/health`, `/retrieve`, `/ask`, `/chat/stream` |
| `Knowledge/Main Spine/00 Base nền tảng/ML - DL basics.md` | training loop, loss, optimization, metrics, leakage, evaluation | Giải thích metric và failure mode theo bài toán |
| `Knowledge/Main Spine/00 Base nền tảng/Transformers - LLM basics.md` | tokenization, attention, QKV, causal mask, KV cache, decoding | Giải thích text đi từ token tới output như thế nào |

## 01 RAG Foundation

`Knowledge/Main Spine/01 RAG Foundation/RAG Foundation.md`

Core concepts:
- data ingestion, parsing, metadata
- chunking, embedding, vector DB/indexing
- `Knowledge/Main Spine/01 RAG Foundation/Chunking.md`
- `Knowledge/Main Spine/01 RAG Foundation/Embedding.md`
- retrieval, hybrid search, reranking
- context construction, citation, grounded generation
- RAG evaluation, observability, failure modes

Proof of learning:
- build simple retriever
- compare chunk size/top_k
- add reranker
- measure Recall@k/MRR
- log retrieved chunks, latency and failure cases

## 02 Agentic AI Systems

`Knowledge/Main Spine/02 Agentic AI Systems/Agentic AI Systems.md`

Core concepts:
- agent loop
- tool calling and tool schema
- planner/executor
- state and memory policy
- verifier, guardrails, human-in-the-loop
- trace logs and task success metrics

Proof of learning:
- build tool-calling demo
- add typed state
- log every tool call
- verify final answer against evidence/tool output

## 03 LLM Serving

`Knowledge/Main Spine/03 LLM Serving/LLM Serving.md`

Core concepts:
- prefill and decode
- KV cache and context length
- OpenAI-compatible API
- streaming
- vLLM, continuous batching, PagedAttention
- TTFT, TPOT, tokens/sec, concurrency, cost

Proof of learning:
- serve model through an OpenAI-compatible endpoint
- run streaming test
- benchmark TTFT/TPOT/tokens/sec
- explain slow request or CUDA OOM from metrics

## 04 GPU Serving

`Knowledge/Main Spine/04 GPU Serving/GPU Inference - Serving Optimization.md`

Core concepts:
- GPU memory: weights, KV cache, activations, buffers
- precision and quantization
- batching, caching, parallelism
- Triton, TensorRT-LLM, SGLang, Dynamo
- latency/throughput/VRAM/cost trade-off

Proof of learning:
- compare baseline vs optimized config
- record VRAM, TTFT, TPOT, tokens/sec
- explain bottleneck: compute, memory, queue, scheduler or runtime

## 05 AI Factory - GenAI Platform

`Knowledge/Main Spine/05 AI Factory - GenAI Platform/AI Factory - GenAI Platform.md`

Core concepts:
- model service, retrieval service, agent service
- tool registry, evaluation service, monitoring
- service boundary, config/versioning, rollback
- request tracing, metrics, logs, runbook

Proof of learning:
- design a mini AI platform
- trace request across services
- define eval regression and failure runbook

## Practice Layer

`Practice/Main Spine/README.md` giữ các đề bài doanh nghiệp để biến knowledge thành bài tập có user, dữ liệu, system behavior và metric rõ ràng.

Recommended path:

```text
Knowledge Copilot
→ IT Service Desk Agent
→ LLM Gateway
→ GPU Inference Optimizer
→ Enterprise AI Factory
```
