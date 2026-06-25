# AE — AI Engineer Main Spine Knowledge Vault

Repo này là vault học tập cho hướng **AI Engineer / GenAI Systems / NVIDIA-ready inference path**. Nội dung được tổ chức theo main spine từ nền tảng tới hệ thống production-style.

## Goal

Xây nền vững để hiểu, triển khai, đo đạc và debug các hệ thống AI hiện đại:

```text
Understand → Practice → Benchmark → Debug → Explain
```

## Main Spine

| Stage | Folder | Trọng tâm |
|---|---|---|
| 00 | `Knowledge/Main Spine/00 Base nền tảng/` | Python engineering, Linux/Git/Docker, API/backend, ML/DL, Transformers/LLM |
| 01 | `Knowledge/Main Spine/01 RAG Foundation/` | Ingestion, chunking, embedding, vector DB, retrieval, reranking, citation, evaluation |
| 02 | `Knowledge/Main Spine/02 Agentic AI Systems/` | Tool calling, planner/executor, state, verifier, guardrails, trace logs |
| 03 | `Knowledge/Main Spine/03 LLM Serving/` | vLLM, OpenAI-compatible API, streaming, KV cache, batching, TTFT/TPOT/tokens/sec |
| 04 | `Knowledge/Main Spine/04 GPU Serving/` | GPU memory, quantization, Triton, TensorRT-LLM, SGLang, Dynamo, cost/token |
| 05 | `Knowledge/Main Spine/05 AI Factory - GenAI Platform/` | Model service, retrieval service, agent service, eval service, monitoring, deployment |

## Current Status

| Stage | Status | Next proof |
|---|---|---|
| 00 Base nền tảng | studying | Giải thích lại được từng base và làm mini demo nhỏ |
| 01 RAG Foundation | drafted | Build retriever, đo Recall@k/MRR, log retrieved chunks |
| 02 Agentic AI Systems | drafted | Tool-calling demo có state, verifier và trace log |
| 03 LLM Serving | drafted | Serve model qua OpenAI-compatible API và benchmark TTFT/TPOT |
| 04 GPU Serving | drafted | Benchmark memory/latency trước-sau một cấu hình tối ưu |
| 05 AI Factory | drafted | Vẽ service boundary và runbook debug request end-to-end |

## Navigation

Đọc nhanh qua `Knowledge/00_INDEX.md`, sau đó đi theo thứ tự stage. Mỗi note chính có block đầu file gồm:

- `Status`: vị trí của note trong roadmap.
- `Must understand`: khái niệm bắt buộc nắm.
- `Must practice`: bài thực hành hoặc benchmark cần làm.
- `Can explain when ready`: câu hỏi tự kiểm tra trước khi coi là đạt.

Obsidian graph bắt đầu từ `AI Engineer.md`: `AI Engineer -> Main Spine -> stage -> note con`.

## Practice

Bộ đề bài thực tế nằm ở `Practice/Main Spine/README.md`. Nguồn dữ liệu synthetic/public/mock cho các đề nằm ở `Practice/Main Spine/Datasets.md`. Các đề chưa phải phase triển khai; chúng dùng để chọn bài toán, xác định user, dữ liệu, system behavior, metric và stage liên quan.

## Scope

Repo này hiện tách rõ hai lớp: `Knowledge/Main Spine/` để học concept và `Practice/Main Spine/` để giữ đề bài doanh nghiệp bám theo main spine. Tech radar chưa tách riêng; cập nhật công nghệ vẫn nằm trong từng note/stage khi cần.
