# Observability / Logging — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: RAG Evaluation, RAG Failure Modes, Performance Latency Cost
- Vai trò: log đủ trace để debug RAG theo từng tầng, đo latency/cost và chạy feedback loop.

## Must understand

- Không log retrieved chunks thì không debug được RAG.
- Không log prompt/model/index version thì không biết thay đổi nào làm hệ thống tốt/xấu.
- Observability không chỉ là error log; nó là request trace xuyên pipeline.
- Log phải cân bằng giữa debug value, privacy và cost.
- Production RAG cần trace theo stage: query rewrite, retrieval, rerank, context, generation, citation.

## Must practice

- Log một RAG request end-to-end với `trace_id`.
- Ghi query gốc, rewritten query, filters, retrieved IDs, scores, reranker scores.
- Ghi final context IDs, prompt version, model version, token usage, latency từng stage.
- Ghi feedback và failure type.
- Mask hoặc không log raw PII nếu chính sách không cho phép.

## Can explain when ready

- Trace khác log dòng thường ở đâu?
- Vì sao cần prompt version và index version?
- Log gì để debug answer sai?
- Log gì để tối ưu latency/cost?

---

## 1. RAG trace nên có gì?

```json
{
  "trace_id": "rag_20260625_001",
  "user_id_hash": "u_abc",
  "query": "enterprise refund sau 30 ngày?",
  "rewritten_query": "Enterprise refund policy after 30 days",
  "filters": {
    "permission_group": "employee",
    "version": "latest"
  },
  "retrieved_chunks": [
    {"chunk_id": "c1", "score": 0.82, "source": "refund_policy.pdf"},
    {"chunk_id": "c2", "score": 0.79, "source": "enterprise_sla.pdf"}
  ],
  "reranked_chunks": [
    {"chunk_id": "c2", "score": 0.91},
    {"chunk_id": "c1", "score": 0.66}
  ],
  "final_context_ids": ["E1", "E2"],
  "prompt_version": "rag_answer_v3",
  "model": "llm-x",
  "answer": "...",
  "citations": ["E1", "E2"],
  "latency_ms": {
    "rewrite": 80,
    "retrieval": 45,
    "rerank": 210,
    "generation": 1200
  },
  "tokens": {"input": 4200, "output": 350},
  "cost_usd": 0.012,
  "failure_type": null
}
```

---

## 2. Log theo từng tầng

| Tầng | Cần log |
| --- | --- |
| Query processing | query gốc, rewritten query, language, query type |
| Filtering | user/group/role, applied filters, filter source |
| Retrieval | chunk IDs, scores, source, top-k, index version |
| Reranking | reranker model, reranker score, rerank top-n |
| Context construction | final evidence IDs, compression, ordering, token count |
| Prompt | prompt template/version, system/developer instruction version |
| Generation | model, params, answer, citations, finish reason |
| Evaluation/feedback | user feedback, judge score, failure label |
| Performance | latency per stage, token usage, cost |

---

## 3. Versioning trong log

Cần log:

```text
parser version
chunker version
embedding model/version
index/collection version
retrieval config
reranker model/version
prompt version
LLM model/version
context construction version
```

Nếu không có versioning, khi answer hôm nay khác hôm qua, bạn không biết do model, prompt, index hay data update.

---

## 4. Failure labels

Nên gán failure type:

```text
parse_error
empty_retrieval
wrong_retrieval
permission_filter_error
rerank_wrong
context_noise
citation_mismatch
unsupported_claim
stale_document
timeout
model_error
```

Failure label giúp gom nhóm lỗi để cải thiện đúng tầng.

---

## 5. Privacy trong logging

Không phải mọi thứ đều nên log raw.

Cần cân nhắc:

```text
PII trong user query
tài liệu private trong retrieved context
secret/API key trong document
tenant isolation
retention policy
access control cho log viewer
```

Pattern an toàn hơn:

```text
hash user_id
mask PII nếu cần
log chunk_id thay vì full text cho production
lưu raw context trong secure debug store có retention ngắn
```

---

## 6. Dashboard nên có

```text
query volume
latency p50/p95/p99
cost per query
retrieval empty rate
answer abstention rate
citation mismatch rate
top failure types
cache hit rate
permission violation count
model/index/prompt version distribution
```

---

## 7. Kết luận

Observability biến RAG từ demo thành hệ thống debug được. Khi answer sai, trace phải cho biết lỗi nằm ở retrieval, filter, rerank, context, prompt, generation, citation hay data freshness.
