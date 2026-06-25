# Performance / Latency / Cost — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Rerank, Context Construction, Caching, Observability Logging
- Vai trò: hiểu RAG chậm/tốn ở đâu và tối ưu mà không phá quality.

## Must understand

- RAG quality cao nhưng latency quá lâu vẫn fail production.
- Latency gồm nhiều stage: rewrite, embedding, retrieval, rerank, context construction, generation.
- Context dài làm tăng TTFT, cost và nguy cơ noise.
- Reranker thường là bottleneck nếu rerank quá nhiều candidates.
- Tối ưu performance phải đo p50/p95/p99, không chỉ average.

## Must practice

- Log latency từng stage.
- Benchmark dense-only vs hybrid vs hybrid+rerank.
- Đo token usage khi thay `final_context_k`.
- Test cache hit rate cho query lặp.
- Tạo timeout/fallback nếu reranker hoặc LLM quá chậm.

## Can explain when ready

- Stage nào thường tốn latency nhất trong RAG?
- Tăng top-k ảnh hưởng quality và latency thế nào?
- Context dài ảnh hưởng cost và generation latency ra sao?
- Khi nào nên bỏ reranker trong fast path?

---

## 1. Latency breakdown

```text
query rewrite: 20-200 ms hoặc LLM call nếu phức tạp
query embedding: vài ms đến vài trăm ms
vector/BM25 search: vài ms đến vài trăm ms
reranking: 50 ms đến vài giây tùy model/top-n
context construction: thường nhỏ nhưng compression có thể tốn
LLM generation: thường lớn nhất nếu answer dài/context dài
```

Cần log:

```text
p50
p95
p99
timeout rate
cost/query
tokens/query
```

---

## 2. Quality-latency trade-off

| Tăng | Quality | Latency/cost |
| --- | --- | --- |
| candidate_top_k | recall tăng | retrieval/rerank tăng |
| rerank_top_n | precision tăng | rerank tăng |
| final_context_k | có thể đủ evidence hơn | generation cost/latency tăng |
| multi-query | recall tăng | LLM/retrieval calls tăng |
| LLM reranker | ranking tốt hơn | cost/latency cao |

Không có config tối ưu chung. Phải benchmark theo eval set và workload thật.

---

## 3. Context cost

Context dài làm:

```text
input token cost tăng
TTFT tăng
generation chậm hơn
lost-in-the-middle dễ hơn
LLM bị noise
```

Rule:

```text
retrieve rộng
rerank/nén
final context gọn
```

---

## 4. Timeout và fallback

Fallback patterns:

```text
reranker timeout -> dùng fusion ranking
LLM timeout -> trả lỗi thân thiện hoặc retry model nhỏ hơn
query rewrite timeout -> dùng query gốc
hybrid sparse timeout -> dense-only fallback
```

Fallback phải được log để biết quality có giảm không.

---

## 5. Fast path / slow path

Fast path:

```text
simple query
metadata filter
hybrid retrieval
cross-encoder small reranker
short answer
```

Slow path:

```text
complex query
multi-query/decomposition
larger reranker hoặc LLM reranker
more context checks
```

Không phải query nào cũng cần pipeline đắt nhất.

---

## 6. Kết luận

Performance là một phần của RAG Foundation vì mọi quyết định retrieval/rerank/context đều có cost. Production RAG phải biết trade-off giữa recall, precision, latency, token cost và reliability.
