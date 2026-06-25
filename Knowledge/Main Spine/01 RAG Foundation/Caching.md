# Caching — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Performance Latency Cost, Data Freshness Update Pipeline, Observability Logging
- Vai trò: giảm latency/cost nhưng vẫn kiểm soát stale cache khi data thay đổi.

## Must understand

- Cache giúp nhanh và rẻ hơn, nhưng có thể trả lời bằng dữ liệu cũ nếu invalidation sai.
- RAG có nhiều tầng cache: parsing, embedding, retrieval, reranker, final answer.
- Cache key phải chứa version/config liên quan, không chỉ query text.
- Permission và tenant phải là một phần của cache key.
- Cache hit rate là metric production quan trọng.

## Must practice

- Cache query embedding theo normalized query + embedding model version.
- Cache retrieval theo query + filters + index version + retriever config.
- Cache reranker score theo query + chunk_id + reranker version.
- Invalidate cache khi document/index/prompt/model version đổi.
- Log cache hit/miss và stale cache incidents.

## Can explain when ready

- Vì sao answer cache nguy hiểm hơn embedding cache?
- Cache key cho retrieval cần chứa gì?
- Stale cache xảy ra khi nào?
- Permission-aware cache cần chú ý gì?

---

## 1. Các loại cache

| Cache | Ý nghĩa | Rủi ro |
| --- | --- | --- |
| Document parsing cache | không parse lại file chưa đổi | parser version đổi nhưng cache chưa invalid |
| Embedding cache | query/doc giống nhau không embed lại | embedding model đổi |
| Retrieval cache | query giống nhau dùng lại top-k | index/filter/data đổi |
| Reranker cache | cache score query-chunk pair | reranker model đổi |
| Context cache | reuse final context | stale evidence/version |
| LLM response cache | reuse answer | permission/freshness/prompt đổi |

---

## 2. Cache key

Retrieval cache key không nên chỉ là query.

Cần gồm:

```text
normalized_query
tenant_id
permission_scope
metadata_filters
index_version
embedding_model_version
retriever_config
top_k
```

Reranker cache key:

```text
query_hash
chunk_id
chunk_version/hash
reranker_model_version
```

Answer cache key:

```text
query_hash
user permission scope
final_context_hash
prompt_version
model_version
answer_format
```

---

## 3. Invalidation

Invalidate khi:

```text
document content đổi
permission đổi
document archived/deleted
index rebuilt
embedding model đổi
chunking strategy đổi
reranker model đổi
prompt version đổi
LLM model đổi nếu answer cache
```

Freshness-sensitive query nên có TTL ngắn hơn.

---

## 4. Permission-aware caching

Nguy hiểm:

```text
User A có quyền HR hỏi query.
Answer/cache lưu private context.
User B không có quyền hỏi query giống vậy.
Cache trả lại answer của User A.
```

Cache key phải chứa permission scope hoặc không cache final answer cho sensitive domains.

---

## 5. Metric

```text
cache hit rate
cache miss rate
latency saved
cost saved
stale cache incidents
permission cache violation
TTL expiration count
```

---

## 6. Kết luận

Caching là tối ưu quan trọng nhưng phải đi cùng versioning, permission scope và invalidation. Trong RAG, cache sai có thể không chỉ trả lời cũ mà còn leak dữ liệu.
