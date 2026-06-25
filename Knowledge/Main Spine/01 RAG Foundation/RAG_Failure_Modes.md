# RAG Failure Modes — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Observability Logging, RAG Evaluation, Security Permissions Privacy
- Vai trò: phân loại lỗi RAG theo tầng để debug đúng chỗ.

## Must understand

- RAG sai không chỉ vì LLM yếu.
- Cùng một answer sai có thể bắt nguồn từ ingestion, chunking, embedding, filtering, retrieval, rerank, context, prompt, generation hoặc citation.
- Debug RAG phải bắt đầu bằng câu hỏi: evidence đúng có vào candidate set không?
- Failure modes cần được log và gắn nhãn để cải thiện theo nhóm.

## Must practice

- Với mỗi answer sai, kiểm tra top-k có evidence đúng không.
- Nếu top-k không có evidence, debug ingestion/chunking/embedding/filter/retrieval.
- Nếu top-k có evidence nhưng final context không có, debug rerank/context construction.
- Nếu final context có evidence nhưng answer sai, debug prompt/generation/citation.
- Gắn failure label cho từng case.

## Can explain when ready

- Làm sao phân biệt retrieval failure và generation failure?
- Citation mismatch nằm ở tầng nào?
- Freshness failure khác retrieval failure thế nào?
- Vì sao eval set quá dễ cũng là failure mode?

---

## 1. Bảng failure modes

| Tầng | Lỗi thường gặp |
| --- | --- |
| Ingestion | parse PDF lỗi, mất table, OCR sai |
| Chunking | chunk mất context, cắt ngang ý, overlap sai |
| Metadata | thiếu source/page/version/permission |
| Embedding | model không hợp domain/ngôn ngữ |
| Indexing | vector cũ, metric sai, duplicate chunks |
| Filtering | metadata filter loại mất tài liệu đúng |
| Retrieval | top-k không có evidence đúng |
| Hybrid search | fusion sai, sparse/dense noise |
| Reranking | reranker xếp sai hoặc quá chậm |
| Context construction | nhồi quá nhiều chunk nhiễu, citation map sai |
| Prompt grounding | không ép abstention/citation đủ chặt |
| Generation | LLM hallucinate dù có context |
| Citation | citation không support claim |
| Permission | user thấy tài liệu không được phép |
| Freshness | trả lời theo tài liệu cũ |
| Evaluation | eval set quá dễ, bị leak, không có hard negatives |

---

## 2. Debug pattern

```text
Answer sai
-> top-k có evidence đúng không?
   -> không:
      debug ingestion/chunking/embedding/filter/retrieval
   -> có:
      final context có evidence đúng không?
         -> không:
            debug rerank/context construction
         -> có:
            answer có dùng đúng evidence không?
               -> không:
                  debug prompt/generation
               -> có nhưng citation sai:
                  debug citation mapping
```

Nếu chỉ sai ngoài production:

```text
distribution shift
stale data
permission/filter khác môi trường
index version khác
prompt/model version khác
```

---

## 3. Triệu chứng và nguyên nhân

| Triệu chứng | Nguyên nhân thường gặp |
| --- | --- |
| Empty retrieval | query rewrite sai, filter quá chặt, index thiếu data |
| Top-k toàn noise | embedding kém, chunking sai, top-k thấp, hybrid thiếu sparse |
| Chunk đúng rank thấp | cần reranker, hybrid, query rewrite |
| Answer đúng nhưng cite sai | citation mapping hoặc prompt cite yếu |
| Answer theo policy cũ | freshness/version filter sai |
| User thấy private docs | permission filter sai hoặc post-filter quá muộn |
| Latency cao | reranker/LLM/context quá dài |
| Cost cao | top-k/context quá lớn, cache thiếu |

---

## 4. Failure analysis template

```text
Query:
Expected answer:
Actual answer:
Expected evidence:
Retrieved top-k:
Final context:
Citations:
Failure type:
Root cause:
Fix:
Regression test added:
```

---

## 5. Kết luận

RAG failure analysis là kỹ năng production quan trọng. Khi phân tầng lỗi tốt, bạn sẽ biết nên sửa parser, chunker, embedding, retriever, reranker, context construction, prompt hay data lifecycle thay vì đổi LLM một cách mù.
