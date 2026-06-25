# Hybrid Search — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Embedding, VectorDB, Retriever
- Vai trò: mở rộng phần dense retrieval, sparse/BM25 retrieval, fusion, reranking và hybrid retrieval eval trong RAG Foundation.

## Must understand

- Hybrid search kết hợp nhiều retrieval signals, thường là dense vector search và sparse/BM25 search.
- Dense retrieval mạnh về semantic similarity nhưng yếu với mã, số, entity, function name, legal clause và exact keyword.
- Sparse/BM25 mạnh với exact match nhưng yếu với synonym, paraphrase, query tự nhiên và đa ngôn ngữ.
- Fusion không được cộng score ngây thơ vì dense score và BM25 score khác scale.
- RRF là baseline fusion tốt vì dùng rank thay vì tin vào score gốc.
- Hybrid search không thay thế metadata filter; permission/version/source filter vẫn nên nằm trước retrieval.
- Hybrid retrieval ưu tiên recall, reranker ưu tiên precision.

## Must practice

- So sánh dense-only, sparse-only và dense+sparse hybrid trên cùng eval set.
- Tạo hard negatives cho entity/code/legal/finance cases.
- Implement baseline: dense top-k, sparse top-k, RRF, reranker, final context.
- Đo Recall@10, Recall@50, MRR, nDCG, latency và answer groundedness.
- Test query tiếng Việt, tiếng Anh và mixed Vietnamese-English.
- Kiểm tra lỗi duplicate candidates, keyword noise, semantic-near-but-wrong và wrong version.
- Tune `dense_top_k`, `sparse_top_k`, `final_context_k` theo latency/cost target.

## Can explain when ready

- Vì sao dense-only dễ sai với code, số, mã sản phẩm hoặc điều khoản pháp lý?
- BM25 khác sparse vector ở điểm nào?
- RRF giải quyết vấn đề gì của score fusion?
- Khi nào nên dùng weighted fusion thay vì RRF?
- Vì sao hybrid search vẫn cần reranker?
- Khi nào hybrid là overkill?

---

## 1. Hybrid search là gì?

Hybrid search là kỹ thuật kết hợp nhiều kiểu retrieval để lấy candidate tốt hơn.

```text
hybrid search = dense retrieval + sparse/BM25 retrieval + fusion/rerank
```

Dense retrieval giỏi hiểu nghĩa:

```text
"refund policy"
gần "money back rules"
gần "reimbursement condition"
gần "chính sách hoàn tiền"
```

Sparse/BM25 giỏi exact keyword:

```text
A-134
BAD_FPD10_ASOF
INV-2026-001
RTX 3090
Article 7.2
calculate_score()
```

Qdrant mô tả hybrid query theo hướng kết hợp dense, sparse và multivector results rồi fuse bằng RRF hoặc DBSF. ([Qdrant Hybrid Queries][1])

---

## 2. Vì sao cần hybrid search?

Dense và sparse có điểm mù khác nhau.

Dense fail case:

```text
Query:
"BAD_FPD10_ASOF definition"

Dense có thể retrieve:
BAD_FPD30_ASOF definition
BAD_FPD7_ASOF definition
BAD score definition
```

Các field này semantic gần nhau, nhưng trong business rule hoặc risk model, `BAD_FPD10_ASOF` và `BAD_FPD30_ASOF` là hai field khác nhau.

Sparse fail case:

```text
Query:
"khách hàng có được lấy lại tiền không?"

Document:
"Refund requests are allowed within 30 days of purchase."
```

BM25 có thể miss vì không trùng keyword, khác ngôn ngữ hoặc khác cách diễn đạt. Dense multilingual embedding có thể bắt được nghĩa “lấy lại tiền” gần với “refund”.

Hybrid search sinh ra để lấy cả hai nguồn:

```text
Dense cứu semantic mismatch.
Sparse cứu exact keyword/entity/code/number.
Fusion gộp candidate.
Reranker chọn evidence tốt nhất.
```

---

## 3. Từ keyword search đến hybrid RAG

| Giai đoạn | Retrieval style | Vì sao có | Điểm yếu |
| --- | --- | --- | --- |
| Search cổ điển | TF-IDF / BM25 | Tìm keyword nhanh, explainable | Không hiểu synonym/paraphrase |
| Neural retrieval | Dense embedding | Tìm theo meaning | Yếu exact match, entity, số, code |
| Production RAG | Dense + sparse hybrid | Cần cả meaning và exact keyword | Cần fusion/rerank |
| Advanced hybrid | Dense + sparse + reranker + metadata | Cần recall và precision tốt | Pipeline phức tạp hơn |
| Multi-vector hybrid | Dense + sparse + image/title/code/table vectors | Tài liệu không chỉ plain text | Schema/index/eval khó hơn |

Cách nhớ:

```text
BM25 tìm chữ.
Dense embedding tìm nghĩa.
Hybrid search cần cả chữ lẫn nghĩa.
```

---

## 4. BM25 là gì?

BM25 là scoring function cổ điển cho keyword retrieval. Nó đánh giá document dựa trên:

```text
term frequency: từ query xuất hiện nhiều không
inverse document frequency: từ đó hiếm hay phổ biến
document length normalization: document dài/ngắn ảnh hưởng thế nào
```

Ví dụ:

```text
Query:
"refund policy enterprise"

Document A:
"Our refund policy for enterprise customers..."

Document B:
"Our payment method for customers..."
```

BM25 thích A vì có exact terms `refund`, `policy`, `enterprise`.

BM25 mạnh với:

```text
tên riêng
mã sản phẩm
số hợp đồng
error code
API name
function/class name
legal clause
field name
acronym
```

BM25 yếu khi query và document không dùng cùng từ:

```text
Query:
"tôi có được lấy lại tiền không?"

Document:
"refund requests are allowed..."
```

Elasticsearch dùng BM25 làm scoring model mặc định cho text relevance, dựa trên term frequency, inverse document frequency và field-length normalization. ([Elastic BM25][2])

---

## 5. Sparse vector là gì?

Sparse vector là vector thưa, hầu hết chiều bằng 0.

Dense vector:

```text
[0.12, -0.04, 0.88, ..., 0.05]
```

Sparse vector có thể nhìn gần giống:

```json
{
  "indices": [104, 9821, 18821],
  "values": [0.92, 0.78, 0.65]
}
```

Sparse signal có thể đến từ:

```text
BM25 / lexical search
SPLADE / learned sparse model
sparse embedding model
full-text search engine
```

BM25 và sparse vector có liên quan nhưng không hoàn toàn giống nhau:

```text
BM25 = keyword scoring cổ điển trong search engine.
Sparse vector = retrieval signal dạng vector thưa, có thể BM25-like hoặc learned sparse.
```

Trong nhiều vector DB, sparse vector cho phép lưu dense và sparse signal trong cùng collection/index để chạy hybrid query.

---

## 6. Dense vector trong hybrid search

Dense vector là semantic embedding.

Dense retrieval tốt khi:

```text
query tự nhiên
query dài
query dùng synonym
query và document khác ngôn ngữ
tài liệu không viết đúng keyword user hỏi
```

Ví dụ:

```text
Query:
"quy định lấy lại tiền"

Document:
"refund policy"
```

Dense multilingual model có thể retrieve được.

Nhưng dense vector dễ sai trong các trường hợp:

```text
"refund policy" gần "payment policy"
"API delete user" gần "API create user"
"A-134" gần "A-143"
"Article 7.2" gần "Article 7.3"
```

Hybrid không thay dense. Hybrid bọc dense bằng sparse để giữ exact signal.

---

## 7. Flow hybrid search chuẩn

Pipeline hybrid đơn giản:

```text
User query
-> metadata pre-filter
-> dense retriever lấy top_k_dense
-> sparse/BM25 retriever lấy top_k_sparse
-> merge/fusion candidates
-> deduplicate
-> rerank
-> context assembly
-> LLM answer
```

Baseline config:

```text
dense_top_k = 30
sparse_top_k = 30
fusion = RRF
rerank_top_n = 10
final_context_k = 4-6
```

Trực giác:

```text
Retriever lấy rộng để tăng recall.
Fusion gộp nguồn.
Reranker chọn hẹp để tăng precision.
```

---

## 8. RRF — Reciprocal Rank Fusion

Dense và BM25 có score khác scale:

```text
Dense cosine score: 0.82
BM25 score: 14.7
```

Không thể cộng thẳng:

```text
0.82 + 14.7
```

vì BM25 sẽ áp đảo.

RRF không tin score gốc. Nó dùng rank.

```text
RRF_score(d) = sum(1 / (k + rank_i(d)))
```

Trong đó:

```text
rank_i(d): thứ hạng của document d trong retriever i
k: hằng số làm mượt, thường dùng khoảng 60 trong nhiều setup
```

Ví dụ:

```text
Dense ranking:
A rank 1
B rank 2
C rank 3

Sparse ranking:
B rank 1
D rank 2
A rank 3
```

A tốt vì rank cao trong dense và cũng xuất hiện trong sparse. B tốt vì rank cao trong sparse và cũng xuất hiện trong dense. D chỉ xuất hiện trong sparse, vẫn có cơ hội nhưng yếu hơn nếu nguồn khác không ủng hộ.

Điểm mạnh:

```text
dễ dùng
không cần normalize score phức tạp
robust khi retriever score khác scale
tốt để combine dense/sparse/BM25/multi-query
```

Điểm yếu:

```text
bỏ qua magnitude của score
rank 1 sát nút và rank 1 áp đảo bị đối xử giống nhau
phụ thuộc top-k của từng retriever
chưa chắc tối ưu nếu score đã calibrate tốt
```

RRF là baseline fusion tốt khi chưa có eval/calibration phức tạp.

---

## 9. Weighted fusion

Weighted fusion gán trọng số cho dense và sparse.

```text
final_score = 0.7 * dense_score_norm + 0.3 * sparse_score_norm
```

Nếu query có nhiều exact keyword:

```text
final_score = 0.4 * dense_score_norm + 0.6 * sparse_score_norm
```

Dùng weighted fusion khi có lý do rõ để tin một nguồn quan trọng hơn:

```text
Customer support natural language -> dense weight cao hơn.
Code/legal/API/error code -> sparse weight cao hơn.
Mixed Vietnamese-English -> cân bằng dense multilingual và sparse keyword.
```

Điểm mạnh:

```text
kiểm soát bias dense/sparse
có thể optimize bằng eval set
có thể adaptive theo query type
```

Điểm yếu:

```text
cần score normalization
trọng số sai làm retrieval tệ
score distribution thay đổi theo model/index
khó generalize cho mọi query
```

Không nên tự đặt `dense = 0.5`, `sparse = 0.5` rồi tin là công bằng. Nếu score scale khác nhau, 0.5/0.5 vẫn có thể lệch hẳn về một nguồn.

OpenSearch hybrid search docs nhấn mạnh lexical và semantic score khác scale, nên cần normalization hoặc rank fusion để kết hợp công bằng. ([OpenSearch Hybrid Search][3])

---

## 10. Hybrid reranking

Hybrid reranking là pattern thực dụng nhất:

```text
Dense retrieve top 30
Sparse retrieve top 30
Union candidates
Reranker score query-document
Final top 5
```

Reranker quan trọng vì dense và sparse là retriever nhanh, nhưng không đọc query-document sâu như cross-encoder/reranker.

Ví dụ:

```text
Query:
"enterprise refund after 30 days"

Candidate A:
"standard refund within 30 days"

Candidate B:
"enterprise customers follow separate SLA after 30 days"
```

Dense/BM25 có thể rank A cao vì trùng `refund` và `30 days`. Reranker có thể đưa B lên vì đúng intent `enterprise + after 30 days`.

Điểm mạnh:

```text
tăng precision rõ
sửa ranking sau hybrid
giảm noise trước khi đưa vào LLM
tốt cho legal/finance/policy/code
```

Điểm yếu:

```text
tốn latency
tốn compute
chỉ rerank được candidate, không cứu được nếu retriever miss evidence
```

---

## 11. Khi nào cần hybrid search?

Hybrid rất đáng dùng khi tài liệu có exact tokens quan trọng.

| Domain | Vì sao cần hybrid |
| --- | --- |
| Technical docs | API name, config key, error message, version, library name |
| Legal/compliance | điều khoản, clause number, ngày hiệu lực, tên văn bản |
| Finance/risk/lending | metric name, mã sản phẩm, DPD/FPD/WO, policy code |
| Product catalog/ecommerce | SKU, brand, model, size, color, version |
| Customer support | user hỏi tự nhiên, docs viết formal |
| Codebase assistant | function/class name, import path, file path, stack trace |

Ví dụ codebase:

```text
Query:
"HierarchicalADPPruner r_keep_final bug"
```

Dense có thể lấy pruning docs chung. Sparse giữ đúng class/variable.

---

## 12. Khi nào chưa cần hybrid?

Dense-only có thể tạm đủ khi:

```text
corpus nhỏ
query tự nhiên
không nhiều mã/entity/số
prototype nhanh
có reranker tốt
dữ liệu cùng ngôn ngữ, cùng style
```

Ví dụ:

```text
small FAQ bot
simple knowledge base
demo nội bộ vài trăm docs
```

Nhưng với RAG doanh nghiệp, hybrid thường nên nằm trong baseline hoặc ít nhất là option để bật.

---

## 13. Hybrid với metadata filter

Hybrid search không thay thế metadata filter.

Pipeline đúng:

```text
User query
-> backend permission/date/source filters
-> dense search trong allowed subset
-> sparse search trong allowed subset
-> fusion/rerank
```

Không nên:

```text
dense search toàn DB
+ sparse search toàn DB
-> fusion
-> lọc permission sau
```

Vì top results có thể toàn tài liệu user không được xem. Sau filter, evidence đúng trong allowed subset có thể đã bị miss.

Security filter phải do backend enforce từ auth context, không để LLM quyết định.

---

## 14. Hybrid với query rewriting và multi-query

Hybrid có thể kết hợp query rewriting.

Ví dụ chat history:

```text
User:
"cái đó áp dụng cho intern không?"

Rewrite:
"Chính sách nghỉ phép năm 2026 có áp dụng cho intern không?"
```

Sau đó hybrid:

```text
Dense tìm meaning "áp dụng cho intern".
Sparse giữ keyword "intern", "nghỉ phép", "2026".
```

Multi-query hybrid:

```text
1. chính sách nghỉ phép intern 2026
2. annual leave policy intern
3. leave entitlement for interns
```

Mỗi query chạy dense+sparse, rồi RRF/rerank.

Pattern này tăng recall mạnh nhưng cũng tăng cost/noise. Chỉ dùng khi query mơ hồ, domain có nhiều synonym hoặc tài liệu đa ngôn ngữ.

---

## 15. Hybrid search và chunking

Hybrid không cứu được chunking quá tệ.

Ví dụ table bị mất header:

```text
A | 20M | 40%
```

Sparse có thể match `A`, dense có thể match policy, nhưng LLM vẫn không biết `20M` là min income hay max loan.

Ví dụ code bị cắt ngang:

```text
r_keep_final = max(...)
```

Nếu chunk thiếu class/function context, hybrid retrieve được nhưng context không đủ để answer.

Hybrid cần đi cùng:

```text
table-aware chunking
code-aware chunking
heading injection
metadata tốt
parent-child context expansion
```

---

## 16. Hybrid search và multilingual RAG

Trong môi trường Việt Nam, query thường mixed:

```text
"refund policy cho enterprise customer"
"lỗi HDMI mouse stutter trên Ubuntu"
"BAD rate tính theo FPD30 như nào"
```

Dense multilingual giúp nối nghĩa giữa tiếng Việt và tiếng Anh.

Sparse giữ keyword technical:

```text
refund
enterprise
HDMI
Ubuntu
BAD
FPD30
```

Với tiếng Việt có dấu/không dấu, cần cân nhắc normalize:

```text
hoàn tiền
hoan tien
refund
```

Một setup tốt có thể lưu thêm field normalized text để BM25/sparse bắt được cả dạng không dấu.

---

## 17. Query-adaptive hybrid

Không phải query nào cũng cần dense/sparse weight giống nhau.

Query thiên exact keyword:

```text
"INV-2026-001 refund"
"Article 7.2 termination"
"BAD_FPD10_ASOF"
"TypeError: NoneType is not subscriptable"
```

Nên:

```text
sparse weight cao hơn
exact match boost
dense vẫn chạy để lấy semantic context
```

Query thiên semantic:

```text
"tôi nghỉ bệnh thì lương tính sao?"
"khách có được lấy lại tiền không?"
"chính sách làm remote như nào?"
```

Nên:

```text
dense weight cao hơn
sparse vẫn chạy để giữ keyword
```

Query mixed:

```text
"refund policy cho enterprise customer sau 30 ngày"
```

Nên:

```text
dense và sparse cân bằng
reranker bắt buộc
```

Query-adaptive hybrid cần query classifier hoặc heuristic. Chỉ nên làm sau khi baseline RRF + reranker đã có eval.

---

## 18. Lỗi hybrid search thường gặp

| Lỗi | Triệu chứng | Cách sửa |
| --- | --- | --- |
| Cộng dense + BM25 trực tiếp | BM25 hoặc dense áp đảo vì khác scale | RRF, score normalization, reranker |
| Sparse kéo keyword noise | Nhiều chunk chứa keyword phụ nhưng sai intent | reranker, field boosting, phrase match |
| Dense kéo semantic gần nhưng sai fact | `delete user API` retrieve `create user API` | sparse exact match, code-aware chunking |
| Duplicate candidates | Dense và sparse cùng lấy nhiều chunk giống nhau | dedup by `chunk_id`, group by `parent_id` |
| Top-k từng retriever quá nhỏ | Hybrid không tăng recall đáng kể | tăng `dense_top_k`, `sparse_top_k` rồi rerank |
| Không eval ablation | Không biết hybrid có thật sự giúp không | đo dense-only, sparse-only, hybrid, hybrid+reranker |
| Filter sau fusion | Security/correctness sai hoặc miss allowed evidence | metadata pre-filter trong từng retriever |

---

## 19. Eval hybrid search

Cần eval set có hard negatives.

Ví dụ policy:

```text
Query:
"khách enterprise có được hoàn tiền sau 30 ngày không?"

Positive:
enterprise refund SLA exception

Hard negatives:
standard refund within 30 days
enterprise payment terms
warranty policy
```

Ví dụ code:

```text
Query:
"HierarchicalADPPruner r_keep_final"

Positive:
class/function chứa r_keep_final logic

Hard negatives:
ActionAwarePRouter rho_target
generic token pruning docs
other keep ratio configs
```

Metrics nên đo:

```text
Recall@10 / Recall@50
MRR
nDCG@10
Precision@k sau rerank
Answer groundedness
Latency
Cost
```

Ablation nên có:

```text
A. Dense only
B. Sparse only
C. Dense + sparse RRF
D. Dense + sparse + reranker
E. Dense + sparse + reranker + metadata filter
```

Nếu hybrid tốt, thường thấy:

```text
Recall@k tăng
MRR/nDCG tăng sau rerank
hallucination giảm
citation đúng hơn
```

---

## 20. Baseline config

Cho RAG doanh nghiệp vừa phải:

```text
dense_top_k = 30
sparse_top_k = 30
fusion = RRF
rerank_top_n = 10
final_context_k = 4-6
metadata_pre_filter = permission + source + version/date
dedup = chunk_id + parent_id
```

Nếu data có code/legal/finance:

```text
sparse_top_k = 50
exact token boost
reranker bắt buộc
```

Nếu data customer support tự nhiên:

```text
dense_top_k = 50
multi-query optional
sparse_top_k = 20-30
reranker nếu noise cao
```

Nếu latency rất chặt:

```text
dense_top_k = 20
sparse_top_k = 20
RRF
rerank_top_n = 5 hoặc bỏ reranker
cache frequent queries
```

---

## 21. Tool support cần biết

| Tool | Ghi chú |
| --- | --- |
| Qdrant | dense/sparse/named vectors, hybrid query, RRF/DBSF, payload filtering |
| OpenSearch / Elasticsearch | mạnh về BM25/lexical search, có vector search và hybrid pipeline |
| Pinecone | managed vector DB, hỗ trợ dense/sparse/full-text tùy index type |
| Weaviate | production vector DB, structured filtering, hybrid search tùy setup |
| Milvus | vector DB scale lớn, nhiều index type, cần hiểu deployment/ops |
| FAISS | mạnh cho dense ANN/local benchmark, không phải full hybrid DB nếu không tự build thêm |

Pinecone docs mô tả serverless index có thể dùng dense vector field, sparse vector field và string fields cho full-text/BM25. ([Pinecone Indexing][4])

---

## 22. Thứ tự học thực dụng

```text
1. BM25
2. Dense retrieval
3. Sparse vector
4. Dense vs sparse failure cases
5. RRF
6. Weighted fusion
7. Hybrid reranking
8. Metadata pre-filter + hybrid
9. Query-adaptive hybrid
10. Eval dense-only vs sparse-only vs hybrid
```

---

## 23. Kết luận

Hybrid search là một trong những nâng cấp đáng tiền nhất cho RAG production.

Naive RAG:

```text
query -> dense vector search top-5 -> LLM
```

Production baseline tốt hơn:

```text
query
-> metadata pre-filter
-> dense retrieval top-30
-> sparse/BM25 retrieval top-30
-> RRF hoặc normalized fusion
-> reranker top-5
-> context assembly
-> grounded answer
```

Mindset:

```text
Dense giúp không bị kẹt vào keyword.
Sparse giúp không bỏ mất keyword quan trọng.
Hybrid giúp tăng recall.
Reranker giúp tăng precision.
```

Khi tài liệu có mã lỗi, tên sản phẩm, API name, điều khoản pháp lý, số hợp đồng, field name, metric name hoặc thuật ngữ domain, dense-only rất dễ sai. Trong nhiều RAG doanh nghiệp, hybrid search nên được xem là baseline nghiêm túc.

[1]: https://qdrant.tech/documentation/search/hybrid-queries/ "Hybrid Queries - Qdrant"
[2]: https://www.elastic.co/search-labs/blog/elasticsearch-scoring-and-explain-api "Understanding Elasticsearch scoring and the Explain API"
[3]: https://opensearch.org/blog/building-effective-hybrid-search-in-opensearch-techniques-and-best-practices/ "Building effective hybrid search in OpenSearch"
[4]: https://docs.pinecone.io/guides/index-data/indexing-overview "Indexing overview - Pinecone Docs"
