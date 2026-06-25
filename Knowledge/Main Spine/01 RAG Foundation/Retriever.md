# Retriever — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Chunking, Embedding, VectorDB
- Vai trò: mở rộng phần retrieval, top-k, filter, hybrid search, query transformation và retrieval eval trong RAG Foundation.

## Must understand

- Retrieval không phải answer generation; retrieval là bước chọn candidate evidence.
- Retriever phải ưu tiên recall: nếu evidence đúng không vào candidate set, reranker và LLM gần như không cứu được.
- `candidate_top_k` và `final_context_k` là hai thứ khác nhau.
- Similarity score không phải điểm đúng tuyệt đối; nó chỉ hữu ích trong cùng model/index/config.
- Metadata filter là một phần của retrieval correctness và security, không phải tính năng phụ.
- Query rewrite, multi-query, decomposition, self-query và multi-hop retrieval chỉ nên dùng khi bài toán thật sự cần.
- Retrieval phải được đánh giá bằng Recall@k, MRR, nDCG, hit rate và failure cases, không đánh giá bằng cảm giác.

## Must practice

- Build simple vector retriever trên một tập Markdown/PDF nhỏ.
- Log query, filters, top-k chunk IDs, scores, source, latency và empty results.
- So sánh `candidate_top_k = 5/20/50` trước và sau rerank.
- Test metadata pre-filter theo permission, version, language hoặc department.
- So sánh dense-only, sparse/BM25-only và hybrid retrieval.
- Tạo eval set có positive evidence và hard negatives.
- Debug ít nhất 5 failure cases: miss evidence, noisy context, wrong version, wrong permission, wrong entity.

## Can explain when ready

- Vì sao retriever nên ưu tiên recall còn reranker ưu tiên precision?
- Khi nào top-k nhỏ làm RAG hallucinate?
- Vì sao score cao không đồng nghĩa evidence đúng?
- Pre-filter khác post-filter ở đâu và vì sao security filter nên pre-filter?
- Khi nào cần query rewrite, multi-query, self-query, decomposition hoặc multi-hop retrieval?
- Hybrid search giải quyết lỗi gì mà dense retrieval không giải quyết tốt?

---

## 1. Retriever trong RAG là gì?

Retriever là thành phần nhận query và trả về các document/chunk có khả năng chứa evidence liên quan.

```text
User query
-> retriever
-> candidate chunks
-> reranker / context assembly
-> LLM answer
```

Câu quan trọng:

```text
Retrieval không phải là answer.
Retrieval là candidate selection.
```

Nhiệm vụ của retriever là đưa evidence đúng vào danh sách ứng viên. Sau đó reranker, context assembly và LLM mới xử lý tiếp.

LangChain định nghĩa retriever là interface trả về documents khi nhận unstructured query; docs của LangChain cũng phân biệt RAG chain đơn giản với agentic RAG, nơi agent có thể quyết định khi nào gọi retrieval tool. ([LangChain Retrieval][1])

---

## 2. Retrieval nằm ở đâu trong pipeline?

Pipeline RAG cơ bản:

```text
User query
-> query processing / rewrite
-> retrieve candidate chunks
-> rerank
-> context assembly
-> LLM answer + citations
```

Naive RAG thường là:

```text
query
-> vector search top-k
-> LLM
```

Production RAG thường có nhiều lớp hơn:

```text
query
-> rewrite / decompose / expand nếu cần
-> metadata pre-filter
-> dense + sparse retrieval
-> candidate merge/fusion
-> deduplicate
-> rerank
-> context assembly
-> LLM
```

Mindset:

```text
Retriever ưu tiên recall.
Reranker ưu tiên precision.
Context assembly ưu tiên đủ evidence nhưng không nhiễu.
LLM ưu tiên grounded answer.
```

---

## 3. Top-k: candidate_top_k và final_context_k

`top-k` là số lượng kết quả retriever lấy ra.

```text
top_k = 5
-> lấy 5 chunks gần query nhất
```

Sai lầm phổ biến là đặt `top_k = 3` hoặc `top_k = 5` rồi coi như xong.

Top-k phụ thuộc vào:

```text
chunk size
embedding quality
query ngắn hay dài
metadata filter
có reranker hay không
câu hỏi đơn giản hay multi-hop
context window của LLM
latency/cost target
```

Cần phân biệt:

```text
candidate_top_k = số chunk lấy ra trước rerank
final_context_k = số chunk đưa vào prompt
```

Ví dụ production hợp lý hơn:

```text
Retriever lấy top 30 hoặc 50
-> reranker chọn top 5
-> LLM chỉ đọc 5 chunks tốt nhất
```

Trade-off:

| Top-k nhỏ | Top-k lớn |
| --- | --- |
| Nhanh hơn | Recall cao hơn |
| Ít noise hơn | Dễ kéo đúng evidence vào candidate set |
| Dễ miss evidence | Tốn latency/cost hơn |
| Context gọn | Cần rerank/dedup/compress |

Kết luận:

```text
Top-k nhỏ tốt cho latency.
Top-k lớn tốt cho recall.
Production thường retrieve rộng rồi rerank hẹp.
```

---

## 4. Similarity score

Similarity score cho biết query vector và chunk vector gần nhau tới mức nào.

Ví dụ cosine similarity:

```text
Chunk A: score 0.86
Chunk B: score 0.79
Chunk C: score 0.72
```

Nhưng:

```text
Score cao không đồng nghĩa chắc chắn đúng.
Score thấp không đồng nghĩa chắc chắn sai.
```

Score phụ thuộc vào:

```text
embedding model
metric: cosine, dot, L2
vector normalization
chunk length
domain
query phrasing
vector DB implementation
```

Không nên so score giữa hai embedding model khác nhau. Cũng không nên cộng trực tiếp dense score với BM25 score:

```text
dense cosine = 0.82
BM25 score = 14.7
```

Hai score này khác scale. Cần fusion, normalize hoặc rerank.

Similarity score hữu ích cho:

```text
ranking tương đối trong cùng setup
threshold sơ bộ
debug retrieval
detect empty/low-confidence retrieval
```

Không dùng score như “độ đúng tuyệt đối”.

---

## 5. Metadata filtering

Metadata filtering là lọc theo thông tin ngoài vector similarity.

Ví dụ metadata:

```json
{
  "source": "hr_policy.pdf",
  "department": "HR",
  "version": "2026",
  "language": "vi",
  "permission_group": "employee",
  "page": 12,
  "effective_date": "2026-01-01"
}
```

Query:

```text
"chính sách nghỉ phép năm nay"
```

Retrieval nên có filter:

```text
department = HR
version = latest
permission_group in user_permissions
effective_date <= today
```

Vì vector similarity không hiểu:

```text
user có quyền xem không
tài liệu đã hết hạn chưa
version nào mới nhất
source nào đáng tin
category nào phù hợp
```

Trong RAG doanh nghiệp, metadata filter là correctness + security.

Ví dụ lỗi nghiêm trọng:

```text
User hỏi về lương thưởng.
Retriever lấy chunk từ executive_compensation_private.pdf.
LLM trả lời dựa trên tài liệu user không có quyền xem.
```

Đây là lỗi phân quyền, không chỉ là lỗi retrieval.

---

## 6. Pre-filter và post-filter

Pre-filter là lọc trước khi search.

```text
1. Lọc documents user được phép xem.
2. Search vector trong subset hợp lệ.
```

Ưu điểm:

```text
đúng security hơn
top-k nằm trong tập hợp hợp lệ
giảm nguy cơ retrieve private/outdated docs
```

Nhược điểm:

```text
khó implement hơn
filter phức tạp có thể làm ANN chậm hơn
phụ thuộc vector DB hỗ trợ filter tốt
```

Post-filter là search trước, lọc sau.

```text
1. Vector search toàn DB top-k.
2. Lọc metadata sau.
```

Rủi ro:

```text
Search toàn DB top 10:
8 private docs
2 outdated docs

Post-filter:
remove private/outdated
-> còn 0 result

Trong khi relevant public doc nằm rank 25.
```

Kết luận:

```text
Permission/security filter nên là pre-filter.
Date/version/category filter quan trọng cũng nên pre-filter nếu DB hỗ trợ.
Post-filter chỉ hợp với cleanup nhẹ.
```

---

## 7. Threshold

Threshold là ngưỡng score tối thiểu.

```text
Chỉ lấy chunk nếu score >= 0.75
```

Mục tiêu là tránh đưa chunk quá irrelevant vào LLM. Nhưng threshold cứng dễ nguy hiểm vì score thay đổi theo query, model, domain, chunk size và metric.

Ví dụ query rất cụ thể:

```text
"BAD_FPD10_ASOF"
```

Embedding model có thể cho score không cao nếu không hiểu field name, nhưng chunk đúng vẫn cần được lấy.

Cách dùng tốt hơn:

```text
retrieve top 30
rerank
nếu best_rerank_score thấp:
  trả lời không tìm thấy đủ bằng chứng
  hoặc hỏi rõ hơn
```

Threshold nên được calibrate bằng eval set, không đặt theo cảm giác.

---

## 8. Query rewriting

Query rewriting là viết lại query người dùng thành câu rõ hơn cho retrieval, đặc biệt trong chat history.

Ví dụ:

```text
User: "cái đó áp dụng cho nhân viên mới không?"

Rewrite:
"Chính sách nghỉ phép năm 2026 có áp dụng cho nhân viên mới thử việc không?"
```

Nó hữu ích khi user query:

```text
ngắn
mơ hồ
dùng đại từ: cái đó, nó, chính sách này
thiếu context từ chat history
sai chính tả
trộn tiếng Việt/Anh
```

Nhưng rewrite có thể đổi intent. Vì vậy rule tốt là:

```text
Rewrite để rõ hơn, không rewrite để thông minh hơn.
```

Query rewrite nên preserve intent, chỉ thêm context đã có, không tự thêm fact và không đoán metadata quan trọng nếu không chắc.

---

## 9. Multi-query retrieval

Multi-query retrieval tạo nhiều biến thể query để tăng recall.

Query gốc:

```text
"chính sách hoàn tiền cho khách doanh nghiệp"
```

Query variants:

```text
1. "enterprise refund policy"
2. "điều kiện refund cho enterprise customer"
3. "SLA hoàn tiền khách hàng doanh nghiệp"
4. "ngoại lệ hoàn tiền sau 30 ngày cho enterprise"
```

Flow:

```text
generate query variants
-> retrieve từng query
-> union candidates
-> deduplicate
-> rerank
```

LangChain `MultiQueryRetriever` dùng LLM viết nhiều query, retrieve docs cho từng query, rồi trả về unique union của retrieved docs. ([LangChain MultiQueryRetriever][2])

Điểm mạnh:

```text
tăng recall khi query ngắn/mơ hồ
hữu ích khi docs dùng từ chuyên môn còn user dùng từ đời thường
hữu ích cho tài liệu song ngữ hoặc nhiều synonym
```

Điểm yếu:

```text
tăng noise
tốn LLM call và retrieval calls
cần dedup/rerank
query phụ quá rộng có thể kéo sai context
```

Cách dùng tốt:

```text
5 query variants
mỗi query retrieve top 10
union còn 30 unique chunks
rerank chọn top 5
```

---

## 10. Query decomposition

Query decomposition tách câu hỏi phức tạp thành các câu hỏi nhỏ.

Query:

```text
So sánh chính sách hoàn tiền của khách thường và khách enterprise sau 30 ngày,
và cho biết trường hợp nào cần approval từ manager.
```

Subqueries:

```text
1. Chính sách hoàn tiền của khách thường sau 30 ngày là gì?
2. Chính sách hoàn tiền của khách enterprise sau 30 ngày là gì?
3. Trường hợp hoàn tiền nào cần manager approval?
4. So sánh điều kiện giữa khách thường và enterprise.
```

Decomposition tốt cho:

```text
compare questions
multi-condition questions
multi-document questions
reasoning questions
questions cần tổng hợp nhiều facts
```

Điểm yếu là LLM có thể sinh subquery sai hoặc thừa. Không nên decompose rồi nhét tất cả chunks vào prompt một cách mù.

Flow tốt hơn:

```text
decompose
-> retrieve từng subquery
-> kiểm tra evidence
-> synthesize answer
```

LlamaIndex docs mô tả query transformations là các module chuyển query thành query khác trước khi chạy qua index, bao gồm các biến đổi như rewrite/decomposition để cải thiện retrieval. ([LlamaIndex Query Transformations][3])

---

## 11. Self-query retrieval

Self-query retrieval dùng LLM chuyển natural language query thành:

```text
semantic query + structured metadata filter
```

Ví dụ:

```text
User:
"Cho tôi chính sách nghỉ phép mới nhất của HR, áp dụng cho nhân viên full-time năm 2026"
```

Parser tạo:

```json
{
  "query": "chính sách nghỉ phép áp dụng cho nhân viên full-time",
  "filter": {
    "department": "HR",
    "employee_type": "full_time",
    "year": 2026,
    "version": "latest"
  }
}
```

Self-query hữu ích khi domain có metadata rõ:

```text
product catalog
policy database
legal docs
ticket search
CRM
document management
real estate
ecommerce
```

Nhưng nó phụ thuộc metadata schema và khả năng parse của LLM. Nếu metadata nghèo hoặc field description không rõ, filter sinh ra dễ sai.

Security filter không nên để LLM quyết định:

```text
LLM có thể tạo filter department/year/source.
Permission filter phải do backend enforce từ auth context.
```

LangChain `SelfQueryRetriever` là retriever dùng vector store và LLM để generate vector store queries. ([LangChain SelfQueryRetriever][4])

---

## 12. Multi-hop retrieval

Multi-hop retrieval là retrieve nhiều bước vì câu trả lời cần evidence nối tiếp nhau.

Ví dụ:

```text
"Người quản lý của team phụ trách policy X có quyền approve refund exception không?"
```

Có thể cần:

```text
Hop 1: policy X thuộc team nào?
Hop 2: manager của team đó là ai?
Hop 3: quyền approve refund exception của manager là gì?
```

Multi-hop tốt cho:

```text
câu hỏi cần reasoning nhiều bước
tài liệu phân tán
compliance/legal
technical troubleshooting
knowledge graph nhẹ
```

Điểm yếu:

```text
chậm hơn
dễ drift
hop đầu sai thì hop sau sai theo
khó eval/debug hơn simple retrieval
```

Nên kiểm soát:

```text
giới hạn số hop
mỗi hop phải lưu evidence
không có evidence thì dừng
final answer phải cite đủ evidence
```

---

## 13. Hybrid search

Hybrid search kết hợp dense semantic search và sparse keyword search.

```text
Dense:
"hoàn tiền" gần "refund" và "reimbursement"

Sparse:
"Article 7.2", "BAD_FPD10_ASOF", "INV-2026-001"
```

Flow:

```text
query
-> dense retrieve top-k
-> sparse/BM25 retrieve top-k
-> merge/fuse candidates
-> rerank
```

Dense và sparse có lỗi ngược nhau:

| Dense | Sparse |
| --- | --- |
| Hiểu meaning tốt | Exact match tốt |
| Yếu số/mã/entity | Yếu synonym/paraphrase |
| Dễ semantic gần nhưng sai | Dễ miss nếu không trùng keyword |

Ví dụ:

```text
Query: "BAD_FPD10_ASOF definition"
```

Dense có thể lấy `BAD_FPD30_ASOF` vì gần cấu trúc. Sparse giữ chính xác `BAD_FPD10_ASOF`.

Fusion có thể dùng:

```text
weighted score
RRF - Reciprocal Rank Fusion
reranker sau khi union candidates
```

RRF hay dùng vì dense score và sparse score khác scale. Qdrant hybrid query docs mô tả việc fuse dense, sparse và multivector results bằng RRF hoặc DBSF. ([Qdrant Hybrid Queries][5])

---

## 14. Retrieval và reranking

Retriever:

```text
Nhiệm vụ: lấy candidate rộng
Ưu tiên: recall
Input: query
Output: top 20/50/100 chunks
```

Reranker:

```text
Nhiệm vụ: xếp lại candidate chính xác hơn
Ưu tiên: precision
Input: query + candidate chunk
Output: top 3/5/10 chunks tốt nhất
```

Flow:

```text
Retriever lấy top 50
-> Reranker chọn top 5
-> LLM answer
```

Tư duy:

```text
Retriever miss evidence thì reranker không cứu được.
Retriever lấy được evidence nhưng rank thấp thì reranker có thể cứu.
```

---

## 15. Retrieval patterns chính

| Pattern | Flow | Dùng khi |
| --- | --- | --- |
| Simple vector retrieval | query -> embed -> vector search top-k | prototype, dữ liệu nhỏ, query đơn giản |
| Filtered vector retrieval | query -> metadata filter -> vector search | enterprise docs, permission, version, category |
| Hybrid retrieval | dense + sparse -> fusion | code/API, legal clause, mã sản phẩm, song ngữ |
| Multi-query retrieval | query variants -> retrieve each -> union -> rerank | query ngắn, mơ hồ, nhiều synonym |
| Decomposition retrieval | complex query -> subqueries -> retrieve each | compare, multi-condition, multi-document |
| Self-query retrieval | natural language -> semantic query + filter | metadata schema tốt, query có điều kiện rõ |
| Agentic retrieval | agent quyết định search tool/query/hop | nhiều nguồn dữ liệu, workflow phức tạp |

Không nên dùng pattern phức tạp chỉ vì nó “ngầu”. Mỗi pattern phải giải quyết một lỗi retrieval cụ thể.

---

## 16. Retrieval eval

Không đánh giá retrieval bằng cảm giác. Cần eval set có:

```text
query
positive evidence chunks
hard negatives
metadata constraints
```

Ví dụ:

```text
Query:
"khách enterprise có được hoàn tiền sau 30 ngày không?"

Positive:
chunk nói enterprise refund SLA exception

Hard negatives:
standard refund 30 days
enterprise payment terms
warranty policy
```

Metrics:

| Metric | Ý nghĩa |
| --- | --- |
| Recall@k | Evidence đúng có nằm trong top-k không |
| Precision@k | Trong top-k có bao nhiêu chunk đúng |
| MRR | Evidence đúng xuất hiện càng sớm càng tốt |
| nDCG | Ranking có phản ánh mức relevance không |
| Hit rate | Có ít nhất một evidence đúng không |
| Answer groundedness | LLM answer có dựa đúng context không |

Với retriever, metric đầu tiên nên nhìn là:

```text
Recall@k
```

Vì retriever là candidate generator.

---

## 17. Lỗi retrieval phổ biến

| Lỗi | Triệu chứng | Cách sửa |
| --- | --- | --- |
| Top-k không chứa evidence đúng | LLM hallucinate dù prompt yêu cầu grounded | tăng `candidate_top_k`, hybrid search, query rewrite, multi-query, chunking lại |
| Retrieve đúng nhưng quá nhiều noise | Context có nhiều chunk không liên quan | reranker, threshold mềm, context compression, dedup |
| Query mơ hồ do chat history | “cái đó áp dụng cho intern không?” | query rewriting / standalone question generation |
| Metadata filter sai | Hỏi policy 2026 nhưng retrieve policy 2024 | version/effective_date metadata, pre-filter latest |
| Dense retrieval sai entity | `A-134 refund` retrieve `A-143 refund` | sparse/BM25, exact match boost, metadata field cho code |
| Multi-query tạo noise | Query phụ quá rộng kéo sai context | giới hạn variants, preserve exact entities, rerank bắt buộc |
| Decomposition sai intent | LLM thêm subquery không liên quan | validate subqueries, chỉ decompose khi cần |
| Security filter để LLM tự quyết | User thấy private context | backend enforce permission filter |

---

## 18. Chọn strategy theo bài toán

| Bài toán | Retrieval baseline |
| --- | --- |
| HR / policy chatbot | metadata pre-filter + dense + BM25/hybrid + reranker |
| Legal/compliance RAG | metadata filter + hybrid + parent-child + reranker + citation strict |
| Codebase assistant | hybrid + code-aware chunking + symbol exact match + repo metadata |
| Research paper QA | section-aware retrieval + query decomposition + table/figure caption retrieval + reranker |
| Customer support bot | query rewriting + multi-query + hybrid + product/version filter |
| Financial/report RAG | metadata filter + table-aware retrieval + hybrid + decomposition + reranker |

---

## 19. Baseline config

Một baseline thực tế để bắt đầu:

```text
candidate_top_k_dense = 30
candidate_top_k_sparse = 30
fusion = RRF
rerank_top_n = 10
final_context_k = 4-6
metadata_filter = permission + source/category/version
```

Điều chỉnh theo query:

```text
Query có mã/entity cụ thể -> ưu tiên hybrid/sparse boost.
Query mơ hồ hoặc conversational -> query rewrite trước.
Query nhiều ý -> query decomposition.
Query có filter tự nhiên -> self-query retrieval, nhưng permission filter do backend enforce.
```

---

## 20. Thứ tự học thực dụng

```text
1. Simple top-k vector retrieval
2. Similarity score và threshold
3. Metadata filtering
4. Pre-filter vs post-filter
5. Hybrid search
6. Reranking
7. Query rewriting
8. Multi-query retrieval
9. Self-query retrieval
10. Query decomposition
11. Multi-hop retrieval
12. Agentic retrieval
```

---

## 21. Kết luận

Retrieval là xương sống của RAG. LLM chỉ trả lời grounded nếu retriever đưa được evidence đúng vào context.

Một baseline production không nên dừng ở:

```text
vector_store.similarity_search(query, k=5)
```

Baseline tốt hơn:

```text
query rewrite nếu có chat history
+ metadata pre-filter
+ hybrid dense/sparse retrieval
+ candidate_top_k đủ lớn
+ reranker
+ dedup/context compression
+ citation/evidence check
```

Nhiều lỗi RAG không phải do LLM yếu, mà do retriever không đưa được evidence đúng vào candidate set, hoặc đưa quá nhiều noise vào context.

[1]: https://docs.langchain.com/oss/python/langchain/retrieval "Retrieval - LangChain Docs"
[2]: https://reference.langchain.com/python/langchain-classic/retrievers/multi_query/MultiQueryRetriever "MultiQueryRetriever - LangChain Reference"
[3]: https://developers.llamaindex.ai/python/framework/optimizing/advanced_retrieval/query_transformations/ "Query Transformations - LlamaIndex"
[4]: https://reference.langchain.com/python/langchain-classic/retrievers/self_query/base/SelfQueryRetriever "SelfQueryRetriever - LangChain Reference"
[5]: https://qdrant.tech/documentation/search/hybrid-queries/ "Hybrid Queries - Qdrant"
