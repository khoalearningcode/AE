# Embedding — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Chunking
- Vai trò: mở rộng phần `## 5. Embedding` trong RAG Foundation.

## Must understand

- Embedding biến query/document/chunk thành vector để tạo retrieval candidates, không phải bằng chứng đúng tuyệt đối.
- Dense, sparse, hybrid, multi-vector/late interaction giải các loại lỗi retrieval khác nhau.
- Similarity metric, normalization và embedding dimension ảnh hưởng trực tiếp tới ranking, latency, storage và cost.
- Domain, multilingual và instruction embedding phải được chọn theo dữ liệu thật và eval set.
- Embedding quality phải đo bằng retrieval eval, không đo bằng cảm giác search.

## Must practice

- So sánh dense-only, sparse-only và hybrid retrieval trên cùng query set.
- Tạo hard negatives cho các case gần nghĩa nhưng sai evidence.
- Đo Recall@k, MRR, Hit rate và latency.
- Test query tiếng Việt, tiếng Anh và mixed Vietnamese-English.
- Kiểm tra tác động của chunk size lên embedding retrieval.

## Can explain when ready

- Vì sao embedding similarity không đồng nghĩa correctness?
- Khi nào dense retrieval fail và sparse/BM25 cứu được?
- Vì sao cần reranker sau embedding retrieval?
- Chọn embedding model cho RAG doanh nghiệp dựa trên tiêu chí nào?

---

## 1. Embedding trong RAG là gì?

Embedding là cách biến text/image/document/code/table thành một vector số để máy có thể so sánh “độ gần” giữa query và tài liệu.

Flow cơ bản:

```text
document
→ chunk
→ embedding model
→ vector
→ lưu vào vector DB

user query
→ embedding model
→ query vector
→ so với document vectors
→ lấy top-k gần nhất
→ đưa vào LLM
```

Sentence Transformers mô tả semantic search đúng theo flow này: embed query và corpus vào cùng vector space, rồi tìm corpus embeddings gần query embedding nhất, mặc định thường dùng cosine similarity. ([sbert.net][1])

Nhưng câu quan trọng nhất là:

```text
Embedding không nói “đây là bằng chứng đúng”.
Embedding chỉ nói “model nghĩ hai đoạn này gần nhau trong vector space”.
```

Ví dụ:

```text
Query: chính sách hoàn tiền

Chunk A: chính sách hoàn tiền trong vòng 30 ngày
Chunk B: chính sách thanh toán qua thẻ ngân hàng
```

Dense embedding có thể thấy A và B đều gần chủ đề “payment/customer policy”, nhưng chỉ A mới là evidence đúng.

Vậy trong RAG, embedding chỉ là **retrieval candidate generator**, không phải bộ phán xét cuối cùng.

---

# 2. Timeline tiến hoá của embedding trong retrieval/RAG

Đây là timeline thực dụng, không phải ngày phát minh tuyệt đối.

| Giai đoạn              | Kiểu retrieval/embedding                           | Vì sao xuất hiện                                                             | Điểm yếu làm sinh ra đời sau                                  |
| ---------------------- | -------------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------------------- |
| Trước neural retrieval | **Sparse keyword retrieval: TF-IDF, BM25**         | Search theo keyword nhanh, dễ giải thích                                     | Không hiểu synonym/paraphrase tốt                             |
| 2013–2018              | **Word embedding / static embedding**              | Word có vector semantic: king, queen, city, country                          | Không đủ tốt cho sentence/document meaning                    |
| 2019                   | **Sentence embedding / SBERT**                     | Cần vector cho cả câu/đoạn để semantic search nhanh                          | General semantic tốt, nhưng chưa tối ưu mạnh cho QA retrieval |
| 2020                   | **Dense Passage Retrieval / dual encoder**         | Train query encoder + passage encoder cho open-domain QA                     | Dense đôi khi miss keyword chính xác, entity, số, code        |
| 2020+                  | **Late interaction / multi-vector: ColBERT**       | Muốn giữ fine-grained token-level matching nhưng vẫn nhanh hơn cross-encoder | Index nặng hơn dense vector đơn                               |
| 2022+                  | **Instruction embedding**                          | Cùng một text nhưng task khác nhau cần representation khác nhau              | Cần prompt/query format đúng                                  |
| 2023–2026              | **Hybrid dense + sparse + rerank**                 | Dense hiểu nghĩa, sparse giữ keyword/entity/số                               | Pipeline phức tạp hơn nhưng production tốt hơn                |
| 2024+                  | **Multilingual / domain / long-context embedding** | RAG doanh nghiệp có nhiều ngôn ngữ, domain, document dài                     | Cần eval riêng theo domain                                    |

SBERT năm 2019 giải quyết vấn đề BERT gốc không phù hợp cho large-scale semantic similarity search, vì BERT kiểu pairwise phải chạy quá nhiều cặp; SBERT dùng siamese/triplet network để tạo sentence embeddings so được bằng cosine similarity. ([ACL Anthology][2])

DPR năm 2020 đẩy dense retrieval lên một mốc quan trọng cho open-domain QA: thay vì chỉ dùng BM25/TF-IDF, nó học dense representations bằng dual-encoder query/passage và cho thấy dense retrieval có thể cạnh tranh mạnh với sparse retrieval truyền thống. ([arXiv][3])

ColBERT cũng năm 2020 sinh ra vì dense vector một-vector-cho-một-passage có thể mất fine-grained token matching; ColBERT encode query và document độc lập rồi dùng late interaction để so chi tiết hơn ở mức token, nhanh hơn nhiều so với re-rank toàn bộ bằng cross-encoder nặng. ([arXiv][4])

Instruction embedding như INSTRUCTOR năm 2022 xuất hiện vì cùng một đoạn text có thể cần vector khác nhau tùy task/domain; INSTRUCTOR embed text kèm instruction mô tả use case, task hoặc domain. ([arXiv][5])

---

# 3. Dense embedding

## Ý tưởng

Dense embedding là vector dày, thường hàng trăm đến vài nghìn chiều.

Ví dụ đơn giản:

```text
"chính sách hoàn tiền"
→ [0.12, -0.03, 0.44, ..., 0.08]
```

Vector này không interpretable từng chiều theo kiểu:

```text
dimension 1 = refund
dimension 2 = price
dimension 3 = customer
```

Thực tế mỗi chiều là latent feature model học được.

Dense embedding giỏi ở semantic similarity:

```text
"chính sách hoàn tiền"
≈ "điều kiện refund"
≈ "quy định trả lại tiền"
≈ "refund policy"
```

## Vì sao dense embedding ra đời?

Vì sparse keyword search gặp lỗi synonym/paraphrase.

Ví dụ query:

```text
"được trả lại tiền không?"
```

Document viết:

```text
"khách hàng có thể yêu cầu hoàn tiền trong 30 ngày"
```

BM25/sparse keyword có thể miss nếu không trùng từ. Dense embedding có thể bắt được nghĩa “trả lại tiền” gần với “hoàn tiền”.

## Điểm mạnh

Dense embedding mạnh ở:

```text
- synonym
- paraphrase
- câu hỏi tự nhiên
- semantic search
- cross-lingual nếu model hỗ trợ
- document không dùng đúng keyword của query
```

Ví dụ trong chatbot doanh nghiệp:

```text
Query: Nhân viên nghỉ bệnh thì lương tính sao?

Relevant chunk:
"Nhân viên nghỉ ốm có giấy xác nhận y tế sẽ được hưởng chế độ..."
```

Dense retrieval có thể tìm được dù query không dùng đúng cụm “nghỉ ốm có giấy xác nhận y tế”.

## Điểm yếu

Dense embedding dễ fail ở những thứ cần match chính xác:

```text
- mã sản phẩm
- tên người
- số hợp đồng
- version
- ngày tháng
- lỗi code
- điều khoản pháp lý
- tên field trong database
```

Ví dụ:

```text
Query: policy A-134 refund
```

Dense model có thể retrieve policy A-143 vì semantic gần, nhưng sai mã.

Hoặc:

```text
Query: BAD_FPD10_ASOF
```

Dense embedding có thể retrieve `BAD_FPD30_ASOF` vì gần nghĩa, nhưng với kỹ thuật thì đây là sai.

## Kết luận

Dense embedding là nền tảng của semantic RAG, nhưng không nên dùng một mình nếu tài liệu có nhiều keyword/entity/số/code.

---

# 4. Sparse embedding

## Ý tưởng

Sparse embedding là vector thưa, phần lớn chiều bằng 0. Nó gần với keyword retrieval.

Ví dụ vocabulary có 1 triệu term, một document chỉ active vài chục/nghìn term liên quan:

```text
refund: 0.92
payment: 0.51
policy: 0.74
...
còn lại gần như 0
```

Sparse truyền thống gồm:

```text
TF-IDF
BM25
keyword inverted index
```

Sparse hiện đại có thể là learned sparse retrieval, tức model học cách expand hoặc weight token.

## Vì sao sparse tồn tại lâu và vẫn sống?

Vì nó cực kỳ tốt với exact match.

Ví dụ:

```text
Query: "Điều 12.4.1"
Query: "invoice INV-2025-0093"
Query: "function calculate_fpd_score"
Query: "RTX 3090 CUDA out of memory"
```

Sparse/BM25 thường tốt hơn dense ở các case này.

DPR paper cũng nói trước dense retrieval, sparse vector space models như TF-IDF/BM25 là phương pháp de facto trong open-domain QA retrieval. ([arXiv][3])

## Điểm mạnh

Sparse mạnh ở:

```text
- exact keyword
- số liệu
- mã định danh
- tên riêng
- acronym
- technical term
- code symbol
- pháp lý/điều khoản
```

## Điểm yếu

Sparse yếu ở semantic mismatch.

Ví dụ:

```text
Query: có được lấy lại tiền không?
Document: refund policy allows reimbursement...
```

Nếu query tiếng Việt, document tiếng Anh, hoặc dùng từ khác, sparse có thể miss.

## Kết luận

Sparse không outdated. Trong production RAG, sparse thường là lớp bảo hiểm cho dense.

---

# 5. Hybrid retrieval: dense + sparse

Hybrid retrieval sinh ra vì dense và sparse sửa lỗi cho nhau.

```text
Dense:
+ hiểu nghĩa
- dễ sai keyword/entity/số

Sparse:
+ exact match tốt
- kém synonym/paraphrase
```

Hybrid làm kiểu:

```text
query
→ dense search top-k
→ sparse/BM25 search top-k
→ merge results
→ rerank
→ chọn context cuối
```

Ví dụ:

```text
Query: "A-134 refund policy for enterprise customer"
```

Dense sẽ bắt được:

```text
refund policy
enterprise customer
```

Sparse sẽ giữ:

```text
A-134
```

Nếu chỉ dense, có thể retrieve nhầm policy gần nghĩa. Nếu chỉ sparse, có thể miss đoạn viết “reimbursement” thay vì “refund”. Hybrid là cách cân bằng.

Một số embedding model hiện đại đi theo hướng multi-function. BGE-M3 được mô tả là hỗ trợ multilingual, dense retrieval, sparse retrieval, multi-vector retrieval và xử lý input granularity tới 8192 tokens. ([arXiv][6])

---

# 6. Multi-vector / late interaction embedding

Dense embedding thường compress cả passage thành một vector:

```text
passage 500 tokens → 1 vector
```

Vấn đề: một vector duy nhất có thể làm mất chi tiết.

Ví dụ passage có 5 ý:

```text
refund
warranty
shipping
enterprise SLA
payment method
```

Một vector phải đại diện cho tất cả, dễ bị loãng.

Multi-vector/late interaction làm khác:

```text
passage → nhiều token vectors / nhiều sub-vectors
query → nhiều token vectors
so fine-grained hơn
```

ColBERT là ví dụ tiêu biểu: query và document được encode độc lập, sau đó dùng late interaction để model fine-grained similarity giữa query/document, giúp giữ chi tiết hơn so với single-vector dense retrieval. ([arXiv][4])

## Điểm mạnh

Tốt cho:

```text
- passage dài nhiều chi tiết
- query cần match nhiều cụm nhỏ
- technical/legal retrieval
- khi dense single-vector bị loãng
```

## Điểm yếu

Đắt hơn:

```text
- index lớn hơn
- search phức tạp hơn
- serving khó hơn
- vector DB support không phải lúc nào cũng đơn giản
```

Trong project doanh nghiệp thông thường, nên học dense/sparse/hybrid trước, sau đó mới tới ColBERT/multi-vector.

---

# 7. Embedding dimension

## Dimension là gì?

Embedding dimension là số chiều của vector.

Ví dụ:

```text
384 dimensions
768 dimensions
1024 dimensions
1536 dimensions
3072 dimensions
```

Vector 768 chiều nghĩa là mỗi chunk được biểu diễn bằng 768 số.

## Dimension cao hơn có tốt hơn không?

Không chắc.

Dimension cao hơn thường có khả năng chứa nhiều thông tin hơn, nhưng:

```text
- tốn storage hơn
- search nặng hơn
- index build lâu hơn
- RAM/VRAM cao hơn
- chưa chắc retrieval tốt hơn nếu model/training kém
```

Quan trọng là **model quality + domain fit + eval**, không phải chỉ dimension.

Ví dụ:

```text
Model A: 768 dim, train tốt cho retrieval
Model B: 3072 dim, general embedding nhưng không hợp domain

→ Model A có thể retrieve tốt hơn.
```

## Chi phí dimension

Nếu có 1 triệu chunks:

```text
1 vector 768 dim float32 ≈ 3 KB
1 triệu vectors ≈ 3 GB raw vector

1 vector 1536 dim float32 ≈ 6 KB
1 triệu vectors ≈ 6 GB raw vector
```

Chưa tính metadata, index overhead, replicas.

Vậy dimension ảnh hưởng trực tiếp đến:

```text
- storage
- latency
- memory
- cost
- index performance
```

Trong production, embedding dimension là trade-off giữa quality và cost.

---

# 8. Cosine similarity

## Ý tưởng

Cosine similarity so hướng của hai vector, không quan tâm nhiều đến độ dài vector.

Công thức:

```text
cosine(a, b) = (a · b) / (||a|| * ||b||)
```

Nếu hai vector cùng hướng:

```text
cosine gần 1
```

Nếu vuông góc:

```text
cosine gần 0
```

Nếu ngược hướng:

```text
cosine gần -1
```

## Vì sao hay dùng cosine?

Vì trong semantic embedding, ta thường quan tâm:

```text
hai vector có hướng semantic giống nhau không?
```

không nhất thiết quan tâm vector dài/ngắn.

Sentence Transformers semantic search mặc định dùng cosine similarity cho query embeddings và corpus embeddings. ([sbert.net][1])

## Điểm yếu

Cosine không phải phép màu.

Paper 2024 **Is Cosine-Similarity of Embeddings Really About Similarity?** cảnh báo không nên dùng cosine một cách mù quáng, vì trong một số setting cosine similarity có thể cho similarity opaque hoặc thậm chí không meaningful như ta tưởng. ([arXiv][7])

Trong RAG, nghĩa thực dụng là:

```text
cosine cao chưa chắc chunk là evidence đúng.
cosine thấp chưa chắc chunk hoàn toàn không liên quan.
```

---

# 9. Dot product

## Ý tưởng

Dot product là tích vô hướng:

```text
dot(a, b) = a · b
```

Nó bị ảnh hưởng bởi cả:

```text
- hướng vector
- độ dài vector
```

Nếu vector đã normalize về unit length:

```text
||a|| = 1
||b|| = 1
```

thì:

```text
dot(a, b) = cosine(a, b)
```

OpenAI FAQ nói embeddings của họ được normalized về length 1, nên cosine similarity có thể tính nhanh hơn bằng dot product, và cosine/dot product cho ranking giống nhau trong trường hợp này. ([OpenAI Help Center][8])

Qdrant cũng giải thích nếu normalize vectors về norm 1 thì dot product tương đương cosine similarity. ([qdrant.tech][9])

## Khi nào dùng dot product?

Dùng dot product khi:

```text
- model được train/eval với dot product
- vector đã normalized
- vector DB/index tối ưu cho dot
```

Nhưng không được tự ý đổi metric nếu model card khuyên dùng metric khác.

---

# 10. Normalization

## Normalization là gì?

Normalization thường là biến vector về độ dài 1:

```text
v_norm = v / ||v||
```

Sau normalization:

```text
vector chỉ còn hướng
độ dài bị loại bỏ
```

## Vì sao cần normalization?

Vì nếu không normalize, vector có norm lớn có thể thắng dot product không phải vì semantic gần hơn, mà vì vector dài hơn.

Ví dụ:

```text
A gần nghĩa vừa phải nhưng norm lớn
B gần nghĩa hơn nhưng norm nhỏ

dot product có thể chọn A
```

Normalize giúp ranking tập trung vào hướng semantic.

## Nhưng có phải lúc nào cũng normalize?

Không. Phải theo model.

Có model được train để dùng cosine với normalized embeddings. Có model dùng dot product. Có vector DB tự normalize khi dùng cosine. Có model có norm mang signal riêng.

Qdrant hỗ trợ các metric phổ biến như dot product, cosine similarity và Euclidean distance trong collection/vector search config. ([qdrant.tech][10])

Mindset đúng:

```text
Không chọn metric theo cảm giác.
Chọn theo model card + benchmark thực tế.
```

---

# 11. Domain embedding

## Ý tưởng

Domain embedding là embedding model hợp với ngôn ngữ/chuyên môn của dữ liệu.

Ví dụ RAG cho:

```text
- legal
- finance
- medical
- code
- academic paper
- Vietnamese internal policy
- customer support
```

thì general embedding model có thể không hiểu đủ tốt thuật ngữ domain.

## Vì sao domain embedding quan trọng?

Vì semantic similarity phụ thuộc vào representation mà model học được.

Ví dụ trong credit risk:

```text
BAD
FPD
DPD
WO
roll rate
vintage
```

Người ngoài domain có thể không hiểu. General embedding cũng có thể map sai hoặc không đủ sắc.

Trong code:

```text
class Router
token pruning
layer skipping
attention mask
```

Embedding general có thể hiểu lỏng, nhưng code/domain embedding hoặc hybrid keyword sẽ tốt hơn.

## Các cách xử lý domain

Có 4 mức:

```text
Level 1: chọn embedding model tốt hơn, multilingual/domain-friendly
Level 2: query/document instruction đúng domain
Level 3: hybrid dense + sparse để giữ thuật ngữ
Level 4: fine-tune embedding bằng query-positive-negative pairs domain
```

Fine-tune chỉ nên làm khi đã có eval set đủ tốt. Không nên fine-tune chỉ vì “domain-specific nghe hay”.

---

# 12. Multilingual embedding

## Ý tưởng

Multilingual embedding giúp query và document khác ngôn ngữ vẫn gần nhau.

Ví dụ:

```text
Query tiếng Việt:
"chính sách hoàn tiền cho khách doanh nghiệp"

Document tiếng Anh:
"Enterprise customers are covered by a separate refund SLA."
```

Multilingual embedding tốt có thể map hai đoạn này gần nhau.

## Vì sao quan trọng trong bối cảnh Việt Nam?

Vì môi trường Việt Nam thường có dữ liệu lẫn:

```text
- tài liệu tiếng Anh
- chat/email tiếng Việt
- thuật ngữ kỹ thuật tiếng Anh
- câu hỏi người dùng tiếng Việt
- document policy song ngữ
```

Nếu dùng English-only embedding, query tiếng Việt có thể retrieve rất kém.

BGE-M3 paper mô tả model này hỗ trợ hơn 100 ngôn ngữ, đồng thời hỗ trợ dense retrieval, sparse retrieval và multi-vector retrieval. ([arXiv][6])

## Lỗi thường gặp

Multilingual không có nghĩa là tốt mọi ngôn ngữ/domain.

Bạn vẫn phải test:

```text
Vietnamese query → Vietnamese doc
Vietnamese query → English doc
English query → Vietnamese doc
Mixed query → mixed doc
```

Ví dụ mixed query:

```text
"refund policy cho enterprise customer"
```

Đây là kiểu query rất thực tế ở Việt Nam.

---

# 13. Instruction embedding

## Ý tưởng

Một số embedding model cần hoặc hưởng lợi từ instruction.

Thay vì embed query trần:

```text
"chính sách hoàn tiền"
```

ta embed:

```text
"Represent this question for retrieving relevant documents: chính sách hoàn tiền"
```

Hoặc:

```text
"query: chính sách hoàn tiền"
"passage: khách hàng có thể yêu cầu hoàn tiền..."
```

INSTRUCTOR là hướng rõ ràng nhất: embed text cùng instruction mô tả task/domain để tạo embedding phù hợp cho downstream task. ([arXiv][5])

BGE model card cũng có hướng dẫn rằng với short-query-to-long-passage retrieval, query nên có instruction, còn passage thì không cần instruction. ([Hugging Face][11])

## Vì sao instruction embedding ra đời?

Vì cùng một text có thể dùng cho nhiều task:

```text
Task A: semantic search
Task B: classification
Task C: clustering
Task D: duplicate detection
Task E: question-answer retrieval
```

Câu:

```text
"Apple released a new product"
```

Nếu task là finance, “Apple” là công ty. Nếu task là food taxonomy, “apple” có thể là trái cây. Instruction giúp model hiểu context sử dụng.

## Điểm yếu

Instruction sai có thể làm retrieval tệ hơn.

Ví dụ document cũng bị nhét query instruction:

```text
"Represent this question..."
```

trong khi nó là passage, sẽ làm embedding lệch.

Hoặc query instruction tiếng Anh nhưng query/document chủ yếu tiếng Việt, model có thể vẫn ổn, nhưng nên test.

---

# 14. Embedding gần nhau không đồng nghĩa chắc chắn đúng

Đây là lỗi rất hay gặp khi mới học RAG.

Embedding similarity cao có thể do:

```text
- cùng topic rộng
- cùng từ khóa
- cùng sentiment
- cùng style
- cùng domain
- cùng entity nhưng khác fact
```

Ví dụ:

```text
Query:
"chính sách hoàn tiền"

Retrieved:
"chính sách thanh toán"
```

Hai đoạn gần nhau vì đều liên quan payment/customer policy, nhưng không phải evidence đúng.

Ví dụ khác:

```text
Query:
"lãi suất vay mua nhà năm 2026"

Retrieved:
"lãi suất vay mua xe năm 2026"
```

Semantic rất gần, nhưng entity “nhà” và “xe” khác nhau.

Hoặc:

```text
Query:
"delete user API"

Retrieved:
"create user API"
```

Code/API context gần, nhưng action ngược nhau.

Kết luận:

```text
Embedding retrieval chỉ lấy candidate.
Reranker / metadata filter / exact match / LLM verification mới giúp giảm sai.
```

---

# 15. Embedding model khác reranker như thế nào?

Đây là phần rất quan trọng trong RAG.

## Embedding model / bi-encoder

Embedding model thường encode query và document riêng:

```text
query → vector
document → vector
similarity(query_vector, document_vector)
```

Ưu điểm:

```text
- nhanh
- precompute document vectors được
- scale tốt tới hàng triệu chunks
```

Nhược điểm:

```text
- interaction giữa query-document bị nén
- dễ retrieve gần nghĩa nhưng sai evidence
```

## Reranker / cross-encoder

Reranker thường đọc query và document cùng lúc:

```text
[query, document] → relevance score
```

Ưu điểm:

```text
- chính xác hơn
- hiểu quan hệ query-document tốt hơn
- bắt được contradiction/condition/entity tốt hơn
```

Nhược điểm:

```text
- chậm hơn
- không thể chạy trên toàn bộ corpus
- thường chỉ rerank top 20/50/100 từ retriever
```

Production RAG thường làm:

```text
embedding retrieve top 50
→ reranker chọn top 5/10
→ LLM answer
```

Đây là lý do embedding không cần hoàn hảo tuyệt đối, nhưng phải retrieve đủ recall.

---

# 16. Các lỗi embedding phổ biến trong RAG

## Lỗi 1: Semantic gần nhưng evidence sai

```text
Query: chính sách hoàn tiền
Retrieved: chính sách thanh toán
```

Cách sửa:

```text
- hybrid dense + sparse
- reranker
- metadata filter theo document type/section
- eval bằng query thật
```

## Lỗi 2: Miss keyword/entity/số

```text
Query: "Điều 7.2"
Dense retrieve: Điều 7.1 hoặc 7.3
```

Cách sửa:

```text
- BM25/sparse
- exact filter
- metadata
- chunk chứa heading/section number
```

## Lỗi 3: Multilingual mismatch

```text
Query tiếng Việt
Document tiếng Anh
Model English-only
```

Cách sửa:

```text
- multilingual embedding
- translate query
- dual index: original + translated fields
- eval cross-lingual
```

## Lỗi 4: Domain mismatch

```text
Query: DPD bucket transition
General model retrieve: generic loan document
```

Cách sửa:

```text
- domain-specific examples
- query expansion
- fine-tune embedding
- hybrid sparse giữ acronym
```

## Lỗi 5: Chunk quá lớn làm embedding loãng

```text
Chunk chứa refund + warranty + shipping + SLA
Query refund
Vector đại diện nhiều topic nên retrieval không sắc
```

Cách sửa:

```text
- chunk nhỏ hơn
- semantic chunking
- parent-child retrieval
```

## Lỗi 6: Chunk quá nhỏ làm mất context

```text
Chunk:
"Enterprise customers are excluded."

Nhưng excluded khỏi điều kiện nào?
```

Cách sửa:

```text
- parent-child
- overlap hợp lý
- heading injection
- context expansion
```

---

# 17. Chọn embedding model như thế nào cho RAG doanh nghiệp?

Không chọn theo leaderboard mù. Chọn theo bài toán.

Checklist:

```text
1. Ngôn ngữ
- Vietnamese?
- English?
- mixed Vietnamese-English?
- cross-lingual query/doc?

2. Domain
- policy/legal?
- finance?
- code?
- support ticket?
- research paper?

3. Query type
- natural language?
- keyword/entity?
- code symbol?
- số hợp đồng?
- câu hỏi dài?

4. Document type
- plain text?
- PDF?
- table?
- code?
- slide?
- email/chat?

5. Scale
- 10k chunks?
- 1M chunks?
- 100M chunks?

6. Latency/cost
- local embedding?
- API embedding?
- batch offline?
- real-time query?

7. Metric support
- cosine?
- dot?
- normalized vectors?
- vector DB config?

8. Eval
- recall@k?
- MRR?
- nDCG?
- answer correctness?
```

---

# 18. Eval embedding đúng cách

Đừng đánh giá embedding bằng cảm giác “search thấy có vẻ đúng”.

Cần tạo test set:

```text
query
positive chunk/document
hard negative chunks/documents
```

Ví dụ:

```text
Query:
"khách enterprise có được hoàn tiền sau 30 ngày không?"

Positive:
section nói enterprise refund SLA

Hard negative:
standard refund policy 30 days
payment policy
enterprise payment terms
```

Metrics nên biết:

| Metric              | Ý nghĩa                                          |
| ------------------- | ------------------------------------------------ |
| **Recall@k**        | Positive có nằm trong top-k không                |
| **MRR**             | Positive xuất hiện càng sớm càng tốt             |
| **nDCG**            | Xếp hạng có đúng mức relevance không             |
| **Hit rate**        | Có retrieve được ít nhất một evidence đúng không |
| **Answer accuracy** | Sau retrieval + LLM, answer có đúng không        |

Với RAG, ưu tiên đầu tiên là:

```text
Recall@k cao
```

Vì nếu evidence không vào top-k, LLM không có gì để trả lời đúng.

Sau đó mới tối ưu:

```text
precision
reranking
context compression
latency
cost
```

---

# 19. Công thức pipeline embedding production

Một pipeline RAG thực dụng thường là:

```text
1. Parse document
2. Clean text
3. Chunk
4. Add metadata
5. Embed chunks
6. Store vectors + metadata
7. Query rewrite/normalize
8. Embed query
9. Dense retrieve
10. Sparse retrieve
11. Merge candidates
12. Rerank
13. Deduplicate
14. Context assembly
15. LLM answer with citations
```

Embedding chỉ nằm ở bước 5 và 8, nhưng chất lượng phụ thuộc cả pipeline.

Nếu parser/chunking/metadata sai, embedding model tốt cũng không cứu được.

---

# 20. Thứ tự học embedding hợp lý

Thứ tự học khuyến nghị:

```text
1. Dense embedding
   → hiểu semantic vector search

2. Cosine / dot product / normalization
   → hiểu vì sao vector gần nhau

3. Embedding dimension
   → hiểu cost, memory, latency

4. Sparse retrieval / BM25
   → hiểu exact keyword search

5. Hybrid retrieval
   → hiểu vì sao production không chỉ dense

6. Multilingual embedding
   → rất quan trọng với tiếng Việt + tiếng Anh

7. Instruction embedding
   → biết query/document format theo model

8. Domain embedding / fine-tuning
   → chỉ học sâu sau khi biết eval

9. Multi-vector / ColBERT / late interaction
   → advanced retrieval, tốt cho precision nhưng phức tạp

10. Embedding evaluation
   → phần quan trọng nhất để đi làm
```

---

# 21. Một cách nhớ nhanh

```text
Dense embedding:
"Tìm theo nghĩa."

Sparse embedding:
"Tìm theo từ khóa/chính xác."

Hybrid:
"Dense hiểu nghĩa, sparse giữ keyword."

Dimension:
"Số chiều vector, ảnh hưởng quality/cost nhưng không quyết định tất cả."

Cosine:
"So hướng vector."

Dot product:
"So tích vô hướng; nếu normalized thì gần như cosine."

Normalization:
"Đưa vector về cùng độ dài để so hướng công bằng hơn."

Domain embedding:
"Model phải hiểu ngôn ngữ chuyên ngành."

Multilingual embedding:
"Query và document khác ngôn ngữ vẫn phải gặp nhau."

Instruction embedding:
"Nói cho model biết đang embed để làm gì."

Reranker:
"Embedding tìm ứng viên, reranker chọn bằng chứng tốt hơn."
```

---

# 22. Kết luận thực dụng cho RAG Engineer

Với RAG doanh nghiệp, embedding tốt không phải là model dimension cao nhất hay model mới nhất. Embedding tốt là model giúp pipeline retrieve đúng evidence trong dữ liệu thật.

Baseline nên bắt đầu như này:

```text
Recursive/semantic chunking hợp lý
+ multilingual dense embedding nếu có tiếng Việt/Anh
+ BM25/sparse hybrid
+ metadata filter
+ reranker
+ eval set có hard negatives
```

Mindset quan trọng nhất:

```text
Embedding similarity ≠ correctness.
Embedding retrieval ≠ final answer.
Top-k chunks ≠ evidence đủ tốt.
```

Một RAG engineer giỏi không chỉ biết gọi embedding API. Họ phải biết khi nào dense sai, khi nào sparse cứu được, khi nào cần reranker, khi nào cần domain/multilingual model, và phải đo bằng retrieval eval chứ không đo bằng cảm giác.

[1]: https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html "Semantic Search — Sentence Transformers documentation"
[2]: https://aclanthology.org/D19-1410/ "Sentence Embeddings using Siamese BERT-Networks"
[3]: https://arxiv.org/abs/2004.04906 "Dense Passage Retrieval for Open-Domain Question Answering"
[4]: https://arxiv.org/abs/2004.12832 "ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT"
[5]: https://arxiv.org/abs/2212.09741 "One Embedder, Any Task: Instruction-Finetuned Text Embeddings"
[6]: https://arxiv.org/abs/2402.03216 "BGE M3-Embedding: Multi-Lingual, Multi-Functionality, Multi-Granularity Text Embeddings Through Self-Knowledge Distillation"
[7]: https://arxiv.org/abs/2403.05440 "Is Cosine-Similarity of Embeddings Really About Similarity?"
[8]: https://help.openai.com/en/articles/6824809-embeddings-faq "Embeddings FAQ"
[9]: https://qdrant.tech/course/essentials/day-1/distance-metrics/ "Distance Metrics"
[10]: https://qdrant.tech/documentation/manage-data/collections/ "Collections"
[11]: https://huggingface.co/BAAI/bge-large-en-v1.5 "BAAI/bge-large-en-v1.5"
