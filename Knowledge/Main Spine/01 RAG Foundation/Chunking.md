# Chunking — RAG Foundation Deep Dive

## Status

Stage: 01 RAG Foundation  
Current level: deep dive note  
Last updated: 2026-06-25

## Vị trí học

- Parent note: RAG Foundation
- Related note: Embedding
- Vai trò: mở rộng phần `## 4. Chunking` trong RAG Foundation.

## Must understand

- Chunk paradox: chunk nhỏ giúp retrieve tốt hơn nhưng dễ thiếu context; chunk lớn giữ context tốt hơn nhưng dễ loãng embedding và tốn token.
- Sự khác nhau giữa fixed-size, recursive, semantic, parent-child, hierarchical, late, table-aware và code-aware chunking.
- Chunking không chỉ là cắt text; nó phụ thuộc parser, metadata, document structure, token budget và retrieval eval.
- Chọn chunking phải dựa trên bài toán, loại tài liệu và metric retrieval/generation.

## Must practice

- Tạo baseline fixed-size/recursive trên cùng một tập tài liệu.
- So sánh chunk size, overlap, top_k bằng Recall@k/MRR/context precision.
- Thử parent-child cho policy/legal docs.
- Log retrieved chunk, parent context, source, score và prompt token cost.
- Tạo failure cases: cắt ngang bảng, mất header, thiếu context cha, chunk quá lớn kéo nhiễu.

## Can explain when ready

- Vì sao không có một `chunk_size` tối ưu cho mọi RAG?
- Khi nào recursive đủ tốt, khi nào cần parent-child/hierarchical?
- Vì sao table/code cần chunking riêng?
- Debug lỗi RAG do chunking như thế nào?

---

## 1. Bức tranh lớn: chunking tiến hóa vì “chunk paradox”

Trong RAG có một mâu thuẫn cốt lõi:

```text
Chunk nhỏ  → embedding sắc, retrieve đúng ý hơn
           → nhưng khi đưa vào LLM thì thiếu context

Chunk lớn  → giữ context tốt hơn
           → nhưng embedding bị loãng, query match kém, tốn token
```

Toàn bộ lịch sử chunking gần như là quá trình giải bài toán này:

```text
Fixed-size
→ xử lý được document dài, nhưng cắt thô theo kích thước

Recursive
→ cắt theo paragraph/sentence tốt hơn, nhưng vẫn không hiểu meaning

Semantic
→ cắt theo ý nghĩa, nhưng tốn compute và khó ổn định

Parent-child
→ retrieve bằng chunk nhỏ, trả về context lớn

Hierarchical
→ biến tài liệu thành cây tri thức nhiều tầng

Late chunking
→ cho embedding model nhìn context dài trước, rồi mới tách chunk

Table/code-aware
→ không coi mọi tài liệu là plain text nữa
```

LangChain hiện vẫn khuyên dùng `RecursiveCharacterTextSplitter` cho generic text vì nó cân bằng khá tốt giữa giữ context và kiểm soát kích thước chunk; default separator của nó ưu tiên `paragraph → line → word → character`, tức là cố giữ các đơn vị ngữ nghĩa gần nhau trước khi phải cắt nhỏ hơn. ([LangChain Docs][1])

---

# 2. Timeline tiến hoá của chunking

Đây không phải “ngày phát minh chính thức”, mà là timeline thực dụng theo cách RAG systems phát triển.

| Giai đoạn              | Kiểu chunking                                             | Vì sao ra đời                                                        | Điểm yếu làm sinh ra đời sau                              |
| ---------------------- | --------------------------------------------------------- | -------------------------------------------------------------------- | --------------------------------------------------------- |
| Trước/đầu RAG          | **Fixed-size**                                            | Cần cắt document cho vừa embedding model/context window              | Cắt giữa câu, giữa bảng, giữa function, mất meaning       |
| RAG phổ biến 2022–2023 | **Recursive**                                             | Giữ paragraph/sentence tốt hơn fixed-size                            | Vẫn không hiểu topic boundary thật sự                     |
| Advanced RAG 2023–2024 | **Parent-child**                                          | Giải mâu thuẫn “chunk nhỏ để retrieve, chunk lớn để answer”          | Context trả về có thể dư, cần docstore + metadata tốt     |
| Advanced RAG 2023–2025 | **Hierarchical**                                          | Tài liệu dài có cấu trúc section → subsection → paragraph            | Index phức tạp, retrieval/merge phức tạp                  |
| 2024 trở đi            | **Semantic / Meta chunking**                              | Muốn cắt theo meaning thay vì ký tự/token                            | Tốn embedding/LLM, khó debug, đôi khi split không ổn định |
| 2024 trở đi            | **Late chunking**                                         | Chunk embedding bị mất context xung quanh                            | Cần long-context embedding model, pipeline phức tạp hơn   |
| 2024–2026              | **Table-aware / code-aware / multimodal structure-aware** | Enterprise docs không chỉ là text: có bảng, PDF, code, slide, layout | Cần parser tốt, metadata tốt, khó generalize              |

Paper **Late Chunking** năm 2024 mô tả rất rõ vấn đề của chunking truyền thống: nếu encode từng chunk riêng lẻ, embedding của chunk có thể mất thông tin từ các chunk xung quanh; late chunking xử lý bằng cách cho long-context embedding model encode toàn văn bản trước, rồi mới chunk representation sau transformer. ([arXiv][2])

Paper **Hierarchical Text Segmentation Chunking** công bố ngày 14/07/2025 cũng đi đúng hướng này: traditional chunking thường không tạo chunk đủ semantic meaning vì không xét cấu trúc văn bản, nên paper đề xuất kết hợp hierarchical text segmentation và clustering để tạo chunk coherent hơn. ([arXiv][3])

---

# 3. Fixed-size chunking

## Ý tưởng

Fixed-size chunking là kiểu đơn giản nhất:

```text
Cứ mỗi 500 tokens cắt một chunk
hoặc mỗi 2,000 characters cắt một chunk
```

Ví dụ:

```text
chunk_size = 500 tokens
chunk_overlap = 50 tokens
```

Document sẽ thành:

```text
Chunk 1: token 0–500
Chunk 2: token 450–950
Chunk 3: token 900–1400
...
```

## Vì sao nó ra đời?

Vì embedding model và LLM đều có giới hạn context. Không thể nhét nguyên PDF 100 trang vào embedding hoặc prompt, nên cách đơn giản nhất là cắt đều ra.

## Điểm mạnh

Fixed-size chunking mạnh ở chỗ **dễ kiểm soát**.

Nó phù hợp khi:

```text
- Dữ liệu khá đồng đều
- Text không có cấu trúc phức tạp
- Cần ingest nhanh
- Cần baseline đầu tiên
- Cần predict cost/token dễ
```

Ví dụ dùng được:

```text
FAQ ngắn
log text
transcript đều đều
simple article
raw notes không nhiều heading
```

## Điểm yếu

Vấn đề lớn nhất: nó **không hiểu ranh giới ý nghĩa**.

Ví dụ câu này:

```text
Refund is only available within 30 days after purchase.
However, enterprise customers are covered by a separate SLA.
```

Nếu cắt ngang giữa hai câu, query hỏi “enterprise refund SLA” có thể retrieve thiếu điều kiện quan trọng.

Với bảng thì còn tệ hơn:

```text
Header: Product | Price | Warranty
Row: A | $100 | 2 years
```

Nếu header nằm chunk trước, row nằm chunk sau, LLM nhìn row sẽ không biết `2 years` là gì.

Với code cũng tương tự: cắt giữa function/class làm mất import, signature, docstring, dependency.

## Khi nào nên học/dùng?

Nên học đầu tiên vì nó là baseline. Nhưng trong production RAG, fixed-size thuần thường chỉ dùng như baseline hoặc fallback.

---

# 4. Recursive chunking

## Ý tưởng

Recursive chunking vẫn có `chunk_size`, nhưng thay vì cắt thẳng theo token/ký tự, nó thử cắt theo separator theo thứ tự ưu tiên:

```text
1. Paragraph: "\n\n"
2. Line: "\n"
3. Space: " "
4. Character: ""
```

Tức là nó sẽ cố giữ paragraph nguyên vẹn. Nếu paragraph quá dài, nó mới cắt xuống sentence/line/word. LangChain document nói rõ recursive splitter thử các separator theo thứ tự đến khi chunk đủ nhỏ, và default separator là `["\n\n", "\n", " ", ""]`. ([LangChain Docs][1])

## Vì sao nó ra đời sau fixed-size?

Vì fixed-size cắt quá “mù”. Recursive chunking ra đời như một bản nâng cấp thực dụng:

```text
Không cần hiểu semantic sâu,
nhưng ít nhất đừng cắt nát paragraph/sentence.
```

## Điểm mạnh

Recursive là default rất tốt cho nhiều RAG hệ thống.

Nó mạnh vì:

```text
- Dễ dùng
- Không cần embedding phụ để quyết định split
- Không cần LLM
- Ít phá câu hơn fixed-size
- Có thể config separator theo Markdown, HTML, code, tiếng Việt, v.v.
```

Ví dụ với Markdown:

```markdown
# Chính sách nghỉ phép

## Nghỉ phép năm
...

## Nghỉ bệnh
...
```

Recursive/Markdown-aware splitter có thể ưu tiên cắt theo heading, tốt hơn fixed-size rất nhiều.

## Điểm yếu

Recursive không thật sự hiểu meaning. Nó chỉ hiểu **ký hiệu bề mặt**.

Ví dụ một paragraph dài có 3 ý:

```text
Chính sách hoàn tiền. Chính sách bảo hành. Chính sách đổi trả cho enterprise.
```

Nếu paragraph ngắn hơn `chunk_size`, recursive có thể giữ nguyên cả 3 ý trong một chunk. Embedding sẽ bị loãng vì một vector phải đại diện cho nhiều ý.

Ngược lại, nếu một ý kéo dài qua nhiều paragraph, recursive có thể tách nó ra.

## Khi nào dùng?

Dùng làm **default baseline production** cho text thường.

Cấu hình khởi điểm:

```text
chunk_size: 300–800 tokens
chunk_overlap: 10–20%
separator: paragraph → sentence → word
```

Với tiếng Việt, cần cẩn thận vì tokenization khác tiếng Anh. Nên đo bằng tokenizer thật của embedding model, không chỉ đo character.

---

# 5. Semantic chunking

## Ý tưởng

Semantic chunking không hỏi:

```text
“Đủ 500 tokens chưa?”
```

Nó hỏi:

```text
“Đoạn này còn cùng một ý không?”
```

Cách làm phổ biến:

```text
1. Split document thành sentence/paragraph nhỏ
2. Embed từng sentence/paragraph
3. Tính similarity giữa các đoạn liên tiếp
4. Nếu semantic distance tăng mạnh → đó là boundary
5. Merge các đoạn liên quan thành chunk
```

Ví dụ:

```text
Paragraph 1–3: nói về refund
Paragraph 4–6: nói về warranty
Paragraph 7–9: nói về enterprise SLA
```

Semantic chunking sẽ cố tạo:

```text
Chunk A: refund
Chunk B: warranty
Chunk C: enterprise SLA
```

## Vì sao nó ra đời sau recursive?

Vì recursive chỉ biết hình thức văn bản. Nhưng tài liệu thật không phải lúc nào cũng viết đẹp.

Có document không có heading rõ. Có document một paragraph chứa nhiều ý. Có document heading sai. Có PDF extract ra line break lung tung.

Semantic chunking sinh ra để giải bài toán:

```text
Cắt theo meaning, không chỉ theo dấu xuống dòng.
```

Unstructured có chiến lược `by_similarity`, dùng embedding để nhận diện các element liên tiếp có cùng topic rồi combine thành chunk, nhưng vẫn giới hạn bởi max chunk size. ([Unstructured][4])

Research gần đây cũng đi theo hướng này. Paper **Meta-Chunking** năm 2024 nói RAG thường bỏ qua vai trò của text chunking, rồi đề xuất framework tìm điểm segmentation tốt hơn và giữ global information, thay vì chỉ dựa vào similarity đơn giản. ([arXiv][5])

## Điểm mạnh

Semantic chunking tốt khi document:

```text
- Dài
- Ít heading
- Heading không đáng tin
- Một section có nhiều topic
- Cần chunk coherent theo ý
```

Ví dụ tốt:

```text
policy document
legal text
research paper
manual dài
customer support knowledge base
```

## Điểm yếu

Semantic chunking có 4 điểm yếu lớn.

Thứ nhất, **tốn compute**. Bạn phải embed sentence/paragraph trước khi chunk chính thức.

Thứ hai, **khó debug hơn recursive**. Với recursive, nhìn separator là hiểu. Với semantic, cần giải thích vì sao similarity drop tại vị trí đó.

Thứ ba, **có thể split sai trong tài liệu kỹ thuật**. Hai đoạn có vẻ semantic khác nhau nhưng thực ra phải đi chung.

Ví dụ:

```text
Đoạn 1: định nghĩa metric
Đoạn 2: công thức metric
Đoạn 3: ví dụ tính metric
```

Semantic distance có thể khác, nhưng nếu tách ra thì chunk mất nghĩa.

Thứ tư, **phụ thuộc embedding model**. Embedding yếu domain thì boundary yếu.

## Khi nào dùng?

Dùng khi recursive cho retrieval kém vì chunk bị lẫn nhiều topic hoặc document format không ổn định.

Không nên dùng semantic chunking chỉ vì nghe advanced. Phải test bằng retrieval eval.

---

# 6. Parent-child chunking

## Ý tưởng

Parent-child chunking giải quyết trực tiếp chunk paradox:

```text
Child chunk nhỏ → dùng để embedding/retrieve
Parent chunk lớn → dùng để đưa vào prompt
```

Ví dụ:

```text
Parent chunk: toàn bộ section "Refund Policy" 1,500 tokens

Child chunks:
- refund condition, 200 tokens
- refund timeline, 200 tokens
- enterprise exception, 200 tokens
- non-refundable items, 200 tokens
```

Khi query:

```text
"enterprise refund exception"
```

Retriever search trên child chunks. Nó tìm đúng child:

```text
enterprise exception
```

Nhưng khi đưa context cho LLM, hệ thống lấy parent:

```text
toàn bộ Refund Policy section
```

LangChain `ParentDocumentRetriever` mô tả đúng pattern này: lưu và retrieve small chunks trước, sau đó lookup parent IDs để trả về larger documents/context. ([Tài liệu tham khảo LangChain][6])

## Vì sao nó ra đời sau recursive/semantic?

Vì recursive và semantic vẫn phải chọn một kích thước chunk duy nhất.

Nhưng một kích thước duy nhất không thể tối ưu cả hai việc:

```text
Embedding/retrieval muốn nhỏ
Generation/answer muốn đủ context
```

Parent-child tách hai mục tiêu này ra.

## Điểm mạnh

Rất mạnh cho enterprise RAG.

Đặc biệt tốt khi câu trả lời cần:

```text
- Một chi tiết nhỏ để match query
- Nhưng cần cả section để answer không sai
```

Ví dụ:

```text
Query: "khách enterprise có được refund sau 30 ngày không?"

Child giúp match:
"enterprise customers are covered by separate SLA"

Parent giúp answer:
"standard refund 30 days, except enterprise SLA..."
```

## Điểm yếu

Parent-child có thể **kéo quá nhiều context vào prompt**.

Nếu retrieve 5 child chunks và mỗi child kéo một parent 1,500 tokens, hệ thống có thể tiêu tốn 7,500 tokens rất nhanh.

Nó cũng cần hệ thống lưu metadata tốt:

```text
child_id
parent_id
document_id
section_title
page_number
source_url
```

Nếu metadata sai, retrieve đúng child nhưng lấy nhầm parent là hỏng.

## Khi nào dùng?

Dùng khi xuất hiện lỗi kiểu:

```text
Retriever tìm đúng đoạn nhỏ,
nhưng LLM trả lời thiếu điều kiện xung quanh.
```

Đây là pattern rất đáng học cho RAG doanh nghiệp.

---

# 7. Hierarchical chunking

## Ý tưởng

Hierarchical chunking không chỉ có parent-child 2 tầng. Nó tổ chức document thành nhiều tầng:

```text
Document
└── Section
    └── Subsection
        └── Paragraph
            └── Sentence
```

Hoặc theo token size:

```text
Level 1: 2048 tokens
Level 2: 512 tokens
Level 3: 128 tokens
```

LlamaIndex `HierarchicalNodeParser` chia document thành hierarchy node; tài liệu của họ mô tả top-level node lớn, child node nhỏ hơn, và leaf node thường được index/retrieve bằng vector store trước. ([Developer Documentation][7])

## Khác gì parent-child?

Parent-child là một trường hợp đơn giản của hierarchical.

```text
Parent-child:
2 tầng: parent → child

Hierarchical:
nhiều tầng: document → section → subsection → paragraph → sentence
```

Parent-child thường trả về parent trực tiếp.

Hierarchical có thể thông minh hơn:

```text
- retrieve leaf node
- nếu nhiều leaf cùng thuộc một parent → merge lên parent
- nếu chỉ một leaf liên quan → giữ leaf
- nếu query broad → retrieve section lớn hơn
- nếu query specific → retrieve paragraph/sentence nhỏ hơn
```

LlamaIndex Auto Merging Retriever dùng hierarchy node và có thể “stitch”/merge chunks lại; default hierarchy trong ví dụ của họ là 2048 → 512 → 128 tokens. ([Developer Documentation][7])

## Vì sao nó ra đời?

Vì tài liệu dài không phẳng.

Một policy document, annual report, research paper, hoặc technical manual thường có cấu trúc:

```text
Chapter
Section
Subsection
Clause
Table
Footnote
Appendix
```

Nếu flatten toàn bộ thành list chunk ngang hàng, hệ thống làm mất cấu trúc tri thức.

Hierarchical chunking ra đời để giữ cấu trúc đó.

## Điểm mạnh

Rất tốt cho:

```text
- legal documents
- financial reports
- technical manuals
- research papers
- product documentation
- internal company wiki
- policy/compliance docs
```

Nó giúp answer tốt hơn với cả query hẹp và query rộng.

Ví dụ:

```text
Query hẹp:
"late fee là bao nhiêu?"
→ retrieve clause nhỏ

Query rộng:
"tóm tắt chính sách thanh toán"
→ retrieve section/subsection lớn
```

## Điểm yếu

Hierarchical chunking khó hơn nhiều ở production.

Bạn phải xử lý:

```text
- parser có nhận đúng heading không?
- PDF extract có giữ section không?
- Table/caption có nằm đúng node không?
- Retrieve leaf hay parent?
- Merge bao nhiêu là đủ?
- Có duplicate context không?
```

Nó cũng cần eval tốt hơn, không chỉ xem top-k similarity.

## Vì sao research gần đây tập trung vào hierarchical?

Vì RAG truyền thống thường coi tài liệu là plain text. Nhưng enterprise document có layout, section, bảng, ảnh, footnote. Paper 2025 này nói traditional chunking không xét underlying textual structure nên chunk không đủ semantic meaning; họ dùng hierarchical segmentation và clustering để tạo chunk coherent hơn. ([arXiv][3])

Paper 2026 **MultiDocFusion** còn đi xa hơn cho long industrial documents: detect document regions bằng vision-based parsing, OCR, dựng lại cây hierarchy bằng LLM-based parsing, rồi tạo hierarchical chunks; mục tiêu là giảm information loss trong tài liệu công nghiệp dài/phức tạp. ([arXiv][8])

---

# 8. Late chunking

## Ý tưởng

Chunking truyền thống làm như này:

```text
Document dài
→ cắt thành chunks
→ embed từng chunk riêng
→ lưu vector
```

Late chunking làm ngược lại một phần:

```text
Document dài
→ long-context embedding model encode toàn document
→ sau đó mới pool/token-slice representation thành chunk embeddings
```

Tức là chunk embedding được tạo ra sau khi model đã nhìn context rộng hơn.

Paper **Late Chunking** mô tả phương pháp này: dùng long-context embedding model để embed tất cả token của long text trước, rồi áp dụng chunking sau transformer và trước mean pooling; mục tiêu là chunk embedding vẫn giữ contextual information từ toàn văn bản. ([arXiv][2])

## Vì sao nó ra đời?

Vì chunk embedding truyền thống bị “mù context”.

Ví dụ có đoạn:

```text
He was appointed CEO in 2021.
```

Nếu chunk trước đó nói:

```text
John Smith joined Acme Corp in 2015.
```

Chunk riêng lẻ `He was appointed CEO...` có embedding yếu vì `He` không rõ là ai.

Late chunking cho embedding model nhìn cả đoạn dài trước, nên representation của `He` có thể mang context về `John Smith`.

## Điểm mạnh

Late chunking mạnh khi:

```text
- document có nhiều coreference: he, she, it, this policy, above method
- paragraph phụ thuộc context trước/sau
- chunk nhỏ nhưng cần global context
- long-context embedding model đủ tốt
```

Nó giữ được lợi ích của small chunk retrieval nhưng giảm mất context trong embedding.

## Điểm yếu

Không phải pipeline nào cũng dùng được.

Bạn cần:

```text
- embedding model hỗ trợ long context
- access token-level hidden states hoặc pooling linh hoạt
- pipeline embedding phức tạp hơn
- memory/latency cao hơn
```

Nó cũng không thay thế table-aware/code-aware. Nếu parser đã làm hỏng bảng/code trước khi embed thì late chunking không cứu được.

## Khi nào học?

Học sau khi đã hiểu recursive, semantic, parent-child. Late chunking là advanced embedding-side technique, không phải thứ cần làm đầu tiên cho mọi RAG.

---

# 9. Table-aware chunking

## Ý tưởng

Table-aware chunking hiểu rằng bảng không thể bị cắt như text thường.

Một table chunk tốt cần giữ:

```text
- table title
- caption
- column headers
- row labels
- units
- footnotes
- surrounding paragraph giải thích bảng
```

Ví dụ bảng:

```text
Table 2. Loan Approval Rules

| Segment | Min income | Max DTI |
| A       | 20M        | 40%     |
| B       | 15M        | 35%     |
```

Nếu chỉ retrieve row:

```text
A | 20M | 40%
```

LLM không biết `20M` là gì, `40%` là gì. Table-aware chunk nên convert thành:

```text
Table: Loan Approval Rules
Segment A:
- Min income: 20M
- Max DTI: 40%
```

## Vì sao nó ra đời?

Vì recursive/fixed-size coi bảng là text. Nhưng bảng có ý nghĩa theo **quan hệ hàng-cột**, không phải theo thứ tự câu.

Unstructured có chunking dựa trên document elements/metadata, và strategy `by_title` giữ section boundary; đây là bước quan trọng để không trộn nội dung từ nhiều section khác nhau vào một chunk. ([Unstructured][4])

Nghiên cứu 2026 về **Structure-Aware Chunking for Tabular Data** chỉ ra chunking cho RAG thường thiết kế cho unstructured text, không xét cấu trúc bảng; paper đề xuất row-level hierarchical representation và token-constrained splitting aligned với structural boundaries. ([arXiv][9])

## Điểm mạnh

Cực kỳ quan trọng cho enterprise.

Vì doanh nghiệp có rất nhiều:

```text
- Excel
- CSV
- financial statement
- pricing table
- policy matrix
- comparison table
- risk rule table
- SLA table
```

## Điểm yếu

Khó nhất là parsing.

Nếu PDF parser extract bảng sai, chunking phía sau cũng sai.

Các lỗi hay gặp:

```text
- mất header
- merge cell sai
- row bị đảo
- caption tách khỏi table
- unit nằm ngoài table
- footnote bị mất
```

## Khi nào dùng?

Bất kỳ RAG nào ingest PDF/Excel/report có bảng đều nên có table-aware logic. Không có table-aware thì retrieval nhìn có vẻ đúng nhưng answer rất dễ sai số/cột/đơn vị.

---

# 10. Code-aware chunking

## Ý tưởng

Code-aware chunking không cắt theo paragraph như text. Nó cắt theo đơn vị code:

```text
- file
- class
- function
- method
- import block
- config block
- test case
```

LangChain có code splitter với separator theo ngôn ngữ lập trình; docs liệt kê nhiều language như Python, Java, JS/TS, Go, Rust, C/C++, C#, HTML, Markdown, LaTeX, v.v. ([LangChain Docs][10])

## Vì sao nó ra đời?

Vì code có structure khác text.

Nếu cắt ngang function:

```python
def calculate_score(x):
    ...
```

mà chunk không có signature, import, class context, docstring, thì LLM khó hiểu.

Với repo-level RAG, một function còn phụ thuộc:

```text
- import
- type definition
- config
- caller/callee
- tests
- adjacent files
```

## Điểm mạnh

Tốt cho:

```text
- code search
- coding assistant
- repo Q&A
- bug explanation
- API documentation
- internal SDK assistant
```

## Điểm yếu

Code-aware chunking không đơn giản là “function chunking là tốt nhất”.

Một nghiên cứu 2026 về RAG-based code completion test nhiều chunking strategies như Function, Declaration, Sliding Window và cAST; kết quả cho thấy chunking strategy ảnh hưởng đáng kể tới chất lượng, và function-based chunking thậm chí underperform trong benchmark RepoEval, trong khi cross-file context length lại là yếu tố rất quan trọng. ([arXiv][11])

Điều này có nghĩa là với code RAG, không nên nghĩ máy móc:

```text
function chunking = best
```

Thực tế có thể cần:

```text
function/class chunk
+ sliding window
+ import/context expansion
+ call graph
+ repo metadata
+ reranking
```

---

# 11. So sánh nhanh các loại chunking

| Loại          |    Retrieve precision |                 Giữ context |    Độ phức tạp | Dùng tốt cho                       |
| ------------- | --------------------: | --------------------------: | -------------: | ---------------------------------- |
| Fixed-size    |            Trung bình |                        Thấp |           Thấp | Baseline, text đều                 |
| Recursive     |                   Khá |                         Khá |           Thấp | Generic RAG                        |
| Semantic      | Cao hơn nếu model tốt |                         Khá | Trung bình/cao | Long text, topic boundary không rõ |
| Parent-child  |                   Cao |                         Cao |     Trung bình | Enterprise docs, policy, legal     |
| Hierarchical  |                   Cao |                     Rất cao |            Cao | Reports, manuals, research papers  |
| Late chunking |                   Cao |         Cao trong embedding |            Cao | Long-context embedding pipelines   |
| Table-aware   |          Cao với bảng |                         Cao |            Cao | PDF, Excel, financial docs         |
| Code-aware    |     Cao với repo/code | Phụ thuộc context expansion |            Cao | Coding assistant, code search      |

---

# 12. Thứ tự học hợp lý

Thứ tự học khuyến nghị:

```text
1. Fixed-size
   → hiểu chunk_size, overlap, token budget

2. Recursive
   → hiểu separator, paragraph/sentence boundary
   → đây là default baseline

3. Parent-child
   → hiểu chunk paradox: small for retrieval, large for generation

4. Hierarchical
   → hiểu document tree, multi-level retrieval, auto-merge

5. Semantic
   → hiểu similarity boundary, embedding-based split, topic coherence

6. Table-aware / code-aware
   → hiểu structure-aware RAG cho dữ liệu doanh nghiệp thật

7. Late chunking
   → hiểu embedding-side innovation, long-context embedding
```

Lý do không nên học late chunking quá sớm: nó advanced nhưng không giải quyết được lỗi parser, metadata, table, hierarchy. Trong doanh nghiệp, phần ingest/parse/chunk structure thường giết RAG trước khi late chunking có cơ hội phát huy.

---

# 13. Cách chọn chunking theo bài toán doanh nghiệp

## Chatbot hỏi đáp policy nội bộ

Nên dùng:

```text
Recursive + parent-child
```

Vì policy thường có section rõ, query thường hỏi chi tiết nhưng answer cần điều kiện xung quanh.

## RAG cho PDF tài chính / báo cáo doanh nghiệp

Nên dùng:

```text
Hierarchical + table-aware + parent-child
```

Vì financial report có section, bảng, footnote, page reference.

## RAG cho legal/compliance

Nên dùng:

```text
Hierarchical + parent-child + semantic reranking
```

Vì câu trả lời thường cần clause nhỏ nhưng phải giữ context điều khoản cha.

## RAG cho research paper

Nên dùng:

```text
Hierarchical section-aware
+ table/figure caption-aware
+ parent-child
```

Vì paper có abstract, method, experiment, table, figure, appendix.

## RAG cho codebase

Nên dùng:

```text
Code-aware
+ repo metadata
+ import/caller context expansion
+ hybrid search
```

Không nên chỉ fixed-size cắt code.

## RAG cho transcript meeting/call center

Nên dùng:

```text
Semantic chunking
+ time-window metadata
+ speaker-aware chunking
```

Vì transcript ít heading, topic chuyển liên tục.

---

# 14. Các tham số phải hiểu sâu

## `chunk_size`

Không có số thần thánh.

Nhưng mindset là:

```text
chunk_size nhỏ:
- match query tốt hơn
- dễ thiếu context
- nhiều vector hơn
- retrieve nhiều hơn

chunk_size lớn:
- context đầy hơn
- vector bị loãng
- tốn prompt token
- dễ kéo irrelevant info
```

Starting point thực dụng:

```text
FAQ/simple docs: 200–400 tokens
Policy/docs thường: 400–800 tokens
Research/legal/manual: 600–1200 tokens
Parent chunk: 1000–2500 tokens
Child chunk: 100–400 tokens
```

## `chunk_overlap`

Overlap dùng để tránh mất thông tin ở boundary.

Ví dụ:

```text
chunk_size = 500
overlap = 50–100
```

Nhưng overlap quá cao gây:

```text
- duplicate chunks
- vector DB phình
- retrieve trùng lặp
- LLM đọc context lặp
- tốn tiền
```

Overlap thường nên là **10–20%**, nhưng với parent-child/hierarchical tốt thì overlap có thể giảm.

## `separator`

Separator quyết định “tôn trọng cấu trúc nào”.

Ví dụ:

```text
Markdown: #, ##, ###, paragraph
HTML: h1, h2, h3, p, table
Code: class, function, method
Legal: Article, Section, Clause
PDF: title, heading, page, table, caption
```

Separator sai thì chunk sai.

## `parent_context_size`

Parent quá nhỏ thì không đủ context. Parent quá lớn thì prompt bị bloat.

Mindset:

```text
child = đơn vị match
parent = đơn vị answer
```

Nếu user hỏi một điều khoản, parent nên là section chứa điều khoản đó, không nhất thiết là cả document.

## `max_context_budget`

Đây là giới hạn sống còn.

Ví dụ LLM có 32k context không có nghĩa là nhét 32k retrieved text. Bạn còn cần:

```text
system prompt
user query
chat history
instructions
citations
reasoning space
answer budget
```

Nên RAG tốt thường có bước:

```text
retrieve nhiều
→ rerank
→ deduplicate
→ compress/select
→ đưa context vừa đủ
```

---

# 15. Một cách nhớ rất nhanh

```text
Fixed-size:
"Cắt cho vừa."

Recursive:
"Cắt cho vừa nhưng đừng phá câu/paragraph."

Semantic:
"Cắt theo ý."

Parent-child:
"Tìm bằng mảnh nhỏ, trả lời bằng mảnh lớn."

Hierarchical:
"Giữ cây cấu trúc của tài liệu."

Late chunking:
"Cho embedding model đọc rộng trước, rồi mới tách chunk."

Table-aware:
"Giữ hàng-cột-header-caption."

Code-aware:
"Giữ function/class/file/dependency."
```

---

# 16. Kết luận thực dụng

Với RAG engineer, đừng học chunking như danh sách thuật ngữ. Hãy học theo lỗi mà từng loại sửa:

```text
Fixed-size sửa lỗi document quá dài.
Recursive sửa lỗi cắt nát câu.
Semantic sửa lỗi chunk không cùng ý.
Parent-child sửa lỗi retrieve đúng nhưng answer thiếu context.
Hierarchical sửa lỗi mất cấu trúc tài liệu dài.
Late chunking sửa lỗi embedding chunk bị mù context xung quanh.
Table-aware sửa lỗi bảng mất header/caption/unit.
Code-aware sửa lỗi code bị cắt ngang logic.
```

Nếu làm project RAG doanh nghiệp thật, baseline hợp lý nhất thường là:

```text
Recursive / Markdown-aware chunking
+ metadata tốt
+ parent-child retrieval
+ reranker
+ table-aware riêng cho bảng
+ eval theo query set thật
```

Sau đó mới nâng cấp semantic, hierarchical hoặc late chunking nếu metric cho thấy baseline đang fail ở đúng loại lỗi đó.

[1]: https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter "Splitting recursively - Text splitter integration guide - Docs by LangChain"
[2]: https://arxiv.org/abs/2409.04701 "Late Chunking: Contextual Chunk Embeddings Using Long-Context Embedding Models"
[3]: https://arxiv.org/abs/2507.09935 "[2507.09935] Enhancing Retrieval Augmented Generation with Hierarchical Text Segmentation Chunking"
[4]: https://docs.unstructured.io/api-reference/legacy-api/partition/chunking "Chunking strategies - Unstructured"
[5]: https://arxiv.org/abs/2410.12788 "[2410.12788] Meta-Chunking: Learning Text Segmentation and Semantic Completion via Logical Perception"
[6]: https://reference.langchain.com/python/langchain-classic/retrievers/parent_document_retriever/ParentDocumentRetriever "ParentDocumentRetriever | langchain_classic"
[7]: https://developers.llamaindex.ai/python/framework/integrations/retrievers/auto_merging_retriever/ "Auto Merging Retriever
 \| Developer Documentation"
[8]: https://arxiv.org/abs/2604.12352 "MultiDocFusion: Hierarchical and Multimodal Chunking Pipeline for Enhanced RAG on Long Industrial Documents"
[9]: https://arxiv.org/abs/2605.00318 "Structure-Aware Chunking for Tabular Data in Retrieval-Augmented Generation"
[10]: https://docs.langchain.com/oss/python/integrations/splitters/code_splitter "Splitting code text splitter integration guide - Docs by LangChain"
[11]: https://arxiv.org/abs/2605.04763 "How Does Chunking Affect Retrieval-Augmented Code Completion? A Controlled Empirical Study"
