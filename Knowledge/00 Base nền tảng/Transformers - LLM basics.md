Đây là ghi chú **kiến thức nền Transformer/LLM**, không phải checklist phỏng vấn. Trong roadmap, phần này nằm ngay dưới nền của **RAG, agentic AI, LLM serving, GPU inference, multimodal/VLM**. Roadmap cũng nhấn mạnh các skill như RAG evaluation, reranking, citations, observability, streaming, KV cache, batching, TTFT, TPOT, tokens/sec. Transformer/LLM basics cần được học để hiểu **LLM sinh chữ như thế nào, vì sao context có giới hạn, vì sao hallucinate, và vì sao inference tốn GPU/memory**. ([Khoalearningcode](https://khoalearningcode.github.io/AI_Engineer_Roadmap/ "AI Systems Engineer Roadmap"))

---

# Transformer / LLM Basics — Knowledge cần học kỹ

## 1. Transformer là gì?

Transformer là kiến trúc neural network dùng **attention** làm cơ chế chính để xử lý sequence. Paper gốc _Attention Is All You Need_ giới thiệu Transformer như một kiến trúc dựa hoàn toàn trên attention, không cần recurrence hay convolution cho sequence transduction. ([arXiv](https://arxiv.org/abs/1706.03762 "Attention Is All You Need"))

Trước Transformer, các mô hình sequence thường dùng RNN/LSTM/GRU. Vấn đề của RNN là xử lý tuần tự:

```text
token 1 → token 2 → token 3 → token 4
```

Transformer cho phép xử lý các token song song hơn trong training, đồng thời dùng attention để mỗi token “nhìn” các token khác trong sequence.

Cách hiểu đơn giản:

```text
Input text
→ tokenization
→ token embeddings
→ positional information
→ transformer layers
→ output hidden states
→ predict next token / class / embedding / answer
```

Transformer không chỉ dùng cho text. Nó còn là nền của:

```text
LLM
BERT-style encoder
GPT-style decoder
Vision Transformer
CLIP / VLM
multimodal models
speech models
time-series models
```

---

## 2. Tokenization

LLM không đọc trực tiếp chữ như con người. Nó đọc **token**.

Ví dụ câu:

```text
I love machine learning.
```

Có thể được tách thành:

```text
["I", " love", " machine", " learning", "."]
```

Hoặc với tiếng Việt / từ lạ / code:

```text
"không hiểu hallucination"
→ ["kh", "ông", " hiểu", " halluc", "ination"]
```

Tùy tokenizer, cùng một câu có thể bị tách khác nhau.

Hugging Face docs ghi các thuật toán subword tokenization phổ biến trong Transformers gồm **BPE**, **WordPiece**, và **Unigram**; chúng chia text thành đơn vị nằm giữa word-level và character-level để giữ vocabulary gọn nhưng vẫn xử lý được từ hiếm. ([Hugging Face](https://huggingface.co/docs/transformers/en/tokenizer_summary "Tokenization algorithms"))

---

## 2.1 Token là gì?

Token có thể là:

```text
một từ
một phần của từ
một ký tự
một dấu câu
một khoảng trắng
một token đặc biệt
```

Ví dụ token đặc biệt:

```text
<BOS>  beginning of sequence
<EOS>  end of sequence
<PAD>  padding
<UNK>  unknown token
<SEP>  separator
<CLS>  classification token, thường gặp trong BERT-style models
```

Cần hiểu:

> Token không giống word. Một word có thể thành nhiều token, nhất là với tiếng Việt, code, tên riêng, từ hiếm, emoji, ký hiệu toán học.

---

## 2.2 Vocabulary

Vocabulary là tập token mà tokenizer biết.

Ví dụ:

```text
vocab size = 32k
vocab size = 50k
vocab size = 100k
```

Vocabulary quá nhỏ:

```text
nhiều từ bị tách vụn
sequence dài hơn
tốn context hơn
```

Vocabulary quá lớn:

```text
embedding matrix lớn hơn
model tốn memory hơn
```

Cần hiểu:

```text
Text → tokenizer → token ids
token ids → embedding layer → vectors
```

Ví dụ:

```text
"hello" → [15339]
"machine learning" → [18712, 6975]
```

Model thật sự nhận vào **token IDs**, không phải string.

---

## 2.3 Tokenization ảnh hưởng gì trong LLM/RAG?

Tokenization ảnh hưởng trực tiếp đến:

```text
context length
cost
latency
chunking
truncation
generation length
prompt design
multilingual performance
```

Ví dụ cùng một ý, nếu viết dài hơn thì tốn token hơn:

```text
"Summarize this document."
```

ít token hơn:

```text
"Please carefully read the following document and provide a detailed summary..."
```

Trong RAG, nếu chunk quá dài:

```text
retrieved context tốn nhiều token
ít chỗ cho conversation history/output
LLM có thể bị nhiễu
```

Nếu chunk quá ngắn:

```text
mất context
retrieval có thể lấy đoạn không đủ ý
answer thiếu grounding
```

---

# 3. Embedding trong Transformer/LLM

Sau tokenization, token ID được biến thành vector.

```text
token id → embedding vector
```

Ví dụ:

```text
"cat" → vector 768 chiều
"dog" → vector 768 chiều
```

Embedding layer là một bảng tra cứu:

```text
vocab_size × hidden_dim
```

Nếu:

```text
vocab_size = 50,000
hidden_dim = 4096
```

thì embedding matrix có kích thước:

```text
50,000 × 4096
```

Embedding trong Transformer gồm hai loại quan trọng:

|Loại|Ý nghĩa|
|---|---|
|**Token embedding**|Biểu diễn token là gì|
|**Position information**|Biểu diễn token nằm ở vị trí nào|

Vì attention tự nó không biết thứ tự, model cần positional information.

---

# 4. Positional Encoding / Position Embedding

Câu:

```text
"dog bites man"
```

khác:

```text
"man bites dog"
```

Dù có cùng token, thứ tự khác làm nghĩa khác.

Transformer cần thông tin vị trí để biết token nào đứng trước/sau.

Các kiểu positional information:

|Kiểu|Ý nghĩa|
|---|---|
|**Absolute position embedding**|Mỗi vị trí có vector riêng|
|**Sinusoidal positional encoding**|Dùng hàm sin/cos như Transformer gốc|
|**Relative position**|Quan tâm khoảng cách giữa token|
|**RoPE**|Rotary Positional Embedding, phổ biến trong nhiều LLM hiện đại|
|**ALiBi**|Attention bias theo khoảng cách|

Cần hiểu ở mức nền:

```text
Token embedding nói token là gì.
Position embedding nói token ở đâu.
Transformer cần cả hai để hiểu sequence.
```

---

# 5. Attention

Attention là lõi của Transformer.

Ý tưởng trực giác:

> Khi xử lý một token, model quyết định nên chú ý tới token nào khác trong context.

Ví dụ câu:

```text
The animal didn't cross the street because it was tired.
```

Khi xử lý `"it"`, model cần chú ý tới `"animal"` để hiểu “it” là con vật.

Ví dụ khác:

```text
Khoa put the book on the table because it was flat.
```

`"it"` có thể liên quan tới `"table"` hơn là `"book"`.

Attention giúp mỗi token lấy thông tin từ token khác.

---

## 5.1 Query, Key, Value

Attention dùng ba vector chính:

```text
Query = token hiện tại đang tìm thông tin gì
Key   = token khác đang chứa loại thông tin gì
Value = nội dung thật sự được lấy về
```

Ví dụ:

```text
Token hiện tại: "it"
Query: cần biết "it" đang refer tới phần gì
Key của "animal": có thông tin entity
Value của "animal": nội dung về animal
```

Công thức scaled dot-product attention trong Transformer gốc:

```text
Attention(Q, K, V) = softmax(QKᵀ / sqrt(d_k)) V
```

Paper Transformer mô tả kiến trúc dựa trên stacked self-attention và feed-forward layers trong encoder/decoder. ([papers.neurips.cc](https://papers.neurips.cc/paper/7181-attention-is-all-you-need.pdf "Attention is All you Need"))

Cách đọc công thức:

```text
Q so với K → ra attention score
softmax → biến score thành trọng số
trọng số nhân với V → lấy thông tin cần thiết
```

---

## 5.2 Self-attention

Self-attention nghĩa là các token trong cùng một sequence chú ý lẫn nhau.

Ví dụ:

```text
"The cat sat on the mat"
```

Token `"sat"` có thể chú ý tới:

```text
"The"
"cat"
"on"
"mat"
```

Self-attention giúp model học quan hệ:

```text
subject - verb
object - attribute
pronoun - referent
cause - effect
question - evidence
```

---

## 5.3 Causal attention

LLM kiểu GPT sinh text từ trái sang phải.

Khi predict token tiếp theo, nó không được nhìn token tương lai.

Ví dụ:

```text
Input: "The capital of France is"
Target: "Paris"
```

Khi model đang ở token `"France"`, nó không được nhìn trước `"Paris"`.

Causal mask đảm bảo:

```text
token hiện tại chỉ nhìn token hiện tại và token trước đó
không nhìn token tương lai
```

Đây là lý do decoder-only LLM phù hợp với next-token generation.

---

## 5.4 Multi-head attention

Một attention head có thể học một loại quan hệ.

Ví dụ:

```text
head 1: subject-verb
head 2: entity-reference
head 3: punctuation/format
head 4: long-range dependency
```

Multi-head attention cho phép model nhìn thông tin theo nhiều “góc” khác nhau cùng lúc.

Cách hiểu:

```text
single-head = một cách chú ý
multi-head = nhiều cách chú ý song song
```

---

# 6. Transformer layer gồm gì?

Một Transformer block thường có:

```text
self-attention
feed-forward network
residual connection
layer normalization
dropout
```

Flow đơn giản:

```text
input hidden states
→ self-attention
→ residual + layer norm
→ feed-forward network
→ residual + layer norm
→ output hidden states
```

## 6.1 Feed-forward network

Sau attention, mỗi token đi qua MLP/feed-forward network.

Attention trộn thông tin giữa token.

Feed-forward xử lý thông tin bên trong từng token.

```text
attention = token nói chuyện với nhau
feed-forward = mỗi token tự biến đổi representation của nó
```

## 6.2 Residual connection

Residual giúp gradient đi qua mạng sâu dễ hơn.

Thay vì chỉ:

```text
x → layer → y
```

Residual làm:

```text
x → layer → y
output = x + y
```

Ý nghĩa:

```text
layer mới học phần cần thêm vào
không phải học lại toàn bộ từ đầu
```

## 6.3 LayerNorm

LayerNorm giúp ổn định activation, làm training ổn định hơn.

---

# 7. Các loại Transformer model

## 7.1 Encoder-only

Ví dụ:

```text
BERT
RoBERTa
DistilBERT
```

Đặc điểm:

```text
nhìn được toàn bộ input hai chiều
phù hợp hiểu văn bản
không phải mô hình sinh text tự nhiên kiểu GPT
```

Dùng cho:

```text
classification
token classification
embedding
reranking
named entity recognition
sentence understanding
```

## 7.2 Decoder-only

Ví dụ:

```text
GPT-style LLM
LLaMA-style LLM
Mistral-style LLM
Qwen-style LLM
```

Đặc điểm:

```text
causal attention
predict next token
sinh text tốt
```

Dùng cho:

```text
chatbot
completion
reasoning
agent
code generation
RAG answer generation
```

Roadmap có phần LLM Generation với “decoder-only transformers, instruction tuning, LoRA/QLoRA, alignment concept, evaluation”, nên đây là nhóm phải nắm rõ nhất. ([Khoalearningcode](https://khoalearningcode.github.io/AI_Engineer_Roadmap/ "AI Systems Engineer Roadmap"))

## 7.3 Encoder-decoder

Ví dụ:

```text
T5
BART
original Transformer translation model
```

Đặc điểm:

```text
encoder đọc input
decoder sinh output
```

Dùng cho:

```text
translation
summarization
text-to-text tasks
structured generation
```

---

# 8. LLM học như thế nào?

LLM thường học bằng **next-token prediction**.

Ví dụ:

```text
Input: "The capital of France is"
Target: "Paris"
```

Model học phân phối xác suất:

```text
Paris: 0.80
London: 0.05
Berlin: 0.03
...
```

Training objective thường là cross-entropy loss trên token đúng.

Cách hiểu:

```text
LLM không lưu tri thức như database.
LLM học pattern xác suất từ dữ liệu text.
```

Nó học:

```text
ngữ pháp
facts xuất hiện trong dữ liệu
style viết
reasoning patterns
code patterns
dialogue patterns
format
```

Nhưng vì nó tối ưu next-token probability, output của nó là **chuỗi token có xác suất hợp lý**, không tự động đảm bảo đúng sự thật.

---

# 9. Context length

Context length là số token tối đa model có thể xử lý trong một lần gọi.

Ví dụ:

```text
4k tokens
8k tokens
32k tokens
128k tokens
1M tokens
```

Context gồm tất cả:

```text
system prompt
developer instruction
chat history
user message
retrieved documents
tool outputs
examples
format instruction
output tokens
```

Cần hiểu:

```text
context window không chỉ dành cho input.
Output cũng chiếm token budget.
```

Ví dụ nếu model max context 8192 tokens:

```text
prompt + history + RAG context = 7000 tokens
max output còn khoảng 1192 tokens
```

---

## 9.1 Context dài không đồng nghĩa hiểu tốt hơn

Long context giúp nhét nhiều thông tin hơn, nhưng không đảm bảo model dùng tốt toàn bộ thông tin. Nghiên cứu _Lost in the Middle_ cho thấy hiệu năng có thể giảm khi thông tin cần thiết nằm ở giữa context; nhiều model dùng thông tin ở đầu hoặc cuối context tốt hơn phần giữa. ([arXiv](https://arxiv.org/abs/2307.03172 "Lost in the Middle: How Language Models Use Long Contexts"))

Ý nghĩa thực tế cho RAG:

```text
Không nên nhồi càng nhiều document càng tốt.
Cần retrieve đúng, rerank tốt, chunk sạch, đặt evidence quan trọng ở vị trí hợp lý.
```

Context dài mà nhiễu:

```text
LLM dễ lạc trọng tâm
latency tăng
cost tăng
KV cache memory tăng
hallucination vẫn có thể xảy ra
```

---

## 9.2 Context length và RAG

Trong RAG, context length quyết định:

```text
top_k lấy được bao nhiêu chunk
mỗi chunk dài bao nhiêu
có giữ conversation history không
có đủ chỗ cho answer không
```

Ví dụ:

```text
top_k = 20
chunk_size = 800 tokens
→ riêng retrieved context đã khoảng 16k tokens
```

Nếu model chỉ có 8k context thì không thể nhét hết.

Do đó phải học:

```text
chunk size
chunk overlap
top_k
reranking
context compression
summary memory
citation-aware context
```

---

# 10. KV Cache

Khi LLM generate token, nếu mỗi bước phải tính lại attention cho toàn bộ token cũ thì rất chậm.

KV cache lưu lại Key/Value của các token trước đó để tái sử dụng khi sinh token tiếp theo.

Cách hiểu:

```text
Token cũ đã tính K/V rồi
→ lưu lại
→ token mới chỉ cần attention vào cache cũ
```

KV cache giúp inference nhanh hơn nhưng tốn GPU memory.

vLLM documentation mô tả PagedAttention tương thích với paged KV caches, trong đó key/value cache được lưu theo các block riêng. ([vLLM](https://docs.vllm.ai/en/latest/design/paged_attention/ "Paged Attention - vLLM Documentation"))

Ý nghĩa với AI Engineer:

```text
context càng dài → KV cache càng lớn
concurrency càng cao → nhiều request giữ KV cache cùng lúc
GPU memory càng dễ đầy
```

Đây là lý do roadmap đưa KV cache, batching, TTFT, TPOT, tokens/sec vào phần LLM serving/GPU inference. ([Khoalearningcode](https://khoalearningcode.github.io/AI_Engineer_Roadmap/ "AI Systems Engineer Roadmap"))

---

# 11. Decoding

Decoding là cách model chọn token tiếp theo từ phân phối xác suất.

LLM không “viết câu” một lần. Nó sinh từng token:

```text
token 1 → token 2 → token 3 → ...
```

Ở mỗi bước:

```text
model output logits
→ softmax thành probability
→ decoding strategy chọn token
```

Hugging Face có tài liệu/guide giải thích các decoding strategy phổ biến như greedy search, beam search, sampling, top-k, top-p. ([Hugging Face](https://huggingface.co/blog/how-to-generate "How to generate text: using different decoding methods for ..."))

---

## 11.1 Logits và probability

Model output ban đầu là **logits**.

```text
Paris: 8.2
London: 3.1
Berlin: 2.7
```

Softmax biến logits thành probability:

```text
Paris: 0.92
London: 0.04
Berlin: 0.03
```

Decoding chọn token từ distribution này.

---

## 11.2 Greedy decoding

Greedy chọn token xác suất cao nhất ở mỗi bước.

```text
mỗi bước chọn token có probability cao nhất
```

Ưu điểm:

```text
ổn định
dễ đoán
ít randomness
```

Nhược điểm:

```text
dễ lặp
có thể khô cứng
không luôn tạo chuỗi tốt nhất toàn cục
```

Dùng khi cần:

```text
output deterministic
classification-like generation
format nghiêm
data extraction
```

---

## 11.3 Beam search

Beam search giữ nhiều candidate sequence cùng lúc.

Ví dụ beam size = 4:

```text
giữ 4 chuỗi có score tốt nhất
mở rộng từng bước
chọn chuỗi cuối tốt nhất
```

Ưu điểm:

```text
tốt cho translation/summarization kiểu cổ điển
ít random hơn sampling
```

Nhược điểm với chatbot/creative generation:

```text
có thể tạo câu chung chung
tốn compute hơn
không luôn tốt cho open-ended dialogue
```

---

## 11.4 Sampling

Sampling chọn token theo xác suất, không nhất thiết chọn token cao nhất.

Ví dụ:

```text
Paris: 0.70
London: 0.10
Rome: 0.08
```

Model có thể chọn `"Paris"` thường xuyên, nhưng đôi khi chọn token khác nếu sampling.

Sampling tạo diversity.

---

## 11.5 Temperature

Temperature điều chỉnh độ “ngẫu nhiên” của distribution.

Công thức tư duy:

```text
probability = softmax(logits / temperature)
```

Temperature thấp:

```text
distribution sắc hơn
token top cao càng áp đảo
output ổn định hơn
ít sáng tạo hơn
```

Temperature cao:

```text
distribution phẳng hơn
nhiều token có cơ hội được chọn
output đa dạng hơn
dễ sai/hallucinate hơn
```

Cách hiểu:

|Temperature|Hiệu ứng|
|---|---|
|0 hoặc gần 0|Gần deterministic|
|0.2–0.5|Ổn định, phù hợp factual/extraction|
|0.7–1.0|Cân bằng|
|>1.0|Sáng tạo hơn nhưng dễ loạn hơn|

Với RAG/factual QA:

```text
temperature thấp thường tốt hơn
```

Với brainstorming/creative writing:

```text
temperature cao có thể hữu ích hơn
```

---

## 11.6 Top-k sampling

Top-k chỉ cho model chọn trong k token có xác suất cao nhất.

Ví dụ `top_k = 5`:

```text
chỉ 5 token tốt nhất được phép sampling
token còn lại bị loại
```

Tác dụng:

```text
giảm token quá kỳ lạ
giữ diversity vừa phải
```

---

## 11.7 Top-p / nucleus sampling

Top-p chọn tập token nhỏ nhất có tổng xác suất ≥ p.

Ví dụ `top_p = 0.9`:

```text
chọn nhóm token có tổng probability khoảng 90%
sampling trong nhóm đó
```

Khác top-k:

```text
top-k luôn lấy đúng k token
top-p lấy số token linh hoạt tùy distribution
```

Nếu distribution rất chắc:

```text
top-p có thể chỉ lấy vài token
```

Nếu distribution mơ hồ:

```text
top-p có thể lấy nhiều token hơn
```

---

## 11.8 Repetition penalty

Repetition penalty giảm khả năng model lặp lại token/cụm từ đã sinh.

Dùng khi model hay sinh:

```text
"the the the..."
"Based on the context, based on the context..."
```

Nhưng đặt quá mạnh có thể làm output mất tự nhiên.

---

## 11.9 Max tokens / stop sequence

`max_tokens` giới hạn số token output.

Stop sequence dừng generation khi gặp chuỗi cụ thể.

Ví dụ:

```text
stop = ["\n\nUser:", "</answer>"]
```

Dùng để:

```text
kiểm soát format
tránh model sinh quá dài
dừng đúng chỗ trong agent/tool call
```

---

# 12. Prompt, instruction, system message

LLM nhận vào một chuỗi context gồm nhiều phần.

Thông thường có:

```text
system instruction
developer instruction
user message
conversation history
retrieved context
tool outputs
format constraints
```

Cần hiểu:

```text
LLM không có ý định riêng.
Nó condition output dựa trên toàn bộ context.
```

Prompt tốt là prompt giúp model hiểu:

```text
nhiệm vụ là gì
dữ liệu nào được phép dùng
format output ra sao
khi không biết thì làm gì
ràng buộc an toàn/chính xác là gì
```

Trong RAG, prompt nên phân biệt rõ:

```text
question
retrieved context
instruction
citation requirement
unknown behavior
```

Ví dụ pattern:

```text
Use only the provided context.
If the answer is not in the context, say you do not know.
Cite the source chunk for each factual claim.
```

Nhưng cần nhớ:

> Prompt chỉ giảm rủi ro, không đảm bảo tuyệt đối model không hallucinate.

---

# 13. Hallucination

Hallucination là khi model tạo ra nội dung nghe hợp lý nhưng sai hoặc không được support bởi context. OpenAI mô tả hallucinations là các phát biểu plausible nhưng false do language model tạo ra. ([OpenAI](https://openai.com/index/why-language-models-hallucinate/ "Why language models hallucinate"))

Ví dụ:

```text
Model bịa tên paper.
Model bịa citation.
Model nói một API có tham số không tồn tại.
Model trả lời chắc chắn dù context không có.
Model suy ra quá mức từ retrieved documents.
```

---

## 13.1 Vì sao hallucination xảy ra?

Các nguyên nhân chính:

### 1. Next-token prediction không phải truth verification

LLM học sinh token tiếp theo có xác suất cao, không phải kiểm tra sự thật như database.

```text
plausible text ≠ true fact
```

### 2. Training data không hoàn hảo

Dữ liệu train có thể:

```text
thiếu thông tin
cũ
mâu thuẫn
sai
nhiễu
```

### 3. Model không biết rõ khi nào nó không biết

Model có thể tiếp tục generate vì objective thường khuyến khích trả lời hơn là dừng.

OpenAI research về hallucination nhấn mạnh một nguyên nhân là các hệ thống đánh giá/training thường reward câu trả lời đúng nhưng cũng có thể vô tình khuyến khích guessing thay vì abstaining khi không chắc. ([OpenAI](https://openai.com/index/why-language-models-hallucinate/ "Why language models hallucinate"))

### 4. Context thiếu hoặc nhiễu

Trong RAG:

```text
retrieved context sai
context không chứa answer
context quá dài và nhiễu
citation không match claim
```

### 5. Decoding quá random

Temperature cao, top-p rộng có thể làm output đa dạng hơn nhưng cũng tăng rủi ro sinh thông tin không chắc.

---

## 13.2 Các loại hallucination

|Loại|Ví dụ|
|---|---|
|**Factual hallucination**|Bịa sự kiện, số liệu, tên người|
|**Citation hallucination**|Bịa nguồn, bịa paper, citation không support|
|**Instruction hallucination**|Làm sai instruction nhưng vẫn nói đã làm|
|**Tool hallucination**|Bịa tool/API result|
|**Code hallucination**|Bịa library/function/argument|
|**RAG hallucination**|Context không có nhưng answer vẫn khẳng định|
|**Visual hallucination**|VLM nói có object/chi tiết không có trong ảnh|

---

## 13.3 Giảm hallucination bằng gì?

Không có cách triệt tiêu hoàn toàn, nhưng có thể giảm bằng:

```text
RAG với evidence đúng
retrieval evaluation
reranking
citation checking
lower temperature
clear abstention instruction
tool verification
structured output
fact-checking step
source-grounded answer
human review cho task quan trọng
```

Trong RAG, cần tách:

```text
retrieval hallucination: context sai/thiếu
generation hallucination: context có nhưng model vẫn bịa
citation hallucination: citation không support claim
```

---

# 14. Instruction tuning

Base LLM chỉ học next-token prediction từ text lớn.

Instruction-tuned LLM được fine-tune để làm theo instruction.

Ví dụ base model có thể chỉ complete text:

```text
User: Explain attention
Model: attention is...
```

Instruction model hiểu format hội thoại hơn:

```text
User asks → assistant answers
```

Instruction tuning giúp model:

```text
theo lệnh tốt hơn
trả lời dạng chat tốt hơn
biết format output hơn
ít lan man hơn
```

Nhưng instruction tuning không tự biến model thành nguồn sự thật tuyệt đối.

---

# 15. Alignment / RLHF / preference tuning

Alignment là nhóm kỹ thuật làm model output phù hợp hơn với mong muốn con người.

Các hướng thường gặp:

```text
SFT - supervised fine-tuning
RLHF - reinforcement learning from human feedback
DPO - direct preference optimization
RLAIF - feedback từ AI
constitutional/self-critique style methods
```

Cần hiểu ở mức nền:

```text
pretraining học language patterns
instruction tuning học làm theo lệnh
preference tuning học chọn response được ưa thích hơn
```

Nhưng alignment có trade-off:

```text
model có thể từ chối nhiều hơn
model có thể nói an toàn hơn nhưng ít trực tiếp hơn
model vẫn có thể hallucinate
```

---

# 16. LLM memory không giống database

Cần phân biệt:

|Thứ|Ý nghĩa|
|---|---|
|**Weights**|Tri thức/pattern được nén trong model|
|**Context**|Thông tin đưa vào trong request hiện tại|
|**External memory**|DB/vector DB/file/tool mà model có thể truy cập|
|**Conversation history**|Lịch sử chat được nhét vào context|
|**Long-term memory system**|Hệ thống ngoài model lưu preference/facts|

LLM không “tra database nội bộ” theo cách chính xác. Nó generate dựa trên weights + context.

Vì vậy với enterprise/private knowledge:

```text
Không nên mong model tự biết tài liệu nội bộ.
Cần RAG hoặc tool/database.
```

---

# 17. Fine-tuning vs RAG vs Prompting

Ba thứ này dễ bị nhầm.

|Cách|Thay đổi gì|Dùng khi|
|---|---|---|
|**Prompting**|Không đổi weights|Điều khiển behavior nhẹ|
|**RAG**|Không đổi weights, thêm external context|Cần knowledge riêng/cập nhật/citation|
|**Fine-tuning**|Update weights hoặc adapter|Cần model học style/task/domain pattern|
|**LoRA/QLoRA**|Fine-tune nhẹ bằng adapter/quantization|Tiết kiệm compute/memory|
|**Tool use**|Model gọi hệ thống ngoài|Cần tính toán, search, API, DB|

Cách nhớ:

```text
Prompting = chỉ dẫn.
RAG = đưa tài liệu đúng vào context.
Fine-tuning = dạy model thói quen/pattern mới.
Tool use = cho model khả năng hành động/tra cứu.
```

Không nên fine-tune chỉ để “nhét tài liệu” nếu tài liệu thay đổi thường xuyên. RAG phù hợp hơn.

---

# 18. LLM output không deterministic hoàn toàn

Nếu dùng sampling, cùng prompt có thể ra nhiều output khác nhau.

Nguyên nhân:

```text
temperature
top-p/top-k
random seed
server batching
model serving implementation
floating point differences
tool results/context thay đổi
```

Nếu cần output ổn định:

```text
temperature thấp
top_p thấp hoặc greedy
schema rõ
validation parser
retry có kiểm soát
unit/eval test
```

---

# 19. LLM serving basics liên quan kiến thức nền

Phần này không phải backend, mà là hiểu vì sao LLM chạy tốn tài nguyên.

Cần học:

|Khái niệm|Ý nghĩa|
|---|---|
|**Prefill**|Xử lý toàn bộ input prompt ban đầu|
|**Decode**|Sinh từng token output|
|**TTFT**|Time to first token|
|**TPOT**|Time per output token|
|**Throughput**|Tokens/sec hoặc requests/sec|
|**Batching**|Gộp request để tăng throughput|
|**KV cache**|Cache attention K/V của token cũ|
|**Quantization**|Giảm precision weights để tiết kiệm memory|
|**Context length**|Ảnh hưởng memory và latency|
|**Concurrency**|Số request đồng thời|

Roadmap yêu cầu stage LLM serving đo TTFT, TPOT, tokens/sec, concurrency, GPU memory report; vì vậy các khái niệm này phải nằm trong docs nền. ([Khoalearningcode](https://khoalearningcode.github.io/AI_Engineer_Roadmap/ "AI Systems Engineer Roadmap"))

---

# 20. Transformer/LLM và RAG liên quan thế nào?

RAG dùng LLM làm generator, nhưng toàn pipeline liên quan nhiều phần:

```text
query
→ tokenizer
→ embedding model
→ vector search
→ reranker
→ context construction
→ LLM generation
→ decoding
→ citation
→ evaluation
```

Các lỗi có thể nằm ở:

```text
tokenization làm query/chunk dài bất thường
embedding retrieve sai
context vượt length
lost-in-the-middle
temperature quá cao
prompt thiếu ràng buộc
LLM hallucinate
citation không support answer
```

Vì vậy học Transformer/LLM không chỉ để biết model bên trong, mà để debug RAG/agent/serving.

---

# 21. Kiến thức cần học theo thứ tự

## Phase 1 — Token và input representation

```text
tokenization
token IDs
vocabulary
special tokens
token embedding
position embedding
context window
```

Mục tiêu:

```text
Hiểu text đi vào model như thế nào.
```

---

## Phase 2 — Transformer core

```text
self-attention
Q/K/V
scaled dot-product attention
multi-head attention
causal mask
feed-forward layer
residual connection
layer norm
encoder-only / decoder-only / encoder-decoder
```

Mục tiêu:

```text
Hiểu vì sao Transformer xử lý sequence tốt và vì sao GPT sinh token trái sang phải.
```

---

## Phase 3 — LLM generation

```text
next-token prediction
logits
softmax
greedy decoding
beam search
sampling
temperature
top-k
top-p
repetition penalty
max tokens
stop sequences
```

Mục tiêu:

```text
Hiểu vì sao cùng một prompt có thể ra output khác nhau và cách chỉnh generation behavior.
```

---

## Phase 4 — Context và serving

```text
context length
prompt token budget
RAG context
KV cache
prefill
decode
TTFT
TPOT
tokens/sec
batching
GPU memory
quantization
```

Mục tiêu:

```text
Hiểu vì sao LLM inference chậm/tốn memory và cách context ảnh hưởng latency/cost.
```

---

## Phase 5 — Hallucination và grounding

```text
hallucination
factuality
groundedness
faithfulness
citation accuracy
abstention
tool verification
RAG failure modes
```

Mục tiêu:

```text
Hiểu vì sao LLM sai và cách giảm sai bằng retrieval, verification, evaluation.
```

---

# Bản gom ngắn để đưa vào bảng roadmap

|Nhóm base|Cần học gì|Mức cần đạt|
|---|---|---|
|**Transformer/LLM basics**|tokenization, token IDs, vocabulary, embeddings, positional encoding, self-attention, Q/K/V, multi-head attention, causal mask, encoder/decoder architecture, context length, KV cache, next-token prediction, logits/softmax, decoding, temperature, top-k/top-p, prompt/context construction, hallucination, grounding, instruction tuning, RAG vs fine-tuning|Hiểu được LLM theo chuỗi **text → tokens → embeddings → attention layers → logits → decoding → generated output**; biết context ảnh hưởng chất lượng/latency ra sao, vì sao hallucination xảy ra, và cách giảm lỗi bằng RAG, prompt constraint, decoding control, citation/evidence verification|

---

# Cấu trúc ghi chú

```text
Transformer / LLM Basics

1. Transformer overview
2. Tokenization
   - token
   - token IDs
   - vocabulary
   - BPE / WordPiece / Unigram
   - special tokens
3. Embedding and positional information
   - token embedding
   - position embedding
   - RoPE / relative position concept
4. Attention
   - Q, K, V
   - scaled dot-product attention
   - self-attention
   - causal attention
   - multi-head attention
5. Transformer block
   - attention
   - feed-forward network
   - residual connection
   - layer normalization
6. Transformer model types
   - encoder-only
   - decoder-only
   - encoder-decoder
7. LLM training objective
   - next-token prediction
   - cross-entropy
8. Context length
   - token budget
   - RAG context
   - truncation
   - lost-in-the-middle
9. KV cache and inference
   - prefill
   - decode
   - memory growth
10. Decoding
   - logits
   - softmax
   - greedy
   - beam search
   - sampling
   - temperature
   - top-k
   - top-p
   - repetition penalty
   - max tokens
   - stop sequence
11. Prompt and instruction
   - system/user/tool context
   - format constraint
   - grounded answering
12. Hallucination
   - factual hallucination
   - citation hallucination
   - tool hallucination
   - RAG hallucination
   - causes and mitigation
13. Instruction tuning and alignment
14. Fine-tuning vs RAG vs prompting
15. LLM serving concepts
   - TTFT
   - TPOT
   - tokens/sec
   - batching
   - concurrency
   - GPU memory
```

Dòng roadmap chuẩn hóa:

> **Transformer/LLM basics:** hiểu tokenization, token IDs, embeddings, positional encoding, attention/QKV, causal mask, Transformer block, decoder-only LLM, context length, KV cache, next-token prediction, decoding strategies, temperature/top-k/top-p, prompt construction, hallucination và grounding.  
> **Mức cần đạt:** đọc được một LLM/RAG pipeline từ input text đến output token; biết vì sao model sinh câu, vì sao bị giới hạn context, vì sao hallucinate, và những biến nào ảnh hưởng chất lượng, latency, cost, memory.