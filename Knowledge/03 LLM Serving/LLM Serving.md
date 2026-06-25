Stage 3 là **module kiến thức riêng** về LLM Serving, không chỉ là vài keyword. Đây là giai đoạn chuyển từ **AI app engineer** sang **AI systems / inference-aware engineer**.

---

# 3. Sau Agent mới học LLM Serving

Sau khi đã hiểu **RAG** và **Agentic AI**, bước tiếp theo là **LLM Serving**. Đây là đoạn biến việc “gọi API từ model provider” thành việc hiểu một LLM được chạy như một **production service**: nhận request, xử lý prompt, sinh token, stream response, đo latency, đo throughput, kiểm soát VRAM, batch nhiều request và tính cost.

Trong roadmap, Stage 3 là **LLM Serving Engineer**. Mục tiêu là serve LLM hiệu quả và đo được **latency, throughput, memory, cost**. Key skills gồm **vLLM, OpenAI-compatible API, streaming, KV cache, batching, TTFT, TPOT, tokens/sec**. Deliverable là một self-hosted LLM serving system có benchmark report, concurrency test, memory report và deployment guide. ([Khoalearningcode][1])

---

## Vì sao học LLM Serving sau Agent?

RAG và Agent tạo ra workload thật cho model.

```text
RAG → prompt dài vì có retrieved context
Agent → nhiều bước, nhiều tool calls, nhiều lần gọi model
```

Nếu chỉ dùng API bên ngoài, chỉ thấy:

```text
prompt → response
```

Nhưng khi học LLM Serving, cần hiểu:

```text
client request
→ API server
→ tokenizer
→ scheduler / batching engine
→ prefill
→ KV cache
→ decode từng token
→ streaming response
→ logs / metrics / benchmark
```

Stage này bắt đầu tạo nền cho tư duy inference/GPU: không chỉ biết gọi model, mà bắt đầu hiểu **model chạy như service trên GPU**.

---

# Mục tiêu của Stage 3

Sau Stage 3, cần hiểu được một request LLM đi qua toàn bộ pipeline:

```text
HTTP request
→ OpenAI-compatible server
→ tokenizer
→ model runtime
→ prefill
→ decode
→ KV cache update
→ streaming chunks
→ final response
→ metrics/logs
```

Và khi hệ thống chậm hoặc lỗi, cần phân tích được:

```text
chậm do prompt dài?
chậm do queue?
chậm do decode?
chậm do batch quá lớn?
OOM do model weights hay KV cache?
context overflow do RAG nhồi quá nhiều chunk?
tokens/sec thấp do GPU utilization kém?
```

---

# Bảng kiến thức chính

| Mảng                     | Nội dung cần nắm                                                            |
| ------------------------ | --------------------------------------------------------------------------- |
| **LLM inference basics** | prefill, decode, autoregressive generation, KV cache                        |
| **Serving API**          | OpenAI-compatible API, chat/completions, streaming response                 |
| **vLLM**                 | continuous batching, PagedAttention, chunked prefill, prefix caching        |
| **Benchmark**            | TTFT, TPOT/ITL, tokens/sec, end-to-end latency, concurrency                 |
| **Memory**               | VRAM usage, model weights, KV cache, context length, batch size             |
| **Cost thinking**        | cost/request, cost/token, latency-throughput-quality trade-off              |
| **Observability**        | request logs, token usage, latency, GPU memory, queue time, errors          |
| **Failure modes**        | OOM, context overflow, queue overload, slow TTFT, slow TPOT, stream failure |

---

# 1. LLM Serving là gì?

**LLM Serving** là quá trình biến một model thành một service có thể nhận request từ app, RAG system, agent hoặc frontend rồi trả về output qua API.

Một LLM serving system tối thiểu gồm:

```text
model runtime
API server
tokenizer
request scheduler
batching mechanism
KV cache manager
streaming response handler
metrics/logging
```

Khác biệt giữa “gọi API” và “serve model”:

| Gọi API                      | Serve model                                           |
| ---------------------------- | ----------------------------------------------------- |
| Provider lo model runtime    | Hiểu runtime hoạt động                              |
| Chỉ quan tâm prompt/response | Quan tâm prefill/decode/KV cache                      |
| Không thấy GPU memory        | Phải đo VRAM                                          |
| Không kiểm soát batching     | Phải hiểu scheduler/batching                          |
| Không tự benchmark backend   | Phải đo TTFT, TPOT, tokens/sec                        |
| Không lo model loading       | Phải biết dtype, context length, quantization concept |

Định nghĩa ngắn:

```text
LLM Serving là tầng biến model thành service đo được, scale được và debug được.
```

---

# 2. LLM inference lifecycle

LLM inference thường có hai pha chính:

```text
prefill → decode → decode → decode → ...
```

Đây là kiến thức lõi của Stage 3.

---

## 2.1 Prefill

**Prefill** là pha model đọc toàn bộ input prompt ban đầu.

Prompt có thể gồm:

```text
system prompt
user message
chat history
RAG retrieved context
agent tool results
format instruction
```

Ví dụ RAG:

```text
system instruction: 300 tokens
user question: 80 tokens
retrieved context: 6,000 tokens
format/citation instruction: 200 tokens

total prompt ≈ 6,580 tokens
```

Trước khi sinh token đầu tiên, model phải xử lý toàn bộ prompt này.

Prefill ảnh hưởng mạnh đến:

```text
TTFT
GPU compute
prompt length
RAG latency
context length
```

Cần nhớ:

```text
Prompt càng dài → prefill càng nặng → token đầu tiên càng lâu xuất hiện.
```

Với RAG, prefill thường là bottleneck vì retrieved context có thể rất dài.

---

## 2.2 Decode

**Decode** là pha model sinh output từng token.

Sau prefill, model bắt đầu sinh:

```text
token 1
→ token 2
→ token 3
→ ...
```

Mỗi token mới lại được đưa vào context để sinh token tiếp theo. Đây là **autoregressive generation**.

Decode ảnh hưởng mạnh đến:

```text
TPOT / ITL
tokens/sec
streaming speed
output latency
KV cache growth
```

Cần nhớ:

```text
Prefill = đọc input.
Decode = sinh output.
```

Ví dụ các workload:

| Workload                     | Nặng ở đâu                   |
| ---------------------------- | ---------------------------- |
| RAG prompt dài, answer ngắn  | Prefill                      |
| Chat prompt ngắn, answer dài | Decode                       |
| Agent gọi model nhiều lần    | Nhiều prefill/decode lặp lại |
| Long-context summarization   | Cả prefill và decode         |
| Code generation dài          | Decode                       |

---

# 3. Autoregressive generation

LLM dạng GPT/decoder-only sinh text theo kiểu **next-token prediction**.

```text
input tokens → predict next token
input + token mới → predict token tiếp theo
input + token mới + token mới nữa → tiếp tục
```

Ví dụ:

```text
Input: "The capital of France is"
Model predicts: "Paris"
Next input: "The capital of France is Paris"
Model predicts: "."
```

Điểm quan trọng:

```text
LLM không sinh nguyên câu một lần.
LLM sinh từng token, và mỗi token mới phụ thuộc vào toàn bộ context trước đó.
```

Điều này giải thích vì sao inference có hai đặc điểm:

```text
output càng dài → decode càng lâu
context càng dài → attention/KV cache càng tốn memory
```

---

# 4. KV cache

**KV cache** là khái niệm cực kỳ quan trọng trong LLM Serving.

Trong Transformer attention, mỗi token tạo ra **Key** và **Value**. Khi sinh token mới, model cần attention tới các token trước đó. Nếu mỗi bước decode đều tính lại Key/Value của toàn bộ token cũ, inference sẽ rất chậm.

KV cache lưu lại Key/Value của các token đã xử lý.

```text
token cũ đã tính K/V
→ lưu vào KV cache
→ token mới dùng lại cache
→ decode nhanh hơn
```

PagedAttention paper chỉ ra KV cache là phần memory lớn trong LLM serving, tăng/giảm động theo request, và nếu quản lý kém sẽ gây fragmentation, duplication, làm giảm batch size và throughput. ([arXiv][2])

---

## 4.1 KV cache giúp gì?

KV cache giúp:

```text
giảm compute khi decode
tăng tốc sinh token
giúp streaming mượt hơn
hỗ trợ nhiều request generate song song
```

Nếu không có KV cache, mỗi token mới phải tính lại quá nhiều phần từ đầu.

---

## 4.2 KV cache tốn gì?

KV cache tốn **VRAM**.

VRAM tăng theo:

```text
số layer
hidden size
số attention heads
context length
batch size / concurrency
output length
dtype của cache
```

Cách hiểu thực tế:

```text
Prompt dài → cache nhiều input tokens.
Output dài → cache tiếp tục tăng.
Nhiều users đồng thời → mỗi request có cache riêng.
```

Vì vậy, khi serving LLM, câu hỏi không chỉ là:

```text
Model có load vừa GPU không?
```

Mà là:

```text
Model load xong còn đủ VRAM cho KV cache của bao nhiêu request?
Context length tối đa là bao nhiêu?
Concurrency chịu được bao nhiêu?
```

---

# 5. Memory / VRAM thinking

VRAM trong LLM Serving thường gồm nhiều phần:

```text
model weights
KV cache
temporary activations / buffers
CUDA runtime overhead
batching overhead
workspace memory
```

Cần hiểu từng phần.

---

## 5.1 Model weights

Model weights là tham số của model.

Ví dụ tư duy đơn giản:

```text
7B parameters ở FP16/BF16 ≈ khoảng 14GB cho weights
```

Chưa tính KV cache và overhead.

Nếu quantized INT8/INT4 thì weights nhỏ hơn, nhưng chất lượng và tốc độ có thể thay đổi tùy model/runtime/kernel.

---

## 5.2 KV cache memory

KV cache phụ thuộc mạnh vào:

```text
context length
batch size
concurrency
output length
```

Ví dụ:

```text
1 user, context 2k tokens → cache nhỏ hơn
16 users, context 8k tokens → cache lớn hơn rất nhiều
```

Đây là lý do LLM serving có thể OOM dù model đã load được.

---

## 5.3 Context length

Context length càng dài:

```text
prefill càng nặng
KV cache càng lớn
TTFT càng cao
VRAM càng căng
cost càng tăng
```

Trong RAG, nếu nhồi quá nhiều chunk:

```text
top_k cao
chunk_size lớn
history dài
instruction dài
```

thì model serving sẽ bị:

```text
TTFT cao
context overflow
VRAM tăng
cost/request tăng
```

---

## 5.4 Batch size / concurrency

Batch size và concurrency giúp tăng throughput, nhưng cũng làm tăng memory.

```text
Concurrency cao → nhiều request cùng giữ KV cache
Batch lớn → GPU utilization tốt hơn
Batch quá lớn → latency từng request có thể tăng
```

Cần nhớ:

```text
VRAM quyết định không chỉ model size, mà cả context length, batch size và concurrency.
```

---

# 6. Serving API

LLM server thường expose API theo kiểu **OpenAI-compatible**, để app/RAG/agent có thể dễ đổi backend.

Tài liệu vLLM ghi rằng vLLM cung cấp HTTP server implement OpenAI Completions API, Chat API và nhiều API khác, giúp serve model và tương tác với model qua API tương thích OpenAI. ([vLLM][3])

Các endpoint cần biết:

```text
GET  /v1/models
POST /v1/chat/completions
POST /v1/completions
POST /v1/embeddings nếu server hỗ trợ
GET  /health
GET  /metrics nếu có
```

---

## 6.1 Chat completions

Chat completions dùng messages:

```json
{
  "model": "llama-3.1-8b-instruct",
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Explain KV cache."}
  ],
  "temperature": 0.2,
  "max_tokens": 512
}
```

Cần hiểu các field:

| Field           | Ý nghĩa                       |
| --------------- | ----------------------------- |
| **model**       | Model server expose           |
| **messages**    | System/user/assistant context |
| **temperature** | Mức random                    |
| **top_p**       | Nucleus sampling              |
| **max_tokens**  | Giới hạn output               |
| **stream**      | Có stream token không         |
| **stop**        | Dừng khi gặp chuỗi nào        |

---

## 6.2 Usage tokens

Response thường có usage:

```json
{
  "prompt_tokens": 1200,
  "completion_tokens": 300,
  "total_tokens": 1500
}
```

Usage giúp tính:

```text
cost/request
prompt dài hay ngắn
answer dài hay ngắn
latency có hợp lý không
```

Cần hiểu:

```text
Prompt tokens ảnh hưởng prefill.
Completion tokens ảnh hưởng decode.
Total tokens ảnh hưởng cost.
```

---

# 7. Streaming response

Streaming là trả token/chunk dần thay vì đợi model sinh xong toàn bộ answer.

Không streaming:

```text
request
→ model generate full answer
→ trả response cuối
```

Streaming:

```text
request
→ token/chunk 1
→ token/chunk 2
→ token/chunk 3
→ ...
→ done
```

Hugging Face Text Generation Inference là một toolkit deploy/serve LLM có hỗ trợ continuous batching và token streaming cho production text generation. ([Hugging Face][4])

---

## 7.1 Vì sao cần streaming?

Streaming giúp:

```text
người dùng thấy phản hồi sớm
chatbot cảm giác nhanh hơn
answer dài vẫn có UX tốt
agent có thể hiển thị progress
```

Streaming không nhất thiết làm model generate nhanh hơn, nhưng làm **perceived latency** tốt hơn.

---

## 7.2 Streaming liên quan metric nào?

Streaming liên quan trực tiếp đến:

```text
TTFT: bao lâu thấy token đầu tiên
TPOT/ITL: token tiếp theo ra nhanh không
E2E latency: toàn bộ answer xong mất bao lâu
```

Nếu TTFT thấp nhưng TPOT cao:

```text
model phản hồi sớm nhưng gõ chậm
```

Nếu TTFT cao nhưng TPOT thấp:

```text
đợi lâu ban đầu nhưng sau đó chữ chạy nhanh
```

---

# 8. Batching

Batching là gom nhiều request để GPU xử lý hiệu quả hơn.

---

## 8.1 Static batching

Static batching gom request thành batch cố định.

Vấn đề:

```text
request dài làm request ngắn phải chờ
batch chưa đầy thì GPU có thể idle
khó xử lý request đến liên tục
```

---

## 8.2 Continuous batching

Continuous batching là cơ chế reschedule batch ở mỗi generation step. Khi request nào xong, request mới có thể vào batch ngay, giúp GPU luôn bận hơn và throughput cao hơn. Tài liệu Hugging Face mô tả continuous batching là cách tối đa hóa GPU utilization bằng cách dynamic reschedule batch ở mỗi generation step, request mới join ngay khi request cũ hoàn thành. ([Hugging Face][5])

Cách hiểu:

```text
Static batching = chờ cả nhóm đi cùng nhau.
Continuous batching = ai xong thì ra, người mới vào ngay.
```

Tác dụng:

```text
tăng GPU utilization
tăng total tokens/sec
xử lý workload nhiều request tốt hơn
giảm idle time
```

Trade-off:

```text
scheduler phức tạp hơn
latency từng request có thể bị ảnh hưởng nếu tải quá cao
cần quản lý KV cache tốt
```

---

# 9. vLLM

**vLLM** là engine phổ biến để serve LLM hiệu năng cao.

Stage 3 nên lấy vLLM làm trọng tâm vì nó gom đúng các concept cần học:

```text
OpenAI-compatible API
PagedAttention
continuous batching
KV cache management
streaming
prefix caching
chunked prefill
benchmark serving
```

vLLM cung cấp OpenAI-compatible server, giúp app có thể gọi model self-hosted qua API shape giống OpenAI. ([vLLM][3])

---

## 9.1 PagedAttention

PagedAttention là concept cốt lõi của vLLM.

Vấn đề:

```text
KV cache của từng request dài/ngắn khác nhau
KV cache tăng dần khi sinh token
memory allocation kém gây fragmentation
VRAM bị lãng phí
batch size/concurrency bị giới hạn
```

PagedAttention quản lý KV cache theo block/page, lấy cảm hứng từ virtual memory/paging trong operating system. Paper PagedAttention cho biết vLLM nhờ cơ chế này có thể giảm waste trong KV cache memory và cải thiện throughput so với một số serving system trước đó. ([arXiv][2])

Cách hiểu:

```text
Thay vì cấp phát một khối cache lớn cứng nhắc,
PagedAttention chia cache thành nhiều block/page
và cấp phát linh hoạt theo nhu cầu từng request.
```

---

## 9.2 Continuous batching trong vLLM

vLLM dùng batching/scheduling để xử lý nhiều request hiệu quả hơn.

Cần hiểu:

```text
nhiều request không sinh cùng độ dài
request đến liên tục
request hoàn thành ở thời điểm khác nhau
server cần đưa request mới vào batch càng sớm càng tốt
```

Đây là lý do continuous batching rất quan trọng cho production serving.

---

## 9.3 Chunked prefill

Chunked prefill là chia prefill của prompt dài thành các phần nhỏ hơn.

Vấn đề:

```text
một request prompt rất dài có thể chiếm GPU lâu
các request decode ngắn khác bị chờ
TTFT/TPOT toàn hệ thống bị ảnh hưởng
```

Chunked prefill giúp scheduler xen kẽ prefill dài với decode của request khác, giảm hiện tượng request dài làm nghẽn toàn server.

Cần hiểu ở Stage 3:

```text
Chunked prefill giúp xử lý workload có prompt length rất khác nhau.
```

---

## 9.4 Prefix caching

Prefix caching cache phần prefix giống nhau giữa nhiều request.

Ví dụ:

```text
system prompt giống nhau
RAG instruction template giống nhau
agent policy prompt giống nhau
```

Nếu nhiều request có cùng prefix, server có thể tái sử dụng phần đã tính thay vì prefill lại từ đầu.

Cần hiểu:

```text
Prefix caching giảm compute khi nhiều request share prompt prefix.
```

---

# 10. Benchmark metrics

Đây là phần cần học cực kỹ trong Stage 3.

Roadmap nhấn mạnh inference phải đo **TTFT, TPOT, tokens/sec, concurrency, GPU memory, cost per task**. ([Khoalearningcode][1])

---

## 10.1 TTFT — Time To First Token

**TTFT** là thời gian từ lúc gửi request tới lúc nhận token đầu tiên. Tài liệu NVIDIA NIM benchmarking định nghĩa TTFT là thời gian từ request đến token đầu tiên được nhận. ([NVIDIA Docs][6])

TTFT gồm:

```text
network/API overhead
queue waiting time
tokenization
prefill
scheduler delay
```

TTFT bị tăng khi:

```text
prompt dài
RAG context quá nhiều
server đang quá tải
queue dài
model lớn
GPU utilization cao
chunked prefill/scheduling chưa tối ưu
```

Ý nghĩa:

```text
TTFT thấp → user thấy model phản hồi nhanh.
TTFT cao → chatbot có cảm giác đơ trước khi chữ đầu tiên xuất hiện.
```

---

## 10.2 TPOT / ITL — Time Per Output Token

**TPOT** là thời gian trung bình cho mỗi output token. Tài liệu NVIDIA NIM gọi Inter-token Latency là thời gian trung bình giữa các token liên tiếp và cũng gọi là TPOT. ([NVIDIA Docs][6])

TPOT bị ảnh hưởng bởi:

```text
decode speed
model size
batch size
KV cache efficiency
sampling config
GPU utilization
quantization/runtime
```

Ý nghĩa:

```text
TPOT thấp → chữ stream ra nhanh.
TPOT cao → model gõ chậm.
```

---

## 10.3 Tokens/sec

Tokens/sec có nhiều cách nhìn.

| Metric                            | Ý nghĩa                           |
| --------------------------------- | --------------------------------- |
| **Output tokens/sec per request** | Tốc độ sinh token của một request |
| **Total output tokens/sec**       | Throughput toàn server            |
| **Input tokens/sec**              | Tốc độ xử lý prompt/prefill       |
| **Total tokens/sec**              | Input + output throughput         |

NVIDIA giải thích tokens per second per system là tổng output tokens/sec của hệ thống qua các concurrent requests; khi số request tăng, TPS tăng tới điểm GPU bão hòa rồi có thể giảm. ([NVIDIA Developer][7])

Cần hiểu:

```text
Latency đo trải nghiệm một request.
Throughput đo năng lực toàn hệ thống.
```

---

## 10.4 End-to-end latency

End-to-end latency là tổng thời gian từ lúc request vào tới lúc response hoàn tất.

```text
E2E latency = queue time + prefill + decode toàn bộ output + overhead
```

Với streaming app, E2E latency chưa đủ. Phải tách:

```text
TTFT: token đầu tiên
TPOT: tốc độ token tiếp theo
E2E: toàn bộ answer xong
```

---

## 10.5 Concurrency

Concurrency là số request đồng thời.

Cần benchmark theo nhiều mức:

```text
concurrency = 1
concurrency = 2
concurrency = 4
concurrency = 8
concurrency = 16
concurrency = 32
```

Ở mỗi mức đo:

```text
TTFT
TPOT
tokens/sec
E2E latency
GPU memory
GPU utilization
error rate
timeout rate
```

Cách đọc:

```text
Concurrency tăng → throughput tăng tới điểm bão hòa.
Sau điểm bão hòa → latency tăng mạnh, timeout/OOM có thể xuất hiện.
```

---

# 11. Memory benchmark

Memory report không chỉ ghi “GPU dùng bao nhiêu GB”. Phải biết memory tăng vì yếu tố nào.

Cần log:

```text
model name
dtype
max context length
prompt length
output length
concurrency
GPU memory used
KV cache usage nếu runtime expose
OOM threshold
```

Các thí nghiệm cần có:

```text
cùng model, tăng context length
cùng model, tăng concurrency
cùng concurrency, tăng max_tokens
cùng workload, đổi dtype/quantization
```

Cách đọc:

| Hiện tượng                  | Có thể do                                 |
| --------------------------- | ----------------------------------------- |
| Load model đã OOM           | Model weights quá lớn                     |
| Load được nhưng request OOM | KV cache/context/concurrency quá lớn      |
| TTFT tăng mạnh              | Prompt dài, queue dài, prefill bottleneck |
| TPOT tăng mạnh              | Decode bottleneck, batch quá tải          |
| Tokens/sec thấp             | GPU chưa được tận dụng, batching kém      |
| Timeout khi concurrency cao | Server quá tải hoặc queue quá dài         |

---

# 12. Cost thinking

Stage này phải học cách nghĩ theo cost.

Cost không chỉ là tiền API. Nếu self-host, cost gồm:

```text
GPU rental/server cost
idle GPU time
storage
network
engineering/maintenance
monitoring/logging
```

Các cách tính:

```text
cost/request
cost/1k input tokens
cost/1k output tokens
cost/successful task
cost/user/month
```

Trade-off quan trọng:

| Tăng cái gì               | Đổi lại                                               |
| ------------------------- | ----------------------------------------------------- |
| Model lớn hơn             | Có thể quality tốt hơn nhưng latency/cost cao hơn     |
| Context dài hơn           | Grounding tốt hơn nhưng TTFT/VRAM/cost tăng           |
| top_k cao hơn trong RAG   | Có thể recall tốt hơn nhưng prompt dài hơn            |
| Reranker mạnh hơn         | Ranking tốt hơn nhưng latency tăng                    |
| Batch/concurrency cao hơn | Throughput tăng nhưng per-request latency có thể tăng |
| Quantization mạnh hơn     | Memory giảm nhưng quality có thể giảm                 |

Định nghĩa ngắn:

```text
LLM Serving tốt là cân bằng quality, latency, throughput, memory và cost.
```

---

# 13. Model loading và serving config

Stage 3 cũng cần hiểu các config khi serve model.

| Config                     | Ý nghĩa                                  |
| -------------------------- | ---------------------------------------- |
| **model path/name**        | Model nào được load                      |
| **tokenizer**              | Tokenizer phải khớp model                |
| **dtype**                  | FP16, BF16, FP8, INT8, INT4              |
| **max model length**       | Context length server cho phép           |
| **served model name**      | Tên model expose qua API                 |
| **GPU memory utilization** | Phần VRAM runtime được phép dùng         |
| **tensor parallel size**   | Chia model qua nhiều GPU                 |
| **quantization**           | Cách nén weights                         |
| **revision**               | Version model cụ thể                     |
| **chat template**          | Format prompt đúng cho instruction model |

Cần hiểu:

```text
Cùng một model nhưng sai tokenizer/chat template có thể làm output tệ.
Cùng một model nhưng max context/config khác nhau có thể benchmark khác nhau.
Không version hóa config thì không reproduce được kết quả.
```

---

# 14. Quantization concept

Quantization là giảm precision của weights để tiết kiệm memory và đôi khi tăng tốc.

Các dạng thường gặp:

```text
FP16 / BF16
FP8
INT8
INT4
AWQ
GPTQ
GGUF
```

Ở Stage 3, chưa cần học kernel sâu, nhưng cần hiểu trade-off:

```text
VRAM giảm
có thể chạy model lớn hơn
concurrency có thể cao hơn
quality có thể giảm
latency không phải lúc nào cũng giảm
runtime/hardware support rất quan trọng
```

Phần quantization sâu hơn nên để Stage 4 GPU Inference.

---

# 15. Serving engine landscape

Stage 3 lấy **vLLM** làm chính, nhưng nên biết landscape.

| Engine                      | Cần hiểu                                                                          |
| --------------------------- | --------------------------------------------------------------------------------- |
| **vLLM**                    | Trọng tâm Stage 3: OpenAI-compatible serving, PagedAttention, continuous batching |
| **Hugging Face TGI**        | Production text-generation server, streaming, continuous batching                 |
| **SGLang**                  | Runtime/serving cho structured generation và agentic workloads                    |
| **TensorRT-LLM**            | NVIDIA-optimized inference, học sâu hơn ở Stage 4                                 |
| **Triton Inference Server** | General inference serving platform, Stage 4/5                                     |
| **NVIDIA NIM**              | Packaged inference microservice concept, liên quan enterprise/AI Factory          |

Hugging Face TGI được mô tả là toolkit deploy và serve LLM, có continuous batching cho incoming requests để tăng total throughput. ([Hugging Face][4])

Cần nhớ:

```text
Stage 3 không phải học hết engine.
Stage 3 là hiểu serving concepts qua vLLM trước.
```

---

# 16. Observability cho LLM Serving

Một serving system phải log/monitor được.

Cần log:

```text
request_id
model_name
model_version
prompt_tokens
completion_tokens
total_tokens
streaming or non-streaming
TTFT
TPOT / ITL
E2E latency
queue time
concurrency
GPU memory
error type
timeout
context overflow
OOM
```

Cần có dashboard hoặc report cho:

```text
latency p50 / p95 / p99
tokens/sec
requests/sec
error rate
timeout rate
GPU memory usage
GPU utilization
cost/request
cost/token
```

Cần hiểu:

```text
Không đo queue time thì không biết chậm do server quá tải hay do model decode chậm.
Không đo prompt/output tokens thì không giải thích được latency và cost.
Không đo GPU memory thì không biết vì sao concurrency bị OOM.
```

---

# 17. Failure modes

LLM Serving có các lỗi riêng.

| Lỗi                             | Nguyên nhân có thể                                       |
| ------------------------------- | -------------------------------------------------------- |
| **CUDA OOM khi load model**     | Model quá lớn, dtype quá nặng, GPU không đủ VRAM         |
| **OOM khi chạy request**        | KV cache quá lớn, context dài, concurrency cao           |
| **Context overflow**            | Prompt + output vượt max context                         |
| **TTFT cao**                    | Prompt dài, queue dài, prefill bottleneck                |
| **TPOT cao**                    | Decode chậm, batch quá tải, model lớn                    |
| **Tokens/sec thấp**             | GPU utilization thấp, batching kém                       |
| **Stream bị ngắt**              | Client disconnect, timeout, server overload              |
| **Output kém bất thường**       | Sai chat template, sai tokenizer, quantization ảnh hưởng |
| **Latency tăng khi nhiều user** | Concurrency vượt điểm bão hòa                            |
| **Benchmark không reproduce**   | Model/config/runtime/version thay đổi                    |

Pattern debug:

```text
Request chậm
→ xem queue time
→ xem prompt tokens
→ xem TTFT
→ xem TPOT
→ xem concurrency
→ xem GPU memory
→ xem tokens/sec
→ xem error/timeout
```

---

# 18. Quan hệ với RAG và Agent

LLM Serving không học tách rời. Nó giải thích vì sao RAG/Agent chạy chậm hoặc tốn tiền.

## Với RAG

RAG thường làm prompt dài:

```text
retrieved chunks
citations
conversation history
instructions
```

Ảnh hưởng:

```text
prompt tokens tăng
prefill tăng
TTFT tăng
KV cache tăng
cost tăng
```

Vì vậy khi RAG chậm, không chỉ optimize vector DB. Còn phải xem:

```text
top_k có quá cao không
chunk_size có quá lớn không
context có nhiễu không
prompt có quá dài không
reranker có làm latency tăng không
```

## Với Agent

Agent thường gọi model nhiều lần:

```text
plan
tool selection
tool result interpretation
verification
final answer
```

Ảnh hưởng:

```text
nhiều API/model calls
latency cộng dồn
cost tăng
trace phức tạp
```

Vì vậy agent production cần:

```text
giảm model calls không cần thiết
cache tool/retrieval result
stream final answer
log token usage từng step
```

---

# 19. Ranh giới Stage 3 và Stage 4

Stage 3 — LLM Serving:

```text
serve model
OpenAI-compatible API
vLLM
streaming
KV cache
batching
benchmark
latency/throughput/memory/cost
```

Stage 4 — GPU Inference:

```text
TensorRT-LLM
Triton
CUDA/GPU memory sâu hơn
kernel/runtime optimization
quantization benchmark sâu
multi-GPU optimization
NVIDIA deployment stack
```

Cần nhớ:

```text
Stage 3 học cách serve và đo.
Stage 4 học cách tối ưu sâu trên GPU/NVIDIA stack.
```

---

# Bảng roadmap

| Nhóm            | Cần học gì                                                                                                                                                                                                                                                                                              | Mức cần đạt                                                                                                                                                                                                                                                                      |
| --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **LLM Serving** | prefill, decode, autoregressive generation, KV cache, OpenAI-compatible API, streaming, vLLM, continuous batching, PagedAttention, chunked prefill, prefix caching, TTFT, TPOT, tokens/sec, concurrency, VRAM usage, context length, batch size, cost/request, cost/token, observability, failure modes | Hiểu được request LLM đi qua **API → tokenizer → scheduler → prefill → decode → KV cache → streaming** như thế nào; biết benchmark latency/throughput/memory/cost, debug OOM/context overflow/slow TTFT/slow TPOT, và tạo được memory + concurrency report cho model self-hosted |

---

# Dòng roadmap chuẩn hóa

```text
Sau Agent mới học LLM Serving.

Đây là đoạn biến “dùng API” thành “biết serve model như engineer”. 
Ở Stage này, cần hiểu một LLM request đi qua API server, tokenizer, scheduler, prefill, KV cache, decode loop và streaming response như thế nào.

Các kiến thức chính gồm: prefill, decode, autoregressive generation, KV cache, OpenAI-compatible API, streaming response, vLLM, PagedAttention, continuous batching, chunked prefill, prefix caching, benchmark TTFT/TPOT/tokens-sec/concurrency, VRAM usage, context length, batch size, quantization concept, cost/request, cost/token, observability và serving failure modes.

Mức cần đạt: không chỉ gọi API từ model provider, mà hiểu model chạy như một inference service; biết đo latency, throughput, memory, cost; biết debug vì sao request chậm, vì sao OOM, vì sao context overflow, vì sao concurrency tăng làm latency tăng; và biết tạo benchmark report cho một self-hosted LLM server.
```

---

# Cấu trúc ghi chú

```text
LLM Serving

1. LLM Serving overview
   - From API user to serving engineer
   - Why this comes after RAG and Agent

2. Inference lifecycle
   - autoregressive generation
   - prefill
   - decode
   - prompt tokens vs output tokens

3. KV cache
   - why KV cache exists
   - how it speeds up decode
   - why it consumes VRAM
   - relation to context length and concurrency

4. Memory / VRAM
   - model weights
   - KV cache
   - context length
   - batch size
   - output length
   - concurrency

5. Serving API
   - OpenAI-compatible server
   - /v1/chat/completions
   - /v1/completions
   - /v1/models
   - usage tokens
   - error handling

6. Streaming response
   - stream=true
   - token delta
   - SSE concept
   - finish reason
   - TTFT and user experience

7. vLLM
   - PagedAttention
   - continuous batching
   - chunked prefill
   - prefix caching
   - OpenAI-compatible API

8. Batching and scheduling
   - static batching
   - continuous batching
   - queue time
   - throughput vs latency

9. Benchmark metrics
   - TTFT
   - TPOT / ITL
   - tokens/sec
   - end-to-end latency
   - concurrency
   - GPU memory
   - error/timeout rate

10. Cost thinking
   - cost/request
   - cost/token
   - cost/successful task
   - quality-latency-cost trade-off

11. Model config
   - model name/path
   - tokenizer
   - dtype
   - max model length
   - chat template
   - quantization concept
   - served model name

12. Observability
   - request logs
   - token usage
   - queue time
   - TTFT/TPOT
   - GPU memory
   - errors/timeouts

13. Failure modes
   - CUDA OOM
   - context overflow
   - queue overload
   - slow TTFT
   - slow TPOT
   - streaming disconnect
   - tokenizer/chat template mismatch
   - config drift

14. Relation to RAG and Agent
   - RAG increases prompt length and prefill
   - Agent increases number of model calls
   - Serving metrics explain real system cost and latency
```

Phạm vi Stage 3 đủ sâu để học LLM Serving nghiêm túc, nhưng chưa lấn sang tối ưu GPU kernel/TensorRT của Stage 4.

[1]: https://khoalearningcode.github.io/AI_Engineer_Roadmap/ "AI Systems Engineer Roadmap"
[2]: https://arxiv.org/abs/2309.06180 "Efficient Memory Management for Large Language Model Serving with PagedAttention"
[3]: https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/ "OpenAI-Compatible Server - vLLM Documentation"
[4]: https://huggingface.co/docs/text-generation-inference/en/index "Text Generation Inference"
[5]: https://huggingface.co/docs/transformers/continuous_batching "Continuous batching"
[6]: https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html "Metrics — NVIDIA NIM LLMs Benchmarking"
[7]: https://developer.nvidia.com/blog/llm-benchmarking-fundamental-concepts/ "LLM Inference Benchmarking: Fundamental Concepts"
