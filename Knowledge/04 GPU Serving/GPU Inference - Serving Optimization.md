Sau **LLM Serving** là **GPU Inference**, khi nền tảng về cách model được serve ra API đã rõ. Stage 4 đi sâu hơn một tầng: **vì sao inference tốn GPU, memory bị ăn ở đâu, tối ưu bằng gì, benchmark trước/sau ra sao, và NVIDIA stack như Triton / TensorRT / TensorRT-LLM / Dynamo nằm ở đâu**.

Roadmap ghi Stage 4 là **GPU Inference Engineer**, mục tiêu là hiểu NVIDIA-style model deployment và optimization. Key skills gồm **GPU memory, quantization, Triton, TensorRT basics, TensorRT-LLM concept, SGLang concept, Dynamo concept, cost per token**; deliverable là GPU-aware inference benchmark và deployment demo có before/after latency, memory notes, config, benchmark chart. ([Khoalearningcode][1])

---

# 4. Sau Serving mới học GPU Inference

Stage 3 — LLM Serving trả lời câu hỏi:

```text
Làm sao serve LLM thành API?
Làm sao đo TTFT, TPOT, tokens/sec, concurrency?
Làm sao dùng vLLM/OpenAI-compatible server?
```

Stage 4 — GPU Inference trả lời sâu hơn:

```text
Vì sao model tốn VRAM?
Tối ưu memory như thế nào?
Quantization giúp gì và đánh đổi gì?
Batch/context ảnh hưởng latency ra sao?
Khi nào dùng Triton?
TensorRT/TensorRT-LLM tối ưu ở tầng nào?
Dynamo/SGLang nằm đâu trong inference stack?
Cost per token bị chi phối bởi yếu tố nào?
```

Tóm tắt quan hệ:

```text
Stage 3 = serve và đo.
Stage 4 = hiểu GPU-aware optimization và deployment stack.
```

---

# GPU Inference — Nội dung cần học

| Mảng                        | Nội dung cần nắm                                                                           |
| --------------------------- | ------------------------------------------------------------------------------------------ |
| **GPU memory**              | weights, activations, KV cache, temporary buffers, batch/context ảnh hưởng VRAM            |
| **Quantization**            | FP16, BF16, FP8, INT8, INT4, GPTQ, AWQ concept                                             |
| **Optimization**            | batching, caching, tensor parallel, pipeline parallel, speculative decoding concept        |
| **Triton Inference Server** | model server, model repository, dynamic batching, ensemble, metrics                        |
| **TensorRT / TensorRT-LLM** | optimization engine concept, TensorRT engine, optimized runtime for NVIDIA GPU             |
| **SGLang**                  | high-performance serving framework, structured generation, prefix/KV reuse concept         |
| **Dynamo**                  | distributed inference framework, disaggregated prefill/decode, routing, multi-node serving |
| **Benchmark mindset**       | trước/sau optimize, latency, throughput, memory, quality, cost/token                       |

---

# 1. GPU Inference là gì?

**GPU Inference** là quá trình chạy model đã train/fine-tune xong trên GPU để phục vụ prediction/generation hiệu quả.

Với LLM, inference gồm:

```text
load model weights lên GPU
→ nhận request
→ tokenize prompt
→ prefill prompt
→ sinh token bằng decode loop
→ dùng KV cache
→ trả output
```

Với CV/classic model, inference có thể là:

```text
image / tensor input
→ preprocessing
→ model forward
→ postprocessing
→ output
```

Điểm Stage 4 cần học là:

```text
GPU không chỉ là nơi chạy model.
GPU là tài nguyên bị giới hạn bởi compute, memory, bandwidth, batching, runtime, precision và deployment engine.
```

---

# 2. GPU memory

Đây là phần quan trọng nhất.

Khi chạy inference, VRAM thường bị dùng bởi:

```text
model weights
KV cache
activations / intermediate tensors
temporary buffers / workspace
CUDA runtime overhead
batching overhead
```

---

## 2.1 Model weights

Weights là tham số của model.

Cách ước lượng đơn giản:

```text
FP32 = 4 bytes / parameter
FP16/BF16 = 2 bytes / parameter
INT8 = 1 byte / parameter
INT4 = 0.5 byte / parameter
```

Ví dụ gần đúng:

```text
7B model FP16 ≈ 14GB weights
13B model FP16 ≈ 26GB weights
70B model FP16 ≈ 140GB weights
```

Đây mới chỉ là weights. Chưa tính KV cache, buffers, runtime overhead.

Cần nhớ:

```text
Model load vừa GPU không có nghĩa là serve được workload thật.
Còn cần VRAM cho KV cache và concurrency.
```

---

## 2.2 KV cache

KV cache là phần memory cực lớn trong LLM inference. PagedAttention paper chỉ ra KV cache memory trong serving rất lớn, thay đổi động theo request, và nếu quản lý kém sẽ gây fragmentation/waste, làm giảm throughput. ([arXiv][2])

KV cache tăng theo:

```text
số layer
hidden size
số token trong context
batch size
concurrency
output length
dtype của cache
```

Cách hiểu:

```text
context dài → cache nhiều token
output dài → cache tiếp tục tăng
nhiều user đồng thời → nhiều cache cùng tồn tại
```

Vì vậy OOM thường xảy ra ở hai mức:

| Trường hợp               | Nguyên nhân                          |
| ------------------------ | ------------------------------------ |
| **OOM khi load model**   | weights quá lớn                      |
| **OOM khi chạy request** | KV cache/context/concurrency quá lớn |

---

## 2.3 Activations và temporary buffers

Dù inference không cần lưu activation cho backprop như training, runtime vẫn cần intermediate tensors và workspace.

Các phần này phụ thuộc vào:

```text
batch size
sequence length
model architecture
runtime engine
kernel implementation
dtype
```

Cần hiểu ở mức Stage 4:

```text
Inference ít tốn activation memory hơn training, nhưng không phải chỉ có weights.
Runtime vẫn cần buffers để tính toán.
```

---

## 2.4 Batch size và context ảnh hưởng VRAM

Batch lớn hơn:

```text
+ GPU utilization tốt hơn
+ throughput cao hơn
- KV cache nhiều hơn
- latency từng request có thể tăng
- dễ OOM hơn
```

Context dài hơn:

```text
+ chứa được nhiều RAG context hơn
- prefill chậm hơn
- KV cache lớn hơn
- TTFT cao hơn
- cost/token/request tăng
```

Cần nhớ:

```text
Trong LLM inference, context length là memory decision, không chỉ là prompt design decision.
```

---

# 3. GPU performance bottleneck

GPU inference bị giới hạn bởi nhiều yếu tố.

| Bottleneck           | Nghĩa là gì                                  |
| -------------------- | -------------------------------------------- |
| **Compute-bound**    | GPU bận tính toán matrix multiply            |
| **Memory-bound**     | GPU chờ đọc/ghi memory                       |
| **Bandwidth-bound**  | Tốc độ truyền data giới hạn                  |
| **CPU overhead**     | CPU scheduler/tokenizer/postprocess chậm     |
| **Network overhead** | Distributed serving cần truyền KV/cache/data |
| **Queueing**         | Request chờ trong scheduler quá lâu          |

LLM prefill và decode có đặc tính khác nhau: prefill xử lý prompt tokens song song hơn, còn decode sinh từng token và dùng KV cache liên tục. TensorRT-LLM blog về disaggregated serving mô tả LLM inference có context/prefill phase để compute KV cache cho prompt tokens và generation/decode phase để sinh token từng bước bằng cached values. ([NVIDIA GitHub][3])

Cần hiểu:

```text
Prefill thường compute-heavy hơn.
Decode thường memory/KV-cache sensitive hơn.
```

Đây là lý do các hệ thống mới quan tâm đến **prefill-decode disaggregation**.

---

# 4. Quantization

Quantization là giảm precision của weights hoặc computation để tiết kiệm memory và tăng hiệu quả inference.

Các precision thường gặp:

| Precision | Ý nghĩa                                     |
| --------- | ------------------------------------------- |
| **FP32**  | 32-bit float, chính xác cao, nặng           |
| **FP16**  | 16-bit float, phổ biến cho GPU inference    |
| **BF16**  | 16-bit, dynamic range tốt hơn FP16          |
| **FP8**   | 8-bit float, rất quan trọng trên GPU mới    |
| **INT8**  | 8-bit integer                               |
| **INT4**  | 4-bit integer, tiết kiệm mạnh hơn           |
| **GPTQ**  | Post-training quantization phổ biến cho LLM |
| **AWQ**   | Activation-aware weight quantization        |
| **GGUF**  | Format phổ biến trong llama.cpp ecosystem   |

TensorRT-LLM documentation có phần riêng về quantization và hỗ trợ triển khai nhiều kiểu tối ưu/quantization cho LLM inference trên NVIDIA GPUs. ([NVIDIA GitHub][4])

---

## 4.1 Quantization giúp gì?

Quantization có thể giúp:

```text
giảm VRAM weights
chạy model lớn hơn trên cùng GPU
tăng batch/concurrency trong một số trường hợp
giảm memory bandwidth pressure
có thể giảm cost/token
```

Ví dụ:

```text
7B FP16 ≈ 14GB weights
7B INT4 ≈ 3.5GB weights, chưa tính overhead
```

---

## 4.2 Quantization đánh đổi gì?

Quantization có thể gây:

```text
giảm quality
sai số số học lớn hơn
một số task nhạy hơn task khác
runtime/kernel không hỗ trợ tốt thì không nhanh hơn
debug khó hơn
```

Cần hiểu:

```text
Quantization không phải cứ thấp bit là tốt.
Phải benchmark quality + latency + memory + throughput cùng lúc.
```

Một nghiên cứu systematic characterization năm 2025 cho thấy trade-off của quantization phụ thuộc mạnh vào task, workload, phương pháp quantization, parallelism và GPU architecture. ([arXiv][5])

---

## 4.3 Khi nào dùng quantization?

Nên cân nhắc khi:

```text
model FP16 không fit GPU
cần tăng concurrency
cần giảm cost/token
throughput đang bị memory-bound
chấp nhận đánh đổi nhẹ quality
```

Không nên dùng mù quáng khi:

```text
task yêu cầu accuracy rất cao
benchmark quality chưa rõ
runtime không support tốt
latency không cải thiện
```

---

# 5. Batching, caching, parallelism

Stage 4 cần hiểu các hướng optimization chính.

---

## 5.1 Batching

Batching gom nhiều request để GPU chạy hiệu quả hơn.

Trong Triton, **dynamic batching** cho phép server gom một hoặc nhiều inference requests thành batch động để tối đa throughput. ([GitHub][6])

Cần hiểu:

```text
batch nhỏ → latency thấp hơn nhưng GPU có thể chưa tận dụng hết
batch lớn → throughput cao hơn nhưng latency/request có thể tăng
```

Với LLM, còn có continuous batching ở serving engine như vLLM/SGLang, nhưng Stage 4 nhìn rộng hơn: batching là kỹ thuật tối ưu utilization.

---

## 5.2 Caching

Các loại cache trong inference:

| Cache               | Ý nghĩa                             |
| ------------------- | ----------------------------------- |
| **KV cache**        | Cache K/V attention cho decode      |
| **Prefix cache**    | Cache prompt prefix giống nhau      |
| **Embedding cache** | Cache embedding query/document      |
| **Response cache**  | Cache answer cho request giống nhau |
| **Reranker cache**  | Cache score query-chunk             |

Cần hiểu:

```text
Caching giảm compute nhưng tăng memory/storage và cần xử lý stale cache.
```

---

## 5.3 Tensor parallelism

Tensor parallelism chia computation của một layer qua nhiều GPU.

Dùng khi:

```text
model quá lớn cho một GPU
muốn tăng throughput bằng nhiều GPU
```

Đánh đổi:

```text
cần communication giữa GPU
phụ thuộc NVLink/PCIe/network
config phức tạp hơn
```

---

## 5.4 Pipeline parallelism

Pipeline parallelism chia các layer model thành nhiều stage trên nhiều GPU.

Ví dụ:

```text
GPU 0: layers 0-15
GPU 1: layers 16-31
```

Đánh đổi:

```text
giúp fit model lớn
nhưng có pipeline bubble/communication overhead
```

---

## 5.5 Data parallel serving / replicas

Chạy nhiều bản copy của model để phục vụ nhiều request.

Ví dụ:

```text
GPU 0: model replica A
GPU 1: model replica B
GPU 2: model replica C
```

Dùng khi:

```text
model fit trên một GPU
muốn tăng requests/sec
```

---

## 5.6 Speculative decoding concept

Speculative decoding dùng model nhỏ/draft model đoán trước token, model lớn verify lại.

Mục tiêu:

```text
tăng tốc decode
giảm latency nếu draft tokens được accept nhiều
```

Stage 4 chỉ cần hiểu concept, chưa cần tự implement.

TensorRT-LLM GitHub nêu các runtime optimizations gồm speculative decoding, prefill-decode disaggregation, expert parallelism và custom kernels cho inference operations. ([GitHub][7])

---

# 6. Triton Inference Server

**NVIDIA Triton Inference Server** là model serving server tổng quát cho nhiều loại model/runtime, không chỉ LLM.

Triton phục vụ model từ **model repository**; inference requests đến server rồi được Triton route đến model tương ứng. Tài liệu NVIDIA mô tả model repository là file-system repository chứa các model mà Triton sẽ make available for inferencing. ([NVIDIA Docs][8])

---

## 6.1 Triton cần học gì?

| Mảng                           | Nội dung                                        |
| ------------------------------ | ----------------------------------------------- |
| **Model repository**           | Folder chứa model/config                        |
| **config.pbtxt**               | Config input/output, batching, instance group   |
| **Backend**                    | TensorRT, ONNX Runtime, PyTorch, Python backend |
| **Dynamic batching**           | Gom request để tăng throughput                  |
| **Concurrent model execution** | Chạy nhiều model/request song song              |
| **Ensemble model**             | Ghép nhiều model thành pipeline                 |
| **Metrics endpoint**           | Prometheus metrics                              |
| **Model versioning**           | Serve nhiều version                             |
| **Health/readiness**           | Kiểm tra server/model sẵn sàng                  |

---

## 6.2 Ensemble trong Triton

Triton ensemble model biểu diễn một pipeline gồm một hoặc nhiều model và kết nối input/output tensors giữa các model. ([NVIDIA Docs][9])

Ví dụ:

```text
preprocess model
→ detector model
→ postprocess model
```

Hoặc:

```text
OCR preprocess
→ vision model
→ text postprocess
```

Cần hiểu:

```text
Triton mạnh cho production inference pipeline, đặc biệt khi có nhiều model/backend cần serve chung.
```

Với LLM thuần, vLLM/TensorRT-LLM/SGLang thường là trọng tâm hơn; với multi-model inference platform, Triton rất quan trọng.

---

# 7. TensorRT và TensorRT-LLM

## 7.1 TensorRT là gì?

TensorRT là NVIDIA inference optimization SDK. Mục tiêu là tối ưu model inference trên NVIDIA GPU bằng các kỹ thuật như graph optimization, precision optimization, kernel/tactic selection và engine build.

Stage 4 chỉ cần hiểu:

```text
TensorRT biến model thành optimized engine để chạy nhanh hơn trên NVIDIA GPU.
```

Chưa cần học sâu plugin/kernel.

---

## 7.2 TensorRT-LLM là gì?

TensorRT-LLM là thư viện NVIDIA để tối ưu inference cho LLM và Visual Gen models. Tài liệu NVIDIA mô tả TensorRT-LLM cung cấp Python API để define LLMs và build TensorRT engines có tối ưu hóa để inference hiệu quả trên NVIDIA GPUs; nó cũng có runtime Python/C++ để execute engines. ([NVIDIA Docs][10])

TensorRT-LLM GitHub mô tả nó có custom kernels cho các inference operations như attention, GEMMs, MoE và runtime optimizations như prefill-decode disaggregation, speculative decoding, expert parallelism. ([GitHub][7])

Cần học:

| Mảng                 | Nội dung                                |
| -------------------- | --------------------------------------- |
| **Engine build**     | Convert/build optimized engine          |
| **Runtime**          | Chạy engine đã optimize                 |
| **Quantization**     | FP8/INT8/INT4 concept                   |
| **Parallelism**      | Tensor/pipeline/expert parallel concept |
| **Serving**          | `trtllm-serve` concept                  |
| **Benchmark**        | So với baseline vLLM/PyTorch            |
| **Hardware support** | Tối ưu theo NVIDIA GPU generation       |

Cần hiểu:

```text
vLLM/SGLang thường dễ bắt đầu hơn.
TensorRT-LLM gần NVIDIA stack hơn và quan trọng khi cần tối ưu sâu trên NVIDIA GPU.
```

---

# 8. SGLang concept

SGLang là high-performance serving framework cho LLM và multimodal models. Documentation mô tả SGLang được thiết kế để deliver low-latency và high-throughput inference cho nhiều workload. ([SGLang Documentation][11])

SGLang đáng học ở Stage 4 vì nó liên quan tới:

```text
structured generation
agentic workloads
prefix/KV reuse
RadixAttention
continuous batching
paged attention
quantization
parallelism
```

SGLang GitHub mô tả runtime có RadixAttention cho prefix caching, zero-overhead CPU scheduler, prefill-decode disaggregation, speculative decoding, continuous batching, paged attention, tensor/pipeline/expert/data parallelism, structured outputs và multi-LoRA batching. ([GitHub][12])

Cần hiểu ở mức Stage 4:

```text
SGLang là một inference/runtime framework tối ưu cho workload nhiều generation calls, structured output, agent/RAG/multi-turn.
```

Chưa cần học sâu code SGLang ngay, nhưng nên biết nó cạnh vLLM và TensorRT-LLM trong inference engine landscape.

---

# 9. NVIDIA Dynamo concept

Dynamo là tầng cao hơn inference engine đơn lẻ.

NVIDIA mô tả Dynamo là open-source, low-latency, modular inference framework cho serving generative AI models trong distributed environments, với resource scheduling, request routing, optimized memory management và data transfer. ([NVIDIA Developer][13])

Tài liệu Dynamo về disaggregated serving mô tả: trong deployment tách rời, **prefill workers** và **decode workers** là hai pool riêng; request đi qua prefill trước, sau đó KV cache state được chuyển/expose cho decode, rồi response được stream từ decode worker. ([Dynamo Documentation][14])

Cần hiểu:

```text
vLLM/SGLang/TensorRT-LLM = inference engines/runtimes.
Dynamo = distributed orchestration layer để phối hợp engines trên nhiều GPU/node.
```

Dynamo quan trọng khi workload lớn hơn single-node:

```text
multi-node serving
reasoning models output dài
agent workload nhiều bước
prefill/decode bottleneck khác nhau
KV cache routing
autoscaling GPU workers
```

Stage 4 chỉ cần hiểu concept, chưa cần deploy Dynamo sâu.

---

# 10. Cost per token

Stage 4 phải học cách nghĩ theo **cost per token**.

Cost/token bị ảnh hưởng bởi:

```text
model size
GPU type
GPU utilization
batching efficiency
prompt length
output length
KV cache memory
quantization
parallelism overhead
engine/runtime
idle time
```

Cách tính tư duy:

```text
cost/hour của GPU
÷ tokens/hour thực tế
= cost/token
```

Hoặc:

```text
cost/request = prompt_tokens_cost + output_tokens_cost + overhead
```

Cần đo:

```text
tokens/sec
requests/sec
GPU utilization
VRAM usage
latency p50/p95/p99
error/timeout rate
quality before/after optimization
```

Cần nhớ:

```text
Tối ưu inference không chỉ là giảm latency.
Mục tiêu thật là giảm cost/token trong khi giữ quality và SLO.
```

---

# 11. Benchmark mindset

Đây là phần phải học cực kỹ.

Mỗi optimization phải có benchmark **trước/sau**.

Ví dụ:

```text
Baseline: vLLM FP16
Optimization 1: vLLM INT4/AWQ
Optimization 2: TensorRT-LLM FP8
Optimization 3: context length giảm
Optimization 4: batch/concurrency tuning
```

Mỗi lần so phải ghi:

```text
model
GPU
dtype/quantization
runtime engine
context length
input tokens
output tokens
concurrency
batching config
max tokens
temperature/top_p
TTFT
TPOT
tokens/sec
E2E latency
VRAM usage
quality metric
error rate
```

---

## 11.1 Không benchmark mơ hồ

Không được chỉ ghi:

```text
TensorRT nhanh hơn.
Quantization nhẹ hơn.
Batching tốt hơn.
```

Phải ghi kiểu:

```text
Model: Qwen/Llama 7B
GPU: RTX 3090 24GB
Runtime: vLLM
Precision: FP16 vs INT4
Prompt length: 2k tokens
Output length: 256 tokens
Concurrency: 1/4/8/16

Metrics:
TTFT
TPOT
tokens/sec
VRAM
quality sample/failure cases
```

---

## 11.2 Quality phải đi cùng performance

Tối ưu performance mà làm output tệ thì không phải tối ưu thật.

Cần đo hoặc kiểm tra:

```text
answer correctness
RAG faithfulness
format validity
tool-call validity
code correctness
hallucination rate
human spot-check
```

Đặc biệt với quantization:

```text
phải so quality trước/sau
không chỉ so memory
```

---

# 12. Quan hệ Stage 3 vs Stage 4

| Nội dung   | Stage 3 — LLM Serving          | Stage 4 — GPU Inference                                     |
| ---------- | ------------------------------ | ----------------------------------------------------------- |
| Trọng tâm  | Serve model thành API          | Tối ưu inference system                                     |
| Tool chính | vLLM, OpenAI-compatible API    | Triton, TensorRT, TensorRT-LLM, SGLang, Dynamo concept      |
| Metric     | TTFT, TPOT, tokens/sec         | Before/after latency, throughput, VRAM, cost/token, quality |
| Memory     | Hiểu KV cache/context          | Phân tích weights/KV/buffers/parallelism/quantization       |
| Scope      | Single service/self-host model | GPU-aware deployment and optimization                       |
| Độ sâu     | Serving engineer               | Inference systems engineer                                  |

Cần nhớ:

```text
Stage 3 biết chạy và đo.
Stage 4 biết vì sao nó nhanh/chậm/tốn VRAM và tối ưu theo GPU stack.
```

---

# 13. Ranh giới: chưa cần CUDA kernel sâu

Ở Stage 4, chưa cần học sâu:

```text
CUDA kernel programming
custom attention kernel
Triton language kernel writing
CUTLASS sâu
compiler internals
GPU assembly
```

Trước mắt cần hiểu:

```text
GPU memory đi đâu
runtime engine làm gì
quantization trade-off gì
batching/caching/parallelism ảnh hưởng gì
Triton/TensorRT-LLM/SGLang/Dynamo nằm tầng nào
benchmark trước/sau optimize ra sao
```

CUDA kernel sâu để sau nếu đi nhánh inference optimization rất nặng.

---

# Bảng roadmap

| Nhóm              | Cần học gì                                                                                                                                                                                                                                                                                                                           | Mức cần đạt                                                                                                                                                                                                                                                                                                                                   |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **GPU Inference** | GPU memory, weights, activations, KV cache, batch/context VRAM, FP16/BF16/FP8/INT8/INT4, GPTQ/AWQ concept, batching, caching, tensor parallel, pipeline parallel, Triton Inference Server, dynamic batching, ensemble, TensorRT basics, TensorRT-LLM concept, SGLang concept, Dynamo concept, cost per token, before/after benchmark | Hiểu được inference system trên GPU: VRAM bị dùng bởi weights/KV cache/buffers ra sao, batch/context/concurrency ảnh hưởng latency-memory thế nào, quantization đánh đổi quality-memory ra sao, Triton/TensorRT-LLM/SGLang/Dynamo nằm ở tầng nào, và biết tạo benchmark trước/sau tối ưu với latency, throughput, VRAM, quality và cost/token |

---

# Dòng roadmap chuẩn hóa

```text
Sau LLM Serving mới học GPU Inference.

Đây là đoạn đi từ “serve model được” sang “hiểu và tối ưu inference system trên GPU”. 
Stage này tập trung vào GPU memory, quantization, batching, caching, parallelism concept, Triton Inference Server, TensorRT/TensorRT-LLM concept, SGLang, Dynamo và cost per token.

Cần hiểu VRAM bị dùng bởi model weights, KV cache, activations/buffers; context length, batch size và concurrency ảnh hưởng memory/latency ra sao; quantization như FP16/BF16/FP8/INT8/INT4, GPTQ/AWQ giúp gì và đánh đổi gì; Triton phục vụ nhiều model/backend ra sao; TensorRT-LLM tối ưu LLM inference trên NVIDIA GPU ở tầng engine/runtime; SGLang tối ưu structured/agentic generation workload; Dynamo là distributed inference orchestration concept cho multi-node serving và prefill/decode disaggregation.

Mức cần đạt: chưa cần viết CUDA kernel sâu, nhưng phải biết đọc và tạo GPU-aware inference benchmark: trước/sau optimize, TTFT, TPOT, tokens/sec, throughput, latency p95/p99, VRAM, quality, error rate và cost/token. Khi inference chậm hoặc OOM, biết phân tích lỗi nằm ở weights, KV cache, context length, batch size, concurrency, quantization, scheduler hay runtime engine.
```

---

# Cấu trúc ghi chú

```text
GPU Inference

1. GPU Inference overview
   - from serving API to GPU-aware inference system
   - why this comes after LLM Serving

2. GPU memory
   - model weights
   - KV cache
   - activations / buffers
   - CUDA/runtime overhead
   - batch size
   - context length
   - concurrency

3. Inference bottlenecks
   - compute-bound
   - memory-bound
   - bandwidth-bound
   - CPU overhead
   - queueing
   - prefill vs decode characteristics

4. Quantization
   - FP32
   - FP16
   - BF16
   - FP8
   - INT8
   - INT4
   - GPTQ
   - AWQ
   - quality-memory-latency trade-off

5. Optimization concepts
   - batching
   - dynamic batching
   - continuous batching concept
   - KV cache
   - prefix caching
   - tensor parallelism
   - pipeline parallelism
   - data parallel replicas
   - speculative decoding concept

6. Triton Inference Server
   - model repository
   - config.pbtxt
   - backends
   - dynamic batching
   - concurrent execution
   - ensemble models
   - metrics
   - model versioning

7. TensorRT basics
   - optimized engine concept
   - graph/layer optimization
   - precision optimization
   - runtime execution

8. TensorRT-LLM concept
   - LLM engine build
   - optimized runtime
   - quantization support
   - parallelism strategies
   - speculative decoding concept
   - NVIDIA GPU optimization

9. SGLang concept
   - high-performance LLM/multimodal serving
   - structured generation
   - RadixAttention / prefix reuse concept
   - agentic workload relevance

10. NVIDIA Dynamo concept
   - distributed inference framework
   - request routing
   - disaggregated prefill/decode
   - KV cache transfer
   - multi-node serving
   - relation to vLLM/SGLang/TensorRT-LLM

11. Cost per token
   - GPU cost/hour
   - tokens/hour
   - utilization
   - model size
   - context length
   - quantization
   - batching
   - idle time

12. Benchmark methodology
   - baseline
   - optimized config
   - same workload
   - same model/GPU
   - TTFT
   - TPOT
   - tokens/sec
   - latency p50/p95/p99
   - VRAM
   - quality metric
   - error/timeout rate

13. Failure modes
   - OOM on load
   - OOM during request
   - context overflow
   - low GPU utilization
   - slow TTFT
   - slow TPOT
   - poor quality after quantization
   - bad batching config
   - parallelism overhead
   - config drift
```

Phạm vi Stage 4 tập trung vào **inference system** và NVIDIA-style deployment, nhưng chưa đi sâu vào CUDA kernel.

[1]: https://khoalearningcode.github.io/AI_Engineer_Roadmap/ "AI Systems Engineer Roadmap"
[2]: https://arxiv.org/abs/2312.07104 "SGLang: Efficient Execution of Structured Language Model Programs"
[3]: https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html "Disaggregated Serving in TensorRT LLM"
[4]: https://nvidia.github.io/TensorRT-LLM/ "Welcome to TensorRT LLM's Documentation!"
[5]: https://arxiv.org/abs/2508.16712 "Systematic Characterization of LLM Quantization: A Performance, Energy, and Quality Perspective"
[6]: https://github.com/triton-inference-server/tutorials/blob/main/Conceptual_Guide/Part_2-improving_resource_utilization/README.md "Dynamic Batching & Concurrent Model Execution"
[7]: https://github.com/NVIDIA/TensorRT-LLM "NVIDIA/TensorRT-LLM"
[8]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html "NVIDIA Triton Inference Server"
[9]: https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/user_guide/ensemble_models.html "Ensemble Models — NVIDIA Triton Inference Server"
[10]: https://docs.nvidia.com/tensorrt-llm/index.html "NVIDIA TensorRT-LLM"
[11]: https://docs.sglang.ai/ "SGLang Documentation: Welcome to SGLang"
[12]: https://github.com/sgl-project/sglang "SGLang is a high-performance serving framework for large ..."
[13]: https://developer.nvidia.com/dynamo "Dynamo Inference Framework | NVIDIA Developer"
[14]: https://docs.dynamo.nvidia.com/dynamo/dev/user-guides/disaggregated-serving "Disaggregated Serving | NVIDIA Dynamo Documentation"
