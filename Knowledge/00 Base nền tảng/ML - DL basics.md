# ML/DL Basics — Knowledge cần học kỹ

## 1. Machine Learning loop

Đây là xương sống của machine learning.

Một model học theo vòng lặp:

```text
data → model → prediction → loss → gradient → optimizer → update weights → validation
```

Cần hiểu kỹ các khái niệm:

|Khái niệm|Cần hiểu|
|---|---|
|**Input / x**|Dữ liệu đưa vào model: ảnh, text, vector, tabular feature|
|**Label / y**|Đáp án đúng dùng trong supervised learning|
|**Prediction / ŷ**|Output model dự đoán|
|**Loss**|Độ sai giữa prediction và label|
|**Gradient**|Hướng cần chỉnh weight để loss giảm|
|**Optimizer**|Thuật toán cập nhật weight|
|**Epoch**|Một lần model đi qua toàn bộ training data|
|**Batch**|Một nhóm sample được xử lý trong một update|
|**Validation**|Tập dùng để kiểm tra model có generalize không|

Trong PyTorch, training loop thường gồm dataset/dataloader, model forward, loss function, optimizer, backward, và update weight. PyTorch docs mô tả optimizer là thành phần điều chỉnh model weights dựa trên kết quả của loss function trong training loop. ([docs.pytorch.org](https://docs.pytorch.org/tutorials/beginner/introyt/trainingyt.html "Training with PyTorch"))

Cách đọc đơn giản:

```text
Model dự đoán sai → loss cao.
Loss cao → gradient chỉ hướng sửa.
Optimizer sửa weight.
Lặp lại nhiều lần → model học tốt hơn.
```

---

## 2. Training vs Inference

Đây là cặp khái niệm cần phân biệt rõ.

|Phần|Training|Inference|
|---|---|---|
|Mục đích|Làm model học|Dùng model đã học|
|Có label không|Có hoặc self-supervised target|Không|
|Có loss không|Có|Thường không|
|Có update weight không|Có|Không|
|Quan tâm chính|Loss, metric, generalization|Latency, memory, throughput, quality|

Training:

```text
input + label → prediction → loss → backpropagation → update weights
```

Inference:

```text
input → model → output
```

Ví dụ RAG/LLM:

```text
Training: fine-tune embedding model, reranker, LLM.
Inference: user hỏi → retrieve context → LLM generate answer.
```

Ví dụ CV:

```text
Training: ảnh + label/object box → model học.
Inference: ảnh mới → model detect/predict.
```

Trong inference, model không nên update weight. Với PyTorch thường dùng `model.eval()` và tắt gradient bằng `torch.no_grad()` để giảm tính toán và memory.

---

## 3. Data, feature, representation

Model không hiểu dữ liệu như con người. Nó làm việc với biểu diễn số.

```text
text → token ids / embedding
image → tensor pixel / visual embedding
tabular data → feature vector
document → chunk embedding
```

### Feature

Feature là thông tin đầu vào được dùng để dự đoán.

Ví dụ tabular ML:

```text
age, income, transaction_count, fail_payment_count
```

Ví dụ CV:

```text
pixel, edge, texture, object shape
```

Ví dụ RAG:

```text
query text, document chunk, metadata, embedding vector
```

### Representation

Representation là cách model biểu diễn input bên trong.

Ví dụ:

```text
"A dog running in the park"
→ embedding vector 768 chiều
```

Cần hiểu:

> Deep Learning mạnh vì nó tự học representation, thay vì chỉ dùng feature do con người thiết kế thủ công.

---

## 4. Dataset split

Phải hiểu rõ 3 tập:

|Tập|Vai trò|
|---|---|
|**Train set**|Dùng để model học|
|**Validation set**|Dùng để chọn model, tune hyperparameter, chọn checkpoint|
|**Test set**|Dùng để đánh giá cuối cùng|

Sai lầm phổ biến:

```text
Tune nhiều lần trên test set → test set không còn khách quan.
```

Cần hiểu thêm:

|Khái niệm|Ý nghĩa|
|---|---|
|**Random split**|Chia ngẫu nhiên|
|**Stratified split**|Giữ tỷ lệ class giữa train/val/test|
|**Time-aware split**|Train bằng quá khứ, test bằng tương lai|
|**Group split**|Không để cùng user/document/video xuất hiện ở cả train và val|
|**Leakage**|Thông tin không nên có lọt vào train/eval|

Với AI Engineer, data split quan trọng không kém model. Split sai thì metric đẹp nhưng giả.

---

## 5. Data leakage

Data leakage là khi model hoặc evaluation nhìn thấy thông tin không nên thấy.

Ví dụ:

```text
Duplicate sample xuất hiện cả train và validation.
Feature chứa trực tiếp hoặc gián tiếp label.
Dữ liệu tương lai lọt vào training.
RAG index chứa luôn answer của eval set.
Cùng document xuất hiện ở train và test dưới dạng gần giống nhau.
```

Hậu quả:

```text
Validation/test score rất cao
nhưng deploy thật thì fail
```

Cần học leakage theo các dạng:

| Dạng leakage              | Ví dụ                                                                               |
| ------------------------- | ----------------------------------------------------------------------------------- |
| **Target leakage**        | Feature chứa thông tin của label                                                    |
| **Temporal leakage**      | Dùng dữ liệu tương lai để predict quá khứ                                           |
| **Duplicate leakage**     | Bản sao/gần sao có ở cả train và test                                               |
| **Preprocessing leakage** | Fit scaler/vectorizer trên toàn bộ data trước khi split                             |
| **Evaluation leakage**    | Test set bị dùng để tune nhiều lần                                                  |
| **RAG leakage**           | Eval answers hoặc ground-truth passages nằm sẵn trong index theo cách không thực tế |

---

## 6. Loss function

Loss là tín hiệu để model học.

```text
Prediction càng sai → loss càng cao.
Prediction càng đúng → loss càng thấp.
```

Cần phân biệt:

```text
Loss = thứ model tối ưu.
Metric = thứ con người dùng để đánh giá bài toán thật.
```

---

## 6.1 Cross-Entropy Loss

Dùng cho classification và next-token prediction.

Ví dụ classification:

```text
Label thật: cat

Model A:
cat = 0.90 → loss thấp

Model B:
cat = 0.10 → loss cao
```

Ý nghĩa:

> Cross-entropy phạt mạnh khi model tự tin vào class sai.

Dùng trong:

```text
image classification
text classification
token classification
LLM next-token prediction
```

Trong LLM:

```text
Input: "The capital of France is"
Target next token: "Paris"
```

Model càng gán xác suất cao cho token đúng thì loss càng thấp.

---

## 6.2 MSE / MAE

Dùng cho regression.

### MSE

```text
MSE = mean((prediction - target)^2)
```

Tính chất:

```text
Phạt lỗi lớn rất mạnh.
Nhạy với outlier.
```

### MAE

```text
MAE = mean(|prediction - target|)
```

Tính chất:

```text
Dễ hiểu hơn.
Ít nhạy với outlier hơn MSE.
```

Dùng trong:

```text
price prediction
depth estimation
risk score regression
latency prediction
```

---

## 6.3 Contrastive Loss

Rất quan trọng cho embedding, retrieval, CLIP, VLM.

Ý tưởng:

```text
positive pairs → kéo gần nhau
negative pairs → đẩy xa nhau
```

Ví dụ image-text:

```text
image: ảnh con chó
positive text: "a dog running"
negative text: "a red car"
```

Model học để:

```text
similarity(image, correct text) cao
similarity(image, wrong text) thấp
```

CLIP là ví dụ kinh điển: model học visual concepts từ natural language supervision và có thể dùng text label để làm zero-shot classification. ([OpenAI](https://openai.com/index/clip/ "CLIP: Connecting text and images"))

---

## 6.4 Triplet Loss

Triplet loss gồm:

```text
anchor
positive
negative
```

Mục tiêu:

```text
distance(anchor, positive) < distance(anchor, negative)
```

Ví dụ person re-ID:

```text
anchor = ảnh người A
positive = ảnh khác của người A
negative = ảnh người B
```

Dùng trong:

```text
face recognition
person re-identification
image retrieval
metric learning
```

---

## 6.5 Ranking Loss

Dùng khi bài toán cần xếp hạng.

Ví dụ:

```text
query q
doc A relevant
doc B irrelevant
```

Mục tiêu:

```text
score(q, doc A) > score(q, doc B)
```

Dùng trong:

```text
search
recommendation
retrieval
reranking
RAG
```

---

## 7. Optimization

Optimization là quá trình cập nhật weight để giảm loss.

Công thức tư duy:

```text
new_weight = old_weight - learning_rate × gradient
```

Cần học:

|Khái niệm|Ý nghĩa|
|---|---|
|**Gradient descent**|Cập nhật weight theo hướng giảm loss|
|**Learning rate**|Độ lớn mỗi bước cập nhật|
|**Optimizer**|Thuật toán update weight|
|**SGD**|Optimizer cơ bản|
|**Adam / AdamW**|Optimizer phổ biến trong deep learning|
|**Weight decay**|Regularization bằng cách phạt weight lớn|
|**Scheduler**|Điều chỉnh learning rate theo thời gian|
|**Warmup**|Tăng learning rate dần ở đầu training|

Learning rate quá lớn:

```text
Loss dao động, training không ổn định.
```

Learning rate quá nhỏ:

```text
Loss giảm rất chậm, training lâu.
```

Learning rate hợp lý:

```text
Train loss giảm ổn định, validation metric cải thiện.
```

---

## 8. Overfitting

Overfitting là khi model học quá kỹ train set nhưng không generalize tốt.

Dấu hiệu:

```text
train loss giảm tiếp
train metric tăng cao
validation loss tăng
validation metric đứng yên hoặc giảm
```

Ví dụ:

```text
Train accuracy = 99%
Validation accuracy = 72%
```

Cách hiểu:

```text
Model học thuộc dữ liệu train thay vì học quy luật tổng quát.
```

Ví dụ CV:

```text
Model học background hoặc camera style thay vì object thật.
```

Ví dụ RAG/retrieval:

```text
Retriever tốt trên eval query giống hệt document,
nhưng user hỏi paraphrase thì retrieve sai.
```

TensorFlow docs mô tả overfitting/underfitting theo quan hệ giữa train data và khả năng học pattern: underfitting xảy ra khi model vẫn còn nhiều chỗ cải thiện trên train data; ngược lại, overfitting là khi model học train data nhưng không generalize tốt. ([TensorFlow](https://www.tensorflow.org/tutorials/keras/overfit_and_underfit "Overfit and underfit | TensorFlow Core"))

---

## 9. Underfitting

Underfitting là khi model chưa học đủ pattern.

Dấu hiệu:

```text
train loss cao
validation loss cao
train metric thấp
validation metric thấp
```

Nguyên nhân:

```text
model quá đơn giản
feature/representation yếu
training chưa đủ
learning rate không phù hợp
regularization quá mạnh
input thiếu thông tin
data quá nhiễu
```

Ví dụ:

```text
Dùng linear model cho ảnh phức tạp.
Dùng embedding model không hợp domain cho tài liệu chuyên ngành.
```

---

## 10. Generalization

Generalization là khả năng model làm tốt trên dữ liệu chưa thấy.

Cần hiểu:

```text
Mục tiêu thật của ML không phải train score cao.
Mục tiêu thật là performance tốt trên unseen data.
```

Các yếu tố ảnh hưởng generalization:

|Yếu tố|Tác động|
|---|---|
|Data diversity|Data đa dạng giúp model bớt học thuộc|
|Model capacity|Model quá lớn dễ overfit nếu data ít|
|Regularization|Giảm overfit|
|Data split|Split sai làm metric ảo|
|Distribution shift|Data deploy khác train làm model fail|
|Label noise|Label sai làm model học sai|
|Metric choice|Metric sai làm tối ưu sai mục tiêu|

---

## 11. Regularization

Regularization là nhóm kỹ thuật giúp giảm overfitting.

|Kỹ thuật|Ý nghĩa|
|---|---|
|**More data**|Tăng độ đa dạng|
|**Data augmentation**|Tạo biến thể từ dữ liệu|
|**Dropout**|Random tắt neuron khi train|
|**Weight decay**|Phạt weight quá lớn|
|**Early stopping**|Dừng khi validation không cải thiện|
|**Smaller model**|Giảm capacity|
|**Label smoothing**|Không cho model quá tự tin|
|**Noise injection**|Tăng robustness|
|**Better split**|Tránh leakage|

Ví dụ augmentation trong CV:

```text
crop
flip
resize
color jitter
blur
brightness change
```

Ví dụ augmentation trong text/RAG:

```text
paraphrase query
hard negative mining
query variation
domain-specific examples
```

---

## 12. Evaluation metrics

Metric phải chọn theo bài toán. `sklearn.metrics` có nhiều nhóm metric cho classification, ranking, regression, probability score, pairwise distance, clustering. ([Scikit-learn](https://scikit-learn.org/stable/modules/model_evaluation.html "3.4. Metrics and scoring: quantifying the quality of predictions"))

---

## 12.1 Classification metrics

### Accuracy

```text
accuracy = correct predictions / total samples
```

Phù hợp khi class tương đối cân bằng.

Không phù hợp khi class imbalance.

Ví dụ:

```text
98% normal
2% fraud

Model luôn predict normal
→ accuracy = 98%
→ nhưng fraud recall = 0%
```

---

### Precision

```text
precision = TP / (TP + FP)
```

Ý nghĩa:

```text
Trong những case model báo positive,
bao nhiêu case thật sự positive?
```

Quan trọng khi false positive đắt.

Ví dụ:

```text
Model báo 100 giao dịch fraud.
80 giao dịch fraud thật.
precision = 80%.
```

---

### Recall

```text
recall = TP / (TP + FN)
```

Ý nghĩa:

```text
Trong toàn bộ positive thật,
model bắt được bao nhiêu?
```

Quan trọng khi false negative đắt.

Ví dụ:

```text
Có 100 fraud thật.
Model bắt được 70.
recall = 70%.
```

---

### F1-score

```text
F1 = 2 × precision × recall / (precision + recall)
```

F1 là harmonic mean của precision và recall, thường hữu ích hơn accuracy khi class imbalance. Google ML Crash Course cũng nhấn mạnh F1 cân bằng precision/recall và phù hợp hơn accuracy trong dataset mất cân bằng. ([Google for Developers](https://developers.google.com/machine-learning/crash-course/classification/accuracy-precision-recall "Classification: Accuracy, recall, precision, and related metrics"))

---

### ROC-AUC

ROC-AUC đo khả năng model phân biệt positive và negative trên nhiều threshold. Google ML Crash Course mô tả AUC như xác suất một positive example được xếp cao hơn một negative example ngẫu nhiên. ([Google for Developers](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc "Classification: ROC and AUC | Machine Learning"))

Cần hiểu:

```text
AUC cao → model rank positive cao hơn negative tốt hơn.
```

Nhưng với positive class cực hiếm, PR-AUC thường hữu ích hơn ROC-AUC.

---

### PR-AUC

Precision-Recall curve hữu ích khi class imbalance. Scikit-learn docs nói precision-recall là thước đo hữu ích khi class rất mất cân bằng. ([Scikit-learn](https://scikit-learn.org/stable/auto_examples/model_selection/plot_precision_recall.html "Precision-Recall"))

Dùng khi:

```text
fraud detection
rare disease detection
spam detection với spam hiếm
BAD user scoring
```

---

## 12.2 Regression metrics

|Metric|Công thức/ý nghĩa|Khi dùng|
|---|---|---|
|**MAE**|Mean absolute error|Muốn sai số dễ hiểu, cùng đơn vị target|
|**MSE**|Mean squared error|Muốn phạt lỗi lớn mạnh|
|**RMSE**|Căn của MSE|Muốn cùng đơn vị target nhưng vẫn phạt lỗi lớn|
|**R²**|Mức giải thích variance|Đánh giá độ fit tổng quát|

Ví dụ:

```text
true depth = 20m
pred depth = 23m
absolute error = 3m
```

---

## 12.3 Retrieval metrics

Đây là nhóm cực quan trọng cho RAG, semantic search, image retrieval, person re-ID, VLM retrieval.

Sentence Transformers mô tả semantic search là quá trình embed query và corpus vào vector space rồi tìm corpus embeddings gần nhất với query embedding, mặc định thường dùng cosine similarity. ([sbert.net](https://www.sbert.net/examples/sentence_transformer/applications/semantic-search/README.html "Semantic Search — Sentence Transformers documentation"))

### Recall@k

```text
Trong top-k retrieved results, có chứa relevant item không?
```

Ví dụ:

```text
100 queries
85 queries có correct document trong top-5
Recall@5 = 85%
```

Ý nghĩa với RAG:

```text
Recall@k thấp → LLM không có evidence đúng → answer dễ sai.
```

---

### Precision@k

```text
Trong top-k retrieved results, bao nhiêu phần relevant?
```

Ví dụ:

```text
Top-5 có 3 document đúng
Precision@5 = 3/5 = 0.6
```

Ý nghĩa:

```text
Precision@k thấp → context nhiều nhiễu → LLM dễ bị distract.
```

---

### MRR

```text
MRR = trung bình 1 / rank của relevant result đầu tiên
```

Ví dụ:

```text
Correct doc ở rank 1 → score = 1
Correct doc ở rank 5 → score = 0.2
```

Ý nghĩa:

```text
MRR cao → evidence đúng thường nằm rất sớm.
```

---

### MAP

MAP dùng khi mỗi query có nhiều relevant documents.

Ý nghĩa:

```text
Đánh giá cả việc retrieve đúng và xếp relevant documents lên cao.
```

---

### NDCG

NDCG dùng khi relevance có nhiều mức.

Ví dụ:

```text
3 = rất liên quan
2 = liên quan
1 = hơi liên quan
0 = không liên quan
```

Ý nghĩa:

```text
Relevant item càng cao trong ranking càng tốt.
Highly relevant item đứng trên mildly relevant item càng tốt.
```

---

## 13. Embedding

Embedding là vector biểu diễn semantic meaning.

```text
text/image/document/user/item → vector
```

Ví dụ:

```text
"dog" gần "puppy"
"dog" xa "airplane"
```

Trong RAG:

```text
document chunk → embedding
query → embedding
similarity search → top-k chunks
```

Trong VLM:

```text
image → image embedding
text → text embedding
similarity(image, text)
```

Cần học các khái niệm:

|Khái niệm|Ý nghĩa|
|---|---|
|**Embedding dimension**|Số chiều vector|
|**Dense embedding**|Vector dense, thường dùng semantic search|
|**Sparse embedding**|Vector thưa, gần keyword/sparse retrieval|
|**Cosine similarity**|Đo hướng giống nhau giữa vector|
|**Dot product**|Đo similarity có phụ thuộc magnitude|
|**Normalization**|Chuẩn hóa vector trước khi so sánh|
|**Nearest neighbor search**|Tìm vector gần nhất|
|**Approximate nearest neighbor**|Tìm gần đúng để scale lớn|
|**Domain mismatch**|Embedding model không hợp domain|
|**Hard negative**|Negative nhìn rất giống positive nhưng thật ra sai|

Hugging Face course mô tả semantic search bằng embeddings như một cách xây search engine dựa trên ý nghĩa thay vì chỉ keyword matching. ([Hugging Face](https://huggingface.co/learn/llm-course/chapter5/6 "Semantic search with FAISS"))

---

## 14. Vector similarity

Các cách so sánh vector phổ biến:

|Similarity|Ý nghĩa|
|---|---|
|**Cosine similarity**|So sánh hướng vector|
|**Dot product**|So sánh tích vô hướng|
|**Euclidean distance**|Khoảng cách hình học|
|**Manhattan distance**|Tổng khoảng cách theo từng chiều|

Trong semantic search/RAG, cosine similarity và dot product là phổ biến nhất.

Cần hiểu:

```text
Similarity cao không đảm bảo answer đúng.
Nó chỉ nói embedding gần nhau theo model.
```

Ví dụ lỗi:

```text
Query: "chính sách hoàn tiền"
Retrieved chunk: "chính sách thanh toán"

Semantic gần, nhưng evidence không đúng.
```

---

## 15. Reranking

Reranking là bước chấm lại các kết quả retrieved.

Pipeline:

```text
query
→ embedding retriever lấy top-50
→ reranker chấm lại top-50
→ chọn top-5 tốt nhất
```

Khác biệt:

|Retriever|Reranker|
|---|---|
|Nhanh|Chậm hơn|
|Dùng vector similarity|Đọc query-document kỹ hơn|
|Tốt để lọc candidate|Tốt để xếp hạng lại|
|Có thể miss fine-grained relevance|Bắt relevance chi tiết hơn|

Cần hiểu:

```text
Retriever tối ưu recall.
Reranker tối ưu precision/ranking quality.
```

---

## 16. RAG evaluation

RAG phải đánh giá theo từng tầng, không gộp một điểm duy nhất.

RAGAS docs nhấn mạnh đánh giá RAG nên tách từng component trong pipeline, với các metric như faithfulness, answer relevancy, context recall, context precision, context utilization. ([docs.ragas.io](https://docs.ragas.io/en/v0.1.21/concepts/metrics/ "Metrics"))

### 16.1 Retrieval-level metrics

|Metric|Câu hỏi|
|---|---|
|**Recall@k**|Top-k có chứa evidence đúng không?|
|**Precision@k**|Top-k có nhiều nhiễu không?|
|**MRR**|Evidence đúng nằm sớm hay muộn?|
|**NDCG**|Các chunk relevant có được xếp cao không?|
|**Context recall**|Retrieved context có bao phủ đủ thông tin cần thiết không?|
|**Context precision**|Relevant chunks có được xếp trên irrelevant chunks không?|

RAGAS stable docs định nghĩa context precision là metric đánh giá khả năng retriever xếp relevant chunks cao hơn irrelevant chunks trong retrieved context. ([docs.ragas.io](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/context_precision/ "Context Precision"))

---

### 16.2 Generation-level metrics

|Metric|Câu hỏi|
|---|---|
|**Answer correctness**|Answer có đúng không?|
|**Answer relevancy**|Answer có trả lời đúng câu hỏi không?|
|**Faithfulness**|Answer có bám vào context không?|
|**Groundedness**|Claim có được support bởi retrieved evidence không?|
|**Citation accuracy**|Citation có thật sự support câu được trích không?|
|**Hallucination rate**|Answer có thêm thông tin không có trong context không?|

RAGAS paper mô tả RAG evaluation khó vì cần xét nhiều chiều: retrieval có lấy đúng/focused context không, LLM có khai thác context một cách faithful không, và generation có chất lượng không. ([arXiv](https://arxiv.org/abs/2309.15217 "RAGAS: Automated Evaluation of Retrieval Augmented Generation"))

---

### 16.3 System-level metrics

|Metric|Ý nghĩa|
|---|---|
|**Latency**|Tổng thời gian trả lời|
|**Retrieval latency**|Thời gian search vector/rerank|
|**Generation latency**|Thời gian LLM generate|
|**Token usage**|Số input/output tokens|
|**Cost per query**|Chi phí mỗi câu hỏi|
|**Failure rate**|Tỷ lệ lỗi|
|**Timeout rate**|Tỷ lệ request quá lâu|
|**Context length usage**|Context có bị nhồi quá nhiều không|

---

## 17. LLM inference metrics

Vì roadmap có LLM Serving và GPU Inference, phần ML/DL basics nên có kiến thức inference metric.

NVIDIA NIM benchmarking docs định nghĩa TPS là tổng output tokens per second của hệ thống qua các concurrent requests; TPS tăng theo số request đến khi GPU compute bão hòa rồi có thể giảm. ([NVIDIA Docs](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html "Metrics — NVIDIA NIM LLMs Benchmarking"))

Cần học:

|Metric|Ý nghĩa|
|---|---|
|**TTFT**|Time To First Token: thời gian tới token đầu tiên|
|**TPOT**|Time Per Output Token: thời gian trung bình giữa các token output|
|**TPS / tokens/sec**|Throughput token|
|**End-to-end latency**|Tổng thời gian từ request tới complete answer|
|**Concurrency**|Số request xử lý đồng thời|
|**Batching**|Gom nhiều request để tăng throughput|
|**KV cache**|Cache key/value attention để generate nhanh hơn|
|**GPU memory**|Memory dùng bởi weights, activations, KV cache|
|**Cost per token/task**|Chi phí vận hành|

Cách hiểu:

```text
TTFT ảnh hưởng cảm giác "server phản hồi nhanh không".
TPOT ảnh hưởng tốc độ chữ chạy ra.
TPS ảnh hưởng throughput tổng hệ thống.
GPU memory quyết định batch/concurrency/context length.
```

---

## 18. Model capacity

Model capacity là khả năng biểu diễn/học pattern của model.

|Capacity thấp|Capacity cao|
|---|---|
|Dễ underfit|Dễ overfit nếu data ít|
|Nhanh, nhẹ|Chậm, tốn memory|
|Ít tham số|Nhiều tham số|
|Khó học pattern phức tạp|Học pattern phức tạp tốt hơn|

Ví dụ:

```text
Linear model cho ảnh phức tạp → capacity thấp.
Large Transformer cho dataset nhỏ → có thể overfit.
```

Cần hiểu:

```text
Model lớn không tự động tốt hơn.
Model lớn cần data, regularization, compute, và evaluation đúng.
```

---

## 19. Bias-variance intuition

Đây là khái niệm giúp hiểu overfit/underfit.

|Hiện tượng|Ý nghĩa|
|---|---|
|**High bias**|Model quá đơn giản, underfit|
|**High variance**|Model quá nhạy với train data, overfit|

Cách đọc:

```text
High bias:
train kém, validation kém.

High variance:
train tốt, validation kém.
```

---

## 20. Class imbalance

Class imbalance là khi các class không đều.

Ví dụ:

```text
98% normal
2% fraud
```

Vấn đề:

```text
Accuracy cao nhưng model không bắt được class hiếm.
```

Cần học:

|Khái niệm|Ý nghĩa|
|---|---|
|**Minority class**|Class ít mẫu|
|**Majority class**|Class nhiều mẫu|
|**Class weight**|Tăng trọng số loss cho class hiếm|
|**Oversampling**|Tăng sample class hiếm|
|**Undersampling**|Giảm sample class nhiều|
|**Threshold tuning**|Điều chỉnh ngưỡng decision|
|**PR-AUC**|Metric tốt hơn khi positive hiếm|
|**Recall/precision trade-off**|Bắt nhiều hơn thì có thể báo nhầm nhiều hơn|

---

## 21. Threshold

Nhiều model output probability/score, sau đó cần threshold để ra decision.

Ví dụ:

```text
score >= 0.5 → positive
score < 0.5 → negative
```

Nhưng threshold 0.5 không phải lúc nào cũng tốt.

Ví dụ fraud:

```text
threshold thấp → bắt nhiều fraud hơn, nhiều false positive hơn
threshold cao → ít false positive hơn, bỏ sót fraud nhiều hơn
```

Cần hiểu:

```text
Threshold quyết định precision-recall trade-off.
```

---

## 22. Calibration

Calibration là mức độ probability của model phản ánh đúng xác suất thật.

Ví dụ:

```text
Trong 100 case model dự đoán 0.8,
nếu khoảng 80 case đúng → model calibrated tốt.
```

Model có accuracy cao nhưng calibration kém vẫn nguy hiểm nếu dùng score để ra quyết định.

Dùng trong:

```text
risk scoring
medical prediction
fraud detection
confidence-based filtering
LLM routing
```

---

## 23. Distribution shift

Distribution shift là khi dữ liệu deploy khác dữ liệu train.

Ví dụ:

```text
Train: ảnh ban ngày
Deploy: ảnh ban đêm
```

```text
Train: query tiếng Anh chuẩn
Deploy: tiếng Việt + typo + teencode
```

```text
Train: PDF sạch
Deploy: scan OCR lỗi
```

Các dạng shift:

|Dạng|Ý nghĩa|
|---|---|
|**Covariate shift**|Input distribution thay đổi|
|**Label shift**|Tỷ lệ class thay đổi|
|**Concept drift**|Quan hệ input-label thay đổi|
|**Domain shift**|Domain deploy khác train|

Cần hiểu:

```text
Benchmark tốt không đảm bảo production tốt nếu distribution thay đổi.
```

---

## 24. Error analysis

Error analysis là phân tích model sai vì đâu.

Không được chỉ nói:

```text
Model yếu.
```

Phải tách lỗi:

### Classification error

```text
label sai
class imbalance
threshold sai
feature thiếu
overfit
underfit
distribution shift
```

### Retrieval error

```text
chunking sai
embedding không hợp domain
query quá mơ hồ
metadata filter sai
top-k quá nhỏ
reranker xếp sai
hard negative quá giống positive
```

### Generation error

```text
context thiếu
context nhiễu
prompt yếu
LLM hallucinate
citation không support answer
context quá dài làm model mất focus
```

### CV error

```text
object nhỏ
occlusion
lighting khác
motion blur
background bias
box label sai
domain shift
```

Cần học theo nguyên tắc:

```text
Mỗi lỗi phải map được về một nguyên nhân có thể kiểm tra.
```

---

## 25. Baseline

Baseline là phương pháp đơn giản để làm mốc.

Ví dụ:

```text
Classification baseline: logistic regression
Retrieval baseline: BM25
RAG baseline: embedding search only
CV baseline: pretrained detector
```

Ý nghĩa:

```text
Không có baseline thì không biết cải tiến có thật hay không.
```

---

## 26. Ablation

Ablation là bỏ từng thành phần để xem nó có tác dụng không.

Ví dụ RAG:

```text
Full system: Recall@5 = 88%
Without reranker: Recall@5 = 80%
Without query rewrite: Recall@5 = 83%
Without metadata filter: Recall@5 = 76%
```

Ý nghĩa:

```text
Ablation giúp biết thành phần nào thật sự đóng góp.
```

---

## 27. Fine-tuning vs RAG vs Prompting

Cần phân biệt rõ.

|Cách|Thay đổi gì|Khi dùng|
|---|---|---|
|**Prompting**|Không đổi weights|Muốn điều khiển behavior nhanh|
|**RAG**|Không đổi weights, thêm external context|Cần knowledge riêng, cập nhật, có citation|
|**Fine-tuning**|Update weights|Cần model học style/task/domain pattern|
|**Instruction tuning**|Fine-tune để làm theo instruction|Cần cải thiện instruction following|
|**LoRA/QLoRA**|Fine-tune nhẹ bằng adapter/quantization|Khi muốn tiết kiệm GPU/memory|

Cần hiểu:

```text
RAG tốt cho knowledge.
Fine-tuning tốt cho behavior/pattern.
Prompting tốt cho control nhẹ.
```

Không nên fine-tune chỉ để nhét tài liệu mới vào model nếu RAG giải quyết tốt hơn.

---

## 28. CV/DL basics liên quan hướng Vision

Vì roadmap có visual intelligence, VLM, image-text retrieval, document AI, video understanding, nên DL basics nên có thêm phần vision.

Cần học:

|Khái niệm|Ý nghĩa|
|---|---|
|**CNN**|Mạng học local visual pattern|
|**Convolution**|Kernel quét qua ảnh để trích feature|
|**Pooling**|Giảm kích thước feature map|
|**Backbone**|Mạng trích feature chính|
|**Detection head**|Phần predict class/box|
|**Bounding box**|Vị trí object|
|**IoU**|Độ overlap giữa predicted box và ground-truth box|
|**mAP**|Metric object detection|
|**ViT**|Transformer áp dụng cho image patches|
|**DETR-style detection**|Detection như set prediction|
|**CLIP/VLM**|Align image-text embedding|

---

## 29. VLM / multimodal basics

Cần học các khái niệm:

|Khái niệm|Ý nghĩa|
|---|---|
|**Image encoder**|Biến ảnh thành visual embedding|
|**Text encoder**|Biến text thành text embedding|
|**Shared embedding space**|Không gian chung để so sánh image/text|
|**Cross-modal retrieval**|Text tìm image hoặc image tìm text|
|**Visual grounding**|Gắn text phrase với vùng ảnh|
|**Image-text alignment**|Mức độ ảnh và text khớp nhau|
|**Hard negatives**|Caption/ảnh gần giống nhưng sai quan hệ|
|**Compositionality**|Hiểu tổ hợp object-attribute-relation|
|**Hallucination in VLM**|Model nói có object/quan hệ không có trong ảnh|

Liên hệ CLIP:

```text
image encoder → image embedding
text encoder → text embedding
contrastive objective → align image/text
```

---

# Bản gom ngắn để đưa vào bảng roadmap

|Nhóm base|Cần học gì|Mức cần đạt|
|---|---|---|
|**ML/DL basics**|ML loop, train/inference, data split, feature/representation, embedding, loss functions, optimization, overfit/underfit, generalization, regularization, metrics, retrieval metrics, RAG evaluation, error analysis, leakage, distribution shift|Hiểu được một AI pipeline theo chuỗi **data → representation → model → loss → optimization → validation → metric → failure analysis → inference behavior**; biết phân biệt lỗi do data, model, retrieval, generation, metric, serving hay distribution shift|

---

# Cấu trúc ghi chú

```text
ML/DL Basics

1. ML learning loop
2. Training vs inference
3. Data, label, feature, representation
4. Dataset split and leakage
5. Loss functions
   - cross-entropy
   - MSE/MAE
   - contrastive loss
   - triplet loss
   - ranking loss
6. Optimization
   - gradient descent
   - learning rate
   - optimizer
   - scheduler
7. Overfit, underfit, generalization
8. Regularization
9. Evaluation metrics
   - classification metrics
   - regression metrics
   - retrieval metrics
   - RAG metrics
   - inference metrics
10. Embedding and vector similarity
11. Retrieval and reranking
12. RAG evaluation
13. LLM inference metrics
14. Class imbalance and threshold
15. Calibration
16. Distribution shift
17. Error analysis
18. Baseline and ablation
19. Fine-tuning vs RAG vs prompting
20. CV/VLM basics
```

Đây là phần **kiến thức nền cần học kỹ**. Cấu trúc này tạo nền cho RAG, Agent, LLM Serving, CV/VLM hay GPU inference đều có nền để hiểu bản chất, không bị học keyword rời rạc.
