# Python Engineering — Knowledge cần học kỹ

## Status

Stage: 00 Base nền tảng  
Current level: concept → coding practice  
Last updated: 2026-06-25

## Must understand

- Project structure cho repo AI/RAG/CV.
- Package, module, import và cách chạy bằng `python -m`.
- Function/class boundary trong pipeline.
- Typing, config, logging, error handling.
- CLI script, environment và test cơ bản.

## Must practice

- Refactor một notebook thành package nhỏ trong `src/`.
- Tạo CLI bằng `argparse` hoặc `typer`.
- Đưa config ra YAML/env thay vì hard-code.
- Thêm logging và error handling cho data/model pipeline.
- Viết test cho chunking, preprocessing hoặc scoring function.

## Can explain when ready

- Vì sao project AI không nên nằm trong một notebook duy nhất?
- Khi nào dùng function, khi nào dùng class?
- Vì sao config/logging quan trọng hơn `print` và hard-code path?

---

| Nhóm                  | Cần hiểu kỹ                                                                             | Mức cần đạt thực tế                                                           |
| --------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Project structure** | Chia code thành `src/`, `configs/`, `scripts/`, `tests/`, `data/`, `notebooks/`         | Nhìn repo là biết file nào làm gì, không nhét toàn bộ vào 1 notebook          |
| **Package / import**  | `module`, `package`, `__init__.py`, relative/absolute import, chạy bằng `python -m`     | Không còn lỗi import lung tung khi chạy từ terminal                           |
| **Function design**   | Tách hàm rõ input/output, không viết hàm quá dài, tránh global variable                 | Một pipeline train/eval/RAG có thể chia thành nhiều hàm sạch                  |
| **Class cơ bản**      | Khi nào dùng class, `__init__`, method, inheritance nhẹ                                 | Biết viết class cho `Dataset`, `Retriever`, `LLMClient`, `Config`, `Pipeline` |
| **Typing**            | `str`, `int`, `list[str]`, `dict`, `Optional`, `Union`, `dataclass`, `TypedDict` cơ bản | Người khác đọc code hiểu dữ liệu đi qua pipeline là gì                        |
| **Config**            | `.env`, YAML/JSON config, argparse/typer, không hard-code path/model/key                | Chạy được kiểu `python -m scripts.train --config configs/rag.yaml`            |
| **Logging**           | `logging` thay vì `print`, log level: info/warning/error, ghi log ra file               | Khi training/indexing lỗi thì biết lỗi ở bước nào                             |
| **Error handling**    | `try/except`, custom error đơn giản, validate input                                     | Code không crash ngu khi thiếu file, sai path, API lỗi                        |
| **CLI**               | `argparse` hoặc `typer`                                                                 | Biến notebook thành script chạy được bằng terminal                            |
| **File I/O**          | `pathlib`, đọc ghi JSON/CSV/YAML, pickle/parquet cơ bản                                 | Làm data pipeline sạch, không hard-code đường dẫn Windows/Linux               |
| **Testing nhẹ**       | `pytest`, test function quan trọng                                                      | Biết test hàm chunking, retrieval, scoring, preprocessing                     |
| **Async cơ bản**      | `async/await`, khi nào dùng cho API/request I/O                                         | Đủ hiểu khi làm LLM API, crawler, RAG service; chưa cần quá sâu               |
| **Environment**       | `venv/conda`, `pip`, `requirements.txt`, `.gitignore`                                   | Clone repo về cài được, không phụ thuộc máy cá nhân                           |

Phần cần hiểu **kỹ nhất** là 5 phần này:

**1. Package + import + project structure**

Đây là phần thường gây lỗi trong project AI. Làm AI project mà không hiểu phần này thì repo rất dễ nát. Cần biết vì sao nên viết:

```bash
project/
  src/
    rag/
      retriever.py
      chunker.py
      pipeline.py
  configs/
    rag.yaml
  scripts/
    build_index.py
    query.py
  tests/
```

Thay vì để:

```bash
final_final_test_v3.ipynb
rag_utils_copy.py
new_test2.py
```

Mức đạt: có thể tự tạo một repo RAG nhỏ, chạy được bằng terminal, import không lỗi.

---

**2. Function/class design**

Phần này quan trọng hơn syntax nâng cao. Cần biết chia:

```python
def load_documents(path): ...
def chunk_documents(docs): ...
def build_vector_index(chunks): ...
def retrieve(query, index): ...
def generate_answer(query, context): ...
```

Không viết một file 500 dòng từ trên xuống dưới.

Class thì chưa cần OOP quá sâu. Chỉ cần biết dùng class khi có **state/config/client/model**. Ví dụ:

```python
class Retriever:
    def __init__(self, embedding_model, vector_db):
        self.embedding_model = embedding_model
        self.vector_db = vector_db

    def search(self, query: str, top_k: int = 5):
        ...
```

Mức đạt: nhìn một pipeline AI và biết phần nào nên là function, phần nào nên là class.

---

**3. Config + logging + error handling**

Đây là nhóm năng lực phân biệt code thử nghiệm và code có thể vận hành.

Không nên viết:

```python
model_name = "gpt-4"
path = "C:/Users/Khoa/Desktop/data"
top_k = 5
```

Mà nên đưa vào config:

```yaml
model_name: gpt-4
top_k: 5
chunk_size: 512
data_path: data/raw
```

Logging cũng vậy. Không chỉ `print("done")`, mà cần:

```python
logger.info("Loaded %d documents", len(docs))
logger.warning("Empty document found: %s", path)
logger.error("Failed to build index", exc_info=True)
```

Mức đạt: khi code lỗi, log cho biết lỗi nằm ở bước nào.

---

**4. CLI/script hóa notebook**

Với AI Engineer, notebook chỉ nên dùng để thử nghiệm; project thật cần được script hóa.

Ví dụ cần chạy được:

```bash
python -m scripts.build_index --config configs/rag.yaml
python -m scripts.query --question "What is RAG?"
python -m scripts.evaluate --dataset data/eval.json
```

Mức đạt: refactor notebook thí nghiệm thành script sạch.

---

**5. Typing vừa đủ**

Typing dùng để làm rõ logic, cấu trúc dữ liệu và tự kiểm tra các giả định quan trọng.

Ví dụ:

```python
def chunk_text(text: str, chunk_size: int) -> list[str]:
    ...
```

Hoặc:

```python
from dataclasses import dataclass

@dataclass
class Document:
    id: str
    text: str
    source: str
```

Mức đạt: function quan trọng đều có type hint, data object quan trọng có `dataclass` hoặc schema rõ ràng.

---


> Có thể tự build một repo AI hoàn chỉnh từ đầu: đọc data → xử lý → chạy model/API → lưu kết quả → evaluate → log lỗi → chạy bằng CLI → người khác clone về chạy được.

Cụ thể, sau phần Python engineering, nên làm được mini project kiểu:

```bash
rag-mini/
  src/
    ragmini/
      config.py
      loader.py
      chunker.py
      embedder.py
      vectorstore.py
      generator.py
      pipeline.py
  configs/
    default.yaml
  scripts/
    build_index.py
    ask.py
    evaluate.py
  tests/
    test_chunker.py
  requirements.txt
  README.md
```

Chạy được:

```bash
python -m scripts.build_index --config configs/default.yaml
python -m scripts.ask --question "Explain attention"
```

Các phần **chưa cần học sâu ngay**:

|Chưa cần ưu tiên|Vì sao|
|---|---|
|Metaclass|Hiếm dùng trong AI project thường ngày|
|Decorator nâng cao|Biết dùng cơ bản là đủ|
|Async sâu / event loop sâu|Chỉ cần hiểu khi gọi API song song|
|Multiprocessing nâng cao|Để sau khi cần optimize data pipeline|
|Design pattern nặng|Dễ học lan man|
|Package publishing lên PyPI|Chưa cần cho giai đoạn này|

Dòng năng lực chuẩn hóa:

> **Python engineering:** biết tổ chức repo, chia module/package, viết function/class sạch, dùng typing/config/logging/error handling, tạo CLI script, quản lý môi trường, test cơ bản.  
> **Mức đạt:** biến được notebook AI/RAG/CV thành một project chạy được bằng terminal, có cấu trúc rõ, dễ debug, dễ tái sử dụng, người khác clone về chạy được.

Đây là nền tảng dùng lại khi làm RAG, Agent, CV training, research code, deploy API đều đụng lại đúng mấy thứ này.
