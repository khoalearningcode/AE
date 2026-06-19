Cái **Linux/Git/Docker** này là base cực quan trọng, thậm chí với AI Engineer còn quan trọng ngang Python engineering, vì sau này bạn sẽ liên tục gặp: server GPU, SSH, training remote, Docker deploy RAG, Git branch làm team, `.env`, port, volume, compose.

Nên hiểu dòng này như sau:

> **Linux/Git/Docker:** biết làm việc trên terminal/server, quản lý môi trường, code bằng Git đúng quy trình, đóng gói project bằng Docker, chạy nhiều service bằng Docker Compose.  
> **Mức đạt:** clone một repo AI/RAG/CV về server, setup env, chạy train/infer/API, debug lỗi cơ bản, version code bằng Git, dockerize được app và chạy lại được trên máy khác.

---

## 1. Linux / Terminal

Cần học:

|Mảng|Cần biết|Mức đạt|
|---|---|---|
|Terminal cơ bản|`cd`, `ls`, `pwd`, `mkdir`, `rm`, `cp`, `mv`, `cat`, `less`, `grep`, `find`|Di chuyển, tìm file, đọc log, sửa lỗi trên server được|
|File permission|`chmod`, `chown`, quyền `rwx`|Biết sửa lỗi permission khi chạy script/model|
|Process|`ps`, `top`, `htop`, `kill`, `nohup`, `tmux`|Biết xem tiến trình train đang chạy, kill job lỗi, giữ job chạy khi tắt SSH|
|Disk/GPU check|`df -h`, `du -sh`, `nvidia-smi`|Biết server đầy disk/GPU đang bị ai chiếm|
|Env variable|`export`, `.bashrc`, `PATH`, `CUDA_VISIBLE_DEVICES`|Biết set GPU, API key, path model|
|Package/system|`apt`, `wget`, `curl`, `unzip`, `tar`|Tải data/code/model trên server được|

Cái cần hiểu kỹ nhất:

```bash
cd
ls
pwd
cp
mv
rm
grep
find
df -h
du -sh
ps aux
kill -9
nvidia-smi
tmux
```

Đặc biệt với AI project, bạn phải quen:

```bash
nvidia-smi
CUDA_VISIBLE_DEVICES=0 python train.py
du -sh checkpoints/
df -h
tail -f logs/train.log
```

Mức đạt thực tế: SSH vào server GPU, tự tìm data ở đâu, chạy training, xem log, kiểm tra GPU/disk, kill process lỗi, không cần mở GUI.

---

## 2. SSH / Remote server

Cần học:

|Mảng|Cần biết|Mức đạt|
|---|---|---|
|SSH login|`ssh user@host`|Vào server được|
|SSH key|`ssh-keygen`, `~/.ssh/id_rsa.pub`, `authorized_keys`|Không cần nhập password mỗi lần|
|Copy file|`scp`, `rsync`|Đẩy code/data/checkpoint qua lại|
|Port forwarding|`ssh -L local:remote`|Mở Jupyter, FastAPI, W&B local từ server|
|Remote dev|VSCode Remote SSH|Code trực tiếp trên server được|

Cần hiểu kỹ đoạn này:

```bash
ssh minhkhoa@gpuserver1
scp file.zip minhkhoa@gpuserver1:/home/minhkhoa/
rsync -avP ./project/ minhkhoa@gpuserver1:/home/minhkhoa/project/
ssh -L 8888:localhost:8888 minhkhoa@gpuserver1
```

Mức đạt: bạn tự dùng server GPU mượt, không bị kẹt ở mấy lỗi như port, copy file, SSH key, process bị tắt khi disconnect.

---

## 3. Environment / Python env

Cái này nằm giữa Linux và Python.

Cần học:

|Mảng|Cần biết|Mức đạt|
|---|---|---|
|`venv` / `conda`|tạo env riêng cho mỗi project|Không cài bừa vào global Python|
|`pip`|install/uninstall/freeze|Tái tạo môi trường được|
|`requirements.txt`|lưu dependency|Người khác clone repo cài được|
|`.env`|lưu API key/path/config nhạy cảm|Không hard-code secret vào code|
|CUDA/PyTorch version|torch + CUDA match|Không bị lỗi GPU do cài sai bản|

Ví dụ cần quen:

```bash
conda create -n rag python=3.10
conda activate rag
pip install -r requirements.txt
pip freeze > requirements.txt
```

Hoặc:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Mức đạt: clone repo về, tạo env, cài dependency, chạy được project mà không phá môi trường khác.

---

## 4. Git

Git không chỉ là `git add . && git commit`. Cần học theo hướng làm project thật.

|Mảng|Cần biết|Mức đạt|
|---|---|---|
|Basic flow|`clone`, `status`, `add`, `commit`, `push`, `pull`|Lưu version code sạch|
|Branch|`branch`, `checkout`, `switch`, `merge`|Làm feature riêng không phá main|
|Conflict|merge conflict cơ bản|Biết sửa conflict khi làm team|
|Remote|GitHub, origin, upstream|Đồng bộ code local/server/GitHub|
|Ignore file|`.gitignore`|Không push data/checkpoint/env|
|Reset/revert|`reset`, `restore`, `revert`|Biết quay lại khi commit sai|
|Tag/release|`git tag`|Đánh dấu version model/experiment|

Cần hiểu kỹ nhất:

```bash
git clone <repo>
git status
git add .
git commit -m "message"
git push
git pull
git switch -c feature/rag-pipeline
git merge feature/rag-pipeline
```

Và `.gitignore` rất quan trọng với AI:

```bash
.venv/
__pycache__/
.env
data/
checkpoints/
outputs/
wandb/
*.pt
*.pth
*.ckpt
```

Mức đạt: bạn biết làm project theo branch, không phá code chính, không push data/checkpoint nặng lên GitHub.

---

## 5. Docker

Docker là để đóng gói môi trường chạy project. Với AI/RAG/deploy, nó rất quan trọng.

Cần học:

|Mảng|Cần biết|Mức đạt|
|---|---|---|
|Image/container|image là template, container là instance đang chạy|Hiểu Docker đang chạy cái gì|
|Dockerfile|`FROM`, `WORKDIR`, `COPY`, `RUN`, `CMD`|Tự viết Dockerfile cho FastAPI/RAG app|
|Build/run|`docker build`, `docker run`|Build image và chạy container được|
|Port mapping|`-p 8000:8000`|Mở API ra ngoài|
|Volume|`-v local:/app/data`|Mount data/model vào container|
|Env|`--env-file .env`|Đưa API key/config vào container|
|Logs/debug|`docker logs`, `docker exec`|Debug container lỗi|

Ví dụ Dockerfile cơ bản cho RAG/FastAPI:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

Chạy:

```bash
docker build -t rag-api .
docker run --env-file .env -p 8000:8000 rag-api
```

Mức đạt: bạn có thể dockerize một app RAG/API đơn giản để người khác chạy không cần setup thủ công.

---

## 6. Docker Compose

Compose dùng khi project có nhiều service. Ví dụ RAG thường có:

```text
FastAPI app + vector database + database + redis
```

Cần học:

|Mảng|Cần biết|Mức đạt|
|---|---|---|
|`docker-compose.yml`|định nghĩa nhiều service|Chạy cả stack bằng 1 lệnh|
|Service|app, db, vector_db|Biết service nào làm gì|
|Network|service gọi nhau bằng tên|App gọi `qdrant:6333`, không gọi localhost|
|Volume|lưu database/vector index|Restart không mất data|
|Env file|`.env`|Config gọn|

Ví dụ Compose cho RAG + Qdrant:

```yaml
services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      - qdrant

  qdrant:
    image: qdrant/qdrant
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage

volumes:
  qdrant_data:
```

Chạy:

```bash
docker compose up --build
docker compose down
docker compose logs -f app
```

Mức đạt: bạn chạy được một mini RAG stack bằng Docker Compose.

---

# Cái nào cần học kỹ nhất?

Với bạn, thứ tự ưu tiên nên là:

## Ưu tiên 1: Linux terminal + SSH

Vì bạn làm CV/RAG/research sẽ dùng server GPU rất nhiều. Không vững cái này là rất dễ bị kẹt ở lỗi nhỏ.

Cần đạt:

```text
SSH vào server
copy file
xem GPU
xem disk
chạy train bằng tmux
đọc log
kill process
set env variable
```

---

## Ưu tiên 2: Git branch workflow

Vì sau này làm research/project mà không version code thì rất dễ mất kiểm soát.

Cần đạt:

```text
main giữ code ổn định
feature branch để thử nghiệm
commit rõ ràng
.gitignore sạch
biết rollback khi sai
```

Ví dụ flow chuẩn:

```bash
git switch -c feature/add-reranker
# code
git add .
git commit -m "Add reranker module"
git push origin feature/add-reranker
```

---

## Ưu tiên 3: Dockerfile

Dockerfile giúp bạn biến project từ “chạy được trên máy tôi” thành “ai cũng chạy được”.

Cần đạt:

```text
viết Dockerfile cho FastAPI/RAG app
build image
run container
map port
mount volume
load .env
```

---

## Ưu tiên 4: Docker Compose

Compose để chạy nhiều service. RAG gần như chắc chắn sẽ đụng.

Cần đạt:

```text
app + qdrant/chroma + postgres/redis chạy chung bằng docker compose
biết xem logs
biết restart/down/up
biết volume lưu data
```

---

# Những cái chưa cần học quá sâu ngay

|Chưa cần sâu|Lý do|
|---|---|
|Bash scripting nâng cao|Biết cơ bản là đủ, sau này cần thì học|
|Kubernetes|Chưa cần cho intern/junior AI ngay|
|Docker networking nâng cao|Biết service name/port là đủ|
|CI/CD phức tạp|Sau khi Git/Docker ổn rồi học|
|Linux admin sâu|Không cần thành sysadmin|
|Git rebase nâng cao|Biết merge là đủ giai đoạn đầu|

---

# Mini project nên làm để chứng minh đã đạt base này

Làm một project nhỏ:

```text
rag-docker-demo/
  app/
    main.py
    rag_pipeline.py
  configs/
    default.yaml
  data/
    docs/
  Dockerfile
  docker-compose.yml
  requirements.txt
  .env.example
  .gitignore
  README.md
```

Yêu cầu chạy được:

```bash
git clone <repo>
cd rag-docker-demo
docker compose up --build
```

Sau đó mở:

```text
http://localhost:8000/docs
```

Và có endpoint:

```text
POST /ask
```

Như vậy là bạn đã cover đủ:

```text
Linux terminal
Git
env
Dockerfile
Docker Compose
RAG app structure
```

---

Tóm lại, dòng roadmap này nên viết lại thành:

> **Linux/Git/Docker:** sử dụng terminal/server qua SSH, quản lý process/log/disk/GPU/env, dùng Git branch workflow sạch, viết Dockerfile để đóng gói app, dùng Docker Compose để chạy app + database/vector database.  
> **Mức đạt:** tự clone một repo AI lên server, setup env, chạy train/infer/API, debug lỗi cơ bản, quản lý version bằng Git, và dockerize được một mini RAG/FastAPI project chạy lại được trên máy khác.