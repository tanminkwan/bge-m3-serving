# Embedding Service

BAAI/bge-m3 모델을 ONNX Runtime + GPU로 서빙하는 FastAPI 웹 서비스.

## 목차

- [사전 요구사항](#사전-요구사항)
- [모델 다운로드 (인터넷 되는 환경)](#모델-다운로드-인터넷-되는-환경)
- [오프라인 환경으로 전달](#오프라인-환경으로-전달)
- [Docker 이미지 구조](#docker-이미지-구조)
- [최초 배포](#최초-배포)
- [코드 수정 후 재배포](#코드-수정-후-재배포)
- [Production 실행](#production-실행)
- [로컬 개발/테스트](#로컬-개발테스트)
- [API](#api)
- [환경변수](#환경변수)
- [트러블슈팅](#트러블슈팅)

---

## 사전 요구사항

### Production 서버 (GPU 환경)

- NVIDIA GPU (Compute Capability 7.0 이상)
- NVIDIA Driver **560.28 이상** (CUDA 12.6 지원에 필요)
- Docker Engine 20.10 이상
- NVIDIA Container Toolkit (nvidia-docker)

드라이버 버전 확인:

```bash
nvidia-smi
# 오른쪽 상단의 "Driver Version"과 "CUDA Version" 확인
# Driver Version >= 560.28 이어야 함
```

NVIDIA Container Toolkit 확인:

```bash
nvidia-container-cli info
# 정상 출력되면 설치됨
```

NVIDIA Container Toolkit이 없으면 설치:

```bash
# Ubuntu
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 빌드 환경 (인터넷 되는 환경)

- Docker Engine 20.10 이상
- Python 3.10 이상 (모델 다운로드용)
- 디스크 여유 공간 10GB 이상

---

## 모델 다운로드 (인터넷 되는 환경)

### 1. huggingface_hub 설치

```bash
pip install huggingface_hub
```

### 2. 모델 다운로드

```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="./bge-m3",
    allow_patterns=["onnx/*", "*.json"]
)
```

### 3. 불필요한 캐시 삭제

```bash
rm -rf ./bge-m3/.cache
```

### 4. 다운로드 확인

다음 파일들이 있어야 정상:

```
bge-m3/
├── onnx/
│   ├── model.onnx          # (~708KB) ONNX 모델 구조
│   ├── model.onnx_data     # (~2.2GB) 모델 가중치
│   ├── config.json
│   ├── sentencepiece.bpe.model
│   ├── special_tokens_map.json
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   └── Constant_7_attr__value
├── config.json
├── tokenizer.json
├── special_tokens_map.json
├── tokenizer_config.json
├── config_sentence_transformers.json
├── modules.json
├── sentence_bert_config.json
└── 1_Pooling/
    └── config.json
```

특히 `onnx/model.onnx_data` (~2.2GB)가 있는지 반드시 확인.

---

## 오프라인 환경으로 전달

### 모델 아카이브

```bash
# 묶기만 (압축 없음, 빠름 - ONNX 바이너리는 압축 효과 거의 없음)
tar cf bge-m3.tar bge-m3/

# 압축 필요 시 (느리지만 약간 작아짐)
tar czf bge-m3.tar.gz bge-m3/
```

### 오프라인 환경에서 압축 해제

```bash
# tar의 경우
tar xf bge-m3.tar

# tar.gz의 경우
tar xzf bge-m3.tar.gz
```

해제 후 `bge-m3/onnx/model.onnx_data` 파일이 ~2.2GB인지 확인.
0바이트이거나 누락되면 모델이 정상 동작하지 않음.

---

## Docker 이미지 구조

이미지를 **base**와 **app** 두 단계로 분리하여, 코드 수정 시 대용량 파일을 다시 전달하지 않도록 함.

```
embedding-server-base  (~4GB, 최초 1회만 전달)
├── nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04
├── python3, pip
└── pip 패키지 (onnxruntime-gpu, transformers, fastapi, uvicorn, numpy)

embedding-server  (수MB, 코드 변경 시마다 전달)
├── FROM embedding-server-base
└── app/ (Python 소스코드)
```

### 관련 파일

- `Dockerfile.base` : base 이미지 빌드용 (CUDA + pip 패키지)
- `Dockerfile` : app 이미지 빌드용 (Python 소스코드만)
- `requirements.txt` : pip 패키지 목록

### 재빌드 기준

| 변경 사항 | 재빌드 대상 | production 전달 크기 |
|---|---|---|
| Python 소스코드 수정 (`app/`) | app 이미지만 | **수KB** (app.tar) |
| pip 패키지 추가/변경 (`requirements.txt`) | base + app 둘 다 | ~2GB (base 이미지) |
| CUDA/cuDNN 버전 변경 | base + app 둘 다 | ~2GB (base 이미지) |

---

## 최초 배포

최초에는 base 이미지, app 소스, 모델 파일 모두 전달해야 함.

### 빌드 환경 (인터넷 되는 곳)에서 수행

#### 1. base 이미지 빌드

```bash
docker build -f Dockerfile.base -t embedding-server-base .
```

빌드 확인:

```bash
docker images embedding-server-base
# SIZE ~4.05GB 이면 정상
```

#### 2. base 이미지 저장

```bash
docker save embedding-server-base | gzip > embedding-server-base.tar.gz
```

- 시간이 걸림 (4GB+ 압축)
- 진행 중 `ls -al`로 보면 파일 크기가 0으로 보일 수 있음 (정상, 파일 쓰기 완료 전)
- 진행 중 실제 크기 확인: `du -h embedding-server-base.tar.gz`
- 완료 후 파일 크기: ~2GB 내외

#### 3. app 이미지 빌드

```bash
docker build -t embedding-server .
```

빌드 확인 (수초 내 완료):

```bash
docker images embedding-server
```

#### 4. production으로 전달할 파일 준비

app 이미지는 `docker save`로 저장하면 base 레이어까지 포함되어 4GB가 됨.
따라서 **app 소스코드와 Dockerfile만 직접 전달**하고, production에서 빌드함.

```bash
tar cf app.tar app/ Dockerfile
```

#### 5. production으로 파일 전달

USB, SCP 등으로 다음 파일들을 전달:

| 파일 | 크기 | 설명 |
|---|---|---|
| `embedding-server-base.tar.gz` | ~2GB | base Docker 이미지 (CUDA + pip 패키지) |
| `app.tar` | **수KB** | Python 소스코드 + Dockerfile |
| `bge-m3.tar` 또는 `bge-m3.tar.gz` | ~2.2GB | ONNX 모델 파일 |
| `docker-compose.yml` | 수KB | Docker Compose 설정 |

#### 6. production 서버에서 수행

```bash
# 1. 작업 디렉토리 생성 및 이동
mkdir -p /path/to/deploy && cd /path/to/deploy

# 2. 전달받은 파일들을 작업 디렉토리에 복사 후 압축 해제

# 2-1. base 이미지 로드 (최초 1회)
docker load < embedding-server-base.tar.gz

# 로드 확인
docker images embedding-server-base
# REPOSITORY                TAG       SIZE
# embedding-server-base     latest    ~4.05GB

# 2-2. app 소스코드 압축 해제
tar xf app.tar

# 2-3. 모델 압축 해제
tar xf bge-m3.tar
# 또는
tar xzf bge-m3.tar.gz

# 모델 파일 확인 (model.onnx_data가 ~2.2GB인지 반드시 확인)
ls -lh bge-m3/onnx/model.onnx_data

# 2-4. docker-compose.yml 복사 (이미 작업 디렉토리에 있어야 함)

# 3. app 이미지 빌드 (수초 내 완료)
docker build -t embedding-server .

# 빌드 확인
docker images embedding-server

# 4. 실행
docker compose up -d

# 5. 정상 기동 확인 (아래 "Production 실행" 섹션 참고)
```

최종 디렉토리 구조:

```
/path/to/deploy/
├── docker-compose.yml
├── Dockerfile
├── app/
│   ├── __init__.py (없어도 됨)
│   ├── config.py
│   ├── model.py
│   └── main.py
└── bge-m3/
    └── onnx/
        ├── model.onnx
        ├── model.onnx_data
        ├── config.json
        ├── tokenizer.json
        ├── tokenizer_config.json
        ├── special_tokens_map.json
        ├── sentencepiece.bpe.model
        └── Constant_7_attr__value
```

---

## 코드 수정 후 재배포

Python 소스코드(`app/`)만 수정한 경우, **base 이미지는 이미 production에 있으므로 다시 전달할 필요 없음.**

### 빌드 환경에서 수행

```bash
# 1. 코드 수정 후 app.tar만 다시 만듦
tar cf app.tar app/ Dockerfile
```

### production으로 전달

- `app.tar` (수KB) 1개만 전달

### production 서버에서 수행

```bash
cd /path/to/deploy

# 1. 기존 컨테이너 중지
docker compose down

# 2. 기존 app 소스 삭제 후 새 소스 압축 해제
rm -rf app/
tar xf app.tar

# 3. app 이미지 재빌드 (수초 내 완료, base 이미지 위에 app만 얹음)
docker build -t embedding-server .

# 4. 실행
docker compose up -d

# 5. 정상 기동 확인
docker logs embedding-server
curl http://localhost:8000/health
```

### requirements.txt가 변경된 경우

pip 패키지가 추가/변경되면 base 이미지를 다시 빌드하고 전달해야 함.

```bash
# 빌드 환경에서
docker build -f Dockerfile.base -t embedding-server-base .
docker save embedding-server-base | gzip > embedding-server-base.tar.gz

# production으로 embedding-server-base.tar.gz 전달 후
docker load < embedding-server-base.tar.gz
docker build -t embedding-server .
docker compose up -d
```

---

## Production 실행

### docker-compose.yml 설정

```yaml
services:
  embedding:
    container_name: embedding-server
    image: embedding-server
    restart: unless-stopped
    ports:
      - "8000:8000"
    volumes:
      - ./bge-m3/onnx:/model
    environment:
      - MODEL_PATH=/model
      - USE_GPU=true
      - GPU_DEVICE_ID=0
      - DUMMY_MODE=false
      - DUMMY_DIM=1024
      - MAX_LENGTH=8192
      - LOG_LEVEL=INFO
      - TRANSFORMERS_OFFLINE=1
      - HF_HUB_OFFLINE=1
    deploy:
      resources:
        reservations:
          devices:
            - capabilities: [gpu]
```

### 실행

```bash
docker compose up -d
```

### 정상 기동 확인

```bash
# 1. 컨테이너 상태 확인
docker ps
# STATUS가 "Up"이어야 함. "Restarting"이면 로그 확인

# 2. 로그 확인
docker logs embedding-server
# 다음 로그가 순서대로 나와야 정상:
# [INFO] app.model: Loading tokenizer from /model
# [INFO] app.model: Tokenizer loaded (vocab_size=250002)
# [INFO] app.model: Loading ONNX model from /model/model.onnx (providers=...)
# [INFO] app.model: ONNX model loaded (active_providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
# [INFO] app.main: Starting with config: ...
# INFO: Uvicorn running on http://0.0.0.0:8000

# 주의: active_providers에 CUDAExecutionProvider가 없으면 GPU를 사용하지 않는 것임!
# "GPU requested but CUDAExecutionProvider not active" 경고가 나오면 트러블슈팅 참고

# 3. Health Check
curl http://localhost:8000/health
# {"status":"ok","dummy_mode":false,"model_path":"/model"}

# 4. 실제 임베딩 테스트
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["테스트 문장입니다"]}'
# embeddings, dim=1024, count=1 이 응답되면 정상
```

### 중지

```bash
docker compose down
```

### GPU 선택 (멀티 GPU 환경)

GPU가 4장인 경우, `GPU_DEVICE_ID`로 사용할 GPU를 지정:

```yaml
environment:
  - GPU_DEVICE_ID=0   # 첫 번째 GPU (0, 1, 2, 3 중 선택)
```

GPU 번호 확인:

```bash
nvidia-smi
# GPU 0, 1, 2, 3의 모델명, 메모리 사용량 확인 가능
```

### docker run으로 직접 실행하는 경우

```bash
docker run -d \
  --name embedding-server \
  --restart unless-stopped \
  --gpus all \
  -p 8000:8000 \
  -v /path/to/bge-m3/onnx:/model \
  -e MODEL_PATH=/model \
  -e USE_GPU=true \
  -e GPU_DEVICE_ID=0 \
  -e DUMMY_MODE=false \
  -e MAX_LENGTH=8192 \
  -e LOG_LEVEL=INFO \
  -e TRANSFORMERS_OFFLINE=1 \
  -e HF_HUB_OFFLINE=1 \
  embedding-server
```

---

## 로컬 개발/테스트

GPU 없는 환경에서 더미 모드로 테스트 가능. 소스 코드 변경 없이 환경변수만 다름.

### 설치

```bash
pip install -r requirements.txt
```

### 더미 모드 실행 (모델 파일 불필요)

```bash
DUMMY_MODE=true USE_GPU=false uvicorn app.main:app --port 8000
```

더미 모드에서는 실제 모델 없이 랜덤 벡터(1024차원)를 반환.
동일 입력에 대해 동일 벡터를 반환하므로 API 연동 테스트에 적합.

### 실제 모델로 로컬 실행 (GPU 있는 경우)

```bash
USE_GPU=true MODEL_PATH=./bge-m3/onnx uvicorn app.main:app --port 8000
```

### Docker 더미 모드 테스트

```bash
docker run -d \
  --name embedding-test \
  -p 8000:8000 \
  -e DUMMY_MODE=true \
  -e USE_GPU=false \
  embedding-server

# 테스트
curl http://localhost:8000/health
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world"]}'

# 정리
docker stop embedding-test && docker rm embedding-test
```

---

## API

### GET /health

서버 상태 확인.

```bash
curl http://localhost:8000/health
```

응답:

```json
{
  "status": "ok",
  "dummy_mode": false,
  "model_path": "/model"
}
```

### POST /embed

텍스트 목록을 받아 임베딩 벡터를 반환.

```bash
curl -X POST http://localhost:8000/embed \
  -H "Content-Type: application/json" \
  -d '{"texts": ["hello world", "안녕하세요"]}'
```

요청:

```json
{
  "texts": ["hello world", "안녕하세요"]
}
```

응답:

```json
{
  "embeddings": [[0.019, 0.026, ...], [0.012, -0.031, ...]],
  "dim": 1024,
  "count": 2
}
```

- `embeddings`: 각 텍스트에 대한 1024차원 정규화된 벡터
- `dim`: 벡터 차원 (bge-m3는 1024)
- `count`: 입력 텍스트 수
- 빈 배열 `{"texts": []}` 입력 시 `{"embeddings": [], "dim": 0, "count": 0}` 반환

---

## 환경변수

| 변수 | 기본값 | 설명 |
|---|---|---|
| `MODEL_PATH` | `./bge-m3/onnx` | 모델 디렉토리 경로. Docker에서는 볼륨 마운트 경로 (예: `/model`) |
| `USE_GPU` | `true` | GPU 사용 여부. `true`/`false` |
| `GPU_DEVICE_ID` | `0` | 사용할 GPU 번호 (0부터 시작). 멀티 GPU 환경에서 선택 |
| `DUMMY_MODE` | `false` | 더미 모드. `true`면 모델 로드 없이 랜덤 벡터 반환 |
| `DUMMY_DIM` | `1024` | 더미 모드에서 반환할 벡터 차원 |
| `MAX_LENGTH` | `8192` | 토크나이저 최대 토큰 수. 이 값을 초과하는 입력은 잘림(truncation). bge-m3의 모델 한계가 8192이므로 이보다 크게 설정해도 의미 없음. 짧은 문서만 처리하는 경우 줄이면 메모리/속도 절약 가능 |
| `HOST` | `0.0.0.0` | 서버 바인드 주소 |
| `PORT` | `8000` | 서버 포트 |
| `LOG_LEVEL` | `INFO` | 로그 레벨. `DEBUG`, `INFO`, `WARNING`, `ERROR` 중 선택 |
| `TRANSFORMERS_OFFLINE` | (미설정) | `1`로 설정하면 transformers가 인터넷 접근을 시도하지 않음. 오프라인 환경 필수 |
| `HF_HUB_OFFLINE` | (미설정) | `1`로 설정하면 huggingface_hub가 인터넷 접근을 시도하지 않음. 오프라인 환경 필수 |

---

## 트러블슈팅

### GPU를 사용하지 않고 CPU로 fallback되는 경우

로그에 다음 메시지가 보이면:

```
GPU requested but CUDAExecutionProvider not active. Falling back to CPU.
```

원인과 해결:

1. **NVIDIA Driver 버전이 낮음**
   - `nvidia-smi`로 확인. Driver Version이 560.28 미만이면 드라이버 업그레이드 필요

2. **NVIDIA Container Toolkit 미설치**
   - `nvidia-container-cli info` 실행. 에러 나면 위 사전 요구사항의 설치 가이드 참고

3. **docker-compose에 GPU 설정 누락**
   - `deploy.resources.reservations.devices` 섹션이 있는지 확인
   - `docker run`의 경우 `--gpus all` 플래그 필요

4. **cuDNN 미포함 이미지 사용**
   - base 이미지가 `nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04` 기반인지 확인
   - `python:3.10-slim` 등 다른 이미지에는 cuDNN이 없어서 GPU 사용 불가

### 모델 로드 실패

```
Failed to load tokenizer from /model
```

- 볼륨 마운트 확인: `docker exec embedding-server ls -la /model/`
- `tokenizer.json`, `config.json` 등이 보여야 함
- 비어있으면 호스트의 모델 경로가 잘못된 것

```
Failed to load ONNX model from /model/model.onnx
```

- `model.onnx`와 `model.onnx_data`가 모두 있어야 함
- `model.onnx_data`가 0바이트면 모델 전달 과정에서 손상된 것. 다시 복사

### PyTorch warning

```
PyTorch was not found. Models won't be available...
```

무시해도 됨. 이 서비스는 ONNX Runtime으로 추론하므로 PyTorch 불필요. tokenizer만 transformers에서 사용.

### 컨테이너가 계속 재시작되는 경우

```bash
docker logs embedding-server
```

로그 마지막 부분에서 에러 원인 확인. 주로:
- 모델 파일 누락 (볼륨 마운트 문제)
- GPU 관련 라이브러리 누락
- 메모리 부족 (OOM)

### 오프라인 환경에서 시작 시 느림/타임아웃

`TRANSFORMERS_OFFLINE=1`과 `HF_HUB_OFFLINE=1`이 설정되어 있는지 확인.
미설정 시 transformers가 인터넷에 접속 시도하다가 타임아웃까지 대기함.

### app 이미지 빌드 시 "embedding-server-base not found" 에러

```
ERROR: pull access denied for embedding-server-base
```

production 서버에 base 이미지가 로드되지 않은 상태.

```bash
# base 이미지 로드 확인
docker images embedding-server-base

# 없으면 로드
docker load < embedding-server-base.tar.gz
```
