# BGE-M3 Sparse 출력 추가 계획

현재 `app/model.py`는 BGE-M3의 dense(cosine용) 임베딩만 반환한다. Sparse(lexical weights)를 추가하기 위한 두 가지 접근을 정리한다.

## 현재 상태

- `bge-m3/onnx/model.onnx`는 `XLMRobertaModel` 백본만 export된 상태 (config.json `architectures: ["XLMRobertaModel"]`).
- ONNX 그래프 출력은 `last_hidden_state` 한 개. BGE-M3의 `sparse_linear`(Linear(1024, 1)) 및 ColBERT 헤드는 포함되지 않음.
- `encode()`는 mean pooling + L2 normalize 후 dense 벡터만 반환.

## 옵션 A — sparse_linear 헤드를 Python에서 적용

ONNX는 그대로 두고, sparse_linear 가중치만 별도로 로드해 numpy로 적용.

### 절차
1. HuggingFace `BAAI/bge-m3` 저장소에서 `sparse_linear.pt`를 받아 `bge-m3/sparse_linear.pt`로 저장.
   - shape: weight `[1, 1024]`, bias `[1]`.
2. `EmbeddingModel.__init__`에서 torch 없이 numpy로 변환해 보관.
   - 예: `torch.load`로 읽고 `.numpy()` 후 weight/bias를 인스턴스 속성으로 저장. 또는 사전에 `.npz`로 변환해 두면 torch 의존성 제거 가능.
3. `encode()`에서 이미 받아둔 `outputs[0]` (last_hidden_state, shape `[B, T, 1024]`)에 적용:
   - `token_weights = relu(hidden @ W.T + b)` → shape `[B, T, 1]` → squeeze.
   - `attention_mask`로 패딩 마스킹.
   - 특수 토큰(CLS=0, SEP=2, PAD=1, UNK=3) 제외.
   - 같은 `input_id`가 여러 위치에 있으면 **max** 값으로 집약 → `{token_id: weight}` dict.
4. API 변경
   - `EmbedResponse`에 `sparse_embeddings: list[dict[int, float]]` 필드 추가.
   - `model.encode()`가 dense, sparse 튜플을 반환하도록 시그니처 변경.

### 장단점
- 장점: ONNX 재export 불필요. 변경량 작음. dense 경로 영향 없음.
- 단점: hidden_state를 CPU numpy로 끌어와 곱하는 단계 추가. GPU 사용 시 H2D/D2H 비용 발생. ColBERT까지 추가하면 동일한 패턴이 또 필요.

## 옵션 B — ONNX 재export (장기 운영 권장)

sparse_linear(필요하면 ColBERT까지)를 포함해 ONNX를 다시 export. 한 번의 추론으로 dense + sparse를 모두 얻음.

### 절차
1. PyTorch + `FlagEmbedding` + `optimum`/`onnx` 호환 환경 구성.
2. `BGEM3FlagModel`의 forward를 wrapping하는 `nn.Module`을 만들어 다음을 출력하도록 한다:
   - dense embedding (CLS 또는 mean pooling 후 normalize)
   - sparse token weights (relu(sparse_linear(hidden)))
   - (선택) ColBERT vectors (colbert_linear(hidden) 후 normalize)
3. `torch.onnx.export` 또는 `optimum.exporters.onnx`로 export. opset/dynamic_axes 설정 시 batch와 seq_len을 동적으로.
4. 새 모델 파일을 `bge-m3/onnx/`에 교체 (또는 별도 경로). `app/model.py`는 출력 인덱스를 늘려 받기만 하면 됨.
5. sparse 후처리(특수 토큰 제외, token_id별 max 집약)는 여전히 Python에서 수행.
   - 또는 후처리도 그래프에 ScatterND/SegmentMax 등으로 넣을 수 있으나 이식성·디버깅 측면에서 보통 Python에 두는 편이 낫다.

### 장단점
- 장점: dense/sparse(/ColBERT)가 한 번의 GPU 추론에 끝남. ONNX 그래프 최적화(fusion, fp16, TensorRT)에 sparse 경로 포함. 모델 파일 자기-완결적.
- 단점: export 환경 세팅 비용. PyTorch/transformers/optimum 버전 호환 이슈 가능. 모델 파일 사이즈 약간 증가(매우 작음).

## 권장

- 단기 실험/기능 검증용: **옵션 A**.
- 운영/장기 사용, 또는 ColBERT까지 같이 쓸 계획: **옵션 B**로 한 번에 정리.

## 참고

- BGE-M3 원본 sparse 처리 로직: `FlagEmbedding/BGE_M3/modeling.py`의 `_process_token_weights` 등.
- sparse_linear는 `nn.Linear(hidden_size=1024, out_features=1)`. weight·bias 모두 사용.
- 특수 토큰 ID는 `tokenizer.special_tokens_map`/`tokenizer_config.json`로 확인.
- 응답 포맷 예: `{"token_id": weight}` dict. Milvus/Vespa/Qdrant 등 sparse 인덱스 입력에 그대로 사용 가능.
