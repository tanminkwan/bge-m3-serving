import logging
import re
import time

import numpy as np

from .config import DUMMY_DIM, DUMMY_MODE, GPU_DEVICE_ID, MAX_LENGTH, MODEL_PATH, USE_GPU

logger = logging.getLogger(__name__)

WORD_PATTERN = re.compile(r"\w+", re.UNICODE)


class EmbeddingModel:
    def __init__(self):
        if DUMMY_MODE:
            logger.info("DUMMY_MODE enabled, skipping model load")
            self.session = None
            self.tokenizer = None
            return

        logger.info("Loading tokenizer from %s", MODEL_PATH)
        try:
            from transformers import AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            logger.info("Tokenizer loaded (vocab_size=%d)", self.tokenizer.vocab_size)
        except Exception:
            logger.exception("Failed to load tokenizer from %s", MODEL_PATH)
            raise

        if USE_GPU:
            providers = [("CUDAExecutionProvider", {"device_id": GPU_DEVICE_ID}), "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        logger.info("Loading ONNX model from %s/model.onnx (providers=%s, gpu_device_id=%d)", MODEL_PATH, providers, GPU_DEVICE_ID)
        try:
            import onnxruntime as ort

            self.session = ort.InferenceSession(f"{MODEL_PATH}/model.onnx", providers=providers)
            active_providers = self.session.get_providers()
            logger.info("ONNX model loaded (active_providers=%s)", active_providers)
            if USE_GPU and "CUDAExecutionProvider" not in active_providers:
                logger.warning("GPU requested but CUDAExecutionProvider not active. Falling back to CPU.")
        except Exception:
            logger.exception("Failed to load ONNX model from %s/model.onnx", MODEL_PATH)
            raise

    def encode(self, texts: list[str]) -> list[list[float]]:
        text_lengths = [len(t) for t in texts]
        logger.info("encode called: count=%d, text_lengths=%s", len(texts), text_lengths)

        if DUMMY_MODE:
            rng = np.random.default_rng(hash(tuple(texts)) % 2**32)
            vecs = rng.standard_normal((len(texts), DUMMY_DIM))
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
            logger.info("Dummy encode done: count=%d, dim=%d", len(texts), DUMMY_DIM)
            return vecs.tolist()

        t0 = time.perf_counter()
        inputs = self.tokenizer(
            texts, padding=True, truncation=True, max_length=MAX_LENGTH, return_tensors="np"
        )
        token_counts = inputs["attention_mask"].sum(axis=1).tolist()
        t_tok = time.perf_counter()
        logger.info("Tokenized: token_counts=%s (%.3fs)", token_counts, t_tok - t0)

        outputs = self.session.run(None, dict(inputs))
        t_inf = time.perf_counter()
        logger.info("Inference done: output_shape=%s (%.3fs)", outputs[0].shape, t_inf - t_tok)

        # mean pooling
        token_embeddings = outputs[0]
        mask = inputs["attention_mask"][..., np.newaxis]
        embeddings = (token_embeddings * mask).sum(axis=1) / mask.sum(axis=1)
        # normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms

        t_total = time.perf_counter() - t0
        logger.info("Encode complete: count=%d, dim=%d, total=%.3fs", len(texts), embeddings.shape[1], t_total)
        return embeddings.tolist()

    def extract_unknown_terms(
        self,
        texts: list[str],
        min_pieces: int = 4,
        pieces_per_char: float = 0.6,
    ) -> list[dict]:
        if DUMMY_MODE or self.tokenizer is None:
            return [{"text": t, "unknown_terms": []} for t in texts]

        unk_id = self.tokenizer.unk_token_id
        results = []
        for text in texts:
            seen: set[str] = set()
            unknown: list[dict] = []
            for word in WORD_PATTERN.findall(text):
                if word in seen:
                    continue
                seen.add(word)
                ids = self.tokenizer.encode(word, add_special_tokens=False)
                if not ids:
                    continue
                reasons = []
                if unk_id is not None and unk_id in ids:
                    reasons.append("unk")
                ratio = len(ids) / max(len(word), 1)
                if len(ids) >= min_pieces:
                    reasons.append("fragmented_count")
                if len(ids) >= 2 and ratio >= pieces_per_char:
                    reasons.append("fragmented_density")
                if reasons:
                    unknown.append({
                        "term": word,
                        "reasons": reasons,
                        "pieces": self.tokenizer.convert_ids_to_tokens(ids),
                        "num_pieces": len(ids),
                    })
            logger.info("extract_unknown_terms: text_len=%d, unknown_count=%d", len(text), len(unknown))
            results.append({"text": text, "unknown_terms": unknown})
        return results


model = EmbeddingModel()
