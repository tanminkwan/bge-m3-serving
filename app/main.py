import logging

from fastapi import FastAPI, Request
from pydantic import BaseModel

from .config import DUMMY_DIM, DUMMY_MODE, MAX_LENGTH, MODEL_PATH, PORT, USE_GPU
from .model import model

logger = logging.getLogger(__name__)

app = FastAPI(title="Embedding Service")


@app.on_event("startup")
def log_startup_config():
    logger.info(
        "Starting with config: MODEL_PATH=%s, USE_GPU=%s, DUMMY_MODE=%s, DUMMY_DIM=%d, MAX_LENGTH=%d, PORT=%d",
        MODEL_PATH, USE_GPU, DUMMY_MODE, DUMMY_DIM, MAX_LENGTH, PORT,
    )


class EmbedRequest(BaseModel):
    texts: list[str]


class EmbedResponse(BaseModel):
    embeddings: list[list[float]]
    dim: int
    count: int


class UnknownTermsRequest(BaseModel):
    texts: list[str]
    min_pieces: int = 4
    pieces_per_char: float = 0.6


class UnknownTerm(BaseModel):
    term: str
    reasons: list[str]
    pieces: list[str]
    num_pieces: int


class UnknownTermsItem(BaseModel):
    text: str
    unknown_terms: list[UnknownTerm]


class UnknownTermsResponse(BaseModel):
    results: list[UnknownTermsItem]
    count: int


@app.get("/health")
def health():
    return {"status": "ok", "dummy_mode": DUMMY_MODE, "model_path": MODEL_PATH}


@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest, request: Request):
    if not req.texts:
        client = request.client.host if request.client else "unknown"
        logger.warning("Empty texts received from %s", client)
        return EmbedResponse(embeddings=[], dim=0, count=0)

    try:
        vectors = model.encode(req.texts)
    except Exception:
        logger.exception("Encode failed: count=%d, text_lengths=%s", len(req.texts), [len(t) for t in req.texts])
        raise

    return EmbedResponse(
        embeddings=vectors,
        dim=len(vectors[0]),
        count=len(vectors),
    )


@app.post("/unknown-terms", response_model=UnknownTermsResponse)
def unknown_terms(req: UnknownTermsRequest, request: Request):
    if not req.texts:
        client = request.client.host if request.client else "unknown"
        logger.warning("Empty texts received from %s", client)
        return UnknownTermsResponse(results=[], count=0)

    try:
        results = model.extract_unknown_terms(
            req.texts,
            min_pieces=req.min_pieces,
            pieces_per_char=req.pieces_per_char,
        )
    except Exception:
        logger.exception("extract_unknown_terms failed: count=%d", len(req.texts))
        raise

    return UnknownTermsResponse(results=results, count=len(results))
