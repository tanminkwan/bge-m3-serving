import logging
import os

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    level=getattr(logging, LOG_LEVEL, logging.INFO),
)

MODEL_PATH = os.getenv("MODEL_PATH", "./bge-m3/onnx")
DUMMY_MODE = os.getenv("DUMMY_MODE", "false").lower() == "true"
DUMMY_DIM = int(os.getenv("DUMMY_DIM", "1024"))  # bge-m3 output dim
USE_GPU = os.getenv("USE_GPU", "true").lower() == "true"
GPU_DEVICE_ID = int(os.getenv("GPU_DEVICE_ID", "0"))
MAX_LENGTH = int(os.getenv("MAX_LENGTH", "8192"))
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
