
import numpy as np
import logging
from functools import lru_cache
from contextlib import asynccontextmanager

from fastapi import APIRouter, Depends, HTTPException, Body
from fastapi import Header, Query
from starlette.concurrency import run_in_threadpool

from settings import settings
from schemas import PredictRequest, PredictResponse, LabelScore, Top5Response

from drivers.onnx_driver  import OnnxDriver
from drivers.torch_driver import TorchDriver

logger = logging.getLogger("app.api")

# driver registry
if settings.device.type == "cuda":
  from drivers.trt_driver import TRTDriver
  DRIVER_REGISTRY = {
    "onnx":     OnnxDriver,
    "torch":    TorchDriver,
    "trt_fp16": TRTDriver,
  }
else:
  DRIVER_REGISTRY = {
    "onnx":  OnnxDriver,
    "torch": TorchDriver,
  }

# instantiate one driver object per backend at startup
_drivers = { name: DriverCls() for name, DriverCls in DRIVER_REGISTRY.items() }

# lifespan (startup/shutdown)
@asynccontextmanager
async def lifespan(app):
  for name, driver in _drivers.items():
    logger.info(f"Loading model driver: {name}")
    driver.load()
  yield
  # optional: driver teardown if needed

# dependency to resolve model name
@lru_cache(maxsize=None)
def resolve_model(
  x_model: str | None = Header(
    default=None,
    alias="X-Model-Name",
    description="Which model driver to use: 'onnx', 'torch', or 'trt_fp16'",
    example="torch",
  )
) -> str:
  name = x_model or settings.default_model
  if name not in _drivers:
    raise HTTPException(404, f"Unknown model '{name}'")
  return name

# router & endpoints
router = APIRouter()

@router.get("/health")
async def health():
    return {"status": "ok"}

# predict endpoint
@router.post("/predict", response_model=PredictResponse)
async def predict(
    payload: PredictRequest,
    model_name: str = Depends(resolve_model),
):
  """
  - **model_name** in the `X-Model-Name` header
  - **text** in the JSON body
  """
  driver = _drivers[model_name]

  # run model and flatten
  raw = await run_in_threadpool(driver.predict, payload.text)
  if isinstance(raw, list) and isinstance(raw[0], (list, tuple)):
      probs = raw[0]
  else:
      probs = raw

  # normalize id2label
  raw_map = driver.config.id2label
  id2label = {int(k): v for k, v in raw_map.items()}

  # find the highest-scoring index
  top_idx = int(np.argmax(probs))
  top_score = float(probs[top_idx])
  top_label = id2label[top_idx]

  # return only the top‐1
  top = LabelScore(code=top_idx, label=top_label, score=top_score)
  return PredictResponse(model=model_name, category=top)

# top‐5 endpoint
@router.post("/top5", response_model=Top5Response)
async def top5(
    payload: PredictRequest,
    model_name: str = Depends(resolve_model),
):
  """
  - **model_name** in the `X-Model-Name` header
  - **text** in the JSON body
  """
  driver = _drivers[model_name]

  # run model and flatten
  raw = await run_in_threadpool(driver.predict, payload.text)
  if isinstance(raw, list) and isinstance(raw[0], (list, tuple)):
      probs = raw[0]
  else:
      probs = raw

  # normalize id2label
  raw_map = driver.config.id2label
  id2label = {int(k): v for k, v in raw_map.items()}

  # find the top‐5 indices
  top_idx = np.argsort(probs)[::-1][:5]
  top_scores = [float(probs[i]) for i in top_idx]
  top_labels = [id2label[i] for i in top_idx]

  # return only the top‐5
  top = [LabelScore(code=i, label=l, score=s) for i, l, s in zip(top_idx, top_labels, top_scores)]
  return Top5Response(model=model_name, top5=top)
