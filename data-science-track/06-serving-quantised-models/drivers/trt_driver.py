import os
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
from transformers import AutoTokenizer, AutoConfig
from drivers.base import ModelDriver
from settings import settings
from pathlib import Path

class TRTDriver(ModelDriver):
  def __init__(self):
    # pull the directory straight from settings
    super().__init__(settings.model_trt_fp16_path)

  def _load_model(self) -> None:
    # shared metadata
    self.config    = AutoConfig   .from_pretrained(settings.model_trt_fp16_path)
    self.tokenizer = AutoTokenizer.from_pretrained(settings.model_trt_fp16_path)

    # initialize CUDA + TRT
    cuda.init()
    device = cuda.Device(0)
    self.ctx = device.make_context()
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    model_location = Path(os.path.join(settings.model_trt_fp16_path, settings.model_trt_fp16_name)).resolve()
    with open(model_location, "rb") as f:
      runtime = trt.Runtime(TRT_LOGGER)
      engine  = runtime.deserialize_cuda_engine(f.read())
    self.context = engine.create_execution_context()
    self.engine  = engine

  def _preprocess(self, text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    toks = self.tokenizer(
      text,
      return_tensors="np",
      padding="max_length",
      truncation=True,
      max_length=128
    )
    # TRT often expects int32
    input_ids      = toks["input_ids"     ].astype(np.int32)
    attention_mask = toks["attention_mask"].astype(np.int32)
    token_type_ids = toks.get(
      "token_type_ids",
      np.zeros_like(input_ids, dtype=np.int32)
    )
    return input_ids, attention_mask, token_type_ids

  def _infer(self, inputs: tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:

    self.ctx.push()
    try:
      input_ids, attention_mask, token_type_ids = inputs

      # 1) set dynamic shapes if supported
      self.context.set_input_shape("input_ids",      input_ids.shape)
      self.context.set_input_shape("attention_mask", attention_mask.shape)
      self.context.set_input_shape("token_type_ids", token_type_ids.shape)

      # 2) allocate & copy inputs
      bindings       = []
      device_buffers = []
      for arr in (input_ids, attention_mask, token_type_ids):
        d_in = cuda.mem_alloc(arr.nbytes)
        cuda.memcpy_htod(d_in, arr)
        bindings.append(int(d_in))
        device_buffers.append(d_in)

      # 3) prepare output buffer
      out_shape = self.context.get_tensor_shape("logits")
      output    = np.empty(out_shape, dtype=np.float32)
      d_out     = cuda.mem_alloc(output.nbytes)
      bindings.append(int(d_out))
      device_buffers.append(d_out)

      # 4) launch inference
      self.context.execute_v2(bindings)

      # 5) copy back
      cuda.memcpy_dtoh(output, d_out)

      return output
    finally:
      self.ctx.pop()
