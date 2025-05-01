# drivers/driver_onnx.py

import numpy as np
from onnxruntime import InferenceSession
from transformers import AutoTokenizer, AutoConfig
from drivers.base import ModelDriver
from settings import settings

class OnnxDriver(ModelDriver):
  def __init__(self):
    # pull the directory straight from settings
    super().__init__(settings.model_onnx_path)

  def _load_model(self) -> None:
    # shared metadata
    self.config    = AutoConfig   .from_pretrained(settings.model_onnx_path)
    self.tokenizer = AutoTokenizer.from_pretrained(settings.model_onnx_path)

    # ONNX session on CPU
    model_path = str(self.model_dir / "model.onnx")
    self.session = InferenceSession(
      model_path,
      providers=["CPUExecutionProvider"]
    )

  def _preprocess(self, text: str) -> dict:
    toks = self.tokenizer(
      text,
      return_tensors="np",
      padding=True,
      truncation=True,
      max_length=128
    )
    # ensure int64 for ONNX
    input_ids      = toks["input_ids"].astype(np.int64)
    attention_mask = toks["attention_mask"].astype(np.int64)
    # some HF tokenizers don’t emit token_type_ids
    token_type_ids = toks.get(
      "token_type_ids",
      np.zeros_like(input_ids, dtype=np.int64)
    )
    # map to the ONNX input names
    inputs = {
      self.session.get_inputs()[0].name: input_ids,
      self.session.get_inputs()[1].name: attention_mask,
      self.session.get_inputs()[2].name: token_type_ids,
    }
    return inputs

  def _infer(self, inputs: dict) -> "np.ndarray":
    # returns [batch_size, num_labels]
    logits = self.session.run(None, inputs)[0]
    return logits
