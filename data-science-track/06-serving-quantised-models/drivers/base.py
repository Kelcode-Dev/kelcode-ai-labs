from abc import ABC, abstractmethod
from pathlib import Path
from transformers import AutoConfig, AutoTokenizer
import numpy as np
import torch
import onnxruntime as ort
import tensorrt as trt  # if you’re using TRT Python API…

class ModelDriver(ABC):
  def __init__(self, model_dir: str):
    self.model_dir = Path(model_dir)

  @abstractmethod
  def _load_model(self) -> None:
    """Load the actual model artifact (ONNX, TorchScript, TRT engine…)."""
    raise NotImplementedError("Subclasses must implement `_load_model`")

  @abstractmethod
  def _preprocess(self, text: str) -> dict:
    """Turn `text` into model inputs (e.g. numpy arrays or torch tensors)."""
    raise NotImplementedError("Subclasses must implement `_preprocess`")

  @abstractmethod
  def _infer(self, inputs: dict) -> np.ndarray:
    """Run the model and return raw logits as a NumPy array."""
    raise NotImplementedError("Subclasses must implement `_infer`")

  def load(self) -> None:
    """Template method: loads shared metadata, then concrete model."""
    # load tokenizer & config
    self.config = AutoConfig.from_pretrained(self.model_dir)
    self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

    # load the runtime engine
    self._load_model()

  def predict(self, text: str) -> list[float]:
    """Generic predict wrapper: tokenize, infer, softmax."""
    inputs = self._preprocess(text)
    logits = self._infer(inputs)
    # convert logits to probabilities
    exp = np.exp(logits - logits.max())
    probs = (exp / exp.sum()).tolist()
    return probs
