import numpy as np
import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForSequenceClassification
from drivers.base import ModelDriver
from settings import settings
from pathlib import Path

import logging
logger = logging.getLogger(__name__)

class TorchDriver(ModelDriver):
  def __init__(self):
    # pull the directory straight from settings
    super().__init__(settings.model_torch_path)

  def _load_model(self) -> None:
    self.config = AutoConfig.from_pretrained(settings.model_torch_path)
    self.tokenizer = AutoTokenizer.from_pretrained(settings.model_torch_path)

    pt_path = Path(settings.model_torch_path) / settings.model_torch_name
    model = torch.load(
      pt_path,
      map_location=settings.model_torch_device,
      weights_only=False   # allow full-module unpickling
    )

    model.to(settings.model_torch_device)
    model.eval()
    self.model = model

  def _preprocess(self, text: str) -> dict:
    toks = self.tokenizer(
      text,
      return_tensors="pt",
      padding=True,
      truncation=True,
      max_length=128
    )
    return {k: v.to(settings.model_torch_device) for k, v in toks.items()}

  def _infer(self, inputs: dict) -> "np.ndarray":
    with torch.no_grad():
      outputs = self.model(**inputs)
      logits  = outputs.logits
    return logits.cpu().numpy()
