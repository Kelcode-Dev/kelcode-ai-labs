import os
import torch
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from pathlib import Path

class Settings(BaseSettings):
  # model configurations
  cuda_availalbe: bool = torch.cuda.is_available()
  model_base_path: Path = Path("models")
  model_onnx_path: str = os.path.join(model_base_path, "onnx")
  model_torch_path: str = os.path.join(model_base_path, "torch")
  model_trt_fp16_path: str = os.path.join(model_base_path, "trt_fp16")
  model_onnx_name: str = "model.onnx"
  model_torch_name: str = "model.pt"
  model_trt_fp16_name: str = "model_fp16.engine"
  model_torch_device: str = "cpu"
  model_onnx_device: str = "cpu"
  model_trt_fp16_device: str = ("cuda" if cuda_availalbe else "cpu")

  # CORS settings
  cors_urls: list[str]  = Field(
    default_factory=lambda: [
    "http://localhost:8080",
    "https://headlinecats.ai.kelcode.co.uk",
    "https://headlinecats.ai.dev.kelcode.co.uk",
    ],
    env="CORS_URLS",
  )
  cors_allow_origins:     list[str] = cors_urls
  cors_allow_methods:     list[str] = ["GET", "POST"]
  cors_allow_headers:     list[str] = ["*"]
  cors_allow_credentials: bool      = True

  # tell Pydantic where to read env vars from (optional)
  model_config = SettingsConfigDict(
    env_file=".env",
    env_file_encoding="utf-8"
  )

  # app metadata
  app_title:   str = "Kelcode Quantised Model Server"
  app_version: str = "1.0.0"
  app_description: str = "FastAPI server for switching between quantized ONNX, TorchScript, and TensorRT models."

  # determine the device to run inference on
  @property
  def device(self) -> torch.device:
    # Evaluates once per access; returns cuda if available, else cpu
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# instantiate once for the app
settings = Settings()
