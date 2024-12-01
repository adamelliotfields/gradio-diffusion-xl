import os
from importlib import import_module
from importlib.util import find_spec
from types import SimpleNamespace
from warnings import filterwarnings

from diffusers import (
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)
from diffusers.utils import logging as diffusers_logging
from transformers import logging as transformers_logging

# Improved GPU handling and progress bars; set before importing spaces
os.environ["ZEROGPU_V2"] = "1"

# Use Rust-based downloader; errors if enabled and not installed
if find_spec("hf_transfer"):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

filterwarnings("ignore", category=FutureWarning, module="diffusers")
filterwarnings("ignore", category=FutureWarning, module="transformers")

diffusers_logging.set_verbosity_error()
transformers_logging.set_verbosity_error()

# Standard refiner structure
_sdxl_refiner_files = [
    "scheduler/scheduler_config.json",
    "text_encoder_2/config.json",
    "text_encoder_2/model.fp16.safetensors",
    "tokenizer_2/merges.txt",
    "tokenizer_2/special_tokens_map.json",
    "tokenizer_2/tokenizer_config.json",
    "tokenizer_2/vocab.json",
    "unet/config.json",
    "unet/diffusion_pytorch_model.fp16.safetensors",
    "vae/config.json",
    "vae/diffusion_pytorch_model.fp16.safetensors",
    "model_index.json",
]

# Standard SDXL structure
_sdxl_files = [
    *_sdxl_refiner_files,
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
]

_sdxl_files_with_vae = [*_sdxl_files, "vae_1_0/config.json"]

# Using namespace instead of dataclass for simplicity
Config = SimpleNamespace(
    HF_TOKEN=os.environ.get("HF_TOKEN", None),
    ZERO_GPU=import_module("spaces").config.Config.zero_gpu,
    PIPELINES={
        "txt2img": StableDiffusionXLPipeline,
        "img2img": StableDiffusionXLImg2ImgPipeline,
    },
    MODEL="segmind/Segmind-Vega",
    MODELS=[
        "cyberdelia/CyberRealsticXL",
        "fluently/Fluently-XL-Final",
        "segmind/Segmind-Vega",
        "SG161222/RealVisXL_V5.0",
        "stabilityai/stable-diffusion-xl-base-1.0",
    ],
    HF_REPOS={
        "ai-forever/Real-ESRGAN": ["RealESRGAN_x2.pth", "RealESRGAN_x4.pth"],
        "cyberdelia/CyberRealsticXL": ["CyberRealisticXLPlay_V1.0.safetensors"],
        "fluently/Fluently-XL-Final": ["FluentlyXL-Final.safetensors"],
        "madebyollin/sdxl-vae-fp16-fix": ["config.json", "diffusion_pytorch_model.fp16.safetensors"],
        "segmind/Segmind-Vega": _sdxl_files,
        "SG161222/RealVisXL_V5.0": ["RealVisXL_V5.0_fp16.safetensors"],
        "stabilityai/stable-diffusion-xl-base-1.0": _sdxl_files_with_vae,
        "stabilityai/stable-diffusion-xl-refiner-1.0": _sdxl_refiner_files,
    },
    SINGLE_FILE_MODELS=[
        "cyberdelia/cyberrealsticxl",
        "fluently/fluently-xl-final",
        "sg161222/realvisxl_v5.0",
    ],
    VAE_MODEL="madebyollin/sdxl-vae-fp16-fix",
    REFINER_MODEL="stabilityai/stable-diffusion-xl-refiner-1.0",
    SCHEDULER="Euler",
    SCHEDULERS={
        "DDIM": DDIMScheduler,
        "DEIS 2M": DEISMultistepScheduler,
        "DPM++ 2M": DPMSolverMultistepScheduler,
        "Euler": EulerDiscreteScheduler,
        "Euler a": EulerAncestralDiscreteScheduler,
    },
    WIDTH=1024,
    HEIGHT=1024,
    NUM_IMAGES=1,
    SEED=-1,
    GUIDANCE_SCALE=6,
    INFERENCE_STEPS=40,
    DEEPCACHE_INTERVAL=1,
    SCALE=1,
    SCALES=[1, 2, 4],
)
