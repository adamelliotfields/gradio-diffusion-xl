import os
from importlib import import_module
from types import SimpleNamespace

from diffusers import (
    DDIMScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    StableDiffusionXLImg2ImgPipeline,
    StableDiffusionXLPipeline,
)

# improved GPU handling and progress bars; set before importing spaces
os.environ["ZEROGPU_V2"] = "true"

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

_sdxl_files = [
    *_sdxl_refiner_files,
    "text_encoder/config.json",
    "text_encoder/model.fp16.safetensors",
    "tokenizer/merges.txt",
    "tokenizer/special_tokens_map.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer/vocab.json",
]

Config = SimpleNamespace(
    HF_TOKEN=os.environ.get("HF_TOKEN", None),
    CIVIT_TOKEN=os.environ.get("CIVIT_TOKEN", None),
    ZERO_GPU=import_module("spaces").config.Config.zero_gpu,
    MONO_FONTS=["monospace"],
    SANS_FONTS=[
        "sans-serif",
        "Apple Color Emoji",
        "Segoe UI Emoji",
        "Segoe UI Symbol",
        "Noto Color Emoji",
    ],
    PIPELINES={
        "txt2img": StableDiffusionXLPipeline,
        "img2img": StableDiffusionXLImg2ImgPipeline,
    },
    HF_MODELS={
        "segmind/Segmind-Vega": [*_sdxl_files],
        "stabilityai/stable-diffusion-xl-base-1.0": [*_sdxl_files, "vae_1_0/config.json"],
        "stabilityai/stable-diffusion-xl-refiner-1.0": [*_sdxl_refiner_files],
    },
    MODEL="segmind/Segmind-Vega",
    MODELS=[
        "cagliostrolab/animagine-xl-3.1",
        "cyberdelia/CyberRealsticXL",
        "fluently/Fluently-XL-Final",
        "segmind/Segmind-Vega",
        "SG161222/RealVisXL_V5.0",
        "stabilityai/stable-diffusion-xl-base-1.0",
    ],
    MODEL_CHECKPOINTS={
        # keep keys lowercase
        "cagliostrolab/animagine-xl-3.1": "animagine-xl-3.1.safetensors",
        "cyberdelia/cyberrealsticxl": "CyberRealisticXLPlay_V1.0.safetensors",  # typo in "realistic"
        "fluently/fluently-xl-final": "FluentlyXL-Final.safetensors",
        "sg161222/realvisxl_v5.0": "RealVisXL_V5.0_fp16.safetensors",
    },
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
    STYLE="enhance",
    WIDTH=1024,
    HEIGHT=1024,
    NUM_IMAGES=1,
    SEED=-1,
    GUIDANCE_SCALE=7.5,
    INFERENCE_STEPS=40,
    DEEPCACHE_INTERVAL=1,
    SCALE=1,
    SCALES=[1, 2, 4],
)
