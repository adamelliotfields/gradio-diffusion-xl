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

Config = SimpleNamespace(
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
    MODEL="fluently/Fluently-XL-Final",
    MODELS=[
        # TODO: CyberRealisticXL once single file support is added
        "cagliostrolab/animagine-xl-3.1",
        "fluently/Fluently-XL-Final",
        "SG161222/RealVisXL_V5.0",
        "stabilityai/stable-diffusion-xl-base-1.0",
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
    STYLE="sai-enhance",
    WIDTH=896,
    HEIGHT=1152,
    NUM_IMAGES=1,
    SEED=-1,
    GUIDANCE_SCALE=6,
    INFERENCE_STEPS=35,
    DEEPCACHE_INTERVAL=1,
    SCALE=1,
    SCALES=[1, 2, 4],
)
