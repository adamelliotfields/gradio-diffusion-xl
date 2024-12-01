import time
from datetime import datetime

import torch
from compel import Compel, ReturnedEmbeddingsType
from compel.prompt_parser import PromptParser
from spaces import GPU

from .config import Config
from .loader import Loader
from .logger import Logger
from .utils import cuda_collect, safe_progress, timer


# Dynamic signature for the GPU duration function; max 60s per image
def gpu_duration(**kwargs):
    loading = 15
    duration = 15
    width = kwargs.get("width", 1024)
    height = kwargs.get("height", 1024)
    scale = kwargs.get("scale", 1)
    num_images = kwargs.get("num_images", 1)
    use_refiner = kwargs.get("use_refiner", False)
    size = width * height
    if use_refiner:
        loading += 10
    if size > 1_100_000:
        duration += 5
    if size > 1_600_000:
        duration += 5
    if scale == 2:
        duration += 5
    if scale == 4:
        duration += 10
    return loading + (duration * num_images)


@GPU(duration=gpu_duration)
def generate(
    positive_prompt,
    negative_prompt="",
    seed=None,
    model="stabilityai/stable-diffusion-xl-base-1.0",
    scheduler="Euler",
    width=1024,
    height=1024,
    guidance_scale=6.0,
    inference_steps=40,
    deepcache=1,
    scale=1,
    num_images=1,
    use_karras=False,
    use_refiner=False,
    Error=Exception,
    Info=None,
    progress=None,
):
    KIND = "txt2img"
    CURRENT_STEP = 0
    CURRENT_IMAGE = 1
    EMBEDDINGS_TYPE = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED

    start = time.perf_counter()
    log = Logger("generate")
    log.info(f"Generating {num_images} image{'s' if num_images > 1 else ''}...")

    if Config.ZERO_GPU:
        safe_progress(progress, 100, 100, "ZeroGPU init")

    if not torch.cuda.is_available():
        raise Error("CUDA not available")

    # https://pytorch.org/docs/stable/generated/torch.manual_seed.html
    if seed is None or seed < 0:
        seed = int(datetime.now().timestamp() * 1e6) % (2**64)

    # custom progress bar for multiple images
    def callback_on_step_end(pipeline, step, timestep, latents):
        nonlocal CURRENT_IMAGE, CURRENT_STEP
        if progress is not None:
            # calculate total steps for img2img based on denoising strength
            strength = 1
            total_steps = min(int(inference_steps * strength), inference_steps)

            # if steps are different we're in the refiner
            refining = False
            if CURRENT_STEP == step:
                CURRENT_STEP = step + 1
            else:
                refining = True
                CURRENT_STEP += 1
            progress(
                (CURRENT_STEP, total_steps),
                desc=f"{'Refining' if refining else 'Generating'} image {CURRENT_IMAGE}/{num_images}",
            )
        return latents

    loader = Loader()
    loader.load(
        KIND,
        model,
        scheduler,
        deepcache,
        scale,
        use_karras,
        use_refiner,
        progress,
    )

    refiner = loader.refiner
    pipeline = loader.pipeline
    upscaler = loader.upscaler

    if pipeline is None:
        raise Error(f"Error loading {model}")

    # prompt embeds for base and refiner
    compel_1 = Compel(
        text_encoder=[pipeline.text_encoder, pipeline.text_encoder_2],
        tokenizer=[pipeline.tokenizer, pipeline.tokenizer_2],
        requires_pooled=[False, True],
        returned_embeddings_type=EMBEDDINGS_TYPE,
        dtype_for_device_getter=lambda _: pipeline.dtype,
        device=pipeline.device,
    )
    compel_2 = Compel(
        text_encoder=[pipeline.text_encoder_2],
        tokenizer=[pipeline.tokenizer_2],
        requires_pooled=[True],
        returned_embeddings_type=EMBEDDINGS_TYPE,
        dtype_for_device_getter=lambda _: pipeline.dtype,
        device=pipeline.device,
    )

    images = []
    current_seed = seed
    safe_progress(progress, 0, num_images, f"Generating image 0/{num_images}")

    for i in range(num_images):
        try:
            generator = torch.Generator(device=pipeline.device).manual_seed(current_seed)
            conditioning_1, pooled_1 = compel_1([positive_prompt, negative_prompt])
            conditioning_2, pooled_2 = compel_2([positive_prompt, negative_prompt])
        except PromptParser.ParsingException:
            raise Error("Invalid prompt")

        # refiner expects latents; upscaler expects numpy array
        pipe_output_type = "pil"
        refiner_output_type = "pil"
        if use_refiner:
            pipe_output_type = "latent"
            if scale > 1:
                refiner_output_type = "np"
        else:
            if scale > 1:
                pipe_output_type = "np"

        pipe_kwargs = {
            "width": width,
            "height": height,
            "denoising_end": 0.8 if use_refiner else None,
            "generator": generator,
            "output_type": pipe_output_type,
            "guidance_scale": guidance_scale,
            "num_inference_steps": inference_steps,
            "prompt_embeds": conditioning_1[0:1],
            "pooled_prompt_embeds": pooled_1[0:1],
            "negative_prompt_embeds": conditioning_1[1:2],
            "negative_pooled_prompt_embeds": pooled_1[1:2],
        }

        refiner_kwargs = {
            "denoising_start": 0.8,
            "generator": generator,
            "output_type": refiner_output_type,
            "guidance_scale": guidance_scale,
            "num_inference_steps": inference_steps,
            "prompt_embeds": conditioning_2[0:1],
            "pooled_prompt_embeds": pooled_2[0:1],
            "negative_prompt_embeds": conditioning_2[1:2],
            "negative_pooled_prompt_embeds": pooled_2[1:2],
        }

        if progress is not None:
            pipe_kwargs["callback_on_step_end"] = callback_on_step_end
            refiner_kwargs["callback_on_step_end"] = callback_on_step_end

        try:
            image = pipeline(**pipe_kwargs).images[0]
            if use_refiner:
                refiner_kwargs["image"] = image
                image = refiner(**refiner_kwargs).images[0]
            images.append((image, str(current_seed)))
            current_seed += 1
        finally:
            CURRENT_STEP = 0
            CURRENT_IMAGE += 1

    # Upscale
    if scale > 1:
        msg = f"Upscaling {scale}x"
        with timer(msg):
            safe_progress(progress, 0, num_images, desc=msg)
            for i, image in enumerate(images):
                images = upscaler.predict(image[0])
                images[i] = image
                safe_progress(progress, i + 1, num_images, desc=msg)

    # Flush memory after generating
    cuda_collect()

    end = time.perf_counter()
    msg = f"Generated {len(images)} image{'s' if len(images) > 1 else ''} in {end - start:.2f}s"
    log.info(msg)

    # Alert if notifier provided
    if Info:
        Info(msg)

    return images
