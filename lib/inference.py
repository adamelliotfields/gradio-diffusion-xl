import time
from datetime import datetime

import torch
from compel import Compel, ReturnedEmbeddingsType
from compel.prompt_parser import PromptParser
from gradio import Error, Info, Progress
from spaces import GPU

from .loader import get_loader
from .logger import Logger
from .utils import cuda_collect, get_output_types, timer


@GPU
def generate(
    positive_prompt="",
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
    _=Progress(track_tqdm=True),
):
    if not torch.cuda.is_available():
        raise Error("CUDA not available")

    if positive_prompt.strip() == "":
        raise Error("You must enter a prompt")

    KIND = "txt2img"
    EMBEDDINGS_TYPE = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED

    start = time.perf_counter()
    log = Logger("generate")
    log.info(f"Generating {num_images} image{'s' if num_images > 1 else ''}...")

    loader = get_loader()
    loader.load(
        KIND,
        model,
        scheduler,
        deepcache,
        scale,
        use_karras,
        use_refiner,
    )

    refiner = loader.refiner
    pipeline = loader.pipeline
    upscaler = loader.upscaler

    # Probably a typo in the config
    if pipeline is None:
        raise Error(f"Error loading {model}")

    # Prompt embeddings for base and refiner
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

    # https://pytorch.org/docs/stable/generated/torch.manual_seed.html
    if seed is None or seed < 0:
        seed = int(datetime.now().timestamp() * 1e6) % (2**64)

    # Increment the seed after each iteration
    images = []
    current_seed = seed

    for i in range(num_images):
        try:
            generator = torch.Generator(device=pipeline.device).manual_seed(current_seed)
            conditioning_1, pooled_1 = compel_1([positive_prompt, negative_prompt])
            conditioning_2, pooled_2 = compel_2([positive_prompt, negative_prompt])
        except PromptParser.ParsingException:
            raise Error("Invalid prompt")

        pipeline_output_type, refiner_output_type = get_output_types(scale, use_refiner)

        pipeline_kwargs = {
            "width": width,
            "height": height,
            "denoising_end": 0.8 if use_refiner else None,
            "generator": generator,
            "output_type": pipeline_output_type,
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

        image = pipeline(**pipeline_kwargs).images[0]

        if use_refiner:
            refiner_kwargs["image"] = image
            image = refiner(**refiner_kwargs).images[0]

        # Use a tuple so gallery images get captions
        images.append((image, str(current_seed)))
        current_seed += 1

    # Upscale
    if scale > 1:
        with timer(f"Upscaling {num_images} images {scale}x", logger=log.info):
            for i, image in enumerate(images):
                image = upscaler.predict(image[0])
                seed = images[i][1]
                images[i] = (image, seed)

    end = time.perf_counter()
    msg = f"Generated {len(images)} image{'s' if len(images) > 1 else ''} in {end - start:.2f}s"
    log.info(msg)

    if Info:
        Info(msg)

    # Flush cache before returning
    cuda_collect()

    return images
