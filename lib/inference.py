import gc
import re
import time
from datetime import datetime
from itertools import product

import torch
from compel import Compel, ReturnedEmbeddingsType
from compel.prompt_parser import PromptParser
from spaces import GPU

from .config import Config
from .loader import Loader
from .utils import load_json


def parse_prompt_with_arrays(prompt: str) -> list[str]:
    arrays = re.findall(r"\[\[(.*?)\]\]", prompt)

    if not arrays:
        return [prompt]

    tokens = [item.split(",") for item in arrays]  # [("a", "b"), (1, 2)]
    combinations = list(product(*tokens))  # [("a", 1), ("a", 2), ("b", 1), ("b", 2)]

    # find all the arrays in the prompt and replace them with tokens
    prompts = []
    for combo in combinations:
        current_prompt = prompt
        for i, token in enumerate(combo):
            current_prompt = current_prompt.replace(f"[[{arrays[i]}]]", token.strip(), 1)
        prompts.append(current_prompt)
    return prompts


def apply_style(positive_prompt, negative_prompt, style_id="none"):
    if style_id.lower() == "none":
        return (positive_prompt, negative_prompt)

    styles = load_json("./data/styles.json")
    style = styles.get(style_id)
    if style is None:
        return (positive_prompt, negative_prompt)

    style_base = style.get("_base", {})
    return (
        style.get("positive").format(prompt=positive_prompt, _base=style_base.get("positive")).strip(),
        style.get("negative").format(prompt=negative_prompt, _base=style_base.get("negative")).strip(),
    )


# max 60s per image
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
    style=None,
    seed=None,
    model="stabilityai/stable-diffusion-xl-base-1.0",
    scheduler="DDIM",
    width=1024,
    height=1024,
    guidance_scale=7.5,
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
    if not torch.cuda.is_available():
        raise Error("CUDA not available")

    # https://pytorch.org/docs/stable/generated/torch.manual_seed.html
    if seed is None or seed < 0:
        seed = int(datetime.now().timestamp() * 1e6) % (2**64)

    KIND = "txt2img"
    CURRENT_STEP = 0
    CURRENT_IMAGE = 1
    EMBEDDINGS_TYPE = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED

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

    start = time.perf_counter()
    print(f"Generating {num_images} image{'s' if num_images > 1 else ''}")

    if Config.ZERO_GPU and progress is not None:
        progress((100, 100), desc="ZeroGPU init")

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

    if loader.pipe is None:
        raise Error(f"Error loading {model}")

    pipe = loader.pipe
    refiner = loader.refiner
    upscaler = loader.upscaler

    # prompt embeds for base and refiner
    compel_1 = Compel(
        text_encoder=[pipe.text_encoder, pipe.text_encoder_2],
        tokenizer=[pipe.tokenizer, pipe.tokenizer_2],
        requires_pooled=[False, True],
        returned_embeddings_type=EMBEDDINGS_TYPE,
        dtype_for_device_getter=lambda _: pipe.dtype,
        device=pipe.device,
    )
    compel_2 = Compel(
        text_encoder=[pipe.text_encoder_2],
        tokenizer=[pipe.tokenizer_2],
        requires_pooled=[True],
        returned_embeddings_type=EMBEDDINGS_TYPE,
        dtype_for_device_getter=lambda _: pipe.dtype,
        device=pipe.device,
    )

    images = []
    current_seed = seed
    for i in range(num_images):
        generator = torch.Generator(device=pipe.device).manual_seed(current_seed)

        try:
            positive_prompts = parse_prompt_with_arrays(positive_prompt)
            index = i % len(positive_prompts)
            positive_styled, negative_styled = apply_style(positive_prompts[index], negative_prompt, style)

            if negative_styled.startswith("(), "):
                negative_styled = negative_styled[4:]

            conditioning_1, pooled_1 = compel_1([positive_styled, negative_styled])
            conditioning_2, pooled_2 = compel_2([positive_styled, negative_styled])
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

        if progress is not None:
            pipe_kwargs["callback_on_step_end"] = callback_on_step_end

        try:
            image = pipe(**pipe_kwargs).images[0]

            refiner_kwargs = {
                "image": image,
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
                refiner_kwargs["callback_on_step_end"] = callback_on_step_end
            if use_refiner:
                image = refiner(**refiner_kwargs).images[0]
            if scale > 1:
                image = upscaler.predict(image)
            images.append((image, str(current_seed)))
            current_seed += 1
        except Exception as e:
            raise Error(f"{e}")
        finally:
            CURRENT_STEP = 0
            CURRENT_IMAGE += 1

    # cleanup
    loader.collect()
    gc.collect()

    end = time.perf_counter()
    msg = f"Generated {len(images)} image{'s' if len(images) > 1 else ''} in {end - start:.2f}s"
    print(msg)
    if Info:
        Info(msg)
    return images
