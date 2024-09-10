import functools
import inspect
import json
import re
import time
from datetime import datetime
from itertools import product
from typing import Callable, TypeVar

import anyio
import spaces
import torch
from anyio import Semaphore
from compel import Compel, ReturnedEmbeddingsType
from compel.prompt_parser import PromptParser
from typing_extensions import ParamSpec

from .loader import Loader

__import__("warnings").filterwarnings("ignore", category=FutureWarning, module="transformers")
__import__("transformers").logging.set_verbosity_error()

T = TypeVar("T")
P = ParamSpec("P")

MAX_CONCURRENT_THREADS = 1
MAX_THREADS_GUARD = Semaphore(MAX_CONCURRENT_THREADS)

with open("./data/styles.json") as f:
    STYLES = json.load(f)


# like the original but supports args and kwargs instead of a dict
# https://github.com/huggingface/huggingface-inference-toolkit/blob/0.2.0/src/huggingface_inference_toolkit/async_utils.py
async def async_call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    async with MAX_THREADS_GUARD:
        sig = inspect.signature(fn)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        partial_fn = functools.partial(fn, **bound_args.arguments)
        return await anyio.to_thread.run_sync(partial_fn)


# parse prompts with arrays
def parse_prompt(prompt: str) -> list[str]:
    arrays = re.findall(r"\[\[(.*?)\]\]", prompt)

    if not arrays:
        return [prompt]

    tokens = [item.split(",") for item in arrays]
    combinations = list(product(*tokens))
    prompts = []

    for combo in combinations:
        current_prompt = prompt
        for i, token in enumerate(combo):
            current_prompt = current_prompt.replace(f"[[{arrays[i]}]]", token.strip(), 1)
        prompts.append(current_prompt)

    return prompts


def apply_style(prompt, style_id, negative=False):
    global STYLES
    if not style_id or style_id == "None":
        return prompt
    for style in STYLES:
        if style["id"] == style_id:
            if negative:
                return prompt + " . " + style["negative_prompt"]
            else:
                return style["prompt"].format(prompt=prompt)
    return prompt


# TODO: fine-tune these
def gpu_duration(**kwargs):
    duration = 20
    scale = kwargs.get("scale", 1)
    num_images = kwargs.get("num_images", 1)
    if scale == 4:
        duration += 10
    return duration * num_images


@spaces.GPU(duration=gpu_duration)
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
    Info: Callable[[str], None] = None,
    Error=Exception,
    progress=None,
):
    if not torch.cuda.is_available():
        raise Error("RuntimeError: CUDA not available")

    # https://pytorch.org/docs/stable/generated/torch.manual_seed.html
    if seed is None or seed < 0:
        seed = int(datetime.now().timestamp() * 1_000_000) % (2**64)

    KIND = "txt2img"
    CURRENT_STEP = 0
    CURRENT_IMAGE = 1
    EMBEDDINGS_TYPE = ReturnedEmbeddingsType.PENULTIMATE_HIDDEN_STATES_NON_NORMALIZED

    if progress is not None:
        TQDM = False
        progress((0, inference_steps), desc=f"Generating image 1/{num_images}")
    else:
        TQDM = True

    def callback_on_step_end(pipeline, step, timestep, latents):
        nonlocal CURRENT_IMAGE, CURRENT_STEP

        if progress is None:
            return latents

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
    loader = Loader()
    pipe, refiner, upscaler = loader.load(
        KIND,
        model,
        scheduler,
        deepcache,
        scale,
        use_karras,
        use_refiner,
        TQDM,
    )

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
        # seeded generator for each iteration
        generator = torch.Generator(device=pipe.device).manual_seed(current_seed)

        try:
            styled_negative_prompt = apply_style(negative_prompt, style, negative=True)
            all_positive_prompts = parse_prompt(positive_prompt)
            prompt_index = i % len(all_positive_prompts)
            prompt = all_positive_prompts[prompt_index]
            styled_prompt = apply_style(prompt, style)
            conditioning_1, pooled_1 = compel_1([styled_prompt, styled_negative_prompt])
            conditioning_2, pooled_2 = compel_2([styled_prompt, styled_negative_prompt])
        except PromptParser.ParsingException:
            raise Error("ValueError: Invalid prompt")

        # refiner expects latents; upscaler expects numpy array
        pipe_output_type = "pil"
        refiner_output_type = "pil"
        if refiner:
            pipe_output_type = "latent"
            if scale > 1:
                refiner_output_type = "np"
        else:
            if scale > 1:
                pipe_output_type = "np"

        pipe_kwargs = {
            "width": width,
            "height": height,
            "denoising_end": 0.8 if refiner else None,
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
        except Exception as e:
            raise Error(f"RuntimeError: {e}")
        finally:
            # reset step and increment image
            CURRENT_STEP = 0
            CURRENT_IMAGE += 1
            current_seed += 1

    diff = time.perf_counter() - start
    if Info:
        Info(f"Generated {len(images)} image{'s' if len(images) > 1 else ''} in {diff:.2f}s")
    return images
