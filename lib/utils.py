import functools
import json
import time
from contextlib import contextmanager

import torch
from diffusers.utils import logging as diffusers_logging
from transformers import logging as transformers_logging


@contextmanager
def timer(message="Operation", logger=print):
    start = time.perf_counter()
    logger(message)
    try:
        yield
    finally:
        end = time.perf_counter()
        logger(f"{message} took {end - start:.2f}s")


@functools.lru_cache()
def read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        data = json.load(file)
        return json.dumps(data, indent=4)


def disable_progress_bars():
    transformers_logging.disable_progress_bar()
    diffusers_logging.disable_progress_bar()


def enable_progress_bars():
    # warns if `HF_HUB_DISABLE_PROGRESS_BARS` env var is not None
    transformers_logging.enable_progress_bar()
    diffusers_logging.enable_progress_bar()


def get_output_types(scale=1, use_refiner=False):
    if use_refiner:
        pipeline_type = "latent"
        refiner_type = "np" if scale > 1 else "pil"
    else:
        refiner_type = "pil"
        pipeline_type = "np" if scale > 1 else "pil"
    return (pipeline_type, refiner_type)


def cuda_collect():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
