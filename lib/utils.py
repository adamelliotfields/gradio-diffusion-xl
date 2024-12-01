import functools
import json
import time
from contextlib import contextmanager

import torch
from diffusers.utils import logging as diffusers_logging
from huggingface_hub._snapshot_download import snapshot_download
from huggingface_hub.utils import are_progress_bars_disabled
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


@functools.lru_cache()
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def disable_progress_bars():
    transformers_logging.disable_progress_bar()
    diffusers_logging.disable_progress_bar()


def enable_progress_bars():
    # warns if `HF_HUB_DISABLE_PROGRESS_BARS` env var is not None
    transformers_logging.enable_progress_bar()
    diffusers_logging.enable_progress_bar()


def safe_progress(progress, current=0, total=0, desc=""):
    if progress is not None:
        progress((current, total), desc=desc)


def cuda_collect():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()


def download_repo_files(repo_id, allow_patterns, token=None):
    was_disabled = are_progress_bars_disabled()
    enable_progress_bars()
    snapshot_path = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=None,
    )
    if was_disabled:
        disable_progress_bars()
    return snapshot_path
