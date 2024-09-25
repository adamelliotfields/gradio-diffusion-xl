import functools
import inspect
import json
import os
from typing import Callable, TypeVar

import anyio
import httpx
from anyio import Semaphore
from huggingface_hub._snapshot_download import snapshot_download
from typing_extensions import ParamSpec

T = TypeVar("T")
P = ParamSpec("P")

MAX_CONCURRENT_THREADS = 1
MAX_THREADS_GUARD = Semaphore(MAX_CONCURRENT_THREADS)


@functools.lru_cache()
def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


@functools.lru_cache()
def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def download_repo_files(repo_id, allow_patterns, token=None):
    return snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        revision="main",
        token=token,
        allow_patterns=allow_patterns,
        ignore_patterns=None,
    )


def download_civit_file(lora_id, version_id, file_path=".", token=None):
    base_url = "https://civitai.com/api/download/models"
    file = f"{file_path}/{lora_id}.{version_id}.safetensors"

    if os.path.exists(file):
        return

    try:
        params = {"token": token}
        response = httpx.get(
            f"{base_url}/{version_id}",
            timeout=None,
            params=params,
            follow_redirects=True,
        )

        response.raise_for_status()
        os.makedirs(file_path, exist_ok=True)

        with open(file, "wb") as f:
            f.write(response.content)
    except httpx.HTTPStatusError as e:
        print(f"HTTPError: {e.response.status_code} {e.response.text}")
    except httpx.RequestError as e:
        print(f"RequestError: {e}")


# like the original but supports args and kwargs instead of a dict
# https://github.com/huggingface/huggingface-inference-toolkit/blob/0.2.0/src/huggingface_inference_toolkit/async_utils.py
async def async_call(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    async with MAX_THREADS_GUARD:
        sig = inspect.signature(fn)
        bound_args = sig.bind(*args, **kwargs)
        bound_args.apply_defaults()
        partial_fn = functools.partial(fn, **bound_args.arguments)
        return await anyio.to_thread.run_sync(partial_fn)
