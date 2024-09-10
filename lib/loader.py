import gc
from threading import Lock
from warnings import filterwarnings

import torch
from DeepCache import DeepCacheSDHelper
from diffusers.models import AutoencoderKL

from .config import Config
from .upscaler import RealESRGAN

__import__("diffusers").logging.set_verbosity_error()
filterwarnings("ignore", category=FutureWarning, module="torch")
filterwarnings("ignore", category=FutureWarning, module="diffusers")


class Loader:
    _instance = None
    _lock = Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance.pipe = None
                cls._instance.model = None
                cls._instance.refiner = None
                cls._instance.upscaler = None
        return cls._instance

    def _flush(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    def _should_unload_pipeline(self, model=""):
        if self.pipe is None:
            return False
        if self.model.lower() != model.lower():
            return True
        return False

    def _should_unload_refiner(self, refiner):
        if self.refiner is not None and not refiner:
            return True
        return False

    def _should_unload_upscaler(self, scale=1):
        return self.upscaler is not None and scale == 1

    def _unload(self, model, refiner, scale):
        to_unload = []
        if self._should_unload_pipeline(model):
            to_unload.append("model")
            to_unload.append("pipe")
        if self._should_unload_refiner(refiner):
            to_unload.append("refiner")
        if self._should_unload_upscaler(scale):
            to_unload.append("upscaler")
        for component in to_unload:
            delattr(self, component)
        self._flush()
        for component in to_unload:
            setattr(self, component, None)

    def _load_pipeline(self, kind, model, tqdm, **kwargs):
        pipeline = Config.PIPELINES[kind]
        if self.pipe is None:
            try:
                print(f"Loading {model}...")
                self.model = model
                self.pipe = pipeline.from_pretrained(model, **kwargs).to("cuda")
                if self.refiner is not None:
                    self.refiner.vae = self.pipe.vae
                    self.refiner.scheduler = self.pipe.scheduler
                    self.refiner.tokenizer_2 = self.pipe.tokenizer_2
                    self.refiner.text_encoder_2 = self.pipe.text_encoder_2
            except Exception as e:
                print(f"Error loading {model}: {e}")
                self.model = None
                self.pipe = None
                return
        if not isinstance(self.pipe, pipeline):
            self.pipe = pipeline.from_pipe(self.pipe).to("cuda")
        if self.pipe is not None:
            self.pipe.set_progress_bar_config(disable=not tqdm)

    def _load_refiner(self, refiner, tqdm, **kwargs):
        if self.refiner is None and refiner:
            model = Config.REFINER_MODEL
            pipeline = Config.PIPELINES["img2img"]
            try:
                print(f"Loading {model}...")
                self.refiner = pipeline.from_pretrained(model, **kwargs).to("cuda")
            except Exception as e:
                print(f"Error loading {model}: {e}")
                self.refiner = None
                return
        if self.refiner is not None:
            self.refiner.set_progress_bar_config(disable=not tqdm)

    def _load_upscaler(self, scale=1):
        if scale > 1 and self.upscaler is None:
            print(f"Loading {scale}x upscaler...")
            self.upscaler = RealESRGAN(scale, "cuda")
            self.upscaler.load_weights()

    def _load_deepcache(self, interval=1):
        has_deepcache = hasattr(self.pipe, "deepcache")
        if has_deepcache and self.pipe.deepcache.params["cache_interval"] == interval:
            return
        if has_deepcache:
            self.pipe.deepcache.disable()
        else:
            self.pipe.deepcache = DeepCacheSDHelper(pipe=self.pipe)
        self.pipe.deepcache.set_params(cache_interval=interval)
        self.pipe.deepcache.enable()

    def load(self, kind, model, scheduler, deepcache, scale, karras, refiner, tqdm):
        model_lower = model.lower()

        scheduler_kwargs = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "timestep_spacing": "leading",
            "steps_offset": 1,
        }

        if scheduler not in ["DDIM", "Euler a"]:
            scheduler_kwargs["use_karras_sigmas"] = karras

        # https://github.com/huggingface/diffusers/blob/8a3f0c1/scripts/convert_original_stable_diffusion_to_diffusers.py#L939
        if scheduler == "DDIM":
            scheduler_kwargs["clip_sample"] = False
            scheduler_kwargs["set_alpha_to_one"] = False

        # no fp16 variant (already half-precision)
        if model_lower not in ["cagliostrolab/animagine-xl-3.1", "fluently/fluently-xl-final"]:
            variant = "fp16"
        else:
            variant = None

        dtype = torch.float16
        pipe_kwargs = {
            "variant": variant,
            "torch_dtype": dtype,
            "add_watermarker": False,
            "scheduler": Config.SCHEDULERS[scheduler](**scheduler_kwargs),
            "vae": AutoencoderKL.from_pretrained(Config.VAE_MODEL, torch_dtype=dtype),
        }

        self._unload(model, refiner, scale)
        self._load_pipeline(kind, model, tqdm, **pipe_kwargs)

        # error loading model
        if self.pipe is None:
            return None, None, None

        same_scheduler = isinstance(self.pipe.scheduler, Config.SCHEDULERS[scheduler])
        same_karras = (
            not hasattr(self.pipe.scheduler.config, "use_karras_sigmas")
            or self.pipe.scheduler.config.use_karras_sigmas == karras
        )

        # same model, different scheduler
        if self.model.lower() == model_lower:
            if not same_scheduler:
                print(f"Switching to {scheduler}...")
            if not same_karras:
                print(f"{'Enabling' if karras else 'Disabling'} Karras sigmas...")
            if not same_scheduler or not same_karras:
                self.pipe.scheduler = Config.SCHEDULERS[scheduler](**scheduler_kwargs)
                self.refiner.scheduler = self.pipe.scheduler

        # https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0/blob/main/model_index.json
        refiner_kwargs = {
            "variant": "fp16",
            "torch_dtype": dtype,
            "add_watermarker": False,
            "requires_aesthetics_score": True,
            "force_zeros_for_empty_prompt": False,
            "vae": self.pipe.vae,
            "scheduler": self.pipe.scheduler,
            "tokenizer_2": self.pipe.tokenizer_2,
            "text_encoder_2": self.pipe.text_encoder_2,
        }

        self._load_refiner(refiner, tqdm, **refiner_kwargs)
        self._load_upscaler(scale)
        self._load_deepcache(deepcache)
        return self.pipe, self.refiner, self.upscaler
