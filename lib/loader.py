import gc
from threading import Lock

import torch
from DeepCache import DeepCacheSDHelper
from diffusers.models import AutoencoderKL

from .config import Config
from .logger import Logger
from .upscaler import RealESRGAN
from .utils import clear_cuda_cache, timer


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
                cls._instance.log = Logger("Loader")
        return cls._instance

    def _should_unload_refiner(self, refiner=False):
        if self.refiner is None:
            return False
        if not refiner:
            return True
        return False

    def _should_unload_upscaler(self, scale=1):
        if self.upscaler is not None and self.upscaler.scale != scale:
            return True
        return False

    def _should_unload_deepcache(self, interval=1):
        has_deepcache = hasattr(self.pipe, "deepcache")
        if has_deepcache and interval == 1:
            return True
        if has_deepcache and self.pipe.deepcache.params["cache_interval"] != interval:
            return True
        return False

    def _should_unload_pipeline(self, model=""):
        if self.pipe is None:
            return False
        if self.model and self.model.lower() != model.lower():
            return True
        return False

    def _unload_refiner(self):
        if self.refiner is not None:
            with timer("Unloading refiner"):
                self.refiner.to("cpu", silence_dtype_warnings=True)

    def _unload_upscaler(self):
        if self.upscaler is not None:
            with timer(f"Unloading {self.upscaler.scale}x upscaler"):
                self.upscaler.to("cpu")

    def _unload_deepcache(self):
        if self.pipe.deepcache is not None:
            self.log.info("Disabling DeepCache")
            self.pipe.deepcache.disable()
            delattr(self.pipe, "deepcache")
            if self.refiner is not None:
                if hasattr(self.refiner, "deepcache"):
                    self.refiner.deepcache.disable()
                    delattr(self.refiner, "deepcache")

    def _unload_pipeline(self):
        if self.pipe is not None:
            with timer(f"Unloading {self.model}"):
                self.pipe.to("cpu", silence_dtype_warnings=True)
                if self.refiner is not None:
                    self.refiner.vae = None
                    self.refiner.scheduler = None
                    self.refiner.tokenizer_2 = None
                    self.refiner.text_encoder_2 = None

    def _unload(self, model, refiner, deepcache, scale):
        to_unload = []
        if self._should_unload_deepcache(deepcache):  # remove deepcache first
            self._unload_deepcache()

        if self._should_unload_refiner(refiner):
            self._unload_refiner()
            to_unload.append("refiner")

        if self._should_unload_upscaler(scale):
            self._unload_upscaler()
            to_unload.append("upscaler")

        if self._should_unload_pipeline(model):
            self._unload_pipeline()
            to_unload.append("model")
            to_unload.append("pipe")

        # Flush cache and run garbage collector
        clear_cuda_cache()
        for component in to_unload:
            setattr(self, component, None)
        gc.collect()

    def _should_load_refiner(self, refiner=False):
        if self.refiner is None and refiner:
            return True
        return False

    def _should_load_upscaler(self, scale=1):
        if self.upscaler is None and scale > 1:
            return True
        return False

    def _should_load_deepcache(self, interval=1):
        has_deepcache = hasattr(self.pipe, "deepcache")
        if not has_deepcache and interval != 1:
            return True
        if has_deepcache and self.pipe.deepcache.params["cache_interval"] != interval:
            return True
        return False

    def _should_load_pipeline(self):
        if self.pipe is None:
            return True
        return False

    def _load_refiner(self, refiner, progress, **kwargs):
        if self._should_load_refiner(refiner):
            model = Config.REFINER_MODEL
            pipeline = Config.PIPELINES["img2img"]
            try:
                with timer(f"Loading {model}"):
                    self.refiner = pipeline.from_pretrained(model, **kwargs).to("cuda")
            except Exception as e:
                self.log.error(f"Error loading {model}: {e}")
                self.refiner = None
                return
        if self.refiner is not None:
            self.refiner.set_progress_bar_config(disable=progress is not None)

    def _load_upscaler(self, scale=1):
        if self._should_load_upscaler(scale):
            try:
                with timer(f"Loading {scale}x upscaler"):
                    self.upscaler = RealESRGAN(scale, device=self.pipe.device)
                    self.upscaler.load_weights()
            except Exception as e:
                self.log.error(f"Error loading {scale}x upscaler: {e}")
                self.upscaler = None

    def _load_deepcache(self, interval=1):
        if self._should_load_deepcache(interval):
            self.log.info("Enabling DeepCache")
            self.pipe.deepcache = DeepCacheSDHelper(pipe=self.pipe)
            self.pipe.deepcache.set_params(cache_interval=interval)
            self.pipe.deepcache.enable()
            if self.refiner is not None:
                self.refiner.deepcache = DeepCacheSDHelper(pipe=self.refiner)
                self.refiner.deepcache.set_params(cache_interval=interval)
                self.refiner.deepcache.enable()

    def _load_pipeline(self, kind, model, progress, **kwargs):
        pipeline = Config.PIPELINES[kind]
        if self._should_load_pipeline():
            try:
                with timer(f"Loading {model}"):
                    self.model = model
                    if model.lower() in Config.MODEL_CHECKPOINTS.keys():
                        self.pipe = pipeline.from_single_file(
                            f"https://huggingface.co/{model}/{Config.MODEL_CHECKPOINTS[model.lower()]}",
                            **kwargs,
                        ).to("cuda")
                    else:
                        self.pipe = pipeline.from_pretrained(model, **kwargs).to("cuda")
                    if self.refiner is not None:
                        self.refiner.vae = self.pipe.vae
                        self.refiner.scheduler = self.pipe.scheduler
                        self.refiner.tokenizer_2 = self.pipe.tokenizer_2
                        self.refiner.text_encoder_2 = self.pipe.text_encoder_2
                        self.refiner.to(self.pipe.device)
            except Exception as e:
                self.log.error(f"Error loading {model}: {e}")
                self.model = None
                self.pipe = None
                self.refiner = None
                return
        if not isinstance(self.pipe, pipeline):
            self.pipe = pipeline.from_pipe(self.pipe).to("cuda")
        if self.pipe is not None:
            self.pipe.set_progress_bar_config(disable=progress is not None)

    def load(self, kind, model, scheduler, deepcache, scale, karras, refiner, progress):
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

        if model.lower() not in Config.MODEL_CHECKPOINTS.keys():
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

        self._unload(model, refiner, deepcache, scale)
        self._load_pipeline(kind, model, progress, **pipe_kwargs)

        # error loading model
        if self.pipe is None:
            return

        same_scheduler = isinstance(self.pipe.scheduler, Config.SCHEDULERS[scheduler])
        same_karras = (
            not hasattr(self.pipe.scheduler.config, "use_karras_sigmas")
            or self.pipe.scheduler.config.use_karras_sigmas == karras
        )

        # same model, different scheduler
        if self.model.lower() == model.lower():
            if not same_scheduler:
                self.log.info(f"Enabling {scheduler}")
            if not same_karras:
                self.log.info(f"{'Enabling' if karras else 'Disabling'} Karras sigmas")
            if not same_scheduler or not same_karras:
                self.pipe.scheduler = Config.SCHEDULERS[scheduler](**scheduler_kwargs)
                if self.refiner is not None:
                    self.refiner.scheduler = self.pipe.scheduler

        if self._should_load_refiner(refiner):
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
            self._load_refiner(refiner, progress, **refiner_kwargs)  # load refiner before deepcache

        if self._should_load_deepcache(deepcache):
            self._load_deepcache(deepcache)

        if self._should_load_upscaler(scale):
            self._load_upscaler(scale)
