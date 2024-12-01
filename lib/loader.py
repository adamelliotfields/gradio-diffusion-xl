import torch
from DeepCache import DeepCacheSDHelper
from diffusers.models import AutoencoderKL

from .config import Config
from .logger import Logger
from .upscaler import RealESRGAN
from .utils import cuda_collect, timer


class Loader:
    def __init__(self):
        self.model = ""
        self.refiner = None
        self.pipeline = None
        self.upscaler = None
        self.log = Logger("Loader")

    def should_unload_refiner(self, use_refiner=False):
        return self.refiner is not None and not use_refiner

    def should_unload_upscaler(self, scale=1):
        return self.upscaler is not None and self.upscaler.scale != scale

    def should_unload_deepcache(self, interval=1):
        has_deepcache = hasattr(self.pipeline, "deepcache")
        if has_deepcache and interval == 1:
            return True
        if has_deepcache and self.pipeline.deepcache.params["cache_interval"] != interval:
            return True
        return False

    def should_unload_pipeline(self, model=""):
        return self.pipeline is not None and self.model != model

    def should_load_refiner(self, use_refiner=False):
        return self.refiner is None and use_refiner

    def should_load_upscaler(self, scale=1):
        return self.upscaler is None and scale > 1

    def should_load_deepcache(self, interval=1):
        has_deepcache = hasattr(self.pipeline, "deepcache")
        if not has_deepcache and interval != 1:
            return True
        if has_deepcache and self.pipeline.deepcache.params["cache_interval"] != interval:
            return True
        return False

    def should_load_pipeline(self):
        return self.pipeline is None

    def unload(self, model, use_refiner, deepcache_interval, scale):
        if self.should_unload_deepcache(deepcache_interval):
            self.log.info("Disabling DeepCache")
            self.pipeline.deepcache.disable()
            delattr(self.pipeline, "deepcache")
            if self.refiner:
                self.refiner.deepcache.disable()
                delattr(self.refiner, "deepcache")

        if self.should_unload_refiner(use_refiner):
            self.log.info("Unloading refiner")
            self.refiner = None

        if self.should_unload_upscaler(scale):
            self.log.info("Unloading upscaler")
            self.upscaler = None

        if self.should_unload_pipeline(model):
            self.log.info(f"Unloading {self.model}")
            if self.refiner:
                self.refiner.vae = None
                self.refiner.scheduler = None
                self.refiner.tokenizer_2 = None
                self.refiner.text_encoder_2 = None
            self.pipeline = None
            self.model = None

        # Flush cache
        cuda_collect()

    def load_refiner(self, progress=None):
        model = Config.REFINER_MODEL
        try:
            with timer(f"Loading {model}", logger=self.log.info):
                refiner_kwargs = {
                    "variant": "fp16",
                    "torch_dtype": self.pipeline.dtype,
                    "add_watermarker": False,
                    "requires_aesthetics_score": True,
                    "force_zeros_for_empty_prompt": False,
                    "vae": self.pipeline.vae,
                    "scheduler": self.pipeline.scheduler,
                    "tokenizer_2": self.pipeline.tokenizer_2,
                    "text_encoder_2": self.pipeline.text_encoder_2,
                }
                Pipeline = Config.PIPELINES["img2img"]
                self.refiner = Pipeline.from_pretrained(model, **refiner_kwargs).to("cuda")
        except Exception as e:
            self.log.error(f"Error loading {model}: {e}")
            self.refiner = None
            return
        if self.refiner is not None:
            self.refiner.set_progress_bar_config(disable=progress is not None)

    def load_upscaler(self, scale=1):
        if self.should_load_upscaler(scale):
            try:
                with timer(f"Loading {scale}x upscaler", logger=self.log.info):
                    self.upscaler = RealESRGAN(scale, device=self.pipeline.device)
                    self.upscaler.load_weights()
            except Exception as e:
                self.log.error(f"Error loading {scale}x upscaler: {e}")
                self.upscaler = None

    def load_deepcache(self, interval=1):
        if self.should_load_deepcache(interval):
            self.log.info("Enabling DeepCache")
            self.pipeline.deepcache = DeepCacheSDHelper(pipe=self.pipeline)
            self.pipeline.deepcache.set_params(cache_interval=interval)
            self.pipeline.deepcache.enable()
            if self.refiner:
                self.refiner.deepcache = DeepCacheSDHelper(pipe=self.refiner)
                self.refiner.deepcache.set_params(cache_interval=interval)
                self.refiner.deepcache.enable()

    def load(self, kind, model, scheduler, deepcache_interval, scale, use_karras, use_refiner, progress=None):
        scheduler_kwargs = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "timestep_spacing": "leading",
            "steps_offset": 1,
        }

        if scheduler not in ["DDIM", "Euler a"]:
            scheduler_kwargs["use_karras_sigmas"] = use_karras

        if scheduler == "DDIM":
            scheduler_kwargs["clip_sample"] = False
            scheduler_kwargs["set_alpha_to_one"] = False

        if model not in Config.SINGLE_FILE_MODELS:
            variant = "fp16"
        else:
            variant = None

        dtype = torch.float16
        pipeline_kwargs = {
            "variant": variant,
            "torch_dtype": dtype,
            "add_watermarker": False,
            "scheduler": Config.SCHEDULERS[scheduler](**scheduler_kwargs),
            "vae": AutoencoderKL.from_pretrained(Config.VAE_MODEL, torch_dtype=dtype),
        }

        self.unload(model, use_refiner, deepcache_interval, scale)

        Pipeline = Config.PIPELINES[kind]
        Scheduler = Config.SCHEDULERS[scheduler]

        try:
            with timer(f"Loading {model}", logger=self.log.info):
                self.model = model
                if model in Config.SINGLE_FILE_MODELS:
                    checkpoint = Config.HF_REPOS[model][0]
                    self.pipeline = Pipeline.from_single_file(
                        f"https://huggingface.co/{model}/{checkpoint}",
                        **pipeline_kwargs,
                    ).to("cuda")
                else:
                    self.pipeline = Pipeline.from_pretrained(model, **pipeline_kwargs).to("cuda")
        except Exception as e:
            self.log.error(f"Error loading {model}: {e}")
            self.model = None
            self.pipeline = None
            return

        if not isinstance(self.pipeline, Pipeline):
            self.pipeline = Pipeline.from_pipe(self.pipeline).to("cuda")

        if self.pipeline is not None:
            self.pipeline.set_progress_bar_config(disable=progress is not None)

        # Check and update scheduler if necessary
        same_scheduler = isinstance(self.pipeline.scheduler, Scheduler)
        same_karras = (
            not hasattr(self.pipeline.scheduler.config, "use_karras_sigmas")
            or self.pipeline.scheduler.config.use_karras_sigmas == use_karras
        )

        if self.model == model:
            if not same_scheduler:
                self.log.info(f"Enabling {scheduler}")
            if not same_karras:
                self.log.info(f"{'Enabling' if use_karras else 'Disabling'} Karras sigmas")
            if not same_scheduler or not same_karras:
                self.pipeline.scheduler = Scheduler(**scheduler_kwargs)
                if self.refiner is not None:
                    self.refiner.scheduler = self.pipeline.scheduler

        if self.should_load_refiner(use_refiner):
            self.load_refiner(progress)

        if self.should_load_deepcache(deepcache_interval):
            self.load_deepcache(deepcache_interval)

        if self.should_load_upscaler(scale):
            self.load_upscaler(scale)
