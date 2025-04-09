import torch
from DeepCache import DeepCacheSDHelper
from diffusers.models import AutoencoderKL

from .config import Config
from .logger import Logger
from .upscaler import RealESRGAN
from .utils import timer


class Loader:
    def __init__(self):
        self.model = ""
        self.vae = None
        self.refiner = None
        self.pipeline = None
        self.upscaler = None
        self.log = Logger("Loader")
        self.device = torch.device("cuda")  # always called in CUDA context

    def should_unload_deepcache(self, cache_interval=1):
        has_deepcache = hasattr(self.pipeline, "deepcache")
        if has_deepcache and cache_interval == 1:
            return True
        if has_deepcache and self.pipeline.deepcache.params["cache_interval"] != cache_interval:
            return True
        return False

    def should_unload_upscaler(self, scale=1):
        return self.upscaler is not None and self.upscaler.scale != scale

    def should_unload_refiner(self, use_refiner=False):
        return self.refiner is not None and not use_refiner

    def should_unload_pipeline(self, model=""):
        return self.pipeline is not None and self.model != model

    def should_load_deepcache(self, cache_interval=1):
        has_deepcache = hasattr(self.pipeline, "deepcache")
        if not has_deepcache and cache_interval > 1:
            return True
        return False

    def should_load_upscaler(self, scale=1):
        return self.upscaler is None and scale > 1

    def should_load_refiner(self, use_refiner=False):
        return self.refiner is None and use_refiner

    def should_load_pipeline(self, pipeline_id=""):
        if self.pipeline is None:
            return True
        if not isinstance(self.pipeline, Config.PIPELINES[pipeline_id]):
            return True
        return False

    def should_load_scheduler(self, cls, use_karras=False):
        has_karras = hasattr(self.pipeline.scheduler.config, "use_karras_sigmas")
        if not isinstance(self.pipeline.scheduler, cls):
            return True
        if has_karras and self.pipeline.scheduler.config.use_karras_sigmas != use_karras:
            return True
        return False

    def unload_all(self, model, deepcache_interval, scale, use_refiner):
        if self.should_unload_deepcache(deepcache_interval):
            self.log.info("Disabling DeepCache")
            self.pipeline.deepcache.disable()
            delattr(self.pipeline, "deepcache")
            if self.refiner:
                self.refiner.deepcache.disable()
                delattr(self.refiner, "deepcache")

        if self.should_unload_upscaler(scale):
            self.log.info("Unloading upscaler")
            self.upscaler = None

        if self.should_unload_refiner(use_refiner):
            self.log.info("Unloading refiner")
            self.refiner = None

        if self.should_unload_pipeline(model):
            self.log.info(f"Unloading {self.model}")
            if self.refiner:
                self.refiner.vae = None
                self.refiner.scheduler = None
                self.refiner.tokenizer_2 = None
                self.refiner.text_encoder_2 = None
            self.pipeline = None
            self.model = ""

    def load_deepcache(self, interval=1):
        self.log.info("Enabling DeepCache")
        self.pipeline.deepcache = DeepCacheSDHelper(pipe=self.pipeline)
        self.pipeline.deepcache.set_params(cache_interval=interval)
        self.pipeline.deepcache.enable()
        if self.refiner:
            self.refiner.deepcache = DeepCacheSDHelper(pipe=self.refiner)
            self.refiner.deepcache.set_params(cache_interval=interval)
            self.refiner.deepcache.enable()

    def load_upscaler(self, scale=1):
        with timer(f"Loading {scale}x upscaler", logger=self.log.info):
            self.upscaler = RealESRGAN(scale, device=self.device)
            self.upscaler.load()

    def load_refiner(self):
        model = Config.REFINER_MODEL
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
            self.refiner = Pipeline.from_pretrained(model, **refiner_kwargs).to(self.device)
            self.refiner.set_progress_bar_config(disable=True)

    def load_pipeline(self, pipeline_id, model, **kwargs):
        Pipeline = Config.PIPELINES[pipeline_id]

        # Load VAE first
        if self.vae is None:
            self.vae = AutoencoderKL.from_pretrained(
                Config.VAE_MODEL,
                torch_dtype=torch.float16,
            ).to(self.device)

        kwargs["vae"] = self.vae

        # Load from scratch
        if self.pipeline is None:
            with timer(f"Loading {model} ({pipeline_id})", logger=self.log.info):
                if model in Config.SINGLE_FILE_MODELS:
                    checkpoint = Config.HF_REPOS[model][0]
                    self.pipeline = Pipeline.from_single_file(
                        f"https://huggingface.co/{model}/{checkpoint}",
                        **kwargs,
                    ).to(self.device)
                else:
                    self.pipeline = Pipeline.from_pretrained(model, **kwargs).to(self.device)

        # Change to a different one
        else:
            with timer(f"Changing pipeline to {pipeline_id}", logger=self.log.info):
                self.pipeline = Pipeline.from_pipe(self.pipeline).to(self.device)

        # Update model and disable terminal progress bars
        self.model = model
        self.pipeline.set_progress_bar_config(disable=True)

    def load_scheduler(self, cls, use_karras=False, **kwargs):
        self.log.info(f"Loading {cls.__name__}{' with Karras' if use_karras else ''}")
        self.pipeline.scheduler = cls(**kwargs)
        if self.refiner is not None:
            self.refiner.scheduler = self.pipeline.scheduler

    def load(self, pipeline_id, model, scheduler, deepcache_interval, scale, use_karras, use_refiner):
        Scheduler = Config.SCHEDULERS[scheduler]

        scheduler_kwargs = {
            "beta_start": 0.00085,
            "beta_end": 0.012,
            "beta_schedule": "scaled_linear",
            "timestep_spacing": "leading",
            "steps_offset": 1,
        }

        if scheduler not in ["Euler a"]:
            scheduler_kwargs["use_karras_sigmas"] = use_karras

        pipeline_kwargs = {
            "torch_dtype": torch.float16,
            "add_watermarker": False,
            "scheduler": Scheduler(**scheduler_kwargs),
        }

        # Single-file models don't need a variant
        if model not in Config.SINGLE_FILE_MODELS:
            pipeline_kwargs["variant"] = "fp16"
        else:
            pipeline_kwargs["variant"] = None

        # Unload
        self.unload_all(model, deepcache_interval, scale, use_refiner)

        # Load
        if self.should_load_pipeline(pipeline_id):
            self.load_pipeline(pipeline_id, model, **pipeline_kwargs)

        if self.should_load_refiner(use_refiner):
            self.load_refiner()

        if self.should_load_scheduler(Scheduler, use_karras):
            self.load_scheduler(Scheduler, use_karras, **scheduler_kwargs)

        if self.should_load_deepcache(deepcache_interval):
            self.load_deepcache(deepcache_interval)

        if self.should_load_upscaler(scale):
            self.load_upscaler(scale)


# Get a singleton or a new instance of the Loader
def get_loader():
    if not hasattr(get_loader, "_instance"):
        get_loader._instance = Loader()
    assert isinstance(get_loader._instance, Loader)
    return get_loader._instance
