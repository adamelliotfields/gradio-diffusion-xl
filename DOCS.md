## Usage

TL;DR: Enter a prompt or roll the `🎲` and press `Generate`.

### Prompting

Positive and negative prompts are embedded by [Compel](https://github.com/damian0815/compel). See [syntax features](https://github.com/damian0815/compel/blob/main/doc/syntax.md) to learn more.

#### Weighting

Use `+` or `-` to increase the weight of a token. The weight grows exponentially when chained. For example, `blue+` means 1.1x more attention is given to `blue`, while `blue++` means 1.1^2 more, and so on. The same applies to `-`.

Groups of tokens can be weighted together by wrapping in parentheses and multiplying by a float between 0 and 2. For example, `(masterpiece, best quality)1.2` will increase the weight of both `masterpiece` and `best quality` by 1.2x.

### Models

* [cyberdelia/CyberRealisticXL](https://huggingface.co/cyberdelia/CyberRealsticXL)
* [fluently/Fluently-XL-Final](https://huggingface.co/fluently/Fluently-XL-Final)
* [segmind/Segmind-Vega](https://huggingface.co/segmind/Segmind-Vega) (default)
* [SG161222/RealVisXL_V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0)
* [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

### Scale

Rescale up to 4x using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) with weights from [ai-forever](ai-forever/Real-ESRGAN). Necessary for high-resolution images.

### Advanced

#### DeepCache

[DeepCache](https://github.com/horseee/DeepCache) caches lower UNet layers and reuses them every _n_ steps. Trade quality for speed:
* `1`: no caching (default)
* `2`: more quality
* `3`: balanced
* `4`: more speed

#### Refiner

Use the [ensemble of expert denoisers](https://research.nvidia.com/labs/dir/eDiff-I/) technique, where the first 80% of timesteps are denoised by the base model and the remaining 80% by the [refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0). Not available with image-to-image pipelines.
