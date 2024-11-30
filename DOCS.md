## Usage

TL;DR: Enter a prompt or roll the `🎲` and press `Generate`.

### Prompting

Positive and negative prompts are embedded by [Compel](https://github.com/damian0815/compel) for weighting. See [syntax features](https://github.com/damian0815/compel/blob/main/doc/syntax.md) to learn more.

Use `+` or `-` to increase the weight of a token. The weight grows exponentially when chained. For example, `blue+` means 1.1x more attention is given to `blue`, while `blue++` means 1.1^2 more, and so on. The same applies to `-`.

For groups of tokens, wrap them in parentheses and multiply by a float between 0 and 2. For example, `a (birthday cake)1.3 on a table` will increase the weight of both `birthday` and `cake` by 1.3x. This also means the entire scene will be more birthday-like, not just the cake. To counteract this, you can use `-` inside the parentheses on specific tokens, e.g., `a (birthday-- cake)1.3`, to reduce the birthday aspect.

### Models

Each model checkpoint has a different aesthetic:

* [cyberdelia/CyberRealisticXL](https://huggingface.co/cyberdelia/CyberRealsticXL): photorealistic
* [fluently/Fluently-XL-Final](https://huggingface.co/fluently/Fluently-XL-Final): general purpose
* [segmind/Segmind-Vega](https://huggingface.co/segmind/Segmind-Vega): lightweight general purpose (default)
* [SG161222/RealVisXL_V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0): photorealistic
* [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0): base

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
