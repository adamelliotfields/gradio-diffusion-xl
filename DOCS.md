# Diffusion XL

TL;DR: Enter a prompt or roll the `🎲` and press `Generate`.

## Prompting

Positive and negative prompts are embedded by [Compel](https://github.com/damian0815/compel) for weighting. See [syntax features](https://github.com/damian0815/compel/blob/main/doc/syntax.md) to learn more and read [Civitai](https://civitai.com)'s guide on [prompting](https://education.civitai.com/civitais-prompt-crafting-guide-part-1-basics/) for best practices.

### Arrays

Arrays allow you to generate different images from a single prompt. For example, `[[cat,corgi]]` will expand into 2 separate prompts. Make sure `Images` is set accordingly (e.g., 2). Only works for the positive prompt. Inspired by [Fooocus](https://github.com/lllyasviel/Fooocus/pull/1503).

## Styles

Styles are prompt templates from twri's [sdxl_prompt_styler](https://github.com/twri/sdxl_prompt_styler) Comfy node. Start with a subject like "cat", pick a style, and iterate from there.

## Scale

Rescale up to 4x using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) from [ai-forever](https://huggingface.co/ai-forever/Real-ESRGAN).

## Models

Each model checkpoint has a different aesthetic:

* [cagliostrolab/animagine-xl-3.1](https://huggingface.co/cagliostrolab/animagine-xl-3.1): anime
* [cyberdelia/CyberRealisticXL](https://huggingface.co/cyberdelia/CyberRealsticXL): photorealistic
* [fluently/Fluently-XL-Final](https://huggingface.co/fluently/Fluently-XL-Final): general purpose
* [segmind/Segmind-Vega](https://huggingface.co/segmind/Segmind-Vega): lightweight general purpose (default)
* [SG161222/RealVisXL_V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0): photorealistic
* [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0): base

## Advanced

### DeepCache

[DeepCache](https://github.com/horseee/DeepCache) caches lower UNet layers and reuses them every `Interval` steps. Trade quality for speed:
* `1`: no caching (default)
* `2`: more quality
* `3`: balanced
* `4`: more speed

### Refiner

Use the [ensemble of expert denoisers](https://research.nvidia.com/labs/dir/eDiff-I/) technique, where the first 80% of timesteps are denoised by the base model and the remaining 80% by the [refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0). Not available with image-to-image pipelines.
