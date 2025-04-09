# gradio-diffusion-xl

Gradio app for Stable Diffusion XL featuring:

* txt2img pipeline with refiner
* Real-ESRGAN resizing up to 8x
* Compel prompt weighting support
* Multiple samplers with Karras scheduling
* DeepCache available for faster inference

## Installation

```sh
uv venv
uv pip install -r requirements.txt
uv run app.py
```

## Usage

Enter a prompt or roll the `🎲` and press `Generate`.

### Prompting

Positive and negative prompts are embedded by [Compel](https://github.com/damian0815/compel). See [syntax features](https://github.com/damian0815/compel/blob/main/doc/syntax.md) to learn more.

### Models

* [cyberdelia/CyberRealisticXL](https://huggingface.co/cyberdelia/CyberRealsticXL)
* [fluently/Fluently-XL-Final](https://huggingface.co/fluently/Fluently-XL-Final)
* [segmind/Segmind-Vega](https://huggingface.co/segmind/Segmind-Vega) (default)
* [SG161222/RealVisXL_V5.0](https://huggingface.co/SG161222/RealVisXL_V5.0)
* [stabilityai/stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0)

### Upscaler

Resize up to 8x using [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) with weights from [ai-forever](ai-forever/Real-ESRGAN).

### Advanced

#### DeepCache

[DeepCache](https://github.com/horseee/DeepCache) caches lower UNet layers and reuses them every _n_ steps. Trade quality for speed:
- *1*: no caching (default)
- *2*: more quality
- *3*: balanced
- *4*: more speed

#### Refiner

Use the [ensemble of expert denoisers](https://research.nvidia.com/labs/dir/eDiff-I/) technique, where the first 80% of timesteps are denoised by the base model and the remaining 20% by the [refiner](https://huggingface.co/stabilityai/stable-diffusion-xl-refiner-1.0).
