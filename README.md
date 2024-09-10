---
# https://huggingface.co/docs/hub/en/spaces-config-reference
title: Diffusion XL
short_description: Stable Diffusion XL image generation studio
emoji: 🦣
colorFrom: gray
colorTo: red
sdk: gradio
sdk_version: 4.41.0
python_version: 3.11.9
app_file: app.py
fullWidth: false
pinned: false
header: mini
license: apache-2.0
models:
- ai-forever/Real-ESRGAN
- cagliostrolab/animagine-xl-3.1
- fluently/Fluently-XL-Final
- madebyollin/sdxl-vae-fp16-fix
- SG161222/RealVisXL_V5.0
- stabilityai/stable-diffusion-xl-base-1.0
- stabilityai/stable-diffusion-xl-refiner-1.0
preload_from_hub:
- >-
  ai-forever/Real-ESRGAN
  RealESRGAN_x2.pth,RealESRGAN_x4.pth
- >-
  cagliostrolab/animagine-xl-3.1
  text_encoder/model.safetensors,text_encoder_2/model.safetensors,unet/diffusion_pytorch_model.safetensors,model_index.json
- >-
  fluently/Fluently-XL-Final
  text_encoder/model.safetensors,text_encoder_2/model.safetensors,unet/diffusion_pytorch_model.safetensors,model_index.json
- >-
  madebyollin/sdxl-vae-fp16-fix
  diffusion_pytorch_model.safetensors,config.json
- >-
  SG161222/RealVisXL_V5.0
  text_encoder/model.fp16.safetensors,text_encoder_2/model.fp16.safetensors,unet/diffusion_pytorch_model.fp16.safetensors,model_index.json
- >-
  stabilityai/stable-diffusion-xl-base-1.0
  text_encoder/model.fp16.safetensors,text_encoder_2/model.fp16.safetensors,unet/diffusion_pytorch_model.fp16.safetensors,model_index.json
- >-
  stabilityai/stable-diffusion-xl-refiner-1.0
  text_encoder_2/model.fp16.safetensors,unet/diffusion_pytorch_model.fp16.safetensors,model_index.json
---

# diffusion-xl

Gradio app for Stable Diffusion XL featuring:

* txt2img pipeline with refiner (img2img with IP-Adapter and ControlNet coming soon)
* Curated models (LoRAs and TIs coming soon)
* 100+ styles from sdxl_prompt_styler
* 150+ prompts from StableStudio
* Compel prompt weighting
* Multiple samplers with Karras scheduling
* DeepCache for speed
* Real-ESRGAN upscaling

## Usage

See [DOCS.md](https://huggingface.co/spaces/adamelliotfields/diffusion-xl/blob/main/DOCS.md).

## Installation

```sh
# clone
git clone https://huggingface.co/spaces/adamelliotfields/diffusion-xl.git
cd diffusion-xl
git remote set-url origin https://adamelliotfields:$HF_TOKEN@huggingface.co/spaces/adamelliotfields/diffusion-xl

# install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# gradio
python app.py --port 7860
```

## Development

See [pull requests and discussions](https://huggingface.co/docs/hub/en/repositories-pull-requests-discussions).

```sh
git fetch origin refs/pr/42:pr/42
git checkout pr/42
# ...
git add .
git commit -m "Commit message"
git push origin pr/42:refs/pr/42
```
