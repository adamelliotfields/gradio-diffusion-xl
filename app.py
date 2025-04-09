import argparse
import os
from importlib.util import find_spec

# Use Rust-based downloader
if find_spec("hf_transfer"):
    os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"

import gradio as gr

# from huggingface_hub._snapshot_download import snapshot_download
from lib import (
    Config,
    generate,
    read_file,
    read_json,
)

# Update refresh button hover text
seed_js = """
(seed) => {
    const button = document.getElementById("refresh");
    button.style.setProperty("--seed", `"${seed}"`);
    return seed;
}
"""

# The CSS `content` attribute expects a string so we need to wrap the number in quotes
refresh_seed_js = """
() => {
    const n = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    const button = document.getElementById("refresh");
    button.style.setProperty("--seed", `"${n}"`);
    return n;
}
"""

# Update width and height on aspect ratio change
aspect_ratio_js = """
(ar, w, h) => {
    if (!ar) return [w, h];
    const [width, height] = ar.split(",");
    return [parseInt(width), parseInt(height)];
}
"""

# Show "Custom" aspect ratio when manually changing width or height, or one of the predefined ones
custom_aspect_ratio_js = """
(w, h) => {
    if (w === 768 && h === 1344) return "768,1344";
    if (w === 896 && h === 1152) return "896,1152";
    if (w === 1024 && h === 1024) return "1024,1024"; 
    if (w === 1152 && h === 896) return "1152,896";
    if (w === 1344 && h === 768) return "1344,768";
    return null;
}
"""

# Inject prompts into random function
random_prompt_js = f"""
(prompt) => {{
    const prompts = {read_json("data/prompts.json")};
    const filtered = prompts.filter(p => p !== prompt);
    return filtered[Math.floor(Math.random() * filtered.length)];
}}
"""

# Gradio styles
head_html = """
<style>
  @media (min-width: 1536px) {
    gradio-app > .gradio-container {
      max-width: 1280px !important;
    }
  }
</style>
"""

# Header
header_html = """
<div id="header">
  <div>
    <h1>
      Diffusion&nbsp;<span>XL</span>
    </h1>
    <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" aria-hidden="true" focusable="false" role="img" width="1em" height="1em" preserveAspectRatio="xMidYMid meet" viewBox="0 0 15 15">
      <path d="M7.48877 6.75C7.29015 6.75 7.09967 6.82902 6.95923 6.96967C6.81879 7.11032 6.73989 7.30109 6.73989 7.5C6.73989 7.69891 6.81879 7.88968 6.95923 8.03033C7.09967 8.17098 7.29015 8.25 7.48877 8.25C7.68738 8.25 7.87786 8.17098 8.0183 8.03033C8.15874 7.88968 8.23764 7.69891 8.23764 7.5C8.23764 7.30109 8.15874 7.11032 8.0183 6.96967C7.87786 6.82902 7.68738 6.75 7.48877 6.75ZM7.8632 0C11.2331 0 11.3155 2.6775 9.54818 3.5625C8.80679 3.93 8.47728 4.7175 8.335 5.415C8.69446 5.565 9.00899 5.7975 9.24863 6.0975C12.0195 4.5975 15 5.19 15 7.875C15 11.25 12.3265 11.325 11.4428 9.5475C11.0684 8.805 10.2746 8.475 9.57813 8.3325C9.42836 8.6925 9.19621 9 8.89665 9.255C10.3869 12.0225 9.79531 15 7.11433 15C3.74438 15 3.67698 12.315 5.44433 11.43C6.17823 11.0625 6.50774 10.2825 6.65751 9.5925C6.29056 9.4425 5.96855 9.2025 5.72891 8.9025C2.96555 10.3875 0 9.8025 0 7.125C0 3.75 2.666 3.6675 3.54967 5.445C3.92411 6.1875 4.71043 6.51 5.40689 6.6525C5.54918 6.2925 5.78882 5.9775 6.09586 5.7375C4.60559 2.97 5.1972 0 7.8632 0Z"></path>
    </svg>
  </div>
  <p>
    Image generation studio for Stable Diffusion XL.
  </p>
</div>
"""

with gr.Blocks(
    head=head_html,
    css="./app.css",
    theme=gr.themes.Default(
        # colors
        neutral_hue=gr.themes.colors.gray,
        primary_hue=gr.themes.colors.orange,
        secondary_hue=gr.themes.colors.blue,
        # sizing
        text_size=gr.themes.sizes.text_md,
        radius_size=gr.themes.sizes.radius_sm,
        spacing_size=gr.themes.sizes.spacing_md,
        # fonts
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
        font_mono=[gr.themes.GoogleFont("Ubuntu Mono"), "monospace"],
    ).set(
        layout_gap="8px",
        block_shadow="0 0 #0000",
        block_shadow_dark="0 0 #0000",
        block_background_fill=gr.themes.colors.gray.c50,
        block_background_fill_dark=gr.themes.colors.gray.c900,
    ),
) as demo:
    gr.HTML(header_html)

    with gr.Tabs():
        with gr.TabItem("🏠 Home"):
            with gr.Column():
                output_images = gr.Gallery(
                    elem_classes=["gallery"],
                    show_share_button=False,
                    object_fit="cover",
                    interactive=False,
                    show_label=False,
                    label="Output",
                    format="png",
                    columns=2,
                )
                prompt = gr.Textbox(
                    placeholder="What do you want to see?",
                    autoscroll=False,
                    show_label=False,
                    label="Prompt",
                    max_lines=3,
                    lines=3,
                )
                with gr.Row():
                    generate_btn = gr.Button("Generate", variant="primary")
                    random_btn = gr.Button(
                        elem_classes=["icon-button", "popover"],
                        variant="secondary",
                        elem_id="random",
                        min_width=0,
                        value="🎲",
                    )
                    refresh_btn = gr.Button(
                        elem_classes=["icon-button", "popover"],
                        variant="secondary",
                        elem_id="refresh",
                        min_width=0,
                        value="🔄",
                    )
                    clear_btn = gr.ClearButton(
                        elem_classes=["icon-button", "popover"],
                        components=[output_images],
                        variant="secondary",
                        elem_id="clear",
                        min_width=0,
                        value="🗑️",
                    )

        with gr.TabItem("⚙️ Settings", elem_id="settings"):
            # Prompt settings
            gr.HTML("<h3>Prompt</h3>")
            with gr.Row():
                negative_prompt = gr.Textbox(
                    value="nsfw",
                    label="Negative Prompt",
                    lines=1,
                )

            # Model settings
            gr.HTML("<h3>Model</h3>")
            with gr.Row():
                model = gr.Dropdown(
                    choices=Config.MODELS,
                    value=Config.MODEL,
                    filterable=False,
                    label="Checkpoint",
                    min_width=240,
                )
                scheduler = gr.Dropdown(
                    choices=Config.SCHEDULERS.keys(),
                    value=Config.SCHEDULER,
                    elem_id="scheduler",
                    label="Scheduler",
                    filterable=False,
                )

            # Generation settings
            gr.HTML("<h3>Generation</h3>")
            with gr.Row():
                guidance_scale = gr.Slider(
                    value=Config.GUIDANCE_SCALE,
                    label="Guidance Scale",
                    minimum=1.0,
                    maximum=15.0,
                    step=0.1,
                )
                inference_steps = gr.Slider(
                    value=Config.INFERENCE_STEPS,
                    label="Inference Steps",
                    minimum=1,
                    maximum=50,
                    step=1,
                )
                deepcache_interval = gr.Slider(
                    value=Config.DEEPCACHE_INTERVAL,
                    label="DeepCache",
                    minimum=1,
                    maximum=4,
                    step=1,
                )
            with gr.Row():
                width = gr.Slider(
                    value=Config.WIDTH,
                    label="Width",
                    minimum=512,
                    maximum=1536,
                    step=64,
                )
                height = gr.Slider(
                    value=Config.HEIGHT,
                    label="Height",
                    minimum=512,
                    maximum=1536,
                    step=64,
                )
                aspect_ratio = gr.Dropdown(
                    value=f"{Config.WIDTH},{Config.HEIGHT}",
                    label="Aspect Ratio",
                    filterable=False,
                    choices=[
                        ("Custom", None),
                        ("4:7 (768x1344)", "768,1344"),
                        ("7:9 (896x1152)", "896,1152"),
                        ("1:1 (1024x1024)", "1024,1024"),
                        ("9:7 (1152x896)", "1152,896"),
                        ("7:4 (1344x768)", "1344,768"),
                    ],
                )
            with gr.Row():
                num_images = gr.Dropdown(
                    choices=list(range(1, 5)),
                    value=Config.NUM_IMAGES,
                    filterable=False,
                    label="Images",
                )
                scale = gr.Dropdown(
                    choices=[(f"{s}x", s) for s in Config.SCALES],
                    filterable=False,
                    value=Config.SCALE,
                    label="Scale",
                )
                seed = gr.Number(
                    minimum=-1,
                    maximum=(2**64) - 1,
                    label="Seed",
                    value=-1,
                )
            with gr.Row():
                use_karras = gr.Checkbox(
                    elem_classes=["checkbox"],
                    label="Karras σ",
                    value=True,
                )
                use_refiner = gr.Checkbox(
                    elem_classes=["checkbox"],
                    label="Use refiner",
                    value=False,
                )

        with gr.TabItem("ℹ️ Info"):
            gr.Markdown(read_file("DOCS.md"))

    # Random prompt on click
    random_btn.click(None, inputs=[prompt], outputs=[prompt], js=random_prompt_js)

    # Update seed on click
    refresh_btn.click(None, inputs=[], outputs=[seed], js=refresh_seed_js)

    # Update seed button hover text
    seed.change(None, inputs=[seed], outputs=[], js=seed_js)

    # Update width and height on aspect ratio change
    aspect_ratio.input(
        None,
        inputs=[aspect_ratio, width, height],
        outputs=[width, height],
        js=aspect_ratio_js,
    )

    # Show "Custom" aspect ratio when manually changing width or height
    gr.on(
        triggers=[width.input, height.input],
        fn=None,
        inputs=[width, height],
        outputs=[aspect_ratio],
        js=custom_aspect_ratio_js,
    )

    # Generate images
    gr.on(
        triggers=[generate_btn.click, prompt.submit],
        fn=generate,
        api_name="generate",
        outputs=[output_images],
        inputs=[
            prompt,
            negative_prompt,
            seed,
            model,
            scheduler,
            width,
            height,
            guidance_scale,
            inference_steps,
            deepcache_interval,
            scale,
            num_images,
            use_karras,
            use_refiner,
        ],
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("-s", "--server", type=str, metavar="STR", default="127.0.0.1")
    parser.add_argument("-p", "--port", type=int, metavar="INT", default=7860)
    args = parser.parse_args()

    # Download all models from Hugging Face
    # token = os.environ.get("HF_TOKEN", None)
    # for repo_id, allow_patterns in Config.HF_REPOS.items():
    #     snapshot_download(
    #         repo_id=repo_id,
    #         repo_type="model",
    #         revision="main",
    #         token=token,
    #         allow_patterns=allow_patterns,
    #         ignore_patterns=None,
    #     )

    # https://www.gradio.app/docs/gradio/interface#interface-queue
    demo.queue(default_concurrency_limit=1).launch(
        server_name=args.server,
        server_port=args.port,
    )
