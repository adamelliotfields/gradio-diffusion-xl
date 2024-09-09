import argparse
import json
import random

import gradio as gr

from lib import Config, async_call, generate

# the CSS `content` attribute expects a string so we need to wrap the number in quotes
refresh_seed_js = """
() => {
    const n = Math.floor(Math.random() * Number.MAX_SAFE_INTEGER);
    const button = document.getElementById("refresh");
    button.style.setProperty("--seed", `"${n}"`);
    return n;
}
"""

seed_js = """
(seed) => {
    const button = document.getElementById("refresh");
    button.style.setProperty("--seed", `"${seed}"`);
    return seed;
}
"""

aspect_ratio_js = """
(ar, w, h) => {
    if (!ar) return [w, h];
    const [width, height] = ar.split(",");
    return [parseInt(width), parseInt(height)];
}
"""


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as file:
        return file.read()


def random_fn():
    prompts = read_file("data/prompts.json")
    prompts = json.loads(prompts)
    return gr.Textbox(value=random.choice(prompts))


async def generate_fn(*args):
    if len(args) > 0:
        prompt = args[0]
    else:
        prompt = None
    if prompt is None or prompt.strip() == "":
        raise gr.Error("ValueError: Invalid prompt")
    try:
        images = await async_call(
            generate,
            *args,
            Info=gr.Info,
            Error=gr.Error,
            progress=gr.Progress(),
        )
    except RuntimeError:
        raise gr.Error("RuntimeError: Please try again later")
    return images


with gr.Blocks(
    head=read_file("./partials/head.html"),
    css="./app.css",
    js="./app.js",
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
        font=[gr.themes.GoogleFont("Inter"), *Config.SANS_FONTS],
        font_mono=[gr.themes.GoogleFont("Ubuntu Mono"), *Config.MONO_FONTS],
    ).set(
        layout_gap="8px",
        block_shadow="0 0 #0000",
        block_shadow_dark="0 0 #0000",
        block_background_fill=gr.themes.colors.gray.c50,
        block_background_fill_dark=gr.themes.colors.gray.c900,
    ),
) as demo:
    gr.HTML(read_file("./partials/intro.html"))

    with gr.Accordion(
        elem_classes=["accordion"],
        elem_id="menu",
        label="Menu",
        open=False,
    ):
        with gr.Tabs():
            with gr.TabItem("⚙️ Settings"):
                with gr.Group():
                    negative_prompt = gr.Textbox(
                        value=None,
                        label="Negative Prompt",
                        placeholder="ugly, bad",
                        lines=2,
                    )

                    with gr.Row():
                        model = gr.Dropdown(
                            choices=Config.MODELS,
                            filterable=False,
                            value=Config.MODEL,
                            label="Model",
                            min_width=240,
                        )
                        scheduler = gr.Dropdown(
                            choices=Config.SCHEDULERS.keys(),
                            value=Config.SCHEDULER,
                            elem_id="scheduler",
                            label="Scheduler",
                            filterable=False,
                        )

                    with gr.Row():
                        styles = json.loads(read_file("data/styles.json"))
                        style = gr.Dropdown(
                            value=Config.STYLE,
                            label="Style",
                            choices=[("None", None)] + [(s["name"], s["id"]) for s in styles],
                        )
                        # embeddings = gr.Dropdown(
                        #     elem_id="embeddings",
                        #     label="Embeddings",
                        #     choices=[(f"<{e}>", e) for e in Config.EMBEDDINGS],
                        #     multiselect=True,
                        #     value=[Config.EMBEDDING],
                        #     min_width=240,
                        # )

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
                            choices=[
                                ("Custom", None),
                                ("4:7 (768x1344)", "768,1344"),
                                ("7:9 (896x1152)", "896,1152"),
                                ("1:1 (1024x1024)", "1024,1024"),
                                ("9:7 (1152x896)", "1152,896"),
                                ("7:4 (1344x768)", "1344,768"),
                            ],
                            value="896,1152",
                            filterable=False,
                            label="Aspect Ratio",
                        )

                    with gr.Row():
                        file_format = gr.Dropdown(
                            choices=["png", "jpeg", "webp"],
                            label="File Format",
                            filterable=False,
                            value="png",
                        )
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
                            value=Config.SEED,
                            label="Seed",
                            minimum=-1,
                            maximum=(2**64) - 1,
                        )

                    with gr.Row():
                        use_karras = gr.Checkbox(
                            elem_classes=["checkbox"],
                            label="Karras σ",
                            value=True,
                        )
                        use_refiner = gr.Checkbox(
                            elem_classes=["checkbox"],
                            label="Refiner",
                            value=True,
                        )

    # Main content
    with gr.Column(elem_id="content"):
        with gr.Group():
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
                placeholder="corgi, beach, 8k",
                autoscroll=False,
                show_label=False,
                label="Prompt",
                max_lines=3,
                lines=3,
            )

        # Buttons
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

    random_btn.click(random_fn, inputs=[], outputs=[prompt], show_api=False)

    refresh_btn.click(None, inputs=[], outputs=[seed], js=refresh_seed_js)

    seed.change(None, inputs=[seed], outputs=[], js=seed_js)

    file_format.change(
        lambda f: (gr.Gallery(format=f), gr.Image(format=f), gr.Image(format=f)),
        inputs=[file_format],
        outputs=[output_images],
        show_api=False,
    )

    # input events are only user input; change events are both user and programmatic
    aspect_ratio.input(
        None,
        inputs=[aspect_ratio, width, height],
        outputs=[width, height],
        js=aspect_ratio_js,
    )

    # show "Custom" aspect ratio when manually changing width or height
    gr.on(
        triggers=[width.input, height.input],
        fn=None,
        inputs=[],
        outputs=[aspect_ratio],
        js="() => { return null; }",
    )

    gr.on(
        triggers=[generate_btn.click, prompt.submit],
        fn=generate_fn,
        api_name="generate",
        concurrency_limit=5,
        outputs=[output_images],
        inputs=[
            prompt,
            negative_prompt,
            style,
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
    parser.add_argument("-s", "--server", type=str, metavar="STR", default="0.0.0.0")
    parser.add_argument("-p", "--port", type=int, metavar="INT", default=7860)
    args = parser.parse_args()

    # https://www.gradio.app/docs/gradio/interface#interface-queue
    demo.queue().launch(
        server_name=args.server,
        server_port=args.port,
    )
