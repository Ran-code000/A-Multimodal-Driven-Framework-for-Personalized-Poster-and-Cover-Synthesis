"""
Gradio web interface for the Personalized Poster & Cover Synthesis Framework.
"""

from __future__ import annotations

import gc
import os
import traceback
from typing import Optional

import gradio as gr
import torch
from PIL import Image

from engine import PosterEngine
from llm_client import PromptEnhancer


CSS = """
.container { max-width: 1200px; margin: 0 auto; }
.output-panel { border: 2px solid #4a90d9; border-radius: 12px; padding: 16px; }
.status-ok { color: #2ecc71; font-weight: bold; }
.status-warn { color: #e67e22; font-weight: bold; }
"""

_engine: Optional[PosterEngine] = None
_enhancer: Optional[PromptEnhancer] = None


def _get_engine() -> PosterEngine:
    global _engine
    if _engine is None:
        _engine = PosterEngine()
        _engine.load()
    return _engine


def _get_enhancer() -> PromptEnhancer:
    global _enhancer
    if _enhancer is None:
        _enhancer = PromptEnhancer()
    return _enhancer


def enhance_prompt(user_input: str) -> str:
    if not user_input.strip():
        return ""
    enhancer = _get_enhancer()
    try:
        return enhancer.enhance(user_input)
    except ValueError:
        return "[ERROR] DEEPSEEK_API_KEY not set. Export it or skip enhancement."
    except Exception as e:
        return f"[ERROR] Enhancement failed: {e}"


def run_generation(
    user_theme: str,
    enhanced_prompt: str,
    layout_img,
    style_img,
    ctrl_weight: float,
    ip_weight: float,
    steps: int,
    cfg_scale: float,
    seed: int,
) -> tuple:
    if not enhanced_prompt.strip():
        return None, "Please provide a prompt (or click 'Enhance Prompt' first)."

    engine = _get_engine()

    prompt = enhanced_prompt
    neg = "low quality, blurry, distorted, ugly, watermark, text, logo, worst quality, jpeg artifacts, signature"

    vram_before = engine.estimate_vram_usage()

    try:
        result = engine.generate(
            prompt=prompt,
            negative_prompt=neg,
            layout_image=layout_img,
            style_image=style_img,
            controlnet_scale=ctrl_weight,
            ip_adapter_scale=ip_weight,
            num_steps=int(steps),
            guidance_scale=cfg_scale,
            seed=int(seed),
        )
    except torch.cuda.OutOfMemoryError:
        gc.collect()
        torch.cuda.empty_cache()
        return None, "VRAM OOM — try enabling sequential_cpu_offload or reducing resolution."
    except Exception as e:
        return None, f"Generation failed: {traceback.format_exc()}"

    vram_after = engine.estimate_vram_usage()

    info = (
        f"**VRAM before:** {vram_before.get('allocated_gb', 'N/A')} GB allocated | "
        f"**after:** {vram_after.get('allocated_gb', 'N/A')} GB allocated\n"
        f"**Device:** {vram_after.get('device_name', 'Unknown')} | "
        f"**Total VRAM:** {vram_after.get('total_vram_gb', 'N/A')} GB\n"
        f"**ControlNet weight:** {ctrl_weight} | **IP-Adapter weight:** {ip_weight}"
    )
    return result, info


def build_ui():
    with gr.Blocks(css=CSS, title="Poster & Cover Synthesis") as demo:
        gr.Markdown(
            """
            # Multimodal-Driven Poster & Cover Synthesis
            Generate personalized posters and covers with SDXL + ControlNet + IP-Adapter.
            Provide a theme, optional layout sketch, and style reference.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                theme_input = gr.Textbox(
                    label=" Poster Theme",
                    placeholder='e.g. "A tech-themed campus recruitment poster in cyberpunk style"',
                    lines=3,
                )
                enhance_btn = gr.Button(" Enhance Prompt via DeepSeek", variant="primary")
                prompt_output = gr.Textbox(
                    label=" Enhanced SDXL Prompt",
                    placeholder="Click 'Enhance' to generate a refined prompt, or type your own...",
                    lines=4,
                )

                gr.Markdown("---")
                gr.Markdown("###  Generation Settings")

                with gr.Accordion("Advanced Settings", open=False):
                    with gr.Row():
                        ctrl_slider = gr.Slider(
                            0.0, 1.5, value=0.55, step=0.05, label="ControlNet Weight"
                        )
                        ip_slider = gr.Slider(
                            0.0, 1.0, value=0.4, step=0.05, label="IP-Adapter Weight"
                        )
                    with gr.Row():
                        steps_slider = gr.Slider(
                            15, 50, value=30, step=1, label="Inference Steps"
                        )
                        cfg_slider = gr.Slider(
                            1.0, 15.0, value=7.5, step=0.5, label="CFG Scale"
                        )
                    seed_input = gr.Number(value=42, label="Random Seed", precision=0)

            with gr.Column(scale=1):
                with gr.Row():
                    layout_upload = gr.Image(
                        label=" Layout Sketch (for ControlNet)",
                        type="pil",
                        height=220,
                    )
                    style_upload = gr.Image(
                        label=" Style Reference (for IP-Adapter)",
                        type="pil",
                        height=220,
                    )

                generate_btn = gr.Button(" Generate Poster", variant="primary", size="lg")
                output_image = gr.Image(
                    label="Generated Poster",
                    type="pil",
                    height=480,
                    elem_classes=["output-panel"],
                )
                status_text = gr.Markdown("")

        # ----- wiring -----
        enhance_btn.click(fn=enhance_prompt, inputs=[theme_input], outputs=[prompt_output])

        generate_btn.click(
            fn=run_generation,
            inputs=[
                theme_input,
                prompt_output,
                layout_upload,
                style_upload,
                ctrl_slider,
                ip_slider,
                steps_slider,
                cfg_slider,
                seed_input,
            ],
            outputs=[output_image, status_text],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
