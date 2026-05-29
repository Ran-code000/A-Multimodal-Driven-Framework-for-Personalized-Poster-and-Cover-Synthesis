"""
PosterEngine: Core image generation pipeline integrating SDXL, ControlNet-Canny, and IP-Adapter.
Memory-optimized for 16GB VRAM GPUs.
"""

from __future__ import annotations

import os

_HF_MIRROR = "https://hf-mirror.com"
os.environ.setdefault("HF_ENDPOINT", _HF_MIRROR)
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gc
import warnings
from typing import Optional

import cv2
import numpy as np
import torch
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)


def _canny_preprocess(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    img = np.array(image)
    img = cv2.Canny(img, low, high)
    img = img[:, :, None]
    img = np.concatenate([img, img, img], axis=2)
    return Image.fromarray(img)


def _neutral_ip_image(width: int = 224, height: int = 224) -> Image.Image:
    return Image.new("RGB", (width, height), color=(128, 128, 128))


class PosterEngine:
    def __init__(
        self,
        sdxl_model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        controlnet_model: str = "diffusers/controlnet-canny-sdxl-1.0",
        ip_adapter_model: str = "h94/IP-Adapter",
        device: Optional[str] = None,
    ):
        self.sdxl_model = sdxl_model
        self.controlnet_model = controlnet_model
        self.ip_adapter_model = ip_adapter_model

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.pipe = None
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return

        from diffusers import (
            StableDiffusionXLControlNetPipeline,
            ControlNetModel,
            DPMSolverMultistepScheduler,
        )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        controlnet = ControlNetModel.from_pretrained(
            self.controlnet_model,
            torch_dtype=torch.float16,
        )
        gc.collect()

        print("[PosterEngine] Loading SDXL ControlNet pipeline ...")
        self.pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
            self.sdxl_model,
            controlnet=controlnet,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
        )
        gc.collect()

        self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            self.pipe.scheduler.config,
            use_karras_sigmas=True,
        )

        print("[PosterEngine] Loading IP-Adapter ...")
        self.pipe.load_ip_adapter(
            self.ip_adapter_model,
            subfolder="sdxl_models",
            weight_name="ip-adapter_sdxl.safetensors",
            low_cpu_mem_usage=True,
        )

        self._apply_memory_optimizations()
        self._loaded = True
        print("[PosterEngine] Pipeline loaded successfully.")

    def _apply_memory_optimizations(self) -> None:
        if self.device == "cpu":
            return

        self.pipe.enable_model_cpu_offload()
        self.pipe.enable_vae_slicing()
        self.pipe.enable_vae_tiling()

        # maybe_free_model_hooks re-calls enable_model_cpu_offload which
        # does self.to("cpu") — this triggers CPU allocator issues on Blackwell.
        self.pipe.maybe_free_model_hooks = lambda: None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[PosterEngine] Memory optimizations applied (model_cpu_offload + vae_slicing + vae_tiling).")

    def unload(self) -> None:
        if self.pipe is not None:
            del self.pipe
            self.pipe = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("[PosterEngine] Pipeline unloaded from memory.")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: str = "",
        layout_image: Optional[Image.Image] = None,
        style_image: Optional[Image.Image] = None,
        controlnet_scale: float = 0.55,
        ip_adapter_scale: float = 0.4,
        num_steps: int = 30,
        guidance_scale: float = 7.5,
        seed: int = 42,
        height: int = 1024,
        width: int = 1024,
    ) -> Image.Image:
        if not self._loaded:
            self.load()

        if layout_image is None:
            layout_image = Image.new("RGB", (width, height), color=(0, 0, 0))
        layout_image = _canny_preprocess(layout_image)

        generator_device = "cuda" if torch.cuda.is_available() else "cpu"
        generator = torch.Generator(device=generator_device).manual_seed(seed)

        # IP-Adapter: when no real style image is provided, use a neutral grey
        # image with scale=0.0 so IP-Adapter has zero effect. The neutral image
        # is still needed for pipeline compatibility (UNet expects image_embeds).
        if style_image is None:
            style_image = _neutral_ip_image()
            self.pipe.set_ip_adapter_scale(0.0)
        else:
            self.pipe.set_ip_adapter_scale(ip_adapter_scale)

        result = self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=layout_image,
            ip_adapter_image=style_image,
            controlnet_conditioning_scale=controlnet_scale,
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            height=height,
            width=width,
            control_guidance_start=0.0,
            control_guidance_end=0.9,
        )

        return result.images[0]

    @staticmethod
    def estimate_vram_usage() -> dict:
        info = {}
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            info["device_name"] = props.name
            info["total_vram_gb"] = round(props.total_memory / (1024**3), 1)
            info["allocated_gb"] = round(torch.cuda.memory_allocated(0) / (1024**3), 2)
            info["reserved_gb"] = round(torch.cuda.memory_reserved(0) / (1024**3), 2)
        return info
