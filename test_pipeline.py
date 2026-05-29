"""
Validation script: tests pipeline loading, generation, and VRAM behavior.
Run with: python test_pipeline.py
"""

from __future__ import annotations

import gc
import sys
import time
import traceback

import torch
from PIL import Image

from engine import PosterEngine


def create_dummy_layout() -> Image.Image:
    """Create a simple geometric layout image to test ControlNet."""
    from PIL import ImageDraw

    img = Image.new("RGB", (1024, 1024), color=(240, 240, 240))
    draw = ImageDraw.Draw(img)
    # header bar
    draw.rectangle([100, 80, 924, 200], fill=(180, 180, 180), outline=(100, 100, 100), width=3)
    # center circle
    draw.ellipse([362, 350, 662, 650], fill=(200, 200, 200), outline=(100, 100, 100), width=3)
    # footer bar
    draw.rectangle([100, 750, 924, 870], fill=(160, 160, 160), outline=(100, 100, 100), width=3)
    return img


def create_dummy_style() -> Image.Image:
    """Create a simple colored style image to test IP-Adapter."""
    from PIL import ImageDraw

    img = Image.new("RGB", (512, 512), color=(30, 30, 60))
    draw = ImageDraw.Draw(img)
    for i in range(0, 512, 40):
        draw.rectangle([i, 0, i + 20, 512], fill=(60, 30, 80))
        draw.rectangle([0, i, 512, i + 20], fill=(80, 60, 30))
    return img


def print_vram_info(label: str) -> None:
    if not torch.cuda.is_available():
        print(f"[{label}] CUDA not available — skipping VRAM check.")
        return
    alloc = torch.cuda.memory_allocated(0) / (1024**3)
    reserved = torch.cuda.memory_reserved(0) / (1024**3)
    total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    free = total - reserved
    print(f"[{label}] VRAM: {alloc:.2f} GB allocated | {reserved:.2f} GB reserved | {free:.2f} GB free | {total:.1f} GB total")


def main():
    print("=" * 60)
    print("PosterEngine Pipeline Validation")
    print("=" * 60)

    # --- Step 1: Check environment ---
    print_vram_info("START")

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. Tests will run on CPU (slow).")

    # --- Step 2: Load pipeline ---
    print("\n[TEST 1] Loading PosterEngine ...")
    try:
        engine = PosterEngine()
        t0 = time.time()
        engine.load()
        elapsed = time.time() - t0
        print(f"[TEST 1] PASS — Pipeline loaded in {elapsed:.1f}s")
    except Exception as e:
        print(f"[TEST 1] FAIL — {e}")
        traceback.print_exc()
        sys.exit(1)

    print_vram_info("AFTER LOAD")

    # --- Step 3: Generate with text-only (no control/style) ---
    print("\n[TEST 2] Text-only generation (no ControlNet, no IP-Adapter) ...")
    try:
        result = engine.generate(
            prompt="A professional tech conference poster with blue gradient, modern abstract geometry, cinematic lighting, masterpiece, best quality",
            negative_prompt="low quality, blurry",
            num_steps=15,
            seed=99,
        )
        if result is not None and result.size == (1024, 1024):
            result.save("test_output_text_only.png")
            print(f"[TEST 2] PASS — Output size: {result.size} (saved to test_output_text_only.png)")
        else:
            print(f"[TEST 2] FAIL — Unexpected output: {result}")
    except torch.cuda.OutOfMemoryError:
        print("[TEST 2] OOM on text-only — VRAM is very tight.")
    except Exception as e:
        print(f"[TEST 2] FAIL — {e}")
        traceback.print_exc()

    print_vram_info("AFTER TEST 2")

    # --- Step 4: Generate with ControlNet + IP-Adapter ---
    print("\n[TEST 3] Full pipeline (ControlNet + IP-Adapter) ...")
    layout = create_dummy_layout()
    style = create_dummy_style()

    try:
        result = engine.generate(
            prompt="A futuristic cyberpunk poster with neon lights, dark gradient background, abstract tech patterns, masterpiece, best quality, highly detailed",
            negative_prompt="low quality, blurry",
            layout_image=layout,
            style_image=style,
            controlnet_scale=0.55,
            ip_adapter_scale=0.4,
            num_steps=20,
            seed=42,
        )
        if result is not None and result.size == (1024, 1024):
            result.save("test_output_full_pipeline.png")
            print(f"[TEST 3] PASS — Output size: {result.size} (saved to test_output_full_pipeline.png)")
        else:
            print(f"[TEST 3] FAIL — Unexpected output: {result}")
    except torch.cuda.OutOfMemoryError:
        print("[TEST 3] OOM! VRAM insufficient. Consider:")
        print("  - Switching to enable_sequential_cpu_offload() in engine.py")
        print("  - Reducing resolution to 768x768")
        print("  - Reducing inference steps further")
        sys.exit(1)
    except Exception as e:
        print(f"[TEST 3] FAIL — {e}")
        traceback.print_exc()

    print_vram_info("AFTER TEST 3")

    # --- Step 5: Cleanup ---
    print("\n[TEST 4] Cleanup and VRAM release ...")
    engine.unload()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    print_vram_info("AFTER UNLOAD")

    residual = torch.cuda.memory_allocated(0) / (1024**3) if torch.cuda.is_available() else 0
    if residual < 0.5:
        print(f"[TEST 4] PASS — VRAM properly released (residual: {residual:.2f} GB)")
    else:
        print(f"[TEST 4] WARN — Residual VRAM: {residual:.2f} GB")

    print("\n" + "=" * 60)
    print("All tests completed.")
    print("=" * 60)


if __name__ == "__main__":
    main()
