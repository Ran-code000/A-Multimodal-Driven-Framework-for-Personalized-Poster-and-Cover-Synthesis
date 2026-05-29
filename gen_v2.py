"""Standalone generator — no engine.py, manual memory management for Blackwell GPU."""
import os, gc, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import numpy as np
import torch, cv2
from PIL import Image, ImageDraw

theme_key = sys.argv[1]   # ancient-beauty | green-landscape | cyberpunk | spring-flowers
group_label = sys.argv[2]  # G1_TextOnly | G2_ControlNet | G3_IPAdapter | G4_FullSystem

THEMES = {
    "ancient-beauty": dict(
        prompt="Traditional Chinese ink wash painting of a graceful classical beauty in flowing silk hanfu, holding a round silk fan, soft peach blossoms floating, elegant court atmosphere, silk scroll texture, subtle gold accents, Song dynasty aesthetic, masterpiece, best quality, highly detailed",
        neg="modern clothing, western style, ugly, deformed, low quality, blurry, watermark, text",
        style="test1.jpg"),
    "green-landscape": dict(
        prompt="Traditional Chinese blue-green shan shui landscape painting, majestic layered mountains in jade and azure, winding river with mist, ancient pine trees, small pavilion on cliff, waterfall cascading, silk scroll texture, Wang Ximeng style, masterpiece, best quality, highly detailed",
        neg="modern buildings, people, ugly, deformed, low quality, blurry, watermark, text",
        style="test2.jpg"),
    "cyberpunk": dict(
        prompt="Neon-lit cyberpunk city at night, rain-slicked streets reflecting purple and cyan holographic signs, towering futuristic skyscrapers, flying vehicles, atmospheric fog, cinematic lighting, blade runner aesthetic, masterpiece, best quality, highly detailed",
        neg="rural, natural landscape, daylight, ugly, deformed, low quality, blurry, watermark, text",
        style="test3.jpg"),
    "spring-flowers": dict(
        prompt="Beautiful spring flower meadow in full bloom, cherry blossoms and tulips, warm golden morning sunlight, soft bokeh background, butterflies, macro photography aesthetic, pastel pink and yellow color palette, dreamy atmosphere, masterpiece, best quality, highly detailed",
        neg="winter, snow, dead plants, dark, ugly, deformed, low quality, blurry, watermark, text",
        style="test4.jpg"),
}

GROUPS = {
    "G1_TextOnly":    dict(cn=False, ip=False, cn_w=0.0,  ip_w=0.0),
    "G2_ControlNet":  dict(cn=True,  ip=False, cn_w=0.18, ip_w=0.0),
    "G3_IPAdapter":   dict(cn=False, ip=True,  cn_w=0.0,  ip_w=0.33),
    "G4_FullSystem":  dict(cn=True,  ip=True,  cn_w=0.18, ip_w=0.33),
}

import math

def make_layout(tk: str) -> Image.Image:
    s = 1024
    img = Image.new("RGB", (s,s), (255,255,255)); d = ImageDraw.Draw(img)
    if tk == "ancient-beauty":
        cx,cy=s//2,s//2
        d.ellipse([cx-80,cy-200,cx+80,cy+200], fill=(200,200,200), outline=(120,120,120), width=5)
        d.ellipse([cx-30,cy-240,cx+30,cy-180], fill=(180,180,180), outline=(120,120,120), width=3)
        for ox,oy in [(60,60),(s-60,60),(60,s-60),(s-60,s-60)]:
            d.ellipse([ox-40,oy-40,ox+40,oy+40], fill=(230,230,230), outline=(150,150,150), width=3)
        d.rectangle([120,40,s-120,100], fill=(210,210,210), outline=(130,130,130), width=4)
    elif tk == "green-landscape":
        for i,(h,yo) in enumerate([(300,350),(400,300),(250,420)]):
            xo=150+i*250
            d.polygon([(xo,yo+h),(xo-200,yo+450),(xo+50,yo+100),(xo+250,yo+450)], fill=(210,210,210), outline=(140,140,140), width=4)
        d.rectangle([0,600,s,s], fill=(230,230,230), outline=(160,160,160), width=3)
        d.rectangle([s//2-30,500,s//2+30,540], fill=(190,190,190), outline=(120,120,120), width=3)
        d.rectangle([120,40,s-120,100], fill=(210,210,210), outline=(130,130,130), width=4)
    elif tk == "cyberpunk":
        img = Image.new("RGB", (s,s), (40,40,40)); d = ImageDraw.Draw(img)
        vx,vy=s//2,s//3
        for i in range(5): d.line([(vx,vy),(100+i*200,s)], fill=(180,180,180), width=3)
        d.line([(0,vy),(s,vy)], fill=(200,200,200), width=4)
        for bx,bw,bh in [(150,120,400),(380,100,500),(600,140,350),(750,110,450)]:
            d.rectangle([bx,s-bh,bx+bw,s], fill=(150,150,150), outline=(200,200,200), width=3)
        d.rectangle([120,30,s-120,80], fill=(120,120,120), outline=(200,200,200), width=3)
    else:  # spring-flowers
        for px,py in [(200,200),(600,180),(400,400),(750,500),(180,550),(850,300),(500,650),(300,750)]:
            r=30+hash((px,py))%50
            d.ellipse([px-r,py-r,px+r,py+r], fill=(230,230,230), outline=(160,160,160), width=3)
            for a in range(0,360,72):
                rad=a*3.14159/180
                d.ellipse([px+int(r*1.3*math.cos(rad))-15,py+int(r*1.3*math.sin(rad))-15,px+int(r*1.3*math.cos(rad))+15,py+int(r*1.3*math.sin(rad))+15], fill=(240,240,240), outline=(170,170,170), width=2)
        d.line([(0,800),(s,800)], fill=(200,200,200), width=4)
        d.rectangle([120,40,s-120,100], fill=(210,210,210), outline=(130,130,130), width=4)
    return img

def canny(img: Image.Image) -> Image.Image:
    a = np.array(img)
    e = cv2.Canny(a, 100, 200)
    return Image.fromarray(np.concatenate([e[:,:,None]]*3, axis=2))

# --- Main ---
t = THEMES[theme_key]
g = GROUPS[group_label]

print(f"[gen_v2] {theme_key}/{group_label} CN={g['cn_w']} IP={g['ip_w']}", flush=True)

# Load layout
layout = canny(make_layout(theme_key)) if g["cn"] else canny(Image.new("RGB", (1024,1024), (0,0,0)))

# Load style (always provide an image for pipeline compat; scale=0 disables effect)
style_img = Image.new("RGB", (224, 224), (128, 128, 128))  # neutral grey
if g["ip"] and os.path.exists(t["style"]):
    style_img = Image.open(t["style"]).convert("RGB").resize((512, 512))

from diffusers import StableDiffusionXLControlNetPipeline, ControlNetModel, DPMSolverMultistepScheduler

print("  Loading ControlNet...", flush=True)
cn_model = ControlNetModel.from_pretrained("diffusers/controlnet-canny-sdxl-1.0", torch_dtype=torch.float16)

print("  Loading SDXL pipeline...", flush=True)
pipe = StableDiffusionXLControlNetPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    controlnet=cn_model,
    torch_dtype=torch.float16, variant="fp16", use_safetensors=True,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config, use_karras_sigmas=True)

print("  Loading IP-Adapter...", flush=True)
pipe.load_ip_adapter("h94/IP-Adapter", subfolder="sdxl_models", weight_name="ip-adapter_sdxl.safetensors", low_cpu_mem_usage=True)

# --- Memory: move pipeline to GPU (monolithic, with expandable_segments) ---
print("  Moving pipeline to CUDA...", flush=True)
pipe.to("cuda")
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()
gc.collect(); torch.cuda.empty_cache()

vram = torch.cuda.memory_allocated(0)/1024**3
print(f"  VRAM after load: {vram:.1f} GB", flush=True)

# --- Generate ---
print("  Generating...", flush=True)
gen = torch.Generator(device="cuda").manual_seed(42)
pipe.set_ip_adapter_scale(g["ip_w"])

with torch.inference_mode():
    result = pipe(
        prompt=t["prompt"], negative_prompt=t["neg"],
        image=layout,
        ip_adapter_image=style_img,
        controlnet_conditioning_scale=g["cn_w"],
        num_inference_steps=25, guidance_scale=7.5,
        generator=gen, height=1024, width=1024,
        control_guidance_start=0.0, control_guidance_end=0.9,
    )

out_path = f"evaluation_v2_output/{theme_key}_{group_label}.png"
os.makedirs("evaluation_v2_output", exist_ok=True)
result.images[0].save(out_path)
print(f"  OK -> {out_path}", flush=True)

# Cleanup
del pipe, cn_model; gc.collect(); torch.cuda.empty_cache()
