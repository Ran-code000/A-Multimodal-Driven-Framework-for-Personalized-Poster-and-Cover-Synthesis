"""Generate a single evaluation image and exit. Called via bash loop."""
import os, sys, time
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
from engine import PosterEngine
from PIL import Image

theme_key, group_label = sys.argv[1], sys.argv[2]
use_cn = sys.argv[3] == "cn"
use_ip = sys.argv[4] == "ip"
cn_w = float(sys.argv[5])
ip_w = float(sys.argv[6])
out_path = sys.argv[7]

THEMES = {
    "ancient-beauty": dict(
        prompt="Traditional Chinese ink wash painting of a graceful classical beauty in flowing silk hanfu, holding a round silk fan, soft peach blossoms floating, elegant court atmosphere, silk scroll texture, subtle gold accents, Song dynasty aesthetic, masterpiece, best quality, highly detailed",
        neg="modern clothing, western style, ugly, deformed, low quality, blurry, watermark, text"),
    "green-landscape": dict(
        prompt="Traditional Chinese blue-green shan shui landscape painting, majestic layered mountains in jade and azure, winding river with mist, ancient pine trees, small pavilion on cliff, waterfall cascading, silk scroll texture, Wang Ximeng style, masterpiece, best quality, highly detailed",
        neg="modern buildings, people, ugly, deformed, low quality, blurry, watermark, text"),
    "cyberpunk": dict(
        prompt="Neon-lit cyberpunk city at night, rain-slicked streets reflecting purple and cyan holographic signs, towering futuristic skyscrapers, flying vehicles, atmospheric fog, cinematic lighting, blade runner aesthetic, masterpiece, best quality, highly detailed",
        neg="rural, natural landscape, daylight, ugly, deformed, low quality, blurry, watermark, text"),
    "spring-flowers": dict(
        prompt="Beautiful spring flower meadow in full bloom, cherry blossoms and tulips, warm golden morning sunlight, soft bokeh background, butterflies, macro photography aesthetic, pastel pink and yellow color palette, dreamy atmosphere, masterpiece, best quality, highly detailed",
        neg="winter, snow, dead plants, dark, ugly, deformed, low quality, blurry, watermark, text"),
}

import numpy as np, math

def make_style(theme_key: str) -> Image.Image | None:
    """Load real style reference from user's test images."""
    mapping = {
        "ancient-beauty":   "test1.jpg",
        "green-landscape":  "test2.jpg",
        "cyberpunk":        "test3.jpg",
        "spring-flowers":   "test4.jpg",
    }
    path = mapping.get(theme_key)
    if path and os.path.exists(path):
        return Image.open(path).convert("RGB").resize((512, 512))
    return None

def make_layout(theme_key: str) -> Image.Image:
    from PIL import ImageDraw
    s = 1024
    if theme_key == "ancient-beauty":
        img = Image.new("RGB", (s,s), (255,255,255)); d = ImageDraw.Draw(img)
        cx, cy = s//2, s//2
        d.ellipse([cx-80,cy-200,cx+80,cy+200], fill=(200,200,200), outline=(120,120,120), width=5)
        d.ellipse([cx-30,cy-240,cx+30,cy-180], fill=(180,180,180), outline=(120,120,120), width=3)
        for ox,oy in [(60,60),(s-60,60),(60,s-60),(s-60,s-60)]:
            d.ellipse([ox-40,oy-40,ox+40,oy+40], fill=(230,230,230), outline=(150,150,150), width=3)
        d.rectangle([120,40,s-120,100], fill=(210,210,210), outline=(130,130,130), width=4)
        return img
    elif theme_key == "green-landscape":
        img = Image.new("RGB", (s,s), (255,255,255)); d = ImageDraw.Draw(img)
        for i,(h,yo) in enumerate([(300,350),(400,300),(250,420)]):
            xo=150+i*250; d.polygon([(xo,yo+h),(xo-200,yo+450),(xo+50,yo+100),(xo+250,yo+450)], fill=(210,210,210), outline=(140,140,140), width=4)
        d.rectangle([0,600,s,s], fill=(230,230,230), outline=(160,160,160), width=3)
        d.rectangle([s//2-30,500,s//2+30,540], fill=(190,190,190), outline=(120,120,120), width=3)
        d.rectangle([120,40,s-120,100], fill=(210,210,210), outline=(130,130,130), width=4)
        return img
    elif theme_key == "cyberpunk":
        img = Image.new("RGB", (s,s), (40,40,40)); d = ImageDraw.Draw(img)
        vx,vy=s//2,s//3
        for i in range(5): d.line([(vx,vy),(100+i*200,s)], fill=(180,180,180), width=3)
        d.line([(0,vy),(s,vy)], fill=(200,200,200), width=4)
        for bx,bw,bh in [(150,120,400),(380,100,500),(600,140,350),(750,110,450)]:
            d.rectangle([bx,s-bh,bx+bw,s], fill=(150,150,150), outline=(200,200,200), width=3)
        d.rectangle([120,30,s-120,80], fill=(120,120,120), outline=(200,200,200), width=3)
        return img
    else:  # spring-flowers
        img = Image.new("RGB", (s,s), (255,255,255)); d = ImageDraw.Draw(img)
        for px,py in [(200,200),(600,180),(400,400),(750,500),(180,550),(850,300),(500,650),(300,750)]:
            r=30+hash((px,py))%50; d.ellipse([px-r,py-r,px+r,py+r], fill=(230,230,230), outline=(160,160,160), width=3)
            for a in range(0,360,72):
                rad=a*3.14159/180; d.ellipse([px+int(r*1.3*math.cos(rad))-15,py+int(r*1.3*math.sin(rad))-15,px+int(r*1.3*math.cos(rad))+15,py+int(r*1.3*math.sin(rad))+15], fill=(240,240,240), outline=(170,170,170), width=2)
        d.line([(0,800),(s,800)], fill=(200,200,200), width=4)
        d.rectangle([120,40,s-120,100], fill=(210,210,210), outline=(130,130,130), width=4)
        return img

# --- Generate ---
t = THEMES[theme_key]
layout = make_layout(theme_key) if use_cn else None
style = make_style(theme_key) if use_ip else None

e = PosterEngine()
e.load()
result = e.generate(
    prompt=t["prompt"], negative_prompt=t["neg"],
    layout_image=layout, style_image=style,
    controlnet_scale=cn_w, ip_adapter_scale=ip_w,
    num_steps=25, guidance_scale=7.5, seed=42,
)
os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
result.save(out_path)
e.unload()
print(f"OK -> {out_path}")
