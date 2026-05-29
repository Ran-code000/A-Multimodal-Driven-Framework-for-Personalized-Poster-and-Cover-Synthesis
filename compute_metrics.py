"""Compute metrics, grids, and report from existing evaluation images. No generation needed."""
from __future__ import annotations
import gc, json, os, sys, time, numpy as np, torch
from PIL import Image
from evaluation import (THEMES, GROUPS, CLIPScorer, simplified_fid, aesthetic_heuristic,
                         make_comparison_grid, make_summary_grid, generate_report)

output_dir = sys.argv[1] if len(sys.argv) > 1 else "./evaluation_output"
os.makedirs(output_dir, exist_ok=True)

# Load all images
print("Loading images...")
all_results: dict = {}
for theme_key in THEMES:
    all_results[theme_key] = {}
    for _, _, group_label in GROUPS:
        path = os.path.join(output_dir, f"{theme_key}_{group_label}.png")
        if os.path.exists(path):
            all_results[theme_key][group_label] = Image.open(path).convert("RGB")
        else:
            print(f"  MISSING: {path}")
            all_results[theme_key][group_label] = Image.new("RGB", (1024, 1024), (200, 50, 50))

# Grids
print("Generating comparison grids...")
group_labels_display = ["G1: Text Only", "G2: +ControlNet", "G3: +IP-Adapter", "G4: Full System"]
for theme_key, theme_info in THEMES.items():
    images_4g = [all_results[theme_key][g[2]] for g in GROUPS]
    make_comparison_grid(images_4g, group_labels_display, theme_info["name_zh"],
                         os.path.join(output_dir, f"grid_{theme_key}.png"))

summary_images = {tk: [all_results[tk][g[2]] for g in GROUPS] for tk in THEMES}
make_summary_grid(summary_images, os.path.join(output_dir, "summary_grid.png"))

# Metrics
print("Computing CLIP Score and aesthetic metrics...")
scorer = CLIPScorer()
metrics_table = []
for theme_key, theme_info in THEMES.items():
    prompt = theme_info["prompt"]
    for use_cn, use_ip, group_label in GROUPS:
        img = all_results[theme_key][group_label]
        arr = np.array(img)
        clip_score = round(scorer.score(img, prompt), 4) if arr.std() > 10 else 0.0
        aesthetics = aesthetic_heuristic(img) if arr.std() > 10 else 1.0
        metrics_table.append(dict(
            theme=theme_info["name_zh"], theme_key=theme_key, group=group_label,
            clip_score=clip_score, aesthetics=aesthetics,
            use_controlnet=use_cn, use_ip_adapter=use_ip,
        ))
scorer.cleanup()

# FID
print("Computing FID...")
from transformers import CLIPProcessor, CLIPModel
_device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(_device)
clip_proc = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
clip_model.eval()

group_clip_embeddings: dict = {}
@torch.inference_mode()
def get_image_embedding(image):
    inputs = clip_proc(images=image, return_tensors="pt").to(_device)
    # Use vision_model + visual_projection explicitly for robustness
    vision_out = clip_model.vision_model(pixel_values=inputs["pixel_values"])
    pooled = vision_out.pooler_output  # (batch, hidden_dim)
    emb = clip_model.visual_projection(pooled)  # (batch, proj_dim)
    emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().flatten()

for _, _, group_label in GROUPS:
    embs = []
    for theme_key in THEMES:
        img = all_results[theme_key][group_label]
        if np.array(img).std() > 10:
            embs.append(get_image_embedding(img))
    group_clip_embeddings[group_label] = np.stack(embs) if embs else np.zeros((1, 768))

clip_model.to("cpu"); del clip_model, clip_proc; gc.collect()
if torch.cuda.is_available(): torch.cuda.empty_cache()

baseline_emb = group_clip_embeddings["G1_TextOnly"]
fid_scores = {}
for _, _, group_label in GROUPS:
    fid_scores[group_label] = 0.0 if group_label == "G1_TextOnly" else simplified_fid(baseline_emb, group_clip_embeddings[group_label])

# Print tables
print(f"\n{'Theme':<12} {'Group':<16} {'CLIP Score':>10} {'Aesthetics':>10}")
print("-" * 52)
for row in metrics_table:
    print(f"{row['theme']:<12} {row['group']:<16} {row['clip_score']:>10.4f} {row['aesthetics']:>10.1f}")

print(f"\n{'GROUP AVERAGES':^52}")
print(f"{'Group':<16} {'Avg CLIP':>10} {'Avg Aes':>10} {'FID vs G1':>10}")
print("-" * 52)
for _, _, gl in GROUPS:
    rows = [r for r in metrics_table if r["group"] == gl]
    print(f"{gl:<16} {np.mean([r['clip_score'] for r in rows]):>10.4f} {np.mean([r['aesthetics'] for r in rows]):>10.1f} {fid_scores[gl]:>10.2f}")

# Save JSON
with open(os.path.join(output_dir, "metrics.json"), "w", encoding="utf-8") as f:
    json.dump(dict(metrics_table=metrics_table, fid_scores=fid_scores), f, ensure_ascii=False, indent=2)

# Generate report
eval_results = dict(metrics_table=metrics_table, fid_scores=fid_scores, all_results=all_results)
generate_report(eval_results, output_dir)
print(f"\nDone. All outputs in: {output_dir}/")
