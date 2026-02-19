import torch
import cv2
import numpy as np
from PIL import Image
from pathlib import Path
import pandas as pd

GEN_DIR = "/home/linuxu/Desktop/hi eden/text2img_generate/Stable Diffusion_text2img"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------
# Load MiDaS (official)
# ---------------------------
midas_model = torch.hub.load(
    "intel-isl/MiDaS",
    "DPT_Large",
    pretrained=True
)
midas_model.to(DEVICE)
midas_model.eval()

midas_transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms"
)
midas_transform = midas_transforms.dpt_transform

# ---------------------------
# helpers
# ---------------------------
def load_image(path):
    return Image.open(path).convert("RGB")

def compute_depth(img):
    img_np = np.array(img)   # PIL → numpy

    input_batch = midas_transform(img_np).to(DEVICE)

    with torch.no_grad():
        depth = midas_model(input_batch)

    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    return depth


def shading_score(img):
    gray = np.array(img.convert("L")) / 255.0
    return gray.mean(), gray.std()

def normals_score(depth):
    dzdx = cv2.Sobel(depth, cv2.CV_64F, 1, 0, ksize=3)
    dzdy = cv2.Sobel(depth, cv2.CV_64F, 0, 1, ksize=3)
    grad = np.sqrt(dzdx**2 + dzdy**2)
    return grad.mean(), grad.std()

# ---------------------------
# Main loop
# ---------------------------
gen_paths = sorted(Path(GEN_DIR).glob("*.jpeg"))
results = []

for gen_path in gen_paths:
    img = load_image(gen_path)

    depth = compute_depth(img)
    shading_mean, shading_std = shading_score(img)
    normals_mean, normals_std = normals_score(depth)

    results.append({
        "image": gen_path.name,
        "Depth_mean": depth.mean(),
        "Depth_std": depth.std(),
        "Shading_mean": shading_mean,
        "Shading_std": shading_std,
        "Normals_mean": normals_mean,
        "Normals_std": normals_std
    })

df = pd.DataFrame(results)
df.to_csv(
    "Stable Diffusion_text2img_physics_metrics_no_reference.csv",
    index=False
)
print(df)
print("\n=== Global Averages ===")
print(df.mean(numeric_only=True))
