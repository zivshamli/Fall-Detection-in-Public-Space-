import os
import torch
from PIL import Image
import torchvision.transforms as T
import pyiqa
import pandas as pd

# -----------------------------
# Settings
# -----------------------------
IMAGE_DIR = "/home/linuxu/Desktop/hi eden/text2img_generate/Stable Diffusion_text2img"               # directory containing your images
RESULT_DIR = "result"              # output directory
OUTPUT_CSV = os.path.join(RESULT_DIR, "result Stable Diffusion_text2img .csv")

device = "cuda" if torch.cuda.is_available() else "cpu"

# Create result folder if missing
os.makedirs(RESULT_DIR, exist_ok=True)

# Image transform
transform = T.Compose([T.ToTensor()])

# -----------------------------
# Load IQA models once
# -----------------------------
niqe = pyiqa.create_metric('niqe').to(device)
brisque = pyiqa.create_metric('brisque').to(device)
nima = pyiqa.create_metric('nima').to(device)

# -----------------------------
# Scan directory

# -----------------------------
images = [f for f in os.listdir(IMAGE_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg"))]

records = []

for img_name in images:
    path = os.path.join(IMAGE_DIR, img_name)
    img = Image.open(path).convert("RGB")

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        niqe_score = float(niqe(img_tensor))
        brisque_score = float(brisque(img_tensor))
        nima_score = float(nima(img_tensor))

    print(f"{img_name}: NIQE={niqe_score:.3f}, BRISQUE={brisque_score:.3f}, NIMA={nima_score:.3f}")

    records.append({
        "image": img_name,
        "NIQE": niqe_score,
        "BRISQUE": brisque_score,
        "NIMA": nima_score,
    })

# -----------------------------
# Save CSV into result/
# -----------------------------
df = pd.DataFrame(records)
df.to_csv(OUTPUT_CSV, index=False)
print(f"\nSaved CSV to: {OUTPUT_CSV}")

# -----------------------------
# Compute averages
# -----------------------------
avg_niqe = df["NIQE"].mean()
avg_brisque = df["BRISQUE"].mean()
avg_nima = df["NIMA"].mean()

print("\n------ AVERAGE SCORES ------")
print(f"Avg NIQE:     {avg_niqe:.3f}")
print(f"Avg BRISQUE:  {avg_brisque:.3f}")
print(f"Avg NIMA:     {avg_nima:.3f}")
