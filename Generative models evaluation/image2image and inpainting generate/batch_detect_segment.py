import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# =========================
# Paths
# =========================
INPUT_DIR = "mask_not_fall/new"
OUTPUT_DIR = "mask_not_fall_result"
SAM_CHECKPOINT = "sam_vit_h_4b8939.pth"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Models
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

detector = YOLO("yolov8m.pt")  # you can upgrade to yolov8m.pt
sam = sam_model_registry["vit_h"](checkpoint=SAM_CHECKPOINT)
sam.to(device)
predictor = SamPredictor(sam)

# =========================
# Helper functions
# =========================
def expand_box(box, img_w, img_h, scale=0.2):
    x1, y1, x2, y2 = box
    dx = int((x2 - x1) * scale)
    dy = int((y2 - y1) * scale)
    return np.array([
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(img_w, x2 + dx),
        min(img_h, y2 + dy)
    ])

def person_score(p, img_center):
    center_dist = np.linalg.norm(np.array(p["center"]) - img_center)
    area = (p["box"][2] - p["box"][0]) * (p["box"][3] - p["box"][1])
    return center_dist - 0.002 * area

# =========================
# Process images
# =========================
for filename in os.listdir(INPUT_DIR):
    if not filename.lower().endswith((".jpg", ".png", ".jpeg")):
        continue

    input_path = os.path.join(INPUT_DIR, filename)
    output_path = os.path.join(
        OUTPUT_DIR,
        os.path.splitext(filename)[0] + "_fg.png"
    )

    print(f"\n📷 Processing {filename}")

    image = cv2.imread(input_path)
    if image is None:
        print("❌ Could not load image")
        continue

    img_h, img_w = image.shape[:2]
    img_center = np.array([img_w / 2, img_h / 2])
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # -------------------------
    # YOLO detection (low conf!)
    # -------------------------
    results = detector(image_rgb, conf=0.1, iou=0.45)[0]

    person_boxes = []
    for box, cls, conf in zip(
        results.boxes.xyxy,
        results.boxes.cls,
        results.boxes.conf
    ):
        if int(cls) == 0:  # person
            x1, y1, x2, y2 = box.cpu().numpy()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            person_boxes.append({
                "box": np.array([x1, y1, x2, y2], dtype=int),
                "center": (cx, cy),
                "conf": conf.item()
            })

    # -------------------------
    # Select person
    # -------------------------
    if len(person_boxes) > 0:
        selected = min(
            person_boxes,
            key=lambda p: person_score(p, img_center)
        )
        box = selected["box"]
        print("✅ Person selected:", box)
    else:
        print("⚠️ YOLO failed — using center fallback")
        bw, bh = img_w // 3, img_h // 3
        box = np.array([
            img_w // 2 - bw // 2,
            img_h // 2 - bh // 2,
            img_w // 2 + bw // 2,
            img_h // 2 + bh // 2
        ])

    box = expand_box(box, img_w, img_h)

    # -------------------------
    # SAM segmentation
    # -------------------------
    predictor.set_image(image_rgb)
    masks, _, _ = predictor.predict(
        box=box,
        multimask_output=False
    )

    mask = masks[0]

    # -------------------------
    # Save RGBA foreground
    # -------------------------
    alpha = (mask * 255).astype(np.uint8)
    rgba = np.dstack((image_rgb, alpha))
    rgba = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA)

    cv2.imwrite(output_path, rgba)
    print(f"💾 Saved: {output_path}")

print("\n🎉 Batch processing completed")
