import os
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO

# -------------------------
# Paths
# -------------------------
input_dir = "input_images"
mask_dir = "masks"
os.makedirs(mask_dir, exist_ok=True)

# -------------------------
# Load YOLOv8 segmentation model
# -------------------------
model = YOLO("yolov8n-seg.pt")  # make sure you have downloaded the model

# -------------------------
# Process images
# -------------------------
for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_dir, img_name)
    print(f"Processing {img_name}...")

    # Load original image
    img = Image.open(img_path).convert("RGB")
    img_width, img_height = img.size

    # Run YOLO segmentation
    results = model(img_path)[0]

    # Initialize empty mask (white = background)
    mask = np.ones((img_height, img_width), dtype=np.uint8) * 255

    # Loop through detected objects and pick "person" (class 0 in COCO)
    for i, cls in enumerate(results.boxes.cls):
        if int(cls) == 0:  # person
            person_mask = results.masks.data[i].cpu().numpy()  # 0/1 mask

            # Resize mask to match original image
            person_mask_resized = cv2.resize(
                person_mask, 
                (img_width, img_height), 
                interpolation=cv2.INTER_NEAREST
            )

            # Invert mask: black = keep person
            mask[person_mask_resized > 0] = 0

    # Save mask
    mask_image = Image.fromarray(mask)
    mask_save_path = os.path.join(mask_dir, img_name)
    mask_image.save(mask_save_path)
    print(f"Saved mask: {mask_save_path}")

print("All masks created in 'masks/' folder!")
