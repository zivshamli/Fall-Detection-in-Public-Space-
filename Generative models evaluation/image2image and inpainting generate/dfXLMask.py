import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from PIL import Image
import numpy as np
import os

# Clear GPU memory
torch.cuda.empty_cache()

# -------------------------
# Paths
# -------------------------
input_dir = "input_images"        # folder with original images
mask_dir = "masks"                # folder with masks (white=keep, black=replace)
output_dir = "img2img_SXDL_masked_results"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Load SDXL VAE + Img2Img pipeline
# -------------------------
vae = AutoencoderKL.from_pretrained("stabilityai/sdxl-vae", torch_dtype=torch.float16)

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    vae=vae,
    torch_dtype=torch.float16
).to("cuda")

pipe.enable_xformers_memory_efficient_attention()

# -------------------------
# Prompt and settings
# -------------------------
prompt = (
    "A photorealistic urban city street with modern buildings, sidewalks, cars, "
    "and pedestrians walking naturally in the background, people interacting casually, "
    "cinematic sunlight, realistic shadows, crisp sharp details, ultra-realistic cityscape, "
    "high-end professional photography, subtle depth of field to place people in the background"
)

strength = 0.55
num_inference_steps = 40
guidance_scale = 7.5
num_variations = 1

# -------------------------
# Process images
# -------------------------
for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_dir, img_name)
    mask_path = os.path.join(mask_dir, img_name)

    if not os.path.exists(mask_path):
        print(f"Skipping {img_name} - mask not found")
        continue

    # Load images
    init_image = Image.open(img_path).convert("RGB").resize((1024, 1024))
    mask_image = Image.open(mask_path).convert("L").resize((1024, 1024))

    # Convert to numpy to check inversion if needed
    mask_np = np.array(mask_image)
    # Invert mask if necessary (make sure black=replace, white=keep)
    if mask_np.max() < 255:
        mask_np = 255 - mask_np
        mask_image = Image.fromarray(mask_np)

    base = os.path.splitext(img_name)[0]

    for i in range(num_variations):
        print(f"Processing {img_name}, variation {i+1}/{num_variations}...")

        out = pipe(
            prompt=prompt,
            image=init_image,
            mask_image=mask_image,
            strength=strength,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=torch.manual_seed(i)
        )

        result = out.images[0]
        save_path = os.path.join(output_dir, f"{base}_var{i+1}.png")
        result.save(save_path)

print("All masked SDXL images generated successfully with realistic backgrounds!")
