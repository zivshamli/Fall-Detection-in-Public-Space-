import torch
from diffusers import DiffusionPipeline,AutoencoderKL
from PIL import Image
import os

# -------------------------
# Paths
# -------------------------
input_dir = "input_images"
output_dir = "img2img_results_v1_5_1"
os.makedirs(output_dir, exist_ok=True)


# -------------------------
# Load VAE + SD v1.5 Img2Img pipeline
# -------------------------
vae = AutoencoderKL.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="vae", torch_dtype=torch.float16
)

# -------------------------
# Load SD v1.5 Img2Img pipeline
# -------------------------
pipe = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    vae=vae,
    torch_dtype=torch.float16
)
pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

# -------------------------
# Prompt settings
# -------------------------
prompt = (
    "A photorealistic urban background, modern city buildings, streets, sidewalks, cars, "
    "pedestrians, cinematic natural sunlight, realistic shadows, crisp sharp detail, "
    "ultra realistic cityscape, high-end photography"
)



# -------------------------
# Generation settings (high-fidelity)
# -------------------------
num_inference_steps = 40
guidance_scale = 7.5
strength = 0.55
num_variations = 1

# Fixed generator for reproducibility
#generator = torch.Generator("cuda").manual_seed(seed)

# -------------------------
# Process all images
# -------------------------
for img_name in os.listdir(input_dir):
    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_dir, img_name)
    try:
        init_image = Image.open(img_path).convert("RGB")
        init_image = init_image.resize((512, 512))  # SD v1.5 native size
    except Exception as e:
        print(f"Skipping corrupted image: {img_name} ({e})")
        continue

    print(f"Processing {img_name}...")

    base = os.path.splitext(img_name)[0]

    for i in range(num_variations):
        print(f"  Variation {i+1}/{num_variations}...")
        try:
            out = pipe(
                prompt=prompt,
                init_image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=torch.Generator("cuda").manual_seed(i)
            )

            if out is None or out.images is None:
                print(f"  FAILED: pipeline returned None for {img_name}")
                continue

            result = out.images[0]
            save_path = os.path.join(output_dir, f"{base}_var{i+1}.png")
            result.save(save_path)
            print(f"  Saved to {save_path}")

        except Exception as e:
            print(f"  FAILED on variation {i+1} for {img_name}: {e}")

print("All images processed successfully!")
