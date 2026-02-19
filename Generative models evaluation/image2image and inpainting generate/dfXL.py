import torch
from diffusers import StableDiffusionXLImg2ImgPipeline, AutoencoderKL
from PIL import Image
import os

torch.cuda.empty_cache()

# -------------------------
# Paths
# -------------------------
input_dir = "input_images"
output_dir = "img2img_SXDL_results"
os.makedirs(output_dir, exist_ok=True)

# -------------------------
# Load SDXL Base Model + SDXL VAE
# -------------------------
model_name = "stabilityai/stable-diffusion-xl-base-1.0"

vae = AutoencoderKL.from_pretrained(
    "stabilityai/sdxl-vae",
    torch_dtype=torch.float16
)

pipe = StableDiffusionXLImg2ImgPipeline.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    vae=vae,
    use_safetensors=True
)

pipe = pipe.to("cuda")
pipe.enable_xformers_memory_efficient_attention()

prompt = (
    "A photorealistic urban background, modern city buildings, streets, sidewalks, cars, "
    "pedestrians, cinematic natural sunlight, realistic shadows, crisp sharp detail, "
    "ultra realistic cityscape, high-end photography"
)

num_inference_steps = 40
guidance_scale = 7.5
strength = 0.55

num_variations = 1

# -------------------------
# Process directory
# -------------------------
for img_name in os.listdir(input_dir):

    if not img_name.lower().endswith((".png", ".jpg", ".jpeg")):
        continue

    img_path = os.path.join(input_dir, img_name)

    try:
        init_image = Image.open(img_path).convert("RGB")
        init_image = init_image.resize((1024, 1024))  # SDXL native size
    except:
        print(f"Skipping corrupted image: {img_name}")
        continue

    print(f"Processing {img_name}...")

    base = os.path.splitext(img_name)[0]

    for i in range(num_variations):

        print(f"  Variation {i+1}/{num_variations}...")

        try:
            out = pipe(
                prompt=prompt,
                image=init_image,
                strength=strength,
                num_inference_steps=num_inference_steps,
                generator=torch.manual_seed(i)
            )

            result = out.images[0]
            save_path = os.path.join(output_dir, f"{base}_var{i+1}.png")
            result.save(save_path)

        except Exception as e:
            print(f"  FAILED on variation {i+1} for {img_name}: {e}")

print("All images generated successfully with SDXL !")
