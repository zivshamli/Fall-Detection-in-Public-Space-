from diffusers import AutoPipelineForText2Image
import torch
import os

# --- Clear GPU memory ---
torch.cuda.empty_cache()

# --- Create output directory ---
output_dir = "result diffusion SDXL"
os.makedirs(output_dir, exist_ok=True)

# --- Find next available index ---
existing_files = [f for f in os.listdir(output_dir) if f.startswith("image_") and f.endswith(".png")]
if existing_files:
    indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
    next_index = max(indices) + 1
else:
    next_index = 0

# --- Load SDXL pipeline ---
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    variant="fp16"
).to("cuda")

# --- 10 prompts ---
prompts=[
    "Ultra-realistic high-quality photograph of a person who has just fallen off a bicycle "
    "and is now on the ground in the middle of a city street. Clean lighting, balanced exposure, "
    "low noise, natural color grading, professional photography style, smooth textures, high dynamic range, "
    "crisp details, shallow depth of field, realistic human posture, no injuries, natural expression, 8k detailed.",
    
    "Ultra-clean realistic street photograph taken from behind, showing a person sitting or lying still on a clean urban sidewalk after a fall, face not visible. "
    "Modern commercial street with straight building lines, smooth pavement textures, minimal reflections. "
    "Pedestrians calmly walking in the background, even natural daylight, low noise, smooth gradients, balanced exposure.",

    "Realistic back-view street photography of a person who has fallen beside a bicycle, face not visible. Cars parked along the street, passing traffic, bus stop, sidewalks, pedestrians in the background, daylight, detailed urban environment.",
    "Ultra-clean realistic street photo from behind, showing a person who has fallen on a well-maintained urban sidewalk, face not visible. Smooth pavement textures, clear lane markings, modern building facades, clean lighting, low noise, balanced exposure. Pedestrians in the background, subtle reflections on windows, controlled contrast, high dynamic range, optimized for NIQE, BRISQUE, and NIMA performance.",
    "High-quality street shot from behind of a fallen person, face hidden, surrounded by modern glass buildings and metal structures. Uniform natural light, smooth reflective surfaces, sharp architectural lines, minimal noise, clean gradients, people walking in the background, photo-realistic clarity. Ideal for high aesthetic and low distortion metrics.",
    "Candid street photo from behind, showing a fallen person, face hidden. Urban street with neatly arranged trees, clean sidewalks, soft natural daylight filtered through leaves, people walking in the background. Balanced color tones, soft highlights, smooth shadows, realistic textures, high aesthetic composition optimized for NIMA.",
    "Ultra-realistic back-view street shot of a person on the ground after a fall, face unseen. Old European-style buildings with symmetrical windows, clean stone pavements, clear architectural repetition, pedestrians in the background. Smooth gradients, low noise, perfect exposure, visually stable geometry for optimal NIQE/BRISQUE performance.",
    "High-resolution street photograph from behind, capturing a fallen person, face not visible of the fallen person. Wide boulevard with long depth, clean perspective lines, distant pedestrians, modern lampposts, soft daylight, crisp textures, low noise, high dynamic range. Composition designed for strong NIMA aesthetic scoring.",
    "Clean and realistic back-view photo of a person lying near a bicycle after a fall, face hidden of the fallen preson. Urban transit area with bus stop signage, marked lanes, uniform pavement, pedestrians in the background. High clarity, balanced exposure, minimal artifacts, smooth lighting optimized for low NIQE and BRISQUE.",
    "Photorealistic street image from behind of a person that he fall, face  person is hidden. Commercial street with shop windows, subtle reflections, clean storefronts, pedestrians passing by. Sharp details, low noise, realistic textures, evenly distributed lighting, optimized for perceptual quality metrics."
]

# --- Generator for reproducibility ---
generator = torch.Generator("cuda").manual_seed(31)

# --- Generate images ---
for i, prompt in enumerate(prompts):
    image = pipeline(prompt, generator=generator).images[0]
    output_path = os.path.join(output_dir, f"image_{next_index + i}.png")
    image.save(output_path)
    print(f"Saved: {output_path}")
