from diffusers import AutoPipelineForText2Image, DiffusionPipeline
import torch
import os


# --- Create output directory ---
output_dir = "result diffusion 1.5"
os.makedirs(output_dir, exist_ok=True)

# --- Find next available index ---
existing_files = [f for f in os.listdir(output_dir) if f.startswith("fall_") and f.endswith(".png")]

if existing_files:
    # extract numbers like fall_3.png → 3
    indices = [int(f.split("_")[1].split(".")[0]) for f in existing_files]
    next_index = max(indices) + 1
else:
    next_index = 0

output_path = os.path.join(output_dir, f"fall_{next_index}.png")

# --- Stable Diffusion v1-5 Pipeline ---
v1_pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float32,
    variant="fp16"
).to("cuda")

'''
    "Ultra-realistic high-quality photograph of a person who has just fallen off a bicycle "
    "and is now on the ground in the middle of a city street. Clean lighting, balanced exposure, "
    "low noise, natural color grading, professional photography style, smooth textures, high dynamic range, "
    "crisp details, shallow depth of field, realistic human posture, no injuries, natural expression, 8k detailed."
    '''
#  "Realistic street photography from behind, showing a person who has fallen on the street, seen from the back, with no face visible. Urban environment with other people walking in the background, natural daylight, candid moment, sharp details, professional street camera look, subtle motion blur, high-resolution 8k."
#  "Realistic back-view street photography of a person who has fallen beside a bicycle, face not visible. Cars parked along the street, passing traffic, bus stop, sidewalks, pedestrians in the background, daylight, detailed urban environment."
#    "Ultra-clean realistic street photo from behind, showing a person who has fallen on a well-maintained urban sidewalk, face not visible. Smooth pavement textures, clear lane markings, modern building facades, clean lighting, low noise, balanced exposure. Pedestrians in the background, subtle reflections on windows, controlled contrast, high dynamic range, optimized for NIQE, BRISQUE, and NIMA performance."
#    "High-quality street shot from behind of a fallen person, face hidden, surrounded by modern glass buildings and metal structures. Uniform natural light, smooth reflective surfaces, sharp architectural lines, minimal noise, clean gradients, people walking in the background, photo-realistic clarity. Ideal for high aesthetic and low distortion metrics."
#      "Candid street photo from behind, showing a fallen person, face hidden. Urban street with neatly arranged trees, clean sidewalks, soft natural daylight filtered through leaves, people walking in the background. Balanced color tones, soft highlights, smooth shadows, realistic textures, high aesthetic composition optimized for NIMA."
#     "Ultra-realistic back-view street shot of a person on the ground after a fall, face unseen. Old European-style buildings with symmetrical windows, clean stone pavements, clear architectural repetition, pedestrians in the background. Smooth gradients, low noise, perfect exposure, visually stable geometry for optimal NIQE/BRISQUE performance."
#     "High-resolution street photograph from behind, capturing a fallen person, face not visible of the fallen person. Wide boulevard with long depth, clean perspective lines, distant pedestrians, modern lampposts, soft daylight, crisp textures, low noise, high dynamic range. Composition designed for strong NIMA aesthetic scoring."
#     "Clean and realistic back-view photo of a person lying near a bicycle after a fall, face hidden of the fallen preson. Urban transit area with bus stop signage, marked lanes, uniform pavement, pedestrians in the background. High clarity, balanced exposure, minimal artifacts, smooth lighting optimized for low NIQE and BRISQUE."
#     "Photorealistic street image from behind of a person that he fall, face  person is hidden. Commercial street with shop windows, subtle reflections, clean storefronts, pedestrians passing by. Sharp details, low noise, realistic textures, evenly distributed lighting, optimized for perceptual quality metrics."


prompt = (
)

v1_image = v1_pipeline(prompt).images[0]

# --- Save image ---
v1_image.save(output_path)

print(f"Image saved at: {output_path}")
