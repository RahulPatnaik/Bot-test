import os
from PIL import Image
import torch
from diffusers.pipelines import FluxPipeline
from diffusers import FluxTransformer2DModel
from src.condition import Condition
from src.generate import generate

# Initialize global variables
pipe = None
use_int8 = False


def get_gpu_memory():
    return torch.cuda.get_device_properties(0).total_memory / 1024**3


def init_pipeline():
    """Initialize the pipeline based on GPU memory and settings."""
    global pipe
    if use_int8 or get_gpu_memory() < 33:
        transformer_model = FluxTransformer2DModel.from_pretrained(
            "sayakpaul/flux.1-schell-int8wo-improved",
            torch_dtype=torch.bfloat16,
            use_safetensors=False,
        )
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell",
            transformer=transformer_model,
            torch_dtype=torch.bfloat16,
        )
    else:
        pipe = FluxPipeline.from_pretrained(
            "black-forest-labs/FLUX.1-schnell", torch_dtype=torch.bfloat16
        )
    pipe = pipe.to("cuda")
    pipe.load_lora_weights(
        "Yuanshi/OminiControl",
        weight_name="omini/subject_512.safetensors",
        adapter_name="subject",
    )


def process_image(image_path, text):
    """Process a single image and generate output."""
    # Open and preprocess the image
    image = Image.open(image_path)
    w, h, min_size = image.size[0], image.size[1], min(image.size)
    image = image.crop(
        (
            (w - min_size) // 2,
            (h - min_size) // 2,
            (w + min_size) // 2,
            (h + min_size) // 2,
        )
    )
    image = image.resize((512, 512))

    condition = Condition("subject", image)

    if pipe is None:
        init_pipeline()

    result_img = generate(
        pipe,
        prompt=text.strip(),
        conditions=[condition],
        num_inference_steps=8,
        height=512,
        width=512,
    ).images[0]

    return result_img


def main():
    # Define assets and output paths
    assets_dir = "assets"
    output_dir = "output_images"
    os.makedirs(output_dir, exist_ok=True)

    # Image-text pairs
    samples = [
        {
            "image": "oranges.jpg",
            "text": "A very close up view of this item. It is placed on a wooden table. The background is a dark room, the TV is on, and the screen is showing a cooking show. With text on the screen that reads 'Omini Control!'",
        },
        {
            "image": "penguin.jpg",
            "text": "On Christmas evening, on a crowded sidewalk, this item sits on the road, covered in snow and wearing a Christmas hat, holding a sign that reads 'Omini Control!'",
        },
        {
            "image": "rc_car.jpg",
            "text": "A film style shot. On the moon, this item drives across the moon surface. The background is that Earth looms large in the foreground.",
        },
        {
            "image": "clock.jpg",
            "text": "In a Bauhaus style room, this item is placed on a shiny glass table, with a vase of flowers next to it. In the afternoon sun, the shadows of the blinds are cast on the wall.",
        },
        {
            "image": "tshirt.jpg",
            "text": "On the beach, a lady sits under a beach umbrella with 'Omini' written on it. She's wearing this shirt and has a big smile on her face, with her surfboard behind her.",
        },
    ]

    # Process each sample and save the output
    for idx, sample in enumerate(samples):
        image_path = os.path.join(assets_dir, sample["image"])
        text = sample["text"]

        print(f"Processing {image_path} with prompt: {text}")
        output_image = process_image(image_path, text)

        output_path = os.path.join(output_dir, f"result_{idx + 1}.png")
        output_image.save(output_path)
        print(f"Saved output to {output_path}")


if __name__ == "__main__":
    init_pipeline()
    main()
