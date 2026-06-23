from pathlib import Path

import pandas as pd
import torch
from geoclip import GeoCLIP
from PIL import Image
from transformers import CLIPProcessor


def main():
    model = GeoCLIP()
    model.eval()

    processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")

    image_path = Path("models/geoclip/images/mexico_cities/cdmx.jpg")
    output_dir = Path("models/geoclip/examples/image_geolocation_mexico")
    output_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(image_path).convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt",
    )

    pixel_values = inputs["pixel_values"]

    with torch.no_grad():
        image_embedding = model.image_encoder(pixel_values).detach().cpu()

    print("Image embedding shape:", image_embedding.shape)
    print("First 10 values:", image_embedding[0, :10])

    embedding_df = pd.DataFrame(
        image_embedding.numpy(),
        index=["cdmx"],
    )

    embedding_df.to_csv(
        output_dir / "geoclip_cdmx_image_embedding.csv"
    )

    print("Saved image embedding.")


if __name__ == "__main__":
    main()