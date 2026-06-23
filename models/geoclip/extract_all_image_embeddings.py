from pathlib import Path

import pandas as pd
import torch
from geoclip import GeoCLIP
from PIL import Image
from transformers import CLIPProcessor


def main():
    model = GeoCLIP()
    model.eval()

    processor = CLIPProcessor.from_pretrained(
        "openai/clip-vit-large-patch14"
    )

    image_dir = Path(
        "models/geoclip/images/mexico_cities"
    )

    output_dir = Path(
        "models/geoclip/examples/image_embeddings"
    )
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    embeddings = []

    image_files = sorted(
        image_dir.glob("*.jpg")
    )

    print(f"Found {len(image_files)} images")

    for image_path in image_files:

        city = image_path.stem

        print(f"Processing {city}")

        image = Image.open(
            image_path
        ).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        with torch.no_grad():
            embedding = (
                model.image_encoder(
                    inputs["pixel_values"]
                )
                .detach()
                .cpu()
                .numpy()
                .flatten()
            )

        row = {
            "city": city
        }

        for i in range(len(embedding)):
            row[f"dim_{i}"] = embedding[i]

        embeddings.append(row)

    df = pd.DataFrame(
        embeddings
    )

    output_file = (
        output_dir /
        "geoclip_image_embeddings.csv"
    )

    df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"Saved to {output_file}"
    )


if __name__ == "__main__":
    main()