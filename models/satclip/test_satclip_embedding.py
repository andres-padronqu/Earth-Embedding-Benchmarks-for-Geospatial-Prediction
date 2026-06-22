from pathlib import Path
import sys

import pandas as pd
import torch
from huggingface_hub import hf_hub_download

sys.path.append(str(Path("satclip").resolve()))

from load import get_satclip  # noqa: E402


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    locations = [
        ("ITAM_CDMX", -99.1840, 19.3460),
        ("Paris", 2.3522, 48.8566),
        ("Tokyo", 139.6917, 35.6895),
        ("New_York", -74.0060, 40.7128),
        ("Monterrey", -100.3161, 25.6866),
    ]

    coords = torch.tensor(
        [[lon, lat] for _, lon, lat in locations],
        dtype=torch.double,
    )

    checkpoint = hf_hub_download(
        "microsoft/SatCLIP-ViT16-L40",
        "satclip-vit16-l40.ckpt",
    )

    model = get_satclip(checkpoint, device=device)
    model.eval()

    with torch.no_grad():
        embeddings = model(coords.to(device)).detach().cpu()

    print("Embeddings shape:", embeddings.shape)
    print("First embedding preview:", embeddings[0, :10])

    examples_dir = Path("examples")
    examples_dir.mkdir(exist_ok=True)

    embedding_df = pd.DataFrame(
        embeddings.numpy(),
        index=[name for name, _, _ in locations],
    )

    embedding_df.to_csv(examples_dir / "satclip_example_embeddings.csv")

    metadata_df = pd.DataFrame(
        locations,
        columns=["location", "lon", "lat"],
    )

    metadata_df.to_csv(examples_dir / "satclip_example_locations.csv", index=False)

    print("Saved embeddings to examples/satclip_example_embeddings.csv")
    print("Saved locations to examples/satclip_example_locations.csv")


if __name__ == "__main__":
    main()