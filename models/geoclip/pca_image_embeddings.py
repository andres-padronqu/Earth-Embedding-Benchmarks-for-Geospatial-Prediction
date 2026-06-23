from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


def main():
    input_path = Path(
        "models/geoclip/examples/image_embeddings/geoclip_image_embeddings.csv"
    )

    output_dir = Path("models/geoclip/examples/image_embeddings")
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)

    cities = df["city"]
    embeddings = df.drop(columns=["city"])

    similarity = pd.DataFrame(
        cosine_similarity(embeddings),
        index=cities,
        columns=cities,
    )

    similarity.to_csv(
        output_dir / "geoclip_image_embeddings_similarity.csv"
    )

    pca = PCA(n_components=2)
    pca_values = pca.fit_transform(embeddings)

    pca_df = pd.DataFrame(
        pca_values,
        columns=["PC1", "PC2"],
    )
    pca_df["city"] = cities

    pca_df.to_csv(
        output_dir / "geoclip_image_embeddings_pca.csv",
        index=False,
    )

    plt.figure(figsize=(8, 6))
    plt.scatter(pca_df["PC1"], pca_df["PC2"])

    for _, row in pca_df.iterrows():
        plt.annotate(
            row["city"],
            (row["PC1"], row["PC2"]),
            fontsize=9,
        )

    plt.title("GeoCLIP Image Embeddings PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()

    plt.savefig(
        output_dir / "geoclip_image_embeddings_pca.png",
        dpi=300,
    )

    plt.close()

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()