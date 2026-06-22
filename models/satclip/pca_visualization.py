from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA


def main():
    embeddings_path = Path(
        "models/satclip/examples/satclip_example_embeddings.csv"
    )

    embeddings = pd.read_csv(
        embeddings_path,
        index_col=0,
    )

    pca = PCA(n_components=2)

    embedding_pca = pca.fit_transform(embeddings)

    pca_df = pd.DataFrame(
        embedding_pca,
        columns=["PC1", "PC2"],
        index=embeddings.index,
    )

    print(
        f"Explained variance ratio: "
        f"{pca.explained_variance_ratio_}"
    )

    plt.figure(figsize=(8, 6))

    plt.scatter(
        pca_df["PC1"],
        pca_df["PC2"],
    )

    for location in pca_df.index:
        plt.annotate(
            location,
            (
                pca_df.loc[location, "PC1"],
                pca_df.loc[location, "PC2"],
            ),
        )

    plt.title("SatCLIP Embeddings PCA")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")

    plt.tight_layout()

    output_path = (
        "models/satclip/examples/satclip_embeddings_pca.png"
    )

    plt.savefig(output_path, dpi=300)

    print(f"Saved figure to {output_path}")

    plt.show()


if __name__ == "__main__":
    main()