from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def main():
    embeddings_path = Path("models/satclip/examples/satclip_example_embeddings.csv")

    embeddings = pd.read_csv(embeddings_path, index_col=0)

    similarity_matrix = pd.DataFrame(
        cosine_similarity(embeddings),
        index=embeddings.index,
        columns=embeddings.index,
    )

    output_path = Path("models/satclip/examples/satclip_cosine_similarity.csv")
    similarity_matrix.to_csv(output_path)

    print("Cosine similarity matrix:")
    print(similarity_matrix.round(3))
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()