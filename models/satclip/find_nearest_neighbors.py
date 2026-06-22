from pathlib import Path

import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def get_top_neighbors(city, similarity_matrix, top_k=5):
    neighbors = (
        similarity_matrix.loc[city]
        .drop(city)
        .sort_values(ascending=False)
        .head(top_k)
    )

    return pd.DataFrame(
        {
            "neighbor": neighbors.index,
            "cosine_similarity": neighbors.values,
        }
    )


def main():

    similarity_path = (
        "models/satclip/examples/mexico_capitals/"
        "satclip_mexico_capitals_similarity.csv"
    )

    similarity = pd.read_csv(
        similarity_path,
        index_col=0,
    )

    output_dir = Path(
        "models/satclip/examples/mexico_capitals/neighbors"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cities_to_analyze = [
        "Ciudad de Mexico",
        "Monterrey",
        "Merida",
        "Guadalajara",
    ]

    for city in cities_to_analyze:

        neighbors = get_top_neighbors(
            city,
            similarity,
            top_k=5,
        )

        print(f"\nTop neighbors for {city}")
        print(neighbors)

        neighbors.to_csv(
            output_dir / f"{city}_neighbors.csv",
            index=False,
        )

    # Global nearest neighbor for every city

    records = []

    for city in similarity.index:

        nearest = (
            similarity.loc[city]
            .drop(city)
            .idxmax()
        )

        score = (
            similarity.loc[city]
            .drop(city)
            .max()
        )

        records.append(
            {
                "city": city,
                "nearest_neighbor": nearest,
                "cosine_similarity": score,
            }
        )

    summary = pd.DataFrame(records)

    summary.to_csv(
        output_dir / "all_nearest_neighbors.csv",
        index=False,
    )

    print("\nSaved neighbor analysis.")


if __name__ == "__main__":
    main()