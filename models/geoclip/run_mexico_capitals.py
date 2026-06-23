from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from geoclip import LocationEncoder
from sklearn.decomposition import PCA
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
    capitals = [
        ("Aguascalientes", "Aguascalientes", -102.2916, 21.8853),
        ("Mexicali", "Baja California", -115.4683, 32.6245),
        ("La Paz", "Baja California Sur", -110.3128, 24.1426),
        ("San Francisco de Campeche", "Campeche", -90.5349, 19.8301),
        ("Saltillo", "Coahuila", -100.9864, 25.4380),
        ("Colima", "Colima", -103.7271, 19.2452),
        ("Tuxtla Gutierrez", "Chiapas", -93.1167, 16.7569),
        ("Chihuahua", "Chihuahua", -106.0889, 28.6353),
        ("Ciudad de Mexico", "Ciudad de Mexico", -99.1332, 19.4326),
        ("Durango", "Durango", -104.6532, 24.0277),
        ("Guanajuato", "Guanajuato", -101.2574, 21.0190),
        ("Chilpancingo", "Guerrero", -99.5000, 17.5500),
        ("Pachuca", "Hidalgo", -98.7624, 20.1011),
        ("Guadalajara", "Jalisco", -103.3496, 20.6597),
        ("Toluca", "Estado de Mexico", -99.6557, 19.2826),
        ("Morelia", "Michoacan", -101.1940, 19.7020),
        ("Cuernavaca", "Morelos", -99.2342, 18.9242),
        ("Tepic", "Nayarit", -104.8946, 21.5085),
        ("Monterrey", "Nuevo Leon", -100.3161, 25.6866),
        ("Oaxaca", "Oaxaca", -96.7266, 17.0732),
        ("Puebla", "Puebla", -98.2063, 19.0414),
        ("Queretaro", "Queretaro", -100.3899, 20.5888),
        ("Chetumal", "Quintana Roo", -88.3038, 18.5001),
        ("San Luis Potosi", "San Luis Potosi", -100.9855, 22.1565),
        ("Culiacan", "Sinaloa", -107.3940, 24.8091),
        ("Hermosillo", "Sonora", -110.9559, 29.0729),
        ("Villahermosa", "Tabasco", -92.9303, 17.9892),
        ("Ciudad Victoria", "Tamaulipas", -99.1456, 23.7369),
        ("Tlaxcala", "Tlaxcala", -98.2390, 19.3182),
        ("Xalapa", "Veracruz", -96.9102, 19.5438),
        ("Merida", "Yucatan", -89.5926, 20.9674),
        ("Zacatecas", "Zacatecas", -102.5832, 22.7709),
    ]

    # GeoCLIP uses lat/lon order.
    gps_data = torch.tensor(
        [[lat, lon] for _, _, lon, lat in capitals],
        dtype=torch.float32,
    )

    gps_encoder = LocationEncoder()
    gps_encoder.eval()

    with torch.no_grad():
        embeddings = gps_encoder(gps_data).detach().cpu()

    output_dir = Path("models/geoclip/examples/mexico_capitals")
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = pd.DataFrame(
        capitals,
        columns=["capital", "state", "lon", "lat"],
    )

    embeddings_df = pd.DataFrame(
        embeddings.numpy(),
        index=metadata["capital"],
    )

    metadata.to_csv(
        output_dir / "geoclip_mexico_capitals_locations.csv",
        index=False,
    )

    embeddings_df.to_csv(
        output_dir / "geoclip_mexico_capitals_embeddings.csv"
    )

    similarity = pd.DataFrame(
        cosine_similarity(embeddings_df),
        index=embeddings_df.index,
        columns=embeddings_df.index,
    )

    similarity.to_csv(
        output_dir / "geoclip_mexico_capitals_similarity.csv"
    )

    pca = PCA(n_components=2)
    pca_values = pca.fit_transform(embeddings_df)

    pca_df = pd.DataFrame(
        pca_values,
        columns=["PC1", "PC2"],
        index=embeddings_df.index,
    )

    pca_df = pca_df.merge(
        metadata,
        left_index=True,
        right_on="capital",
    )

    pca_df.to_csv(
        output_dir / "geoclip_mexico_capitals_pca.csv",
        index=False,
    )

    plt.figure(figsize=(12, 8))
    plt.scatter(pca_df["PC1"], pca_df["PC2"])

    for _, row in pca_df.iterrows():
        plt.annotate(
            row["capital"],
            (row["PC1"], row["PC2"]),
            fontsize=8,
        )

    plt.title("GeoCLIP Embeddings PCA: Mexican State Capitals")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.tight_layout()

    plt.savefig(
        output_dir / "geoclip_mexico_capitals_pca.png",
        dpi=300,
    )

    plt.close()

    neighbors_dir = output_dir / "neighbors"
    neighbors_dir.mkdir(parents=True, exist_ok=True)

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

        neighbors.to_csv(
            neighbors_dir / f"{city}_neighbors.csv",
            index=False,
        )

        print(f"\nTop neighbors for {city}")
        print(neighbors)

    records = []

    for city in similarity.index:
        nearest = similarity.loc[city].drop(city).idxmax()
        score = similarity.loc[city].drop(city).max()

        records.append(
            {
                "city": city,
                "nearest_neighbor": nearest,
                "cosine_similarity": score,
            }
        )

    all_neighbors = pd.DataFrame(records)

    all_neighbors.to_csv(
        neighbors_dir / "all_nearest_neighbors.csv",
        index=False,
    )

    print("Embeddings shape:", embeddings.shape)
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print(f"Files saved to: {output_dir}")


if __name__ == "__main__":
    main()