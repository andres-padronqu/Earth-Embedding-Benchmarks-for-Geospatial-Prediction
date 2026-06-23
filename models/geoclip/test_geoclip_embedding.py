import torch
from geoclip import LocationEncoder


def main():
    gps_encoder = LocationEncoder()

    # GeoCLIP uses lat/lon order.
    gps_data = torch.tensor(
        [
            [19.4326, -99.1332],  # Mexico City
            [25.6866, -100.3161], # Monterrey
            [20.9674, -89.5926],  # Merida
        ],
        dtype=torch.float32,
    )

    with torch.no_grad():
        gps_embeddings = gps_encoder(gps_data)

    print("Embeddings shape:", gps_embeddings.shape)
    print("First embedding preview:", gps_embeddings[0, :10])


if __name__ == "__main__":
    main()