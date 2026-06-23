from pathlib import Path

import pandas as pd
from geoclip import GeoCLIP


def main():
    model = GeoCLIP()

    image_dir = Path("models/geoclip/images/mexico_cities")
    output_dir = Path("models/geoclip/examples/image_geolocation_mexico")
    output_dir.mkdir(parents=True, exist_ok=True)

    images = ["cdmx.jpg"]

    records = []

    for image_name in images:
        image_path = image_dir / image_name

        top_pred_gps, top_pred_prob = model.predict(
            str(image_path),
            top_k=5,
        )

        print(f"\nImage: {image_name}")
        print("Top 5 GPS predictions")

        for rank, (gps, prob) in enumerate(
            zip(top_pred_gps, top_pred_prob),
            start=1,
        ):
            lat, lon = gps

            print(
                f"{rank}. lat={lat:.6f}, "
                f"lon={lon:.6f}, "
                f"prob={prob:.6f}"
            )

            records.append(
                {
                    "image": image_name,
                    "rank": rank,
                    "pred_lat": float(lat),
                    "pred_lon": float(lon),
                    "probability": float(prob),
                }
            )

    results = pd.DataFrame(records)

    results.to_csv(
        output_dir / "geoclip_image_predictions.csv",
        index=False,
    )

    print(f"\nSaved results to {output_dir / 'geoclip_image_predictions.csv'}")


if __name__ == "__main__":
    main()