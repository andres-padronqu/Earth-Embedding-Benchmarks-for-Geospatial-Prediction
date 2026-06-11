# The Complete Pipeline

This is where the separate pieces from Chapters 2–6 finally line up into one workflow you can actually run: imagery in one end, a fitted model and predictions out the other. If a chapter earned its place in this book, it's because it shows up here. By the end you should be able to wire Earth embeddings into your own project without treating any step as magic.

## Pipeline Overview

```
SATELLITE IMAGES       →   ENCODER (ViT)    →   EMBEDDINGS (768D)
  [Sentinel-2]              [foundation          [NumPy array]
                              model]
                                                       ↓
                                              DataFrame (pandas)
                                                       ↓
                                       MERGE with socioeconomic data
                                                       ↓
                                     PCA (768D → k components) + OLS
                                                       ↓
                                       PREDICT on unseen locations
```

Each step corresponds to a concept covered in a previous chapter:
- **Images**: Chapter 2 (tensor structure, spectral bands)
- **Encoder**: Chapters 4–5 (ViT, MAE)
- **Embeddings**: Chapter 3 (vectors, cosine similarity)
- **Models**: Chapter 6 (Clay, AlphaEarth, etc.)

## Step 1: Obtain Satellite Imagery

For research purposes, satellite imagery is available from multiple sources:

| Source | Sensor | Resolution | Cost | Access |
|--------|--------|------------|------|--------|
| Copernicus Open Access Hub | Sentinel-2 | 10m | Free | scihub.copernicus.eu |
| Google Earth Engine | Multiple | Varies | Free (academic) | earthengine.google.com |
| USGS EarthExplorer | Landsat 8/9 | 30m | Free | earthexplorer.usgs.gov |
| Planet | PlanetScope | 3–5m | Commercial | planet.com |

For demonstration, the EuroSAT dataset provides a convenient labeled collection of 27,000 Sentinel-2 image patches at 64×64 pixels, classified into 10 land use categories:

```python
from torchvision.datasets import EuroSAT
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

dataset = EuroSAT(root="./data", transform=transform, download=True)
print(f"Images: {len(dataset)}, Classes: {len(dataset.classes)}")
print(f"Classes: {dataset.classes}")
```

In a real research project, you would download imagery for your specific study area—municipalities, grid cells, or administrative units—via Google Earth Engine or the Copernicus API.

## Step 2: Extract Embeddings

Load a pre-trained ViT encoder and extract the embedding for each image:

```python
import torch
import torchvision

device = "cuda" if torch.cuda.is_available() else "cpu"

vit = torchvision.models.vit_b_16(
    weights=torchvision.models.ViT_B_16_Weights.DEFAULT
)
vit.heads = torch.nn.Identity()  # remove classification head → pure encoder
vit = vit.to(device).eval()

embeddings_list = []
with torch.no_grad():
    for img_tensor, label in selected_images:
        img_batch = img_tensor.unsqueeze(0).to(device)
        emb = vit(img_batch).cpu().squeeze().numpy()  # (768,)
        embeddings_list.append(emb)

embeddings = np.stack(embeddings_list)  # (N, 768)
print(f"Embedding matrix: {embeddings.shape}")
```

> **Note — Production vs. Demo.** The ViT-B/16 used here is pre-trained on ImageNet (natural images), not on satellite imagery. It serves as a pedagogical stand-in. In production, you would use Clay, Prithvi, or AlphaEarth, which are specifically trained on Earth observation data and produce embeddings that capture geospatially relevant features.

The parameter count of the encoder (~86M for ViT-Base) tells you how much capacity the model has to represent visual information. Clay has ~307M parameters; AlphaEarth uses a U-Net with EfficientNet backbone.

## Step 3: Build the DataFrame

Once embeddings are extracted, the transition from computer vision to standard data analysis is straightforward—each embedding becomes a row in a DataFrame:

```python
import pandas as pd

# Embedding columns
df_emb = pd.DataFrame(
    embeddings,
    columns=[f"emb_{i:03d}" for i in range(embeddings.shape[1])]
)

# Add metadata
df_emb["zone_id"] = zone_ids
df_emb["land_cover"] = land_cover_labels

# Socioeconomic data (from census, survey, or administrative records)
df_socio = pd.DataFrame({
    "zone_id": zone_ids,
    "gdp_index": gdp_values,
    "gini_coefficient": gini_values,
    "population_density": pop_density,
})

# Merge on zone identifier
df_final = pd.merge(df_emb, df_socio, on="zone_id", how="inner")
print(f"Final dataset: {df_final.shape}")
```

This merge is the bridge: the embedding dimensions become covariates sitting alongside traditional socioeconomic variables. From here on, the analysis runs on standard tools.

## Step 4: Dimensionality Reduction + Regression

With 768 embedding dimensions and potentially limited observations, dimensionality reduction is essential to avoid overfitting:

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Standardize embeddings
emb_cols = [c for c in df_final.columns if c.startswith("emb_")]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_final[emb_cols])

# PCA: 768D → k components
N_COMPONENTS = 10
pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_scaled)

variance_explained = pca.explained_variance_ratio_.sum()
print(f"Variance explained by {N_COMPONENTS} PCs: {variance_explained:.1%}")
```

Then fit OLS on the principal components:

```python
import statsmodels.api as sm

df_pca = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(N_COMPONENTS)])
df_pca["gdp_index"] = df_final["gdp_index"].values

X = sm.add_constant(df_pca[[f"PC{i+1}" for i in range(N_COMPONENTS)]])
y = df_pca["gdp_index"]

model = sm.OLS(y, X).fit()
print(model.summary())
```

The R² of this model tells you how much variation in the outcome is explained by the satellite embeddings alone. Typical values for economic outcomes range from 0.4 to 0.8, depending on the spatial resolution, the embedding model, and the outcome variable.

## Step 5: Prediction on Unseen Locations

The true test of the pipeline is prediction on locations that were not used for training. This requires applying the *same* preprocessing—without re-fitting:

```python
# For new images:
# 1. Extract embedding with the SAME encoder
new_emb = extract_embedding(new_image, vit)

# 2. Standardize with the SAME scaler (not re-fit)
new_scaled = scaler.transform(new_emb.reshape(1, -1))

# 3. Project with the SAME PCA (not re-fit)
new_pca = pca.transform(new_scaled)

# 4. Predict with the SAME model
new_pred = model.predict(sm.add_constant(new_pca))
```

> **Warning — transform, not fit_transform.** When applying the pipeline to new data, use `.transform()` for the scaler and PCA—never `.fit_transform()`. Re-fitting would change the preprocessing parameters, making predictions incomparable to the training set.

## Connection to Published Work

This pipeline mirrors real published research. Yue, Zhao & Hu (2026) estimate economic activity using AlphaEarth embeddings pre-computed on Google Earth Engine. Their approach:

1. Query AlphaEarth embeddings (64D) for grid cells covering countries of interest.
2. Train a neural network to project from 64D embeddings to a 32D "income-aware" space.
3. Predict GDP per capita using the 32D representations.
4. Validate against World Bank statistics.

The pipeline structure is identical to what we demonstrated here. The differences are in scale (global vs. local), embedding model (AlphaEarth vs. ViT-B/16), and the downstream model (neural network vs. OLS).

## Feature Set Comparison

A key analysis for social scientists is comparing the predictive contribution of embeddings against traditional covariates:

```python
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score

feature_sets = {
    "Embeddings only":          X_emb,
    "Sociodemographics only":   X_socio,
    "Coordinates only":         X_coords,
    "Embeddings + Sociodemo":   np.hstack([X_emb, X_socio]),
    "All features":             np.hstack([X_emb, X_socio, X_coords]),
}

for name, X in feature_sets.items():
    model = HistGradientBoostingRegressor(random_state=42)
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    print(f"{name:35s} R² = {r2:.3f}")
```

If embeddings add substantial predictive power beyond sociodemographics, this suggests they capture information not available in census data—potentially visual characteristics of the built environment, infrastructure quality, or environmental conditions that surveys do not measure.

## Recap

Five steps: imagery → encoder → DataFrame → PCA/OLS → prediction. The one that does the real work — turning a computer-vision artifact into something a social scientist can use — is the merge, where embedding columns sit next to census columns in the same table. Two habits keep you out of trouble: reduce dimensions with PCA when you have more embedding columns than rows, and fit your preprocessing on the training set only, then `transform` everything else. And the analysis worth running, every time, is the feature-set comparison: do the embeddings add anything once the usual covariates are already in the model?
