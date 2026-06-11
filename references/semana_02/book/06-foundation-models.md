# Earth Embedding Models

We've covered the architecture (ViT) and the training trick (MAE). Now for the practical question: which actual models can you go and use, and how do they differ? This chapter maps the landscape and ties the abstract machinery from Chapters 4–5 to named systems you can download or query today.

## Explicit vs. Implicit Approaches

The first split to understand is how a model gets to its embedding (Klemmer et al., 2025):

**Explicit feature extraction** models take raw geospatial data (e.g., a satellite image) as input and produce an embedding as output. The embedding is a function of the *content* of the input data at a given location:

$$
\mathbf{emb} \sim E_{\text{explicit}}(\text{data}_{\text{location}})
$$

Examples: Clay, Prithvi, AlphaEarth, MOSAIKS.

**Implicit neural representations** take only *coordinates* (longitude, latitude, optionally time) as input and return an embedding. Location-specific information is stored directly in the network's weights, learned during training:

$$
\mathbf{emb} \sim E_{\text{implicit}}(\text{lon}, \text{lat}, \text{time})
$$

Examples: SatCLIP, GeoCLIP, SINR, Climplicit.

| Property | Explicit | Implicit |
|----------|----------|----------|
| Input at inference | Raw imagery + coordinates | Coordinates only |
| Resolution | High (native sensor resolution) | Low (smooth interpolation) |
| Requires raw data at inference? | Yes | No |
| Model size | Large (100M–1B parameters) | Small (few-layer MLP) |
| Best for | Fine-grained tasks, retrieval | Geographic conditioning, large-scale prediction |

## The Major Models

### Clay

Clay is an open-source geospatial foundation model built on the ViT architecture, trained with MAE on Sentinel-2 and Landsat imagery.

- **Architecture**: ViT encoder (~307M parameters)
- **Training**: MAE self-supervised on multi-spectral satellite imagery
- **Embedding dimension**: 768
- **Input**: Multi-spectral satellite images
- **License**: Apache 2.0 (open-source, open-weight)
- **Best for**: General-purpose feature extraction from optical imagery

Clay produces high-quality embeddings that can be used for classification, segmentation, and prediction tasks. Its open-source nature makes it accessible for academic research.

### Prithvi

Prithvi is a geospatial foundation model developed by IBM and NASA, designed for temporal analysis.

- **Architecture**: ViT with temporal attention
- **Training**: MAE on Harmonized Landsat Sentinel-2 (HLS) data
- **Embedding dimension**: 768
- **Input**: Multi-temporal sequences (4–6 images of the same location over time)
- **Notable feature**: Captures temporal dynamics (seasonal changes, land use transitions)

Prithvi is particularly useful when temporal variation matters—for example, distinguishing irrigated cropland (which greens seasonally) from evergreen forest (which stays green year-round).

### SatCLIP

SatCLIP (Klemmer et al., 2025) is an implicit neural representation model trained via contrastive learning (CLIP-style) on satellite imagery.

- **Architecture**: Location encoder (MLP) trained contrastively with a satellite image encoder
- **Training**: CLIP-style contrastive learning linking geographic coordinates to Sentinel-2 imagery
- **Embedding dimension**: 256 or 512
- **Input at inference**: Longitude and latitude only
- **Notable feature**: No raw imagery needed at inference time; just query with coordinates

SatCLIP learns a smooth embedding surface over the globe. Given any coordinate pair, it returns an embedding that reflects the typical visual characteristics of that location—a useful prior for geographic conditioning.

### GeoCLIP

GeoCLIP (Vivanco Cepeda et al., 2023) applies a similar contrastive approach but trained on ground-level Flickr images instead of satellite imagery.

- **Input at inference**: Longitude and latitude
- **Training data**: Geotagged ground-level photographs
- **Notable feature**: Captures street-level visual patterns rather than overhead patterns

GeoCLIP embeddings reflect what a location *looks like from the ground*—vegetation, architecture, signage—which may complement overhead satellite embeddings.

### AlphaEarth

AlphaEarth (Google) is a commercial-scale explicit embedding model that fuses multiple data modalities.

- **Architecture**: U-Net with EfficientNet backbone
- **Embedding dimension**: 64
- **Input**: Multi-modal (optical, SAR, LiDAR, climate data)
- **Interface**: Pre-computed embedding database available on Google Earth Engine
- **Resolution**: High (up to 4.7m with PlanetScope imagery)

AlphaEarth provides embeddings as a pre-computed database rather than as a downloadable model. Users query coordinates on Google Earth Engine and receive 64-dimensional embedding vectors. This makes it the most accessible option for researchers without GPU infrastructure, but the closed-model approach limits interpretability and customization.

### MOSAIKS

MOSAIKS (Rolf et al., 2021) is a computationally efficient alternative that uses random convolutional features rather than learned features.

- **Architecture**: Random convolutional features (not learned)
- **Training**: No training required; features are fixed random projections
- **Embedding dimension**: Configurable (typically 512–8192)
- **Notable feature**: Extremely fast, no GPU required, strong baseline

MOSAIKS demonstrates that even *random* spatial features from satellite imagery are predictive of many downstream outcomes. It provides an important baseline for evaluating whether the additional complexity of learned embeddings is justified for a given task.

## Embedding Models vs. Embedding Databases

Beyond the explicit/implicit distinction, Earth embeddings differ in how they are *delivered* to users (Klemmer et al., 2025):

**Embedding models** are released as code + weights. The user downloads the model, provides input data, and runs inference locally. This offers maximum flexibility (fine-tuning, custom data, interpretability tools like Grad-CAM) but requires computational resources and deep learning expertise.

**Embedding databases** are pre-computed embedding vectors stored at fixed locations, queryable by coordinates. The user sends a coordinate and receives a vector. This requires no deep learning expertise or GPU, but the embeddings are static and the underlying model cannot be inspected or modified.

| Interface | Pros | Cons |
|-----------|------|------|
| **Model** | Flexible, customizable, inspectable | Requires GPU, DL expertise |
| **Database** | Easy to use, no GPU needed | Static, opaque, may incur storage costs |

In practice, implicit models (SatCLIP, GeoCLIP) function as both model and database: they are lightweight enough to run on a CPU and can be queried for any coordinate. Explicit models (Clay, Prithvi) are more often shared as downloadable models, while their pre-computed outputs (AlphaEarth, Major TOM) are shared as databases.

## Choosing the Right Model

The choice of embedding model depends on the research question:

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| General-purpose prediction (poverty, economic activity) | AlphaEarth or Clay | High resolution, multi-modal, strong baselines |
| Temporal analysis (seasonal change, land use transition) | Prithvi | Temporal attention mechanism |
| Geographic conditioning for other models | SatCLIP | Lightweight, coordinate-only input |
| Street-level visual characterization | GeoCLIP | Ground-level image training |
| Quick baseline, no GPU | MOSAIKS | Random features, very fast |
| Fine-grained land cover classification | Clay or AlphaEarth | High native resolution |

For social science applications, a pragmatic approach is to start with a pre-computed embedding database (AlphaEarth on GEE or MOSAIKS) and compare against traditional covariates before investing in running a full foundation model.

## Recap

The big dividing line is explicit vs. implicit: explicit models (Clay, Prithvi, AlphaEarth) need the actual imagery, implicit ones (SatCLIP, GeoCLIP) need only a coordinate. The heavyweight foundation models are ViTs trained with MAE; the location encoders are small and give you a smooth surface over the globe. For getting started, AlphaEarth on Google Earth Engine is the path of least resistance — no GPU, no weights to manage — and MOSAIKS is the random-feature baseline you compare against to check whether the learned model is even earning its keep. In the end the right choice falls out of your question, your data, and your compute, not out of which model is newest.
