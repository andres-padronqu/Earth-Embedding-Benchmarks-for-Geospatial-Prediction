# Introduction: Why Earth Embeddings?

## The Geospatial Data Problem

Geospatial data are discrete snapshots of our planet—satellite images, sensor readings, geotagged social media posts—captured at specific locations and times. These data are widely used for disaster response, weather forecasting, transportation management, and environmental monitoring. But working with raw geospatial data presents fundamental challenges:

- **Volume**: A single Sentinel-2 satellite image tile contains over 100 million pixel values across 13 spectral bands.
- **Heterogeneity**: Different sensors, resolutions, and modalities produce data in incompatible formats.
- **Expertise**: Processing satellite imagery requires specialized knowledge that most social scientists lack.
- **Computation**: Running models on raw imagery at global scale demands significant infrastructure.

These barriers mean that most quantitative social scientists—who could benefit enormously from geospatial information—rarely work with it directly.

## Embeddings as a Solution

The concept of **embeddings** offers a way through these barriers. An embedding is a compact vector representation that compresses high-dimensional data into a lower-dimensional form while preserving meaningful structure. The idea has its modern origins in natural language processing, where methods like Word2Vec (Mikolov et al., 2013) and GloVe (Pennington et al., 2014) showed that words could be mapped to vectors such that semantic relationships were captured by geometric distances. For example, the vector difference between "kitten" and "cat" is similar to the difference between "puppy" and "dog."

**Earth embeddings** apply this same principle to geographic locations (Klemmer et al., 2025). Instead of representing a location as millions of raw pixel values, we represent it as a vector of, say, 768 numbers that encode the visual and environmental characteristics of that place. Two locations that are functionally similar—two dense urban areas, for instance—will have vectors that are close together in embedding space, even if they are on different continents.

## The Four Central Functions

Earth embeddings serve four central functions (Klemmer et al., 2025):

1. **Compression**: Distilling high-dimensional, multi-modal geospatial data into compact vectors. A 512×512 pixel image with 13 bands contains over 3.4 million values; its embedding might contain 768.

2. **Fusion**: Combining different data modalities—optical imagery, radar, text, demographic data—into a single unified representation indexed by location.

3. **Interpolation**: For models based on implicit neural representations (location encoders), embeddings are available for *any* coordinate in continuous space, even locations absent from the training data.

4. **Interoperability**: Embeddings can serve as "location tokens" that plug into other AI systems, including large language models, enabling text-based queries about geographic locations.

## From Computer Vision to Social Science

The technical machinery behind Earth embeddings comes from computer vision: convolutional neural networks, Vision Transformers, self-supervised learning. But the *applications* extend far beyond image analysis. Once a location is represented as a vector of numbers, it becomes a row in a DataFrame—compatible with regression, classification, clustering, and any other statistical method.

This creates an opportunity for **methodological arbitrage**: taking mature techniques from computer vision and deploying them to answer questions in computational social science. Specifically:

- **Poverty mapping**: Using satellite embeddings as proxies for economic conditions in regions with no survey data (Rolf et al., 2021).
- **Economic activity estimation**: Predicting GDP or economic indicators from spatial representations (Yue et al., 2026).
- **Urban analysis**: Characterizing neighborhoods at fine spatial resolution using embeddings merged with census data.
- **Conflict and governance**: Measuring state capacity, infrastructure, or displacement through observable features of the built and natural environment.

The path from "interesting computer vision technique" to "valid social science measurement" is not automatic—it requires careful evaluation of what embeddings actually capture, where they fail, and under what conditions they can support causal or descriptive inference. This resource works through that path systematically.

## Roadmap

| Chapter | Topic | Source |
|---------|-------|--------|
| 2 | Satellite imagery as tensors, spectral bands, NDVI | Session 1A |
| 3 | What embeddings are, cosine similarity, use in regression | Session 1B |
| 4 | Vision Transformer architecture | Session 2A |
| 5 | Masked Autoencoders and self-supervised learning | Session 2B |
| 6 | Earth embedding models: Clay, Prithvi, SatCLIP, AlphaEarth | Session 3A + paper |
| 7 | Complete pipeline: image → embedding → merge → OLS | Session 3B |

Each chapter builds on the previous one. By the end, you will understand both *how* Earth embeddings work and *when* they can be responsibly used for social science research.
