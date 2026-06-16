# Earth Embeddings: From Computer Vision to Social Science

This resource provides a structured introduction to **Earth embeddings**—compact vector representations of geographic locations learned by AI models from satellite imagery and other geospatial data sources. It is designed for researchers in the computational and data sciences who want to understand how techniques from computer vision can be transferred to empirical social science research.

## What You Will Learn

By working through this material, you will understand:

1. **Where the data comes from**: how satellite sensors capture multispectral images and how those images are structured as numerical tensors.
2. **What embeddings are**: how millions of pixel values can be compressed into a vector of a few hundred numbers that preserves semantic information about a location.
3. **How the compression works**: the mechanics of Vision Transformers (ViT) and Masked Autoencoders (MAE)—the architectures behind modern geospatial foundation models.
4. **Which models exist**: a taxonomy of Earth embedding models (Clay, Prithvi, SatCLIP, AlphaEarth) and how they differ.
5. **How to use them**: the complete pipeline from raw satellite imagery to a DataFrame ready for standard econometric analysis.

## Who This Is For

This resource assumes familiarity with:
- Linear algebra basics (vectors, matrices, dot products)
- Introductory statistics (OLS regression, PCA)
- Python (NumPy, pandas, matplotlib)

No prior knowledge of computer vision, deep learning, or remote sensing is required. The presentation builds from first principles.

## How to Navigate

Each chapter builds on the previous one. Chapters 2–5 cover the technical foundations. Chapters 6–7 connect those foundations to real Earth embedding models and applied workflows.

Code blocks throughout the text are meant to be read and understood—they illustrate mechanisms rather than provide production-ready implementations. Where relevant, they can be executed in Google Colab or a local Python environment.
