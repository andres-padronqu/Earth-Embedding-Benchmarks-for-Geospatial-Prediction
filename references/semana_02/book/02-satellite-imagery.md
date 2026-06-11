# Satellite Imagery Fundamentals

Before we can talk about embeddings, we need to be clear about what goes into them. Everything downstream is built on satellite imagery, and an embedding can only encode what the image captured in the first place — so it's worth understanding the raw material on its own terms.

## What a Satellite Captures

A satellite like Sentinel-2 doesn't take photos the way your phone does. Your phone records three channels — red, green, blue — because that's what the human eye uses. Sentinel-2 carries a multispectral sensor that measures reflected light in 13 separate bands, reaching past visible light into the near-infrared (NIR) and shortwave infrared (SWIR), where a lot of the interesting physics lives.

Each band captures different physical information:

| Band | Name | Wavelength (nm) | What It Captures |
|------|------|-----------------|------------------|
| B2 | Blue | 490 | Water penetration, atmospheric scattering |
| B3 | Green | 560 | Vegetation vigor |
| B4 | Red | 665 | Chlorophyll absorption |
| B8 | NIR | 842 | Vegetation structure, biomass |
| B11 | SWIR 1 | 1610 | Soil moisture, built surfaces |
| B12 | SWIR 2 | 2190 | Mineral content, fire detection |

The spatial resolution varies by band: 10m for visible and NIR bands, 20m for red edge and SWIR, and 60m for atmospheric bands. A single tile covers 100×100 km.

## From Image to Tensor

A satellite image is stored as a GeoTIFF file—a raster format that includes both pixel values and geographic metadata (coordinate reference system, spatial extent, resolution). When loaded into Python, the image becomes a three-dimensional NumPy array—a **tensor** with shape `(C, H, W)`:

- **C** (channels): the number of spectral bands
- **H** (height): the number of pixel rows
- **W** (width): the number of pixel columns

```python
import rasterio

with rasterio.open("Sentinel2_Tile.tif") as src:
    image = src.read()  # shape: (C, H, W)
    print(f"Shape: {image.shape}")
    print(f"Bands: {image.shape[0]}")
    print(f"Pixels: {image.shape[1]} x {image.shape[2]}")
    print(f"Resolution: {src.res[0]} meters/pixel")
    print(f"CRS: {src.crs}")
```

For a Sentinel-2 image with 13 bands at 1000×1000 pixels, the tensor contains $13 \times 1000 \times 1000 = 13{,}000{,}000$ numerical values. Each value represents the reflectance intensity measured by the sensor for that pixel in that band.

> **Note — Notation Convention.** Python and PyTorch use the *channels-first* convention: `(C, H, W)`. Some frameworks (TensorFlow, NumPy for display) use *channels-last*: `(H, W, C)`. The data is the same; only the axis ordering differs.

## Exploring a Single Band

Extracting one band from the tensor produces a 2D matrix—a grayscale image where each value is the reflectance at that pixel for that wavelength:

```python
band_nir = image[7, :, :]  # Band 8 (NIR), zero-indexed
print(f"Shape: {band_nir.shape}")  # (1000, 1000)
print(f"Min: {band_nir.min()}, Max: {band_nir.max()}")
print(f"Mean: {band_nir.mean():.2f}")
```

The NIR band is particularly informative because healthy vegetation reflects strongly in near-infrared wavelengths. This is invisible to the human eye but clearly captured by the sensor: a forest will show high NIR values while a body of water will show very low ones.

## Pixel-Level Values

A single pixel across all bands forms a **spectral signature**—a vector that characterizes the material at that location:

```python
pixel_x, pixel_y = 500, 500
spectral_signature = image[:, pixel_x, pixel_y]
# spectral_signature is a vector of length C (e.g., 13)
```

Different land cover types produce different spectral signatures:
- **Vegetation**: low red reflectance (chlorophyll absorbs red light), high NIR reflectance.
- **Water**: low reflectance across all bands, especially in NIR and SWIR.
- **Urban surfaces**: moderate, relatively uniform reflectance across visible bands; high SWIR.
- **Bare soil**: gradually increasing reflectance from blue to SWIR.

These spectral differences are the foundation of all remote sensing analysis—and the information that embedding models learn to encode.

## Spectral Indices: NDVI

Before embeddings, the dominant approach to extracting information from satellite images was computing **spectral indices**—algebraic combinations of bands designed to highlight specific phenomena. The most widely used is the Normalized Difference Vegetation Index (NDVI):

$$
\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}
$$

```python
red = image[3, :, :].astype(float)   # Band 4
nir = image[7, :, :].astype(float)   # Band 8

denominator = nir + red
denominator[denominator == 0] = 1e-4  # avoid division by zero
ndvi = (nir - red) / denominator
```

NDVI values range from -1 to 1:
- **0.6–1.0**: Dense vegetation (tropical forest, mature canopy)
- **0.3–0.6**: Moderate vegetation (crops, grassland)
- **0.0–0.2**: Bare soil, urban surfaces
- **< 0**: Water, snow, clouds

NDVI has been enormously useful in environmental science. But it uses only 2 of 13 available bands, and it captures only one dimension (vegetation density) of the rich information contained in a satellite image. This is where embeddings enter: they aim to compress *all* the information across *all* bands into a compact representation—not just vegetation, but urban form, water presence, soil type, and their spatial arrangement.

## The Compression Problem

Consider the scale of the problem. A single 1000×1000 pixel Sentinel-2 image with 13 bands contains 13 million values. If you want to use satellite data as an input to a regression model—predicting poverty, economic activity, or crime—you cannot use 13 million raw pixel values as independent variables. You need to compress.

Traditional approaches to compression include:
- **Hand-crafted features**: NDVI, built-up indices, texture statistics. Effective for specific tasks but limited to what the researcher anticipates will matter.
- **Summary statistics**: Mean reflectance per band. Simple but discards spatial structure entirely.
- **Random features**: MOSAIKS (Rolf et al., 2021) uses random convolutional features as a general-purpose baseline.

Deep learning offers a different approach: learn the compression from data. Train a neural network on millions of satellite images so that it learns which patterns matter, and let it produce a compressed representation—an **embedding**—that preserves as much useful information as possible. The next chapters explain exactly how this works.

## Recap

A satellite image is a 3D tensor `(C, H, W)`, and every number in it is a reflectance measurement. Different bands see different physical things, which is why indices like NDVI work — but NDVI only ever uses two of the thirteen bands, so it leaves most of the picture on the table. The real problem for any downstream model is compression: how do you go from millions of pixel values to a handful of features without throwing away the part you actually needed? That question is what the rest of the book answers.
