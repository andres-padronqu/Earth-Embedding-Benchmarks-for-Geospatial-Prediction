# The Encoder: Vision Transformers

So far we know what an embedding is and why a compressed representation is worth having. The missing piece is the machine that actually does the compressing. For almost every modern Earth embedding model, that machine is a **Vision Transformer (ViT)**, and this chapter takes it apart step by step.

## What the Encoder Has to Do

An encoder is just a function that takes a big input (an image) and returns a small output (an embedding vector):

$$
E: \mathbb{R}^{C \times H \times W} \rightarrow \mathbb{R}^{d}
$$

Take a Sentinel-2 image with 4 bands at 512×512 pixels. The input holds $4 \times 512 \times 512 = 1{,}048{,}576$ numbers; the embedding holds $d = 768$. That is a compression of more than 1,300×. The encoder doesn't just pick 768 of the original pixels — it learns a transformation that folds the whole million-value input down into 768 new numbers that still carry the signal.

## A Bit of History: CNNs Then Transformers

Until around 2020, image analysis meant **Convolutional Neural Networks (CNNs)**. A CNN slides learned filters across the image: early layers pick up edges and textures, deeper layers pick up shapes and objects. They work well because they assume what is usually true of images — that nearby pixels belong together — so each filter only ever looks at a small neighborhood.

Then Dosovitskiy et al. (2020) tried something different. The Transformer had already taken over NLP, where a sentence is handled as a sequence of words. Their bet was that you could treat an image the same way: chop it into patches and feed those patches in as if they were words. That is the **Vision Transformer**, and it ended up matching CNNs and scaling better on large datasets — which is exactly the regime satellite data lives in.

This is why Clay, Prithvi, and most of the other models we care about are ViTs under the hood. The rest of the chapter follows one image through the architecture.

## Step 1: Patchification

First we cut the image into a regular grid of non-overlapping patches. Each patch will play the role that a word plays in a sentence — it becomes one "token".

For an image of size $H \times W$ with patch size $P$:

$$
N_{\text{patches}} = \frac{H}{P} \times \frac{W}{P}
$$

With $H = W = 512$ and $P = 32$: $N = 16 \times 16 = 256$ patches.

```python
def patchify(image, patch_size):
    """
    Divide image (C, H, W) into patches of size patch_size × patch_size.
    Returns: (num_patches, C * patch_size * patch_size)
    """
    C, H, W = image.shape
    n_h = H // patch_size
    n_w = W // patch_size
    patches = []

    image_hwc = image.transpose(1, 2, 0)  # (H, W, C)
    for i in range(n_h):
        for j in range(n_w):
            patch = image_hwc[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
                :
            ]
            patches.append(patch.flatten())

    return np.array(patches)

patches = patchify(image, patch_size=32)
# patches.shape: (256, 4096)
# 256 patches, each with 32 * 32 * 4 = 4096 values
```

Each patch is flattened into a vector. A 32×32 patch with 4 bands becomes a vector of 4,096 values. The full image is now a sequence of 256 such vectors.

## Step 2: Linear Projection

Each flattened patch vector is projected into the model's working dimensionality $d$ (typically 768) via a learned linear transformation:

$$
\mathbf{z}_i = W_{\text{proj}} \, \mathbf{x}_i + \mathbf{b}_{\text{proj}}
$$

where $\mathbf{x}_i \in \mathbb{R}^{4096}$ is the flattened patch and $\mathbf{z}_i \in \mathbb{R}^{768}$ is the projected token.

```python
import torch.nn as nn

class PatchEmbedding(nn.Module):
    def __init__(self, patch_dim, d_model):
        super().__init__()
        self.projection = nn.Linear(patch_dim, d_model)

    def forward(self, x):
        return self.projection(x)
        # (batch, num_patches, patch_dim) → (batch, num_patches, d_model)
```

This matrix $W_{\text{proj}}$ has $4096 \times 768 \approx 3.1$ million parameters. In a trained model, these parameters encode which combinations of pixel values in a patch are informative.

## Step 3: Positional Encoding

Here is a quirk worth pausing on: a Transformer has no built-in sense of order. It sees the tokens as an unordered *set*, so on its own it could not tell the top-left patch from the bottom-right one. We fix this by adding a **positional encoding** to each token:

$$
\mathbf{z}_i' = \mathbf{z}_i + \mathbf{p}_i
$$

where $\mathbf{p}_i \in \mathbb{R}^{768}$ is a learned vector specific to position $i$. This tells the model that token 0 comes from the top-left corner, token 255 from the bottom-right, and so on.

```python
class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, d_model):
        super().__init__()
        self.pos_embedding = nn.Parameter(
            torch.randn(1, num_patches + 1, d_model) * 0.02
        )

    def forward(self, x):
        return x + self.pos_embedding
```

Without positional encoding, the model would produce the same embedding regardless of how patches are arranged—a shuffled image would look identical to the original.

## Step 4: The CLS Token

Before the sequence enters the Transformer, we glue one extra learnable vector to the front of it: the **CLS token** (short for classification token). It does not come from any patch — it starts as a free parameter the model learns. Its job is to act as a scratchpad: as attention runs, it reads from every patch and slowly fills up with a summary of the whole scene. When we want a single vector for the image, this is the one we read off.

```python
cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

# Prepend CLS to the sequence
x = torch.cat([cls_token.expand(batch_size, -1, -1), x], dim=1)
# x: (batch, num_patches + 1, d_model) = (1, 257, 768)
```

Once the Transformer has chewed through the sequence, we take the CLS token's final state and call that the embedding of the image.

## Step 5: Self-Attention

Self-attention is the part that makes a Transformer a Transformer. The idea: each token gets to look at every other token and decide how much to borrow from each one. Those "how much" weights — the attention scores — are not fixed; they depend on the content of the tokens themselves, and the token's new value is a weighted sum of the others.

Concretely, for a pair of tokens $(i, j)$, the score answers "how much should token $i$ pay attention to token $j$?" In a satellite scene that buys us things like:

- A patch that looks like a road can check its neighbors to tell whether it sits on a highway or a quiet residential street.
- A patch of vegetation can look far across the image to tell whether it is an isolated park or one corner of a continuous forest.
- The CLS token looks at everything, which is how it ends up holding a global summary.

Mechanically this runs on three learned projections of each token — **Query (Q)**, **Key (K)**, and **Value (V)**:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V
$$

In practice we don't run one of these but several in parallel — the "multi-head" part. With $h$ heads (usually 8 or 12), each with its own projections, the model can track several kinds of relationship at once: one head might follow roads, another might track vegetation boundaries.

## Step 6: The Complete Forward Pass

Stacking the steps together, here is the path from image to embedding:

```
IMAGE (C × H × W)
  ↓ patchify
PATCHES (N × patch_dim)         e.g., (256 × 4096)
  ↓ linear projection
TOKENS (N × d_model)            e.g., (256 × 768)
  ↓ prepend CLS + positional encoding
SEQUENCE ((N+1) × d_model)      e.g., (257 × 768)
  ↓ Transformer blocks (self-attention + feedforward)
REPRESENTATIONS ((N+1) × d_model)
  ↓ extract CLS token
EMBEDDING (d_model)              e.g., (768,)
```

```python
class ViTEncoder(nn.Module):
    def __init__(self, patch_dim, num_patches, d_model, num_heads, num_layers):
        super().__init__()
        self.patch_embed = PatchEmbedding(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_encoding = PositionalEncoding(num_patches, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, patches):
        x = self.patch_embed(patches)
        cls = self.cls_token.expand(patches.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_encoding(x)
        x = self.transformer(x)
        x = self.norm(x)
        embedding = x[:, 0, :]  # CLS token
        return embedding
```

A ViT-Base model has approximately 86 million parameters. Clay uses a variant with approximately 307 million parameters.

## Why the Weights Are the Whole Point

Everything above is just plumbing. The architecture is fixed and deterministic — same weights, same input, same output every time. What actually decides whether an embedding means anything is *which weights* sit inside that plumbing.

- **Random weights**: garbage in, garbage out. The embedding is noise with no semantic content. That is, by the way, all that the toy code in this chapter produces — it is there to show the shapes and the flow, not to give you a usable model.
- **Pre-trained weights** (Clay, Prithvi, and the rest): these came out of training on millions of satellite images, and now the same architecture produces embeddings that actually separate vegetation from urbanization from water from bare soil.

So keep the two ideas apart: the architecture sets *what the model could learn*, the weights record *what it did learn*. The next chapter is about how you get good weights without anyone hand-labeling a single image.

## Recap

A ViT reads an image as a sequence of patches, the same way a Transformer reads a sentence as a sequence of words. Each patch is projected into the model's working dimension, tagged with a positional encoding so the model knows where it came from, and then passed through self-attention so every patch can talk to every other patch. A special CLS token rides along and collects the global summary that we read out as the embedding. None of this matters, though, until the weights have been trained — which is the next chapter.
