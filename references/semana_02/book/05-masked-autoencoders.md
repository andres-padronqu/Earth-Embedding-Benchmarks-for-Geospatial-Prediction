# Learning Without Labels: Masked Autoencoders

The last chapter gave us the architecture but left the hard question open: where do good weights come from? You can't hand-label the planet, so the answer can't be ordinary supervised learning. The trick that makes it work is the **Masked Autoencoder (MAE)** — a way to train the encoder using nothing but the images themselves.

## The Problem With Labels

Supervised learning needs labels: each image tagged "forest," "urban," "water," and so on. At global scale that is a non-starter — far too slow and expensive, and nobody is going to annotate millions of satellite tiles by hand. Self-supervised learning gets around this by inventing a task where the data *is* its own answer key.

MAE (He et al., 2022) is about as clean an example of this as you'll find:

1. **Hide** most of the image — 75% of it.
2. Make the model **reconstruct** the hidden part from the quarter it can still see.
3. The gap between its guess and the truth is the training signal. No human labels anywhere.

The reason this teaches the model anything useful is subtle. To fill in a patch it never saw, the model has to have internalized what satellite imagery *tends to look like* — how vegetation clusters, how urban texture differs from cropland, what water does across the bands. That internalized sense of the data is exactly what later makes the embeddings worth using.

## The Masking Strategy

The input image is divided into patches (as in Chapter 4), and 75% of them are randomly masked—hidden from the encoder. Only the remaining 25% of patches are processed:

```python
def generate_mask(num_patches, mask_ratio, seed=42):
    rng = np.random.default_rng(seed)
    num_masked = int(num_patches * mask_ratio)
    all_indices = np.arange(num_patches)
    masked_indices = rng.choice(all_indices, size=num_masked, replace=False)
    visible_indices = np.setdiff1d(all_indices, masked_indices)
    return np.sort(visible_indices), np.sort(masked_indices)

visible_idx, masked_idx = generate_mask(256, mask_ratio=0.75)
# 64 visible patches, 192 masked patches
```

### Why 75%?

The masking ratio sounds like a minor knob, but it decides how much the model is forced to learn:

| Masking Ratio | Effect |
|---------------|--------|
| **25%** | Too easy. With so many patches still visible, the model can fill the gaps by copying from the neighbors. It picks up local texture and never has to understand the scene. |
| **50%** | Harder, but interpolation is still partly an option. The representations are okay without being great. |
| **75%** | The sweet spot from He et al. (2022). Now there simply aren't enough visible patches to cheat by interpolation, so the only way to reconstruct the rest is to actually grasp what kind of place this is. |

The randomness matters just as much as the ratio. Every image gets a fresh random mask on every pass, so the model can't memorize a fixed "fill in these holes" recipe — it has to generalize.

## The MAE Architecture

The MAE consists of two components: an **encoder** and a **decoder**, with an asymmetric design.

### Encoder

The encoder is identical to the ViT from Chapter 4, with one difference that matters: **it processes only the visible patches** (25% of the image). The masked patches are not seen by the encoder at all.

```python
class MAEEncoder(nn.Module):
    def __init__(self, patch_dim, num_visible, d_model, num_heads, num_layers):
        super().__init__()
        self.patch_embed = nn.Linear(patch_dim, d_model)
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_visible + 1, d_model) * 0.02
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=num_heads,
            dim_feedforward=d_model * 4,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, visible_patches):
        x = self.patch_embed(visible_patches)
        cls = self.cls_token.expand(visible_patches.shape[0], -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.transformer(x)
        return self.norm(x)
```

Because the encoder only processes 25% of patches, training is much cheaper than running it on the full image.

### Decoder

The decoder receives the encoder's representations of visible patches plus **mask tokens**—learnable placeholder vectors, one for each masked position. Each mask token includes positional information so the decoder knows *which* patch it should reconstruct.

```python
class MAEDecoder(nn.Module):
    def __init__(self, d_encoder, d_decoder, num_patches, patch_dim,
                 num_heads, num_layers):
        super().__init__()
        self.encoder_to_decoder = nn.Linear(d_encoder, d_decoder)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_decoder) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches, d_decoder) * 0.02
        )
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=d_decoder, nhead=num_heads,
            dim_feedforward=d_decoder * 4,
            batch_first=True, norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(decoder_layer, num_layers)
        self.norm = nn.LayerNorm(d_decoder)
        self.head = nn.Linear(d_decoder, patch_dim)

    def forward(self, encoder_output, visible_idx, masked_idx, num_patches):
        enc_tokens = self.encoder_to_decoder(encoder_output[:, 1:, :])
        mask_tokens = self.mask_token.expand(
            encoder_output.shape[0], len(masked_idx), -1
        )
        full_tokens = torch.zeros(
            encoder_output.shape[0], num_patches,
            self.head.in_features  # d_decoder
        )
        full_tokens[:, visible_idx, :] = enc_tokens
        full_tokens[:, masked_idx, :] = mask_tokens
        full_tokens = full_tokens + self.pos_embed
        x = self.transformer(full_tokens)
        x = self.norm(x)
        return self.head(x[:, masked_idx, :])
```

The decoder is intentionally **lightweight**—smaller dimension, fewer layers—because its job is temporary. It exists only to provide the training signal.

## The Training Loop

Training proceeds by standard gradient descent on the **mean squared error** between the reconstructed and original patches:

$$
\mathcal{L} = \frac{1}{|M|} \sum_{i \in M} \| \hat{\mathbf{x}}_i - \mathbf{x}_i \|^2
$$

where $M$ is the set of masked patch indices, $\hat{\mathbf{x}}_i$ is the decoder's prediction, and $\mathbf{x}_i$ is the ground truth patch.

```python
# One training step
optimizer.zero_grad()
encoder_output = encoder(visible_patches)
reconstructed = decoder(encoder_output, visible_idx, masked_idx, num_patches)
loss = F.mse_loss(reconstructed, ground_truth_masked_patches)
loss.backward()
optimizer.step()
```

In practice:
- Training runs for hundreds of thousands of iterations.
- Each iteration uses a different image with a different random mask.
- The loss decreases from ~0.1 to <0.001 as the model learns to reconstruct accurately.

## After Training: Throw the Decoder Away

Here is the part that surprises people the first time: once training is done, we delete the decoder and keep only the encoder. It makes sense once you remember why the decoder existed. It was never the product — it was the pressure that forced the encoder to learn something real. Reconstruction was the exam; once it's over, the only thing we want to keep is what the encoder learned in order to pass it.

From here on, getting an embedding is simple: run a new image through the trained encoder and read off the CLS token.

```
Trained MAE
├── Encoder (KEEP) → produces embeddings
└── Decoder (DISCARD) → was only needed for training
```

```python
encoder.eval()
with torch.no_grad():
    encoder_output = encoder(all_patches)
    embedding = encoder_output[0, 0, :]  # CLS token, shape (768,)
```

This embedding—a vector of 768 numbers produced by an encoder trained via MAE on millions of satellite images—is what we have been calling an "Earth embedding" throughout this book.

## Random Weights vs. Trained Weights

The demonstrations in this chapter use random weights. With random weights:
- The encoder produces random embeddings with no semantic content.
- The decoder produces random reconstructions that look nothing like the original.
- The MSE loss is high and does not decrease.

With trained weights (Clay, Prithvi):
- The encoder produces embeddings that capture land cover, vegetation patterns, urban form, and other geographic characteristics.
- The decoder fills in the masked patches well—predicting the right spectral values, textures, and spatial patterns.
- The MSE loss is very low.

The mechanism is identical in both cases. The difference is entirely in the weights, which are the result of optimization over millions of training examples.

## Recap

MAE trains an encoder with no labels at all. Hide 75% of the patches, let the encoder see only the remaining 25%, and force it to produce a representation rich enough that a small decoder can rebuild everything that was hidden. The reconstruction error drives learning; once it's done, the decoder is thrown away and the encoder is what you keep. How good the resulting embeddings are comes down to the training data — how much of it, how varied, which sensors — and the training setup. This same recipe is what sits behind Clay, Prithvi, and the other geospatial foundation models you'll actually use.
