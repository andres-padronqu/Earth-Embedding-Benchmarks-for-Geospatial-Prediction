# What Are Embeddings?

This chapter gets concrete about embeddings: what one actually looks like, how you measure whether two of them are alike, how to visualize a whole space of them, and how to drop them into a regression like any other variable. We'll worry about how they're produced in Chapters 4–5; here we just treat the model as a black box that hands us vectors.

## From Images to Vectors

Say we have 100 satellite images, one per location. We push each through a pre-trained network and get back 768 numbers per image. Stack those and you have an **embedding matrix** of shape `(100, 768)`:

```python
import numpy as np

embeddings = np.load("embeddings.npy")
print(f"Shape: {embeddings.shape}")
# (100, 768)
# 100 images, 768 dimensions per embedding
```

Each row is a single image, compressed from millions of pixel values into 768 numbers. These 768 numbers have no individual interpretation—dimension 42 does not mean "vegetation" and dimension 300 does not mean "urbanization." Their meaning is **relational**: it emerges from comparing embeddings to each other.

This is analogous to factor scores in psychometrics or principal components in multivariate statistics. A single PC score is meaningless in isolation; it gains meaning from the loadings and from how observations differ along that dimension.

## Cosine Similarity

The primary way to compare embeddings is **cosine similarity**, which measures the cosine of the angle between two vectors:

$$
\text{sim}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}
$$

- **sim = 1**: vectors point in the same direction (very similar locations).
- **sim = 0**: vectors are orthogonal (unrelated locations).
- **sim = −1**: vectors point in opposite directions (maximally different).

Cosine similarity is conceptually equivalent to a Pearson correlation between the two embedding vectors. The key property is that it is **invariant to magnitude**—it measures the pattern of values, not their scale.

```python
from sklearn.metrics.pairwise import cosine_similarity

sim_matrix = cosine_similarity(embeddings)
# sim_matrix[i, j] = cosine similarity between image i and image j
```

In practice, images of the same land cover type tend to have high cosine similarity (> 0.8), while images of very different types have lower similarity (< 0.5). This holds even when the images come from different countries or continents:

```python
# Illustrative pairwise similarities
cases = [
    ("Forest A",  "Forest B",       0.92, "same type, different country"),
    ("Forest A",  "Urban A",        0.31, "very different types"),
    ("Forest A",  "Cropland A",     0.68, "both vegetation"),
    ("Urban A",   "Informal Settl.", 0.74, "both urban, different form"),
    ("Water A",   "Water B",        0.95, "same type"),
]
```

This is the whole point: **embeddings capture semantic similarity**. Two forests land close together in embedding space because the network has learned they share visual patterns (canopy texture, spectral signature, spatial structure), even though their raw pixel values are nothing alike.

## Mean Similarity Between Classes

To evaluate whether embeddings systematically capture land cover differences, we can compute the mean cosine similarity between all pairs of images within and across classes:

```python
class_names = ["Forest", "Water", "Urban", "Cropland", "Settlement"]
labels = np.array([0]*20 + [1]*20 + [2]*20 + [3]*20 + [4]*20)

for i, name_i in enumerate(class_names):
    for j, name_j in enumerate(class_names):
        mask_i = labels == i
        mask_j = labels == j
        mean_sim = sim_matrix[np.ix_(mask_i, mask_j)].mean()
        # Within-class similarities are high (~0.85-0.95)
        # Cross-class similarities are lower (~0.30-0.70)
```

The resulting similarity matrix reveals semantic structure:
- Forest and Cropland have moderate cross-class similarity (both involve vegetation).
- Urban and Settlement have moderate cross-class similarity (both involve built environment).
- Forest and Water have low cross-class similarity (very different materials).

These patterns emerge **without any supervision**—the embedding model was never told what "forest" or "urban" means. It learned these distinctions from the statistical structure of satellite imagery.

## Visualizing the Embedding Space

Embedding vectors live in a 768-dimensional space that we cannot visualize directly. To see the structure, we project down to 2 dimensions using **t-SNE** (t-distributed Stochastic Neighbor Embedding):

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, random_state=42, perplexity=15)
embeddings_2d = tsne.fit_transform(embeddings)
# embeddings_2d: (100, 2)
```

> **Warning — Interpreting t-SNE.** The two axes in a t-SNE plot have **no direct interpretation**. They are artificial coordinates computed to preserve local neighborhood structure. Only the **relative positions** of points matter: nearby points have similar embeddings, distant points have different embeddings. Do not interpret axis values, distances between clusters, or cluster sizes literally.

In a well-trained embedding space, the t-SNE visualization will show clear clusters corresponding to land cover types. Points within a cluster are images that the model considers similar; the gaps between clusters represent semantic boundaries.

## Embeddings as Variables in Statistical Models

The key practical insight is that an embedding matrix is just a data matrix. Each row is an observation (a location), and each column is a feature. This means embeddings can be used directly as independent variables in any standard statistical model.

### The Pipeline: PCA → Regression

With 768 embedding dimensions and potentially fewer observations, dimensionality reduction is advisable before regression. A standard pipeline:

```python
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Step 1: Reduce dimensionality
pca = PCA(n_components=20, random_state=42)
X_pca = pca.fit_transform(embeddings)
variance_explained = pca.explained_variance_ratio_.sum()
print(f"Variance explained by 20 components: {variance_explained:.1%}")

# Step 2: Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_pca, poverty_index, test_size=0.2, random_state=42
)

# Step 3: Regression
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
print(f"R² on test set: {r2:.3f}")
```

With 20 PCA components derived from embeddings, a simple linear regression can explain a substantial fraction of variance in socioeconomic outcomes. Worth sitting with for a second: 768 numbers from a vision model trained on satellite images alone end up carrying information about poverty, income, and access to infrastructure.

### Adding Embeddings to Existing Models

In practice, embeddings tend to pull their weight when combined with traditional data sources rather than used alone. A common pattern:

```python
import pandas as pd

# Embedding features
df_emb = pd.DataFrame(X_pca, columns=[f"PC{i+1}" for i in range(20)])

# Census / survey features
df_socio = pd.DataFrame({
    "population": [...],
    "pct_employed": [...],
    "housing_quality_index": [...]
})

# Merge
df_full = pd.concat([df_emb, df_socio], axis=1)

# Now run your model with both feature sets
# Compare: embeddings only, sociodemographics only, both
```

The research question then becomes: **do embeddings add predictive power beyond what traditional sociodemographic variables already capture?** This question of marginal contribution is directly testable through standard model comparison techniques (nested F-tests, AIC/BIC, cross-validated performance).

## Recap

An embedding is a fixed-length vector standing in for a place. You compare two of them with cosine similarity, and when you look across many of them you find real structure — similar land cover lands in the same neighborhood of the space, even though nobody told the model what "forest" or "urban" means. t-SNE helps you see that structure, as long as you remember the axes mean nothing on their own. And because an embedding is just a row of numbers, it slots straight into a regression (usually after a PCA step). For our purposes the question that matters is always the same: does it tell us anything our existing covariates don't already?
