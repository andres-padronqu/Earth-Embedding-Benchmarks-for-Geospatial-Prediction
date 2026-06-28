import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

DATA_PATH = "outputs/alphaearth_mexico_states_embeddings.csv"

df = pd.read_csv(DATA_PATH)

embedding_cols = [col for col in df.columns if col.startswith("A")]

print("Dataset shape:")
print(df.shape)

print("\nEmbedding columns:")
print(len(embedding_cols))

print("\nMissing values in embeddings:")
print(df[embedding_cols].isna().sum().sum())

print("\nStates:")
print(df["state"].tolist())

X = df[embedding_cols].values

# PCA
pca = PCA(n_components=2)
coords = pca.fit_transform(X)

df["PC1"] = coords[:, 0]
df["PC2"] = coords[:, 1]

print("\nExplained variance ratio:")
print(pca.explained_variance_ratio_)

plt.figure(figsize=(10, 7))
plt.scatter(df["PC1"], df["PC2"])

for _, row in df.iterrows():
    plt.annotate(row["state"], (row["PC1"], row["PC2"]), fontsize=8)

plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("PCA of AlphaEarth Embeddings for Mexican States")
plt.tight_layout()
plt.savefig("figures/alphaearth_mexico_states_pca.png", dpi=300)
plt.show()

# Cosine similarity
similarity = cosine_similarity(X)

similarity_df = pd.DataFrame(
    similarity,
    index=df["state"],
    columns=df["state"]
)

similarity_df.to_csv("outputs/alphaearth_mexico_states_similarity.csv")

print("\nMost similar states to Ciudad de Mexico:")
cdmx_sim = similarity_df["Distrito Federal"].sort_values(ascending=False)
print(cdmx_sim.head(10))

df.to_csv("outputs/alphaearth_mexico_states_with_pca.csv", index=False)