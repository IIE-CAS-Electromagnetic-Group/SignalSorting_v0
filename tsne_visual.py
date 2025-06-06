import torch
import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


features = torch.load("tsne_features.pt").numpy()
labels = torch.load("tsne_labels.pt").numpy()

# 步骤1：先用 PCA 降到 20 维
pca = PCA(n_components=20)
features_pca = pca.fit_transform(features)

# 步骤2：再用 t-SNE 降到 2 维
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
features_tsne = tsne.fit_transform(features_pca)

# 步骤3：绘制 t-SNE 散点图
plt.figure(figsize=(10, 8))
for label in np.unique(labels):
    plt.scatter(features_tsne[labels == label, 0],
                features_tsne[labels == label, 1],
                label=f"Class {label}",
                alpha=0.6, s=20)
plt.legend()
plt.title("t-SNE Visualization of CNN Features")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.grid(True)
plt.tight_layout()
plt.savefig("tsne_visualization.png")
plt.show()
