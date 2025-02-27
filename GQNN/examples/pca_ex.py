from GQNN.data.pca import PCA
import numpy as np 
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data  
y = iris.target  

pca = PCA(n_components=2)

X_transformed = pca.fit_transform(X)

print("Principal Components:\n", pca.components)
print("\nTransformed Data (first 5 samples):\n", X_transformed[:5])
print("\nExplained Variance Ratio:\n", pca.explained_variance_ratio())

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))
for target, color, label in zip(np.unique(y), ['r', 'g', 'b'], iris.target_names):
    plt.scatter(
        X_transformed[y == target, 0], 
        X_transformed[y == target, 1], 
        color=color, 
        label=label, 
        alpha=0.7
    )

plt.title("PCA on Iris Dataset (2 Principal Components)")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.legend()
plt.grid()
plt.show()