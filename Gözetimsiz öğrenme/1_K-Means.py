"""
@author: Ritalin
problem: K-Means kümeleme
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

X, _ = make_blobs(n_samples=300, centers=5, cluster_std=1.2, random_state=42)

plt.figure()
plt.title("Example")

plt.scatter(X[:, 0], X[:, 1])
kmeans = KMeans(n_clusters=5)
kmeans.fit(X)


plt.figure()
plt.title("K-Means")

label = kmeans.labels_
plt.scatter(X[:, 0], X[:, 1], c=label, cmap="brg", alpha=.4)

centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c="black")

