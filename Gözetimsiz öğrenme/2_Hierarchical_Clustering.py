"""
@author: Ritalin
problem: Hierarchical Clustering
"""

from sklearn.datasets import make_blobs
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage

X, _ = make_blobs(n_samples=300, centers=3, cluster_std=2, random_state=42)
linkage_methods = ["ward", "single", "average", "complete"]  # 4 farklı metot


plt.figure(dpi=120)
for i, linkage_methods in enumerate(linkage_methods, 1):
    model = AgglomerativeClustering(n_clusters=3, linkage=linkage_methods)
    cluster_labels = model.fit_predict(X)
    # burada train-test-split yapılmamasının sebebi her küme için x,y noktası yok

    plt.subplot(2, 4, i)  # ilk 4 satır
    plt.title(f"{linkage_methods} linkage dendogram'ı")
    dendrogram(linkage(X, method=linkage_methods))
    plt.xlabel("Veri noktaları")
    plt.ylabel("uzaklık")

    plt.subplot(2, 4, i+4)  # ikinci 4 satır
    plt.scatter(X[:, 0], X[:, 1], c=cluster_labels, cmap="viridis")
    plt.title(f"{linkage_methods} linkage scatter'ı")
    plt.xlabel("X")
    plt.ylabel("Y")
