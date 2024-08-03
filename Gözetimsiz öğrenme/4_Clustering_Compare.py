"""
@author: Rita_bey
problem: Clustering_Compare
"""

from sklearn import datasets, cluster
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")

n_samples = 1000
n_cls = 2

# veri setlerinin oluşturulması
noisy_circles = datasets.make_circles(
    n_samples=n_samples, factor=0.5, noise=0.05)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=0.05)
blobs = datasets.make_blobs(n_samples=n_samples)
no_structure = np.random.rand(n_samples, 2), None

# kümeleme algoritması isimleri
clustering_names = ["MiniBatchKMeans", "SpectralClustering", "Ward",
                    "AgglomerativeClustering", "DBSCAN", "Birch"]


color = np.array(["b", "g", "r", "c", "m", "y"])
datasets = [noisy_circles, noisy_moons, blobs, no_structure]

plt.figure()
# grafikleştirme için for döngüsü
i = 1
for i_dataset, dataset in enumerate(datasets):

    X, y = dataset
    X = StandardScaler().fit_transform(X)

    two_means = cluster.MiniBatchKMeans(n_clusters=n_cls)
    ward = cluster.AgglomerativeClustering(n_clusters=n_cls, linkage="ward")
    spectral = cluster.SpectralClustering(n_clusters=n_cls)
    DBSCAN = cluster.DBSCAN(eps=n_cls*0.1)
    average_linkage = cluster.AgglomerativeClustering(n_clusters=n_cls, linkage="average")
    birch = cluster.Birch(n_clusters=n_cls)

    clustering_algorithms = [two_means, ward,
                             spectral, DBSCAN, average_linkage, birch]

    for name, algs, in zip(clustering_names, clustering_algorithms):

        algs.fit(X)

        if hasattr(algs, "labels_"):
            y_pred = algs.labels_.astype(int)
        else:
            y_pred = algs.predict(X)

        plt.subplot(len(datasets), len(clustering_algorithms), i)
        
        if i_dataset == 0:
            plt.title(name)
        plt.scatter(X[:,0],X[:,1], color = color[y_pred].tolist())

        print(f"* {i}. {name} Algoritması -> hesaplandı")
        i += 1
