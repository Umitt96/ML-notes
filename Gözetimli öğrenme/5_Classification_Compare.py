"""
@author: Ritalin
problem: All classification models compare
"""

from sklearn.datasets import make_classification, make_moons, make_circles
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.inspection import DecisionBoundaryDisplay
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np

# [ Datasetleri oluşturuyoruz ]
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2, n_clusters_per_class=1, random_state=42)
X += 1.55 * np.random.uniform(size=X.shape)
Xy = (X, y)

datasets = [
    Xy,
    make_moons(noise=0.1, random_state=42),
    make_circles(noise=0.1, factor=0.3, random_state=42)
]


# [ Öğrenme yöntemlerini tanımlıyoruz ]
names = ["KNN", "Linear SVM", "Decision Tree", "Random Forest", "Naive Bayes"]
classifiers = [KNeighborsClassifier(), SVC(), DecisionTreeClassifier(), RandomForestClassifier(), GaussianNB()]

fig = plt.figure(figsize=(15, 9))
i = 1

for ds_count, ds in enumerate(datasets):
    X, y = ds
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    cm_bright = ListedColormap(["red", "cyan"])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

    if ds_count == 0:
        ax.set_title("Input data")

    # Eğitim verisi
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="black")

    # Test verisi
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="black", alpha=0.8)
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

        clf = make_pipeline(StandardScaler(), clf)
        clf.fit(X_train, y_train)

        score = clf.score(X_test, y_test)
        DecisionBoundaryDisplay.from_estimator(clf, X, cmap=plt.cm.RdBu, alpha=0.7, ax=ax, eps=0.5)

        # Eğitim verisi
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="black")

        # Test verisi
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, edgecolors="black", alpha=0.6)

        if ds_count == 0:
            ax.set_title(name)
        ax.set_xticks(())
        ax.set_yticks(())

        ax.text(
            X[:, 0].max() + 0.25,
            X[:, 1].min() - 0.45,
            f"{score:.2f}",
            size=15,
            horizontalalignment="right"
        )
        i += 1

plt.tight_layout()
