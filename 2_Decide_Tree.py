"""
@author: Ritalin
"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt

# Veri seti incelemesi

iris = load_iris()
X = iris.data #features
y = iris.target #target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# DT modeli oluşturup train etme
tree_clf = DecisionTreeClassifier(criterion="gini",max_depth= 6, random_state=42)
tree_clf.fit(X_train,y_train)

y_pred = tree_clf.predict(X_test)

# Sonuçlar
accurary = accuracy_score(y_test, y_pred)
print("Decision Tree model doğruluğu: ",accurary)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Karmaşıklık matrisi: ")
print(conf_matrix)

# Grafikleştirme
plt.figure(dpi=150)
plot_tree(tree_clf, filled = True, feature_names= iris.feature_names, class_names = list(iris.target_names))

#en tepedeki veri, en önemli veridir çünkü kararları ona göre ayarlıyoruz


# %%

"""

Üstteki grafiğin biraz daha detaylı hali olarak düşün
"""

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import numpy as np
import warnings

warnings.filterwarnings("ignore")

# Veri setini yükleme
iris = load_iris()
number_class = len(iris.target_names)
plot_colors = "ryb"

# Özellik çiftleri üzerinde döngü
for pairidx, pair in enumerate([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]]):
    X = iris.data[:, pair]
    y = iris.target
    
    clf = DecisionTreeClassifier().fit(X, y)
    
    # Alt grafiği oluşturma
    ax = plt.subplot(2, 3, pairidx + 1)
    plt.tight_layout()
    DecisionBoundaryDisplay.from_estimator(clf, X, 
                                           cmap=plt.cm.RdYlBu, 
                                           response_method="predict",
                                           ax=ax,
                                           xlabel=iris.feature_names[pair[0]],
                                           ylabel=iris.feature_names[pair[1]])

    # Veri noktalarını plot etme
    for i, color in zip(range(number_class), plot_colors):
        idx = np.where(y == i)
        plt.scatter(X[idx, 0], X[idx, 1], 
                    c=color, 
                    label=iris.target_names[i],
                    cmap=plt.cm.RdYlBu,
                    edgecolor='black', 
                    s=15)

plt.legend()


# %%
"""
Regresyon veri setini Decision Tree ile anlamak
Diyabet

"""
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import matplotlib.pyplot as plt

X = np.sort(5* np.random.rand(80,1), axis = 0)
y = np.sin(X).ravel()

y[::5] += 1.6 * (0.5 - np.random.rand(16))

#plt.plot(X, y)

regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=8)
regr_1.fit(X,y)
regr_2.fit(X,y)

X_test = np.arange(0,5,0.05)[:,np.newaxis]
y_pred_1 = regr_1.predict(X_test)
y_pred_2 = regr_2.predict(X_test)

plt.figure()
plt.scatter(X, y, c = "red", label = "data")
plt.plot(X, y, c = "red", label = "data")
plt.plot(X_test, y_pred_1, color = "blue", label = "MaxDep: 2")
plt.plot(X_test, y_pred_2, color = "black", label = "MaxDep: 8")

plt.xlabel("data")
plt.ylabel("target")
plt.legend()

#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)






