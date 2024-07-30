from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

# [1] Veri seti incelenmesi
cancer = load_breast_cancer()
DF = pd.DataFrame(data=cancer.data, columns=cancer.feature_names)
DF["target"] = cancer.target

# [2] Modelin eğitilmesi
X = cancer.data  # features
y = cancer.target  # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# [2.2] Ölçeklendirme
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)  # train verileri zaten bilindiği için buna göre değerleri küçültüyoruz
X_test = scaler.transform(X_test)

# [3] Model eğitimi
knn = KNeighborsClassifier(n_neighbors=3)  # komşu parametresi
knn.fit(X_train, y_train)  # fit fonksiyonu veriyi kullanarak knn algoritmasını eğitir

# [4] Sonuçların değerlendirmesi
y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Doğruluk değeri: ", accuracy)

conf_matrix = confusion_matrix(y_test, y_pred)
print("Karmaşıklık matrisi: ")
print(conf_matrix)

# [5] Hiperparametre ayarlaması
"""
KNN hp'si, K olursa
Accuracy , %A %B %C olur

- Böylece en yakın doğruluk değerinini hangi komşuda olduğunu bulabiliriz
- Diğer taraftan işlem gücü açısından şuan 9-10-11. değerler varsa 
en küçük olanı seçmemiz daha mantıklı olacaktır
"""

accuracy_values = []
k_values = []
for k in range(1,21):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    accuracy_values.append(accuracy)
    k_values.append(k)
    
plt.figure(dpi=150)
plt.plot(k_values,accuracy_values, "r*-")
plt.title("K'ya göre doğruluk")
plt.xlabel("k değeri")
plt.ylabel("accuracy değeri")
plt.xticks(k_values)
plt.grid(True)


# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor

# grafiği ve eğitme sürecini daha iyi anlamak için bir örnek
X = np.sort(5 * np.random.rand(40,1), axis=0) # features /rastgele sayılar
y = np.sin(X).ravel() #  target  /sinüsünün vektör hali
  
# add noise
y[::5] += 1 * (0.5 - np.random.rand(8)) #her 5 seferde rastgele bir değişim yap

T = np.linspace(0,5,500)[:, np.newaxis]

for i, weight in enumerate(["uniform","distance"]):
    knn = KNeighborsRegressor(n_neighbors=5, weights=weight)
    y_pred = knn.fit(X, y).predict(T)

    plt.subplot(2,1, i+1)
    plt.scatter(X, y, color = "blue", label = "data")
    plt.plot(T,y_pred, color = "green", label = "predict")
    plt.axis("tight")
    plt.legend()
    plt.title("KNN regression weight = {}".format(weight))

plt.tight_layout()












