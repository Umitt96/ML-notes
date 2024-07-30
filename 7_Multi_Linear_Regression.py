"""
@author: Ritalin
problem: Multi Variable Linear Regression
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

dot_number = 200

# Rastgele veri oluşturma
X = np.random.rand(dot_number, 2)

# Gerçek katsayılar ve hedef değerler
coef = np.array([3, 5])
y = 2*np.random.rand(dot_number) + np.dot(X, coef)


# Lineer regresyon modeli oluşturma ve eğitme
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# 3D grafik oluşturma
fig = plt.figure(dpi=150)
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], y)
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("y")

# Tahmin yüzeyini oluşturma
x1, x2 = np.meshgrid(np.linspace(0, 1, 10), np.linspace(0, 1, 10))
y_pred = lin_reg.predict(np.array([x1.flatten(), x2.flatten()]).T)
ax.plot_surface(x1, x2, y_pred.reshape(x1.shape), alpha=0.3, color = "red")
plt.title("3D Lineer Regresyon")