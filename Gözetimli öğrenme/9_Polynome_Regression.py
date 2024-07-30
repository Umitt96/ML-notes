"""
@author: Ritalin
problem: Polynomial Regression with numpy data
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

noise = 6*np.random.rand(100, 1)

X = 5 * np.random.rand(100, 1)
y = 2 + 3*X**2 + noise  # y = 2 + 3x^2


poly_feature = PolynomialFeatures(degree=2)
X_poly = poly_feature.fit_transform(X)

poly_reg = LinearRegression()
poly_reg.fit(X_poly, y)

plt.scatter(X, y, color="blue", alpha=.7)

X_test = np.linspace(0, 5, 100).reshape(-1, 1)
X_test_poly = poly_feature.transform(X_test)
y_pred = poly_reg.predict(X_test_poly)

plt.plot(X_test, y_pred, color="red")
plt.xlabel("X değeri")
plt.ylabel("Y değeri")
plt.title("Polinom Regresyon modeli")
