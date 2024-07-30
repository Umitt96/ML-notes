"""
@author: Ritalin
problem: Linear Regression example
"""

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt


dot_number = 150


X = np.random.rand(dot_number, 1)
y = 3 + 4 * X + np.random.rand(dot_number, 1)

lin_reg = LinearRegression()
lin_reg.fit(X, y)

plt.figure()
plt.scatter(X, y)
plt.plot(X, lin_reg.predict(X), color="red", alpha=.6)
plt.title("Lineer Regresyon")

# düz çizginin konumu
a1 = lin_reg.coef_[0][0]
a0 = lin_reg.intercept_[0]

for i in range(dot_number):
    y_ = a0 + a1 * X
    plt.plot(X, y_, color="green", alpha=.6)

# %%

# problem: Linear Regression with diabetes


diabet = load_diabetes()

# X = diabet.data yerine bunu kullanabiliriz
diabet_X, diabet_y = load_diabetes(return_X_y=True)
diabet_X = diabet_X[:, np.newaxis, 2]  # bmi sütunu

# test train split ama manuel
diabet_X_train = diabet_X[:-20]
diabet_X_test = diabet_X[-20:]
diabet_y_train = diabet_y[:-20]
diabet_y_test = diabet_y[-20:]

lin_reg = LinearRegression()
lin_reg.fit(diabet_X_train, diabet_y_train)
diabet_y_pred = lin_reg.predict(diabet_X_test)

# sonuçlar
mse = mean_squared_error(diabet_y_test, diabet_y_pred)
r2 = r2_score(diabet_y_test, diabet_y_pred)
print(f'mse: {mse} \nr2 score: {r2}')

# grafik
plt.scatter(diabet_X_test, diabet_y_test, color="black")
plt.plot(diabet_X_test, diabet_y_pred, color="blue")
