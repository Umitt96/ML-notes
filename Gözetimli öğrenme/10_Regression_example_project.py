"""
@author: Ritalin
problem: Regression_example_project
"""
# Gerekli Kütüphanelerin İçe Aktarılması
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# California Konut Verilerinin Yüklenmesi
cf_housing = fetch_california_housing()

# Data ve target Değişkenlerin Ayrılması
X = cf_housing.data
y = cf_housing.target

# Verilerin Eğitim ve Test Setlerine Bölünmesi
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Polinomal Özelliklerin Oluşturulması
poly_feature = PolynomialFeatures(degree=2)
X_train_poly = poly_feature.fit_transform(X_train)
X_test_poly = poly_feature.transform(X_test)

# Polinomal Regresyon Modelinin Eğitilmesi ve Tahmin Yapılması, performansı
poly_reg = LinearRegression()
poly_reg.fit(X_train_poly, y_train)
y_pred = poly_reg.predict(X_test_poly)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"polynomial regression rmse: {rmse}")

# Doğrusal Regresyon Modelinin Eğitilmesi ve Tahmin Yapılması, performansı
lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred = lin_reg.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f"linear regression rmse: {rmse}")
