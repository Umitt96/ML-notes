"""
@author: Ritalin

problem: use olivetti_faces library with classification
"""
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

oli_faces = fetch_olivetti_faces()
    
X = oli_faces.data
y = oli_faces.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


rf_clf = RandomForestClassifier(n_estimators= 110, random_state= 42)
rf_clf.fit(X_train, y_train)
y_pred = rf_clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)  
print(acc)


#%%
"""
problem: California'daki ev fiyatları, Regression
"""

from sklearn.datasets import fetch_california_housing

california_housing = fetch_california_housing()

X = california_housing.data 
y = california_housing.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
