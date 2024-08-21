# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # Features (reshape to 2D array)
y = dataset.iloc[:, 2].values    # Target variable

# No need to split the dataset due to small size
# Splitting the dataset into Training set and Test set is not applied

# Fitting Linear Regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualizing Linear Regression Results
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.scatter(X, y, color='red')
plt.plot(X, lin_reg.predict(X), color='blue')
plt.title('Linear Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')

# Visualizing Polynomial Regression Results
plt.subplot(1, 2, 2)
X_grid = np.arange(min(X), max(X), 0.01)  # Smoother curve
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape for prediction
plt.scatter(X, y, color='red')
plt.plot(X_grid, lin_reg_2.predict(poly_reg.transform(X_grid)), color='blue')
plt.title('Polynomial Regression')
plt.xlabel('Position Level')
plt.ylabel('Salary')

plt.show()

# Predicting a new result with Linear Regression
lin_reg_prediction = lin_reg.predict([[6.5]])
print(f'Linear Regression Prediction for 6.5: {lin_reg_prediction[0]}')

# Predicting a new result with Polynomial Regression
poly_reg_prediction = lin_reg_2.predict(poly_reg.transform([[6.5]]))
print(f'Polynomial Regression Prediction for 6.5: {poly_reg_prediction[0]}')
