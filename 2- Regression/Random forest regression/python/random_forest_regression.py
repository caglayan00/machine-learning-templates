# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # Feature (Position level)
y = dataset.iloc[:, 2].values    # Target (Salary)

# We don't apply training and test set because we don't have enough data

# Fitting Random Forest Regression model to the dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=100, random_state=0)
regressor.fit(X, y)

# Predicting a new result with Random Forest Regression
# Reshape input to be 2-D for prediction
y_pred = regressor.predict(np.array([[6.5]]))

# Visualizing Random Forest Regression results (for high resolution and smoother curves)
X_grid = np.arange(min(X), max(X), 0.01)  # Use a finer step size for a smoother curve
X_grid = X_grid.reshape((len(X_grid), 1))  # Reshape for prediction
plt.scatter(X, y, color='red')  # Plot actual data points
plt.plot(X_grid, regressor.predict(X_grid), color='blue')  # Plot the Random Forest Regression curve
plt.title('Reality vs Bluff (Random Forest Regression)')
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()
