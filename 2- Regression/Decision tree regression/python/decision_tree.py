# Importing the necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values  # Position level (independent variable)
y = dataset.iloc[:, 2].values    # Salary (dependent variable)

# Note: No train-test split as the dataset is very small.

# Fitting the Decision Tree Regression model to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X, y)

# Predicting a new result with Decision Tree Regression
y_pred = regressor.predict([[6.5]])  # Make sure to pass a 2D array for prediction
print(f"Predicted salary for position level 6.5: {y_pred[0]}")

# Visualizing the Decision Tree Regression results (for higher resolution and smoother curve)
X_grid = np.arange(min(X), max(X), 0.01)  # Higher resolution grid for a smoother curve
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color='red')
plt.plot(X_grid, regressor.predict(X_grid), color='blue')
plt.title('Reality vs Bluff (Decision Tree Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()
