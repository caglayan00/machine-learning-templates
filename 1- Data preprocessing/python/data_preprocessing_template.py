# Data Preprocessing Template
#import the dataset and split between train and test set

# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split  # Updated import for train_test_split
from sklearn.preprocessing import StandardScaler

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # Independent variables
y = dataset.iloc[:, -1].values   # Dependent variable (assuming it's in the last column)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

# If you need to scale y (typically for regression tasks, not classification):
# y_train = sc_y.fit_transform(y_train.reshape(-1, 1))  # reshape if y is a 1D array
