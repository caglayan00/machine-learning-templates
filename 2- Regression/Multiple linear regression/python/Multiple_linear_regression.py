#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 23 11:08:12 2017

Multiple linear regression
Import dataset and encode categorical variables, fit multiple linear regression
find best model using backward elimination

@author: Ilaria
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

# Encode categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

# Encode the categorical column (assuming column index 3 for State)
label_encoder_X = LabelEncoder()
X[:, 3] = label_encoder_X.fit_transform(X[:, 3])

# OneHotEncoder to avoid the dummy variable trap
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Avoid the Dummy Variable Trap by removing one dummy variable
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting Test set results
y_pred = regressor.predict(X_test)

# Build the optimal model using Backward Elimination
import statsmodels.api as sm

# Add constant column for intercept
X = np.append(arr=np.ones((X.shape[0], 1)).astype(int), values=X, axis=1)

# Fit model with all predictors
X_opt = X[:, :]
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())

# Backward elimination
def backward_elimination(X, y, significance_level=0.05):
    num_vars = X.shape[1]
    for i in range(num_vars):
        regressor_OLS = sm.OLS(endog=y, exog=X).fit()
        p_values = regressor_OLS.pvalues
        max_p_value = max(p_values)
        if max_p_value > significance_level:
            # Remove the feature with the highest p-value
            for j in range(num_vars - i):
                if p_values[j] == max_p_value:
                    X = np.delete(X, j, axis=1)
        else:
            break
    return X

# Apply backward elimination
X_opt = backward_elimination(X, y)
regressor_OLS = sm.OLS(endog=y, exog=X_opt).fit()
print(regressor_OLS.summary())
