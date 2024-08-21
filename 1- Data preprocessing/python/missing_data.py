import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Data.csv')

# Separating independent variables (X) and dependent variable (y)
x = dataset.iloc[:, :-1].values  # Independent variables
y = dataset.iloc[:, -1].values   # Dependent variable

# Handling missing data using SimpleImputer
from sklearn.impute import SimpleImputer

# Define the imputer to replace missing values with the mean of the column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')

# Fit the imputer to the appropriate columns (assuming columns 1 and 2 have missing data)
imputer = imputer.fit(x[:, 1:3])

# Transform the data to fill in the missing values
x[:, 1:3] = imputer.transform(x[:, 1:3])

# Now `x` and `y` are ready for further processing or model training
