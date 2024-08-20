# Data Preprocessing
#this code imports the dataset, convert missing data in their mean and
#encode dependent and indipendent variable 

# Importing the necessary libraries
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Importing the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values  # Independent variables
y = dataset.iloc[:, -1].values   # Dependent variable (assuming it's in the last column)

# Taking care of missing data using SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

# Encoding categorical data for Independent Variable using ColumnTransformer and OneHotEncoder
# This will apply OneHotEncoder only to the first column (index 0)
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))

# Encoding the Dependent Variable using LabelEncoder
labelencoder_y = LabelEncoder()
y = labelencoder_y.fit_transform(y)

# Now X and y are ready for model training or further processing
