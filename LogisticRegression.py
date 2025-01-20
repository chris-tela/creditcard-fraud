# import necessary libraries 
import numpy as np
import pandas as pd 

import matplotlib.pyplot as plt 
# %matplotlib inline
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error

# import dataset
creditcard_file = pd.ExcelFile("data_UPDATED.csv.xlsx")
trainData = pd.read_excel(creditcard_file, "train")

# Explore the dataset 
sns.countplot(x="isFraud", data=trainData)
sns.countplot(x="isFraud", hue="gender", data=trainData)
sns.countplot(x="isFraud", hue="category", data=trainData)
sns.countplot(x="isFraud", hue="state", data=trainData)

plt.hist(trainData["amount"])

# imputation 