import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd

# Load your data
df = pd.ExcelFile('data_UPDATED.csv.xlsx')
df = pd.read_excel(df, 'train')

# Show columns 
print(df.columns)
# Drop street, lastName, firstName, and gender columns
df = df.drop(['street', 'lastName', 'firstName', 'gender'], axis=1)