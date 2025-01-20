from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pandas as pd

le = LabelEncoder()

# Load the data
df = pd.ExcelFile('data_UPDATED.csv.xlsx')
df = pd.read_excel(df, 'train')
df = df.dropna()
df = df.drop(['gender', 'street', 'city', 'state', 'zip', 'latitude', 'longitude', 'transNum', 'merchLongitude', 'merchLatitude'], axis=1)

# Preprocess test data
# Handling missing values, encode categorical variables, etc.

# Convert the 'date' column to int
df['transDate'] = pd.to_datetime(df['transDate'])
df['transDate'] = df['transDate'].dt.strftime('%Y%m%d').astype(int)

# Using the label encoder to encode categorical variables (for model training)
df['business'] = le.fit_transform(df['business'])
df['category'] = le.fit_transform(df['category'])
df['firstName'] = le.fit_transform(df['firstName'])
df['lastName'] = le.fit_transform(df['lastName'])
df['job'] = le.fit_transform(df['job'])

# Convert date of birth to int
df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'])
df['dateOfBirth'] = df['dateOfBirth'].dt.strftime('%Y%m%d').astype(int)

# Split the data into features (X) and target (y)
X = df.drop('isFraud', axis=1)
y = df['isFraud']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
print("Accuracy:", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Run the model against new data
df_other = pd.read_excel('data_UPDATED.csv.xlsx', 'test')
print(df_other.columns)

# Preprocess test data
df_other['transDate'] = pd.to_datetime(df_other['transDate'])
df_other['transDate'] = df_other['transDate'].dt.strftime('%Y%m%d').astype(int)

df_other = df_other.drop(['gender', 'city', 'state', 'zip', 'street', 'latitude', 'longitude', 'transNum', 'merchLatitude', 'merchLongitude'], axis=1)

df_other['business'] = le.fit_transform(df_other['business'])
df_other['category'] = le.fit_transform(df_other['category'])
df_other['firstName'] = le.fit_transform(df_other['firstName'])
df_other['lastName'] = le.fit_transform(df_other['lastName'])
df_other['job'] = le.fit_transform(df_other['job'])

# Convert date of birth to int
df_other['dateOfBirth'] = pd.to_datetime(df_other['dateOfBirth'])
df_other['dateOfBirth'] = df_other['dateOfBirth'].dt.strftime('%Y%m%d').astype(int)

# Split the data into features (X) and target (y)
X_other = df_other.drop('isFraud', axis=1)
y_other = df_other['isFraud']

# Make predictions on the new data
y_other_pred = model.predict(X_other)

df_other['predicted_isFraud'] = y_other_pred
df_other.to_excel('output.xlsx', index=False)