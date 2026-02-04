import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler

# Load dataset
data = pd.read_csv('../dataset/liver_disease.csv', header=None)

# Rename columns
data.columns = [
    'Age', 'Gender', 'TB', 'DB', 'Alkphos',
    'SGPT', 'SGOT', 'TP', 'ALB', 'A/G', 'Target'
]

# Convert gender to numeric
data['Gender'] = data['Gender'].map({'Male':1, 'Female':0})

# Handle missing values
data.fillna(data.mean(), inplace=True)

X = data.drop(columns='Target', axis=1)
Y = data['Target']

# Convert labels: 1 = disease, 0 = no disease
Y = Y.apply(lambda x: 1 if x == 1 else 0)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, stratify=Y, random_state=2
)

model = SVC(kernel='linear')
model.fit(X_train, Y_train)

# Save model
pickle.dump(model, open('../saved_models/liver_model.sav', 'wb'))

print("Liver disease model trained and saved!")
