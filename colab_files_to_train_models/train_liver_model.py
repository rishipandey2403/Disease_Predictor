import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Load dataset
data = pd.read_csv('../dataset/liver_disease.csv', header=None)

data.columns = [
    'Age','Gender','TB','DB','Alkphos',
    'SGPT','SGOT','TP','ALB','AG','Target'
]

# Encode gender
data['Gender'] = data['Gender'].map({'Male':1, 'Female':0})

# Fill missing values
data.fillna(data.mean(), inplace=True)

X = data.drop('Target', axis=1)
y = data['Target']

# Convert labels
y = y.apply(lambda x: 1 if x == 1 else 0)

# Scale features
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Balanced split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# Balanced model
model = RandomForestClassifier(class_weight='balanced')

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print("Model accuracy:", accuracy)

# Save model + scaler
pickle.dump((model, scaler), open('../saved_models/liver_model.sav', 'wb'))

print("Liver model retrained and saved!")
