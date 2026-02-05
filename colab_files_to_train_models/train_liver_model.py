import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from sklearn.metrics import classification_report

# Load dataset
data = pd.read_csv('../dataset/liver_disease.csv', header=None)

data.columns = [
    'Age','Gender','TB','DB','Alkphos',
    'SGPT','SGOT','TP','ALB','AG','Target'
]

# Encode gender
data['Gender'] = data['Gender'].map({'Male':1, 'Female':0})

data.fillna(data.mean(), inplace=True)

# Separate classes
disease = data[data.Target == 1]
healthy = data[data.Target == 2]

# Downsample disease to match healthy
disease_down = resample(disease,
                        replace=False,
                        n_samples=len(healthy),
                        random_state=42)

balanced = pd.concat([disease_down, healthy])

X = balanced.drop('Target', axis=1)
y = balanced['Target'].apply(lambda x: 1 if x == 1 else 0)

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42
)

model = RandomForestClassifier(n_estimators=300)
model.fit(X_train, y_train)

print(classification_report(y_test, model.predict(X_test)))

pickle.dump((model, scaler), open('../saved_models/liver_model.sav', 'wb'))

print("Balanced liver model saved!")
