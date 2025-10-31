# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import joblib

# 1. Load data
df = pd.read_csv('diabetes_readmission.csv')

# 2. Simple feature engineering
df['age'] = df['age'].astype(int)
# select features
features = ['age','gender','admission_type','discharge_disposition','num_lab_procedures','num_medications','hemoglobin','A1Cresult','diabetesMed']
target = 'readmitted'

X = df[features]
y = df[target]

# 3. Preprocessing
numeric_features = ['age','num_lab_procedures','num_medications','hemoglobin']
categorical_features = ['gender','admission_type','discharge_disposition','A1Cresult','diabetesMed']

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 4. Model pipeline
clf = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf.fit(X_train, y_train)

# 6. Save model
joblib.dump(clf, 'model.pkl')
print("Saved model to model.pkl")
