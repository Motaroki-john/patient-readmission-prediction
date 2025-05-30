import sys
print("Script started, Python version:", sys.version)
print("Current directory:", __file__)

try:
    import pandas as pd
    print("Pandas imported successfully!")
except Exception as e:
    print("Error importing pandas:", str(e))
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    import joblib
    print("All libraries imported successfully!")
except Exception as e:
    print("Error importing libraries:", str(e))
    sys.exit(1)

try:
    data = pd.read_csv('diabetic_data.csv')
    print("Dataset loaded successfully, columns:", data.columns.tolist())
except Exception as e:
    print("Error loading dataset:", str(e))
    sys.exit(1)

print("Dataset Info:")
print(data.info())
print("\nFirst 5 Rows:")
print(data.head())

data['readmitted'] = data['readmitted'].apply(lambda x: 1 if x == '<30' else 0)

print("Saving readmission_distribution.png")
plt.figure(figsize=(8, 6))
sns.countplot(x='readmitted', data=data)
plt.title('Distribution of Readmission Outcomes')
plt.xlabel('Readmitted (0 = No, 1 = Yes)')
plt.ylabel('Count')
plt.savefig('readmission_distribution.png')
plt.close()
print("readmission_distribution.png saved!")

print("Saving readmission_by_age.png")
plt.figure(figsize=(10, 6))
sns.countplot(x='age', hue='readmitted', data=data)
plt.title('Readmission by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.savefig('readmission_by_age.png')
plt.close()
print("readmission_by_age.png saved!")

categorical_cols = ['age', 'diag_1', 'diag_2', 'diag_3', 'glucose_serum', 'A1Cresult']
numeric_cols = ['num_lab_procedures', 'num_procedures', 'num_medications', 
                'number_outpatient', 'number_emergency', 'number_inpatient']

le = LabelEncoder()
for col in categorical_cols:
    if col in data.columns:
        data[col] = data[col].astype(str).replace('?', np.nan)
        data[col] = data[col].fillna(data[col].mode()[0])
        data[col] = le.fit_transform(data[col])
    else:
        print(f"Warning: Column '{col}' not found.")

for col in numeric_cols:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        data[col] = data[col].fillna(data[col].median())
    else:
        print(f"Warning: Column '{col}' not found.")

features = [col for col in categorical_cols + numeric_cols if col in data.columns]
X = data[features]
y = data['readmitted']

scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

print("\nModel Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

print("Saving feature_importance.png")
feature_names = features
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Importance': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance)
plt.title('Feature Importance for Readmission Prediction')
plt.savefig('feature_importance.png')
plt.close()
print("feature_importance.png saved!")

print("Saving readmission_model.pkl")
joblib.dump(model, 'readmission_model.pkl')
print("readmission_model.pkl saved!")

print("Saving confusion_matrix.png")
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('confusion_matrix.png')
plt.close()
print("confusion_matrix.png saved!")

print("Script completed successfully!")