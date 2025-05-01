"""Steps 
1. Load the data
2. Preprocess the data (handle missing values, encode categorical variables, etc.)  
3. Visualize the data (correlation matrix, pairplot, etc.)
4. Split the data into training and testing sets
5. Train a Random Forest model
6. Evaluate the model (accuracy, classification report, confusion matrix)
7. Make predictions on the test set
8. Save the predictions to a CSV file"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Load the data
data = pd.read_csv(r"C:\Users\Rushil\Desktop\training\Titanic\train.csv")
data_test = pd.read_csv(r"C:\Users\Rushil\Desktop\training\Titanic\test.csv")

# Display basic structure and info
print("\n" + "="*60)

print(data.head())

print("\n" + "="*60)

print(data_test.head())

print("\n" + "="*60)

print(data.tail())

print("\n" + "="*60)

print(data_test.tail())

print("\n" + "="*60)

print(data.describe())

print("\n" + "="*60)

print(data.isnull().sum())

print("\n" + "="*60)

print(data_test.isnull().sum())

# Handle missing values by filling with median for numerical features
data['Age'].fillna(data['Age'].median(), inplace=True)
data_test['Age'].fillna(data_test['Age'].median(), inplace=True)
data_test['Fare'].fillna(data_test['Fare'].median(), inplace=True)

# Drop unnecessary columns
data.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True)
data_test.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'], inplace=True)

# Encode categorical variables
label_encoders = {}
categorical_cols = data.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

for col in data_test.select_dtypes(include='object').columns:
    if col in label_encoders:
        data_test[col] = label_encoders[col].transform(data_test[col])
    else:
        data_test[col] = LabelEncoder().fit_transform(data_test[col])

# Generates the Correlation Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Pairplot
sns.pairplot(data, hue='Survived', diag_kind='kde', palette='coolwarm')
plt.show()

# Countplot for Gender
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('Survival by Gender')
plt.show()

# Age Distribution
sns.histplot(data=data, x='Age', hue='Survived', bins=30, kde=True)
plt.title('Age Distribution by Survival')
plt.show()

# Pclass vs Survival
sns.countplot(x='Pclass', hue='Survived', data=data)
plt.title('Survival by Passenger Class')
plt.show()

# Features and Target
Y = data['Survived']
X = data.drop(columns=['Survived', 'PassengerId'])

# Prepare test data
test_passenger_ids = data_test['PassengerId']
X_test_data = data_test.drop(columns=['PassengerId'])

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_test = scaler.transform(X_test_data)

# Train-test split
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=1)

# Random Forest with Overfitting Control
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=42
)

# Model Training
model.fit(X_train, Y_train)

# Predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Evaluation Matrix
print("\n" + "="*60)
print(f"Model: {model}")
print(f"Training Accuracy: {accuracy_score(Y_train, train_preds):.4f}")
print(f"Testing Accuracy : {accuracy_score(Y_test, test_preds):.4f}")

print("\nClassification Report (Test Data):")
print(classification_report(Y_test, test_preds))

#Generates Confusion Matrix
cm = confusion_matrix(Y_test, test_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Final Prediction on test set
final_predictions = model.predict(X_scaled_test)

# Saves output in CSV forma
output = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': final_predictions})
output.to_csv('result_new.csv', index=False)
print("\nSubmission file created: result_new.csv")
