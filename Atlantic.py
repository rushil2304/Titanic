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

# Loads and reads the data
data = pd.read_csv(r"C:\Users\Rushil\Desktop\training\Atlantic\train.csv")
data_test = pd.read_csv(r"C:\Users\Rushil\Desktop\training\Atlantic\test.csv")

# Drop non-useful columns/features
data.drop(columns=['Name','Ticket','Cabin','Embarked'], inplace=True)
data_test.drop(columns=['Name', 'Ticket', 'Cabin','Embarked'], inplace=True)

# Encode categorical variables in training data using LabelEncoder
label_encoders = {}
categorical_cols = data.select_dtypes(include='object').columns

for col in categorical_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Encode categorical variables in test data using same encoders becuase random forest does not support categorical variables so we need to covert them
categorical_cols_test = data_test.select_dtypes(include='object').columns
for col in categorical_cols_test:
    if col in label_encoders:
        data_test[col] = label_encoders[col].transform(data_test[col])
    else:
        data_test[col] = LabelEncoder().fit_transform(data_test[col])

# Heatmap or correlation matrix (only numeric features)
numeric_data = data.select_dtypes(include='number')
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', linewidths=0.2)
plt.gcf().set_size_inches(10, 8)
plt.show()

# generates a pairplot
sns.pairplot(numeric_data, hue='Survived', diag_kind='kde', palette='coolwarm')
plt.show()

# Countplot: Survival by Gender graph
sns.countplot(x='Sex', hue='Survived', data=data)
plt.title('Survival by Gender')
plt.show()

# Prepares the training data
Y = data['Survived']
X = data.drop(columns=['Survived', 'PassengerId'])

# Prepare test data (save PassengerId for final output)
test_passenger_ids = data_test['PassengerId']
X_test_data = data_test.drop(columns=['PassengerId'])

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_test = scaler.transform(X_test_data)

# Train-test split(80/20)
X_train, X_test, Y_train, Y_test = train_test_split(X_scaled, Y, test_size=0.2, stratify=Y, random_state=1)

# Define model(Random forest classifier)
model = RandomForestClassifier(
    random_state=42  
)

# Trains the model
model.fit(X_train,Y_train)

# Make predictions
train_preds = model.predict(X_train)
test_preds = model.predict(X_test)

# Evaluate the model
train_acc = accuracy_score(Y_train, train_preds)
test_acc = accuracy_score(Y_test, test_preds)

print(f"\n{'='*50}")
print(f"Model: {model}")
print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy : {test_acc:.4f}")
print("\nClassification Report (Test Data):")
print(classification_report(Y_test, test_preds))

# Generates the Confusion matrix
cm = confusion_matrix(Y_test, test_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=model.classes_, yticklabels=model.classes_)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show()

# Predict on actual test set
final_predictions = model.predict(X_scaled_test)

# Create and save result CSV file
output = pd.DataFrame({'PassengerId': test_passenger_ids, 'Survived': final_predictions})
output.to_csv('result_new.csv', index=False)
print(f"Submission file created: result_new.csv") 