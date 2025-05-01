# 🚢 Titanic Survival Prediction

This project uses machine learning to predict passenger survival on the Titanic based on various features like age, gender, class, and more. It’s based on the classic Kaggle Titanic dataset.

## 📊 Project Description

The objective is to apply classification algorithms to predict whether a passenger survived or not, using features such as:

- Passenger class (Pclass)
- Sex
- Age
- Fare
- SibSp & Parch (family aboard)
- Embarked location

## 🔍 Dataset

- Source: [Kaggle Titanic Dataset](https://www.kaggle.com/c/titanic/data)
- Files:
  - `train.csv` — Training data with survival labels
  - `test.csv` — Test data for predictions
 
## Approach
1)Load the Data

Used Pandas to load training and test datasets.

2)Preprocess the Data

Filled missing values with median values.

Dropped irrelevant columns (Name, Ticket, etc.).

Converted categorical columns like Sex into numeric form using Label Encoding.

3)Visualize the Data

Plotted correlation heatmaps, survival by gender/class, and age distributions to understand patterns.

4)Split the Data

Separated features and target variable (Survived).

Scaled the data and split it into training and testing sets.

5)Train the Model

Trained a Random Forest Classifier with parameters to avoid overfitting.

6)Evaluate the Model

Evaluated using accuracy, classification report, and confusion matrix.

7)Predict and Export

Made predictions on the test set and saved the results to a CSV file.


## Output
Training Accuracy: 0.8539
Testing Accuracy : 0.8436

Classification Report (Test Data):
              precision    recall  f1-score   support

           0       0.84      0.92      0.88       110
           1       0.85      0.72      0.78        69

    accuracy                           0.84       179
   macro avg       0.84      0.82      0.83       179
weighted avg       0.84      0.84      0.84       179

![Figure_1](https://github.com/user-attachments/assets/d7c77fa6-22c3-4cb9-b1a9-d7ec99f256ef)


## 🛠️ Installation

Clone the repository:

```bash
git clone https://github.com/rushil2304/Titanic.git
cd Titanic





