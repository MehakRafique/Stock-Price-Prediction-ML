# Import required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc

# -----------------------------
# 1. Load Dataset
# -----------------------------
# Replace 'heart.csv' with the path to your dataset
data = pd.read_csv("heart.csv")

# Display first 5 rows
print("First 5 rows of the dataset:")
print(data.head())

# -----------------------------
# 2. Data Cleaning
# -----------------------------
# Check for missing values
print("\nMissing values in each column:")
print(data.isnull().sum())

# If missing values exist, we can fill them with median
data.fillna(data.median(), inplace=True)

# -----------------------------
# 3. Exploratory Data Analysis (EDA)
# -----------------------------
# Basic statistics
print("\nDataset statistics:")
print(data.describe())

# Visualize distribution of target variable
sns.countplot(x='target', data=data)
plt.title('Heart Disease Distribution (0 = No, 1 = Yes)')
plt.show()

# Correlation heatmap
plt.figure(figsize=(10,8))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# -----------------------------
# 4. Feature Selection and Splitting Data
# -----------------------------
X = data.drop('target', axis=1)  # Features
y = data['target']               # Target variable

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# -----------------------------
# 5. Train Models
# -----------------------------

# 5.1 Logistic Regression
log_model = LogisticRegression()
log_model.fit(X_train_scaled, y_train)

# 5.2 Decision Tree Classifier
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X_train, y_train)

# -----------------------------
# 6. Evaluate Models
# -----------------------------

def evaluate_model(model, X_test, y_test, scaled=False):
    if scaled:
        y_pred = model.predict(X_test_scaled)
    else:
        y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # ROC Curve
    if scaled:
        y_prob = model.predict_proba(X_test_scaled)[:,1]
    else:
        y_prob = model.predict_proba(X_test)[:,1]
        
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    
    print(f"Accuracy: {acc:.4f}")
    print("Confusion Matrix:")
    print(cm)
    
    plt.figure()
    plt.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='red', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

# Evaluate Logistic Regression
print("\n--- Logistic Regression Evaluation ---")
evaluate_model(log_model, X_test, y_test, scaled=True)

# Evaluate Decision Tree
print("\n--- Decision Tree Evaluation ---")
evaluate_model(tree_model, X_test, y_test, scaled=False)

# -----------------------------
# 7. Feature Importance (Decision Tree)
# -----------------------------
importance = pd.Series(tree_model.feature_importances_, index=X.columns)
importance.sort_values(ascending=False, inplace=True)
print("\nFeature Importance (Decision Tree):")
print(importance)

# Visualize Feature Importance
plt.figure(figsize=(10,6))
sns.barplot(x=importance, y=importance.index)
plt.title('Feature Importance in Heart Disease Prediction')
plt.show()