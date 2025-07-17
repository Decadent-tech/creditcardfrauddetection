import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import gridspec

data = pd.read_csv("dataset/creditcard.csv")
print(data.head())
print(data.describe())

fraud = data[data['Class'] == 1]
print(fraud.describe())
print(f"Total number of fraud cases: {len(fraud)}")


valid = data[data['Class'] == 0]
print(f"Total number of valid cases: {len(valid)}") 
print(f"Total number of cases: {len(data)}")


print(f"Total number of non-fraud cases: {len(data) - len(fraud)}")
print(f"Percentage of fraud cases: {len(fraud) / len(data) * 100:.2f}%")
#univariate and bivariate analysis
# Visualizing the distribution of the 'Class' feature
#clearly the dataset is highly imbalanced
#let's visualize the distribution of the classes
plt.figure(figsize=(10, 5))
sns.countplot(x='Class', data=data)
plt.title('Distribution of Classes')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks([0, 1], ['Valid', 'Fraud'])
#plt.show()
#plt.savefig('class_distribution.png')
# Visualizing the distribution of the 'Amount' feature
plt.figure(figsize=(10, 5))
sns.histplot(data['Amount'], bins=50, kde=True)
plt.title('Distribution of Transaction Amounts')
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.xlim(0, 2000)  # Limit x-axis for better visibility
#plt.show()
#plt.savefig('amount_distribution.png')
# Visualizing the correlation matrix
plt.figure(figsize=(12, 10))
corr = data.corr()
sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
plt.title('Correlation Matrix')
#plt.show()
#plt.savefig('correlation_matrix.png')

#class and amount distribution
plt.figure(figsize=(12, 6))
gs = gridspec.GridSpec(1, 2, width_ratios=[1, 2])
ax0 = plt.subplot(gs[0])
sns.countplot(x='Class', data=data, ax=ax0)
ax0.set_title('Class Distribution')
ax0.set_xlabel('Class')
ax0.set_ylabel('Count')
ax1 = plt.subplot(gs[1])
sns.boxplot(x='Class', y='Amount', data=data, ax=ax1)
ax1.set_title('Transaction Amount by Class')
ax1.set_xlabel('Class')
ax1.set_ylabel('Amount')
plt.tight_layout()
#plt.show()
#plt.savefig('class_amount_distribution.png')
# Visualizing the time feature
plt.figure(figsize=(12, 6)) 
sns.histplot(data['Time'], bins=50, kde=True)
plt.title('Distribution of Transaction Times')
plt.xlabel('Time (seconds since start of recording)')
plt.ylabel('Frequency')
plt.xlim(0, 172800)  # Limit x-axis for better visibility
#plt.show()
#plt.savefig('time_distribution.png')
# Visualizing the relationship between Time and Amount
plt.figure(figsize=(12, 6))
#sns.scatterplot(x='Time', y='Amount', hue='Class', data=data, alpha=0.5)
sns.kdeplot(data=data[data['Class'] == 1], x='Time', y='Amount', cmap="Reds", fill=True)
plt.title('Transaction Amount vs Time')
plt.xlabel('Time (seconds since start of recording)')
plt.ylabel('Amount')
plt.xlim(0, 172800)  # Limit x-axis for better visibility
plt.ylim(0, 2000)  # Limit y-axis for better visibility
plt.legend(title='Class', loc='upper right')  # Remove or comment out, no legend for kdeplot
# plt.show()  # Comment out to avoid interruption
#plt.savefig('time_amount_relationship.png')
#Time does affect transaction volume, but not dramatically — there are peaks and dips in activity.
#Most fraudulent or genuine transactions occur within the ₹0–₹500 range, which may limit Amount as a strong classifier on its own (it needs to be combined with other features).
#this plot is helpful for understanding volume concentration, but it does not distinguish between fraud and non-fraud directly.


#The strongest correlations (either positive or negative) with Class are:
#V17, V14, V10, V12, and V4 — these show slight negative correlations (darker blue).
#V7, V11, V2 — these may show some mild positive correlation (slight reddish shades).
#This indicates these variables may be useful predictors for the classification task (likely fraud detection, as this looks like the Credit Card Fraud dataset).
# Visualizing the distribution of V17, V14, V10, V12, and V4
features_of_interest = ['V17', 'V14', 'V10', 'V12', 'V4']
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_of_interest):
    plt.subplot(3, 2, i + 1)
    sns.histplot(data[feature], bins=50, kde=True)
    plt.title(f'Distribution of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Frequency') 

plt.tight_layout()
#plt.show()
#plt.savefig('features_of_interest_distribution.png')
# Visualizing the relationship between V17, V14, V10, V12, and V4 with Class
plt.figure(figsize=(15, 10))    
for i, feature in enumerate(features_of_interest):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(x='Class', y=feature, data=data)
    plt.title(f'{feature} by Class')
    plt.xlabel('Class')
    plt.ylabel(feature)
plt.tight_layout()
#plt.show()
#plt.savefig('features_of_interest_by_class.png')


#Transaction Amount vs Time by Class
#Class 0 (Non-Fraudulent Transactions):
    #Has numerous extreme outliers, with transaction amounts reaching as high as ₹25,000+.
    #The distribution is heavily right-skewed, meaning most transactions are of small amounts, but a few are very large.
    #This wide range and the presence of many outliers indicate high variance in legitimate transaction amounts.
#Class 1 (Fraudulent Transactions):
    #Amounts are much lower in general — very few go beyond ₹2,000–₹3,000.
    #There are fewer extreme values, and the spread is much tighter.
    #This implies frauds in this dataset tend to involve lower-value transactions.

#distribution plots of features V4, V10, V12, V14, and V17, which (as seen in the correlation heatmap) appear to have some relationship with the target Class

# Visualizing the distribution of V4, V10, V12, V14, and V17 by Class
plt.figure(figsize=(15, 10))
for i, feature in enumerate(features_of_interest):
    plt.subplot(3, 2, i + 1)
    sns.boxplot(x='Class', y=feature, data=data)
    plt.title(f'{feature} by Class')
    plt.xlabel('Class')
    plt.ylabel(feature)
plt.tight_layout()
#plt.show()
#plt.savefig('features_of_interest_by_class.png')


print("Amount details of the fraudulent transaction")
print(fraud.Amount.describe())

print("details of valid transaction")
print(valid.Amount.describe())

#Preparing Data
#Separate the input features (X) and target variable (Y) then split the data into training and testing sets

X = data.drop(['Class'], axis=1)
Y = data['Class']
print(f"Shape of X: {X.shape}")
print(f"Shape of Y: {Y.shape}")

xData = X.values
yData = Y.values
from sklearn.model_selection import train_test_split
#smote is used to handle class imbalance
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
xData, yData = smote.fit_resample(xData, yData)
print(f"Shape of xData after SMOTE: {xData.shape}")
print(f"Shape of yData after SMOTE: {yData.shape}")
# Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(xData, yData, test_size=0.2, random_state=42, stratify=yData)
print(f"Shape of X_train: {X_train.shape}\nShape of X_test: {X_test.shape}")
print(f"Shape of y_train: {y_train.shape}\nShape of y_test: {y_test.shape}")

#training the model on logistic regression
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train, y_train)
# Evaluating the model
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of the Logistic Regression model: {accuracy * 100:.2f}%")
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'])
plt.title('Confusion Matrix for Logistic Regression')
plt.show()
plt.savefig('confusion_matrix_logistic_regression.png')

#Key Issues Uncovered:
    #1,646 fraudulent transactions misclassified as valid:
        #These are false negatives (FN) — the most dangerous errors in fraud detection.
        #You’re letting actual fraud go undetected — unacceptable in real-world systems.
    #Accuracy ≈ 98% is misleading:
        #That's because it includes all the true negatives (the bulk of the data), which masks the 1,646 fraud misses.
    #Precision and recall need context:
        #If heavily penalizing false negatives (which you should in fraud detection),
        #you may want to optimize recall or F2 score over accuracy.
    
#changing the model to Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
# Evaluating the Random Forest model
y_pred_rf = rf_model.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy of the Random Forest model: {accuracy_rf * 100:.2f}%")
print("Classification Report for Random Forest:")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix for Random Forest:")
print(confusion_matrix(y_test, y_pred_rf))
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues', xticklabels=['Valid', 'Fraud'], yticklabels=['Valid', 'Fraud'])
plt.title('Confusion Matrix for Random Forest')
plt.show()
plt.savefig('confusion_matrix_random_forest.png')


#Handles non-linear relationships and interactions between features better than Logistic Regression.
#Is robust to outliers and feature scaling.
#With enough trees and depth, it captures even subtle fraud patterns.
# Exporting the RandomForestClassifier model using joblib
import joblib
joblib.dump(rf_model, 'model/rf_fraud_model.pkl')