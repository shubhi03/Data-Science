# ROC
# Import necessary modules
from sklearn import model_selection
import pandas as pd
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

col_names = ['pregnant', 'glucose', 'bp', 'skin', 'insulin', 'bmi', 'pedigree', 'age','label']
data = pd.read_csv('/home/azad/Data_Science_Courses/01_DataScience-20190528T065230Z-001/01_DataScience/Datasets/pima-indians-diabetes.data.csv', header=None, names=col_names)
data.head()
# define X and y
feature_cols = ['pregnant', 'insulin', 'bmi', 'age']
# X is a matrix, hence we use [] to access the features we want in feature_cols
X = data[feature_cols]
# y is a vector, hence we use dot to access 'label'
y = data.label
# split X and y into training and testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
# train a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
# instantiate model
logreg = LogisticRegression()
# fit model
logreg.fit(X_train, y_train)
# make class predictions for the testing set
y_pred_class = logreg.predict(X_test)
#Classification accuracy: percentage of correct predictions
# calculate accuracy
from sklearn import metrics
print(metrics.accuracy_score(y_test, y_pred_class))

from sklearn.metrics import roc_curve
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]
# Generate ROC curve values: fpr, tpr, thresholds
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()




# AUC
# Import necessary modules
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
# Compute predicted probabilities: y_pred_prob
y_pred_prob = logreg.predict_proba(X_test)[:,1]
# Compute and print AUC score
print("AUC: {}".format(roc_auc_score(y_test, y_pred_prob)))
# Compute cross-validated AUC scores: cv_auc
cv_auc = cross_val_score(logreg, X, y, cv=5, scoring='roc_auc')# Print list of AUC scores
print("AUC scores computed using 5-fold cross-validation: {}".format(cv_auc))


y_test.mean()

1 - y_test.mean()
max(y_test.mean(), 1 - y_test.mean())

print('True', y_test.values[0:25])
print('Pred', y_pred_class[0:25])

confusion = metrics.confusion_matrix(y_test, y_pred_class)

TP = confusion[1, 1]
TN = confusion[0, 0]
FP = confusion[0, 1]
FN = confusion[1, 0]

print((TP + TN) / float(TP + TN + FP + FN))

classification_error = (FP + FN) / float(TP + TN + FP + FN)
print(classification_error)

print(1 - metrics.accuracy_score(y_test, y_pred_class))



# Sensitivity: When the actual value is positive, how often is the prediction correct?
sensitivity = TP / float(FN + TP)
print(sensitivity)
print(metrics.recall_score(y_test, y_pred_class))
# Specificity: When the actual value is negative, how often is the prediction correct?
specificity = TN / (TN + FP)
print(specificity)


# False Positive Rate: When the actual value is negative, how often is the prediction incorrect?
false_positive_rate = FP / float(TN + FP)
print(false_positive_rate)
print(1 - specificity)
# Precision: When a positive value is predicted, how often is the prediction correct?
#
#How "precise" is the classifier when predicting positive instances?
precision = TP / float(TP + FP)
print(precision)
print(metrics.precision_score(y_test, y_pred_class))


# print the first 10 predicted responses
# 1D array (vector) of binary values (0, 1)
logreg.predict(X_test)[0:10]
# print the first 10 predicted probabilities of class membership
logreg.predict_proba(X_test)[0:10]
# print the first 10 predicted probabilities for class 1
logreg.predict_proba(X_test)[0:10, 1]
# store the predicted probabilities for class 1
y_pred_prob = logreg.predict_proba(X_test)[:, 1]


plt.rcParams['font.size'] = 12
# histogram of predicted probabilities
# 8 bins
plt.hist(y_pred_prob, bins=8)
# x-axis limit from 0 to 1plt.xlim(0,1)
plt.title('Histogram of predicted probabilities')
plt.xlabel('Predicted probability of diabetes')
plt.ylabel('Frequency')
