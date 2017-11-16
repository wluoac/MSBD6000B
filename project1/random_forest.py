# Use an ensemble classifier (random forest) for prediction
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix,roc_auc_score

# Assume files are in the same directory
# Load training features and labels from the csv files
features = np.array(pd.read_csv("./traindata.csv", header=None))
label = np.ravel((pd.read_csv("./trainlabel.csv", header=None)))

# Pre-processing: normalization (convert to mean 0 and scaled by std)
mean_train = np.mean(features, axis=0)
std_train = np.std(features, axis=0)
features -= mean_train
features /= std_train

# The model is a random forest classifier
clf = RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_split=2)

# Try cross validation with K=5
scores = cross_val_score(clf, features, label, cv=5)
print(scores)

# Try split the training data for training and testing parts
cv = 0
while cv <= 5:
    X_train, X_test, y_train, y_test = train_test_split(features, label, test_size=0.4, random_state=1)
    clf.fit(X_train, y_train)
    y_true, y_pred = label, clf.predict(features)
    print(classification_report(y_true, y_pred))
    print(confusion_matrix(y_true, y_pred))
    print ("AUC score " , roc_auc_score(y_true, y_pred))
    cv += 1

# Train the model with all training data
clf.fit(features, label)

# Use the obtained model to predict the labels of testing data
test_features = np.array(pd.read_csv("./testdata.csv",header=None))
test_features -= mean_train
test_features /= std_train
test_predict = clf.predict(test_features)

# Save the prediction as a csv file
np.savetxt("project1_08573584.csv", test_predict, delimiter=",",fmt="%i")