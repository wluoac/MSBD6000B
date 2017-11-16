# Use an ensemble classifier (random forest) for prediction
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score

# Assume files are in the same directory
# Load training features and labels from the csv files
features = np.array(pd.read_csv("./traindata.csv"))
label = np.ravel((pd.read_csv("./trainlabel.csv")))

# Normalization: convert to mean 0 and scaled by std
mean_train = np.mean(features, axis=0)
std_train = np.std(features, axis=0)

features -= mean_train
features /= std_train

# The model is a random forest classifier
clf = RandomForestClassifier(n_estimators=30, max_depth=None,min_samples_split=2,random_state=0)

# Try cross validation with K=10
scores = cross_val_score(clf, features, label,cv=10)

# Print the cross validation scores
print(scores)

# Train the model with all training data
clf.fit(features, label)

# Use the obtained model to predict the labels of testing data
test_features = np.array(pd.read_csv("./testdata.csv"))
test_features -= mean_train
test_features /= std_train
test_predict = clf.predict(test_features).astype(np.int64)

# Save the prediction as a csv file
np.savetxt("test_predict_rf.csv", test_predict, delimiter=",",fmt='%i')