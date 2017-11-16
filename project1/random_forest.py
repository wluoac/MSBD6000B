import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# load training features and labels from files
features = np.array(pd.read_csv("./traindata.csv"))
label = np.ravel((pd.read_csv("./trainlabel.csv")))

# normalization: convert to mean 0 and scaled by std
mean_train = np.mean(features, axis=0)
std_train = np.std(features, axis=0)

features -= np.mean(features, axis=0)
features /= np.std(features, axis=0)

clf = RandomForestClassifier(n_estimators=21, max_depth=None,min_samples_split=2,random_state=0)
scores = cross_val_score(clf, features, label,cv=10)
clf.fit(features, label)

print(scores)

test_features = np.array(pd.read_csv("./testdata.csv"))
test_features -= mean_train
test_features /= std_train
test_predict = clf.predict(test_features).astype(np.int64)

#save the predict as a csv file
np.savetxt("test_predict_rf.csv", test_predict, delimiter=",",fmt='%i')