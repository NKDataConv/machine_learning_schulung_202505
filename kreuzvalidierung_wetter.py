from datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
# from machine_learning_schulung_202505.datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import numpy as np

###### Decision Tree Classifier #####
PARAMS = {"max_depth": 6}
cls = DecisionTreeClassifier(**PARAMS)

cv = cross_validate(cls, x_train, y_train, cv=5, scoring="recall")
print(np.mean(cv["test_score"]))
print(cv)
# leave one out
# from sklearn.model_selection import LeaveOneOut
# loo = LeaveOneOut()
# cv = cross_validate(cls, x_train, y_train, cv=loo, scoring="accuracy")
# print(cv)