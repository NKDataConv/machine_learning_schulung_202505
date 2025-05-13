from datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
# from machine_learning_schulung_202505.datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier

###### Classifier #####
PARAMS = {"n_estimators": 100,
          "learning_rate": 0.15,
          "max_depth": 3}

cls = GradientBoostingClassifier(**PARAMS)
cls.fit(x_train, y_train)


###### Prediction on Train Set #####
y_train_pred = cls.predict(x_train)

df = pd.DataFrame({'actual': y_train,
                'prediction': y_train_pred})

accuracy_train = accuracy_score(df["actual"], df["prediction"])
precision_train = precision_score(df["actual"], df["prediction"])
recall_train = recall_score(df["actual"], df["prediction"])
print("Accuracy auf Train: ", accuracy_train)
print("Precision auf Train: ", precision_train)
print("Recall auf Train: ", recall_train)


##### Prediction on Validation Set #####
y_vali_pred = cls.predict(x_vali)

df_vali = pd.DataFrame({'actual': y_vali,
                'prediction': y_vali_pred})

accuracy_vali = accuracy_score(df_vali["actual"], df_vali["prediction"])
precision_vali = precision_score(df_vali["actual"], df_vali["prediction"])
recall_vali = recall_score(df_vali["actual"], df_vali["prediction"])
print("Accuracy auf Vali: ", accuracy_vali)
print("Precision auf Vali: ", precision_vali)
print("Recall auf Vali: ", recall_vali)

overfitting = accuracy_train - accuracy_vali
overfitting_recall = recall_train - recall_vali
print("Overfitting Accuracy: ", overfitting)
print("Overfitting Recall: ", overfitting_recall)


###### Feature Importances ######
# df_feature_importance = pd.DataFrame({"score": cls.feature_importances_,
#                                       "feature": x_train.columns})
#
# df_feature_importance.sort_values(by="score", ascending=False)
# print(df_feature_importance)