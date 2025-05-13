from datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
# from machine_learning_schulung_202505.datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.neighbors import KNeighborsClassifier

###### Classifier #####
cls = KNeighborsClassifier()
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
