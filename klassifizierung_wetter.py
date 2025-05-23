from datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
# from machine_learning_schulung_202505.datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score

###### Decision Tree Classifier #####
PARAMS = {"max_depth": 6}
cls = DecisionTreeClassifier(**PARAMS)
cls.fit(x_train, y_train)


###### Prediction on Train Set #####
# y_train_pred = cls.predict(x_train)
y_train_pred_proba = cls.predict_proba(x_train)

cutoff = 0.18
y_train_pred = [1 if y[1] > cutoff else 0 for y in y_train_pred_proba]

df = pd.DataFrame({'actual': y_train,
                'prediction': y_train_pred})

accuracy_train = accuracy_score(df["actual"], df["prediction"])
precision_train = precision_score(df["actual"], df["prediction"])
recall_train = recall_score(df["actual"], df["prediction"])
print("Accuracy auf Train: ", accuracy_train)
print("Precision auf Train: ", precision_train)
print("Recall auf Train: ", recall_train)


##### Prediction on Validation Set #####
# y_vali_pred = cls.predict(x_vali)
y_vali_pred_proba = cls.predict_proba(x_vali)
y_vali_pred = [1 if y[1] > cutoff else 0 for y in y_vali_pred_proba]

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
df_feature_importance = pd.DataFrame({"score": cls.feature_importances_,
                                      "feature": x_train.columns})

df_feature_importance.sort_values(by="score", ascending=False)


###### Plot Decision Tree ######
# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# plot_tree(cls,
#           feature_names=x_train.columns,
#           filled=True,
#           fontsize=15)
# plt.show(block=True)
