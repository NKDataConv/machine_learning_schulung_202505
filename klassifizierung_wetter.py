from datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
# from machine_learning_schulung_202505.datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score


PARAMS = {"max_depth": 3}

cls = DecisionTreeClassifier(**PARAMS)
cls.fit(x_train, y_train)

# y_train_pred = cls.predict(x_train)
y_train_pred = cls.predict_proba(x_train)

cutoff = 0.5
y_train_pred = [1 if y[1] > cutoff else 0 for y in y_train_pred]

df = pd.DataFrame({'actual': y_train,
                'prediction': y_train_pred})

# df["correct"] = df["actual"] == df["prediction"]
# anzahl_richtig_train = df["correct"].sum()
# print("Auf Trainingsdaten wurden ", anzahl_richtig_train, " von ", len(df), " richtig klassifiziert.")
# accuracy_train = anzahl_richtig_train / len(df)
# print("Accuracy auf Train: ", accuracy_train)
accuracy_train = accuracy_score(df["actual"], df["prediction"])
precision_train = precision_score(df["actual"], df["prediction"])
recall_train = recall_score(df["actual"], df["prediction"])
print("Accuracy auf Train: ", accuracy_train)
print("Precision auf Train: ", precision_train)
print("Recall auf Train: ", recall_train)

# y_vali_pred = cls.predict(x_vali)
y_vali_pred = cls.predict_proba(x_vali)
y_vali_pred = [1 if y[1] > cutoff else 0 for y in y_vali_pred]

df_vali = pd.DataFrame({'actual': y_vali,
                'prediction': y_vali_pred})

# df_vali["correct"] = df_vali["actual"] == df_vali["prediction"]
# anzahl_richtig_vali = df_vali["correct"].sum()
# print("Auf Trainingsdaten wurden ", anzahl_richtig_vali, " von ", len(df_vali), " richtig klassifiziert.")
# accuracy_vali = anzahl_richtig_vali / len(df_vali)
# print("Accuracy auf Vali: ", accuracy_vali)
accuracy_vali = accuracy_score(df_vali["actual"], df_vali["prediction"])
precision_vali = precision_score(df_vali["actual"], df_vali["prediction"])
recall_vali = recall_score(df_vali["actual"], df_vali["prediction"])
print("Accuracy auf Vali: ", accuracy_vali)
print("Precision auf Vali: ", precision_vali)
print("Recall auf Vali: ", recall_vali)

overfitting = accuracy_train - accuracy_vali
print("Overfitting: ", overfitting)

df_feature_importance = pd.DataFrame({"score": cls.feature_importances_,
                                      "feature": x_train.columns})

df_feature_importance.sort_values(by="score", ascending=False)

# from sklearn.tree import plot_tree
# import matplotlib.pyplot as plt
# plot_tree(cls,
#           feature_names=x_train.columns,
#           filled=True,
#           fontsize=15)
# plt.show(block=True)
