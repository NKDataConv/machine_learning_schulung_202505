from datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
# from machine_learning_schulung_202505.datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

PARAMS = {"max_depth": 4,
          "min_samples_split": 400,
          "min_samples_leaf": 200}

cls = DecisionTreeClassifier(**PARAMS)
cls.fit(x_train, y_train)

y_train_pred = cls.predict(x_train)

df = pd.DataFrame({'actual': y_train,
                'prediction': y_train_pred})

df["correct"] = df["actual"] == df["prediction"]
anzahl_richtig_train = df["correct"].sum()
print("Auf Trainingsdaten wurden ", anzahl_richtig_train, " von ", len(df), " richtig klassifiziert.")
accuracy_train = anzahl_richtig_train / len(df)
print("Accuracy auf Train: ", accuracy_train)

y_vali_pred = cls.predict(x_vali)

df_vali = pd.DataFrame({'actual': y_vali,
                'prediction': y_vali_pred})

df_vali["correct"] = df_vali["actual"] == df_vali["prediction"]
anzahl_richtig_vali = df_vali["correct"].sum()
print("Auf Trainingsdaten wurden ", anzahl_richtig_vali, " von ", len(df_vali), " richtig klassifiziert.")
accuracy_vali = anzahl_richtig_vali / len(df_vali)
print("Accuracy auf Vali: ", accuracy_vali)

overfitting = accuracy_train - accuracy_vali
print("Overfitting: ", overfitting)

df_feature_importance = pd.DataFrame({"score": cls.feature_importances_,
                                      "feature": x_train.columns})

df_feature_importance.sort_values(by="score", ascending=False)

from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
plot_tree(cls,
          feature_names=x_train.columns,
          filled=True,
          fontsize=5)
plt.show(block=True)
