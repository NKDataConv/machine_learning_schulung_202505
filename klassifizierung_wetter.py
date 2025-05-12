from datenaufbereitung_wetter import x_train, x_vali, y_train, y_vali
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

cls = DecisionTreeClassifier(max_depth=4)
cls.fit(x_train, y_train)

y_train_pred = cls.predict(x_train)
print(y_train_pred)

df = pd.DataFrame({'actual': y_train,
                'prediction': y_train_pred})

df["correct"] = df["actual"] == df["prediction"]
anzahl_richtig_train = df["correct"].sum()
print("Auf Trainingsdaten wurden ", anzahl_richtig_train, " von ", len(df), " richtig klassifiziert.")
accuracy_train = anzahl_richtig_train / len(df)

y_vali_pred = cls.predict(x_vali)

df_vali = pd.DataFrame({'actual': y_vali,
                'prediction': y_vali_pred})

df_vali["correct"] = df_vali["actual"] == df_vali["prediction"]
anzahl_richtig_vali = df_vali["correct"].sum()
print("Auf Trainingsdaten wurden ", anzahl_richtig_vali, " von ", len(df_vali), " richtig klassifiziert.")
accuracy_vali = anzahl_richtig_vali / len(df_vali)

overfitting = accuracy_train - accuracy_vali
print("Overfitting: ", overfitting)

