import pandas as pd

df = pd.read_csv("daten/weatherAUS.csv")

df = df.drop(columns=["WindGustDir", "WindDir9am", "WindDir3pm", "Location", "Date", "RISK_MM"])

df = df.dropna()

# Option 1:
rain_dict = {"No": 0, "Yes": 1}
df["RainTomorrow"] = df["RainTomorrow"].map(rain_dict)
df["RainToday"] = df["RainToday"].map(rain_dict)

# Option 2:
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(df["RainTomorrow"])
df["RainTomorrow"] = le.transform(df["RainTomorrow"])
df["RainToday"] = le.transform(df["RainToday"])

y = df["RainTomorrow"]
x = df.drop(columns=["RainTomorrow"])

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
x_vali, x_test, y_vali, y_test = train_test_split(x_test, y_test, test_size=0.25, random_state=42, stratify=y_test)
