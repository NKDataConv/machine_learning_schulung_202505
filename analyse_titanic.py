import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


pd.set_option("display.max_columns", None)
df = pd.read_csv("daten/titanic.csv")

# print(df.head())
# print(df.columns)
# for col in df.columns:
#     print("*" * 50)
#     print(col)
#     print(df[col].value_counts())

df["Cabin"] = df["Cabin"].map(lambda x: str(x)[0] if pd.notna(x) else "N")

df = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch", "Cabin", "Fare"]]
df = df.dropna()

le = LabelEncoder()
le.fit(df["Sex"])
df["Sex"] = le.transform(df["Sex"])

df = pd.get_dummies(df)

# le = LabelEncoder()
# le.fit(df["Cabin"])
# df["Cabin"] = le.transform(df["Cabin"])

y = df["Survived"]
x = df.drop(columns=["Survived"])

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=42, stratify=y)
x_vali, x_test, y_vali, y_test = train_test_split(x_test, y_test, test_size=0.25, random_state=42, stratify=y_test)
