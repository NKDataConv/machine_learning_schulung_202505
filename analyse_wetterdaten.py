import pandas as pd

pd.set_option("display.max_columns", None)

df = pd.read_csv("daten/weatherAUS.csv")
df = df.dropna()
print(df.head())
print(df.info())
df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

print(df.columns)

# Frage 1: Wo und wann max Temperatur
# Option 1:
df = df.set_index("Date")
max_temp = df["MaxTemp"].max()
mask_max_temp = df["MaxTemp"] == max_temp
print(df.loc[mask_max_temp, "Location"])

# Option 2:
df = df.set_index(["Date", "Location"])
max_temp = df["MaxTemp"].idxmax()
print(df.loc[max_temp])

# Frage 2: Stadt mit höchsten Mittel
df_grouped = df.groupby("Location").agg({"MaxTemp": "mean"}).reset_index()
df_grouped.sort_values(by="MaxTemp")

df_grouped["MaxTemp"].max()

# Frage 3: Schwankungen am niedrigsten
# Option 1:
df_grouped = df.groupby("Location").agg({"MaxTemp": "std"})
df_grouped.idxmin()

# Option 2:
df["tages_diff"] = df["MaxTemp"] - df["MinTemp"]
df_grouped = df.groupby("Location").agg({"tages_diff": "mean"})
df_grouped.idxmin()

# Option 3:
df_grouped = df.groupby("Location").agg({"MinTemp": "min", "MaxTemp": "max"})
df_grouped["diff"] = df_grouped["MaxTemp"] - df_grouped["MinTemp"]
df_grouped.idxmin()
df_grouped.sort_values("diff")

# Frage 4: kälteste Monat, meiste Regen
df["monat"] = df.index.month
df_grouped = df.groupby("monat").agg({"MinTemp": "min"})
kaelteste_monat = df_grouped.idxmin()
print("Der käteste Monat ist Monat ", kaelteste_monat.values[0])

# meiste Regen
df["monat"] = df.index.month
df_grouped = df.groupby("monat").agg({"Rainfall": "sum", "Location": "count"})
df_grouped = df_grouped.rename(columns={"Location": "anzahl_tage"})

df_grouped["rainfall_normalisiert"] = df_grouped["Rainfall"] / df_grouped["anzahl_tage"]
df_grouped.sort_values(by="rainfall_normalisiert")

df_grouped = df.groupby("monat").agg({"Rainfall": "mean"})
print(df_grouped)

# Frage 5: Erhöhung der Temperatur
df["year"] = df.index.year
df_grouped = df.groupby("year").agg({"MaxTemp": ["mean", "max"], "MinTemp": "mean"})
print(df_grouped)

df.groupby("year").agg({"Location": "count"})
