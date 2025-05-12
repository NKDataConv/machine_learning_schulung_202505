import pandas as pd

df = pd.read_csv("machine_learning_schulung_202505/daten/BTC_daily.csv")

print(df.head())

df["Date"] = pd.to_datetime(df["Date"])
df = df.set_index("Date")

print(df.head())

# Werte Selektieren mit Index
df.loc["2024-09-17"]
df.loc["2024-09-17", "Open"]

# Werte Selektieren mit Position
df.iloc[0, 0]
df.iloc[1, 1]

# Werte Selektieren mit Attributen
df.Open

df["year"] = df.index.year
print(df.head())

df_grouped = df.groupby("year").agg({"Open": "mean"})
print(df_grouped)


# Was ist der neuste Wert?
max_index = df.index.max()
letzte_wert = df.loc[max_index, "Close"]
print("Der letzte Wert war", letzte_wert)

# Was ist der Durchschnitt?
df.columns
open_durchschnitt = df["Open"].mean()

durchschnitte = df[['Open', 'High', 'Low', 'Close']].mean()
print(durchschnitte)

# Wann wurde das meiste Volumen gehandelt?
# Option 1:
max_volume = df["Volume"].max()
mask = df.Volume == max_volume
max_volume_tag = df[mask].index

# Option 2:
max_volume_tag = df["Volume"].idxmax()
print(max_volume_tag)

# Was war der höchste Wert des Bitcoin?
max_high_tag = df["High"].idxmax()
print("Der höchste Wert war am", max_high_tag)

# Wieviel Geld hätte man, wenn man am Anfang 1000€ investiert hätte?
erste_tag = df.index.min()
wert_am_ersten_tag = df.loc[erste_tag, "Open"]
print(wert_am_ersten_tag)

anzahl_bitcoin = 1000 / wert_am_ersten_tag

letzte_tag = df.index.max()
kurs_am_letzten_tag = df.loc[letzte_tag, "Close"]
wert = anzahl_bitcoin * kurs_am_letzten_tag

print("Die Bitcoins wären ", wert, "€ wert")

# Wieviel Geld hätte man, wenn man jeden Tag 1€ in Bitcoin investiert hätte?
df["investment"] = 1
df["invest_in_bitcoin"] = df["investment"] / df["Open"]

gesamt_invest_in_bitcoin = df["invest_in_bitcoin"].sum()
gesamt_wert = gesamt_invest_in_bitcoin * kurs_am_letzten_tag
print("Die täglichen Investitionen wären ", gesamt_wert, "€ wert")
