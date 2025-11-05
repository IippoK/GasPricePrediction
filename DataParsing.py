import pandas as pd

brent = pd.read_csv("brent_oil_data.csv")
usd = pd.read_csv("usd_eur_weekly.csv")
fuel = pd.read_csv("gas_prices_weekly.csv")

for df in [brent, usd, fuel]:
    df["date"] = pd.to_datetime(df["date"])

usd = usd[["date", "rate"]]

merged = brent.merge(usd, on="date").merge(fuel, on="date")
merged.columns = ["date", "brent_price", "usd_eur", "e95_price", "diesel_price"]

merged = merged.sort_values("date").reset_index(drop=True)
merged = merged.dropna().reset_index(drop=True)

print(merged.info())
print(merged.head())
