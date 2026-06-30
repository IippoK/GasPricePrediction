import pandas as pd
import json

# Load JSON from file
with open("gas_price_data.json", "r", encoding="utf-8") as f:
    raw_json = json.load(f)

# Flatten the "data" list
gas_data = raw_json["data"]

# Load into DataFrame
df_gas = pd.DataFrame(gas_data)

# Parse date: YYYYMMDD → datetime
df_gas["date"] = pd.to_datetime(df_gas["date"], format="%Y%m%d")

# Convert price columns to float
df_gas["euro95"] = pd.to_numeric(df_gas["euro95"], errors="coerce")
df_gas["diesel"] = pd.to_numeric(df_gas["diesel"], errors="coerce")

# Drop duplicate rows
df_gas = df_gas.drop_duplicates(subset=["date", "euro95", "diesel"])

# Drop rows with missing prices (if any)
df_gas = df_gas.dropna(subset=["euro95"])

# Keep only useful columns
df_gas = df_gas[["date", "euro95", "diesel"]]

# Round date to start of ISO week (Monday)
df_gas["date"] = df_gas["date"].dt.to_period('W').apply(lambda r: r.start_time)

# Optional: average duplicate weeks (if some days fall into the same week)
df_gas = df_gas.groupby("date", as_index=False).mean()

# Save cleaned gas prices
df_gas.to_csv("cleaned_gas_prices_weekly.csv", index=False)
