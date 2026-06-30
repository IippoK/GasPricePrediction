import pandas as pd

# Load Brent oil CSV with correct date format
df_oil = pd.read_csv("brent_historical_data.csv")

# Parse the "Date" column, specify format MM/DD/YYYY
df_oil["date"] = pd.to_datetime(df_oil["Date"], format="%m/%d/%Y")

# Keep only needed columns (date and closing price)
df_oil = df_oil[["date", "Price"]].rename(columns={"Price": "brent_usd"})

# Sort by date ascending
df_oil = df_oil.sort_values("date").reset_index(drop=True)

# Optional: round dates to start of the week (Monday)
df_oil["date"] = df_oil["date"].dt.to_period("W").apply(lambda r: r.start_time)

df_oil.to_csv("cleaned_brent_oil_data.csv", index=False)
