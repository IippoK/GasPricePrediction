import pandas as pd
import numpy as np


def get_data():

    # Load the .csv-files
    brent = pd.read_csv("brent_oil_data.csv")
    usd = pd.read_csv("usd_eur_weekly.csv")
    fuel = pd.read_csv("gas_prices_weekly.csv")

    # Consistent date format
    for df in [brent, usd, fuel]:
        df["date"] = pd.to_datetime(df["date"])

    # Drop unnecessary columns
    usd = usd[["date", "rate"]]

    # Merge
    merged = brent.merge(usd, on="date").merge(fuel, on="date")
    merged.columns = ["date", "brent_price", "usd_eur", "e95_price", "diesel_price"]

    # Add time variable
    merged["weeks_since_start"] = range(len(merged))

    # Convert Brent oil price to euros
    merged["brent_eur"] = merged["brent_price"] / merged["usd_eur"]

    # Lag features (1- and 2-week lag)
    merged["brent_eur_lag1"] = merged["brent_eur"].shift(1)
    merged["brent_eur_lag2"] = merged["brent_eur"].shift(2)

    # Seasonal feature with sine and cosine
    merged["weekofyear"] = merged["date"].dt.isocalendar().week.astype(int)
    merged["season_sin"] = np.sin(2 * np.pi * merged["weekofyear"] / 52)
    merged["season_cos"] = np.cos(2 * np.pi * merged["weekofyear"] / 52)

    merged = merged.dropna().reset_index(drop=True)
    merged = merged.sort_values("date").reset_index(drop=True)

    return merged

    # print(merged.info())
    # print(merged.head())
