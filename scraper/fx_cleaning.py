import pandas as pd

# Load CSV without header, then assign column names
df_fx = pd.read_csv("usd_eur_rates.csv")

df_fx = df_fx.rename(columns={"DATE": "date"})

# Parse 'date' column to datetime
df_fx["DATE"] = pd.to_datetime(df_fx["DATE"])

# Normalize to start of the week (Monday)
df_fx["DATE"] = df_fx["DATE"].dt.to_period("W").apply(lambda r: r.start_time)

# Rename for clarity
df_fx = df_fx.rename(columns={"US dollar/Euro (EXR.D.USD.EUR.SP00.A) - Modified value (Weekly)": "usd_per_eur"})

df_fx = df_fx.rename(columns={"US dollar/Euro (EXR.D.USD.EUR.SP00.A) - Modified value (Weekly)": "usd_per_eur"})
# Drop duplicates just in case
df_fx = df_fx.drop_duplicates(subset="date")

# Save cleaned version
df_fx.to_csv("cleaned_usd_eur_weekly.csv", index=False)
