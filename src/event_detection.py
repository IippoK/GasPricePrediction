import pandas as pd
import numpy as np


def detect_price_surges(df, window=30, z_threshold=3):
    """
    Detects sudden surges or drops in fuel prices based on z-scores.

    :param df:
    :param window:
    :param z_threshold:
    :return:
    """

    df = df.copy()
    df["rolling_mean"] = df["euro95"].rolling(window=window).mean()
    df["rolling_std"] = df["euro95"].rolling(window=window).std()
    df["zscore"] = (df["euro95"] - df["rolling_mean"]) / df["rolling_std"]
    df["is_surge"] = df["zscore"].abs() > z_threshold

    return df


def mark_surge_periods(df, min_gap=5):
    """
    Groups consecutive surge days into 'events'.

    :param df:
    :param min_gap:
    :return:
    """

    df = df.copy()
    df["event_id"] = 0

    event_counter = 0
    in_event = False
    gap_counter = 0

    for i in range(len(df)):
        if df.loc[i, "is_surge"]:
            if not in_event:
                event_counter += 1
                in_event = True
            gap_counter = 0
            df.loc[i, "event_id"] = event_counter
        elif in_event:
            gap_counter += 1
            if gap_counter >= min_gap:
                in_event = False
                gap_counter = 0

    return df


def detect_events(df):
    """
    Pipeline to detect and label fuel price surge events.

    :param df:
    :return:
    """

    df = detect_price_surges(df)
    df = mark_surge_periods(df)

    return df


def summarize_events(df):
    """
    Summarize detected surge events by start/end date and magnitude.
    """
    events = df[df["event_id"] > 0]. groupby("event_id").agg(
        start_date=("date", "min"),
        end_date=("date", "max"),
        duration_days=("date", lambda x: (x.max() - x.min()).days),
        avg_zscore=("zscore", "mean"),
        max_zscore=("zscore", "max"),
        mean_price=("euro95", "mean")).reset_index()

    events = events.sort_values("start_date")
    return events


if __name__ == "__main__":

    path = "../data/gas_prices_weekly.csv"
    df = pd.read_csv(path, parse_dates=["date"])
    df = detect_events(df)

    # Save result
    output_path = "../data/processed/fuel_prices_with_events.csv"
    df.to_csv(output_path, index=False)
    print(f"Saved with event flags: {output_path}")

    events_summary = summarize_events(df)
    print(events_summary)
