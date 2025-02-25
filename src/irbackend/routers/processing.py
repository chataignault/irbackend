import pandas as pd


def process_ticker_hist(df: pd.DataFrame) -> pd.DataFrame:
    df["Mid"] = (df["High"] - df["Low"]) / 2.0
    return df[["Mid"]]
