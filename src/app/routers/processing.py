import pandas as pd
import polars as pl


def process_ticker_hist(df: pd.DataFrame) -> pd.DataFrame:
    df["Mid"] = (df["High"] - df["Low"]) / 2.0
    return df[["Mid"]]


MAX_DAILY_CHANGE = 2.0


def filter_jumps(df: pl.LazyFrame) -> pl.LazyFrame:
    dfs = df.select(pl.exclude("time").backward_fill()).head(1).collect().to_pandas()
    df = (
        df.with_columns(
            pl.exclude("time")
            .diff()
            .clip(-MAX_DAILY_CHANGE, MAX_DAILY_CHANGE)
            .replace(-MAX_DAILY_CHANGE, 0.0)
            .replace(MAX_DAILY_CHANGE, 0.0)
            .cum_sum()
        )
        .collect()
        .to_pandas()
        .set_index("time")
        + dfs.values
    )
    return pl.from_dataframe(df.reset_index()).lazy()
