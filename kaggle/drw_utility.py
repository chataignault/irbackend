import numpy as np
import pandas as pd
import polars as pl
from typing import List


def get_clean_crypto_data(train: bool = True) -> pl.LazyFrame:
    """
    Load and clean crypto data, returning either train or test set.

    Args:
        train: If True, return training set. If False, return test set.

    Returns:
        Cleaned lazy frame with columns that have variance and no infinite values.
    """
    global KAGGLE
    global crypto_folder
    
    filename = "train.parquet" if train else "test.parquet"

    # load data
    crypto_lazy = pl.scan_parquet(crypto_folder / filename)
    n_cols = len(crypto_lazy.collect_schema().names())

    if train and KAGGLE:
        # rename timestamp column
        crypto_lazy = crypto_lazy.with_columns(
            pl.col("__index_level_0__").alias("timestamp")
        ).drop(["__index_level_0__"])

    # Remove columns with zero variance in the training set
    train_lazy = pl.scan_parquet(crypto_folder / "train.parquet")
    if KAGGLE:
        train_lazy = train_lazy.with_columns(
            pl.col("__index_level_0__").alias("timestamp")
        ).drop(["__index_level_0__"])

    # Get column names and calculate variance on training set (for consistency)
    crypto_var = train_lazy.select(pl.exclude(["timestamp"]).var())

    crypto_var_cols = (
        crypto_var.select(pl.all() == 0.0)
        .first()
        .collect()
        .to_pandas()
        .T.rename(columns={0: "is_variance_null"})
        .reset_index()
        .rename(columns={"index": "column_name"})
        .groupby("is_variance_null")["column_name"]
        .unique()
    )

    crypto_cols_with_var = crypto_var_cols[False]

    try:
        cols_no_var = crypto_var_cols[True]
        print(f"Columns with no variance : {cols_no_var}")
    except KeyError:
        print("All columns have variance in the train set")

    # remove columns that have no variance in the training set
    train_lazy = train_lazy.select(
        ["timestamp"] + [pl.col(c) for c in crypto_cols_with_var]
    )

    # Remove columns with infinite values (check on training set)
    current_columns = train_lazy.collect_schema().names()
    contains_infinite_cols = (
        train_lazy.select(pl.exclude("timestamp").abs().max().is_infinite())
        .collect()
        .to_pandas()
        .T.rename(columns={0: "contains_infinite"})
        .reset_index()
        .rename(columns={"index": "column_name"})
        .groupby("contains_infinite")["column_name"]
        .unique()
    )

    try:
        cols_with_inf_vals = contains_infinite_cols[True]
        print(f"Columns with infinite values : {cols_with_inf_vals}")
    except KeyError:
        print("No columns with infinite values")

    if not train:
        # add dummy timestamps
        crypto_lazy = crypto_lazy.with_columns(
            ID=range(1, crypto_lazy.select(pl.len()).collect().item() + 1)
        )
    # Filter clean columns based on what's available in the current dataset
    clean_columns = [
        c for c in current_columns if c in contains_infinite_cols[False]
    ] + ["timestamp", "ID"]
    available_columns = crypto_lazy.collect_schema().names()
    final_columns = [c for c in clean_columns if c in available_columns]
    print(f"Eventually {len(final_columns)}, removed {n_cols - len(final_columns)}")

    return crypto_lazy.select(final_columns)


def get_diff_features(df: pl.LazyFrame, stats_columns: List[str]):
    return (
        df.with_columns(pl.exclude(stats_columns).diff())
        .with_row_index()
        .fill_null(strategy="backward")
        .select(pl.exclude("index"))
    )


