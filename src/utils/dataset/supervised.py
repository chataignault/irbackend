import polars as pl
import numpy as np
from typing import Tuple


def get_multidim_ts_scaled_lookback_flat(
    df: pl.LazyFrame, target_name: str, scaling_factor: float = 1.0, lookback: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    diffs = (
        df.with_columns(pl.exclude("time").diff())
        .fill_nan(0.0)
        .fill_null(0.0)
        .drop_nans()
        .drop_nulls()
    )
    diffs_np = diffs.collect().to_pandas().set_index("time").values

    X = (
        np.array(
            [
                diffs_np[i : i + lookback].flatten()
                for i in range((len(diffs_np) - lookback))
            ]
        )
        * scaling_factor
    )
    Y = diffs.select(pl.col(target_name)).collect().to_pandas().values[lookback:]

    Y = scaling_factor * (Y - np.mean(Y))
    Y = Y.T[0]

    return X, Y
