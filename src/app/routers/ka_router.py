import json
import kaggle
import pandas as pd
from pathlib import Path
from fastapi import APIRouter
import pandas as pd
import polars as pl
from deltalake import write_deltalake

from . import DataFolder
from .ka_helpers import CountryCode
from ...utils import householder_qr

ka_router = APIRouter(prefix="/ka", tags=["Kaggle Hub Datasets"])

raw_data_path = Path(DataFolder.RAW)
database_path = Path(DataFolder.INGESTED)


@ka_router.get("/bonds/download_raw")
async def get_bonds_kaggle_raw():
    """
    Download data from Kaggle dataset (first as CSV files)
    Create Delta Lake table (from CSV)
    """

    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "everget/government-bonds", path=raw_data_path, unzip=True
    )

    prices = pl.read_csv(raw_data_path / "prices.csv")
    yields = pl.read_csv(raw_data_path / "yields.csv")  # lazy ?

    write_deltalake(
        table_or_uri=raw_data_path,
        name="bond_prices_raw",
        data=prices,
        mode="overwrite",
    )
    write_deltalake(
        table_or_uri=raw_data_path,
        name="bond_yields_raw",
        data=yields,
        mode="overwrite",
    )

    return


@ka_router.get("/bonds/yields")
async def get_yields():
    if not (database_path / "yields.csv").exists():
        await get_bonds_kaggle_raw()
    df = pd.read_csv(raw_data_path / "yields.csv")
    return df.to_json()


@ka_router.get("/bonds/prices/qr/{country_code}")
async def get_prices_qr(country_code: CountryCode):
    if not (database_path / "prices.csv").exists():
        if not (raw_data_path / "prices.csv").exists():
            await get_bonds_kaggle_raw()
    df = pd.read_csv(database_path / "prices.csv")
    df = df[[c for c in df.columns if c.startswith(country_code)]]
    Q, R = householder_qr(df.tail(100).values)
    print(df.shape, Q.shape, R.shape)
    return json.dumps({"Q": Q.tolist(), "R": R.tolist()})


@ka_router.get("/countries")
async def get_country_codes():
    return json.dumps(sorted([c.value for c in CountryCode]))
