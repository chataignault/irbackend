import json
import kaggle
import pandas as pd
from pathlib import Path
from fastapi import APIRouter

from .ka_helpers import CountryCode
from ...utils import householder_qr

ka_router = APIRouter(prefix="/ka", tags=["Kaggle Hub Datasets"])

bond_data_path = Path("bonds_data_cache")


@ka_router.get("/bonds/download_raw")
async def get_raw_data():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "everget/government-bonds", path=bond_data_path, unzip=True
    )
    return


@ka_router.get("/bonds/yields")
async def get_yields():
    if not (bond_data_path / "yields.csv").exists():
        await get_raw_data()
    df = pd.read_csv(bond_data_path / "yields.csv")
    return df.to_json()


@ka_router.get("/bonds/prices/qr/{country_code}")
async def get_prices_qr(country_code: CountryCode):
    if not (bond_data_path / "prices.csv").exists():
        await get_raw_data()
    df = pd.read_csv(bond_data_path / "prices.csv")
    df = df[[c for c in df.columns if c.startswith(country_code)]]
    Q, R = householder_qr(df.tail(100).values)
    print(df.shape, Q.shape, R.shape)
    return json.dumps({"Q": Q.tolist(), "R": R.tolist()})


@ka_router.get("/countries")
async def get_country_codes():
    return json.dumps([c.value for c in CountryCode])
