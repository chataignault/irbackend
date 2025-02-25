import kaggle
import pandas as pd
from pathlib import Path
from fastapi import APIRouter

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


@ka_router.get("/bonds/prices")
async def get_yields():
    if not (bond_data_path / "prices.csv").exists():
        await get_raw_data()
    df = pd.read_csv(bond_data_path / "prices.csv")
    return df.to_json()
