import os
import logging
import pandas as pd
from fastapi import APIRouter
import yfinance as yf
import datetime as dt

from .constant import Ticker

logger = logging.getLogger(__name__)

yf_router = APIRouter(prefix="/yf")


def read_ticker(ticker: Ticker, start: dt.datetime, end: dt.datetime, save: bool = False):
    """
    yfinance API proxy for historical data
    Read if file already exists else fetch
    """
    file_name = "_".join([ticker, start.strftime("%Y%m%d"), end.strftime("%Y%m%d"), ".parquet"])
    file_path = os.path.join(os.getcwd(), "data", file_name)
    if os.path.exists(file_path):
        logger.info(f"Reading file {file_name}...")
        df = pd.read_parquet(file_name)
    else:
        logger.info(f"Fetch {ticker} data...")
        df = yf.Ticker(ticker).history(start=start, end=end)
        if save:
            df.to_parquet(file_path)
    return df


@yf_router.get("/ticker/{ticker}")
async def read_ticker_daily(
    ticker: Ticker,
    start: dt.datetime = dt.datetime.today(),
    end: dt.datetime = dt.datetime.today(),
    save: bool = False,
):
    df = read_ticker(ticker, start, end, save)
    return df.to_json()


@yf_router.get("/tickers")
async def read_all_tickers(
    start: dt.datetime = dt.datetime.today(),
    end: dt.datetime = dt.datetime.today(),
    save: bool = False,
):
    dfx = []
    for ticker in Ticker:
        df_t = read_ticker(ticker, start, end)
        df_t.columns = [" ".join([c, ticker.value]) for c in df_t.columns]
        dfx.append(df_t)
    df = pd.concat(dfx, axis=1)
    return df.to_json()

