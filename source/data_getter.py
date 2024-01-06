import logging
from fastapi import APIRouter
import yfinance as yf
import datetime as dt

logger = logging.getLogger(__name__)

yf_router = APIRouter(prefix="/yf")


@yf_router.get("/{ticker}")
async def read_ticker_daily(
    ticker: str, start: dt.datetime = dt.datetime.today(), end: dt.datetime = dt.datetime.today()
):
    logger.info("Fetch TNX data...")
    return (yf.download(ticker, period="1d")).to_json()
