from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
import bs4
from dateutil.relativedelta import relativedelta
from dotenv import load_dotenv
import pandas as pd
import requests
import yfinance as yf

import datetime
import os
import pickle


# NOTE: Only use below function if in QuantConnect platform.
# def get_history_qb(
#     qb: QuantBook,
#     symbol_str: str,
#     start: datetime.datetime,
#     end: datetime.datetime,
# ) -> pd.core.frame.DataFrame:
#     """Returns price and volume history for the symbol."""
#     symbol = qb.AddEquity(symbol_str, Resolution.Daily).Symbol
#     return qb.History(symbol, start, end).loc[symbol]


def get_history_yf(
    symbol: str,
    interval: str = "5m",
    period: str = "60d",
    auto_adjust: bool = False,
) -> pd.core.frame.DataFrame:
    """Returns a pandas DataFrame containing the symbol's historical data."""
    return yf.download(symbol, interval=interval, period=period, auto_adjust=False)


def get_history_apca(
    symbol: str,
    interval: TimeFrame = TimeFrame(5, TimeFrameUnit.Minute),
    adjustment: str = "raw",
) -> pd.core.frame.DataFrame:
    """Returns a pandas DataFrame containing the symbol's historical data."""

    # Check for API key in environment variable, load from .env file if possible.
    var_name = "APCA_API_KEY_ID"
    if var_name not in os.environ:
        load_dotenv()
        if var_name not in os.environ:
            raise Exception(
                f"Expected environment variable {var_name} to be set or to be found in .env"
            )
    var_name = "APCA_API_SECRET_KEY"
    if var_name not in os.environ:
        load_dotenv()
        if var_name not in os.environ:
            raise Exception(
                f"Expected environment variable {var_name} to be set or to be found in .env"
            )
    api = REST()

    # NOTE: Cannot get data from the past 15 minutes, so get data up to yesterday.
    earliest = (datetime.datetime(year=1600, month=1, day=1)).strftime("%Y-%m-%d")
    yesterday = (datetime.datetime.today() - relativedelta(days=1)).strftime("%Y-%m-%d")

    return api.get_bars(
        symbol,
        interval,
        earliest,
        yesterday,
        adjustment=adjustment,
    ).df


def get_pcr(kind: str, symbol: str, day_range: int) -> str:
    """Returns the put-call ratio for the symbol.

    Valid kinds: 'volume', 'vol', 'open_interest', 'openinterest', 'oi'
    Example symbol: 'VTI'
    Valid day_ranges: 10, 20, 30, 60, 90, 120, 180
    """
    url = f"https://mktdata.fly.dev/pcr/{kind}/{symbol}/{day_range}"
    pcr_str = requests.get(url).text
    return pcr_str


def get_er(symbol: str) -> str:
    """Returns the expense ratio for the symbol.

    Example symbol: 'TQQQ'
    symbol is case sensitive.
    """
    url = f"https://mktdata.fly.dev/er/{symbol}"
    er_str = requests.get(url).text
    return er_str


def get_fgi() -> dict:
    """Returns the current CNN fear and greed index."""
    url = f"https://mktdata.fly.dev/fgi"
    response_json = requests.get(url).json
    return response_json
