from alpaca_trade_api.rest import REST, TimeFrame, TimeFrameUnit
import bs4
from dotenv import load_dotenv
import pandas as pd
import pendulum
import requests
import yfinance as yf

import os


def get_history_yf(
    symbol,
    interval="5m",
    period="60d",
    auto_adjust=False,
):
    """Returns a pandas DataFrame containing the symbol's historical data.

    Args:
        symbol (str)
        interval (str)
        period (str)
        auto_adjust (bool)

    Returns:
        pd.core.frame.DataFrame
    """
    return yf.download(symbol, interval=interval, period=period, auto_adjust=False)


def get_history_apca(
    symbol,
    interval=TimeFrame(5, TimeFrameUnit.Minute),
    start=None,
    end=None,
    adjustment="raw",
):
    """Returns a pandas DataFrame containing the symbol's historical data.

    Args:
        symbol (str)
        interval (alpaca_trade_api.rest.TimeFrame)
        start (str)
        end (str)
        adjustment (str)

    Returns:
        pd.core.frame.DataFrame
    """

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

    if start is None:
        start = pendulum.datetime(1600, 1, 1).strftime("%Y-%m-%d")
    if end is None:
        # NOTE: Cannot get data from the past 15 minutes, so get data up to yesterday.
        end = (pendulum.yesterday().in_tz('America/New_York')).strftime("%Y-%m-%d")

    return api.get_bars(
        symbol,
        interval,
        start,
        end,
        adjustment=adjustment,
    ).df


def get_pcr(kind, symbol, day_range):
    """Returns the put-call ratio for the symbol.

    Args:
        kind (str): 'volume', 'vol', 'open_interest', 'openinterest', 'oi'
        symbol (str)
        day_ranges (int): 10, 20, 30, 60, 90, 120, 180

    Returns:
        str
    """
    url = f"https://mktdata.fly.dev/pcr/{kind}/{symbol}/{day_range}"
    pcr_str = requests.get(url).text
    return pcr_str


def get_er(symbol):
    """Returns the expense ratio for the symbol.

    Args:
        symbol (str): Case sensitive

    Returns:
        str
    """
    url = f"https://mktdata.fly.dev/er/{symbol}"
    er_str = requests.get(url).text
    return er_str


def get_fgi() -> dict:
    """Returns the current CNN fear and greed index.

    Returns:
        dict
    """
    url = f"https://mktdata.fly.dev/fgi"
    response_json = requests.get(url).json
    return response_json
