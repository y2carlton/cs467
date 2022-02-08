import bs4
import pandas as pd
import datetime
import pickle
import requests


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


def get_pcr(kind: str, symbol_str: str, day_range: int) -> str:
    """Returns the put-call ratio for the symbol.

    Valid kinds: 'volume', 'vol', 'open_interest', 'openinterest', 'oi'
    Example symbol_str: 'VTI'
    Valid day_ranges: 10, 20, 30, 60, 90, 120, 180
    """
    url = f"https://mktdata.fly.dev/pcr/{kind}/{symbol_str}/{day_range}"
    pcr_str = requests.get(url).text
    return pcr_str


def get_er(symbol_str: str) -> str:
    """Returns the expense ratio for the symbol.

    Example symbol_str: 'TQQQ'
    symbol_str is case sensitive.
    """
    url = f"https://mktdata.fly.dev/er/{symbol_str}"
    er_str = requests.get(url).text
    return er_str


def get_fgi() -> dict:
    """Returns the expense ratio for the symbol.

    Example symbol_str: 'TQQQ'
    symbol_str is case sensitive.
    """
    url = f"https://mktdata.fly.dev/fgi"
    response_json = requests.get(url).json
    return response_json
