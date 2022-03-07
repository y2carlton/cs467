from training import Trainer

import pandas as pd
import pendulum

from pathlib import Path
import pickle

SYMBOLS = [
    "NAIL",
    "URTY",
    "UMDD",
    "MIDU",
    "WANT",
    "UDOW",
    "TECL",
    "UBOT",
    "DUSL",
    "UPRO",
    "SPXL",
    "SOXL",
    "UTSL",
    "DRN",
    "CURE",
    "TNA",
    "FAS",
    "TQQQ",
]


def get_volatility_this_week(data_dir=None, trained_models_dir="trained_models"):
    """Returns a list of the assets and their volatility in descending order.
    Will call Alpaca API to retrieve OHLC data. A .env file with Alpaca keys is required if data_dir is not specified or OHLC data is not found in data_dir. Refer to steps 1 to 3 here for more details: https://github.com/y2carlton/cs467#using-get_history_apca-to-create-a-csv-containing-vtis-historical-data

    Args:
        data_dir (str): Will look for OHLC data in this directory if specified. OHLC data is assumed to contain at least 52 weeks of data before the week to be predicted and expected to be named "{data_dir}/{symbol}_ohlc.csv" in CSV format with the following columns:
            timestamp  open  high  low  close  volume  trade_count  vwap
        trained_models_dir (str): Will look for and save trained models in this directory. Models are expected to be named "{trained_models_dir}/{symbol}__Trainer_obj.pkl".

    Returns:
        list: example is [{'symbol': 'NAIL', 'volatility': 0.04109106958}, {'symbol': 'TQQQ', 'volatility': 0.03571258858}, ...]
    """
    week = pendulum.now().in_tz("America/New_York").strftime("%G-W%V")
    return get_volatility_for_week(
        week, data_dir=data_dir, trained_models_dir=trained_models_dir
    )


def get_volatility_next_week(data_dir=None, trained_models_dir="trained_models"):
    """Returns a list of the assets and their volatility in descending order.
    Will call Alpaca API to retrieve OHLC data. A .env file with Alpaca keys is required if data_dir is not specified or OHLC data is not found in data_dir. Refer to steps 1 to 3 here for more details: https://github.com/y2carlton/cs467#using-get_history_apca-to-create-a-csv-containing-vtis-historical-data

    Args:
        data_dir (str): Will look for OHLC data in this directory if specified. OHLC data is assumed to contain at least 52 weeks of data before the week to be predicted and expected to be named "{data_dir}/{symbol}_ohlc.csv" in CSV format with the following columns:
            timestamp  open  high  low  close  volume  trade_count  vwap
        trained_models_dir (str): Will look for and save trained models in this directory. Models are expected to be named "{trained_models_dir}/{symbol}__Trainer_obj.pkl".

    Returns:
        list: example is [{'symbol': 'NAIL', 'volatility': 0.04109106958}, {'symbol': 'TQQQ', 'volatility': 0.03571258858}, ...]
    """
    week = pendulum.now().in_tz("America/New_York").add(weeks=1).strftime("%G-W%V")
    return get_volatility_for_week(
        week, data_dir=data_dir, trained_models_dir=trained_models_dir
    )


def get_volatility_for_week(week, data_dir=None, trained_models_dir="trained_models"):
    """Returns a list of the assets and their volatility in descending order.
    Will call Alpaca API to retrieve OHLC data. A .env file with Alpaca keys is required if data_dir is not specified or OHLC data is not found in data_dir. Refer to steps 1 to 3 here for more details: https://github.com/y2carlton/cs467#using-get_history_apca-to-create-a-csv-containing-vtis-historical-data

    Args:
        week (str): In format YYYY-Www, for example, '2021-W01'
        data_dir (str): Will look for OHLC data in this directory if specified. OHLC data is assumed to contain at least 52 weeks of data before the week to be predicted and expected to be named "{data_dir}/{symbol}_ohlc.csv" in CSV format with the following columns:
            timestamp  open  high  low  close  volume  trade_count  vwap
        trained_models_dir (str): Will look for and save trained models in this directory. Models are expected to be named "{trained_models_dir}/{symbol}__Trainer_obj.pkl".

    Returns:
        list: example is [{'symbol': 'NAIL', 'volatility': 0.04109106958}, {'symbol': 'TQQQ', 'volatility': 0.03571258858}, ...]
    """
    if data_dir is not None and not isinstance(data_dir, str):
        raise Exception(f"Expected data_dir to be a string, but got {type(data_dir)}")

    # Make directories if they do not exist yet
    Path(trained_models_dir).mkdir(parents=True, exist_ok=True)
    if data_dir is not None:
        Path(data_dir).mkdir(parents=True, exist_ok=True)

    ranking = []
    for symbol in SYMBOLS:
        current = {"symbol": symbol}

        # Load trained model or do training
        path = Path(f"{trained_models_dir}/{symbol}__Trainer_obj.pkl")

        if path.is_file():
            with open(path, "rb") as file:
                trainer_object = pickle.load(file)
        else:
            trainer_object = Trainer(symbol)
            trainer_object.train(start="1600-01-01", end="2019-12-31")
            # Save trained model to disk
            with open(f"{trained_models_dir}/{symbol}__Trainer_obj.pkl", "wb") as f:
                pickle.dump(trainer_object, f)

        if data_dir is not None:
            path = Path(f"{data_dir}/{symbol}_ohlc.csv")
            ohlc_df = pd.read_csv(path, index_col="timestamp", parse_dates=True)
            current["volatility"] = trainer_object.predict_by_week_number(
                week, df_to_use=ohlc_df
            )
        else:
            current["volatility"] = trainer_object.predict_by_week_number(week)
        ranking.append(current)
    return sorted(ranking, key=lambda item: item["volatility"], reverse=True)


if __name__ == "__main__":
    print(f"Volatility this week\n{get_volatility_this_week(data_dir='data')}\n")
    print(f"Volatility next week\n{get_volatility_next_week(data_dir='data')}\n")
