from data_retrieval import get_history_apca
from project_467_v03 import calculateRelativeStandardDev
from utils import log

import numpy as np
import pandas as pd
import pendulum
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense, Dropout

import logging
import statistics


class Trainer:
    def __init__(self, symbol):
        self.symbol = symbol
        self.model = None
        self.scaler = None
        self._original_df = None

    @log("Beginning process to train model")
    def train(self):
        df = self._get_historical_data_for_training()
        self._rename_cols(df)
        self._create_helper_col(df)
        df = self._create_volatility_df(df)
        self.scaler = self._create_scaler(df)
        df = self._normalize_values(df)

        # Train to use past 52 weeks to predict the next 1 week.
        self._train_x, self._train_y = self._create_training_set(
            df, n_past=52, n_future=1
        )
        validation_split = 0.2
        optimal_epochs = self._get_optimal_epochs(
            validation_split=validation_split, verbose=0
        )

        self.model = self._create_model()
        self.model.fit(
            self._train_x,
            self._train_y,
            epochs=optimal_epochs,
            validation_split=validation_split,
            verbose=1,
        )

    @log("Predicting next week's volatility")
    def predict_next(self):
        if self.model is None:
            raise Exception("Model has not been trained yet")

        start = (
            pendulum.today()
            .in_tz("America/New_York")
            .subtract(weeks=55)
            .strftime("%Y-%m-%d")
        )
        end = (
            pendulum.today()
            .in_tz("America/New_York")
            .subtract(days=1)
            .strftime("%Y-%m-%d")
        )
        df = get_history_apca(self.symbol, start=start, end=end)

        self._rename_cols(df)
        self._create_helper_col(df)
        df = self._create_volatility_df(df)
        df = self._normalize_values(df)

        input = np.array([df[-52:]])
        prediction_normalized = self.model.predict(input)
        prediction = scaler.inverse_transform(prediction_normalized)[0][0]
        return prediction

    @log("Predicting current week's volatility")
    def predict_current(self):
        if self.model is None:
            raise Exception("Model has not been trained yet")

        start = (
            pendulum.today()
            .in_tz("America/New_York")
            .subtract(weeks=55)
            .strftime("%Y-%m-%d")
        )
        end = (
            pendulum.today()
            .in_tz("America/New_York")
            .subtract(days=1)
            .strftime("%Y-%m-%d")
        )
        df = get_history_apca(self.symbol, start=start, end=end)

        self._rename_cols(df)
        self._create_helper_col(df)
        df = self._create_volatility_df(df)
        df = self._normalize_values(df)

        input = np.array([df[-(52 + 1) : -1]])
        prediction_normalized = self.model.predict(input)
        prediction = scaler.inverse_transform(prediction_normalized)[0][0]
        return prediction

    @log("Getting historical data for training")
    def _get_historical_data_for_training(self, force=False):
        """Returns a dataframe of 5-minute interval historical OHLC data  for the symbol.

        Args:
            force (bool): Only retrieves data from the internet if it hasn't already or if force is set to True.

        Returns:
            pandas.core.frame.DataFrame
        """
        if force is True or self._original_df is None:
            self._original_df = get_history_apca(self.symbol)
        return self._original_df.copy(deep=True)

    @log("Renaming dataframe columns")
    def _rename_cols(self, df):
        """Modifies the dataframe in place.

        Args:
            df (pandas.core.frame.DataFrame)

        Returns:
            pandas.core.frame.DataFrame
        """
        df.index.names = ["Datetime"]
        df.rename(
            columns={
                "timestamp": "Datetime",
                "open": "Open",
                "high": "High",
                "low": "Low",
                "close": "Close",
                "volume": "Volume",
            },
            inplace=True,
        )

    @log("Creating helper column")
    def _create_helper_col(self, df, time_format="%G-W%V"):
        """Creates a helper column in the dataframe to group by later on.
        Modifies the dataframe in place.

        Args:
            df (pandas.core.frame.DataFrame)
            time_format (str)

        Returns:
            pandas.core.frame.DataFrame
        """
        df["Group"] = df.index
        df["Group"] = df["Group"].dt.strftime(time_format)  # YYYY-Www format

    @log("Creating new dataframe of the volatility grouped by the helper column")
    def _create_volatility_df(self, df):
        """Creates a new dataframe of the volatility grouped by the helper column.

        Args:
            df (pandas.core.frame.DataFrame)

        Returns:
            pandas.core.frame.DataFrame
        """
        # Standard deviation of each week will be our measure of volatility
        measure_of_volatility = calculateRelativeStandardDev
        new_df = df.groupby("Group")["Close"].agg(measure_of_volatility).to_frame()
        new_df.rename(columns={"Close": "Volatility"}, inplace=True)
        return new_df

    @log("Fitting scaler to data")
    def _create_scaler(self, df):
        """Creates the scaler to normalize values.

        Args:
            df (pandas.core.frame.DataFrame)

        Returns:
            sklearn.preprocessing.MinMaxScaler
        """
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(np.array(df).reshape(-1, 1))
        return scaler

    @log("Normalizing volatility values")
    def _normalize_values(self, df):
        """Normalize volatility values because LSTM is sensitive to magnitude.

        Args:
            df (pandas.core.frame.DataFrame)

        Returns:
            pandas.core.frame.DataFrame
        """
        return self.scaler.transform(np.array(df).reshape(-1, 1))

    @log("Creating training set")
    def _create_training_set(self, df, n_past, n_future):
        """Creates and returns a training set based on the dataframe.

        Args:
            df (pandas.core.frame.DataFrame)
            n_past (int): Number of weeks will we look back to use as our predictor for the future
            n_future (int): Number of weeks will we try to predict volatility for in the future

        Returns:
            tuple (
                numpy.ndarray: The data to make predictions from
                numpy.ndarray: The correct answers
            ): Training set to be fed into LSTM model
        """
        train_x = []
        train_y = []

        for i in range(n_past, len(df) - n_future + 1):
            train_x.append(df[i - n_past : i])
            train_y.append(df[i + n_future - 1 : i + n_future, 0])

        # Convert to numpy arrays
        return np.array(train_x), np.array(train_y)

    @log("Creating LSTM model")
    def _create_model(self) -> Sequential:
        """Creates and returns an LSTM model.

        Returns:
            keras.engine.sequential.Sequential
        """
        model = Sequential()
        model.add(
            LSTM(
                64,
                input_shape=(self._train_x.shape[1], self._train_x.shape[2]),
                return_sequences=True,
            )
        )  # Return for next LSTM layer
        model.add(LSTM(32, return_sequences=False))
        model.add(
            Dropout(0.2)
        )  # A regularization technique to help prevent overfitting
        model.add(Dense(self._train_y.shape[1]))
        model.compile(loss="mean_squared_error", optimizer="adam")
        return model

    @log(
        "Attempting to find optimal epoch number, will create and get the epoch number with minimum validation loss multiple times"
    )
    def _get_optimal_epochs(
        self,
        num_times=15,
        max_epochs=20,
        validation_split=0.2,
        verbose=0,
    ):
        """Simple attempt to find a decent number of epochs to train for.

        Args:
            num_times (int): Number of trainings to check optimal epoch number for
            max_epochs (int)
            validation_split (float): Between 0 and 1
            verbose (int): 0 = silent, 1 = progress bar, 2 = one line per epoch


        Returns:
            int
        """
        epoch_nums = []
        for attempt in range(num_times):
            model = self._create_model()
            training_history = model.fit(
                self._train_x,
                self._train_y,
                epochs=max_epochs,
                validation_split=validation_split,
                verbose=verbose,
            )
            epoch_nums.append(self._get_epoch_num_with_min_val_loss(training_history))
        return int(statistics.median(epoch_nums))

    @log("Getting the epoch number with the minimum validation loss")
    def _get_epoch_num_with_min_val_loss(self, training_history):
        """Returns the epoch number with the minimum validation loss.

        Args:
            training_history (keras.callbacks.History)
        Returns:
            int
        """
        epoch_num_with_min_val_loss = 1
        min_val_loss = training_history.history["val_loss"][0]
        for i, val_loss in enumerate(training_history.history["val_loss"]):
            current_epoch_num = i + 1
            if val_loss < min_val_loss:
                epoch_num_with_min_val_loss = current_epoch_num
                min_val_loss = val_loss
        return epoch_num_with_min_val_loss


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y/%m/%d %H:%M:%S",
    )

    model = Trainer("SPY")

    model.train()

    print(f"Next week's predicted volatility: {model.predict_next()}")
    print(f"This week's predicted volatility: {model.predict_current()}")
