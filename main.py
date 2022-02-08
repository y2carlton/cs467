############################################################################################################################
# CS467 - Capstone Project (Winter 2022 term)
# Algorithmic Trading Challenge
# Team JAR
# Last Revised:  February 7, 2022
#
# The following application makes algorithmic trading decision by 1) loading a list of candidate securities, 2) selecting
# a universe of currently peferred securities to deal in and 3) buying/selling the identified securities based on current
# momentum.  For the current project, the default list of candidate securities includes 18 leveraged ETFs which are
# considered to be diverse but volatile enough to offer benefit from momentum trading.  By default the selected universe
# will be comprised of a single preferred security and the buy/sell signals will be generated based on daily or hourly
# data.
#
#        THIS APPLICATION INCLUDES CODE WHICH ALREADY EXISTED, AND WAS NOT DEVELOPED AS PART OF THE PROJECT.
#        
# Since universe selection (as defined above) is the primary focus on the current project, our final project will include
# multiple versions of the universe_select() method.  However, since this method will function hand-in-hand with the
# existing buy/sell logic, back testing for model training and performance comparison will be performed using this
# complete application.
#
# Key parameters defining each test may be found in the accompanying test_config.py file.  Parameters for a back test
# may be specified by updating this file, which allows suites of tests to be defined as a batch.
############################################################################################################################
from System.Drawing import Color
import test_configuration as config


class FirstAlgo(QCAlgorithm):
    ########################################################################################################################
    # This is the entry point for setup when running in QuantConnect.  This method loads data, sets up parameters,
    # schedules actions and so forth.  Execution of the trading algorithm will typically occur in the OnData() event
    # handler, which will be triggered each time new data is received (either in real time, or historical data which
    # emulates operation in the past for the purpose of back testing).
    ########################################################################################################################
    def Initialize(self):
        self.load_parameters()
        self.initializeIndicators()

        # Add Equities and Resolutions for Universe Selection
        for ticker in self.equities:
            self.AddEquity(ticker, Resolution.Hour)
        # self.AddEquity("SPY", Resolution.Hour)

        # Trade Information and variables
        self.weighted_resolutions = {}
        self.universal_max_weight = 100
        self.resolutions = self.weight.keys()
        self.trade = True

        # Scheduled Actions
        self.Schedule.On(self.DateRules.MonthStart(), self.TimeRules.At(1, 00), self.set_universe)
        # self.Schedule.On(self.DateRules.MonthStart(14), self.TimeRules.At(1, 00), self.set_universe)
        self.Schedule.On(self.DateRules.MonthStart(), self.TimeRules.At(1, 00), self.log_portfolio)

        self.buildDataDictionaries()

        # Set Initial Universe
        if self._equity is None:
            self.set_universe()


    ########################################################################################################################
    # This is the primary entry point for the trading algorithm.  Each time a new data point is received, this handler
    # will be triggered.
    # 
    # This method already existed prior to the start of the Winter 2022 ML Algorithmic Trading Challenge, and the
    # underlying trading logic will not be alterred within the scope of said project.  Instead, the current project will
    # focus on the selection logic embodied in the set_universe() method, and will incorporate the logic of the following
    # existing trading algorithm to back test performance in tandem with varying universe selection strategies.
    #
    # Arguments:
    #       data: Slice object keyed by symbol containing the stock data
    ########################################################################################################################
    def OnData(self, data):
        # Validate Data and set Daily Data

        # Check to see whether data has been warmed up (i.e., all pre-test data loaded to initialize trends)
        if self.IsWarmingUp is True or self._equity is None:
            return

        re_balance = False

        if not data.Bars.ContainsKey(self._equity):
            return

        bar = data.Bars[self._equity]

        # Buy and Sell Signals
        for resolution in self.resolutions:

            # Buy Signals
            if bar.Close > self.moving_avg[self._equity][resolution].Current.Value and \
                    self.weighted_resolutions[self._equity][resolution]["weight"] != \
                    self.weighted_resolutions[self._equity][resolution]["max_weight"] and \
                    self.weighted_resolutions[self._equity][resolution]["max_weight"] != 0 and self.trade is True:

                # Increase weight for momentum buying
                self.weighted_resolutions[self._equity][resolution]["weight"] = \
                    self.weighted_resolutions[self._equity][resolution]["max_weight"]
                re_balance = True

            # Sell Signal
            elif bar.Close < self.moving_avg_low[self._equity][resolution].Current.Value and \
                    self.weighted_resolutions[self._equity][resolution]["weight"] != 0 and \
                    self.weighted_resolutions[self._equity][resolution][
                        "max_weight"] != 0 and self.trade is True:  # SELF TRADE

                # Decrease weight for momentum selling
                self.weighted_resolutions[self._equity][resolution]["weight"] = 0
                re_balance = True
                # self.Log(bar)

        # Make trades if momentum dictates
        if re_balance is True:
            self.SetHoldings(self._equity, self.buy_signals(self._equity))


    ########################################################################################################################
    # Loads parameters for the current back test from the test_config.py file.  To define a suite of back tests for
    # model training outside of QuantConnect, the user may automate the generation of a set of test_config.py files, after
    # which he/she will run the back test repetitively with each of the files.  Thus far this is a manual process,
    # although automation would be a good feature for the backlog.
    ########################################################################################################################
    def load_parameters(self):
        self.equities = config.universe2
        self.weight = config.weight_1

        self.SetStartDate(config.startYear, config.startMonth, config.startDay)
        self.SetEndDate(config.endYear, config.endMonth, config.endDay)
        self.SetCash(config.startingCash)
        self.SetWarmUp(config.warmupPeriod)
        self._equity = None
        self._hours = 0


    ########################################################################################################################
    # Initializes the indicators (as lists) which selection and/or trading algorithms will rely upon for make decision.
    ########################################################################################################################
    def initializeIndicators(self):
        # Technical Indicators
        self.moving_avg_close = {}
        self.moving_avg_high = {}
        self.moving_avg_low = {}
        self.moving_avg = {}
        self.bb = {}
        self.std = {}
        self.rocp = {}
        
        
    ########################################################################################################################
    # Builds the dictionaries that selection and/or trading algorithms will rely upon for make decision.
    ########################################################################################################################
    def buildDataDictionaries(self):
        for ticker in self.equities:
            self.moving_avg_close[ticker] = {}
            self.moving_avg_high[ticker] = {}
            self.moving_avg_low[ticker] = {}
            self.moving_avg[ticker] = {}
            self.bb[ticker] = {}
            self.std[ticker] = {}
            self.rocp[ticker] = {}
            for resolution in self.resolutions:
                self.moving_avg_close[ticker][resolution] = self.SMA(ticker, resolution, Resolution.Daily, Field.Close)
                self.moving_avg_high[ticker][resolution] = self.SMA(ticker, resolution, Resolution.Daily, Field.High)
                self.moving_avg_low[ticker][resolution] = self.SMA(ticker, resolution, Resolution.Daily, Field.Low)
                self.moving_avg[ticker][resolution] = self.SMA(ticker, resolution, Resolution.Daily)
            self.bb[ticker] = self.BB(ticker, 14, Resolution.Daily)
            self.std[ticker] = self.STD(ticker, 10, Resolution.Daily)
            self.rocp[ticker] = self.ROCP(ticker, 62, Resolution.Daily)


    ########################################################################################################################
    # Generates buy/sell signals based on changes to weight associated with each ticker symbol.
    ########################################################################################################################
    def buy_signals(self, ticker):
        """
        Calculates buy signals for ticker
        """
        buy = 0

        for resolution in self.resolutions:
            buy += self.weighted_resolutions[ticker][resolution]["weight"]

        return buy / self.universal_max_weight


    ########################################################################################################################
    # This method chooses suitable securit(ies) from the list of available ticker symbols, and creates the universe of
    # securities which are considered to be candidates for trading.
    #
    # For the Algorithmic Trading Challenge, the default operation of this method will be to choose a single security from
    # the full list based on volatility and possibly momentum.  Multiple selection strategies will be tested by creating
    # alternate versions of this selection method.  We will also develop an alternate selection method, which will read
    # precalculated trades from a table in the accompanying test_config.py file, the purpose of which will be to
    # back test algorithms in QuantConnect which rely on pre-computed reference tables outside of QuantConnect.
    #
    # Not for future reference... consider a selector field which can be pulled from test_config.py to determine which
    # selection strategy to use in a given test.
    ########################################################################################################################
    def set_universe(self):
        active = []
        top = None

        # Get the tickers that have a positive rate of change
        for ticker in self.equities:
            if self.std[ticker].Current.Value != 0:
                active.append(ticker)

        if len(active) > 0:
            top = active[0]

        for ticker in active:
            # self.Log("---{0} STD {1}--- STD PERCENT {2}".format(ticker, self.std[ticker].Current.Value, self.std[ticker].Current.Value/self.moving_avg[ticker][3].Current.Value))
            if self.std[ticker].Current.Value / self.moving_avg[ticker][10].Current.Value > self.std[
                top].Current.Value / self.moving_avg[top][10].Current.Value:
                top = ticker

        self._equity = top
        self.weighted_resolutions[top] = self.weight

        # If ticker is not in focus liquidate it
        for ticker in self.equities:
            if ticker is not self._equity:
                self.Liquidate(ticker)


    ########################################################################################################################
    # Writes contents of the portfolio to a log file.
    ########################################################################################################################
    def log_portfolio(self):
        invested_tickers = 0
        self.Log("PORTFOLIO TOTAL: {0} || TOTAL MARGIN USED: {1} || IS TRADING: {2}".format(
            self.Portfolio.TotalPortfolioValue, self.Portfolio.TotalMarginUsed, self.trade))
        for ticker in self.equities:
            if self.ActiveSecurities[ticker].Invested is True:
                invested_tickers += 1

        self.Log("INVESTED TICKERS: {0}".format(invested_tickers))
        self.Log("EQUITIES: {0}".format(self._equity))