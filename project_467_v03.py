#####################################################################################################################
#  Following code references daily data (particularly closing prices) for 18 leveraged ETFs, which make up the
#  universe of investable securities for the CS467 Capstone Project.  Based on this data, it calculates monthly
#  statistics, which are potentially indicative of momentum and volatility.  Tables of these results are output
#  to files for further use as inputs for asset selection algorithms, including machine-learning based algorithms.
#####################################################################################################################
import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
import statistics

data_path = "../data/"                                      # Path to folder with input files
output_path = "../output/"                                  # Path to folder in which output files will be written
security_list_source = "security_list.txt"                  # List of security ticker symbols to be considered
DAILY_DATE = 0                                              # Column numbers (0 thru 6) for corresponding data fields
DAILY_OPEN = 1
DAILY_HIGH = 2
DAILY_LOW = 3
DAILY_CLOSE = 4
DAILY_ADJ_CLOSE = 5
DAILY_VOLUME = 6


#####################################################################################################################
# Helper functions
#####################################################################################################################

# Sets up the necessary directories if they do not exist already
def initialize():
    path = Path(data_path)
    if not path.is_dir():
        print("Data directory not found at " + data_path + "\nMaking directory... ", end="")
        path.mkdir()
        print("Data directory created")

    path = Path(output_path)
    if not path.is_dir():
        print("Output directory not found at " + output_path + "\nMaking directory... ", end="")
        path.mkdir()
        print("Output directory created")

    path = Path(data_path + security_list_source)
    if not path.is_file():
        print("Security list file not found at " + security_list_source + "\nMaking empty file...")
        path.touch()




#####################################################################################################################
#  Object representing an instance of a given security, including data and useful methods.
#####################################################################################################################

class Asset:
    def __init__(self, name):
        self.name = name

    # Loads the daily data from files and stores the data in table at self.dailyData[][]
    def GetDailyData(self):
        path = Path(data_path + self.name + ".csv")

        df = yf.download(self.name)
        df.to_csv(path)

        fileIn = open(path, "r")
        fileIn.readline()
        self.dailyData = []

        for line in fileIn:
            currentValues = line.split(",")
            currentRecord = []

            for field in currentValues:
                currentRecord.append(field.rstrip())

            self.dailyData.append(currentRecord)

        fileIn.close()

    # Calculates monthly data based on self.dailyData[][], stores result in self.monthlyData[][]
    def CalculateMonthlyData(self):
        self.dailyData.sort(key=lambda x: x[0])

        self.monthlyData = []
        currentMonth = self.dailyData[0][DAILY_DATE][0:7]
        startingMonth = getNextMonthString(currentMonth)
        currentMonthClosing = []
        previousMonthClosing = 1

        for i in range(len(self.dailyData)):                                # For each record...
            if self.dailyData[i][DAILY_DATE][0:7] != currentMonth:          #   if it is from a new month and not the first (partial) month, calculate stats for previous month
                if self.dailyData[i][DAILY_DATE][0:7] != startingMonth:
                    monthlyRecord = [currentMonth]
                    monthlyRecord.append(calculateStandardDev(currentMonthClosing))                                     # Calculate standard deviation of closing prices for the month
                    monthlyRecord.append(calculateAveragePrice(currentMonthClosing))                                    # Calculate average closing price for the month
                    monthlyRecord.append(calculateRelativeStandardDev(currentMonthClosing))                             # Calculate relative standard deviation of closing prices for the month
                    monthlyRecord.append(calculateGeometricMeanDailyReturn(currentMonthClosing, previousMonthClosing))  # Calculate geometric mean of daily returns for the month
                    self.monthlyData.append(monthlyRecord)

                previousMonthClosing = currentMonthClosing[-1]              # Save last closing price of month for calculations of next month's returns
                currentMonthClosing = []                                    # Clear list of closing prices from last month
                currentMonth = self.dailyData[i][DAILY_DATE][0:7]           # Update the current month to new one

            currentMonthClosing.append(self.dailyData[i][DAILY_CLOSE])      # Add current daily closing price to list for current month

    # Output monthly statistics for current security to a file
    def WriteFileMonthly(self):
        fileOut = open(output_path + self.name + "_monthly.csv", "w")
        fileOut.write("Year-Month, Standard Deviation, Average Price, Relative Standard Deviation, Geometric Mean Daily Return\n")

        for i in range(len(self.monthlyData)):
            fileOut.write(ConvertListToCSV(self.monthlyData[i]) + '\n')

        fileOut.close


# Import the list of ticker symbols for all of the securities to be considered
def GetSecurityList():
       fileIn = open(data_path + security_list_source, "r")
       security_list = []

       for line in fileIn:
           security_list.append(line.rstrip())

       fileIn.close()
       return security_list


# Convert a list of elements to a string in a .csv format
def ConvertListToCSV(list):
    outputString = ""
    blnFirst = True

    for element in list:
        if not blnFirst:
            outputString = outputString + ", "

        outputString = outputString + str(element)
        blnFirst = False

    return outputString


# Accepts a month string in "YYYY-MM" format and returns a string containing the next month in the same format
def getNextMonthString(currentMonth):
    currentYear = int(currentMonth[0:4])
    currentMonth = int(currentMonth[-2:])

    if currentMonth == 12:
        currentMonth = 1
        currentYear += 1
    else:
        currentMonth += 1

    return format(currentYear, '04d') + "-" + format(currentMonth, '02d')


# Calculates and returns the standard deviation of prices contained in the dailyPrices list
def calculateStandardDev(dailyPrices):
    floatPrices = [float(element) for element in dailyPrices]
    return statistics.pstdev(floatPrices)


# Calculates and returns the average of prices contained in the dailyPrices list
def calculateAveragePrice(dailyPrices):
    floatPrices = [float(element) for element in dailyPrices]
    return statistics.mean(floatPrices)


# Calculates and returns the relative standard deviation (as opposed to the absolute standard deviation) of prices contained in the dailyPrices list
def calculateRelativeStandardDev(dailyPrices):
    floatPrices = [float(element) for element in dailyPrices]
    return statistics.pstdev(floatPrices) / statistics.mean(floatPrices)


# Calculates and returns the geometric mean daily return of prices contained in the dailyPrices list
def calculateGeometricMeanDailyReturn(dailyPrices, previousMonthClosing):
    previousMonthEnd = float(previousMonthClosing)
    currentMonthEnd = float(dailyPrices[-1])
    return (currentMonthEnd / previousMonthEnd) ** (1.0 / len (dailyPrices)) - 1.0




#####################################################################################################################
#  Execution starts here
#####################################################################################################################

if __name__ == "__main__":
    initialize()                                    # Sets up the necessary directories if they do not exist already

    security_list = GetSecurityList()               # Load the list of securities (ticker symbols) to be considered

    if len(security_list) == 0:
        print("Please add security symbols line by line in " + data_path + security_list_source)

    for security in security_list:                  # For each security to be considered
        currentSecurity = Asset(security)           #   ...create an Asset object,
        currentSecurity.GetDailyData()              #   ...input the daily data from files,
        currentSecurity.CalculateMonthlyData()      #   ...calculate the monthly data that we need
        currentSecurity.WriteFileMonthly()          #   ...and write these to file for use as security selection inputs.
