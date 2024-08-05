# Apple Stock Price 

## Overview

This project aims to forecast Apple Inc.'s stock prices using both traditional time series models like ARIMA and machine learning models like XGBoost. The forecasting models are built and evaluated using historical stock price data.

## Project Structure

**main.py**: The main script that orchestrates the entire forecasting process.

**src/**: Contains modules responsible for data loading, processing, modeling, and evaluation.

**data/**: Functions related to loading and processing data.

    **model/**: Functions to build, train, and evaluate models.

    **evaluate/**: Functions to backtest and plot predictions.

    **utils/**: Utility functions like logging and saving images.

    **data/**: This directory contains the stock price data used for forecasting.

**logs/**: Directory where log files are stored.

**storage/**: Directory to store intermediate files like plots and results.

**requirements.txt**: A list of Python libraries required to run the code.

**README.md**: This file, providing an overview and instructions for the project.


## Dependencies
Plese see the requirement.txt file

## How to Run
To execute the project, navigate to your project directory and run the main.py script. Ensure that Python is installed and accessible via your command line:

python main.py
