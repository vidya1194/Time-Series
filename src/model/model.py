import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error
from src.utils.utils import setup_logging, save_image
import pandas as pd

logger = setup_logging()

def decompose_series(df):
    try:
        decomposed = seasonal_decompose(df['AAPL'])
        return decomposed.trend, decomposed.seasonal, decomposed.resid
    except Exception as e:
            logger.error(f"Error in load_and_process_data: {e}")
            raise e

def plot_decomposition(df, trend, seasonal, residual):
    try:
        plt.figure(figsize=(12,8))
        plt.subplot(411)
        plt.plot(df['AAPL'], label='Original', color='black')
        plt.legend(loc='upper left')
        plt.subplot(412)
        plt.plot(trend, label='Trend', color='red')
        plt.legend(loc='upper left')
        plt.subplot(413)
        plt.plot(seasonal, label='Seasonal', color='blue')
        plt.legend(loc='upper left')
        plt.subplot(414)
        plt.plot(residual, label='Residual', color='black')
        plt.legend(loc='upper left')
        
        # Get the current figure and save it
        figure = plt.gcf()  # Get the current figure
        save_image(figure, 'plot_decomposition.png')
            
        plt.show() 
        
    except Exception as e:
        logger.error(f"Error in load_and_process_data: {e}")
        raise e

def plot_acf_pacf(df):
    try:
        plot_acf(df['AAPL'].dropna())
        plot_pacf(df['AAPL'].dropna(), lags=11)
        
        # Get the current figure and save it
        figure = plt.gcf()  # Get the current figure
        save_image(figure, 'plot_acf_pacf.png')
        
        plt.show()
    except Exception as e:
        logger.error(f"Error in load_and_process_data: {e}")
        raise e

def adf_test(series):
    try:
        results = adfuller(series)
        return results[1]  # return p-value
    except Exception as e:
        logger.error(f"Error in load_and_process_data: {e}")
        raise e

def difference_series(df):
    try:
        return df['AAPL'].diff().dropna()
    except Exception as e:
            logger.error(f"Error in load_and_process_data: {e}")
            raise e

def train_arima(df, order=(1,1,1)):
    try:
        arima = ARIMA(df.AAPL, order=order)
        ar_model = arima.fit()
        return ar_model
    except Exception as e:
        logger.error(f"Error in load_and_process_data: {e}")
        raise e

def forecast_arima(model, steps=2):
    try:
        forecast = model.get_forecast(steps)
        ypred = forecast.predicted_mean
        conf_int = forecast.conf_int(alpha=0.05)
        return ypred, conf_int
    except Exception as e:
            logger.error(f"Error in load_and_process_data: {e}")
            raise e


def evaluate_model(actual, predicted):
    try:
        return mean_absolute_error(actual, predicted)
    except Exception as e:
            logger.error(f"Error in load_and_process_data: {e}")
            raise e
