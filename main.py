import pandas as pd
from src.data.load_process import load_data, preprocess_data, plot_data
from src.model.model import (decompose_series, plot_decomposition, plot_acf_pacf,
                          adf_test, difference_series, train_arima,
                          forecast_arima, evaluate_model)
from src.model.evaluate import (prepare_yf_ml_data, train_test_split, train_xgboost,
                            evaluate_predictions, plot_predictions, backtest, plot_forecast)
from src.utils.utils import setup_logging

logger = setup_logging()


def main():
    try:
        
        # Load and preprocess data
        data = load_data('data/AAPL.csv')
        logger.info("Data loaded")
        
        df = preprocess_data(data)
        logger.info('Data Preprocessed')
        
        # Plot data
        plot_data(df)
        logger.info('Data Plotted')

        # Decompose series
        trend, seasonal, residual = decompose_series(df)
        plot_decomposition(df, trend, seasonal, residual)
        logger.info('Decompose series Plotted')

        # Plot ACF and PACF
        plot_acf_pacf(df)
        logger.info('Plot ACF and PACF')

        # ADF test
        p_value = adf_test(df['AAPL'])
        print(f'ADF Test p-value: {p_value}')
        logger.info('ADF test')

        # Differencing
        v1 = difference_series(df)
        p_value_diff = adf_test(v1)
        print(f'ADF Test p-value after differencing: {p_value_diff}')
        logger.info('ADF Test p-value after differencing')

        # Train ARIMA model
        arima_model = train_arima(df)
        print(arima_model.summary())
        logger.info('Train ARIMA model')

        # Forecast using ARIMA
        ypred, conf_int = forecast_arima(arima_model)
        dp = pd.DataFrame({
            'Date': pd.to_datetime(['2024-01-01', '2024-02-01']),
            'price_actual': [184.40, 185.04],
            'price_predicted': ypred.values,
            'lower_int': conf_int['lower AAPL'].values,
            'upper_int': conf_int['upper AAPL'].values
        }).set_index('Date')
        print(dp)
        logger.info('Forecast using ARIMA')

        # Plot forecast
        plot_forecast(data, dp['price_predicted'], dp['lower_int'], dp['upper_int'])
        logger.info('Plot forecast')

        # Evaluate ARIMA model
        mae_arima = evaluate_model(dp['price_actual'], dp['price_predicted'])
        print(f'ARIMA MAE: {mae_arima}')
        logger.info('Evaluate ARIMA model')

        # Prepare data for ML model
        data = prepare_yf_ml_data()
        train, test = train_test_split(data)
        logger.info('Prepare data for ML model')

        # Train and evaluate XGBoost model
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        ml_predictions, model = train_xgboost(train, test, features)
        precision = evaluate_predictions(test, ml_predictions)
        print(f'XGBoost Precision: {precision}')
        logger.info('Train and evaluate XGBoost model')

        # Plot XGBoost predictions
        plot_predictions(test, ml_predictions)
        logger.info('Plot XGBoost predictions')

        # Backtest XGBoost model
        backtest_predictions = backtest(data, model, features)
        logger.info('Backtest XGBoost model')
        backtest_precision = evaluate_predictions(backtest_predictions, backtest_predictions['predictions'])
        logger.info('Backtest XGBoost model')
        
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


if __name__ == "__main__":
    main()
