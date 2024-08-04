import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import precision_score
from xgboost import XGBClassifier
import yfinance as yf
from src.utils.utils import setup_logging, save_image
import datetime

def plot_forecast(data, predictions, lower_int, upper_int):
    plt.plot(data.AAPL)
    plt.plot(predictions, color='orange')
    plt.fill_between(predictions.index, lower_int, upper_int, color='k', alpha=.15)
    plt.title('Model Performance')
    plt.legend(['Actual','Prediction'], loc='lower right')
    plt.xlabel('Date')
    plt.xticks(rotation=30)
    plt.ylabel('Price (USD)')
    
    # Get the current figure and save it
    figure = plt.gcf()  # Get the current figure
    save_image(figure, 'Model Performance')
    
    plt.show()

def prepare_yf_ml_data():
    data = yf.download("AAPL", start="2000-01-01", end="2022-05-31")
    data['Next_day'] = data['Close'].shift(-1)
    data['Target'] = (data['Next_day'] > data['Close']).astype(int)
    return data

def train_test_split(data, split_ratio=0.9):
    split_point = int(len(data) * split_ratio)
    train = data.iloc[:split_point]
    test = data.iloc[split_point:]
    return train, test

def train_xgboost(train, test, features):
    model = XGBClassifier(max_depth=3, n_estimators=100, random_state=42)
    model.fit(train[features], train['Target'])
    predictions = model.predict(test[features])
    return predictions, model

def evaluate_predictions(test, predictions):
    precision = precision_score(test['Target'], predictions)
    return precision

def plot_predictions(test, predictions):
    plt.plot(test['Target'], label='Actual')
    plt.plot(predictions, label='Predicted')
    plt.legend()
    
    # Get the current timestamp
    filename = f"prediction_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
    # Get the current figure and save it
    figure = plt.gcf()  # Get the current figure
    save_image(figure, filename)
    
    plt.show()

def backtest(data, model, features, start=5031, step=120):
    all_predictions = []
    for i in range(start, data.shape[0], step):
        train = data.iloc[:i].copy()
        test = data.iloc[i:(i + step)].copy()
        
        # Ensure 'Target' column is present in test
        if 'Target' not in test.columns:
            test['Next_day'] = test['Close'].shift(-1)
            test['Target'] = (test['Next_day'] > test['Close']).astype(int)
        
        model_preds = model.predict(test[features])
        test['predictions'] = model_preds
        all_predictions.append(test[['Target', 'predictions']])
    
    return pd.concat(all_predictions)
