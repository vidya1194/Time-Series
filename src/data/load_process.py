import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from src.utils.utils import setup_logging, save_image

logger = setup_logging()

def load_data(file_path):
    try:
        # Read the dataset
        df = pd.read_csv(file_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        logger.error(f"Error in load_and_process_data: {e}")
        raise e


def preprocess_data(data):
    try:
        df = data.iloc[:-2, 0:2]
        df = df.set_index('Date')
        return df
    except Exception as e:
        logger.error(f"Error in load_and_process_data: {e}")
        raise e
    
    
def plot_data(df):
    try:
        # Create seaborn lineplot
        sns.lineplot(df['AAPL'])

        # Get the current figure and save it
        figure = plt.gcf()  # Get the current figure
        save_image(figure, 'learningrate_loss.png')

        # Display the plot
        plt.show()
        
    except Exception as e:
        logger.error(f"Error in load_and_process_data: {e}")
        raise e