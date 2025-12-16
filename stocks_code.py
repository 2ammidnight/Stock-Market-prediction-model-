import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class StockPredictor:
    """
    A class for predicting stock prices using LSTM neural networks.
    
    Attributes:
        ticker (str): Stock ticker symbol
        start_date (str): Start date for historical data
        end_date (str): End date for historical data
        model: Trained LSTM model
        scaler: MinMaxScaler for data normalization
        data: Historical stock data
    """
    
    def __init__(self, ticker, start_date, end_date):
        """
        Initialize the StockPredictor.
        
        Args:
            ticker (str): Stock ticker symbol (e.g., 'AAPL')
            start_date (str): Start date in 'YYYY-MM-DD' format
            end_date (str): End date in 'YYYY-MM-DD' format
        """
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        
    def fetch_data(self):
        """
        Fetch stock data from Yahoo Finance.
        
        Returns:
            pd.DataFrame: Historical stock data
        """
        print(f"Fetching data for {self.ticker}...")
        self.data = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        print(f"Data fetched: {len(self.data)} records")
        print(f"Date range: {self.data.index[0].date()} to {self.data.index[-1].date()}")
        return self.data
    
    def prepare_data(self, lookback=60):
        """
        Prepare data for LSTM model by creating sequences.
        
        Args:
            lookback (int): Number of previous days to use for prediction
            
        Returns:
            tuple: (X_train, X_test, y_train, y_test, lookback)
        """
        df = self.data[['Close']].values
        
        # Scale the data to 0-1 range
        scaled_data = self.scaler.fit_transform(df)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(scaled_data)):
            X.append(scaled_data[i-lookback:i, 0])
            y.append(scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        # Split into train (80%) and test (20%)
        split = int(0.8 * len(X))
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        
        print(f"Training samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        
        return X_train, X_test, y_train, y_test, lookback
    
    def build_model(self, input_shape):
        """
        Build LSTM neural network model.
        
        Args:
            input_shape (tuple): Shape of input data
            
        Returns:
            Sequential: Compiled Keras model
        """
        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=input_shape),
            Dropout(0.2),
            LSTM(50, return_sequences=True),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model
    
    def train(self, epochs=50, batch_size=32):
        """
        Train the LSTM model.
        
        Args:
            epochs (int): Number of training epochs
            batch_size (int): Batch size for training
            
        Returns:
            tuple: (history, train_predict, test_predict, lookback)
        """
        print("Preparing data...")
        X_train, X_test, y_train, y_test, lookback = self.prepare_data()
        
        print("Building model...")
        self.model = self.build_model((X_train.shape[1], 1))
        
        print("Training model...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Make predictions
        train_predict = self.model.predict(X_train)
        test_predict = self.model.predict(X_test)
        
        # Inverse transform predictions to get actual prices
        train_predict = self.scaler.inverse_transform(train_predict)
        test_predict = self.scaler.inverse_transform(test_predict)
        y_train_actual = self.scaler.inverse_transform(y_train.reshape(-1, 1))
        y_test_actual = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calculate performance metrics
        train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_predict))
        test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_predict))
        test_mae = mean_absolute_error(y_test_actual, test_predict)
        
        print(f"\n{'='*60}")
        print(f"Performance Metrics:")
        print(f"{'='*60}")
        print(f"Train RMSE: ${train_rmse:.2f}")
        print(f"Test RMSE: ${test_rmse:.2f}")
        print(f"Test MAE: ${test_mae:.2f}")
        print(f"{'='*60}")
        
        return history, train_predict, test_predict, lookback
    
    def predict_future(self, days=30, lookback=60):
        """
        Predict future stock prices.
        
        Args:
            days (int): Number of days to predict into the future
            lookback (int): Number of previous days to use for prediction
            
        Returns:
            np.ndarray: Array of predicted prices
        """
        last_data = self.data[['Close']].values[-lookback:]
        scaled_data = self.scaler.transform(last_data)
        
        predictions = []
        current_batch = scaled_data.reshape(1, lookback, 1)
        
        for i in range(days):
            pred = self.model.predict(current_batch, verbose=0)[0]
            predictions.append(pred)
            current_batch = np.append(current_batch[:, 1:, :], [[pred]], axis=1)
        
        predictions = self.scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
        return predictions
    
    def plot_results(self, train_predict, test_predict, lookback):
        """
        Plot historical data and predictions.
        
        Args:
            train_predict (np.ndarray): Training predictions
            test_predict (np.ndarray): Testing predictions
            lookback (int): Number of lookback days
        """
        plt.figure(figsize=(15, 6))
        
        # Plot actual prices
        plt.plot(self.data.index, self.data['Close'], 
                label='Actual Price', color='blue', linewidth=2)
        
        # Plot training predictions
        train_dates = self.data.index[lookback:lookback+len(train_predict)]
        plt.plot(train_dates, train_predict, 
                label='Training Predictions', color='green', alpha=0.7)
        
        # Plot test predictions
        test_dates = self.data.index[lookback+len(train_predict):lookback+len(train_predict)+len(test_predict)]
        plt.plot(test_dates, test_predict, 
                label='Test Predictions', color='red', alpha=0.7)
        
        plt.title(f'{self.ticker} Stock Price Prediction - Historical', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    def plot_future_predictions(self, future_prices, days=30):
        """
        Plot future price predictions.
        
        Args:
            future_prices (np.ndarray): Array of predicted future prices
            days (int): Number of days predicted
        """
        plt.figure(figsize=(15, 6))
        
        # Historical data
        plt.plot(self.data.index, self.data['Close'], 
                label='Historical Price', color='blue', linewidth=2)
        
        # Future predictions
        last_date = self.data.index[-1]
        future_dates = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=days, freq='D')
        plt.plot(future_dates, future_prices, 
                label='Future Predictions', color='red', 
                linewidth=2, linestyle='--', marker='o', markersize=3)
        
        plt.axvline(x=self.data.index[-1], color='gray', 
                   linestyle=':', linewidth=2, alpha=0.7)
        
        plt.title(f'{self.ticker} Stock Price - Future Predictions ({days} Days)', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Price ($)', fontsize=12)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


def main():
    """
    Main function for standalone script execution.
    """
    # Configuration
    TICKER = "AAPL"  # Change to any stock ticker
    START_DATE = "2020-01-01"
    END_DATE = datetime.now().strftime("%Y-%m-%d")
    FUTURE_DAYS = 30
    EPOCHS = 50
    BATCH_SIZE = 32
    
    print("="*60)
    print("STOCK PRICE PREDICTION APP")
    print("="*60)
    
    # Initialize predictor
    predictor = StockPredictor(TICKER, START_DATE, END_DATE)
    
    # Fetch data
    predictor.fetch_data()
    
    # Train model
    history, train_predict, test_predict, lookback = predictor.train(
        epochs=EPOCHS, 
        batch_size=BATCH_SIZE
    )
    
    # Plot historical predictions
    predictor.plot_results(train_predict, test_predict, lookback)
    
    # Predict future
    print(f"\nPredicting next {FUTURE_DAYS} days...")
    future_prices = predictor.predict_future(days=FUTURE_DAYS, lookback=lookback)
    
    # Display predictions
    print(f"\n📊 Future Price Predictions for {TICKER}:")
    print("-" * 40)
    for i, price in enumerate(future_prices, 1):
        print(f"Day {i}: ${price[0]:.2f}")
    
    # Plot future predictions
    predictor.plot_future_predictions(future_prices, FUTURE_DAYS)
    
    # Summary
    current_price = predictor.data['Close'][-1]
    predicted_price = future_prices[-1][0]
    change = predicted_price - current_price
    change_pct = (change / current_price) * 100
    
    print(f"\n{'='*60}")
    print(f"Summary:")
    print(f"{'='*60}")
    print(f"Current Price: ${current_price:.2f}")
    print(f"Predicted Price (Day {FUTURE_DAYS}): ${predicted_price:.2f}")
    print(f"Expected Change: ${change:.2f} ({change_pct:+.2f}%)")
    print(f"{'='*60}")
    
    print("\n⚠️  DISCLAIMER: This is for educational purposes only!")
    print("Do NOT use for actual trading decisions.")


if __name__ == "__main__":
    main()






