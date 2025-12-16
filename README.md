# 📈 Stock Price Prediction App

A machine learning application that predicts stock prices using LSTM (Long Short-Term Memory) neural networks. This project includes both a Python module for programmatic use and an interactive Jupyter Notebook for analysis and visualization.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🌟 Features

- **Real-time Data Fetching**: Automatically downloads historical stock data from Yahoo Finance
- **LSTM Neural Network**: Deep learning model with 3 LSTM layers for accurate predictions
- **Performance Metrics**: RMSE and MAE calculations to evaluate model accuracy
- **Future Predictions**: Forecast stock prices for the next 30 days (customizable)
- **Beautiful Visualizations**: Multiple charts showing historical data, predictions, and forecasts
- **Interactive Notebook**: Jupyter Notebook for step-by-step analysis
- **Easy Customization**: Simple configuration for different stocks and parameters

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Configuration](#configuration)
- [Examples](#examples)
- [Model Architecture](#model-architecture)
- [Performance](#performance)
- [Disclaimer](#disclaimer)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/stock-prediction-app.git
cd stock-prediction-app
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install numpy pandas matplotlib yfinance scikit-learn tensorflow
```

### Step 3: Verify Installation

```bash
python -c "import tensorflow; print(tensorflow.__version__)"
```

## ⚡ Quick Start

### Option 1: Run Python Script

```bash
python stock_predictor.py
```

### Option 2: Use Jupyter Notebook

```bash
jupyter notebook stock_prediction.ipynb
```

### Option 3: Import in Your Code

```python
from stock_predictor import StockPredictor

# Initialize predictor
predictor = StockPredictor("AAPL", "2020-01-01", "2024-12-16")

# Fetch data
predictor.fetch_data()

# Train model
history, train_pred, test_pred, lookback = predictor.train(epochs=50)

# Predict future
future_prices = predictor.predict_future(days=30)

# Visualize
predictor.plot_future_predictions(future_prices, 30)
```

## 📁 Project Structure

```
stock-prediction-app/
│
├── stock_predictor.py          # Main Python module with StockPredictor class
├── stock_prediction.ipynb      # Interactive Jupyter Notebook
├── requirements.txt            # Project dependencies
├── README.md                   # Project documentation
│
├── data/                       # (Optional) Store downloaded data
│   └── .gitkeep
│
├── models/                     # (Optional) Save trained models
│   └── .gitkeep
│
└── results/                    # (Optional) Save predictions and charts
    └── .gitkeep
```

## 💻 Usage

### Basic Usage

```python
from stock_predictor import StockPredictor
from datetime import datetime

# Configuration
TICKER = "TSLA"
START_DATE = "2020-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")

# Create predictor
predictor = StockPredictor(TICKER, START_DATE, END_DATE)

# Fetch and train
data = predictor.fetch_data()
history, train_pred, test_pred, lookback = predictor.train(epochs=50)

# Make predictions
future_prices = predictor.predict_future(days=30, lookback=lookback)
```

### Advanced Usage - Custom Configuration

```python
# Initialize with custom parameters
predictor = StockPredictor("GOOGL", "2019-01-01", "2024-12-16")

# Fetch data
predictor.fetch_data()

# Train with custom parameters
history, train_pred, test_pred, lookback = predictor.train(
    epochs=100,        # More epochs for better accuracy
    batch_size=16      # Smaller batch size
)

# Predict longer timeframe
future_prices = predictor.predict_future(
    days=60,           # Predict 60 days
    lookback=90        # Use 90 days of history
)

# Visualize results
predictor.plot_results(train_pred, test_pred, lookback)
predictor.plot_future_predictions(future_prices, 60)
```

## ⚙️ Configuration

### Stock Configuration

| Parameter | Description | Default | Example |
|-----------|-------------|---------|---------|
| `TICKER` | Stock symbol | `"AAPL"` | `"TSLA"`, `"GOOGL"`, `"MSFT"` |
| `START_DATE` | Start date for data | `"2020-01-01"` | `"2019-01-01"` |
| `END_DATE` | End date for data | Current date | `"2024-12-16"` |

### Model Configuration

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `LOOKBACK_DAYS` | Days to look back | `60` | 30-120 |
| `FUTURE_DAYS` | Days to predict | `30` | 1-90 |
| `EPOCHS` | Training iterations | `50` | 20-200 |
| `BATCH_SIZE` | Training batch size | `32` | 16-64 |

## 📊 Examples

### Example 1: Apple Stock (AAPL)

```python
predictor = StockPredictor("AAPL", "2020-01-01", "2024-12-16")
predictor.fetch_data()
history, train_pred, test_pred, lookback = predictor.train(epochs=50)
future_prices = predictor.predict_future(days=30)
```

**Sample Output:**
```
Fetching data for AAPL...
Data fetched: 1234 records
Train RMSE: $2.45
Test RMSE: $3.21
Test MAE: $2.87

Day 1: $195.32
Day 2: $196.15
...
Day 30: $203.45
```

### Example 2: Tesla Stock (TSLA)

```python
predictor = StockPredictor("TSLA", "2021-01-01", "2024-12-16")
predictor.fetch_data()
history, train_pred, test_pred, lookback = predictor.train(epochs=75)
future_prices = predictor.predict_future(days=45)
```

### Example 3: Multiple Stocks Comparison

```python
tickers = ["AAPL", "GOOGL", "MSFT", "TSLA"]
predictions = {}

for ticker in tickers:
    predictor = StockPredictor(ticker, "2020-01-01", "2024-12-16")
    predictor.fetch_data()
    _, _, _, lookback = predictor.train(epochs=50, batch_size=32)
    predictions[ticker] = predictor.predict_future(days=30, lookback=lookback)
    
# Compare predictions
for ticker, prices in predictions.items():
    print(f"{ticker}: ${prices[-1][0]:.2f}")
```

## 🧠 Model Architecture

The LSTM model consists of:

```
Layer 1: LSTM (50 units, return_sequences=True)
         Dropout (0.2)

Layer 2: LSTM (50 units, return_sequences=True)
         Dropout (0.2)

Layer 3: LSTM (50 units)
         Dropout (0.2)

Output:  Dense (1 unit)

Optimizer: Adam
Loss: Mean Squared Error
```

### Why LSTM?

- **Captures long-term dependencies** in time series data
- **Remembers patterns** from historical prices
- **Handles sequential data** effectively
- **Dropout layers** prevent overfitting

## 📈 Performance

### Typical Performance Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| **Train RMSE** | $2-5 | Root Mean Squared Error on training data |
| **Test RMSE** | $3-7 | Root Mean Squared Error on test data |
| **Test MAE** | $2-6 | Mean Absolute Error on test data |
| **Training Time** | 2-5 min | On standard CPU (50 epochs) |

### Factors Affecting Accuracy

- **Data quality**: More historical data = better predictions
- **Market volatility**: Stable stocks are easier to predict
- **Training epochs**: More epochs = better learning (up to a point)
- **Lookback period**: Balance between too short and too long

## ⚠️ Disclaimer

**IMPORTANT: This project is for educational purposes only!**

- 📚 **Educational Tool**: Designed to teach machine learning and time series prediction
- 🚫 **Not Financial Advice**: Do NOT use for actual trading or investment decisions
- 📉 **No Guarantees**: Past performance does not guarantee future results
- 💼 **Consult Professionals**: Always consult with a qualified financial advisor
- 🎲 **Market Complexity**: Many factors affect stock prices that this model doesn't consider

### Limitations

- Does not account for:
  - News events and announcements
  - Market sentiment and social media
  - Economic indicators and policies
  - Company fundamentals and earnings
  - Global events and geopolitics
  - Trading volume and market depth

## 🤝 Contributing

Contributions are welcome! Here's how you can help:

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Ideas for Contribution

- Add support for multiple stocks comparison
- Implement model saving/loading functionality
- Add more technical indicators (RSI, MACD, etc.)
- Create a web interface using Streamlit or Flask
- Add backtesting functionality
- Implement ensemble methods
- Add sentiment analysis from news

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Yahoo Finance** for providing free stock data via `yfinance`
- **TensorFlow/Keras** team for the deep learning framework
- **scikit-learn** for preprocessing utilities
- The open-source community for various tools and libraries

## 📧 Contact

- **Author**: KARTIKEYA GURURANI
- **Email**: kartikeyagururani17@gmail.com
- **GitHub**: [@2ammidnight](https://github.com/2ammidnight)

## 🔗 Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Keras LSTM Guide](https://keras.io/api/layers/recurrent_layers/lstm/)
- [Yahoo Finance API](https://github.com/ranaroussi/yfinance)
- [Time Series Forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series)

## 📚 Further Reading

- [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- [Stock Price Prediction Using Machine Learning](https://www.investopedia.com/)
- [Time Series Analysis with Python](https://machinelearningmastery.com/time-series-forecasting/)

---

⭐ **If you find this project helpful, please consider giving it a star!** ⭐

Made with ❤️ and Python
