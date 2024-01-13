
"""
José Andrés Baca Salgado
bacasalgado@gmail.com
12/15/2023
"""

# Importing necessary libraries
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Function to fetch historical data for a given ticker
def fetch_data(ticker):
    """
    Fetch historical stock data for the specified ticker.

    :param ticker: Ticker symbol of the stock.
    :return: DataFrame with historical stock data.
    """
    stock_data = yf.Ticker(ticker)
    return stock_data.history(period="max")

# Function to preprocess and prepare dataset
def preprocess_data(stock_data):
    """
    Preprocess the stock data by cleaning and creating new features.

    :param stock_data: DataFrame with stock data.
    :return: DataFrame after preprocessing.
    """
    # Dropping unnecessary columns
    stock_data.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)

    # Creating target variable 'Target' where 1 indicates price increase
    stock_data["Target"] = (stock_data["Close"].shift(-1) > stock_data["Close"]).astype(int)

    # Creating additional technical indicators as features
    stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()
    stock_data['Volume_MA'] = stock_data['Volume'].rolling(window=50).mean()
    stock_data['14d_RSI'] = compute_RSI(stock_data['Close'], 14)
    stock_data['MACD'], stock_data['MACD_Signal'] = compute_MACD(stock_data['Close'])
    stock_data['BB_Upper'], stock_data['BB_Lower'] = compute_bollinger_bands(stock_data['Close'])

    # Fill NaN values using both forward fill and backward fill
    stock_data.fillna(method='ffill', inplace=True)
    stock_data.fillna(method='bfill', inplace=True)

    return stock_data

# Function to calculate RSI
def compute_RSI(series, period):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)

    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Function to calculate MACD and MACD signal
def compute_MACD(series):
    exp1 = series.ewm(span=12, adjust=False).mean()
    exp2 = series.ewm(span=26, adjust=False).mean()
    macd = exp1 - exp2
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal

# Function to compute Bollinger Bands
def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

# Function to train the RandomForest model
def train_model(data, predictors):
    # Check if data is empty
    if data.empty or len(data) < 100:
        raise ValueError("Insufficient data for training. Please check your data preprocessing steps.")

    # Handling NaN values using an imputer
    imputer = SimpleImputer(strategy='mean')
    train_features = imputer.fit_transform(data.iloc[:-100][predictors])
    train_target = data.iloc[:-100]["Target"]

    model = RandomForestClassifier(n_estimators=100, min_samples_split=100, random_state=1)
    model.fit(train_features, train_target)
    return model, imputer

# Function to evaluate and plot model performance
def evaluate_and_plot(model, imputer, test_data, predictors):
    # Imputing NaN values in test data
    test_features = imputer.transform(test_data[predictors])

    preds = model.predict(test_features)
    precision = precision_score(test_data["Target"], preds)
    print(f"Model Precision: {precision}")

    # Plotting the results
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_data.index, test_data['Close'], label='Actual Prices')
    ax.scatter(test_data.index, test_data['Close'], c=preds, cmap='viridis', label='Predictions (1 for Increase, 0 for Decrease)')
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.title('Stock Price Prediction vs Actual Prices')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.legend()
    plt.show()

    # Creating trade table
    trades = test_data[['Close']].copy()
    trades['Predictions'] = preds
    trades['Correct'] = trades['Predictions'] == test_data['Target']
    return trades

# Main function to execute the stock prediction workflow
def main():
    sp500 = fetch_data("^GSPC")
    original_row_count = len(sp500)
    sp500 = preprocess_data(sp500)

    # Check if sufficient data remains after preprocessing
    if len(sp500) < 100 or len(sp500) < original_row_count * 0.5:  # Example threshold
        raise ValueError("Insufficient data after preprocessing. Please adjust preprocessing steps.")

    predictors = ['Close', 'Volume', '50_MA', '200_MA', 'Volume_MA', '14d_RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower']
    model, imputer = train_model(sp500, predictors)

    test = sp500.iloc[-100:]
    trades = evaluate_and_plot(model, imputer, test, predictors)
    print(trades)

# Entry point for the script
if __name__ == "__main__":
    main()

import yfinance as yf
import pandas as pd
import numpy as np

# Definiciones de las funciones de los indicadores técnicos
def compute_RSI(series, period=14):
    delta = series.diff()
    gain = (delta.clip(lower=0)).fillna(0)
    loss = (-delta.clip(upper=0)).fillna(0)

    avg_gain = gain.rolling(window=period, min_periods=1).mean()
    avg_loss = loss.rolling(window=period, min_periods=1).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    rsi[avg_loss == 0] = 100  # Considerar RSI como 100 si la pérdida promedio es cero
    return rsi

def compute_MACD(series, fast_period=12, slow_period=26, signal_period=9):
    fast_ema = series.ewm(span=fast_period, adjust=False).mean()
    slow_ema = series.ewm(span=slow_period, adjust=False).mean()
    macd = fast_ema - slow_ema
    signal = macd.ewm(span=signal_period, adjust=False).mean()
    return macd, signal

def compute_bollinger_bands(series, window=20, num_std=2):
    rolling_mean = series.rolling(window=window).mean()
    rolling_std = series.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return upper_band, lower_band

def compute_MFI(high, low, close, volume, period=14):
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume

    positive_flow = money_flow * (typical_price > typical_price.shift(1))
    negative_flow = money_flow * (typical_price < typical_price.shift(1))

    positive_flow_sum = positive_flow.rolling(window=period, min_periods=1).sum()
    negative_flow_sum = negative_flow.rolling(window=period, min_periods=1).sum()

    mfi = 100 - (100 / (1 + (positive_flow_sum / negative_flow_sum)))
    mfi[negative_flow_sum == 0] = 100  # Considerar MFI como 100 si el flujo negativo es cero
    return mfi

def compute_fibonacci_levels(series):
    max_price = series.max()
    min_price = series.min()
    diff = max_price - min_price
    level1 = max_price - 0.236 * diff
    level2 = max_price - 0.382 * diff
    level3 = max_price - 0.618 * diff
    return level1, level2, level3

# Función para cargar y preprocesar los datos
def fetch_and_preprocess(ticker):
    stock_data = yf.Ticker(ticker).history(period="max")
    stock_data.drop(["Dividends", "Stock Splits"], axis=1, inplace=True)
    stock_data["Target"] = (stock_data["Close"].shift(-1) > stock_data["Close"]).astype(int)

    # Aplicar las funciones de indicadores técnicos
    stock_data['14d_RSI'] = compute_RSI(stock_data['Close'])
    macd, macd_signal = compute_MACD(stock_data['Close'])
    stock_data['MACD'] = macd
    stock_data['MACD_Signal'] = macd_signal
    upper_band, lower_band = compute_bollinger_bands(stock_data['Close'])
    stock_data['BB_Upper'] = upper_band
    stock_data['BB_Lower'] = lower_band
    stock_data['MFI'] = compute_MFI(stock_data['High'], stock_data['Low'], stock_data['Close'], stock_data['Volume'])
    fib_level1, fib_level2, fib_level3 = compute_fibonacci_levels(stock_data['Close'])
    stock_data['Fib_Level1'] = fib_level1
    stock_data['Fib_Level2'] = fib_level2
    stock_data['Fib_Level3'] = fib_level3

    # Calcular medias móviles
    stock_data['50_MA'] = stock_data['Close'].rolling(window=50).mean()
    stock_data['200_MA'] = stock_data['Close'].rolling(window=200).mean()

    stock_data.fillna(method='ffill', inplace=True)
    stock_data.fillna(method='bfill', inplace=True)

    return stock_data

# Cargar y preprocesar los datos
sp500 = fetch_and_preprocess("^GSPC")

import itertools
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

# Asegúrate de que 'features' contenga todas las columnas que quieres analizar
features = ['Close', 'Volume', '50_MA', '200_MA', '14d_RSI', 'MACD', 'MACD_Signal', 'BB_Upper', 'BB_Lower', 'MFI', 'Fib_Level1', 'Fib_Level2', 'Fib_Level3']

# Limitar el número de combinaciones evaluando hasta un máximo de 5 características por combinación
max_features = 5
all_combinations = [combo for r in range(1, max_features + 1)
                    for combo in itertools.combinations(features, r)]

# Dividir los datos en conjuntos de entrenamiento y prueba con muestreo estratificado
X = sp500[features]
y = sp500['Target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Evaluar cada combinación de características
results = []
for combo in all_combinations:
    model = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=42)
    model.fit(X_train[list(combo)], y_train)
    predictions = model.predict(X_test[list(combo)])
    precision = precision_score(y_test, predictions)
    results.append((combo, precision))

# Ordenar los resultados por precisión y mostrar
results.sort(key=lambda x: x[1], reverse=True)

# Mostrar las combinaciones más efectivas
for combo in results[:10]:  # Ajusta esto para mostrar más combinaciones si es necesario
    print(f"Combinación: {combo[0]}, Precisión: {combo[1]:.4f}")