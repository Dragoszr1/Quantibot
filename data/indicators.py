import pandas as pd
import numpy as np

# -------------------------
# Moving Averages
# -------------------------

def SMA(series, period):
    """Simple Moving Average"""
    return series.rolling(window=period).mean()

def EMA(series, period):
    """Exponential Moving Average"""
    return series.ewm(span=period, adjust=False).mean()

# -------------------------
# Momentum Indicators
# -------------------------

def RSI(series, period=14):
    """Relative Strength Index"""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def MACD(series, fast=12, slow=26, signal=9):
    """Moving Average Convergence Divergence"""
    ema_fast = EMA(series, fast)
    ema_slow = EMA(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = EMA(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram

# -------------------------
# Volatility / Trend Indicators
# -------------------------

def Bollinger_Bands(series, period=20, k=2):
    """Upper and Lower Bollinger Bands"""
    sma = SMA(series, period)
    std = series.rolling(window=period).std()
    upper = sma + k * std
    lower = sma - k * std
    return upper, lower

def ATR(df, period=14):
    """Average True Range"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

# -------------------------
# Trend Strength Indicators
# -------------------------

def ADX(df, period=14):
    """Average Directional Index - measures trend strength"""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # True Range
    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smooth the values
    tr_smooth = tr.rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / tr_smooth
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / tr_smooth
    
    # Calculate ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx

def Trend_Strength(df, period=14):
    """Composite trend strength indicator"""
    # Price momentum
    price_momentum = df['Close'].pct_change(period)
    
    # Volume trend
    volume_trend = df['Volume'].rolling(period).mean().pct_change(period)
    
    # Moving average alignment
    sma_short = SMA(df['Close'], 5)
    sma_long = SMA(df['Close'], 20)
    ma_alignment = (sma_short > sma_long).astype(int)
    
    # RSI trend
    rsi = RSI(df['Close'], period)
    rsi_trend = (rsi > 50).astype(int)
    
    # Combine all factors
    trend_strength = (price_momentum + volume_trend + ma_alignment + rsi_trend) / 4
    return trend_strength * 100  # Scale to 0-100

# -------------------------
# Stochastic Oscillator
# -------------------------

def Stochastic_Oscillator(df, period=14):
    """Returns %K and %D"""
    low = df['Low'].rolling(window=period).min()
    high = df['High'].rolling(window=period).max()
    stoch_k = ((df['Close'] - low) / (high - low)) * 100
    stoch_d = SMA(stoch_k, 3)  # Smooth %K to get %D
    return stoch_k, stoch_d

# -------------------------
# Mean Reversion
# -------------------------

def Z_Score(series, period=20):
    """Z-Score for mean reversion"""
    mean = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    z = (series - mean) / std
    return z


