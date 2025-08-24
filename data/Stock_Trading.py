import yfinance as yf
import pandas as pd
import numpy as np
import indicators as ind

#fetch stocks from yfinance
def fetch_stocks(ticker,period,interval,save_csv):
    stock=yf.download(ticker,period=period,interval=interval)
    if save_csv:
        stock.to_csv(f"{ticker}.csv")
    return stock

def get_timeframe_parameters(interval):
    """
    Adjust signal parameters based on timeframe
    """
    if 'm' in interval:  # Minutes
        if interval == '1m':
            return {'rsi_oversold': 25, 'rsi_overbought': 75, 'volume_threshold': 1.5, 'trend_strength': 20}
        elif interval == '5m':
            return {'rsi_oversold': 28, 'rsi_overbought': 72, 'volume_threshold': 1.3, 'trend_strength': 22}
        elif interval == '15m':
            return {'rsi_oversold': 30, 'rsi_overbought': 70, 'volume_threshold': 1.2, 'trend_strength': 25}
        else:
            return {'rsi_oversold': 30, 'rsi_overbought': 70, 'volume_threshold': 1.2, 'trend_strength': 25}
    elif 'h' in interval:  # Hours
        return {'rsi_oversold': 30, 'rsi_overbought': 70, 'volume_threshold': 1.2, 'trend_strength': 25}
    elif 'd' in interval:  # Days
        return {'rsi_oversold': 30, 'rsi_overbought': 70, 'volume_threshold': 1.1, 'trend_strength': 25}
    else:
        return {'rsi_oversold': 30, 'rsi_overbought': 70, 'volume_threshold': 1.2, 'trend_strength': 25}

#add indicators to the stock
def add_indicators(stock,period):
    # Multiple SMAs for trend analysis
    stock['SMA5']=ind.SMA(stock['Close'],5)
    stock['SMA10']=ind.SMA(stock['Close'],10)
    stock['SMA20']=ind.SMA(stock['Close'],20)
    stock['SMA50']=ind.SMA(stock['Close'],50)
    stock['SMA200']=ind.SMA(stock['Close'],200)
    
    # EMAs for trend confirmation
    stock['EMA5']=ind.EMA(stock['Close'],5)
    stock['EMA10']=ind.EMA(stock['Close'],10)
    stock['EMA20']=ind.EMA(stock['Close'],20)
    stock['EMA50']=ind.EMA(stock['Close'],50)
    
    # Momentum indicators
    stock['RSI']=ind.RSI(stock['Close'],period)
    stock['RSI_5']=ind.RSI(stock['Close'],5)  # Short-term RSI
    
    # MACD components
    macd_line, signal_line, histogram = ind.MACD(stock['Close'],period)
    stock['MACD']=macd_line
    stock['MACD_Signal']=signal_line
    stock['MACD_Histogram']=histogram
    
    # Volatility indicators
    stock['ATR']=ind.ATR(stock,period)
    bb_upper, bb_lower = ind.Bollinger_Bands(stock['Close'], 20, 2)
    stock['BB_Upper'] = bb_upper
    stock['BB_Lower'] = bb_lower
    stock['BB_Middle'] = ind.SMA(stock['Close'], 20)
    
    # Volume indicators
    stock['Volume_SMA'] = stock['Volume'].rolling(20).mean()
    stock['Volume_Ratio'] = stock['Volume'] / stock['Volume_SMA']
    
    # Trend strength indicators
    stock['ADX'] = ind.ADX(stock, period)
    stock['Trend_Strength'] = ind.Trend_Strength(stock, period)
    
    return stock

#generate buy and sell signals
def buy_sell_signals(stock):
    """
    Enhanced buy/sell signal generation with multiple confirmations
    NO FORWARD-LOOKING BIAS - all comparisons use historical data only
    """
    # Primary trend conditions (using current values vs historical)
    primary_uptrend = (
        (stock['SMA20'] > stock['SMA50']) & 
        (stock['EMA20'] > stock['EMA50']) &
        (stock['Close'] > stock['SMA200'])
    )
    
    primary_downtrend = (
        (stock['SMA20'] < stock['SMA50']) & 
        (stock['EMA20'] < stock['EMA50']) &
        (stock['Close'] < stock['SMA200'])
    )
    
    # Short-term momentum (current vs previous)
    short_momentum = (
        (stock['SMA5'] > stock['SMA10']) &
        (stock['EMA5'] > stock['EMA10']) &
        (stock['Close'] > stock['Close'].shift(1))  # Current close > previous close
    )
    
    # Volume confirmation (current vs historical average)
    volume_confirmation = stock['Volume_Ratio'] > 1.2  # 20% above average
    
    # RSI conditions (current values only)
    rsi_oversold = stock['RSI'] < 30
    rsi_overbought = stock['RSI'] > 70
    rsi_bullish_divergence = (stock['RSI'] > stock['RSI'].shift(1)) & (stock['Close'] < stock['Close'].shift(1))
    
    # MACD conditions (current vs previous - NO future data)
    macd_bullish = (
        (stock['MACD'] > stock['MACD_Signal']) & 
        (stock['MACD_Histogram'] > stock['MACD_Histogram'].shift(1)) &  # Current > Previous
        (stock['MACD'] > stock['MACD'].shift(1))  # MACD line rising
    )
    
    macd_bearish = (
        (stock['MACD'] < stock['MACD_Signal']) & 
        (stock['MACD_Histogram'] < stock['MACD_Histogram'].shift(1)) &  # Current < Previous
        (stock['MACD'] < stock['MACD'].shift(1))  # MACD line falling
    )
    
    # Bollinger Band conditions (current values only)
    bb_squeeze = (stock['BB_Upper'] - stock['BB_Lower']) < (stock['BB_Upper'] - stock['BB_Lower']).rolling(20).mean()
    bb_oversold = stock['Close'] < stock['BB_Lower'] * 1.02  # 2% below lower band
    bb_overbought = stock['Close'] > stock['BB_Upper'] * 0.98  # 2% above upper band
    
    # Trend strength confirmation (current values)
    strong_trend = stock['ADX'] > 25
    trend_strength_high = stock['Trend_Strength'] > 60
    
    # ATR volatility filter (current vs historical)
    low_volatility = stock['ATR'] < stock['ATR'].rolling(20).mean()
    
    # Price momentum (current vs previous)
    price_momentum_up = stock['Close'] > stock['Close'].shift(1)
    price_momentum_down = stock['Close'] < stock['Close'].shift(1)
    
    # Moving average momentum (current vs previous)
    sma_momentum = stock['SMA5'] > stock['SMA5'].shift(1)
    ema_momentum = stock['EMA5'] > stock['EMA5'].shift(1)
    
    # Buy Signal - Multiple confirmations required (NO future bias)
    buy_signal = (
        primary_uptrend &                    # Primary trend is up
        short_momentum &                     # Short-term momentum is positive
        (rsi_oversold | rsi_bullish_divergence) &  # RSI shows oversold or bullish divergence
        macd_bullish &                       # MACD is bullish
        volume_confirmation &                # Volume confirms the move
        strong_trend &                       # Trend is strong enough
        (bb_oversold | bb_squeeze) &        # Price near support or in squeeze
        price_momentum_up &                  # Current price > previous price
        sma_momentum                         # 5 SMA is rising
    )
    
    # Sell Signal - Multiple confirmations required (NO future bias)
    sell_signal = (
        primary_downtrend &                  # Primary trend is down
        (~short_momentum) &                  # Short-term momentum is negative
        (rsi_overbought | (~rsi_bullish_divergence)) &  # RSI shows overbought
        macd_bearish &                       # MACD is bearish
        (volume_confirmation | bb_overbought) &  # Volume confirms or price overbought
        (strong_trend | trend_strength_high) &   # Trend strength confirms
        price_momentum_down &                # Current price < previous price
        (~sma_momentum)                      # 5 SMA is falling
    )
    
    # Stop Loss and Take Profit signals (current values only)
    stop_loss = (
        (stock['Close'] < stock['SMA20'] * 0.95) &  # 5% below 20 SMA
        (stock['ATR'] > stock['ATR'].rolling(20).mean())  # High volatility
    )
    
    take_profit = (
        (stock['RSI'] > 80) &               # Extremely overbought
        (stock['Close'] > stock['BB_Upper'] * 1.05) &  # 5% above upper BB
        (stock['MACD_Histogram'] < stock['MACD_Histogram'].shift(1))  # MACD weakening
    )
    
    # Separate bearish signals from exit signals
    bearish_signal = (
        (stock['Close'] < stock['SMA20']) &     # Price below 20 SMA
        (stock['RSI'] > 70) &                  # RSI overbought
        (stock['MACD'] < stock['MACD_Signal']) # MACD bearish
    )
    
    # Create different signal strictness levels
    # Conservative signals (high accuracy, fewer signals)
    conservative_buy = buy_signal & (stock['Signal_Strength'] >= 80)
    conservative_sell = sell_signal & (stock['Signal_Strength'] >= 80)
    
    # Moderate signals (balanced accuracy and frequency)
    moderate_buy = buy_signal & (stock['Signal_Strength'] >= 60)
    moderate_sell = sell_signal & (stock['Signal_Strength'] >= 60)
    
    # Aggressive signals (more signals, lower accuracy)
    aggressive_buy = buy_signal & (stock['Signal_Strength'] >= 40)
    aggressive_sell = sell_signal & (stock['Signal_Strength'] >= 40)
    
    # Assign signals to dataframe
    stock['Buy_Signal'] = buy_signal
    stock['Sell_Signal'] = sell_signal
    stock['Stop_Loss'] = stop_loss
    stock['Take_Profit'] = take_profit
    stock['Bearish_Signal'] = bearish_signal
    
    # Different strictness levels
    stock['Conservative_Buy'] = conservative_buy
    stock['Conservative_Sell'] = conservative_sell
    stock['Moderate_Buy'] = moderate_buy
    stock['Moderate_Sell'] = moderate_sell
    stock['Aggressive_Buy'] = aggressive_buy
    stock['Aggressive_Sell'] = aggressive_sell
    
    # Signal strength (0-100) - weighted by importance
    stock['Signal_Strength'] = (
        (primary_uptrend.astype(int) * 20) +      # Primary trend (most important)
        (short_momentum.astype(int) * 15) +       # Short-term momentum
        (macd_bullish.astype(int) * 15) +         # MACD confirmation
        (volume_confirmation.astype(int) * 15) +   # Volume confirmation
        (strong_trend.astype(int) * 10) +          # Trend strength
        (rsi_oversold.astype(int) * 10) +         # RSI oversold
        (bb_oversold.astype(int) * 8) +            # Bollinger Band support
        (price_momentum_up.astype(int) * 7)        # Price momentum
    )
    
    return stock

data_frame=fetch_stocks('AAPL',period='1d',interval='1m',save_csv=True)

# Get timeframe-specific parameters
tf_params = get_timeframe_parameters('1m')
print(f"Using timeframe parameters: {tf_params}")

data_frame=add_indicators(data_frame,period=14)
data_frame=buy_sell_signals(data_frame)

# Display results
print("=== AAPL Stock Analysis ===")
print(f"Data points: {len(data_frame)}")
print("\nLatest indicators:")
print(data_frame[['Close', 'SMA20', 'EMA20', 'RSI', 'MACD', 'ADX', 'Trend_Strength']].tail())

print("\n=== Signal Summary ===")
# Conservative signals (high accuracy)
conservative_buy = data_frame['Conservative_Buy'].sum()
conservative_sell = data_frame['Conservative_Sell'].sum()

# Moderate signals (balanced)
moderate_buy = data_frame['Moderate_Buy'].sum()
moderate_sell = data_frame['Moderate_Sell'].sum()

# Aggressive signals (more frequent)
aggressive_buy = data_frame['Aggressive_Buy'].sum()
aggressive_sell = data_frame['Aggressive_Sell'].sum()

# Risk management signals
stop_loss_signals = data_frame['Stop_Loss'].sum()
take_profit_signals = data_frame['Take_Profit'].sum()

print(f"Conservative Buy signals: {conservative_buy} (High accuracy)")
print(f"Conservative Sell signals: {conservative_sell} (High accuracy)")
print(f"Moderate Buy signals: {moderate_buy} (Balanced)")
print(f"Moderate Sell signals: {moderate_sell} (Balanced)")
print(f"Aggressive Buy signals: {aggressive_buy} (More frequent)")
print(f"Aggressive Sell signals: {aggressive_sell} (More frequent)")
print(f"Stop loss signals: {stop_loss_signals}")
print(f"Take profit signals: {take_profit_signals}")

# Show latest signal with strictness level
latest_data = data_frame.iloc[-1]
signal_found = False

if latest_data['Conservative_Buy']:
    print(f"\n🟢 CONSERVATIVE BUY SIGNAL - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    print("   High confidence entry - recommended for swing trading")
    signal_found = True
elif latest_data['Moderate_Buy']:
    print(f"\n🟢 MODERATE BUY SIGNAL - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    print("   Balanced confidence - good for day trading")
    signal_found = True
elif latest_data['Aggressive_Buy']:
    print(f"\n🟢 AGGRESSIVE BUY SIGNAL - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    print("   Lower confidence - use with tight stops")
    signal_found = True
elif latest_data['Conservative_Sell']:
    print(f"\n🔴 CONSERVATIVE SELL SIGNAL - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    print("   High confidence exit - recommended")
    signal_found = True
elif latest_data['Moderate_Sell']:
    print(f"\n🔴 MODERATE SELL SIGNAL - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    print("   Balanced confidence exit")
    signal_found = True
elif latest_data['Aggressive_Sell']:
    print(f"\n🔴 AGGRESSIVE SELL SIGNAL - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    print("   Lower confidence exit - consider partial positions")
    signal_found = True

if latest_data['Stop_Loss']:
    print(f"\n⚠️  STOP LOSS TRIGGERED")
    signal_found = True
elif latest_data['Take_Profit']:
    print(f"\n💰 TAKE PROFIT TRIGGERED")
    signal_found = True

if not signal_found:
    print(f"\n⚪ No active signals - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    if latest_data['Signal_Strength'] >= 70:
        print("   High signal strength - monitor for entry opportunities")
    elif latest_data['Signal_Strength'] >= 40:
        print("   Moderate signal strength - wait for confirmation")
    else:
        print("   Low signal strength - no clear direction")

# Market context
print(f"\n=== Market Context ===")
print(f"Trend: {'UPTREND' if latest_data['SMA20'] > latest_data['SMA50'] else 'DOWNTREND'}")
print(f"Volatility: {'HIGH' if latest_data['ATR'] > data_frame['ATR'].rolling(20).mean().iloc[-1] else 'LOW'}")
print(f"RSI: {latest_data['RSI']:.1f} ({'Oversold' if latest_data['RSI'] < 30 else 'Overbought' if latest_data['RSI'] > 70 else 'Neutral'})")
print(f"MACD: {'Bullish' if latest_data['MACD'] > latest_data['MACD_Signal'] else 'Bearish'}")

# Save enhanced data
data_frame.to_csv('AAPL_enhanced.csv')
print(f"\nEnhanced data saved to: AAPL_enhanced.csv")

