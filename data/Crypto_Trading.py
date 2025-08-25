import ccxt as cc
import pandas as pd
import numpy as np
import indicators as ind

#fetch crypto from ccxt
def fetch_crypto(symbol,interval,limit):
    exchange=cc.binance()
    # Fix: use the exchange instance to fetch OHLCV
    ohlcv=exchange.fetch_ohlcv(symbol,interval,limit=limit)
    df=pd.DataFrame(ohlcv,columns=['timestamp','open','high','low','close','volume'])
    df['timestamp']=pd.to_datetime(df['timestamp'],unit='ms')
    df.set_index('timestamp',inplace=True)
    return df

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
def add_indicators(crypto,period):
    # Normalize columns to match indicator expectations (Title Case)
    renamed = {}
    for col in crypto.columns:
        if col.lower() == 'open':
            renamed[col] = 'Open'
        elif col.lower() == 'high':
            renamed[col] = 'High'
        elif col.lower() == 'low':
            renamed[col] = 'Low'
        elif col.lower() == 'close':
            renamed[col] = 'Close'
        elif col.lower() == 'volume':
            renamed[col] = 'Volume'
    if renamed:
        crypto = crypto.rename(columns=renamed)
    
    # Multiple SMAs for trend analysis
    crypto['SMA5']=ind.SMA(crypto['Close'],5)
    crypto['SMA10']=ind.SMA(crypto['Close'],10)
    crypto['SMA20']=ind.SMA(crypto['Close'],20)
    crypto['SMA50']=ind.SMA(crypto['Close'],50)
    crypto['SMA200']=ind.SMA(crypto['Close'],200)
    
    # EMAs for trend confirmation (include fast day-trading EMAs)
    crypto['EMA5']=ind.EMA(crypto['Close'],5)
    crypto['EMA9']=ind.EMA(crypto['Close'],9)
    crypto['EMA10']=ind.EMA(crypto['Close'],10)
    crypto['EMA20']=ind.EMA(crypto['Close'],20)
    crypto['EMA21']=ind.EMA(crypto['Close'],21)
    crypto['EMA50']=ind.EMA(crypto['Close'],50)
    
    # Momentum indicators
    crypto['RSI']=ind.RSI(crypto['Close'],period)
    crypto['RSI_5']=ind.RSI(crypto['Close'],5)  # Short-term RSI
    
    # MACD components (faster for aggressive intraday): 8/17/9
    macd_line, signal_line, histogram = ind.MACD(crypto['Close'], fast=8, slow=17, signal=9)
    crypto['MACD']=macd_line
    crypto['MACD_Signal']=signal_line
    crypto['MACD_Histogram']=histogram
    
    # Volatility indicators
    crypto['ATR']=ind.ATR(crypto, period)
    bb_upper, bb_lower = ind.Bollinger_Bands(crypto['Close'], 20, 2)
    crypto['BB_Upper'] = bb_upper
    crypto['BB_Lower'] = bb_lower
    crypto['BB_Middle'] = ind.SMA(crypto['Close'], 20)
    
    # Volume indicators - ensure Series
    volume_data = crypto['Volume']
    crypto['Volume_SMA'] = volume_data.rolling(20).mean()
    crypto['Volume_Ratio'] = volume_data / crypto['Volume_SMA']
    
    # Trend strength indicators
    adx_result = ind.ADX(crypto, period)
    crypto['ADX'] = adx_result if isinstance(adx_result, pd.Series) else pd.Series(adx_result, index=crypto.index)
    trend_strength_result = ind.Trend_Strength(crypto, period)
    crypto['Trend_Strength'] = trend_strength_result if isinstance(trend_strength_result, pd.Series) else pd.Series(trend_strength_result, index=crypto.index)
    
    # Trailing stop reference line (EMA9)
    crypto['Trail_Line'] = crypto['EMA9']
    
    return crypto

#generate buy and sell signals
def buy_sell_signals(crypto, tf_params):
    """
    Enhanced buy/sell signal generation with multiple confirmations
    NO FORWARD-LOOKING BIAS - all comparisons use historical data only
    """
    # Primary trend conditions (use EMA21 over EMA50 for faster trend, keep SMA trend context)
    primary_uptrend = (
        (crypto['SMA20'] > crypto['SMA50']) &
        (crypto['EMA9'] > crypto['EMA21']) &
        (crypto['Close'] > crypto['EMA9'])
    )
    
    primary_downtrend = (
        (crypto['SMA20'] < crypto['SMA50']) &
        (crypto['EMA9'] < crypto['EMA21']) &
        (crypto['Close'] < crypto['EMA9'])
    )
    
    # Short-term momentum (current vs previous)
    short_momentum = (
        (crypto['SMA5'] > crypto['SMA10']) &
        (crypto['EMA5'] > crypto['EMA10']) &
        (crypto['Close'] > crypto['Close'].shift(1))
    )
    
    # Volume confirmation (use timeframe parameter)
    volume_confirmation = crypto['Volume_Ratio'] > tf_params['volume_threshold']
    
    # RSI conditions (use timeframe parameters)
    rsi_oversold = crypto['RSI'] < tf_params['rsi_oversold']
    rsi_overbought = crypto['RSI'] > tf_params['rsi_overbought']
    rsi_bullish_divergence = (crypto['RSI'] > crypto['RSI'].shift(1)) & (crypto['Close'] < crypto['Close'].shift(1))
    
    # MACD conditions (fast MACD already computed)
    macd_bullish = (
        (crypto['MACD'] > crypto['MACD_Signal']) &
        (crypto['MACD_Histogram'] > crypto['MACD_Histogram'].shift(1)) &
        (crypto['MACD'] > crypto['MACD'].shift(1))
    )
    
    macd_bearish = (
        (crypto['MACD'] < crypto['MACD_Signal']) &
        (crypto['MACD_Histogram'] < crypto['MACD_Histogram'].shift(1)) &
        (crypto['MACD'] < crypto['MACD'].shift(1))
    )
    
    # Bollinger Band conditions (current values only)
    bb_squeeze = (crypto['BB_Upper'] - crypto['BB_Lower']) < (crypto['BB_Upper'] - crypto['BB_Lower']).rolling(20).mean()
    bb_oversold = crypto['Close'] < crypto['BB_Lower'] * 1.02
    bb_overbought = crypto['Close'] > crypto['BB_Upper'] * 0.98
    
    # Trend strength confirmation (use timeframe parameter)
    strong_trend = crypto['ADX'] > tf_params['trend_strength']
    trend_strength_high = crypto['Trend_Strength'] > 60
    
    # ATR volatility filter (current vs historical)
    low_volatility = crypto['ATR'] < crypto['ATR'].rolling(20).mean()
    
    # Price momentum (current vs previous)
    price_momentum_up = crypto['Close'] > crypto['Close'].shift(1)
    price_momentum_down = crypto['Close'] < crypto['Close'].shift(1)
    
    # Moving average momentum (current vs previous)
    sma_momentum = crypto['SMA5'] > crypto['SMA5'].shift(1)
    ema_momentum = crypto['EMA5'] > crypto['EMA5'].shift(1)
    
    # Buy Signal - Multiple confirmations required (NO future bias)
    buy_signal = (
        primary_uptrend &
        short_momentum &
        (rsi_oversold | rsi_bullish_divergence) &
        macd_bullish &
        volume_confirmation &
        strong_trend &
        (bb_oversold | bb_squeeze) &
        price_momentum_up &
        sma_momentum
    )
    
    # Sell Signal - Multiple confirmations required (NO future bias)
    sell_signal = (
        primary_downtrend &
        (~short_momentum) &
        (rsi_overbought | (~rsi_bullish_divergence)) &
        macd_bearish &
        (volume_confirmation | bb_overbought) &
        (strong_trend | trend_strength_high) &
        price_momentum_down &
        (~sma_momentum)
    )
    
    # ATR-based stop and take profit suggestions and trailing stop
    # Trail line is EMA9; stop if close crosses below trail for longs or above for shorts
    trail_stop_long = crypto['Close'] < crypto['Trail_Line']
    trail_stop_short = crypto['Close'] > crypto['Trail_Line']
    
    # Suggested static ATR bands (not using entry price here; for reference only)
    crypto['ATR_Stop_Long_Level'] = crypto['Close'] - (0.8 * crypto['ATR'])
    crypto['ATR_TP_Long_Level'] = crypto['Close'] + (1.2 * crypto['ATR'])
    crypto['ATR_Stop_Short_Level'] = crypto['Close'] + (0.8 * crypto['ATR'])
    crypto['ATR_TP_Short_Level'] = crypto['Close'] - (1.2 * crypto['ATR'])
    
    # Exit signals
    stop_loss = (
        (crypto['Close'] < crypto['SMA20'] - 0.8 * crypto['ATR']) |
        trail_stop_long
    )
    
    take_profit = (
        (crypto['RSI'] > (tf_params['rsi_overbought'] + 5)) &
        (crypto['MACD_Histogram'] < crypto['MACD_Histogram'].shift(1))
    )
    
    # Separate bearish signals from exit signals
    bearish_signal = (
        (crypto['Close'] < crypto['SMA20']) &
        (crypto['RSI'] > tf_params['rsi_overbought']) &
        (crypto['MACD'] < crypto['MACD_Signal'])
    )
    
    # Assign signals to dataframe
    crypto['Buy_Signal'] = buy_signal
    crypto['Sell_Signal'] = sell_signal
    crypto['Stop_Loss'] = stop_loss
    crypto['Take_Profit'] = take_profit
    crypto['Bearish_Signal'] = bearish_signal
    crypto['Trail_Stop_Long'] = trail_stop_long
    crypto['Trail_Stop_Short'] = trail_stop_short
    
    # Signal strength (0-100) - adjusted weights for aggressive trading
    crypto['Signal_Strength'] = (
        (primary_uptrend.astype(int) * 15) +
        (short_momentum.astype(int) * 20) +
        (macd_bullish.astype(int) * 20) +
        (volume_confirmation.astype(int) * 15) +
        (strong_trend.astype(int) * 10) +
        (rsi_oversold.astype(int) * 10) +
        (bb_oversold.astype(int) * 5) +
        (price_momentum_up.astype(int) * 15)
    )
    
    # Create different signal strictness levels AFTER Signal_Strength is created
    conservative_buy = buy_signal & (crypto['Signal_Strength'] >= 80)
    conservative_sell = sell_signal & (crypto['Signal_Strength'] >= 80)
    moderate_buy = buy_signal & (crypto['Signal_Strength'] >= 60)
    moderate_sell = sell_signal & (crypto['Signal_Strength'] >= 60)
    aggressive_buy = buy_signal & (crypto['Signal_Strength'] >= 40)
    aggressive_sell = sell_signal & (crypto['Signal_Strength'] >= 40)
    
    # Different strictness levels
    crypto['Conservative_Buy'] = conservative_buy
    crypto['Conservative_Sell'] = conservative_sell
    crypto['Moderate_Buy'] = moderate_buy
    crypto['Moderate_Sell'] = moderate_sell
    crypto['Aggressive_Buy'] = aggressive_buy
    crypto['Aggressive_Sell'] = aggressive_sell
    
    return crypto

symbol = 'BTC/USDT'
interval = '1m'

data_frame=fetch_crypto(symbol,interval=interval,limit=1000)

# Get timeframe-specific parameters
tf_params = get_timeframe_parameters(interval)
print(f"Using timeframe parameters: {tf_params}")

data_frame=add_indicators(data_frame,period=14)
data_frame=buy_sell_signals(data_frame, tf_params)

# Display results
print(f"=== {symbol} Crypto Analysis ({interval}) ===")
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
    print(f"\n CONSERVATIVE BUY SIGNAL - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    print("   High confidence entry - recommended for swing trading")
    signal_found = True
elif latest_data['Moderate_Buy']:
    print(f"\n MODERATE BUY SIGNAL - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    print("   Balanced confidence - good for day trading")
    signal_found = True
elif latest_data['Aggressive_Buy']:
    print(f"\n AGGRESSIVE BUY SIGNAL - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    print("   Lower confidence - use with tight stops")
    signal_found = True
elif latest_data['Conservative_Sell']:
    print(f"\n CONSERVATIVE SELL SIGNAL - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    print("   High confidence exit - recommended")
    signal_found = True
elif latest_data['Moderate_Sell']:
    print(f"\n MODERATE SELL SIGNAL - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    print("   Balanced confidence exit")
    signal_found = True
elif latest_data['Aggressive_Sell']:
    print(f"\n AGGRESSIVE SELL SIGNAL - Signal Strength: {latest_data['Signal_Strength']:.1f}")
    print("   Lower confidence exit - consider partial positions")
    signal_found = True

if latest_data['Stop_Loss']:
    print(f"\n  STOP LOSS TRIGGERED")
    signal_found = True
elif latest_data['Take_Profit']:
    print(f"\n TAKE PROFIT TRIGGERED")
    signal_found = True

if not signal_found:
    print(f"\n No active signals - Signal Strength: {latest_data['Signal_Strength']:.1f}")
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

# Save enhanced data with safe filename
safe_symbol = symbol.replace('/', '-')
data_frame.to_csv(f'{safe_symbol}_{interval}_enhanced.csv')
print(f"\nEnhanced data saved to: {safe_symbol}_{interval}_enhanced.csv")

