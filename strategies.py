import pandas as pd
import numpy as np
import logging
from ml_utils import load_model, predict_signal, prepare_features

logger = logging.getLogger(__name__)

# Load AI Model
try:
    ai_model = load_model()
    if ai_model:
        logger.info("AI Model loaded successfully.")
    else:
        logger.warning("⚠️ No AI model found. AI filtering disabled.")
except Exception as e:
    logger.error(f"Failed to load AI model: {e}")
    ai_model = None

def reload_ai_model():
    """Reloads the AI model from disk."""
    global ai_model
    try:
        new_model = load_model()
        if new_model:
            ai_model = new_model
            logger.info("🧠 AI Model Reloaded Successfully!")
            return True
        else:
            logger.warning("⚠️ Failed to load new AI model (None returned).")
            return False
    except Exception as e:
        logger.error(f"❌ Error reloading AI model: {e}")
        return False

def wma(series, period):
    """Calculates Weighted Moving Average."""
    return series.rolling(period).apply(
        lambda x: ((x * np.arange(1, period + 1)).sum()) / np.arange(1, period + 1).sum(), 
        raw=True
    )

def analyze_strategy(candles_data, use_ai=True):
    """
    Analyzes candle data and returns a signal ('CALL', 'PUT', or None).
    Implements Bollinger Band + RSI Mean Reversion Strategy.
    """
    if not candles_data or len(candles_data) < 35:
        return None

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(candles_data)

    # Standardize timestamp column to 'time'
    if 'time' not in df.columns:
        if 'from' in df.columns:
             df['time'] = df['from']
        elif 'at' in df.columns:
             df['time'] = df['at']

    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    
    # Ensure numeric columns
    cols = ['open', 'close', 'min', 'max']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])
            
    # --- Strategy: Bollinger Band + RSI Mean Reversion ---
    # Best for OTC / Ranging Markets
    
    # 1. Bollinger Bands (20, 2)
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['std_20'] = df['close'].rolling(window=20).std()
    df['bb_upper'] = df['sma_20'] + (df['std_20'] * 2.0)
    df['bb_lower'] = df['sma_20'] - (df['std_20'] * 2.0)
    
    # 2. RSI (14)
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # 3. EMA 50 (Trend Filter)
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Get Previous Completed Candle (c1)
    curr = df.iloc[-2] 
    
    # Limits (Strict Mean Reversion)
    RSI_OVERBOUGHT = 65
    RSI_OVERSOLD = 35
    
    signal = None

    # Trend Logic: (Disabled for Volume Test)
    # is_uptrend = curr['close'] > curr['ema_50']
    # is_downtrend = curr['close'] < curr['ema_50']

    # CALL SIGNAL:
    # Rule: Price dip to Lower Band (Strict) + Oversold
    near_lower = curr['close'] <= (curr['bb_lower'] * 1.0005)
    
    if near_lower and curr['rsi'] < RSI_OVERSOLD:
        signal = "CALL"
        
    # PUT SIGNAL:
    # Rule: Price rally to Upper Band (Strict) + Overbought
    near_upper = curr['close'] >= (curr['bb_upper'] * 0.9995)
    
    if near_upper and curr['rsi'] > RSI_OVERBOUGHT:
        signal = "PUT"

    # --- AI Confirmation ---
    if signal and ai_model and use_ai:
        try:
            df_features = prepare_features(df)
            
            # We need the last row (current candle) for context
            if not df_features.empty:
                current_features = df_features.iloc[[-1]]
                prediction = predict_signal(ai_model, current_features)
                
                if prediction == 0: # 0 = Loss/Reject
                    logger.info(f"[AI] REJECTED {signal} signal on {df.iloc[-1].get('time', 'unknown')}")
                    return None
                else:
                    logger.info(f"[AI] APPROVED {signal} signal.")
        except Exception as e:
            logger.error(f"AI Prediction failed: {e}")
            pass

    return signal

def confirm_trade_with_ai(candles_data, direction):
    """
    Checks if the AI model 'approves' a trade for a given direction.
    """
    if not ai_model:
        return True 
        
    try:
        df = pd.DataFrame(candles_data)
        
        # Standardize timestamp column to 'time'
        if 'time' not in df.columns:
            if 'from' in df.columns:
                df['time'] = df['from']
            elif 'at' in df.columns:
                df['time'] = df['at']
                
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        else:
            logger.warning("Missing timestamp in candle data.")
            
        df['close'] = pd.to_numeric(df['close'])
        df['open'] = pd.to_numeric(df['open'])
        df['min'] = pd.to_numeric(df['min'])
        df['max'] = pd.to_numeric(df['max'])
        
        df_features = prepare_features(df)
        
        if df_features.empty:
            return True 
            
        current_features = df_features.iloc[[-1]]
        
        prediction = predict_signal(ai_model, current_features)
        
        if prediction == 0:
            logger.info(f"[AI] REJECTED external {direction} signal.")
            return False
            
        logger.info(f"[AI] APPROVED external {direction} signal.")
        return True
        
    except Exception as e:
        logger.error(f"AI Confirmation Error: {e}")
        return True 
    
    return None