import pandas as pd
import numpy as np
import logging
from settings import config

logger = logging.getLogger(__name__)

# --- Pattern Recognition Configuration (Dummy for imports) ---
PATTERN_CONFIG = {
    'engulfing': True,
    'pinbar': False,
    'marubozu': False
}

def analyze_strategy(candles_data, asset_name=None):
    """
    Analyzes candle data using ONLY the 'New Script' logic (SMA 3, 7, 200 + Engulfing).
    No AI, No ML features, No heavy processing.
    
    Args:
        candles_data: List of candle dicts
        asset_name: Optional asset name (e.g. "EURUSD") to Determine Real vs OTC logic.
    """
    if not candles_data or len(candles_data) < 200:
        # Need enough data for SMA 200
        return None

    try:
        # Create DataFrame
        df = pd.DataFrame(candles_data)
        
        # Standardize Columns
        cols = ['open', 'close', 'min', 'max', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])
        
        # Calculate Indicators (Lightweight)
        # Strategy: 
        # Fast MAP = 3, Slow MAP = 7, Trend MAP = 200
        # CALL: Green Candle, Prev Red, Engulfing, Close > SMA3 & SMA7 & SMA200
        # PUT: Red Candle, Prev Green, Engulfing, Close < SMA3 & SMA7 & SMA200
        
        df['sma_3'] = df['close'].rolling(window=3).mean()
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_200'] = df['close'].rolling(window=200).mean()
        
        # Get last 2 rows
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Validate SMAs exist (not NaN)
        if pd.isna(curr['sma_200']):
            return None
            
        # Parse Values
        close_curr = curr['close']
        open_curr = curr['open']
        close_prev = prev['close']
        open_prev = prev['open']
        
        sma3 = curr['sma_3']
        sma7 = curr['sma_7']
        sma200 = curr['sma_200']
        
        # Candle Colors
        is_green = close_curr > open_curr
        is_red = close_curr < open_curr
        prev_is_green = close_prev > open_prev
        prev_is_red = close_prev < open_prev
        
        # Engulfing (Body Size)
        body_curr = abs(close_curr - open_curr)
        body_prev = abs(close_prev - open_prev)
        is_engulfing = body_curr > body_prev
        
        # --- VOLUME VALIDATION (REAL MARKETS) ---
        is_volume_valid = True
        if asset_name and "-OTC" not in asset_name and "OTC" not in asset_name:
             # Logic: Current Volume > Previous Volume (Momentum confirmation)
             vol_curr = curr.get('volume', 0)
             vol_prev = prev.get('volume', 0)
             if vol_curr <= vol_prev:
                 is_volume_valid = False
                 # logger.info(f"Filtered {asset_name}: Low Volume Breakout")
        
        # Check Strategy Conditions
        signal = None
        
        if is_engulfing and is_volume_valid:
            # CALL SIGNAL
            # Logic: Green, Prev Red, Engulfing, Above ALL SMAs
            if is_green and prev_is_red:
                if close_curr > sma3 and close_curr > sma7 and close_curr > sma200:
                    signal = "CALL"
            
            # PUT SIGNAL
            # Logic: Red, Prev Green, Engulfing, Below ALL SMAs
            elif is_red and prev_is_green:
                if close_curr < sma3 and close_curr < sma7 and close_curr < sma200:
                    signal = "PUT"
                
        return signal

    except Exception as e:
        logger.error(f"Strategy Error: {e}")
        return None

from ml_utils import load_model, prepare_features, predict_signal
from ml_lstm import predict_lstm, get_model_and_scaler

# Cache for loaded models to avoid disk I/O on every tick
CACHED_MODEL = None

def reload_ai_model():
    """Forces reload of the AI model from disk."""
    global CACHED_MODEL
    CACHED_MODEL = load_model()
    logger.info("🧠 AI Model Reloaded via Strategy.")

def confirm_trade_with_ai(candles_data, direction):
    """
    Validates a signal using the trained Machine Learning Model.
    
    Args:
        candles_data (list): List of candle dictionaries (must be enough for features like RSI/SMA)
        direction (str): 'CALL' or 'PUT'
        
    Returns:
        bool: True if AI approves (or AI is disabled/failed), False if AI rejects.
    """
    global CACHED_MODEL
    
    if not candles_data:
        return True # Default to allow if no data
        
    try:
        # Convert to DataFrame
        df = pd.DataFrame(candles_data)
        
        # 1. Generate Technical Features
        # Note: prepare_features expects standard columns
        features_df = prepare_features(df)
        
        # We need the last row to predict (the current market state)
        last_row = features_df.tail(1)
        
        # 2. Select Model based on Config
        if config.model_type == "LSTM":
            # LSTM needs a sequence (e.g. last 10 candles)
            # We assume prepare_sequences logic is handled inside predict_lstm or we pass a window
            # predict_lstm expects a DataFrame window of 10 candles
            window = df.tail(10)
            if len(window) < 10:
                logger.warning("⚠️ Not enough data for LSTM (Need 10 candles). AI Skipped.")
                return True
                
            prediction = predict_lstm(window)
            is_approved = (prediction == 1)
            
            if not is_approved:
                logger.info(f"🧠 LSTM Rejected Signal: {direction}")
            else:
                logger.info(f"🧠 LSTM Approved Signal: {direction}")
                
            return is_approved
            
        else:
            # XGBoost / Random Forest
            if CACHED_MODEL is None:
                CACHED_MODEL = load_model()
                
            if CACHED_MODEL is None:
                logger.warning("⚠️ No AI Model found. Skipping AI Check.")
                return True # Fail open (allow trade)
            
            # Predict
            # 1 = Win, 0 = Loss
            decision = predict_signal(CACHED_MODEL, last_row, direction)
            
            if decision == 1:
                logger.info(f"🧠 AI Approved {direction} (Confidence High)")
                return True
            else:
                logger.info(f"🧠 AI Rejected {direction} (Low Probability)")
                return False
                
    except Exception as e:
        logger.error(f"AI Confirmation Failed: {e}", exc_info=True)
        return True # Fail open to ensure trading continues if AI bugs out

def analyze_colormillion(candles_data, asset_name=None):
    """
    Analyzes candle data using 'ColorMillion 2023' Logic.
    
    Indicators:
    - EMA 13
    - MACD (12, 26, 9)
    
    Signal Logic:
    - CALL: EMA13 > Prev EMA13  AND  MACD_Hist > Prev MACD_Hist
    - PUT:  EMA13 < Prev EMA13  AND  MACD_Hist < Prev MACD_Hist
    
    Args:
        candles_data: List of candle dicts
        asset_name: Optional, for consistency/logging
    """
    if not candles_data or len(candles_data) < 50:
         return None

    try:
        df = pd.DataFrame(candles_data)
        cols = ['open', 'close', 'min', 'max', 'volume']
        for c in cols:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c])

        # --- 1. EMA 13 ---
        df['ema_13'] = df['close'].ewm(span=13, adjust=False).mean()

        # --- 2. MACD (12, 26, 9) ---
        # Fast EMA 12
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        # Slow EMA 26
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        
        df['macd_line'] = ema_12 - ema_26
        df['signal_line'] = df['macd_line'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd_line'] - df['signal_line']

        # Get Last 2 Candles
        curr = df.iloc[-1]
        prev = df.iloc[-2]
        
        # Check if indicators are valid (not NaN)
        if pd.isna(curr['ema_13']) or pd.isna(curr['macd_hist']):
            return None

        ema_curr = curr['ema_13']
        ema_prev = prev['ema_13']
        
        hist_curr = curr['macd_hist']
        hist_prev = prev['macd_hist']
        
        signal = None
        
        # CALL Logic: Both Rising
        if ema_curr > ema_prev and hist_curr > hist_prev:
            signal = "CALL"
            
        # PUT Logic: Both Falling
        elif ema_curr < ema_prev and hist_curr < hist_prev:
            signal = "PUT"
            
        return signal

    except Exception as e:
        logger.error(f"ColorMillion Error: {e}")
        return None
