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

# --- Pattern Recognition Configuration (Modular) ---
PATTERN_CONFIG = {
    'engulfing': True,
    'pinbar': True,
    'star': True,
    'exhaustion': True,
    '3soldiers': False,
    'tweezer': False,
    'inside': False,
    'piercing': False,
    'marubozu': False
}

def analyze_strategy(candles_data, use_ai=True):
    """
    Analyzes candle data and returns a signal ('CALL', 'PUT', or None)
    based on Modular Candlestick Patterns + AI Confirmation.
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
            
    # --- Strategy: Modular Pattern Recognition ---
    
    # 1. Feature Engineering (Detects patterns)
    try:
        df_features = prepare_features(df)
        if df_features.empty:
            return None
        
        # Get latest fully closed candle analysis (since we trade on Close)
        # Actually pattern detection uses current row (which may be closed candle in backtest)
        # In live trading, we pass closed candles.
        curr = df_features.iloc[-1]
        
    except Exception as e:
        logger.error(f"Feature calculation failed: {e}")
        return None

    signal = None

    # 2. Iterate Configured Patterns
    # If ANY enabled pattern triggers, we take the signal.
    # Priority: If multiple patterns conflicts, we can default to None or prioritize specific ones.
    # Here: First match wins (or we can sum up scores if we wanted).
    
    triggered_pattern = None
    
    # Map pattern columns to our config keys
    # Column format: 'pattern_name' -> Config key: 'name'
    for key, enabled in PATTERN_CONFIG.items():
        if enabled:
            col_name = f'pattern_{key}'
            if col_name in curr:
                val = curr[col_name]
                if val == 1:
                    signal = "CALL"
                    triggered_pattern = key
                    break # Stop at first detected pattern
                elif val == -1:
                    signal = "PUT"
                    triggered_pattern = key
                    break # Stop at first detected pattern

    if not signal:
        return None

    # logger.info(f"Pattern Detected: {triggered_pattern} -> {signal}")

    # --- AI Confirmation ---
    if signal and ai_model and use_ai:
        try:
            # We already have df_features
            current_features = df_features.iloc[[-1]]
            
            # Encode direction for AI (1=CALL, -1=PUT)
            dir_val = 1 if signal == "CALL" else -1
            
            prediction = predict_signal(ai_model, current_features, direction=dir_val)
            
            if prediction == 0: # 0 = Loss/Reject
                logger.info(f"[AI] REJECTED {signal} ({triggered_pattern}) on {df.iloc[-1].get('time', 'unknown')}")
                return None
            else:
                logger.info(f"[AI] APPROVED {signal} ({triggered_pattern}).")
                
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
        
        # Encode direction (Assuming direction string like 'CALL' or 'PUT')
        dir_val = 1 if 'CALL' in str(direction).upper() else -1
        
        prediction = predict_signal(ai_model, current_features, direction=dir_val)
        
        if prediction == 0:
            logger.info(f"[AI] REJECTED external {direction} signal.")
            return False
            
        logger.info(f"[AI] APPROVED external {direction} signal.")
        return True
        
    except Exception as e:
        logger.error(f"AI Confirmation Error: {e}")
        return True 
    
    return None