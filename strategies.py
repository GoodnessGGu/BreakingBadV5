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

def analyze_strategy(candles_data):
    """
    Analyzes candle data using ONLY the 'New Script' logic (SMA 3, 7, 200 + Engulfing).
    No AI, No ML features, No heavy processing.
    """
    if not candles_data or len(candles_data) < 200:
        # Need enough data for SMA 200
        return None

    # Optimize: We only need the last 205 candles to calculate SMA 200 for the last few points
    # Passing 3000 candles creates a massive generic DF.
    # We'll take the whole list for now, but DF creation is fast enough for 3000 rows if we don't do complex features.
    
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
        
        # Check Strategy Conditions
        signal = None
        
        # CALL SIGNAL
        # Logic: Green, Prev Red, Engulfing, Above ALL SMAs
        if is_green and prev_is_red and is_engulfing:
            if close_curr > sma3 and close_curr > sma7 and close_curr > sma200:
                signal = "CALL"
        
        # PUT SIGNAL
        # Logic: Red, Prev Green, Engulfing, Below ALL SMAs
        elif is_red and prev_is_green and is_engulfing:
            if close_curr < sma3 and close_curr < sma7 and close_curr < sma200:
                signal = "PUT"
                
        return signal

    except Exception as e:
        logger.error(f"Strategy Error: {e}")
        return None

def confirm_trade_with_ai(candles_data, direction):
    """
    Dummy function to satisfy imports. 
    Always returns True since AI is disabled for this strategy.
    """
    return True

def reload_ai_model():
    # Placeholder to prevent import errors in main
    pass
