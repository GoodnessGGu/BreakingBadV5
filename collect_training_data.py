import logging
import time
import pandas as pd
import numpy as np
import os
from iqoptionapi.stable_api import IQ_Option
from dotenv import load_dotenv
from ml_utils import prepare_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
EMAIL = os.getenv("IQ_EMAIL")
PASSWORD = os.getenv("IQ_PASSWORD")

if not EMAIL or not PASSWORD:
    logger.error("❌ Credentials not found. Please set EMAIL and PASSWORD in .env file.")
    exit(1)

API = IQ_Option(EMAIL, PASSWORD)

def connect_iq():
    check, reason = API.connect()
    if check:
        logger.info("✅ Connected to IQ Option successfully.")
    else:
        logger.error(f"❌ Connection failed: {reason}")
        exit(1)

def get_candles(asset, timeframe=60, amount=1000):
    """Fetches candles from IQ Option."""
    try:
        # IQ Option API 'get_candles' returns list of dicts
        candles = API.get_candles(asset, timeframe, amount, time.time())
        return candles
    except Exception as e:
        logger.error(f"Error fetching candles for {asset}: {e}")
        return []

def label_data_binary_strategy(df):
    """
    Labels data based on a simple strategy or just Next Candle Color.
    For Training, simpler is often better: 
    Let's label: If next candle close > current close -> CALL WIN.
    
    Actually, to train a model to finding "Winning Conditions" for a specific strategy, 
    we usually need that strategy's triggers.
    
    However, a general "Next Candle Predictor" is also useful.
    Let's create a dataset where Outcome = 1 if the *direction matches the prediction*.
    
    But we don't have predictions yet.
    
    Standard Approach: 
    Target = 1 if Next Candle is GREEN (Close > Open).
    Target = 0 if Next Candle is RED (Close < Open).
    
    Features = Current Candle Indicators.
    
    The model will learn: "When RSI is low, Next Candle is likely Green".
    """
    
    # 1. Target: Next Candle Direction
    # Shift(-1) means looking into the future (next row)
    df['next_close'] = df['close'].shift(-1)
    df['next_open'] = df['open'].shift(-1)
    
    # Define Win Condition: Green Candle
    # If Next Close > Next Open => 1 (CALL)
    # If Next Close < Next Open => 0 (PUT)
    
    # But wait, we want to train for BOTH Call and Put?
    # Usually we train one model for "Is this a WIN?".
    # So we need to know what the signal WAS.
    
    # Let's simplify: Train the model to predict "Will the next candle go UP?".
    # > 0.5 = UP (Call), < 0.5 = DOWN (Put).
    # Then in the bot: If Model > 0.6 -> Approve Call. If Model < 0.4 -> Approve Put.
    
    # Let's try this:
    # outcome = 1 (Green), 0 (Red)
    conditions = [
        (df['next_close'] > df['next_open']), # Green
        (df['next_close'] <= df['next_open']) # Red
    ]
    choices = [1, 0]
    df['outcome'] = np.select(conditions, choices, default=np.nan)
    
    # Drop last row (NaN because of shift)
    df = df.dropna()
    
    return df

def collect_data():
    connect_iq()
    
    assets = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD", "USDCAD"] # Major pairs
    all_data = []
    
    for asset in assets:
        logger.info(f"Fetching data for {asset}...")
        candles = get_candles(asset, timeframe=60, amount=3000) # 1 minute candles
        
        if not candles:
            continue
            
        df = pd.DataFrame(candles)
        
        # Standardize columns
        # IQ Option API returns: 'id', 'from', 'at', 'to', 'open', 'close', 'min', 'max', 'volume'
        # Rename 'from' to 'time' or ensure prepare_features handles it
        if 'from' in df.columns:
            df['time'] = pd.to_datetime(df['from'], unit='s')
        
        # Prepare Features (RSI, Bollinger, etc.)
        df = prepare_features(df)
        
        # Create Target Label
        df = label_data_binary_strategy(df)
        
        # Add metadata
        df['asset'] = asset
        
        all_data.append(df)
        time.sleep(1) # Be nice to API
        
    if all_data:
        final_df = pd.concat(all_data, ignore_index=True)
        final_df.to_csv("training_data.csv", index=False)
        logger.info(f"✅ Successfully saved {len(final_df)} rows to training_data.csv")
    else:
        logger.warning("No data collected.")

if __name__ == "__main__":
    collect_data()
