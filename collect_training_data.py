import logging
import time
import pandas as pd
import numpy as np
import os
import asyncio
from iqclient import IQOptionAPI
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

API = IQOptionAPI(EMAIL, PASSWORD)

def connect_iq():
    # Run async connection
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    connected = loop.run_until_complete(API.ensure_connect())
    
    if connected:
        logger.info("✅ Connected to IQ Option successfully (via iqclient).")
        # Give it a moment to stabilize
        time.sleep(2)
    else:
        logger.error(f"❌ Connection failed.")
        exit(1)

def get_candles(asset, timeframe=60, amount=1000, endtime=None):
    """Fetches candles from IQ Option."""
    if endtime is None:
        endtime = int(time.time())
    else:
        endtime = int(endtime)

    try:
        # iqclient.get_candle_history args: asset_name, count, timeframe, end_time
        candles = API.get_candle_history(asset, count=amount, timeframe=timeframe, end_time=endtime)
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
    
    # Target Asset Only for Pattern Strategy Test
    target_asset = "EURUSD-OTC"
    
    logger.info(f"Fetching data for {target_asset} (Pattern Strategy)...")
    
    all_data = []
    end_time = int(time.time())
    TOTAL_CANDLES = 15000 # Enough for RF
    
    current_candles = []
    
    while len(current_candles) < TOTAL_CANDLES:
        candles = get_candles(target_asset, timeframe=300, amount=1000, endtime=end_time)
        if not candles: break
        
        current_candles = candles + current_candles
        end_time = candles[0]['from'] - 1
        logger.info(f"Collected {len(current_candles)} candles...")
        time.sleep(0.2)
        
    df = pd.DataFrame(current_candles)
    if 'from' in df.columns:
         df['time'] = pd.to_datetime(df['from'], unit='s')
         
    # Prepare Features (Triggering the new Pattern Logic)
    df = prepare_features(df)
    
    # Label Data
    df = label_data_binary_strategy(df)
    
    df.to_csv("training_data.csv", index=False)
    logger.info("Saved data for RF training.")

if __name__ == "__main__":
    collect_data()
