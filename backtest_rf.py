import pandas as pd
import numpy as np
import joblib
import os
import logging
from ml_utils import prepare_features, train_rf_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backtest_strategy():
    # Load Data
    if not os.path.exists("training_data.csv"):
        print("No data found.")
        return
        
    df = pd.read_csv("training_data.csv")
    
    # Ensure patterns are calculated (if data is old)
    # df = prepare_features(df) # Logic is already in CSV if collected recently
    
    # 1. Test Momentum Exhaustion Strategy (Pattern 9)
    # Logic: It was labeled as 'pattern_exhaustion' = 1 (Call) or -1 (Put)
    # The label was created in `prepare_features` inside `collect_training_data`.
    
    if 'pattern_exhaustion' not in df.columns:
        print("Pattern Exhaustion feature missing.")
        return

    print("--- Backtesting Momentum Exhaustion Strategy ---")
    
    signals = df[df['pattern_exhaustion'] != 0].copy()
    print(f"Total Signals Found: {len(signals)}")
    
    if len(signals) == 0:
        return

    # Simulate Outcomes
    # Outcome column: 1 (Next Candle Green), 0 (Next Candle Red)
    # If Signal 1 (Call) and Outcome 1 -> Win
    # If Signal -1 (Put) and Outcome 0 -> Win
    
    signals['win'] = False
    
    # Calls
    calls = signals[signals['pattern_exhaustion'] == 1]
    calls_wins = calls[calls['outcome'] == 1]
    
    # Puts
    puts = signals[signals['pattern_exhaustion'] == -1]
    puts_wins = puts[puts['outcome'] == 0]
    
    total_trades = len(signals)
    total_wins = len(calls_wins) + len(puts_wins)
    win_rate = (total_wins / total_trades * 100)
    
    print(f"Win Rate: {win_rate:.2f}% ({total_wins}/{total_trades})")
    print(f"Calls: {len(calls_wins)}/{len(calls)} ({len(calls_wins)/len(calls)*100:.1f}%)")
    print(f"Puts: {len(puts_wins)}/{len(puts)} ({len(puts_wins)/len(puts)*100:.1f}%)")
    
    # 2. Test Random Forest Filter
    # Does RF predict these specific trades correctly?
    # We trained RF on ALL data. Let's see its prediction for THESE rows.
    
    rf_model = joblib.load("models/rf_pattern_model.pkl")
    
    # Prepare features for RF
    pattern_cols = [c for c in df.columns if 'pattern_' in c]
    context_cols = ['rsi', 'dist_sma_20', 'bb_pos', 'adx']
    features = pattern_cols + context_cols
    valid_features = [c for c in features if c in df.columns]
    
    X_signals = signals[valid_features].fillna(0)
    
    rf_preds = rf_model.predict(X_signals) # Predicts 1 (Green) or 0 (Red)
    
    # RF Filtered Strategy
    # If Strategy=Call AND RF=1 (Green) -> Trade
    # If Strategy=Put AND RF=0 (Red) -> Trade
    
    signals['rf_pred'] = rf_preds
    
    filtered_calls = signals[(signals['pattern_exhaustion'] == 1) & (signals['rf_pred'] == 1)]
    filtered_puts = signals[(signals['pattern_exhaustion'] == -1) & (signals['rf_pred'] == 0)]
    
    filtered_trades = pd.concat([filtered_calls, filtered_puts])
    
    if len(filtered_trades) > 0:
        f_calls_wins = filtered_calls[filtered_calls['outcome'] == 1]
        f_puts_wins = filtered_puts[filtered_puts['outcome'] == 0]
        f_wins = len(f_calls_wins) + len(f_puts_wins)
        f_wr = (f_wins / len(filtered_trades) * 100)
        
        print("\n--- WITH RF FILTER ---")
        print(f"Filtered Trades: {len(filtered_trades)}")
        print(f"Filtered Win Rate: {f_wr:.2f}%")
    else:
        print("\nRF Filter removed all trades.")

if __name__ == "__main__":
    backtest_strategy()
