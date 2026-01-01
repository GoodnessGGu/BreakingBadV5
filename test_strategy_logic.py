import pandas as pd
import numpy as np
from ml_utils import prepare_features
from strategies import analyze_strategy

# Mock Data (250 candles to pass the 200 limit)
data = {
    'open': np.random.rand(250) * 100,
    'close': np.random.rand(250) * 100,
    'min': np.random.rand(250) * 90,
    'max': np.random.rand(250) * 110,
    'volume': np.random.rand(250) * 1000,
    'time': pd.date_range(start='2025-01-01', periods=250, freq='5min')
}
df = pd.DataFrame(data)

print("Testing Features...")
try:
    df_features = prepare_features(df)
    print("Features Calculated. Columns:", df_features.columns)
    print("SMA3:", df_features['sma_3'].iloc[-1])
    print("SMA200:", df_features['sma_200'].iloc[-1])
except Exception as e:
    print(f"Features Failed: {e}")

print("\nTesting Strategy...")
try:
    signal = analyze_strategy(df.to_dict('records'))
    print(f"Strategy Signal: {signal}")
except Exception as e:
    print(f"Strategy Failed: {e}")
