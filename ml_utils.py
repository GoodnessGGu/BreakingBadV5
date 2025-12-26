import pandas as pd
import numpy as np
import joblib
import os
import logging
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "trade_model.pkl")

# ... (Indicators remain unchanged) ...

def train_model(data_path="training_data.csv"):
    """
    Trains an XGBoost model using labeled data.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    logger.info("Loading data...")
    df = pd.read_csv(data_path)
    
    # Separate features (X) and target (y)
    if 'outcome' not in df.columns:
        logger.error("Data missing 'outcome' column.")
        return

    # DROP RAW COLUMNS explicitly here to force model to learn from normalized features only
    drop_raw = ['open', 'close', 'min', 'max', 'volume', 'sma_20', 'sma_50', 'bb_upper', 'bb_lower']
    
    # Also drop metadata
    drop_meta = ['time', 'outcome', 'signal', 'asset', 'from', 'to']
    
    # Combine drops
    annotated_cols = drop_raw + drop_meta
    
    X = df.drop(columns=[c for c in annotated_cols if c in df.columns])
    
    # 2. Drop NaNs from X and align y
    X = X.dropna()
    y = df.loc[X.index, 'outcome'] # Align y with cleaned X
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info("Training XGBoost Classifier...")
    
    # XGBoost Configuration
    clf = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        random_state=42,
        n_jobs=-1
    )
    
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy: {acc:.2f}")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # Save
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    joblib.dump(clf, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    return clf

# --- Indicators ---
def calculate_rsi(series, period=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(series, period=20, std_dev=2):
    """Calculates Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def calculate_adx(df, period=14):
    """Calculates ADX."""
    alpha = 1/period
    # True Range
    h_l = df['max'] - df['min']
    h_yc = abs(df['max'] - df['close'].shift(1))
    l_yc = abs(df['min'] - df['close'].shift(1))
    tr = pd.concat([h_l, h_yc, l_yc], axis=1).max(axis=1)
    
    # Directional Movement
    up = df['max'] - df['max'].shift(1)
    down = df['min'].shift(1) - df['min']
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothing
    tr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
    
    dx = 100 * abs(plus_dm_s - minus_dm_s) / (plus_dm_s + minus_dm_s)
    return dx.ewm(alpha=alpha, adjust=False).mean()

def calculate_atr(df, period=14):
    """Calculates Average True Range (Volatility)."""
    high = df['max']
    low = df['min']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculates MACD (Moving Average Convergence Divergence)."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_stochastic(df, period=14, k_period=3, d_period=3):
    """Calculates Stochastic Oscillator."""
    low_min = df['min'].rolling(window=period).min()
    high_max = df['max'].rolling(window=period).max()
    
    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    # Fast Stochastic %K
    percent_k = k
    # Smooth %D
    percent_d = percent_k.rolling(window=d_period).mean()
    return percent_k, percent_d

def calculate_cci(df, period=20):
    """Calculates Commodity Channel Index (CCI)."""
    tp = (df['max'] + df['min'] + df['close']) / 3
    sma = tp.rolling(window=period).mean()
    mad = (tp - sma).abs().rolling(window=period).mean()
    
    # Avoid division by zero
    mad = mad.replace(0, 0.001)
    
    cci = (tp - sma) / (0.015 * mad)
    return cci

def prepare_features(df):
    """
    Generates technical indicators as features for the ML model.
    """
    df = df.copy()
    
    # Ensure numeric
    cols = ['open', 'close', 'min', 'max', 'volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])
            
    if 'time' in df.columns:
        df['hour'] = df['time'].dt.hour
        # Encoding cyclical time features can be better, but raw hour is a good start
    
    # 1. Momentum & Trend (Already Relative - Good)
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['adx'] = calculate_adx(df, 14)
    df['atr'] = calculate_atr(df, 14)
    
    # 2. Moving Averages -> RELATIVE DISTANCE (%)
    # Don't feed raw 1.0500 vs 1.2000. Feed "Price is 0.5% above SMA"
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    df['dist_sma_20'] = (df['close'] - df['sma_20']) / df['close'] * 100
    df['dist_sma_50'] = (df['close'] - df['sma_50']) / df['close'] * 100
    
    # 3. Bollinger Bands -> POSITION (%)
    # Already computed relative bb_pos and bb_width, which is good.
    # bb_pos: 0 = Lower Band, 0.5 = Middle, 1 = Upper Band
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['close'], 20, 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 4. Price Action -> RELATIVE SIZE (%)
    # Body Size as % of Price (handles different asset scales)
    df['body_size_pct'] = abs(df['close'] - df['open']) / df['close'] * 100
    
    # Shadows as % of Price
    df['upper_shadow_pct'] = (df['max'] - df[['open', 'close']].max(axis=1)) / df['close'] * 100
    df['lower_shadow_pct'] = (df[['open', 'close']].min(axis=1) - df['min']) / df['close'] * 100
    
    # 5. Volatility Ratio
    # ATR as % of Price
    df['atr_pct'] = df['atr'] / df['close'] * 100

    # 5. Advanced Indicators [NEW]
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df)
    df['cci'] = calculate_cci(df)
    
    # Pattern: Engulfing (1 = Bullish, -1 = Bearish, 0 = None)
    # Bullish Engulfing: Prev Red, Curr Green, Curr Open < Prev Close, Curr Close > Prev Open
    # Bearish Engulfing: Prev Green, Curr Red, Curr Open > Prev Close, Curr Close < Prev Open
    
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    prev_high = df['max'].shift(1)
    prev_low = df['min'].shift(1)
    
    curr_open = df['open']
    curr_close = df['close']
    curr_high = df['max']
    curr_low = df['min']
    
    # Pre-calculate Candle Properties
    body_size = (curr_close - curr_open).abs()
    mean_body_size = body_size.rolling(10).mean()
    long_candle = body_size > (mean_body_size * 1.0) # Not super huge, just significant
    
    is_green = curr_close > curr_open
    is_red = curr_close < curr_open
    
    prev_is_green = prev_close > prev_open
    prev_is_red = prev_close < prev_open
    
    # 1. Engulfing
    is_bullish_engulfing = prev_is_red & is_green & (curr_open < prev_close) & (curr_close > prev_open)
    is_bearish_engulfing = prev_is_green & is_red & (curr_open > prev_close) & (curr_close < prev_open)
    
    df['pattern_engulfing'] = 0
    df.loc[is_bullish_engulfing, 'pattern_engulfing'] = 1
    df.loc[is_bearish_engulfing, 'pattern_engulfing'] = -1
    
    # 2. Pinbar (Hammer / Shooting Star)
    # Logic: Wick is 2x larger than body, and body is in top/bottom 30% of range
    total_range = curr_high - curr_low
    upper_wick = curr_high - df[['open', 'close']].max(axis=1)
    lower_wick = df[['open', 'close']].min(axis=1) - curr_low
    
    is_small_body = body_size <= (total_range * 0.3)
    
    # Bullish Pinbar (Hammer): Long Lower Wick
    is_bullish_pinbar = is_small_body & (lower_wick >= (2 * body_size)) & (lower_wick > upper_wick)
    # Bearish Pinbar (Shooting Star): Long Upper Wick
    is_bearish_pinbar = is_small_body & (upper_wick >= (2 * body_size)) & (upper_wick > lower_wick)
    
    df['pattern_pinbar'] = 0
    df.loc[is_bullish_pinbar, 'pattern_pinbar'] = 1
    df.loc[is_bearish_pinbar, 'pattern_pinbar'] = -1
    
    # 3. Marubozu (Momentum)
    # Logic: Huge body, tiny wicks
    is_marubozu = (body_size > (mean_body_size * 1.5)) & \
                  (upper_wick < (body_size * 0.1)) & \
                  (lower_wick < (body_size * 0.1))
                  
    df['pattern_marubozu'] = 0
    df.loc[is_marubozu & is_green, 'pattern_marubozu'] = 1
    df.loc[is_marubozu & is_red, 'pattern_marubozu'] = -1
    
    # 4. Three White Soldiers / Black Crows (Continuity)
    # Logic: 3 consecutive green/red candles with decent bodies
    prev2_close = df['close'].shift(2)
    prev2_open = df['open'].shift(2)
    prev2_is_green = prev2_close > prev2_open
    prev2_is_red = prev2_close < prev2_open
    
    is_3_soldiers = prev2_is_green & prev_is_green & is_green & \
                    (curr_close > prev_close) & (prev_close > prev2_close)
                    
    is_3_crows = prev2_is_red & prev_is_red & is_red & \
                 (curr_close < prev_close) & (prev_close < prev2_close)
                 
    df['pattern_3soldiers'] = 0
    df.loc[is_3_soldiers, 'pattern_3soldiers'] = 1
    df.loc[is_3_crows, 'pattern_3soldiers'] = -1
    
    # 5. Morning / Evening Star (Reversal)
    # Logic: Big Red (2 ago), Small Star (1 ago), Big Green (curr)
    star_body = (prev_close - prev_open).abs()
    prev2_body = (prev2_close - prev2_open).abs()
    
    is_morning_star = prev2_is_red & (prev2_body > mean_body_size) & \
                      (star_body < (mean_body_size * 0.5)) & \
                      is_green & (curr_close > (prev2_close + prev2_open)/2) # Closes > 50% of first candle
                      
    is_evening_star = prev2_is_green & (prev2_body > mean_body_size) & \
                      (star_body < (mean_body_size * 0.5)) & \
                      is_red & (curr_close < (prev2_close + prev2_open)/2)
                      
    df['pattern_star'] = 0
    df.loc[is_morning_star, 'pattern_star'] = 1
    df.loc[is_evening_star, 'pattern_star'] = -1
    
    # 6. Tweezer Top / Bottom (Support/Resistance)
    # Logic: Equal Highs or Lows (within very small margin)
    is_tweezer_bottom = (curr_low - prev_low).abs() < (curr_close * 0.0001)
    is_tweezer_top = (curr_high - prev_high).abs() < (curr_close * 0.0001)
    
    df['pattern_tweezer'] = 0
    df.loc[is_tweezer_bottom, 'pattern_tweezer'] = 1 # Potential Bullish Reversal
    df.loc[is_tweezer_top, 'pattern_tweezer'] = -1 # Potential Bearish Reversal
    
    # 7. Piercing Line / Dark Cloud (Weak Reversal)
    # Logic: 2nd candle opens gap, closes > 50% into previous
    midpoint_prev = (prev_open + prev_close) / 2
    
    is_piercing = prev_is_red & is_green & (curr_open < prev_close) & (curr_close > midpoint_prev)
    is_dark_cloud = prev_is_green & is_red & (curr_open > prev_close) & (curr_close < midpoint_prev)
    
    df['pattern_piercing'] = 0
    df.loc[is_piercing, 'pattern_piercing'] = 1
    df.loc[is_dark_cloud, 'pattern_piercing'] = -1
    
    # 8. Three Inside Up / Down (Confirmed Reversal)
    # Logic: Harami (Mother/Baby) -> Validation Candle
    # Candle 3 (Curr) breaks out of Candle 1 (Prev2) range
    
    prev2_high = df['max'].shift(2)
    prev2_low = df['min'].shift(2)
    
    # Harami (Prev2 = Mother, Prev = Baby)
    is_bullish_harami = prev2_is_red & prev_is_green & (prev_high < prev2_open) & (prev_low > prev2_close)
    is_bearish_harami = prev2_is_green & prev_is_red & (prev_high < prev2_close) & (prev_low > prev2_open)
    
    is_inside_up = is_bullish_harami & is_green & (curr_close > prev2_high)
    is_inside_down = is_bearish_harami & is_red & (curr_close < prev2_low)
    
    df['pattern_inside'] = 0
    df.loc[is_inside_up, 'pattern_inside'] = 1
    df.loc[is_inside_down, 'pattern_inside'] = -1

    # 5. Lagged Features (Percent Change)
    for lag in [1, 2, 3]:
        # Log Returns or Simple Returns
        df[f'return_lag_{lag}'] = df['close'].pct_change(lag) * 100
        df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
    
    # Drop rows with NaN (due to rolling windows)
    df = df.dropna()
    
    # We kept raw columns here because consumers (like collect_data) need them for labeling/logic.
    # The dropping of raw columns happens in train_model!
    
    return df



def load_model():
    """Loads the trained model."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    return None

def predict_signal(model, features_df, direction=None):
    """
    Predicts outcome for a single row of features.
    Returns: 1 (Win) or 0 (Loss)
    """
    if model is None:
        return 1 # Fallback: Assume win if no model
        
    try:
        # 1. Preserve Raw Features for Logic (ADX, SMA, etc.)
        # These columns exist in the input features_df (from prepare_features)
        raw_sma50 = features_df['sma_50'].iloc[0] if 'sma_50' in features_df.columns else None
        raw_adx = features_df['adx'].iloc[0] if 'adx' in features_df.columns else None
        raw_close = features_df['close'].iloc[0] if 'close' in features_df.columns else None
        
        # 2. Align features for Model (XGBoost)
        # We must filter to ONLY the features the model expects, and fill missing with 0
        model_input = features_df.copy()
        
        if hasattr(model, "feature_names_in_"):
            # Add missing columns as 0
            missing = set(model.feature_names_in_) - set(model_input.columns)
            for c in missing:
                model_input.loc[:, c] = 0
            
            # Select and Reorder to match model
            model_input = model_input[model.feature_names_in_]
            
        # 3. Model Prediction (Probability)
        proba = model.predict_proba(model_input)
        win_prob = proba[0][1] # Probability of Class 1 (Win)
        
        # 4. Balanced Precision Filters
        threshold = 0.65
        
        # A) Confidence Check
        if win_prob < threshold:
            return 0
            
        # B) Choppy Market Filter (ADX)
        if raw_adx is not None and raw_adx < 22:
             # logger.info(f"AI REJECT: ADX Weak ({raw_adx:.1f})")
             return 0
            
        return 1 # Approved

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 1 # Fallback
