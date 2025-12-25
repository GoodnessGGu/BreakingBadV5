import sqlite3
import pandas as pd
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

DB_PATH = "trades.db-1003457213931" # Using the specific DB file found in file list
OUTPUT_CSV = "training_data.csv"

def extract_data_to_csv():
    if not os.path.exists(DB_PATH):
        logger.error(f"Database file not found: {DB_PATH}")
        # Try finding any .db file
        files = [f for f in os.listdir('.') if f.endswith('.db') or '.db-' in f]
        if files:
            logger.info(f"Found other DB files: {files}. Please update script to use one of them if needed.")
        return

    try:
        conn = sqlite3.connect(DB_PATH)
        
        # Check tables
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        logger.info(f"Tables in DB: {tables}")
        
        # details, signals? We need to find where the full candle info + result is stored.
        # Often bots store results in a 'trades' or 'signals' table.
        # Let's try to read 'signals' or 'trades'
        
        target_table = None
        for t in tables:
            if 'trades' in t[0] or 'signals' in t[0]:
                target_table = t[0]
                break
        
        if not target_table:
            target_table = tables[0][0] # Fallback to first table
            
        logger.info(f"Reading from table: {target_table}")
        
        df = pd.read_sql_query(f"SELECT * FROM \"{target_table}\"", conn)
        
        if df.empty:
            logger.warning("Database table is empty.")
            return

        logger.info(f"Extracted {len(df)} rows.")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        # We need to ensure we have features + outcome.
        # If the DB only has 'win/loss', we might not have the indicators saved.
        # If indicators are NOT in the DB, we cannot train on this data 
        # (unless we have timestamps and asset names to re-download candles, which is complex).
        
        # Let's save what we have to CSV to inspect or use if it has features.
        df.to_csv(OUTPUT_CSV, index=False)
        logger.info(f"Saved to {OUTPUT_CSV}")
        
    except Exception as e:
        logger.error(f"Error extracting data: {e}")
    finally:
        if 'conn' in locals():
            conn.close()

if __name__ == "__main__":
    extract_data_to_csv()
