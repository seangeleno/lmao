import pandas as pd
from freqtrade.configuration.load_config import load_config_file
from freqtrade.data.history import load_pair_history
from freqtrade.enums import CandleType
from pathlib import Path

# --- Configuration ---
# Path to your Freqtrade configuration file
config_path = Path("configs/static_pairlists.json")

# Pair and timeframe you want to load
pair = "BTC/USDT"
timeframe = "5m"

# --- Load Freqtrade Configuration ---
config = load_config_file(config_path)
datadir = Path("./user_data/data/binance")
# --- Load Historical Data ---
try:
    candles = load_pair_history(
        datadir=datadir,
        timeframe=timeframe,
        pair=pair,
        data_format="feather",  # Or 'feather', 'parquet', etc. depending on your data format
        candle_type=CandleType.SPOT,  # Or CandleType.FUTURES
    )

    # --- Process the Data ---
    if not candles.empty:
        print(
            f"Successfully loaded {len(candles)} candles for {pair} on {timeframe} timeframe."
        )
        print("\nLatest 5 candles:")
        print(candles.tail())

        # Example of accessing specific columns
        print("\nClosing prices of the latest 5 candles:")
        print(candles["close"].tail())
    else:
        print(f"No data found for {pair} on {timeframe} timeframe.")

except Exception as e:
    print(f"An error occurred: {e}")
