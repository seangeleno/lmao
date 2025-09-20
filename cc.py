import os
import json
import requests
from datetime import datetime, timedelta, timezone
from dotenv import load_dotenv

blacklists = [
    "LA/USDT:USDT",
    "APT/USDT:USDT",
    "CAKE/USDT:USDT",
    "TIA/USDT:USDT",
    "ALPHA/USDT:USDT",
    "W/USDT:USDT",
    "MYX/USDT:USDT",
    "DOGE/USDT:USDT",
    "AVAX/USDT:USDT",
    "SAGA/USDT:USDT",
    "AVA/USDT:USDT",
    "SAFE/USDT:USDT",
    "STEEM/USDT:USDT",
    "CELO/USDT:USDT",
    "SCRT/USDT:USDT",
    "ANKR/USDT:USDT",
    "C98/USDT:USDT",
    "COTI/USDT:USDT",
    "HOOK/USDT:USDT",
    "VOXEL/USDT:USDT",
    "PIXEL/USDT:USDT",
    "MASK/USDT:USDT",
    "AXS/USDT:USDT",
    "LUMIA/USDT:USDT",
    "TRU/USDT:USDT",
    "SXP/USDT:USDT",
    "ALICE/USDT:USDT",
    "ZETA/USDT:USDT",
    "MTL/USDT:USDT",
    "POWR/USDT:USDT",
]

# processed blacklist to only have the symbols
blacklisted_symbols = {item.split("/")[0] for item in blacklists}


def get_binance_futures_symbols():
    """
    Fetches all symbols from Binance USDT-M Futures and returns a set of
    base assets for fast lookups. This requires no API key.
    """
    print("Fetching list of all symbols from Binance Futures...")
    try:
        url = "https://fapi.binance.com/fapi/v1/exchangeInfo"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Filter for USDT-margined perpetual contracts only
        futures_symbols = {
            item["baseAsset"]
            for item in data["symbols"]
            if item.get("quoteAsset") == "USDT"
            and item.get("contractType") == "PERPETUAL"
        }

        print(f"Found {len(futures_symbols)} unique symbols on Binance Futures.")
        return futures_symbols
    except requests.exceptions.RequestException as e:
        print(f"Error: Could not fetch data from Binance API: {e}")
        return None


def get_final_filtered_coins():
    """
    Main function to fetch, filter, and format the coin list.
    """
    # Step 1: Get the list of symbols from Binance Futures
    binance_symbols = get_binance_futures_symbols()
    if binance_symbols is None:
        # If the Binance API call fails, we cannot proceed
        return

    # --- Setup for CoinMarketCap ---
    load_dotenv()
    API_KEY = "cf18a96a-1d38-4b23-acb4-c02025f21e13"
    if not API_KEY:
        print(
            json.dumps({"error": "CMC_PRO_API_KEY not found in .env file."}, indent=2)
        )
        return

    blacklisted_tags = {
        "stablecoin",
        "wrapped-tokens",
        "weth",
        "asset-backed-stablecoin",
        "leveraged-token",
    }

    url = "https://pro-api.coinmarketcap.com/v1/cryptocurrency/listings/latest"
    parameters = {"start": "1", "limit": "600", "convert": "USD", "sort": "volume_24h"}
    headers = {
        "Accepts": "application/json",
        "X-CMC_PRO_API_KEY": API_KEY,
    }

    # Step 2: Get the main coin data from CoinMarketCap
    print("\nFetching top 1000 coins by 24h volume from CoinMarketCap...")
    try:
        response = requests.get(url, params=parameters, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.exceptions.RequestException as e:
        print(json.dumps({"error": f"CoinMarketCap API request failed: {e}"}, indent=2))
        return

    all_coins = data.get("data", [])
    if not all_coins:
        print(
            json.dumps(
                {"error": "Could not retrieve coin data from CMC API response."},
                indent=2,
            )
        )
        return

    print(f"Successfully fetched {len(all_coins)} coins. Applying all filters...")

    # --- Step 3: Apply all filters ---
    one_year_ago = datetime.now(timezone.utc) - timedelta(days=365)
    eligible_coins = []

    for coin in all_coins:
        symbol = coin.get("symbol")
        rank = coin.get("cmc_rank")
        date_added_str = coin.get("date_added")
        tags = set(coin.get("tags", []))

        is_blacklisted = not blacklisted_tags.isdisjoint(tags)

        # The final, combined filter logic
        if (
            symbol in binance_symbols
            and rank
            and rank > 100
            and date_added_str
            and not is_blacklisted
            and symbol not in blacklisted_symbols
        ):

            listing_date = datetime.fromisoformat(date_added_str.replace("Z", "+00:00"))
            if listing_date <= one_year_ago:
                eligible_coins.append(coin)

    if not eligible_coins:
        print(json.dumps([], indent=2))
        return

    # Remove duplicates based on symbol, keeping the one with the highest market cap
    unique_coins = {}
    for coin in eligible_coins:
        symbol = coin["symbol"]
        market_cap = coin["quote"]["USD"]["market_cap"]
        if (
            symbol not in unique_coins
            or market_cap > unique_coins[symbol]["quote"]["USD"]["market_cap"]
        ):
            unique_coins[symbol] = coin

    eligible_coins = list(unique_coins.values())

    # Sort the final list by market cap
    sorted_coins = sorted(eligible_coins, key=lambda x: x["quote"]["USD"]["market_cap"])

    # --- Step 4: Prepare and print JSON Output ---
    formatted_list = [f"{coin['symbol']}/USDT:USDT" for coin in sorted_coins]
    print("\n--- Final Results ---")
    print(f"Total eligible coins after all filters: {len(formatted_list)}")
    print(json.dumps(formatted_list, indent=2))

    # generate freqtrade download command
    print("\n--- Freqtrade Download Command ---")
    command = (
        "freqtrade download-data --exchange binance --trading-mode futures --erase --prepend "
        "--timerange 20250101- --pairs " + " ".join(formatted_list)
    )
    print(command)


if __name__ == "__main__":
    get_final_filtered_coins()
