# --- Import Freqtrade Libraries ---
from typing import Optional, Union

import numpy as np
from freqtrade.strategy.interface import IStrategy
from datetime import datetime, timedelta, timezone
from freqtrade.persistence import Trade
from datetime import datetime, timedelta
from freqtrade.exchange import timeframe_to_prev_date

import pandas as pd
import logging

logger = logging.getLogger(__name__)


class pair_trading_run_V1_J_price_0811(IStrategy):
    can_short = True
    timeframe = "5m"
    startup_candle_count = 3
    process_only_new_candles = True
    # Trading parameters
    stoploss = -200
    trailing_stop = False
    use_custom_stoploss = False  # Enable dynamic stop loss
    Zscore_entry = 1  # Z-score entry threshold
    Zscore_entry1 = 1.1 * Zscore_entry  # Z-score entry threshold 1.1x
    Zscore_exit = 0.8  # Z-score exit threshold
    Zscore_stop = 3  # Z-score stop loss threshold (loss exit)
    can_trade_usdt = False
    min_consistency_score = 0.0002
    zscore_mean_window = 10  # Z-score mean calculation window
    zscore_abs_max_window = 100  # Z-score absolute maximum value calculation window
    leverage1 = 10

    # order_types = {
    #    'entry': 'limit',
    #    'exit': 'market',
    #    'emergency_exit': 'market',
    #    'force_entry': 'market',
    #    'force_exit': "market",
    #    'stoploss': 'market',
    #    'stoploss_on_exchange': False,
    #    'stoploss_on_exchange_interval': 60,
    #    'stoploss_on_exchange_market_ratio': 0.99
    # }

    def __init__(self, config):
        super().__init__(config)
        self.whitelist = config.get("exchange", {}).get("pair_whitelist", [])
        self.pvalue_dict = config.get("pvalue_dict", {})
        self.gamma_dict = config.get("gamma_dict", {})
        self.c_dict = config.get("c_dict", {})
        self.z_mean_dict = config.get("z_mean_dict", {})
        self.z_std_dict = config.get("z_std_dict", {})
        self.adfstat_dict = config.get("adfstat_dict", {})
        self.consistency_score_dict = config.get("consistency_score_dict", {})
        self.half_life_dict = config.get("half_life_dict", {})
        self.z_cross_zero_count_dict = config.get("z_cross_zero_count_dict", {})
        # Pair dictionary related
        self.all_candidate_pairs = {}
        self.tradable_pairs = {}
        self.stake_allocations = {}
        self.pair_states = {}
        ### Modification: Add a timer to update pair pool every 4 hours
        self.last_update_time = datetime(1945, 8, 15, tzinfo=timezone.utc)
        self.update_interval = timedelta(days=1)

    def _initialize_all_candidates(self) -> list:
        logger.info(len(self.pair_states))
        pvalue_filtered = {k: v for k, v in self.pvalue_dict.items() if v <= 0.0001}
        logger.info(f"Remaining pairs after p-value filtering: {len(pvalue_filtered)}")
        logger.info(f"First 5 pvalue_filtered: {list(pvalue_filtered.items())[:5]}")

        # Filter pairs based on zmean on top of pvalue_filtered
        zmean_filtered = {}
        for p_key, pvalue in pvalue_filtered.items():
            try:
                a_key, b_key = p_key.replace("_pvalue", "").split("_")
                zmean_key = f"{a_key}_{b_key}_z_mean"
                z_mean_value = self.z_mean_dict.get(zmean_key)

                # Filter based on z_mean value
                if z_mean_value is not None and abs(z_mean_value) < 0.00001:
                    zmean_filtered[p_key] = pvalue

            except Exception as e:
                logger.error(f"Error during zmean filtering: {e} for key {p_key}")
                continue
        logger.info(f"Remaining pairs after zmean filtering: {len(zmean_filtered)}")
        logger.info(f"First 5 zmean_filtered: {list(zmean_filtered.items())[:5]}")

        # Filter pairs based on z_cross_zero_count on top of zstd_filtered
        z_cross_zero_count_filtered = {}
        for p_key, pvalue in zmean_filtered.items():
            try:
                a_key, b_key = p_key.replace("_pvalue", "").split("_")
                z_cross_zero_count_key = f"{a_key}_{b_key}_z_cross_zero_count"
                z_cross_zero_count_value = self.z_cross_zero_count_dict.get(
                    z_cross_zero_count_key
                )

                # Filter based on z_cross_zero_count value
                if (
                    z_cross_zero_count_value is not None
                    and z_cross_zero_count_value >= 200
                ):
                    z_cross_zero_count_filtered[p_key] = pvalue

            except Exception as e:
                logger.error(
                    f"Error during z_cross_zero_count filtering: {e} for key {p_key}"
                )
                continue
        logger.info(
            f"Remaining pairs after z_cross_zero_count filtering: {len(z_cross_zero_count_filtered)}"
        )
        logger.info(
            f"First 5 z_cross_zero_count_filtered: {list(z_cross_zero_count_filtered.items())[:5]}"
        )
        # z_cross_zero_count_filtered = zmean_filtered

        # Filter pairs based on adf on top of z_cross_zero_count_filtered
        adfstat_filtered = {}
        for p_key, pvalue in z_cross_zero_count_filtered.items():
            try:
                a_key, b_key = p_key.replace("_pvalue", "").split("_")
                adfstat_key = f"{a_key}_{b_key}_adfstat"
                adfstat_value = self.adfstat_dict.get(adfstat_key)

                # ADF statistic needs to be less than critical value (more negative) to be considered stationary.
                # -3.0 is an empirical strict threshold.
                if adfstat_value is not None and adfstat_value < -3.0:
                    adfstat_filtered[p_key] = pvalue

            except Exception as e:
                logger.error(f"Error during adf filtering: {e} for key {p_key}")
                continue
        logger.info(f"Remaining pairs after adf filtering: {len(adfstat_filtered)}")
        logger.info(f"First 5 adfstat_filtered: {list(adfstat_filtered.items())[:5]}")

        # Filter pairs based on half_life on top of adfstat_filtered
        half_life_filtered = {}
        for p_key, pvalue in adfstat_filtered.items():
            try:
                a_key, b_key = p_key.replace("_pvalue", "").split("_")
                half_life_key = f"{a_key}_{b_key}_half_life"
                half_life_value = self.half_life_dict.get(half_life_key)

                # Filter based on half_life value
                if half_life_value is not None and half_life_value < 700:
                    half_life_filtered[p_key] = pvalue

            except Exception as e:
                logger.error(f"Error during half_life filtering: {e} for key {p_key}")
                continue
        logger.info(
            f"Remaining pairs after half_life filtering: {len(half_life_filtered)}"
        )
        logger.info(
            f"First 5 half_life_filtered: {list(half_life_filtered.items())[:5]}"
        )

        candidates = []
        max_pairs = self.config.get("max_open_trades", 1) / 2
        for p_key, pvalue in half_life_filtered.items():
            try:
                a_key, b_key = p_key.replace("_pvalue", "").split("_")
                adf_key = f"{a_key}_{b_key}_adfstat"
                adfstat = self.adfstat_dict.get(adf_key)
                a_df = self.dp.get_pair_dataframe(pair=a_key, timeframe=self.timeframe)
                if self.dp.runmode.value in ("live", "dry_run"):
                    a_price = a_df["close"].iloc[-1]
                else:
                    a_price = a_df["close"].iloc[0]

                b_df = self.dp.get_pair_dataframe(pair=b_key, timeframe=self.timeframe)
                if self.dp.runmode.value in ("live", "dry_run"):
                    b_price = b_df["close"].iloc[-1]
                else:
                    b_price = b_df["close"].iloc[0]

                gamma = self.gamma_dict.get(f"{a_key}_{b_key}_gamma")
                if a_price is None or b_price is None or gamma is None:
                    continue

                # 1. Define financial constants
                MIN_STAKE_USDT = 5.0  # Minimum order nominal value required by exchange

                # 2. Use dollar neutrality method to calculate target position value
                #    This method is the correct algorithm that matches the log price regression model
                #    Formula: Stake_B = Budget / (gamma + 1), Stake_A = Budget - Stake_B
                if (gamma + 1) <= 0:
                    continue  # Avoid division by zero or negative numbers

                target_stake_B = (
                    self.wallets.get_total(self.stake_currency)
                    / max_pairs
                    / (gamma + 1)
                )
                target_stake_A = (
                    self.wallets.get_total(self.stake_currency) / max_pairs
                    - target_stake_B
                )

                # 3. Check if position value meets minimum requirements (independent of leverage!)
                if target_stake_A < MIN_STAKE_USDT or target_stake_B < MIN_STAKE_USDT:
                    continue

                half_key = f"{a_key}_{b_key}_half_life"
                half_life = self.half_life_dict.get(half_key)

                ### Modification
                candidates.append(
                    {
                        "raw_pair_key": f"{a_key}_{b_key}",
                        "pair_A": a_key,
                        "pair_B": b_key,
                        # 'consistency_score': consistency_score,
                        "half_life": half_life,
                        "adfstat": adfstat,
                    }
                )
            except Exception as e:
                logger.error(f"Error preloading data: {e} for key {p_key}")

        # Sort by half-life in ascending order
        logger.info(f"Number of pairs before price filtering: {len(candidates)}")
        candidates = sorted(candidates, key=lambda x: x["half_life"], reverse=False)
        # Remove duplicate currencies. If A-B pair appears before, remove later B-C pairs
        seen_currencies = set()
        filtered_candidates = []
        for candidate in candidates:
            pair_A_base = candidate["pair_A"].split("/")[0]
            pair_B_base = candidate["pair_B"].split("/")[0]

            if pair_A_base in seen_currencies or pair_B_base in seen_currencies:
                continue
            seen_currencies.add(pair_A_base)
            seen_currencies.add(pair_B_base)
            filtered_candidates.append(candidate)
        # logger.info(f"Number of pairs after filtering: {len(filtered_candidates)}")
        candidates = filtered_candidates
        logger.info(f"After removing duplicate currencies: {len(candidates)}")
        return candidates
        # return candidates

    def _update_tradable_pairs(self):
        logger.warning(f"Dynamic pair pool update triggered...")
        open_trades = Trade.get_trades_proxy(is_open=True)
        new_tradable_pairs = {}
        locked_currencies = set()

        # 1. Unconditionally retain and lock existing position pairs
        for trade in open_trades:
            if not trade.enter_tag:
                continue
            try:
                trade_pair_key = "_".join(trade.enter_tag.split("_")[2:])
                if (
                    trade_pair_key in self.tradable_pairs
                    and trade_pair_key not in new_tradable_pairs
                ):
                    new_tradable_pairs[trade_pair_key] = self.tradable_pairs[
                        trade_pair_key
                    ]

                    pair_A_base = new_tradable_pairs[trade_pair_key]["pair_A"].split(
                        "/"
                    )[0]
                    pair_B_base = new_tradable_pairs[trade_pair_key]["pair_B"].split(
                        "/"
                    )[0]
                    locked_currencies.add(pair_A_base)
                    locked_currencies.add(pair_B_base)
                    # logger.info(f"Retain existing position pair {trade_pair_key}, and lock currencies: {pair_A_base}, {pair_B_base}")
            except Exception as e:
                logger.error(f"Error processing existing positions: {e}")
        logger.info(f"Number of existing position pairs: {len(new_tradable_pairs)}")
        # 2. Greedy selection to fill remaining slots
        max_allowed_pairs = (self.config.get("max_open_trades", 1)) / 2
        logger.info(
            f"Maximum allowed tradable pairs from maxopentrade: {max_allowed_pairs}"
        )
        logger.info(
            f"Number of pairs after initial filtering: {len(new_tradable_pairs)}"
        )
        logger.info(
            f"Number of pairs after filter function screening: {len(self.all_candidate_pairs)}"
        )
        for candidate in self.all_candidate_pairs:
            # logger.info(f"Check candidate pair: {candidate['raw_pair_key']} ")
            if len(new_tradable_pairs) >= max_allowed_pairs:
                logger.info(f"Reached maximum tradable pair count: {max_allowed_pairs}")
                break

            pair_A_base = candidate["pair_A"].split("/")[0]
            pair_B_base = candidate["pair_B"].split("/")[0]

            if pair_A_base in locked_currencies or pair_B_base in locked_currencies:
                continue

            pair_key_sorted = "_".join((candidate["pair_A"], candidate["pair_B"]))
            if pair_key_sorted not in new_tradable_pairs:
                new_tradable_pairs[pair_key_sorted] = {
                    "pair_A": candidate["pair_A"],
                    "pair_B": candidate["pair_B"],
                    "gamma": self.gamma_dict.get(f"{candidate['raw_pair_key']}_gamma"),
                    "c": self.c_dict.get(f"{candidate['raw_pair_key']}_c"),
                    "z_mean": self.z_mean_dict.get(
                        f"{candidate['raw_pair_key']}_z_mean"
                    ),
                    "z_std": self.z_std_dict.get(f"{candidate['raw_pair_key']}_z_std"),
                    "half_life": candidate["half_life"],
                    "adfstat": candidate["adfstat"],
                    " z_cross_zero_count": self.z_cross_zero_count_dict.get(
                        f"{candidate['raw_pair_key']}_z_cross_zero_count"
                    ),
                }
                locked_currencies.add(pair_A_base)
                locked_currencies.add(pair_B_base)
                # logger.info(f"Dynamically selected new pair {pair_key_sorted} (consistency score: {candidate['consistency_score']:.2f})")

        # 3. Atomically update tradable_pairs and fund allocation
        self.tradable_pairs = new_tradable_pairs
        logger.info(
            f"Number of tradable pairs after update: {len(self.tradable_pairs)}"
        )

        if len(self.tradable_pairs) > 0:
            capital_per_pair = self.wallets.get_total(self.stake_currency) / len(
                self.tradable_pairs
            )
            self.stake_allocations = {
                key: capital_per_pair for key in self.tradable_pairs
            }
        else:
            self.stake_allocations = {}
        # logger.info(f"Fund allocation: {self.stake_allocations}")
        logger.warning(
            f"Dynamic pair pool update completed. Current tradable pairs: {list(self.tradable_pairs.keys())}"
        )
        logger.warning(f"Current tradable pairs are: {self.tradable_pairs}")

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        current_time = dataframe.iloc[-1]["date"].to_pydatetime()
        if current_time >= self.last_update_time + self.update_interval:
            # self.free_usdt = 0.6 * self.wallets.get_total("USDT")
            # self.free_usdt = self.wallets.get_total(self.stake_currency)
            self.all_candidate_pairs = self._initialize_all_candidates()
            self._update_tradable_pairs()
            self.last_update_time = current_time

        current_pair = metadata["pair"]
        dataframe["zero"] = 0
        dataframe["2"] = 2
        dataframe["-2"] = -2
        dataframe["-1"] = -1
        dataframe["1"] = 1
        dataframe["-0.8"] = -0.8
        dataframe["0.8"] = 0.8

        # Collect all new columns
        new_columns = {}

        for pair in self.whitelist:
            pair_df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            new_columns[f"{pair}_log_close"] = pair_df["close"]

        # Calculate Z-score columns for each pair
        for pair_key_str, params in self.tradable_pairs.items():
            pair_A = params["pair_A"]
            pair_B = params["pair_B"]

            if (
                f"{pair_A}_log_close" in new_columns
                and f"{pair_B}_log_close" in new_columns
            ):
                gamma = params.get("gamma")
                c = params.get("c")
                z_mean = params.get("z_mean")
                z_std = params.get("z_std")

                if all(p is not None for p in [gamma, c, z_mean, z_std]) and z_std != 0:
                    y_series = new_columns[f"{pair_A}_log_close"]
                    x_series = new_columns[f"{pair_B}_log_close"]

                    z_value = self.zvalue(y_series, x_series, gamma, c)
                    z_score = (z_value - z_mean) / z_std

                    new_columns[f"{pair_A}_{pair_B}_Zscore"] = z_score
                    new_columns[f"{pair_A}_{pair_B}_Zscore_mean"] = z_score.rolling(
                        window=self.zscore_mean_window
                    ).mean()

        # Add all new columns at once
        if new_columns:
            new_df = pd.DataFrame(new_columns, index=dataframe.index)
            dataframe = pd.concat([dataframe, new_df], axis=1)

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        current_pair = metadata["pair"]
        candle_time = dataframe.iloc[-1]["date"]
        # pair_0, pair_1 = self.pair_0, self.pair_1

        # logger.info(f"--- [populate_entry_trend] for {current_pair} at {dataframe.iloc[-1]['date']} ---")

        for pair_key, params in self.tradable_pairs.items():
            pair_A = params["pair_A"]
            pair_B = params["pair_B"]
            zscore_col_name = f"{pair_A}_{pair_B}_Zscore"
            zscore_col_name_mean = f"{pair_A}_{pair_B}_Zscore_mean"

            if zscore_col_name in dataframe.columns and (
                current_pair == pair_A or current_pair == pair_B
            ):
                zscore = dataframe[zscore_col_name]
                zscore_mean = dataframe[zscore_col_name_mean]
                # zscore_diff = zscore - zscore.shift(1)
                # short_cond = (zscore > self.Zscore_entry) & (zscore_diff < 0)
                # long_cond = (zscore < -self.Zscore_entry) & (zscore_diff > 0)
                # short_cond = (zscore > self.Zscore_entry) & (zscore <= 1.5)
                # long_cond = (zscore < -self.Zscore_entry) & (zscore >= -1.5)
                # short_cond = (zscore > self.Zscore_entry) & (zscore.diff() < 0)
                # long_cond = (zscore < -self.Zscore_entry) & (zscore.diff() > 0)
                # short_cond = (zscore > self.Zscore_entry) &  (zscore < self.Zscore_entry1)
                # long_cond = (zscore < -self.Zscore_entry) &  (zscore > -self.Zscore_entry1)
                short_cond = zscore > self.Zscore_entry
                long_cond = zscore < -self.Zscore_entry
                # short_cond = (zscore > (1.00 * zscore_mean))  &  (zscore > 0.5)
                # long_cond = (zscore < (-1.00 * zscore_mean)) &  (zscore < -0.5)
                short_tag = f"entry_short_{pair_A}_{pair_B}"
                long_tag = f"entry_long_{pair_A}_{pair_B}"
                if current_pair == pair_A:
                    dataframe.loc[short_cond, ["enter_short", "enter_tag"]] = (
                        1,
                        short_tag,
                    )
                    dataframe.loc[long_cond, ["enter_long", "enter_tag"]] = (
                        1,
                        long_tag,
                    )
                if current_pair == pair_B:
                    dataframe.loc[short_cond, ["enter_long", "enter_tag"]] = (
                        1,
                        long_tag,
                    )
                    dataframe.loc[long_cond, ["enter_short", "enter_tag"]] = (
                        1,
                        short_tag,
                    )

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:

        current_pair = metadata["pair"]
        for pair_key, params in self.tradable_pairs.items():
            pair_A = params["pair_A"]
            pair_B = params["pair_B"]
            zscore_col_name = f"{pair_A}_{pair_B}_Zscore"
            if zscore_col_name in dataframe.columns and (
                current_pair == pair_A or current_pair == pair_B
            ):

                zscore = dataframe[zscore_col_name]
                open_trades = Trade.get_trades_proxy(is_open=True)
                # Calculate profit for pairs A and B
                trade_A = next((t for t in open_trades if t.pair == pair_A), None)
                trade_B = next((t for t in open_trades if t.pair == pair_B), None)

                profit_A = self.get_trade_profit(trade_A) if trade_A else 0.0
                profit_B = self.get_trade_profit(trade_B) if trade_B else 0.0

                # Check if profit is greater than 0
                # pair_profit_ratio = (profit_A + profit_B) / ((0.6 * self.wallets.get_total('USDT')) / 6)
                pair_profit_ratio = (profit_A + profit_B) / (
                    self.wallets.get_total(self.stake_currency)
                    / len(self.tradable_pairs)
                )
                # profit_condition = pair_profit_ratio > 0

                # exit_cond = (abs(zscore) < self.Zscore_exit) & profit_condition
                exit_cond = abs(zscore) < self.Zscore_exit
                exit_tag = f"populate_exit_trend_{pair_A}_{pair_B}_get_profit"
                dataframe.loc[exit_cond, ["exit_short", "exit_tag"]] = (1, exit_tag)
                dataframe.loc[exit_cond, ["exit_long", "exit_tag"]] = (1, exit_tag)

        return dataframe

    def custom_exit(
        self,
        pair: str,
        trade: "Trade",
        current_time: "datetime",
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        open_trades = Trade.get_trades_proxy(is_open=True)
        all_profits = {}
        total_profit_amount = 0.0
        for open_trade in open_trades:
            profit_ratio = 0.0
            if open_trade.pair == pair:
                profit_ratio = current_profit
            else:
                other_pair = open_trade.pair
                (dataframe, _) = self.dp.get_analyzed_dataframe(
                    pair=other_pair, timeframe=self.timeframe
                )
                if not dataframe.empty:
                    last_candle = dataframe.iloc[-1]
                    rate_for_other_pair = last_candle["close"]
                    if open_trade.is_short:
                        # Short position profit ratio
                        profit_ratio = 1 - (rate_for_other_pair / open_trade.open_rate)
                    else:
                        # Long position profit ratio
                        profit_ratio = (rate_for_other_pair / open_trade.open_rate) - 1
                    profit_ratio *= open_trade.leverage
            profit_pct = profit_ratio * 100
            profit_amount = open_trade.stake_amount * profit_ratio
            total_profit_amount += profit_amount
            all_profits[open_trade.pair] = (
                f"{profit_pct:.2f}% ({profit_amount:.2f} {open_trade.stake_currency})"
            )
        ratio = total_profit_amount / self.wallets.get_total("USDT")
        # logger.info(f"log_____total_profit_amount:{total_profit_amount},ratio:{ratio}")
        if ratio <= -1:
            logger.info(
                "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            )
            logger.info(
                f"Liquidity ratio at {current_time}: {ratio:.2%},strategy={self.__class__.__name__} failed "
            )
            logger.info(
                "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
            )
        self.liqutation_ratio = ratio

        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        trade_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
        trade_candle = dataframe.loc[dataframe["date"] == trade_date]
        if not trade_candle.empty:
            trade_candle = trade_candle.iloc[0]
        else:
            # No corresponding K-line found, can choose continue or return None
            return None
        current_candle = dataframe.iloc[-1].squeeze()
        current_pair = pair
        for pair_key, params in self.tradable_pairs.items():
            pair_A = params["pair_A"]
            pair_B = params["pair_B"]
            zscore_col_name = f"{pair_A}_{pair_B}_Zscore"
            # 1. Z-score exit
            if zscore_col_name in dataframe.columns and (
                current_pair == pair_A or current_pair == pair_B
            ):
                if abs(current_candle[zscore_col_name]) < 0.30 * abs(
                    trade_candle[zscore_col_name]
                ):
                    return f"Zscore_get_profit_{pair_A}_{pair_B}"

            trade_A = next((t for t in open_trades if t.pair == pair_A), None)
            trade_B = next((t for t in open_trades if t.pair == pair_B), None)
            # Update floating profit
            profit_A = self.get_trade_profit(trade_A) if trade_A else 0.0
            profit_B = self.get_trade_profit(trade_B) if trade_B else 0.0
            pair_profit_ratio = (profit_A + profit_B) / (
                (self.wallets.get_total("USDT")) / len(self.tradable_pairs)
            )
            # pair_profit_ratio = (profit_A + profit_B) / ( self.wallets.get_total('USDT') / 6)
            state = self.pair_states.get(pair_key)

            # logger.info(f"Pair {pair_key} profit ratio: {pair_profit_ratio:.2%} (A: {profit_A:.2f}, B: {profit_B:.2f})")
            if (
                pair_profit_ratio >= (0.01 * self.leverage1)
                and (current_pair == pair_A or current_pair == pair_B)
                and state.get("one_pair_is_already_exit") == 0
            ):
                return f"Pair_profit_get_profit_{pair_A}_{pair_B}"

            if (
                state
                and state.get("one_pair_is_already_exit") == 1
                and (current_pair == pair_A or current_pair == pair_B)
            ):
                return f"Pair_already_exit_{pair_A}_{pair_B}"

        return None

    def _calculate_precise_stakes(
        self, pair_key: str, budget: float, pair_A_price: float, pair_B_price: float
    ) -> dict:
        logger.info(
            f"_calculate_precise_stakes called with pair_key={pair_key}, budget={budget}"
        )
        params = self.tradable_pairs.get(pair_key)
        if not params:
            logger.info(f"log----_calculate_precise_stakes----params={params}")
            return {"A_amount": 0, "B_amount": 0}

        gamma = params["gamma"]
        pair_A = params["pair_A"]
        pair_B = params["pair_B"]

        if gamma is None:
            logger.info(f"log----_calculate_precise_stakes----gamma={gamma}")
            return {"A_amount": 0, "B_amount": 0}

        # fee = 1.0015
        fee = 1
        price_A_with_fee = pair_A_price * fee
        price_B_with_fee = pair_B_price * fee
        min_usdt = 6.0 / self.leverage1

        market_A = self.dp.market(pair_A)
        market_B = self.dp.market(pair_B)

        base_A_precision = market_A.get("precision", {}).get("amount", {})
        base_B_precision = market_B.get("precision", {}).get("amount", {})

        # logger.info(f"A_precision: {A_precision}  B_precision: {B_precision}")
        best_qty_A = 0
        best_qty_B = 0
        best_err = float("inf")

        if (price_A_with_fee + price_B_with_fee * gamma) < 0:
            logger.info(
                f"log----_calculate_precise_stakes----price_A_with_fee + price_B_with_fee * gamma={price_A_with_fee + price_B_with_fee * gamma}"
            )
            return {"A_amount": 0, "B_amount": 0}

        max_qty_A = budget / (price_A_with_fee + price_B_with_fee * gamma)

        qty_A_iterator = base_A_precision

        while qty_A_iterator <= max_qty_A + 1e-8:
            qty_A = round(qty_A_iterator, 8)
            cost_A = qty_A * price_A_with_fee
            if cost_A < min_usdt:
                qty_A_iterator += base_A_precision
                continue

            qty_B_theoretical = qty_A * gamma
            qty_B = round(qty_B_theoretical / base_B_precision) * base_B_precision
            qty_B = round(qty_B, 8)

            cost_B = qty_B * price_B_with_fee
            # logger.info(f"Checking qty_A={qty_A}, cost_A={cost_A}, cost_B={cost_B}")

            if cost_B < min_usdt:
                qty_A_iterator += base_A_precision
                continue

            total_cost = cost_A + cost_B
            if total_cost > budget:
                break

            ratio = qty_B / qty_A if qty_A > 0 else 0
            err = abs(ratio - gamma)
            # logger.info(f"err {err} ")

            if err < best_err:
                best_qty_A = qty_A
                best_qty_B = qty_B
                best_err = err
            # logger.info(f"best_err={best_err}")
            qty_A_iterator += base_A_precision

        final_stake_A = best_qty_A * pair_A_price
        final_stake_B = best_qty_B * pair_B_price

        logger.info(
            f"pair_key={pair_key}, price_A_with_fee={price_A_with_fee}, price_B_with_fee={price_B_with_fee}, gamma={gamma}"
        )
        logger.info(
            f"base_A_precision={base_A_precision}, base_B_precision={base_B_precision}"
        )

        err_pct = best_err / gamma if gamma > 0 else 0
        if best_qty_A > 0:
            logger.info(
                f"Precise position calculation for pair {pair_key}: "
                f"Budget=${budget}, "
                f"{pair_A} quantity={best_qty_A}, "
                f"{pair_B} quantity={best_qty_B}, "
                f"Final position A=${final_stake_A}, B=${final_stake_B}, "
                f"Gamma error={err_pct}"
            )

        result = {"A_amount": best_qty_A, "B_amount": best_qty_B}

        return result

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: float,
        max_stake: float,
        leverage: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:
        logger.info(
            f"----------------------------------------Buy log start---------------------------------------"
        )
        logger.info(
            f"Calculate position for pair {pair}, current time: {current_time},current_rate{current_rate}"
        )
        if not entry_tag:
            return 0.0
        try:
            pair_key = "_".join(entry_tag.split("_")[2:])
        except IndexError:
            return 0.0

        if (
            pair_key not in self.tradable_pairs
            or pair_key not in self.stake_allocations
        ):
            return 0.0

        # Get pair state, empty dict if not exists
        state = self.pair_states.get(pair_key)

        if not state:
            budget = self.stake_allocations[pair_key]
            # logger.info(f"Calculate pair {pair_key} position, budget: ${budget:.2f}")
            params = self.tradable_pairs[pair_key]
            pair_A = params["pair_A"]
            pair_B = params["pair_B"]

            dataframe_A = self.dp.get_pair_dataframe(
                pair=pair_A, timeframe=self.timeframe
            )
            dataframe_B = self.dp.get_pair_dataframe(
                pair=pair_B, timeframe=self.timeframe
            )

            if dataframe_A.empty or dataframe_B.empty:
                return 0.0

            pair_A_price = dataframe_A.iloc[-1]["close"]
            pair_B_price = dataframe_B.iloc[-1]["close"]
            logger.info(
                f"A:dataframe open{dataframe_A.iloc[-1]['open']} close:{dataframe_A.iloc[-1]['close']}"
            )
            logger.info(
                f"B:dataframe open{dataframe_B.iloc[-1]['open']} close:{dataframe_B.iloc[-1]['close']}"
            )
            logger.info(f"pair_A_price {pair_A_price} ，pair_B_price{pair_B_price}")

            stakes = self._calculate_precise_stakes(
                pair_key, budget, pair_A_price, pair_B_price
            )
            if stakes["A_amount"] <= 0 or stakes["B_amount"] <= 0:
                logger.warning(
                    f"Pair {pair_key} calculated position invalid: A leg: {stakes['A_amount']}, B leg: {stakes['B_amount']}"
                )
                return 0.0

            self.pair_states[pair_key] = {
                "stakes_calculated": True,
                "A_amount": stakes["A_amount"],
                "B_amount": stakes["B_amount"],
                "leg_A_opened": False,
                "leg_B_opened": False,
                "one_pair_is_already_exit": 0,
                "entry_tag": entry_tag,
            }

            state = self.pair_states.get(pair_key)

        pair_A = self.tradable_pairs[pair_key]["pair_A"]
        stake_to_return = (
            (state["A_amount"] * current_rate)
            if pair == pair_A
            else (state["B_amount"] * current_rate)
        )
        # logger.info(f"当前配对状态: {pair_key}: {self.pair_states[pair_key]}")
        logger.info(f"Buy amount {stake_to_return}")
        return stake_to_return

    def custom_entry_price(
        self,
        pair: str,
        current_time: datetime,
        proposed_rate: float,
        entry_tag: Optional[str],
        **kwargs,
    ) -> float:
        dataframe = self.dp.get_pair_dataframe(pair, timeframe=self.timeframe)
        pair_price = dataframe.iloc[-1]["close"]
        last_time = dataframe.iloc[-1]["date"].to_pydatetime()

        logger.info(
            f"Custom entry price function calculates buy price for pair {pair}, current time: {current_time}, previous time: {last_time}, proposed_rate: {proposed_rate}, pair_price: {pair_price}"
        )
        return pair_price

    def custom_exit_price(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        proposed_rate: float,
        current_profit: float,
        exit_tag: str | None,
        **kwargs,
    ) -> float:
        dataframe = self.dp.get_pair_dataframe(pair, timeframe=self.timeframe)
        pair_price = dataframe.iloc[-1]["close"]
        last_time = dataframe.iloc[-1]["date"].to_pydatetime()

        logger.info(
            f"Custom exit price function calculates sell price for pair {pair}, current time: {current_time}, previous time: {last_time}, proposed_rate: {proposed_rate}, pair_price: {pair_price}"
        )
        return pair_price

    def confirm_trade_entry(
        self,
        pair: str,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        side: str,
        **kwargs,
    ) -> bool:
        logger.info(
            f"Confirm buy {pair} ({side}), order type: {order_type}, quantity: {amount}, price: {rate}, time validity: {time_in_force}"
        )
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        if dataframe.empty:
            return False

        enter_tag = dataframe["enter_tag"].iloc[-1]
        # logger.info(f"确认买入{pair} ({side})")
        try:
            pair_key = "_".join(enter_tag.split("_")[2:])
        except IndexError:
            # logger.warning(f"无法从 entry_tag '{enter_tag}' 中解析出配对key。")
            return False

        state = self.pair_states.get(pair_key)
        if not state:
            return False
        params = self.tradable_pairs[pair_key]
        pair_A = params["pair_A"]
        pair_B = params["pair_B"]

        if pair == pair_A and state["leg_A_opened"] == False:
            state["leg_A_opened"] = True
            logger.info(
                f"Pair {pair_key} A leg opened successfully, state updated to: {state}"
            )
        if pair == pair_B and state["leg_B_opened"] == False:
            state["leg_B_opened"] = True
            logger.info(
                f"Pair {pair_key} B leg opened successfully, state updated to: {state}"
            )

        logger.info(
            f"Trade {pair} ({side}) opened successfully, current pair state: {state}"
        )
        logger.info(
            f"----------------------------------------Buy log end---------------------------------------"
        )
        return True

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        exit_tag: str = None,
        **kwargs,
    ) -> bool:
        logger.info(
            f"----------------------------------------Sell log start---------------------------------------"
        )
        enter_tag = trade.enter_tag
        # logger.info(f"确认买入{pair} ({side})")
        try:
            pair_key = "_".join(enter_tag.split("_")[2:])
        except IndexError:
            # logger.warning(f"无法从 entry_tag '{enter_tag}' 中解析出配对key。")
            return False

        state = self.pair_states.get(pair_key)
        if not state:
            return False
        params = self.tradable_pairs[pair_key]
        pair_A = params["pair_A"]
        pair_B = params["pair_B"]

        if pair == pair_A:
            if (
                state["leg_A_opened"] == True
                and state["leg_B_opened"] == True
                and exit_reason == f"Pair_profit_get_profit_{pair_A}_{pair_B}"
            ):
                state["one_pair_is_already_exit"] = 1
                logger.info(
                    f"-----------------------Triggered one_pair_is_already_exit set to 1---------------------"
                )
            if (
                state["leg_A_opened"] == True
                and state["leg_B_opened"] == False
                and state["one_pair_is_already_exit"] == 1
                and exit_reason == f"Pair_already_exit_{pair_A}_{pair_B}"
            ):
                logger.info(
                    f"-----------------------Triggered one_pair_is_already_exit set to 0---------------------"
                )
                state["one_pair_is_already_exit"] = 0
            state["leg_A_opened"] = False
        if pair == pair_B:
            if (
                state["leg_A_opened"] == True
                and state["leg_B_opened"] == True
                and exit_reason == f"Pair_profit_get_profit_{pair_A}_{pair_B}"
            ):
                state["one_pair_is_already_exit"] = 1
                logger.info(
                    f"-----------------------Triggered one_pair_is_already_exit set to 1---------------------"
                )
            if (
                state["leg_A_opened"] == False
                and state["leg_B_opened"] == True
                and state["one_pair_is_already_exit"] == 1
                and exit_reason == f"Pair_already_exit_{pair_A}_{pair_B}"
            ):
                logger.info(
                    f"-----------------------Triggered one_pair_is_already_exit set to 0---------------------"
                )
                state["one_pair_is_already_exit"] = 0
            state["leg_B_opened"] = False
        logger.info(
            f"Trade {trade.id} ({pair}) closed, close reason: {exit_reason}, close completed, current pair state: {state}, close quantity{amount}"
        )
        logger.info(
            f"------------- ---------------------------Sell log end---------------------------------------"
        )

        if not state["leg_A_opened"] and not state["leg_B_opened"]:
            self.pair_states.pop(pair_key, None)

        return True

    position_adjustment_enable = True

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float | None,
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ) -> float | None | tuple[float | None, str | None]:
        current_pair = trade.pair
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=current_pair, timeframe=self.timeframe
        )
        current_candle = dataframe.iloc[-1].squeeze()
        for pair_key, params in self.tradable_pairs.items():
            pair_A = params["pair_A"]
            pair_B = params["pair_B"]
            zscore_col_name = f"{pair_A}_{pair_B}_Zscore"
            # 1. Z-score position increase
            last_time = trade.date_last_filled_utc + timedelta(days=3)
            if zscore_col_name in dataframe.columns and (
                current_pair == pair_A or current_pair == pair_B
            ):
                if (
                    abs(current_candle[zscore_col_name])
                    > ((1.5 * trade.nr_of_successful_entries) * self.Zscore_entry)
                    and trade.nr_of_successful_entries < 2
                    and current_time > last_time
                ):
                    entry_tag = current_candle["enter_tag"]
                    pair = current_pair
                    if not entry_tag:
                        return 0.0
                    try:
                        pair_key = "_".join(entry_tag.split("_")[2:])
                    except IndexError:
                        return 0.0

                    if (
                        pair_key not in self.tradable_pairs
                        or pair_key not in self.stake_allocations
                    ):
                        return 0.0

                    # Get pair state, empty dict if not exists
                    state = self.pair_states.get(pair_key)

                    if not state:
                        A_amount = B_amount = 0.0
                        leg_A_opened = leg_B_opened = False
                        for ot in Trade.get_trades_proxy(is_open=True):
                            if ot.enter_tag == entry_tag:
                                if ot.pair == params["pair_A"]:
                                    A_amount, leg_A_opened = ot.stake_amount, True
                                elif ot.pair == pair_B:
                                    B_amount, leg_B_opened = ot.stake_amount, True

                        if not (leg_A_opened or leg_B_opened):
                            budget = self.stake_allocations[pair_key]
                            # logger.info(f"计算配对 {pair_key} 的仓位，预算: ${budget:.2f}")
                            params = self.tradable_pairs[pair_key]
                            pair_A = params["pair_A"]
                            pair_B = params["pair_B"]

                            dataframe_A = self.dp.get_pair_dataframe(
                                pair=pair_A, timeframe=self.timeframe
                            )
                            dataframe_B = self.dp.get_pair_dataframe(
                                pair=pair_B, timeframe=self.timeframe
                            )

                            if dataframe_A.empty or dataframe_B.empty:
                                return 0.0

                            pair_A_price = dataframe_A.iloc[-1]["open"]
                            pair_B_price = dataframe_B.iloc[-1]["open"]

                            stakes = self._calculate_precise_stakes(
                                pair_key, budget, pair_A_price, pair_B_price
                            )
                            if stakes["A_amount"] <= 0 or stakes["B_amount"] <= 0:
                                logger.warning(
                                    f"adjust_trade_position pair {pair_key} calculated position invalid: A leg: {stakes['A_amount']}, B leg: {stakes['B_amount']}"
                                )
                                return 0.0

                        self.pair_states[pair_key] = {
                            "stakes_calculated": True,
                            "A_amount": A_amount,
                            "B_amount": B_amount,
                            "leg_A_opened": leg_A_opened,
                            "leg_B_opened": leg_B_opened,
                            "one_pair_is_already_exit": 0,
                            "enter_tag": entry_tag,
                        }

                        state = self.pair_states.get(pair_key)

                    pair_A = self.tradable_pairs[pair_key]["pair_A"]
                    stake_to_return = (
                        (state["A_amount"] * current_rate)
                        if pair == pair_A
                        else (state["B_amount"] * current_rate)
                    )
                    # logger.info(f"当前配对状态: {pair_key}: {self.pair_states[pair_key]}")
                    logger.info(
                        f"Current currency {pair} pair {pair_key} position increase amount: {stake_to_return:.2f} USDT"
                    )
                    return stake_to_return

        return None

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: str,
        side: str,
        **kwargs,
    ) -> float:
        return self.leverage1

    def zvalue(self, y, x, gamma, c):
        if gamma is None or c is None:
            logger.error("gamma or c value not set, unable to calculate Z value.")
            return None

        Z = y - (gamma * x + c)
        return Z

    def Yvalue(self, y, x, gamma, c):
        if gamma is None or c is None:
            logger.error("gamma or c value not set, unable to calculate Y value.")
            return None

        Y = gamma * x + c
        return Y

    def change_y(self, y, x, gamma, c):
        """
        Calculate percentage difference relative to y value
        """
        if gamma is None or c is None:
            logger.error("gamma or c value not set, unable to calculate Z value.")
            return None

        Y = gamma * x + c
        y_change = (Y - y) / y * 100  # Calculate percentage difference relative to Y
        return y_change

    def get_trade_profit(self, trade):
        if not trade:
            return 0.0
        (dataframe, _) = self.dp.get_analyzed_dataframe(
            pair=trade.pair, timeframe=self.timeframe
        )
        if dataframe.empty:
            return 0.0
        last_candle = dataframe.iloc[-1]
        current_rate = last_candle["close"]
        if trade.is_short:
            profit_ratio = (1 - (current_rate / trade.open_rate)) * trade.leverage
        else:
            profit_ratio = ((current_rate / trade.open_rate) - 1) * trade.leverage
        profit_amount = trade.stake_amount * profit_ratio
        return profit_amount
