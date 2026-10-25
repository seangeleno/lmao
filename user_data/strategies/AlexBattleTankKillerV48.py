import logging
import sys
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import talib.abstract as ta

import freqtrade.vendor.qtpylib.indicators as qtpylib  # noqa: F401  (kept for convenience)
from freqtrade.exchange import timeframe_to_minutes
from freqtrade.persistence import Order, Trade
from freqtrade.strategy import IStrategy, merge_informative_pair

# The hyperopt parameters live in a sibling module. freqtrade only puts the strategy
# folder on sys.path while it imports the strategy, so add it permanently to keep the
# import working for tooling that re-imports later (plot-dataframe, lookahead-analysis...).
_STRATEGY_DIR = str(Path(__file__).resolve().parent)
if _STRATEGY_DIR not in sys.path:
    sys.path.insert(0, _STRATEGY_DIR)
from AlexBattleTankKillerV48_params import AlexBattleTankKillerParams, attach_hyperopt_params  # noqa: E402

warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)
logger = logging.getLogger(__name__)

# Define Murrey Math level names for consistency
MML_LEVEL_NAMES = [
    "[-3/8]P", "[-2/8]P", "[-1/8]P", "[0/8]P", "[1/8]P",
    "[2/8]P", "[3/8]P", "[4/8]P", "[5/8]P", "[6/8]P",
    "[7/8]P", "[8/8]P", "[+1/8]P", "[+2/8]P", "[+3/8]P"
]
# Index of each level in the 16-step ladder ml[k] = mml_val - k * dmml
_MML_LADDER_INDEX = {
    "[-3/8]P": 14, "[-2/8]P": 13, "[-1/8]P": 12,
    "[0/8]P": 11, "[1/8]P": 10, "[2/8]P": 9,
    "[3/8]P": 8, "[4/8]P": 7, "[5/8]P": 6,
    "[6/8]P": 5, "[7/8]P": 4, "[8/8]P": 3,
    "[+1/8]P": 2, "[+2/8]P": 1, "[+3/8]P": 0,
}

# Market-context basket: (base currency, weight)
BREADTH_BASKET = [("BTC", 0.40), ("ETH", 0.25), ("BNB", 0.10), ("SOL", 0.08),
                  ("XRP", 0.07), ("ADA", 0.05), ("AVAX", 0.02)]
MCAP_BASKET = [("BTC", 0.4), ("ETH", 0.2), ("BNB", 0.1), ("SOL", 0.05), ("ADA", 0.05)]


def calculate_minima_maxima(df, window):
    """
    Marks candles whose ha_close is the unique min (-window) / max (+window) of the
    trailing window+1 candles. Vectorised; identical output to the V48 python loop
    but ~100x faster.
    """
    if df is None or df.empty:
        return np.zeros(0), np.zeros(0)

    n = len(df)
    minima = np.zeros(n)
    maxima = np.zeros(n)
    width = window + 1
    if n < width:
        return minima, maxima

    values = df["ha_close"].to_numpy(dtype=float)
    windows = np.lib.stride_tricks.sliding_window_view(values, width)  # row k -> candle k+window
    current = values[window:]
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        w_min = np.nanmin(windows, axis=1)
        w_max = np.nanmax(windows, axis=1)
    occurrences = (windows == current[:, None]).sum(axis=1)
    unique = occurrences == 1
    minima[window:] = np.where((current == w_min) & unique, -window, 0)
    maxima[window:] = np.where((current == w_max) & unique, window, 0)
    return minima, maxima


@attach_hyperopt_params(AlexBattleTankKillerParams)
class AlexBattleTankKillerV4(IStrategy):
    """
    Enhanced strategy with Market Correlation Filters.
    Hyperopt parameters live in AlexBattleTankKillerV48_params.py (AlexBattleTankKillerParams)
    and are attached by the decorator above - edit ranges/defaults there, not here.

    V48 -> V49 bug-hunt summary (details in the accompanying changelog):
      - Market context (BTC filter, breadth, market-cap trend, legacy correlation, BTC regime)
        is now computed per candle and merged by date. V48 broadcast the *latest* value over
        the whole history, which is lookahead bias in backtests/hyperopt.
      - Murrey Math levels computed for every candle (V48 interpolated between every 5th
        candle using future values = lookahead; its cache never hit).
      - Market-regime entry filter applied per candle instead of last-candle-for-all.
      - leverage(): MACD histogram rule was dead (TA-Lib returns a list, not a DataFrame).
      - adjust_trade_position(): partial exits sized from stake (V48 used notional, which
        over-sells with leverage); NameError on empty dataframe; per-order max_stake check.
      - custom_stake_amount(): volatility reduction read a non-existent column.
      - custom_entry_price(): tag names did not match the tags populate_entry_trend sets.
      - stoploss_on_exchange moved into order_types (class attribute was ignored).
      - confirm_trade_exit(): emergency exits always allowed; force-exit blocking opt-in.
      - bot_loop_start(): naive/aware datetime TypeError silently disabled it.
      - Re-entry: exits are now recorded (order_filled) and candle time read from 'date'.
      - Signal strength tiers no longer overwrite stronger tiers with weaker ones.
      - Simple-exit path NameError, choppy_market always True, misc crashers fixed.
    """

    # General strategy parameters
    timeframe = "1h"
    startup_candle_count: int = 200
    stoploss = -0.15  # Fixed 15% stop loss (of stake, i.e. leverage-adjusted)
    use_custom_stoploss = False  # Disable custom logic (custom_stoploss() below is inactive)

    # stoploss_on_exchange is only honoured through order_types (a bare class attribute
    # named stoploss_on_exchange is ignored by freqtrade). Config order_types still win.
    order_types = {
        "entry": "limit",
        "exit": "limit",
        "stoploss": "market",
        "stoploss_on_exchange": True,  # Let exchange handle it
        "stoploss_on_exchange_interval": 60,  # Check every 60 seconds
    }

    trailing_stop = True
    trailing_stop_positive = 0.015  # Trail 1.5% behind the high ...
    trailing_stop_positive_offset = 0.03   # ... once profit reaches 3%
    trailing_only_offset_is_reached = True
    position_adjustment_enable = True
    can_short = True
    use_exit_signal = True
    ignore_roi_if_entry_signal = False
    max_stake_per_trade = 10.0
    max_portfolio_percentage_per_trade = 0.05
    max_entry_position_adjustment = 1
    process_only_new_candles = True
    max_dca_orders = 1
    max_total_stake_per_pair = 10
    max_single_dca_amount = 4
    use_custom_exits_advanced = True
    use_emergency_exits = True

    enable_position_flip = True
    # V48 silently refused /forceexit and emergency exits. Manual force exits are now allowed
    # unless you opt back in here; emergency exits (failed stoploss order etc.) always pass.
    block_force_exits = False

    last_entry_price: Optional[float] = None

    # ROI table (minutes -> profit ratio)
    minimal_roi = {
        "0": 0.17,      # 17% immediately
        "64": 0.113,    # 11.3% after 64 min
        "170": 0.051,   # 5.1% after 170 min
        "304": 0        # break-even after 304 min
    }
    plot_config = {
        "main_plot": {
            # Price action and key levels
            "actual_stoploss": {"color": "white", "type": "line"},
            "follow_price_level": {"color": "green", "type": "line"},

            # Key MML levels on main chart for better visibility
            "[2/8]P": {"color": "#ff6b6b", "type": "line"},  # 25% - Support/Resistance
            "[4/8]P": {"color": "#4ecdc4", "type": "line"},  # 50% - Key level
            "[6/8]P": {"color": "#45b7d1", "type": "line"},  # 75% - Support/Resistance

            # Entry signals on main chart
            "enter_long": {"color": "lime", "type": "scatter"},
            "enter_short": {"color": "red", "type": "scatter"},
        },
        "subplots": {
            "exit_signals": {
                "exit_long": {"color": "#ff4757", "type": "scatter"},
                "exit_short": {"color": "#2ed573", "type": "scatter"},
                "Signal_Flip_Short": {"color": "#ff9ff3", "type": "scatter"},
                "Signal_Flip_Long": {"color": "#54a0ff", "type": "scatter"},
                "Momentum_Loss": {"color": "#ff6b35", "type": "scatter"},
                "Support_Break": {"color": "#ff3838", "type": "scatter"},
                "Resistance_Break": {"color": "#2ecc71", "type": "scatter"},
            },
            "trade_metrics": {
                "price_change_5": {"color": "#74b9ff", "type": "line"},
                "rsi_change": {"color": "#fd79a8", "type": "line"},
                "volume_surge": {"color": "#fdcb6e", "type": "line"},
            },
            "rsi_analysis": {
                "rsi": {"color": "#6c5ce7", "type": "line"},
                "rsi_overbought_70": {"color": "#fd79a8", "type": "line"},
                "rsi_oversold_30": {"color": "#00b894", "type": "line"},
            },
            "extrema_analysis": {
                "s_extrema": {"color": "#f53580", "type": "line"},
                "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
                "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
            },
            "min_max_viz": {
                "maxima": {"color": "#a29db9", "type": "scatter"},
                "minima": {"color": "#aac7fc", "type": "scatter"},
                "maxima_check": {"color": "#e17055", "type": "line"},
                "minima_check": {"color": "#74b9ff", "type": "line"},
            },
            "murrey_math_levels": {
                "[1/8]P": {"color": "#d63031", "type": "line"},
                "[3/8]P": {"color": "#fd79a8", "type": "line"},
                "[4/8]P": {"color": "#0984e3", "type": "line"},
                "[5/8]P": {"color": "#00b894", "type": "line"},
                "[7/8]P": {"color": "#e84393", "type": "line"},
                "[8/8]P": {"color": "#d63031", "type": "line"},
                "[0/8]P": {"color": "#d63031", "type": "line"},
            },
            "market_correlation": {
                "btc_correlation": {"color": "#0984e3", "type": "line"},
                "market_breadth": {"color": "#00b894", "type": "line"},
                "market_score": {"color": "#6c5ce7", "type": "line"},
                "market_direction": {"color": "#fd79a8", "type": "line"},
            },
            "market_regime": {
                "market_volatility": {"color": "#e17055", "type": "line"},
                "mcap_trend": {"color": "#fdcb6e", "type": "line"},
                "market_adx": {"color": "#74b9ff", "type": "line"},
            },
            "volume_analysis": {
                "volume": {"color": "#636e72", "type": "bar"},
                "volume_ratio": {"color": "#00cec9", "type": "line"},
                "avg_volume": {"color": "#fd79a8", "type": "line"},
            },
            "trend_analysis": {
                "trend_consistency": {"color": "#a29bfe", "type": "line"},
                "trend_strength": {"color": "#fd79a8", "type": "line"},
                "strong_uptrend": {"color": "#00b894", "type": "line"},
                "strong_downtrend": {"color": "#d63031", "type": "line"},
            },
            "signal_quality": {
                "combined_signal_strength": {"color": "#fd79a8", "type": "line"},
                "mml_signal_strength": {"color": "#74b9ff", "type": "line"},
                "rsi_signal_strength": {"color": "#00b894", "type": "line"},
                "volume_strength": {"color": "#fdcb6e", "type": "line"},
            }
        },
    }

    # ===========================================
    # STATE / LIFECYCLE
    # ===========================================

    def bot_start(self, **kwargs) -> None:
        self.recent_exits: Dict[str, list] = {}
        self.reentry_count: Dict[str, Dict[str, int]] = {}
        self.signal_strength_cache: Dict = {}
        self._loop_log_times: Dict[int, datetime] = {}

    def get_or_init_reentry_data(self, pair: str):
        """
        📝 Initialize re-entry data for pair if not exists
        """
        if not hasattr(self, 'recent_exits'):
            self.recent_exits = {}
        if pair not in self.recent_exits:
            self.recent_exits[pair] = []
        if not hasattr(self, 'reentry_count'):
            self.reentry_count = {}
        if pair not in self.reentry_count:
            self.reentry_count[pair] = {'long': 0, 'short': 0}
        if not hasattr(self, 'signal_strength_cache'):
            self.signal_strength_cache = {}

    # ===========================================
    # MARKET-CONTEXT HELPERS (per-candle, no lookahead)
    # ===========================================

    @staticmethod
    def _pair_parts(pair: str) -> Tuple[str, str, Optional[str]]:
        """'SOL/USDT:USDT' -> ('SOL', 'USDT', 'USDT');  'SOL/USDT' -> ('SOL', 'USDT', None)"""
        base, rest = pair.split('/', 1) if '/' in pair else (pair, 'USDT')
        if ':' in rest:
            quote, settle = rest.split(':', 1)
        else:
            quote, settle = rest, None
        return base, quote, settle

    def _basket_pair(self, base: str, ref_pair: str) -> List[str]:
        """Candidate pair names for `base`, matching the market type of `ref_pair` first."""
        _, quote, settle = self._pair_parts(ref_pair)
        candidates = []
        if settle:
            candidates.append(f"{base}/{quote}:{settle}")
            candidates.append(f"{base}/USDT:USDT")
        candidates.append(f"{base}/{quote}")
        candidates.append(f"{base}/USDT")
        seen = set()
        return [c for c in candidates if not (c in seen or seen.add(c))]

    def _get_basket_dataframe(self, base: str, ref_pair: str, timeframe: str,
                              min_len: int = 20) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        for candidate in self._basket_pair(base, ref_pair):
            try:
                df = self.dp.get_pair_dataframe(candidate, timeframe)
            except Exception as e:
                logger.debug(f"   ❌ {candidate}: {e}")
                continue
            if df is not None and not df.empty and len(df) >= min_len and 'date' in df.columns:
                return df, candidate
        return None, None

    def _merge_inf_columns(self, dataframe: pd.DataFrame, inf_df: pd.DataFrame, inf_tf: str,
                           columns: List[str]) -> pd.DataFrame:
        """
        Merge per-candle `columns` of another pair onto `dataframe` by candle date using
        freqtrade's merge_informative_pair (shifts slower timeframes so only closed candles
        are visible). Returns a frame aligned row-for-row with `dataframe`.
        """
        inf = inf_df[['date'] + columns].copy()
        merged = merge_informative_pair(dataframe[['date']].copy(), inf, self.timeframe, inf_tf, ffill=True)
        out = pd.DataFrame(index=dataframe.index)
        for c in columns:
            out[c] = merged[f"{c}_{inf_tf}"].to_numpy()
        return out

    def calculate_signal_strength(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Signal strength calculation. Tiers are applied weakest -> strongest so a stronger
        tier is never overwritten by a weaker one (V48 capped volume at 0.6 and MML at 0.8).
        """
        dataframe['rsi_signal_strength'] = 0.0
        dataframe['mml_signal_strength'] = 0.0
        dataframe['mml_bounce_strength'] = 0.0
        dataframe['volume_strength'] = 0.0
        dataframe['extrema_strength'] = 0.0

        rsi = dataframe['rsi'].fillna(50)

        # RSI tiers don't overlap (except shared boundaries), so V48 order is kept
        dataframe.loc[rsi < 25, 'rsi_signal_strength'] = 1.0  # Very oversold
        dataframe.loc[rsi.between(25, 35), 'rsi_signal_strength'] = 0.8  # Oversold
        dataframe.loc[rsi.between(35, 40), 'rsi_signal_strength'] = 0.6  # Mildly oversold

        dataframe.loc[rsi > 75, 'rsi_signal_strength'] = 1.0  # Very overbought
        dataframe.loc[rsi.between(65, 75), 'rsi_signal_strength'] = 0.8  # Overbought
        dataframe.loc[rsi.between(60, 65), 'rsi_signal_strength'] = 0.6  # Mildly overbought

        try:
            if '[2/8]P' in dataframe.columns and '[6/8]P' in dataframe.columns:
                # Medium MML signals
                dataframe.loc[dataframe['close'] < dataframe['[2/8]P'], 'mml_signal_strength'] = 0.8
                dataframe.loc[dataframe['close'] > dataframe['[6/8]P'], 'mml_signal_strength'] = 0.8

                # MML bounces
                bounce_conditions = (
                    ((dataframe['low'] <= dataframe['[2/8]P']) & (dataframe['close'] > dataframe['[2/8]P'])) |
                    ((dataframe['high'] >= dataframe['[6/8]P']) & (dataframe['close'] < dataframe['[6/8]P']))
                )
                dataframe.loc[bounce_conditions, 'mml_bounce_strength'] = 0.9

            if '[1/8]P' in dataframe.columns and '[7/8]P' in dataframe.columns:
                # Extreme MML signals (applied last so they win)
                dataframe.loc[dataframe['close'] < dataframe['[1/8]P'], 'mml_signal_strength'] = 1.0
                dataframe.loc[dataframe['close'] > dataframe['[7/8]P'], 'mml_signal_strength'] = 1.0
        except Exception as e:
            logger.warning(f"MML signal strength calculation error: {e}")

        try:
            avg_volume = dataframe['volume'].rolling(20).mean()
            dataframe.loc[dataframe['volume'] <= avg_volume, 'volume_strength'] = 0.3
            dataframe.loc[dataframe['volume'] > avg_volume, 'volume_strength'] = 0.6
            dataframe.loc[dataframe['volume'] > avg_volume * 1.2, 'volume_strength'] = 0.8
            dataframe.loc[dataframe['volume'] > avg_volume * 1.5, 'volume_strength'] = 1.0
        except Exception as e:
            logger.warning(f"Volume strength calculation error: {e}")
            dataframe['volume_strength'] = 0.5

        try:
            if 'minima' in dataframe.columns:
                dataframe.loc[dataframe['minima'] == 1, 'extrema_strength'] = 0.8
            if 'maxima' in dataframe.columns:
                dataframe.loc[dataframe['maxima'] == 1, 'extrema_strength'] = 0.8
        except Exception as e:
            logger.warning(f"Extrema strength calculation error: {e}")

        dataframe['combined_signal_strength'] = (
            dataframe['rsi_signal_strength'] * 0.3 +
            dataframe['mml_signal_strength'] * 0.25 +
            dataframe['mml_bounce_strength'] * 0.2 +
            dataframe['volume_strength'] * 0.15 +
            dataframe['extrema_strength'] * 0.1
        ).clip(0, 1)

        return dataframe

    def check_signal_confirmation(self, dataframe: pd.DataFrame, signal_type: str) -> pd.Series:
        """
        Check if signal persists for required confirmation period (helper - not called).
        """
        confirmation_period = self.signal_confirmation_candles.value

        if signal_type == 'long':
            bullish_conditions = (
                (dataframe['rsi'] < 45) |
                (dataframe['close'] > dataframe['[4/8]P']) |
                (dataframe['minima'] == 1) |
                (dataframe['close'] > dataframe['close'].shift(1))
            )
            confirmed = bullish_conditions.rolling(confirmation_period).sum() >= (confirmation_period * 0.7)
        elif signal_type == 'short':
            bearish_conditions = (
                (dataframe['rsi'] > 55) |
                (dataframe['close'] < dataframe['[4/8]P']) |
                (dataframe['maxima'] == 1) |
                (dataframe['close'] < dataframe['close'].shift(1))
            )
            confirmed = bearish_conditions.rolling(confirmation_period).sum() >= (confirmation_period * 0.7)
        else:
            confirmed = pd.Series(False, index=dataframe.index)

        return confirmed

    def check_signal_cooldown(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Prevent signals too close together (helper - not called).
        """
        cooldown_period = self.signal_cooldown_period.value
        dataframe['recent_long_signal'] = dataframe['enter_long'].rolling(cooldown_period).sum()
        dataframe['recent_short_signal'] = dataframe['enter_short'].rolling(cooldown_period).sum()
        can_signal = (
            (dataframe['recent_long_signal'] == 0) &
            (dataframe['recent_short_signal'] == 0)
        )
        return can_signal

    def detect_market_state(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Detect current market state with pair-specific calculations (per candle).
        Writes both market_state and market_regime.
        """
        df = df.copy()

        df['price_volatility'] = df['atr'] / df['close']
        df['volatility_percentile'] = df['price_volatility'].rolling(self.regime_lookback_period.value).rank(pct=True)
        df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
        df['volume_increasing'] = df['volume'] > df['volume'].shift(1)

        df['momentum_3'] = df['close'].pct_change(3)
        df['momentum_5'] = df['close'].pct_change(5)
        df['momentum_10'] = df['close'].pct_change(10)

        df['consistent_uptrend'] = (
            (df['momentum_3'] > 0) &
            (df['momentum_5'] > 0) &
            (df['momentum_10'] > 0)
        )
        df['consistent_downtrend'] = (
            (df['momentum_3'] < 0) &
            (df['momentum_5'] < 0) &
            (df['momentum_10'] < 0)
        )

        df['market_state'] = 'neutral'
        df.loc[
            (df['consistent_uptrend'] & (df['adx'] > 25) & (df['volume_ratio'] > 1.2)),
            'market_state'
        ] = 'bull_run'
        df.loc[
            (df['consistent_downtrend'] & (df['adx'] > 25) & (df['volume_ratio'] > 1.2)),
            'market_state'
        ] = 'bear_market'
        df.loc[
            (df['adx'] < 20) & (df['volatility_percentile'] < 0.5),
            'market_state'
        ] = 'ranging'
        df.loc[
            (df['volatility_percentile'] > 0.8),
            'market_state'
        ] = 'high_volatility'

        df['potential_reversal'] = (
            ((df['high'] >= df['high'].rolling(10).max()) &
             (df['momentum_3'] < df['momentum_3'].shift(3))) |
            ((df['low'] <= df['low'].rolling(10).min()) &
             (df['momentum_3'] > df['momentum_3'].shift(3)))
        )

        df['potential_breakout'] = (
            (df['close'] > df['high'].rolling(20).max().shift(1)) |
            (df['close'] < df['low'].rolling(20).min().shift(1))
        ) & (df['volume_ratio'] > 1.5)

        df['market_regime'] = df['market_state']

        if len(df) > 0:
            logger.info(f"🏛️ Market regime: {df['market_regime'].iloc[-1]}, "
                        f"Volatility: {df['price_volatility'].iloc[-1]:.4f}, "
                        f"ADX: {df['adx'].iloc[-1]:.1f}, "
                        f"Trend Strength: {df['momentum_10'].iloc[-1]:.2%}")
        return df

    def get_timing_filters(self, dataframe: pd.DataFrame, direction: str) -> pd.Series:
        """
        Get timing-based filters to avoid bad entry timing (helper - not called).
        Note: market_state never takes the values 'strong_uptrend'/'strong_downtrend'.
        """
        # V48 used `~self.require_volume_confirmation.value` - bitwise NOT of a Python bool is
        # -1/-2 (always truthy), so the volume requirement could never be enforced.
        if self.require_volume_confirmation.value:
            volume_ok = dataframe['volume_ratio'] > 1.1
        else:
            volume_ok = pd.Series(True, index=dataframe.index)

        if direction == 'long':
            good_timing = (
                (dataframe['market_state'] != 'strong_downtrend') &
                (
                    dataframe['potential_breakout'] |
                    (dataframe['rsi'] < 40) |
                    (dataframe['market_state'] == 'strong_uptrend')
                ) &
                (
                    (dataframe['market_state'] != 'high_volatility') |
                    (dataframe['rsi'] < 25)
                ) &
                volume_ok
            )
        else:  # short
            good_timing = (
                (dataframe['market_state'] != 'strong_uptrend') &
                (
                    dataframe['potential_breakout'] |
                    (dataframe['rsi'] > 60) |
                    (dataframe['market_state'] == 'strong_downtrend')
                ) &
                (
                    (dataframe['market_state'] != 'high_volatility') |
                    (dataframe['rsi'] > 75)
                ) &
                volume_ok
            )

        return good_timing

    def has_active_trade(self, pair: str, side: str) -> bool:
        """
        Check if there's an active trade in the specified direction
        """
        try:
            trades = Trade.get_open_trades()
            for trade in trades:
                if trade.pair == pair:
                    if side == "long" and not trade.is_short:
                        return True
                    elif side == "short" and trade.is_short:
                        return True
        except Exception as e:
            logger.warning(f"Error checking active trades for {pair}: {e}")
        return False

    @staticmethod
    def _calculate_mml_core(mn: float, finalH: float, mx: float, finalL: float,
                            mml_c1: float, mml_c2: float) -> Dict[str, float]:
        dmml_calc = ((finalH - finalL) / 8.0) * mml_c1
        if dmml_calc == 0 or np.isinf(dmml_calc) or np.isnan(dmml_calc) or finalH == finalL:
            return {key: finalL for key in MML_LEVEL_NAMES}
        mml_val = (mx * mml_c2) + (dmml_calc * 3)
        if np.isinf(mml_val) or np.isnan(mml_val):
            return {key: finalL for key in MML_LEVEL_NAMES}
        ml = [mml_val - (dmml_calc * i) for i in range(16)]
        return {key: ml[idx] for key, idx in _MML_LADDER_INDEX.items()}

    def calculate_rolling_murrey_math_levels_optimized(self, df: pd.DataFrame, window_size: int) -> Dict[str, pd.Series]:
        """
        Murrey Math Levels on a rolling window, computed for EVERY candle (vectorised).

        V48 calculated every 5th candle and filled the gaps with
        interpolate(limit_direction='both') + bfill, i.e. candles used levels derived from
        up to 4 FUTURE candles (lookahead bias). Its per-pair cache also never hit.
        Candles before the first full window are NaN (startup_candle_count covers them).
        """
        rolling_high = df["high"].rolling(window=window_size, min_periods=window_size).max()
        rolling_low = df["low"].rolling(window=window_size, min_periods=window_size).min()
        close = df["close"]
        mml_c1 = self.mml_const1.value
        mml_c2 = self.mml_const2.value

        dmml = ((rolling_high - rolling_low) / 8.0) * mml_c1
        mml_val = (rolling_high * mml_c2) + (dmml * 3)

        warmup = rolling_high.isna() | rolling_low.isna()
        degenerate = (~warmup) & (
            (rolling_high == rolling_low) | (rolling_low == 0) |
            ~np.isfinite(dmml) | (dmml == 0) | ~np.isfinite(mml_val)
        )

        levels: Dict[str, pd.Series] = {}
        for key, idx in _MML_LADDER_INDEX.items():
            level = mml_val - dmml * idx
            level = level.where(~degenerate, close)  # V48 fallback: use close
            level[warmup] = np.nan
            levels[key] = level
        return {key: levels[key] for key in MML_LEVEL_NAMES}

    def calculate_market_correlation_simple(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Legacy BTC correlation, per candle: rolling 20-candle return correlation with BTC and
        a close > SMA20 > SMA50 trend flag.
        """
        pair = metadata['pair']
        base_currency = self._pair_parts(pair)[0]

        if base_currency == 'BTC':
            dataframe['btc_correlation'] = 1.0
            dataframe['btc_trend'] = 1
            dataframe['btc_close'] = dataframe['close']
            dataframe['btc_sma20'] = dataframe['close'].rolling(20).mean()
            dataframe['btc_sma50'] = dataframe['close'].rolling(50).mean()
            dataframe['returns'] = dataframe['close'].pct_change()
            return dataframe

        dataframe['returns'] = dataframe['close'].pct_change()
        btc_df, btc_pair = self._get_basket_dataframe('BTC', pair, self.timeframe, min_len=50)

        if btc_df is None:
            dataframe['btc_correlation'] = 0.5
            dataframe['btc_trend'] = 0
            dataframe['btc_close'] = dataframe['close']
            dataframe['btc_sma20'] = dataframe['close']
            dataframe['btc_sma50'] = dataframe['close']
            logger.warning(f"No BTC data for {pair}, using neutral correlation")
            return dataframe

        try:
            btc = btc_df[['date', 'close']].copy()
            btc['btc_close'] = btc['close']
            btc['btc_ret'] = btc['close'].pct_change()
            btc['btc_sma20'] = btc['close'].rolling(20).mean()
            btc['btc_sma50'] = btc['close'].rolling(50).mean()
            btc['btc_trend'] = np.select(
                [(btc['close'] > btc['btc_sma20']) & (btc['btc_sma20'] > btc['btc_sma50']),
                 (btc['close'] < btc['btc_sma20']) & (btc['btc_sma20'] < btc['btc_sma50'])],
                [1, -1], default=0)

            merged = self._merge_inf_columns(
                dataframe, btc, self.timeframe, ['btc_close', 'btc_ret', 'btc_sma20', 'btc_sma50', 'btc_trend'])

            correlation_period = 20
            corr = dataframe['returns'].rolling(correlation_period).corr(merged['btc_ret'])
            dataframe['btc_correlation'] = corr.replace([np.inf, -np.inf], np.nan).fillna(0.5)
            dataframe['btc_trend'] = merged['btc_trend'].fillna(0).astype(int)
            dataframe['btc_close'] = merged['btc_close'].fillna(dataframe['close'])
            dataframe['btc_sma20'] = merged['btc_sma20']
            dataframe['btc_sma50'] = merged['btc_sma50']

            logger.debug(f"{pair} Correlation vs {btc_pair}: {dataframe['btc_correlation'].iloc[-1]:.3f}, "
                         f"BTC trend: {dataframe['btc_trend'].iloc[-1]}")
        except Exception as e:
            logger.warning(f"Simple correlation failed for {pair}: {e}")
            dataframe['btc_correlation'] = 0.5
            dataframe['btc_trend'] = 0
            dataframe['btc_close'] = dataframe['close']
            dataframe['btc_sma20'] = dataframe['close']
            dataframe['btc_sma50'] = dataframe['close']

        return dataframe

    def informative_pairs(self):
        """
        Pairs/timeframes read through dp.get_pair_dataframe(). Built from the whitelist's
        quote/settle currency so spot and futures both resolve.
        V48 also requested every whitelist pair on 15m/1h/4h, which nothing read; dropped.
        """
        try:
            whitelist = self.dp.current_whitelist()
        except Exception:
            whitelist = []
        ref_pair = whitelist[0] if whitelist else "BTC/USDT:USDT"
        bases = {b for b, _ in BREADTH_BASKET} | {b for b, _ in MCAP_BASKET}

        informative = []
        for base in sorted(bases):
            informative.append((self._basket_pair(base, ref_pair)[0], self.timeframe))
        # Enhanced BTC filter falls back to a slower timeframe if the main one is short
        btc_pair = self._basket_pair('BTC', ref_pair)[0]
        for tf in ('4h',):
            if timeframe_to_minutes(tf) > timeframe_to_minutes(self.timeframe):
                informative.append((btc_pair, tf))

        seen = set()
        return [p for p in informative if not (p in seen or seen.add(p))]

    def _extract_dataframe_safely(self, result, pair_name):
        """
        Safely extract DataFrame from get_analyzed_dataframe result
        """
        try:
            if isinstance(result, tuple) and len(result) > 0:
                potential_df = result[0]
                if isinstance(potential_df, pd.DataFrame):
                    return potential_df
                elif isinstance(potential_df, list) and len(potential_df) > 0:
                    if isinstance(potential_df[0], pd.DataFrame):
                        return potential_df[0]
            elif isinstance(result, pd.DataFrame):
                return result
            elif isinstance(result, list) and len(result) > 0:
                if isinstance(result[0], pd.DataFrame):
                    return result[0]
                elif isinstance(result[0], tuple) and len(result[0]) > 0:
                    if isinstance(result[0][0], pd.DataFrame):
                        return result[0][0]

            logger.debug(f"   ❌ {pair_name}: Cannot extract DataFrame from result (type: {type(result)})")
            return None
        except Exception as e:
            logger.debug(f"   ❌ {pair_name}: Error extracting DataFrame: {str(e)}")
            return None

    def calculate_market_breadth_weighted(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Weighted market breadth, per candle.

        For every basket pair: close > SMA20 * 1.002 -> bullish, < SMA20 / 1.002 -> bearish,
        otherwise neutral. breadth = bullish_weight / (bullish_weight + bearish_weight).
        Candles with no basket data fall back to the current pair's own trend. Same rules as
        V48, but evaluated on each candle instead of broadcasting the last value.
        """
        current_pair = metadata['pair']
        threshold = 1.002  # 0.2% buffer

        n = len(dataframe)
        bullish_weight = np.zeros(n)
        bearish_weight = np.zeros(n)
        neutral_weight = np.zeros(n)
        pairs_checked = np.zeros(n)

        for base, weight in BREADTH_BASKET:
            try:
                pair_data, used_pair = self._get_basket_dataframe(base, current_pair, self.timeframe, min_len=20)
                if pair_data is None:
                    logger.debug(f"   ❌ {base}: No data")
                    continue

                p = pair_data[['date', 'close']].copy()
                sma20 = p['close'].rolling(20).mean()
                valid = sma20.notna() & (sma20 > 0) & p['close'].notna()
                p['trend'] = np.select(
                    [p['close'] > sma20 * threshold, p['close'] < sma20 / threshold], [1.0, -1.0], default=0.0)
                p.loc[~valid, 'trend'] = np.nan

                trend = self._merge_inf_columns(dataframe, p, self.timeframe, ['trend'])['trend'].to_numpy()
                bullish_weight += np.where(trend == 1, weight, 0.0)
                bearish_weight += np.where(trend == -1, weight, 0.0)
                neutral_weight += np.where(trend == 0, weight, 0.0)
                pairs_checked += ~np.isnan(trend)
                logger.debug(f"   ✅ {used_pair}: last trend {trend[-1] if n else 'n/a'} (weight: {weight:.2f})")
            except Exception as e:
                logger.warning(f"   💥 {base} breadth ERROR: {str(e)}")
                continue

        total_weight = bullish_weight + bearish_weight + neutral_weight
        directional = bullish_weight + bearish_weight
        with np.errstate(invalid='ignore', divide='ignore'):
            breadth = np.where(directional > 0, bullish_weight / np.where(directional > 0, directional, 1), 0.5)
        direction = np.sign(bullish_weight - bearish_weight)

        # Fallback for candles without any basket data: current pair trend
        no_data = (pairs_checked == 0) | (total_weight <= 0)
        if no_data.any():
            sma20_self = dataframe['close'].rolling(20).mean()
            ratio = (dataframe['close'] / sma20_self).to_numpy()
            fb_breadth = np.select([ratio > 1.01, ratio < 0.99], [0.65, 0.35], default=0.5)
            fb_dir = np.select([ratio > 1.01, ratio < 0.99], [1, -1], default=0)
            breadth = np.where(no_data, fb_breadth, breadth)
            direction = np.where(no_data, fb_dir, direction)
            pairs_checked = np.where(no_data, 1, pairs_checked)

        # V48: prevent 0% breadth (all-bearish) -> 35%
        zero = breadth == 0.0
        breadth = np.where(zero, 0.35, breadth)
        direction = np.where(zero, -1, direction)

        dataframe['market_breadth'] = breadth
        dataframe['market_direction'] = direction.astype(int)
        dataframe['bullish_weight'] = bullish_weight
        dataframe['bearish_weight'] = bearish_weight
        dataframe['total_weight'] = total_weight
        dataframe['breadth_pairs_checked'] = pairs_checked.astype(int)

        if n:
            logger.info(f"🎯 {current_pair} Final Breadth: {breadth[-1]:.1%} "
                        f"(Direction: {int(direction[-1])}, pairs: {int(pairs_checked[-1])})")
        return dataframe

    def calculate_market_regime_aligned(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🔍 BTC market regime per candle (volatility, z-score trend strength, ADX).

        Written to `btc_market_regime` (+ market_volatility / market_adx / btc_trend_strength).
        V48 wrote it to `market_regime`, but detect_market_state() overwrote that column right
        afterwards, so the BTC regime never reached a signal; keeping it separate preserves
        that effective behaviour while making the BTC view available.
        """
        lookback = self.regime_lookback_period.value
        current_pair = metadata['pair']

        def _neutral():
            dataframe['btc_market_regime'] = 'neutral'
            dataframe['market_volatility'] = 0.02
            dataframe['market_adx'] = 25.0
            dataframe['btc_trend_strength'] = 0.0
            return dataframe

        btc_data, btc_pair_used = self._get_basket_dataframe('BTC', current_pair, self.timeframe, min_len=lookback)
        if btc_data is None:
            logger.info(f"🔄 {current_pair} No BTC data available, using neutral regime")
            return _neutral()

        try:
            tf_minutes = timeframe_to_minutes(self.timeframe)
            btc = btc_data[['date', 'open', 'high', 'low', 'close', 'volume']].copy()
            returns = btc['close'].pct_change()
            btc['market_volatility'] = returns.rolling(window=lookback).std() * np.sqrt(1440 / tf_minutes)
            btc['market_adx'] = pd.Series(np.asarray(ta.ADX(btc, timeperiod=14), dtype=float), index=btc.index)
            mean = btc['close'].rolling(window=lookback).mean()
            std = btc['close'].rolling(window=lookback).std()
            btc['btc_trend_strength'] = ((btc['close'] - mean) / std.replace(0, np.nan)).fillna(0.0)

            vol, ts, adx = btc['market_volatility'], btc['btc_trend_strength'], btc['market_adx']
            btc['btc_market_regime'] = np.select(
                [(vol > 0.05) & (ts > 1.5) & (adx > 35),
                 (vol > 0.05) & (ts < -1.5) & (adx > 35),
                 (vol < 0.008) & (adx < 15),
                 (vol > 0.08)],
                ['bull_run', 'bear_market', 'ranging', 'high_volatility'],
                default='neutral')

            merged = self._merge_inf_columns(
                dataframe, btc, self.timeframe,
                ['btc_market_regime', 'market_volatility', 'market_adx', 'btc_trend_strength'])
            dataframe['btc_market_regime'] = merged['btc_market_regime'].fillna('neutral')
            dataframe['market_volatility'] = merged['market_volatility'].astype(float).fillna(0.02)
            dataframe['market_adx'] = merged['market_adx'].astype(float).fillna(25.0)
            dataframe['btc_trend_strength'] = merged['btc_trend_strength'].astype(float).fillna(0.0)

            logger.info(f"🏛️ {current_pair} BTC regime ({btc_pair_used}): {dataframe['btc_market_regime'].iloc[-1]}, "
                        f"Volatility: {dataframe['market_volatility'].iloc[-1]:.4f}, "
                        f"ADX: {dataframe['market_adx'].iloc[-1]:.1f}, "
                        f"Trend Strength: {dataframe['btc_trend_strength'].iloc[-1]:.2f}")
            return dataframe

        except Exception as e:
            logger.warning(f"💥 {current_pair} Regime calculation failed: {str(e)}, using neutral regime")
            return _neutral()

    def apply_market_regime_filters(self, df: pd.DataFrame, any_long_signal: pd.Series,
                                    any_short_signal: pd.Series, metadata: dict) -> tuple:
        """
        🧠 Adjust signal sensitivity based on market conditions (helper - not called).
        Evaluated per candle (V48 used the last candle's regime for the whole history).
        """
        try:
            regime = df['market_regime'] if 'market_regime' in df.columns else pd.Series('neutral', index=df.index)
            fear_greed = df['market_fear_greed'] if 'market_fear_greed' in df.columns else pd.Series(50, index=df.index)
            rsi = df['rsi']
            vol_ok = df['volume'] > df['volume'].rolling(20).mean() * 1.2
            mml38 = df['[3/8]P'] if '[3/8]P' in df.columns else df['close']
            mml58 = df['[5/8]P'] if '[5/8]P' in df.columns else df['close']

            long_gate = np.select(
                [regime == 'bull_run', regime == 'bear_market', regime == 'high_volatility',
                 regime == 'sideways', regime == 'transitional'],
                [rsi < 60, rsi < 20, rsi < 25, (rsi < 35) & (df['close'] <= mml38), (rsi < 40) & vol_ok],
                default=True)
            short_gate = np.select(
                [regime == 'bull_run', regime == 'bear_market', regime == 'high_volatility',
                 regime == 'sideways', regime == 'transitional'],
                [rsi > 80, rsi > 40, rsi > 75, (rsi > 65) & (df['close'] >= mml58), (rsi > 60) & vol_ok],
                default=True)

            enhanced_long_signal = any_long_signal & pd.Series(long_gate, index=df.index).astype(bool)
            enhanced_short_signal = any_short_signal & pd.Series(short_gate, index=df.index).astype(bool)

            enhanced_long_signal &= ~((fear_greed > 75) & ~(rsi < 30))
            enhanced_short_signal &= ~((fear_greed < 25) & ~(rsi > 70))
            return enhanced_long_signal, enhanced_short_signal

        except Exception as e:
            logger.error(f"{metadata['pair']} ❌ Market regime filter error: {e}")
            return any_long_signal, any_short_signal

    def calculate_total_market_cap_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Weighted "total market cap" trend proxy, per candle: Σ w * (close - MA) / MA.
        V48 read get_analyzed_dataframe() (empty in backtests -> always 0 there, non-zero live)
        and broadcast the last value; this uses raw candles merged by date.
        """
        ma_period = self.total_mcap_ma_period.value
        n = len(dataframe)
        weighted_trend = np.zeros(n)
        total_weight = np.zeros(n)

        for base, weight in MCAP_BASKET:
            try:
                coin_data, _ = self._get_basket_dataframe(base, metadata['pair'], self.timeframe, min_len=ma_period)
                if coin_data is None:
                    continue
                c = coin_data[['date', 'close']].copy()
                ma = c['close'].rolling(ma_period).mean()
                c['mcap_component'] = (c['close'] - ma) / ma
                comp = self._merge_inf_columns(dataframe, c, self.timeframe, ['mcap_component'])['mcap_component']
                comp = comp.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=float)
                has = ~np.isnan(comp)
                weighted_trend += np.where(has, comp * weight, 0.0)
                total_weight += np.where(has, weight, 0.0)
            except Exception as e:
                logger.debug(f"Could not process {base} for mcap trend: {e}")
                continue

        with np.errstate(invalid='ignore', divide='ignore'):
            mcap_trend = np.where(total_weight > 0, weighted_trend / np.where(total_weight > 0, total_weight, 1), 0.0)

        dataframe['mcap_trend'] = mcap_trend
        dataframe['mcap_status'] = np.select([mcap_trend > 0.05, mcap_trend < -0.05], ['bullish', 'bearish'],
                                             default='neutral')
        return dataframe

    def apply_correlation_filters(self, dataframe: pd.DataFrame, direction: str = 'long') -> pd.Series:
        """
        Apply market correlation filters (helper - not called).
        Returns a boolean Series indicating whether market conditions are favorable
        """
        conditions = pd.Series(True, index=dataframe.index)
        rsi = dataframe['rsi'] if 'rsi' in dataframe.columns else pd.Series(50, index=dataframe.index)

        if self.enable_btc_correlation.value and 'btc_correlation_ok' in dataframe.columns:
            conditions &= dataframe['btc_correlation_ok'].astype(bool)
            if 'btc_trend_score' in dataframe.columns:
                score = dataframe['btc_trend_score']
                if direction == 'long':
                    conditions &= (score >= 2) | score.isna() | (rsi < 25)
                else:
                    conditions &= (score <= 3) | score.isna() | (rsi > 75)

        if self.market_breadth_enabled.value and 'market_breadth' in dataframe.columns:
            if direction == 'long':
                conditions &= (dataframe['market_breadth'] > self.market_breadth_threshold.value)
            else:
                conditions &= (dataframe['market_breadth'] < (1 - self.market_breadth_threshold.value))

        if self.total_mcap_filter_enabled.value and 'mcap_status' in dataframe.columns:
            if direction == 'long':
                conditions &= (dataframe['mcap_status'] != 'bearish')
            else:
                conditions &= (dataframe['mcap_status'] != 'bullish')

        if self.regime_filter_enabled.value and 'market_regime' in dataframe.columns:
            conditions &= (dataframe['market_regime'] != 'high_volatility')

        return conditions

    def calculate_enhanced_trend_filters(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """
        Enhanced trend detection to avoid choppy market entries
        """
        dataframe['ema_5'] = ta.EMA(dataframe['close'], timeperiod=5)
        dataframe['ema_13'] = ta.EMA(dataframe['close'], timeperiod=13)
        dataframe['ema_21'] = ta.EMA(dataframe['close'], timeperiod=21)
        dataframe['ema_50'] = ta.EMA(dataframe['close'], timeperiod=50)

        bull_aligned = (
            (dataframe['close'] > dataframe['ema_5']) &
            (dataframe['ema_5'] > dataframe['ema_13']) &
            (dataframe['ema_13'] > dataframe['ema_21']) &
            (dataframe['ema_21'] > dataframe['ema_50'])
        )
        bear_aligned = (
            (dataframe['close'] < dataframe['ema_5']) &
            (dataframe['ema_5'] < dataframe['ema_13']) &
            (dataframe['ema_13'] < dataframe['ema_21']) &
            (dataframe['ema_21'] < dataframe['ema_50'])
        )
        dataframe['trend_alignment_bull'] = bull_aligned.astype(int)
        dataframe['trend_alignment_bear'] = bear_aligned.astype(int)

        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['strong_trend'] = dataframe['adx'] > 25
        dataframe['very_strong_trend'] = dataframe['adx'] > 40

        dataframe['macd'], dataframe['macd_signal'], dataframe['macd_hist'] = ta.MACD(
            dataframe['close'], fastperiod=12, slowperiod=26, signalperiod=9
        )

        dataframe['macd_bull'] = (
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1))
        )
        dataframe['macd_bear'] = (
            (dataframe['macd'] < dataframe['macd_signal']) &
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1))
        )

        dataframe['higher_highs'] = (
            (dataframe['high'] > dataframe['high'].shift(1)) &
            (dataframe['high'].shift(1) > dataframe['high'].shift(2))
        )
        dataframe['lower_lows'] = (
            (dataframe['low'] < dataframe['low'].shift(1)) &
            (dataframe['low'].shift(1) < dataframe['low'].shift(2))
        )
        dataframe['higher_lows'] = (
            (dataframe['low'] > dataframe['low'].shift(1)) &
            (dataframe['low'].shift(1) > dataframe['low'].shift(2))
        )
        dataframe['lower_highs'] = (
            (dataframe['high'] < dataframe['high'].shift(1)) &
            (dataframe['high'].shift(1) < dataframe['high'].shift(2))
        )

        dataframe['bb_upper'], dataframe['bb_middle'], dataframe['bb_lower'] = ta.BBANDS(
            dataframe['close'], timeperiod=20, nbdevup=2.0, nbdevdn=2.0
        )
        bb_width = (dataframe['bb_upper'] - dataframe['bb_lower']) / dataframe['bb_middle']
        dataframe['bb_squeeze'] = bb_width < 0.1
        dataframe['bb_expansion'] = bb_width > 0.2

        # V48 applied `~` to the int alignment columns (~0 == -1, ~1 == -2: both truthy), which
        # made choppy_market always True. Use the boolean masks instead.
        dataframe['trending_up'] = (
            bull_aligned & dataframe['strong_trend'] & dataframe['macd_bull'] & (~dataframe['bb_squeeze'])
        )
        dataframe['trending_down'] = (
            bear_aligned & dataframe['strong_trend'] & dataframe['macd_bear'] & (~dataframe['bb_squeeze'])
        )
        dataframe['choppy_market'] = (
            (dataframe['adx'] < 20) |
            dataframe['bb_squeeze'] |
            ((~bull_aligned) & (~bear_aligned))
        )

        vol_mean20 = dataframe['volume'].rolling(20).mean()
        dataframe['structure_break_bull'] = (
            (dataframe['close'] > dataframe['high'].rolling(20).max().shift(1)) &
            dataframe['trending_up'] &
            (dataframe['volume'] > vol_mean20 * 1.2)
        )
        dataframe['structure_break_bear'] = (
            (dataframe['close'] < dataframe['low'].rolling(20).min().shift(1)) &
            dataframe['trending_down'] &
            (dataframe['volume'] > vol_mean20 * 1.2)
        )

        dataframe['trend_momentum'] = dataframe['close'].pct_change(10)
        dataframe['trend_momentum_strength'] = abs(dataframe['trend_momentum'])

        dataframe['short_term_trend'] = np.where(
            dataframe['close'] > dataframe['ema_5'], 1,
            np.where(dataframe['close'] < dataframe['ema_5'], -1, 0)
        )
        dataframe['medium_term_trend'] = np.where(
            dataframe['close'] > dataframe['ema_21'], 1,
            np.where(dataframe['close'] < dataframe['ema_21'], -1, 0)
        )
        dataframe['long_term_trend'] = np.where(
            dataframe['close'] > dataframe['ema_50'], 1,
            np.where(dataframe['close'] < dataframe['ema_50'], -1, 0)
        )

        dataframe['trend_consistency'] = (
            dataframe['short_term_trend'] +
            dataframe['medium_term_trend'] +
            dataframe['long_term_trend']
        ) / 3

        dataframe['strong_consistent_uptrend'] = dataframe['trend_consistency'] > 0.6
        dataframe['strong_consistent_downtrend'] = dataframe['trend_consistency'] < -0.6

        dataframe['trend_change_up'] = (
            (dataframe['trend_consistency'] > 0) &
            (dataframe['trend_consistency'].shift(1) <= 0)
        )
        dataframe['trend_change_down'] = (
            (dataframe['trend_consistency'] < 0) &
            (dataframe['trend_consistency'].shift(1) >= 0)
        )

        dataframe['dynamic_resistance'] = np.maximum.reduce([
            dataframe['ema_5'], dataframe['ema_13'],
            dataframe['ema_21'], dataframe['ema_50']
        ])
        dataframe['dynamic_support'] = np.minimum.reduce([
            dataframe['ema_5'], dataframe['ema_13'],
            dataframe['ema_21'], dataframe['ema_50']
        ])

        dataframe['above_all_emas'] = (
            (dataframe['close'] > dataframe['ema_5']) &
            (dataframe['close'] > dataframe['ema_13']) &
            (dataframe['close'] > dataframe['ema_21']) &
            (dataframe['close'] > dataframe['ema_50'])
        )
        dataframe['below_all_emas'] = (
            (dataframe['close'] < dataframe['ema_5']) &
            (dataframe['close'] < dataframe['ema_13']) &
            (dataframe['close'] < dataframe['ema_21']) &
            (dataframe['close'] < dataframe['ema_50'])
        )

        price_peaks = dataframe['high'].rolling(10).max()
        price_troughs = dataframe['low'].rolling(10).min()

        dataframe['bullish_divergence'] = (
            (dataframe['low'] == price_troughs) &
            (dataframe['low'] < dataframe['low'].shift(10)) &
            (dataframe['rsi'] > dataframe['rsi'].shift(10))
        )
        dataframe['bearish_divergence'] = (
            (dataframe['high'] == price_peaks) &
            (dataframe['high'] > dataframe['high'].shift(10)) &
            (dataframe['rsi'] < dataframe['rsi'].shift(10))
        )

        return dataframe

    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength to avoid entering against strong trends (helper - not called).
        Linear-regression slope computed in closed form (V48 ran np.polyfit per candle).
        """
        def rolling_slope(series: pd.Series, period: int) -> pd.Series:
            x = np.arange(period, dtype=float)
            x_c = x - x.mean()
            denom = (x_c ** 2).sum()
            return series.rolling(period).apply(lambda y: float(np.dot(x_c, y - y.mean()) / denom), raw=True)

        df['slope_5'] = rolling_slope(df['close'], 5)
        df['slope_10'] = rolling_slope(df['close'], 10)
        df['slope_20'] = rolling_slope(df['close'], 20)

        df['trend_strength_5'] = df['slope_5'] / df['close'] * 100
        df['trend_strength_10'] = df['slope_10'] / df['close'] * 100
        df['trend_strength_20'] = df['slope_20'] / df['close'] * 100
        df['trend_strength'] = (df['trend_strength_5'] + df['trend_strength_10'] + df['trend_strength_20']) / 3

        strong_up_threshold = self.trend_strength_threshold.value
        strong_down_threshold = -self.trend_strength_threshold.value
        df['strong_uptrend'] = df['trend_strength'] > strong_up_threshold
        df['strong_downtrend'] = df['trend_strength'] < strong_down_threshold
        df['ranging'] = (df['trend_strength'].abs() < strong_up_threshold * 0.5)
        return df

    @property
    def protections(self):
        prot = [{"method": "CooldownPeriod", "stop_duration_candles": self.cooldown_lookback.value}]
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": 72,
                "trade_limit": 2,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": False,
            })
        return prot

    # ===========================================
    # CALLBACKS: STAKE / ENTRY PRICE / STOPLOSS
    # ===========================================

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float, entry_tag: Optional[str], side: str, **kwargs) -> float:

        # Max stake per trade (stake currency)
        MAX_STAKE_PER_TRADE = self.max_stake_per_trade

        # Max stake as a share of the portfolio
        try:
            total_portfolio = self.wallets.get_total_stake_amount()
            max_stake_from_portfolio = total_portfolio * self.max_portfolio_percentage_per_trade
        except Exception:
            # Fallback when wallets are not available
            max_stake_from_portfolio = MAX_STAKE_PER_TRADE
            total_portfolio = 1000.0  # Dummy value

        # Volatility-based stake reduction.
        # V48 read a 'volatility' column that is never created, so this never fired.
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is not None and not dataframe.empty:
            last_candle = dataframe.iloc[-1]
            current_volatility = last_candle.get("price_volatility", np.nan)
            if pd.isna(current_volatility):
                atr, close = last_candle.get("atr", np.nan), last_candle.get("close", np.nan)
                current_volatility = atr / close if close and not pd.isna(atr) else 0.02

            if current_volatility > 0.05:  # 5% ATR/Price ratio
                volatility_reduction = min(0.5, current_volatility * 10)  # Max 50% reduction
                proposed_stake *= (1 - volatility_reduction)
                logger.info(f"{pair} Stake reduced by {volatility_reduction:.1%} due to high volatility ({current_volatility:.2%})")

        # DCA multiplier: reserve room for the configured safety orders
        calculated_max_dca_multiplier = 1.0
        if self.position_adjustment_enable:
            num_safety_orders = int(self.max_safety_orders.value)
            volume_scale = self.safety_order_volume_scale.value
            if num_safety_orders > 0 and volume_scale > 0:
                current_order_relative_size = 1.0
                for _ in range(num_safety_orders):
                    current_order_relative_size *= volume_scale
                    calculated_max_dca_multiplier += current_order_relative_size
            else:
                logger.warning(f"{pair}: Could not calculate max_dca_multiplier due to "
                               f"invalid max_safety_orders ({num_safety_orders}) or "
                               f"safety_order_volume_scale ({volume_scale}). Defaulting to 1.0.")
        else:
            logger.debug(f"{pair}: Position adjustment not enabled. max_dca_multiplier is 1.0.")

        if calculated_max_dca_multiplier <= 0:
            logger.warning(
                f"{pair} Calculated max_dca_multiplier is {calculated_max_dca_multiplier:.2f}, which is invalid. "
                f"Using proposed_stake: {proposed_stake:.8f}")
            return proposed_stake

        stake_amount = proposed_stake / calculated_max_dca_multiplier

        final_stake = min(stake_amount, MAX_STAKE_PER_TRADE, max_stake_from_portfolio, max_stake)

        limit_reason = "calculated"
        if final_stake == MAX_STAKE_PER_TRADE:
            limit_reason = "max_per_trade"
        elif final_stake == max_stake_from_portfolio:
            limit_reason = "portfolio_percentage"
        elif final_stake == max_stake:
            limit_reason = "freqtrade_max"

        portfolio_pct = (final_stake / total_portfolio) * 100 if total_portfolio else 0.0
        logger.info(f"{pair} Initial stake calculated: {final_stake:.8f} (Proposed: {proposed_stake:.8f}, "
                    f"Calculated Max DCA Multiplier: {calculated_max_dca_multiplier:.2f}, "
                    f"Limited by: {limit_reason}, Portfolio %: {portfolio_pct:.1f}%)")

        if min_stake is not None and final_stake < min_stake:
            logger.info(f"{pair} Initial stake {final_stake:.8f} was below min_stake {min_stake:.8f}. "
                        f"Adjusting to min_stake. Consider tuning your DCA parameters or proposed stake.")
            final_stake = min_stake

        return final_stake

    @staticmethod
    def _base_tag(entry_tag: Optional[str]) -> str:
        """Strip the _RE<n> re-entry suffix."""
        tag = entry_tag or ""
        if "_RE" in tag:
            head, _, tail = tag.rpartition("_RE")
            if tail.isdigit():
                return head
        return tag

    def custom_entry_price(self, pair: str, trade: Optional[Trade], current_time: datetime,
                           proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        """
        🎯 Wait for better entry prices to reduce immediate stop loss risk.
        V48 compared against tag names populate_entry_trend never produces
        ('MML_Support_Bounce', 'MML_Bullish_Breakout', ...); the real tags are matched now.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return proposed_rate

        last_candle = dataframe.iloc[-1]
        tag = self._base_tag(entry_tag)

        atr = last_candle.get('atr', 0)
        if pd.isna(atr) or atr == 0:
            atr = (last_candle['high'] - last_candle['low'])
        if pd.isna(atr):
            return proposed_rate

        def level(name):
            v = last_candle.get(name, last_candle['close'])
            return last_candle['close'] if pd.isna(v) else v

        mml_25 = level('[2/8]P')
        mml_50 = level('[4/8]P')
        mml_75 = level('[6/8]P')

        if side == "long":
            if tag in ("MML_50_Reclaim", "Support_Bounce", "MML_Support_Bounce"):
                support_buffer = atr * 0.2
                optimal_entry = max(mml_25, mml_50) + support_buffer
                entry_price = min(proposed_rate, optimal_entry)
            elif tag in ("MML_Breakout_75", "MML_Bullish_Breakout"):
                pullback_entry = proposed_rate - (atr * 0.1)
                entry_price = max(pullback_entry, proposed_rate * 0.999)
            else:
                entry_price = proposed_rate - (atr * 0.05)
        elif side == "short":
            if tag in ("MML_50_Breakdown", "Resistance_Reject", "MML_Resistance_Reject"):
                resistance_buffer = atr * 0.2
                optimal_entry = min(mml_75, mml_50) - resistance_buffer
                entry_price = max(proposed_rate, optimal_entry)
            elif tag in ("Bear_Breakdown_25", "MML_Bearish_Breakdown"):
                bounce_entry = proposed_rate + (atr * 0.1)
                entry_price = min(bounce_entry, proposed_rate * 1.001)
            else:
                entry_price = proposed_rate + (atr * 0.05)
        else:
            entry_price = proposed_rate

        # Keep entry within 0.5% of the proposed rate
        max_deviation = proposed_rate * 0.005
        if side == "long":
            entry_price = max(entry_price, proposed_rate - max_deviation)
        else:
            entry_price = min(entry_price, proposed_rate + max_deviation)

        logger.info(f"{pair} Optimized entry: {entry_price:.6f} (was {proposed_rate:.6f}, tag: {entry_tag})")
        return entry_price

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        """
        🔧 Time-based stop loss loosening + MML support.
        INACTIVE while use_custom_stoploss = False. Note the time buckets were tuned on 15m
        data; the strategy now runs 1h, so revisit them before enabling.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty or 'atr' not in dataframe.columns or len(dataframe) < 10:
            logger.warning(f"{pair} Insufficient data for dynamic stop loss. Using conservative default: -0.03")
            return -0.03

        last_candle = dataframe.iloc[-1]
        last_atr = last_candle.get("atr", 0)

        if pd.isna(last_atr) or last_atr == 0:
            valid_atr = dataframe["atr"].dropna()
            if not valid_atr.empty:
                last_atr = valid_atr.rolling(window=5, min_periods=1).mean().iloc[-1]
                logger.info(f"{pair} Using smoothed ATR: {last_atr:.8f}")
            else:
                logger.warning(f"{pair} No valid ATR found. Using fallback stop loss -0.03")
                return -0.03

        if last_atr == 0 or current_rate == 0:
            logger.warning(f"{pair} ATR or rate is 0. Using fallback stop loss -0.03")
            return -0.03

        time_in_trade = (current_time - trade.open_date_utc).total_seconds() / 3600  # Hours

        if trade.is_short:
            resistance_level = last_candle.get('[6/8]P', current_rate * 1.02)
            if pd.isna(resistance_level):
                resistance_level = current_rate * 1.02
            mml_stop_distance = abs((resistance_level - current_rate) / current_rate)
        else:
            support_level = last_candle.get('[2/8]P', current_rate * 0.98)
            if pd.isna(support_level):
                support_level = current_rate * 0.98
            mml_stop_distance = abs((current_rate - support_level) / current_rate)

        base_atr_stop = (last_atr / current_rate) * 2.0

        if time_in_trade < 0.25:
            time_multiplier = 1.2
        elif time_in_trade < 0.5:
            time_multiplier = 1.5
        elif time_in_trade < 1.0:
            time_multiplier = 2.0
        elif time_in_trade < 1.5:
            time_multiplier = 2.5
        elif time_in_trade < 2.0:
            time_multiplier = 3.0
        elif time_in_trade < 4.0:
            time_multiplier = 2.8
        else:
            time_multiplier = 2.5

        if current_profit > 0.08:
            profit_stop = max(-0.015, -(base_atr_stop * 0.3))
        elif current_profit > 0.06:
            profit_stop = max(-0.02, -(base_atr_stop * 0.5))
        elif current_profit > 0.04:
            profit_stop = max(-0.025, -(base_atr_stop * 0.7))
        elif current_profit > 0.02:
            profit_stop = max(-0.03, -(base_atr_stop * 1.0))
        else:
            profit_stop = -(base_atr_stop * time_multiplier)

        mml_adjusted_stop = min(
            abs(profit_stop),
            max(mml_stop_distance * 1.3, 0.02),
            0.10
        )

        if current_profit <= 0:
            min_stop = -0.08 if time_in_trade < 1.5 else -0.06
            final_stop = -max(mml_adjusted_stop, abs(min_stop))
        else:
            final_stop = -mml_adjusted_stop

        if final_stop > -0.015 and current_profit < 0.06:
            final_stop = -0.015

        final_stop = max(final_stop, -0.12)

        logger.info(f"{pair} 🛑 DYNAMIC STOP LOSS: t={time_in_trade:.2f}h profit={current_profit:.2%} "
                    f"mult={time_multiplier}x atr={base_atr_stop:.4f} mml={mml_stop_distance:.4f} "
                    f"final={final_stop:.4f}")
        return final_stop

    def minimal_roi_market_adjusted(self, pair: str, trade: Trade, current_time: datetime,
                                    current_rate: float, current_profit: float) -> Dict[int, float]:
        """
        Dynamically adjust ROI based on market conditions (helper - not called).
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return self.minimal_roi

        last_candle = dataframe.iloc[-1]
        market_score = last_candle.get('market_score', 0.5)
        market_regime = last_candle.get('market_regime', 'normal')

        adjusted_roi = self.minimal_roi.copy()
        if market_regime == 'high_volatility' or market_score < 0.3:
            adjusted_roi = {k: v * 0.8 for k, v in adjusted_roi.items()}
            logger.info(f"{pair} ROI adjusted down due to market conditions")
        elif market_score > 0.7 and market_regime == 'bull_run':
            adjusted_roi = {k: v * 1.2 for k, v in adjusted_roi.items()}
            logger.info(f"{pair} ROI adjusted up due to bullish market")
        return adjusted_roi

    # ===========================================
    # POSITION ADJUSTMENT (partial exits + DCA)
    # ===========================================

    def adjust_trade_position(self, trade: Trade, current_time: datetime, current_rate: float,
                              current_profit: float, min_stake: Optional[float], max_stake: float,
                              current_entry_rate: float, current_exit_rate: float, current_entry_profit: float,
                              current_exit_profit: float, **kwargs) -> Optional[float]:

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        last_candle = dataframe.iloc[-1] if dataframe is not None and not dataframe.empty else None

        # V48 left btc_trend undefined when the dataframe was empty (NameError), and the
        # 'btc_trend' column only exists when the legacy correlation is enabled (off by
        # default), so the bear-market rule never fired. Fall back to the enhanced BTC score.
        btc_trend = 0
        market_score = 0.5
        market_breadth = 0.5
        market_regime = 'neutral'
        if last_candle is not None:
            market_score = last_candle.get('market_score', 0.5)
            market_breadth = last_candle.get('market_breadth', 0.5)
            market_regime = last_candle.get('market_regime', 'neutral')
            raw_trend = last_candle.get('btc_trend', np.nan)
            if not pd.isna(raw_trend):
                btc_trend = int(raw_trend)
            else:
                score = last_candle.get('btc_trend_score', np.nan)
                if not pd.isna(score):
                    btc_trend = -1 if score <= 1 else (1 if score >= 4 else 0)

        count_of_entries = trade.nr_of_successful_entries
        count_of_exits = trade.nr_of_successful_exits

        # === PARTIAL PROFIT TAKING ===
        # Amounts are in stake currency of the position (margin). V48 used
        # trade.amount * current_rate (the leveraged notional) which, at leverage >= ~3x,
        # asks to sell more than the whole position.
        def partial(fraction: float) -> float:
            return -(trade.stake_amount * fraction)

        if market_score > 0.75 and current_profit > 0.15 and count_of_exits == 0:
            logger.info(f"{trade.pair} Taking profit early due to market greed: {market_score:.2f}")
            return partial(0.33)

        if current_profit > 0.25 and count_of_exits == 0:
            logger.info(f"{trade.pair} Taking partial profit (25%) at {current_profit:.2%}")
            return partial(0.25)

        # Take profits faster on longs while BTC trends down
        if not trade.is_short and btc_trend < 0:
            if current_profit > 0.10 and count_of_exits == 0:
                logger.info(f"{trade.pair} Bear market quick profit taking at {current_profit:.2%}")
                return partial(0.5)

        if current_profit > 0.40 and count_of_exits == 1:
            logger.info(f"{trade.pair} Taking additional profit (33%) at {current_profit:.2%}")
            return partial(1 / 3)

        # === DCA LOGIC WITH STRICT CONTROLS ===
        if not self.position_adjustment_enable:
            return None

        max_dca_for_pair = self.max_dca_orders
        max_total_stake = self.max_total_stake_per_pair
        max_single_dca = self.max_single_dca_amount

        if count_of_entries > max_dca_for_pair:
            logger.debug(f"{trade.pair} 🛑 MAX DCA REACHED: {count_of_entries}/{max_dca_for_pair}")
            return None

        if trade.stake_amount >= max_total_stake:
            logger.debug(f"{trade.pair} 🛑 MAX STAKE REACHED: {trade.stake_amount:.2f}/{max_total_stake}")
            return None

        if market_breadth < 0.25 and not trade.is_short:
            logger.info(f"{trade.pair} Blocking DCA due to bearish market breadth: {market_breadth:.2%}")
            return None

        dca_multiplier = 1.5 if market_regime == 'high_volatility' else 1.0
        trigger = self.initial_safety_order_trigger.value * dca_multiplier

        if (current_profit > trigger / 2.0 and count_of_entries == 1) or \
           (current_profit > trigger and count_of_entries == 2) or \
           (current_profit > trigger * 1.5 and count_of_entries == 3):
            return None

        if count_of_entries >= self.max_safety_orders.value + 1:
            logger.info(f"{trade.pair} 🛑 max_safety_orders reached: {count_of_entries}")
            return None

        try:
            filled_entry_orders = trade.select_filled_orders(trade.entry_side)
            if not filled_entry_orders:
                logger.error(f"{trade.pair} No filled entry orders found for DCA calculation")
                return None

            base_dca_amount = max_single_dca
            dca_multipliers = [1.0, 0.8, 0.6]
            if count_of_entries <= len(dca_multipliers):
                current_multiplier = dca_multipliers[count_of_entries - 1]
            else:
                current_multiplier = 0.5

            dca_stake_amount = base_dca_amount * current_multiplier

            remaining_budget = max_total_stake - trade.stake_amount
            if dca_stake_amount > remaining_budget:
                if remaining_budget > 1:
                    dca_stake_amount = remaining_budget
                    logger.info(f"{trade.pair} 🔧 DCA capped to remaining budget: {dca_stake_amount:.2f}")
                else:
                    logger.info(f"{trade.pair} 🛑 Insufficient remaining budget: {remaining_budget:.2f}")
                    return None

            if min_stake is not None and dca_stake_amount < min_stake:
                logger.warning(f"{trade.pair} DCA below min_stake. Adjusting to {min_stake:.2f}")
                dca_stake_amount = min_stake

            # max_stake is the limit for THIS additional order (available balance), not the
            # position total - V48 compared it against stake + dca.
            if max_stake is not None and dca_stake_amount > max_stake:
                if max_stake > (min_stake or 0):
                    dca_stake_amount = max_stake
                    logger.warning(f"{trade.pair} DCA reduced due to max_stake: {dca_stake_amount:.2f}")
                else:
                    logger.warning(f"{trade.pair} Cannot DCA due to max_stake limit")
                    return None

            new_total_stake = trade.stake_amount + dca_stake_amount
            if new_total_stake > max_total_stake:
                logger.error(f"{trade.pair} 🚨 SAFETY VIOLATION: Would exceed max total stake!")
                return None

            logger.info(f"{trade.pair} ✅ DCA #{count_of_entries}: +{dca_stake_amount:.2f} "
                        f"(Total: {new_total_stake:.2f}/{max_total_stake})")
            return dca_stake_amount

        except Exception as e:
            logger.error(f"Error calculating DCA stake for {trade.pair}: {e}")
            return None

    # ===========================================
    # LEVERAGE
    # ===========================================

    @staticmethod
    def _last_valid(values, default: float) -> float:
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[~np.isnan(arr)]
        return float(arr[-1]) if arr.size else default

    def leverage(self, pair: str, current_time: datetime, current_rate: float, proposed_leverage: float,
                 max_leverage: float, side: str, **kwargs) -> float:
        window_size = self.leverage_window_size.value
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or len(dataframe) < window_size:
            logger.warning(
                f"{pair} Not enough data ({0 if dataframe is None else len(dataframe)} candles) to calculate "
                f"dynamic leverage (requires {window_size}). Using proposed: {proposed_leverage}")
            return proposed_leverage

        close_prices_series = dataframe["close"].tail(window_size)
        high_prices_series = dataframe["high"].tail(window_size)
        low_prices_series = dataframe["low"].tail(window_size)
        base_leverage = self.leverage_base.value

        rsi_array = ta.RSI(close_prices_series, timeperiod=14)
        atr_array = ta.ATR(high_prices_series, low_prices_series, close_prices_series, timeperiod=14)
        sma_array = ta.SMA(close_prices_series, timeperiod=20)
        macd_output = ta.MACD(close_prices_series, fastperiod=12, slowperiod=26, signalperiod=9)

        current_rsi = self._last_valid(rsi_array, 50.0)
        current_atr = self._last_valid(atr_array, 0.0)
        current_sma = self._last_valid(sma_array, current_rate)

        # TA-Lib returns MACD as a list/tuple (macd, signal, hist) for Series/array input and a
        # DataFrame only for DataFrame input. V48 only handled the DataFrame case, so the MACD
        # rule never fired.
        if isinstance(macd_output, pd.DataFrame):
            macd_hist = macd_output['macdhist'] if 'macdhist' in macd_output.columns else []
        elif isinstance(macd_output, (list, tuple)) and len(macd_output) == 3:
            macd_hist = macd_output[2]
        else:
            macd_hist = []
        current_macd_hist = self._last_valid(macd_hist, 0.0)

        high_volatility = current_atr > 0 and current_rate > 0 and \
            (current_atr / current_rate) > self.leverage_atr_threshold_pct.value

        if side == "long":
            if current_rsi < self.leverage_rsi_low.value:
                base_leverage *= self.leverage_long_increase_factor.value
            elif current_rsi > self.leverage_rsi_high.value:
                base_leverage *= self.leverage_long_decrease_factor.value

            if high_volatility:
                base_leverage *= self.leverage_volatility_decrease_factor.value

            if current_macd_hist > 0:
                base_leverage *= self.leverage_long_increase_factor.value

            if current_sma > 0 and current_rate < current_sma:
                base_leverage *= self.leverage_long_decrease_factor.value
        else:
            # Volatility is direction-agnostic risk; V48 applied no adjustment to shorts at all.
            if high_volatility:
                base_leverage *= self.leverage_volatility_decrease_factor.value

        adjusted_leverage = round(max(1.0, min(base_leverage, max_leverage)), 2)
        logger.info(
            f"{pair} Dynamic Leverage ({side}): {adjusted_leverage:.2f} (Base: {base_leverage:.2f}, RSI: {current_rsi:.2f}, "
            f"ATR: {current_atr:.4f}, MACD Hist: {current_macd_hist:.4f}, SMA: {current_sma:.4f})")
        return adjusted_leverage

    # ===========================================
    # INDICATORS
    # ===========================================

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        pair = metadata['pair']
        dataframe['actual_stoploss'] = np.nan
        dataframe['follow_price_level'] = np.nan

        # --- Core indicators ------------------------------------------------
        dataframe["atr"] = ta.ATR(dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=7)
        if dataframe["atr"].isna().all():
            logger.warning(f"{pair} No valid ATR calculated. Filling with 0 temporarily.")
            dataframe["atr"] = 0.0
        else:
            # Fill NaN with a trailing mean (only affects warm-up candles)
            dataframe["atr"] = dataframe["atr"].fillna(dataframe["atr"].rolling(window=5, min_periods=1).mean())
        dataframe["ema20"] = ta.EMA(dataframe["close"], timeperiod=20)
        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["rsi"] = ta.RSI(dataframe["close"], timeperiod=7)
        dataframe["rsi_overbought"] = (dataframe["rsi"] > 70).astype(int)
        dataframe["rsi_oversold"] = (dataframe["rsi"] < 30).astype(int)

        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0

        # --- Extrema --------------------------------------------------------
        extrema_order = self.indicator_extrema_order.value
        dataframe["maxima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).max()
        ).astype(int)
        dataframe["minima"] = (
            dataframe["close"] == dataframe["close"].shift(1).rolling(window=extrema_order).min()
        ).astype(int)

        dataframe["s_extrema"] = 0
        dataframe.loc[dataframe["minima"] == 1, "s_extrema"] = -1
        dataframe.loc[dataframe["maxima"] == 1, "s_extrema"] = 1

        # Heikin-Ashi close and rolling extrema (plot-only columns)
        dataframe["ha_close"] = (dataframe["open"] + dataframe["high"] + dataframe["low"] + dataframe["close"]) / 4
        if dataframe["ha_close"].isna().all():
            logger.error(f"{pair} 🚨 All ha_close values are NaN!")
        dataframe["minh2"], dataframe["maxh2"] = calculate_minima_maxima(dataframe, self.h2.value)
        dataframe["minh1"], dataframe["maxh1"] = calculate_minima_maxima(dataframe, self.h1.value)
        dataframe["minh0"], dataframe["maxh0"] = calculate_minima_maxima(dataframe, self.h0.value)
        dataframe["mincp"], dataframe["maxcp"] = calculate_minima_maxima(dataframe, self.cp.value)

        # --- Murrey Math levels --------------------------------------------
        murrey_levels = self.calculate_rolling_murrey_math_levels_optimized(
            dataframe, window_size=self.indicator_mml_window.value)
        for level_name in MML_LEVEL_NAMES:
            dataframe[level_name] = murrey_levels.get(level_name, dataframe["close"])

        osc_denominator = (dataframe["[+3/8]P"] - dataframe["[-3/8]P"]).replace(0, np.nan)
        dataframe["mmlextreme_oscillator"] = 100 * ((dataframe["close"] - dataframe["[4/8]P"]) / osc_denominator)

        dataframe["DI_catch"] = np.where(dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1)
        rolling_window_threshold = self.indicator_rolling_window_threshold.value
        dataframe["minima_sort_threshold"] = dataframe["close"].rolling(window=rolling_window_threshold, min_periods=1).min()
        dataframe["maxima_sort_threshold"] = dataframe["close"].rolling(window=rolling_window_threshold, min_periods=1).max()

        rolling_check_window = self.indicator_rolling_check_window.value
        dataframe["minima_check"] = (
            dataframe["minima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0
        ).astype(int)
        dataframe["maxima_check"] = (
            dataframe["maxima"].rolling(window=rolling_check_window, min_periods=1).sum() == 0
        ).astype(int)

        dataframe["volatility_range"] = dataframe["high"] - dataframe["low"]
        dataframe["avg_volatility"] = dataframe["volatility_range"].rolling(window=10).mean()
        dataframe["avg_volume"] = dataframe["volume"].rolling(window=20).mean()

        # ===========================================
        # 🚀 MARKET CONTEXT (all per candle, merged by date)
        # ===========================================
        if self.enable_btc_correlation.value:
            dataframe = self.enhanced_btc_correlation_filter(dataframe, metadata)

        if self.btc_correlation_enabled.value or self.btc_trend_filter_enabled.value:
            dataframe = self.calculate_market_correlation_simple(dataframe, metadata)

        if self.market_breadth_enabled.value:
            try:
                dataframe = self.calculate_market_breadth_weighted(dataframe, metadata)
            except Exception as e:
                logger.warning(f"⚠️ {pair} Market breadth calculation failed: {str(e)}")
                dataframe['market_breadth'] = 0.5
                dataframe['breadth_pairs_checked'] = 0

        if self.regime_filter_enabled.value:
            try:
                dataframe = self.calculate_market_regime_aligned(dataframe, metadata)
            except Exception as e:
                logger.warning(f"⚠️ {pair} Market regime calculation failed: {e}")
                dataframe['btc_market_regime'] = 'neutral'
                dataframe['market_volatility'] = 0.02
                dataframe['market_adx'] = 25.0

        if self.total_mcap_filter_enabled.value:
            try:
                dataframe = self.calculate_total_market_cap_trend(dataframe, metadata)
            except Exception as e:
                logger.warning(f"⚠️ {pair} Market cap trend calculation failed: {e}")
                dataframe['mcap_trend'] = 0.0

        # ===========================================
        # 🆕 TREND / STATE / STRENGTH
        # ===========================================
        try:
            dataframe = self.calculate_enhanced_trend_filters(dataframe)
        except Exception as e:
            logger.warning(f"⚠️ {pair} Enhanced trend filters failed: {e}")
            dataframe['trend_alignment_bull'] = 0.5
            dataframe['trend_alignment_bear'] = 0.5
            if 'adx' not in dataframe.columns:
                dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)

        try:
            dataframe = self.detect_market_state(dataframe)
        except Exception as e:
            logger.warning(f"⚠️ {pair} Market state detection failed: {e}")
            dataframe['market_state'] = 'normal'
            dataframe['market_regime'] = 'neutral'

        try:
            dataframe = self.calculate_signal_strength(dataframe)
        except Exception as e:
            logger.warning(f"⚠️ {pair} Signal strength calculation failed: {e}")
            dataframe['signal_strength'] = 0.5

        # ===========================================
        # MARKET SCORE
        # ===========================================
        dataframe['market_score'] = 0.5

        if 'btc_correlation' in dataframe.columns:
            dataframe['market_score'] += dataframe['btc_correlation'] * 0.15

        if 'btc_correlation_ok' in dataframe.columns:
            dataframe['market_score'] += dataframe['btc_correlation_ok'].astype(float) * 0.20

        if 'market_breadth' in dataframe.columns:
            dataframe['market_score'] += (dataframe['market_breadth'] - 0.5) * 0.25

        if 'mcap_trend' in dataframe.columns:
            dataframe['market_score'] += dataframe['mcap_trend'] * 0.10

        if 'market_regime' in dataframe.columns:
            regime_scores = {
                'bull_run': 0.85,
                'bear_market': 0.15,
                'sideways': 0.50,
                'high_volatility': 0.30,
                'transitional': 0.50,
                'neutral': 0.50
            }
            dataframe['regime_score'] = dataframe['market_regime'].map(lambda x: regime_scores.get(x, 0.5))
            dataframe['market_score'] += (dataframe['regime_score'] - 0.5) * 0.15

        if 'trend_alignment_bull' in dataframe.columns:
            dataframe['market_score'] += dataframe['trend_alignment_bull'] * 0.10
            dataframe['market_score'] -= dataframe['trend_alignment_bear'] * 0.10

        if 'macd_bull' in dataframe.columns:
            dataframe['market_score'] += dataframe['macd_bull'].astype(int) * 0.05
            dataframe['market_score'] -= dataframe['macd_bear'].astype(int) * 0.05

        if 'volume_confirmation' in dataframe.columns:
            dataframe['market_score'] += (dataframe['volume_confirmation'] - 0.5) * 0.10

        dataframe['market_score'] = dataframe['market_score'].clip(0, 1)

        # ===========================================
        # SIGNAL QUALITY INDICATORS
        # ===========================================
        dataframe['signal_persistence'] = 0
        for window in [3, 5, 10]:
            dataframe[f'bullish_persistence_{window}'] = (dataframe['rsi'] < 50).rolling(window).sum() / window
            dataframe[f'bearish_persistence_{window}'] = (dataframe['rsi'] > 50).rolling(window).sum() / window

        dataframe['volume_momentum'] = (
            dataframe['volume'].rolling(3).mean() / dataframe['volume'].rolling(20).mean()
        ).fillna(1.0)

        dataframe['price_momentum_strength'] = abs(dataframe['close'].pct_change(5)).rolling(10).mean()

        if 'market_breadth' in dataframe.columns:
            dataframe['breadth_momentum'] = dataframe['market_breadth'].diff()
            dataframe['breadth_trend'] = dataframe['market_breadth'].rolling(5).mean()
            price_direction = (dataframe['close'] > dataframe['close'].shift(5)).astype(int)
            breadth_direction = (dataframe['market_breadth'] > 0.5).astype(int)
            dataframe['breadth_price_divergence'] = abs(price_direction - breadth_direction)

        if 'market_adx' in dataframe.columns:
            dataframe['regime_strength'] = dataframe['market_adx'] / 50
            dataframe['strong_regime'] = dataframe['market_adx'] > 30

        if len(dataframe) > 0:
            last_row = dataframe.iloc[-1]
            logger.info(f"📊 FINAL MARKET CONDITIONS for {pair}: "
                        f"breadth={last_row.get('market_breadth', 0.5):.2%} "
                        f"regime={last_row.get('market_regime', 'unknown')} "
                        f"score={last_row.get('market_score', 0.5):.2%} "
                        f"btc_ok={last_row.get('btc_correlation_ok', True)}")
        else:
            logger.error(f"🚨 {pair} Empty dataframe after indicators!")

        return dataframe

    def enhanced_btc_correlation_filter(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🛡️ BTC health filter, per candle. Five checks on BTC (close>SMA24, close>SMA50,
        SMA24>SMA50, 1-candle momentum > 0.1%, 30<RSI14<70); btc_correlation_ok is False
        when <= 1 pass. Uses the strategy timeframe, falling back to 4h (merged without
        lookahead). Candles without BTC data stay permissive (ok=True), as in V48.
        """
        pair = metadata['pair']

        if 'BTC' in pair:
            dataframe['btc_correlation_ok'] = True
            dataframe['btc_status'] = 'SELF'
            return dataframe

        try:
            candidates = [self.timeframe] + [tf for tf in ('4h',)
                                             if timeframe_to_minutes(tf) > timeframe_to_minutes(self.timeframe)]
            btc_main, main_tf, btc_pair = None, None, None
            for tf in candidates:
                df_tf, used = self._get_basket_dataframe('BTC', pair, tf, min_len=31)
                if df_tf is not None:
                    btc_main, main_tf, btc_pair = df_tf, tf, used
                    break

            if btc_main is None:
                logger.error(f"{pair} 🚨 NO BTC DATA AVAILABLE - DISABLING BTC CORRELATION")
                dataframe['btc_correlation_ok'] = True
                dataframe['btc_status'] = 'NO_DATA'
                return dataframe

            btc = btc_main[['date', 'close']].copy()
            sma_24 = btc['close'].rolling(24).mean()
            sma_50 = btc['close'].rolling(50).mean()
            rsi = pd.Series(np.asarray(ta.RSI(btc['close'], timeperiod=14), dtype=float), index=btc.index)
            momentum = btc['close'].pct_change()

            btc['btc_trend_score'] = (
                (btc['close'] > sma_24).astype(int) +
                (btc['close'] > sma_50).astype(int) +
                (sma_24 > sma_50).astype(int) +
                (momentum > 0.001).astype(int) +
                ((rsi > 30) & (rsi < 70)).astype(int)
            ).astype(float)

            merged = self._merge_inf_columns(dataframe, btc, main_tf, ['btc_trend_score'])
            score = merged['btc_trend_score']
            dataframe['btc_trend_score'] = score
            dataframe['btc_correlation_ok'] = ~(score <= 1)  # NaN (no data) -> True
            dataframe['btc_status'] = np.where(score.notna(), f"OK_{main_tf}", 'NO_DATA')

            last = score.iloc[-1] if len(score) else np.nan
            logger.info(f"{pair} 🔍 BTC filter ({btc_pair} {main_tf}): signals {last}/5, "
                        f"ok={bool(dataframe['btc_correlation_ok'].iloc[-1]) if len(score) else True}")
            return dataframe

        except Exception as e:
            logger.error(f"{pair} 💥 BTC CORRELATION ERROR: {str(e)} - falling back to no BTC filter")
            dataframe['btc_correlation_ok'] = True
            dataframe['btc_status'] = 'ERROR'
            return dataframe

    @staticmethod
    def _to_utc_ts(value) -> pd.Timestamp:
        if isinstance(value, (int, float, np.integer, np.floating)):
            ts = pd.Timestamp(value, unit='s')
        else:
            ts = pd.Timestamp(value)
        return ts.tz_localize('UTC') if ts.tzinfo is None else ts.tz_convert('UTC')

    def safe_time_diff(self, time1, time2):
        """Minutes between two timestamps; tolerates naive/aware mixes (V48 returned 0 then)."""
        try:
            return (self._to_utc_ts(time1) - self._to_utc_ts(time2)).total_seconds() / 60
        except Exception as e:
            logger.warning(f"Time diff error: {e}")
            return 0

    # ===========================================
    # ENTRY
    # ===========================================

    def populate_entry_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        🔄 Long and short signals with adaptive filters.
        """
        pair = metadata['pair']
        self.get_or_init_reentry_data(pair)

        df["enter_long"] = 0
        df["enter_short"] = 0
        df["enter_tag"] = ""
        df["exit_long"] = 0
        df["exit_short"] = 0

        # ===========================================
        # PRELIMINARY CALCULATIONS
        # ===========================================
        if 'adx' not in df.columns:
            df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
            df['plus_di'] = ta.PLUS_DI(df['high'], df['low'], df['close'], timeperiod=14)
            df['minus_di'] = ta.MINUS_DI(df['high'], df['low'], df['close'], timeperiod=14)

        volatility_high = df['atr'] / df['close'] > 0.03  # >3% ATR/close
        df['volume_rising'] = (df['volume'] > df['volume'].shift(1)) & \
                              (df['volume'] > df['volume'].rolling(5).mean() * 1.2)

        df['green_candle'] = (df['close'] > df['open']).astype(int)
        df['red_candle'] = (df['close'] < df['open']).astype(int)
        df['consecutive_green'] = df['green_candle'].rolling(3).sum()
        df['consecutive_red'] = df['red_candle'].rolling(3).sum()

        df['trend_consistency'] = (df['close'] / df['close'].shift(10) - 1).fillna(0)
        if 'market_score' not in df.columns:
            df['market_score'] = 0.5

        df['below_midpoint'] = df['close'] < df['[4/8]P']
        lookback_candles = getattr(self, 'lookback_candles', 15)
        min_drop_pct = getattr(self, 'min_drop_pct', 0.015)
        df['highest_close'] = df['close'].rolling(lookback_candles).max()
        df['pct_drop_from_high'] = (df['highest_close'] - df['close']) / df['highest_close']
        df['significant_pullback'] = df['pct_drop_from_high'] > min_drop_pct

        momentum_lookback = getattr(self, 'momentum_lookback', 3)
        max_drop_pct = getattr(self, 'max_drop_pct', 0.03)
        df['recent_drop'] = (df['close'].shift(1) - df['close'].rolling(momentum_lookback).min()) / df['close'].shift(1)
        df['momentum_safe'] = df['recent_drop'] <= max_drop_pct

        df['above_midpoint'] = df['close'] > df['[4/8]P']
        min_rise_pct = getattr(self, 'min_rise_pct', 0.015)
        df['lowest_close'] = df['close'].rolling(lookback_candles).min()
        df['pct_rise_from_low'] = (df['close'] - df['lowest_close']) / df['lowest_close']
        df['significant_rise'] = df['pct_rise_from_low'] > min_rise_pct

        max_rise_pct = getattr(self, 'max_rise_pct', 0.03)
        df['recent_rise'] = (df['close'].rolling(momentum_lookback).max() - df['close'].shift(1)) / df['close'].shift(1)
        df['momentum_safe_short'] = df['recent_rise'] <= max_rise_pct

        vol_mean10 = df["volume"].rolling(10).mean()
        vol_mean15 = df["volume"].rolling(15).mean()
        vol_mean20 = df["volume"].rolling(20).mean()

        # MML structure
        bullish_mml = (
            (df["close"] > df["[6/8]P"]) |
            ((df["close"] > df["[4/8]P"]) & (df["close"].shift(5) < df["[4/8]P"].shift(5)))
        )
        bearish_mml = (
            (df["close"] < df["[2/8]P"]) |
            ((df["close"] < df["[4/8]P"]) & (df["close"].shift(5) > df["[4/8]P"].shift(5)))
        )
        range_bound = (
            (df["close"] >= df["[2/8]P"]) &
            (df["close"] <= df["[6/8]P"]) &
            (~bullish_mml) & (~bearish_mml)
        )
        mml_support_bounce = (
            ((df["low"] <= df["[2/8]P"]) & (df["close"] > df["[2/8]P"])) |
            ((df["low"] <= df["[4/8]P"]) & (df["close"] > df["[4/8]P"]))
        )
        mml_resistance_reject = (
            ((df["high"] >= df["[6/8]P"]) & (df["close"] < df["[6/8]P"])) |
            ((df["high"] >= df["[8/8]P"]) & (df["close"] < df["[8/8]P"]))
        )

        # ===========================================
        # LONG SIGNALS
        # ===========================================
        long_signal_mml_breakout = (
            (df["close"] > df["[6/8]P"]) &
            (df["close"].shift(1) <= df["[6/8]P"].shift(1)) &
            (df["volume"] > vol_mean20 * 1.1) &
            (df["rsi"] > 50) &
            (df["close"] > df["ema20"]) &
            (df["below_midpoint"] | (df["rsi"].shift(1) < df["rsi"])) &
            (df['volume_rising']) &
            (~volatility_high | (df["rsi"] < 30))
        )

        long_signal_support_bounce = (
            mml_support_bounce &
            (df["rsi"] < 45) &
            (df["close"] > df["close"].shift(1)) &
            (df["volume"] > vol_mean10 * 1.1) &
            (df["minima"] == 1) &
            (df["significant_pullback"]) &
            (df["momentum_safe"]) &
            (df['volume_rising']) &
            (~volatility_high | (df["rsi"] < 25))
        )

        long_signal_50_reclaim = (
            (df["close"] > df["[4/8]P"]) &
            (df["close"].shift(1) <= df["[4/8]P"].shift(1)) &
            (df["rsi"] > 50) &
            (df["volume"] > vol_mean15 * 1.1) &
            (df["close"] > df["ema20"]) &
            (df["market_score"] > 0.2) &
            (df['adx'] > 20) &
            (df['volume_rising'])
        )

        long_signal_range_long = (
            range_bound &
            (df["low"] <= df["[2/8]P"]) &
            (df["close"] > df["[2/8]P"]) &
            (df["rsi"] < 50) &
            (df["close"] > df["close"].shift(1)) &
            (df["minima"] == 1) &
            (df["volume"] > vol_mean10 * 1.1) &
            (df["significant_pullback"]) &
            (df['volume_rising']) &
            (~volatility_high)
        )

        long_signal_trend_continuation = (
            bullish_mml &
            (df["close"] > df["[5/8]P"]) &
            (df["rsi"].between(40, 65)) &
            (df["close"] > df["ema20"]) &
            (df["volume"] > vol_mean15 * 1.1) &
            (df['adx'] > 25) &
            (df['plus_di'] > df['minus_di']) &
            (df["close"] > df["[4/8]P"]) &
            (df['volume_rising']) &
            (~volatility_high)
        )

        long_signal_extreme_oversold = (
            (df["low"] <= df["[1/8]P"]) &
            (df["close"] > df["[1/8]P"]) &
            (df["rsi"] < 35) &
            (df["volume"] > vol_mean20 * 1.2) &
            (df["close"] > df["close"].shift(1)) &
            (df["minima"] == 1) &
            (df["significant_pullback"]) &
            (df["momentum_safe"]) &
            (df['volume_rising']) &
            (~volatility_high | (df["rsi"] < 20))
        )

        any_long_signal = (
            long_signal_mml_breakout |
            long_signal_support_bounce |
            long_signal_50_reclaim |
            long_signal_range_long |
            long_signal_trend_continuation |
            long_signal_extreme_oversold
        )

        # ===========================================
        # SHORT SIGNALS
        # ===========================================
        false_series = pd.Series(False, index=df.index)
        short_signal_bearish_breakdown = short_signal_resistance_reject = false_series
        short_signal_50_breakdown = short_signal_range_short = false_series
        short_signal_trend_continuation = short_signal_extreme_overbought = false_series

        if self.can_short:
            short_signal_bearish_breakdown = (
                (df["close"] < df["[2/8]P"]) &
                (df["close"].shift(1) >= df["[2/8]P"].shift(1)) &
                (df["volume"] > vol_mean20 * 1.1) &
                (df["rsi"] < 50) &
                (df["close"] < df["ema20"]) &
                (df["above_midpoint"] | (df["rsi"].shift(1) > df["rsi"])) &
                (df['volume_rising']) &
                (~volatility_high | (df["rsi"] > 70))
            )

            short_signal_resistance_reject = (
                mml_resistance_reject &
                (df["rsi"] > 50) &
                (df["close"] < df["close"].shift(1)) &
                (df["volume"] > vol_mean10 * 1.1) &
                (df["maxima"] == 1) &
                (df["significant_rise"]) &
                (df["momentum_safe_short"]) &
                (df['volume_rising']) &
                (~volatility_high | (df["rsi"] > 75))
            )

            short_signal_50_breakdown = (
                (df["close"] < df["[4/8]P"]) &
                (df["close"].shift(1) >= df["[4/8]P"].shift(1)) &
                (df["rsi"] < 50) &
                (df["volume"] > vol_mean15 * 1.1) &
                (df["close"] < df["ema20"]) &
                (df["market_score"] < 0.8) &
                (df['adx'] > 20) &
                (df['volume_rising'])
            )

            short_signal_range_short = (
                range_bound &
                (df["high"] >= df["[6/8]P"]) &
                (df["close"] < df["[6/8]P"]) &
                (df["rsi"] > 50) &
                (df["close"] < df["close"].shift(1)) &
                (df["maxima"] == 1) &
                (df["volume"] > vol_mean10 * 1.1) &
                (df["significant_rise"]) &
                (df['volume_rising']) &
                (~volatility_high)
            )

            short_signal_trend_continuation = (
                bearish_mml &
                (df["close"] < df["[3/8]P"]) &
                (df["rsi"].between(35, 60)) &
                (df["close"] < df["ema20"]) &
                (df["volume"] > vol_mean15 * 1.1) &
                (df['adx'] > 25) &
                (df['minus_di'] > df['plus_di']) &
                (df["close"] < df["[4/8]P"]) &
                (df['volume_rising']) &
                (~volatility_high)
            )

            short_signal_extreme_overbought = (
                (df["high"] >= df["[7/8]P"]) &
                (df["close"] < df["[7/8]P"]) &
                (df["rsi"] > 65) &
                (df["volume"] > vol_mean20 * 1.2) &
                (df["close"] < df["close"].shift(1)) &
                (df["maxima"] == 1) &
                (df["significant_rise"]) &
                (df["momentum_safe_short"]) &
                (df['volume_rising']) &
                (~volatility_high | (df["rsi"] > 80))
            )

        any_short_signal = (
            short_signal_bearish_breakdown |
            short_signal_resistance_reject |
            short_signal_50_breakdown |
            short_signal_range_short |
            short_signal_trend_continuation |
            short_signal_extreme_overbought
        )

        # ===========================================
        # MARKET REGIME FILTER (per candle)
        # V48 took df['market_regime'].iloc[-1] and applied that one regime to the entire
        # history - in a backtest every candle was filtered by the regime of the final candle.
        # ===========================================
        if 'market_regime' in df.columns:
            regime = df['market_regime'].fillna('neutral')
            is_bear = regime == 'bear_market'
            is_bull = regime == 'bull_run'
            any_long_signal = any_long_signal & (~is_bear | (df['rsi'] < 35)) & (~is_bull | (df['rsi'] > 40))
            any_short_signal = any_short_signal & (~is_bear | (df['rsi'] > 45)) & (~is_bull | (df['rsi'] > 65))
            logger.info(f"{pair} 🏛️ Market regime: {regime.iloc[-1] if len(regime) else 'n/a'}")

        # ===========================================
        # RE-ENTRY (live only - needs real exit history recorded in order_filled)
        # ===========================================
        if self.enable_reentry.value and len(df) > 0:
            try:
                current_time = df['date'].iloc[-1] if 'date' in df.columns else datetime.now(timezone.utc)
                recent_exits = self.get_recent_exits(pair, current_time)
                if not recent_exits:
                    # Re-entry window closed - reset per-direction counters
                    self.reentry_count[pair] = {'long': 0, 'short': 0}

                logger.info(f"{pair} 🔄 Re-entry check: {len(recent_exits)} recent exits")

                reentry_long_signals = pd.Series(False, index=df.index)
                reentry_short_signals = pd.Series(False, index=df.index)

                market_score = df['market_score'].fillna(0.5)
                market_breadth = (df['market_breadth'] if 'market_breadth' in df.columns
                                  else pd.Series(0.5, index=df.index)).fillna(0.5)
                btc_ok = (df['btc_correlation_ok'] if 'btc_correlation_ok' in df.columns
                          else pd.Series(True, index=df.index)).fillna(True).astype(bool)

                for exit_time, direction, reason, profit in recent_exits:
                    try:
                        time_since_exit = self.safe_time_diff(current_time, exit_time)
                        current_count = self.reentry_count[pair].get(direction, 0)
                        if current_count >= int(self.max_reentries_per_direction.value):
                            logger.info(f"{pair} 🛑 Max {direction} re-entries: {current_count}")
                            continue
                    except Exception as e:
                        logger.warning(f"{pair} Re-entry processing error: {e}, skipping this exit")
                        continue

                    if direction == 'long':
                        if (reason == 'roi' and profit > 0.015 and
                                self.quick_reentry_cooldown.value <= time_since_exit <= 30):
                            roi_long_conditions = (
                                (df["rsi"] < 55) &
                                (df["close"] > df["[2/8]P"]) &
                                (df["volume"] > vol_mean10) &
                                (market_score > 0.4) &
                                (market_breadth > 0.3) &
                                btc_ok
                            )
                            reentry_long_signals.iloc[-3:] = reentry_long_signals.iloc[-3:] | roi_long_conditions.iloc[-3:]
                            if roi_long_conditions.iloc[-1]:
                                logger.info(f"{pair} 🎯 LONG ROI re-entry: {profit:.2%} profit")

                        elif self.reentry_cooldown_minutes.value <= time_since_exit <= 120:
                            original_long_active = (
                                long_signal_mml_breakout |
                                long_signal_support_bounce |
                                long_signal_50_reclaim |
                                long_signal_range_long |
                                long_signal_extreme_oversold
                            )
                            std_long_conditions = (
                                original_long_active &
                                (df["rsi"] < 45) &
                                (df["close"] > df["[1/8]P"]) &
                                (df["volume"] > vol_mean10) &
                                (market_breadth > 0.35) &
                                btc_ok
                            )
                            reentry_long_signals.iloc[-5:] = reentry_long_signals.iloc[-5:] | std_long_conditions.iloc[-5:]
                            if std_long_conditions.iloc[-1]:
                                logger.info(f"{pair} 🔄 LONG standard re-entry ({time_since_exit:.0f}m)")

                    elif direction == 'short' and self.can_short:
                        if (reason == 'roi' and profit > 0.015 and
                                self.quick_reentry_cooldown.value <= time_since_exit <= 30):
                            roi_short_conditions = (
                                (df["rsi"] > 45) &
                                (df["close"] < df["[6/8]P"]) &
                                (df["volume"] > vol_mean10) &
                                (market_score < 0.6) &
                                (market_breadth < 0.7) &
                                btc_ok
                            )
                            reentry_short_signals.iloc[-3:] = reentry_short_signals.iloc[-3:] | roi_short_conditions.iloc[-3:]
                            if roi_short_conditions.iloc[-1]:
                                logger.info(f"{pair} 🎯 SHORT ROI re-entry: {profit:.2%} profit")

                        elif self.reentry_cooldown_minutes.value <= time_since_exit <= 120:
                            original_short_active = (
                                short_signal_bearish_breakdown |
                                short_signal_resistance_reject |
                                short_signal_50_breakdown |
                                short_signal_range_short |
                                short_signal_extreme_overbought
                            )
                            std_short_conditions = (
                                original_short_active &
                                (df["rsi"] > 55) &
                                (df["close"] < df["[7/8]P"]) &
                                (df["volume"] > vol_mean10) &
                                (market_breadth < 0.65) &
                                btc_ok
                            )
                            reentry_short_signals.iloc[-5:] = reentry_short_signals.iloc[-5:] | std_short_conditions.iloc[-5:]
                            if std_short_conditions.iloc[-1]:
                                logger.info(f"{pair} 🔄 SHORT standard re-entry ({time_since_exit:.0f}m)")

                any_long_signal = any_long_signal | reentry_long_signals
                any_short_signal = any_short_signal | reentry_short_signals

                if reentry_long_signals.iloc[-1]:
                    self.reentry_count[pair]['long'] += 1
                    logger.info(f"{pair} 📈 LONG re-entry #{self.reentry_count[pair]['long']}")
                if reentry_short_signals.iloc[-1]:
                    self.reentry_count[pair]['short'] += 1
                    logger.info(f"{pair} 📉 SHORT re-entry #{self.reentry_count[pair]['short']}")

            except Exception as e:
                logger.error(f"{pair} 💥 Re-entry error: {e}")

        # Market breadth filter
        if 'market_breadth' in df.columns:
            breadth = df['market_breadth'].fillna(0.5)
            any_long_signal = any_long_signal & (breadth > 0.25)
            any_short_signal = any_short_signal & (breadth < 0.75)
            logger.info(f"{pair} 📊 Market breadth: {breadth.iloc[-1]:.1%}")

        # BTC correlation is informational here (filter intentionally disabled in V48)
        if 'btc_correlation_ok' in df.columns:
            btc_ok = df['btc_correlation_ok'].fillna(True).astype(bool)
            logger.info(f"{pair} ₿ BTC correlation: {'✅' if btc_ok.iloc[-1] else '❌'}")

        # Market score filter
        if 'market_score' in df.columns:
            score = df['market_score'].fillna(0.5)
            any_long_signal = any_long_signal & (score > 0.20)
            any_short_signal = any_short_signal & (score < 0.80)
            logger.info(f"{pair} 📈 Market score: {score.iloc[-1]:.1%}")

        any_long_signal = any_long_signal.fillna(False).astype(bool)
        any_short_signal = any_short_signal.fillna(False).astype(bool)

        # ===========================================
        # FINAL SIGNAL ASSIGNMENT
        # ===========================================
        df.loc[any_long_signal, "enter_long"] = 1
        if self.can_short:
            df.loc[any_short_signal, "enter_short"] = 1

        df.loc[any_long_signal & long_signal_mml_breakout, "enter_tag"] = "MML_Breakout_75"
        df.loc[any_long_signal & long_signal_support_bounce & (df["enter_tag"] == ""), "enter_tag"] = "Support_Bounce"
        df.loc[any_long_signal & long_signal_50_reclaim & (df["enter_tag"] == ""), "enter_tag"] = "MML_50_Reclaim"
        df.loc[any_long_signal & long_signal_range_long & (df["enter_tag"] == ""), "enter_tag"] = "Range_Long"
        df.loc[any_long_signal & long_signal_trend_continuation & (df["enter_tag"] == ""), "enter_tag"] = "Trend_Cont"
        df.loc[any_long_signal & long_signal_extreme_oversold & (df["enter_tag"] == ""), "enter_tag"] = "Extreme_Oversold"

        if self.can_short:
            df.loc[any_short_signal & short_signal_bearish_breakdown, "enter_tag"] = "Bear_Breakdown_25"
            df.loc[any_short_signal & short_signal_resistance_reject & (df["enter_tag"] == ""), "enter_tag"] = "Resistance_Reject"
            df.loc[any_short_signal & short_signal_50_breakdown & (df["enter_tag"] == ""), "enter_tag"] = "MML_50_Breakdown"
            df.loc[any_short_signal & short_signal_range_short & (df["enter_tag"] == ""), "enter_tag"] = "Range_Short"
            df.loc[any_short_signal & short_signal_trend_continuation & (df["enter_tag"] == ""), "enter_tag"] = "Bear_Trend_Cont"
            df.loc[any_short_signal & short_signal_extreme_overbought & (df["enter_tag"] == ""), "enter_tag"] = "Extreme_Overbought"

        if self.enable_reentry.value and pair in self.reentry_count:
            total_reentries = self.reentry_count[pair]['long'] + self.reentry_count[pair]['short']
            if total_reentries > 0:
                mask = df["enter_tag"] != ""
                df.loc[mask, "enter_tag"] = df.loc[mask, "enter_tag"] + f"_RE{total_reentries}"

        final_long_count = int(any_long_signal.sum())
        final_short_count = int(any_short_signal.sum())
        logger.info(f"{pair} 🎯 FINAL SIGNALS: Long={final_long_count}, Short={final_short_count}")

        if final_long_count > 0:
            signal_breakdown = {
                'mml_breakout': long_signal_mml_breakout.sum(),
                'support_bounce': long_signal_support_bounce.sum(),
                '50_reclaim': long_signal_50_reclaim.sum(),
                'range_long': long_signal_range_long.sum(),
                'trend_continuation': long_signal_trend_continuation.sum(),
                'extreme_oversold': long_signal_extreme_oversold.sum(),
            }
            logger.debug(f"{pair} 📋 LONG SIGNAL BREAKDOWN: " +
                         ", ".join(f"{k}={v}" for k, v in signal_breakdown.items() if v > 0))

        if final_short_count > 0 and self.can_short:
            short_breakdown = {
                'bearish_breakdown': short_signal_bearish_breakdown.sum(),
                'resistance_reject': short_signal_resistance_reject.sum(),
                '50_breakdown': short_signal_50_breakdown.sum(),
                'range_short': short_signal_range_short.sum(),
                'trend_continuation': short_signal_trend_continuation.sum(),
                'extreme_overbought': short_signal_extreme_overbought.sum(),
            }
            logger.debug(f"{pair} 📋 SHORT SIGNAL BREAKDOWN: " +
                         ", ".join(f"{k}={v}" for k, v in short_breakdown.items() if v > 0))

        df["enter_long"] = df["enter_long"].fillna(0).astype(int)
        df["enter_short"] = df["enter_short"].fillna(0).astype(int)
        df["enter_tag"] = df["enter_tag"].fillna("")

        return df

    def calculate_mml_signal_strengths(self, df: pd.DataFrame, bullish_mml, bearish_mml,
                                       mml_support_bounce, mml_resistance_reject, range_bound) -> pd.DataFrame:
        """
        📊 Calculate signal strength for MML signals (used by apply_mml_reentry_logic - not called).
        """
        mml_long_strength = (
            bullish_mml.astype(float) * 0.4 +
            mml_support_bounce.astype(float) * 0.3 +
            ((df["close"] > df["[4/8]P"]).astype(float) * 0.2) +
            ((df["close"] > df["[6/8]P"]).astype(float) * 0.1)
        ).clip(0, 0.4)
        rsi_long_strength = (
            ((df["rsi"] < 30).astype(float) * 0.2) +
            ((df["rsi"] < 40).astype(float) * 0.15) +
            ((df["rsi"] > df["rsi"].shift(1)).astype(float) * 0.05)
        ).clip(0, 0.2)
        volume_strength = (
            ((df["volume"] > df["volume"].rolling(20).mean() * 1.5).astype(float) * 0.2) +
            ((df["volume"] > df["volume"].rolling(10).mean()).astype(float) * 0.1)
        ).clip(0, 0.2)
        momentum_long_strength = (
            ((df["close"] > df["close"].shift(1)).astype(float) * 0.1) +
            ((df["close"] > df["open"]).astype(float) * 0.1)
        ).clip(0, 0.2)

        mml_short_strength = (
            bearish_mml.astype(float) * 0.4 +
            mml_resistance_reject.astype(float) * 0.3 +
            ((df["close"] < df["[4/8]P"]).astype(float) * 0.2) +
            ((df["close"] < df["[2/8]P"]).astype(float) * 0.1)
        ).clip(0, 0.4)
        rsi_short_strength = (
            ((df["rsi"] > 70).astype(float) * 0.2) +
            ((df["rsi"] > 60).astype(float) * 0.15) +
            ((df["rsi"] < df["rsi"].shift(1)).astype(float) * 0.05)
        ).clip(0, 0.2)
        momentum_short_strength = (
            ((df["close"] < df["close"].shift(1)).astype(float) * 0.1) +
            ((df["close"] < df["open"]).astype(float) * 0.1)
        ).clip(0, 0.2)

        df['long_signal_strength'] = (mml_long_strength + rsi_long_strength + volume_strength +
                                      momentum_long_strength).clip(0, 1)
        df['short_signal_strength'] = (mml_short_strength + rsi_short_strength + volume_strength +
                                       momentum_short_strength).clip(0, 1)
        return df

    def apply_mml_reentry_logic(self, df: pd.DataFrame, pair: str,
                                any_long_signal: pd.Series, any_short_signal: pd.Series) -> tuple:
        """
        Per-candle re-entry helper (not called). V48 version crashed: it called .shift() on a
        scalar, read a non-existent 'volume_avg_10' column and used the RangeIndex as time.
        """
        self.get_or_init_reentry_data(pair)
        if df.empty:
            return any_long_signal, any_short_signal
        current_time = df['date'].iloc[-1]
        recent_exits = self.get_recent_exits(pair, current_time)
        reentry_long_mask = pd.Series(False, index=df.index)
        reentry_short_mask = pd.Series(False, index=df.index)
        if not recent_exits:
            return any_long_signal, any_short_signal

        vol_avg_10 = df['volume'].rolling(10).mean()
        prev_close = df['close'].shift(1)
        threshold = self.reentry_signal_strength_threshold.value
        max_re = self.max_reentries_per_direction.value

        for i in range(len(df)):
            candle_time = df['date'].iloc[i]
            row = df.iloc[i]
            for direction in ('long', 'short'):
                if self.reentry_count[pair][direction] >= max_re:
                    continue
                if not self.should_attempt_reentry(pair, direction, candle_time, recent_exits):
                    continue
                strength_col = f'{direction}_signal_strength'
                if strength_col not in df.columns or row[strength_col] < threshold:
                    continue
                if direction == 'long':
                    ok = (row['close'] > row['[2/8]P'] and row['rsi'] < 45 and
                          row['volume'] > vol_avg_10.iloc[i] and (i == 0 or row['close'] > prev_close.iloc[i]))
                else:
                    ok = (row['close'] < row['[6/8]P'] and row['rsi'] > 55 and
                          row['volume'] > vol_avg_10.iloc[i] and (i == 0 or row['close'] < prev_close.iloc[i]))
                if ok:
                    mask = reentry_long_mask if direction == 'long' else reentry_short_mask
                    mask.iloc[i] = True
                    self.reentry_count[pair][direction] += 1
                    logger.info(f"{pair} 🔄 STANDARD {direction.upper()} RE-ENTRY #{self.reentry_count[pair][direction]}")

        return any_long_signal | reentry_long_mask, any_short_signal | reentry_short_mask

    def set_enhanced_entry_tags(self, df: pd.DataFrame, final_long_signal, final_short_signal,
                                long_signal_mml_breakout, long_signal_support_bounce,
                                long_signal_50_reclaim, long_signal_sextrema,
                                long_signal_rolling_MinH2, short_signal_bearish_breakdown,
                                short_signal_confirmed_max_entry, short_signal_rolling_MaxH2_Entry,
                                pair: str) -> pd.DataFrame:
        """
        🏷️ Set enhanced entry tags that include re-entry information (helper - not called)
        """
        df.loc[final_long_signal & long_signal_mml_breakout, "enter_tag"] = "MML_Bullish_Breakout"
        df.loc[final_long_signal & long_signal_support_bounce & (df["enter_tag"] == ""), "enter_tag"] = "MML_Support_Bounce"
        df.loc[final_long_signal & long_signal_50_reclaim & (df["enter_tag"] == ""), "enter_tag"] = "MML_50_Reclaim"
        df.loc[final_long_signal & long_signal_rolling_MinH2 & (df["enter_tag"] == ""), "enter_tag"] = "Rolling_MinH2"
        df.loc[final_long_signal & long_signal_sextrema & (df["enter_tag"] == ""), "enter_tag"] = "Sextrema"

        if self.can_short:
            df.loc[final_short_signal & short_signal_bearish_breakdown, "enter_tag"] = "MML_Bearish_Breakdown"
            df.loc[final_short_signal & short_signal_rolling_MaxH2_Entry & (df["enter_tag"] == ""), "enter_tag"] = "Rolling_MaxH2_Short"
            df.loc[final_short_signal & short_signal_confirmed_max_entry & (df["enter_tag"] == ""), "enter_tag"] = "Confirmed_Max_Entry_Short"

        if pair in getattr(self, 'reentry_count', {}):
            total_reentries = self.reentry_count[pair]['long'] + self.reentry_count[pair]['short']
            if total_reentries > 0:
                mask = df["enter_tag"] != ""
                df.loc[mask, "enter_tag"] = df.loc[mask, "enter_tag"] + f"_RE{total_reentries}"
        return df

    # ===========================================
    # 🛠️ RE-ENTRY HELPERS
    # ===========================================

    def should_attempt_reentry(self, pair: str, direction: str, current_time: datetime, recent_exits: list) -> bool:
        """Check if we should attempt re-entry"""
        try:
            if self.reentry_count[pair][direction] >= self.max_reentries_per_direction.value:
                return False

            recent_exit_in_direction = None
            for exit_time, exit_direction, exit_reason, profit in recent_exits:
                if exit_direction == direction:
                    recent_exit_in_direction = (exit_time, exit_reason)
                    break
            if not recent_exit_in_direction:
                return False

            exit_time, exit_reason = recent_exit_in_direction
            if exit_reason == 'roi' and self.enable_quick_reentry_on_roi.value:
                cooldown = float(self.quick_reentry_cooldown.value)
            else:
                cooldown = float(self.reentry_cooldown_minutes.value)

            return self.safe_time_diff(current_time, exit_time) >= cooldown

        except Exception as e:
            logger.warning(f"Re-entry check error for {pair} {direction}: {e}")
            return False

    def was_recent_roi_exit(self, pair: str, direction: str, current_time: datetime, recent_exits: list) -> bool:
        """Check if there was a recent ROI exit"""
        try:
            for exit_time, exit_direction, exit_reason, profit in recent_exits:
                if exit_direction == direction and exit_reason == 'roi':
                    if self.safe_time_diff(current_time, exit_time) <= float(self.quick_reentry_cooldown.value):
                        return True
            return False
        except Exception as e:
            logger.warning(f"Recent ROI check error for {pair} {direction}: {e}")
            return False

    def get_recent_exits(self, pair: str, current_time: datetime) -> list:
        """
        Exits for this pair inside the re-entry window. The window is 120 minutes so the
        standard re-entry branch (cooldown .. 120 min) is reachable; V48 used 60 minutes,
        which made reentry_cooldown_minutes values above 60 dead.
        """
        try:
            if not hasattr(self, 'recent_exits') or pair not in self.recent_exits:
                return []
            recent = []
            for exit_time, direction, reason, profit in self.recent_exits[pair]:
                try:
                    if 0 <= self.safe_time_diff(current_time, exit_time) <= 120:
                        recent.append((exit_time, direction, reason, profit))
                except Exception as e:
                    logger.warning(f"Exit time comparison error: {e}")
            return recent
        except Exception as e:
            logger.warning(f"Get recent exits error for {pair}: {e}")
            return []

    def order_filled(self, pair: str, trade: Trade, order: Order, current_time: datetime, **kwargs) -> None:
        """
        Record exits so the (live-only) re-entry logic has data. V48 initialised
        self.recent_exits but never wrote to it, so re-entry could never trigger.
        """
        try:
            if order.ft_order_side != trade.exit_side:
                return
            self.get_or_init_reentry_data(pair)
            direction = 'short' if trade.is_short else 'long'
            reason = trade.exit_reason or getattr(order, 'ft_order_tag', None) or ''
            profit = trade.close_profit
            if profit is None:
                profit = trade.calc_profit_ratio(order.safe_price)
            self.recent_exits[pair].append((current_time, direction, reason, float(profit or 0.0)))
            # keep the list short
            cutoff = self._to_utc_ts(current_time) - pd.Timedelta(hours=3)
            self.recent_exits[pair] = [e for e in self.recent_exits[pair] if self._to_utc_ts(e[0]) >= cutoff]
        except Exception as e:
            logger.debug(f"order_filled bookkeeping error for {pair}: {e}")

    # ===========================================
    # EXIT
    # ===========================================

    def populate_exit_trend(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        UNIFIED EXIT SYSTEM - chooses the advanced MML exits or simple opposite-signal exits.
        """
        df["exit_long"] = 0
        df["exit_short"] = 0
        df["exit_tag"] = ""

        if 'adx' not in df.columns:
            df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)

        if self.use_custom_exits_advanced:
            return self._populate_custom_exits_advanced(df, metadata)
        return self._populate_simple_exits(df, metadata)

    def _populate_custom_exits_advanced(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        ADVANCED MML-BASED EXIT SYSTEM
        """
        pair = metadata['pair']
        if 'adx' not in df.columns:
            df['adx'] = ta.ADX(df['high'], df['low'], df['close'], timeperiod=14)
        for col, default in (('consecutive_red', 0), ('market_score', 0.5), ('market_regime', 'neutral')):
            if col not in df.columns:
                df[col] = default

        vol_mean10 = df["volume"].rolling(10).mean()
        vol_mean15 = df["volume"].rolling(15).mean()
        vol_mean20 = df["volume"].rolling(20).mean()
        atr_mean10 = df["atr"].rolling(10).mean()

        df['volume_fading'] = (df['volume'] < vol_mean20 * 0.8) & (df['volume'] < df['volume'].shift(1))
        volatility_high = df['atr'] / df['close'] > 0.03

        bullish_mml = (
            (df["close"] > df["[6/8]P"]) |
            ((df["close"] > df["[4/8]P"]) & (df["close"].shift(5) < df["[4/8]P"].shift(5)))
        )
        bearish_mml = (
            (df["close"] < df["[2/8]P"]) |
            ((df["close"] < df["[4/8]P"]) & (df["close"].shift(5) > df["[4/8]P"].shift(5)))
        )
        at_resistance = (
            (df["high"] >= df["[6/8]P"]) |
            (df["high"] >= df["[7/8]P"]) |
            (df["high"] >= df["[8/8]P"])
        )
        at_support = (
            (df["low"] <= df["[2/8]P"]) |
            (df["low"] <= df["[1/8]P"]) |
            (df["low"] <= df["[0/8]P"])
        )
        # .shift() on a bool Series yields object dtype with NaN; make it a clean bool
        bullish_mml_prev = bullish_mml.shift(1, fill_value=False).astype(bool)
        bearish_mml_prev = bearish_mml.shift(1, fill_value=False).astype(bool)

        # 1. Profit-Taking Exits
        atr_adjusted = df["atr"].fillna(0).astype(float) * self.atr_multiplier.value
        trailing_stop_level = df["high"] - atr_adjusted
        long_exit_resistance_profit = (
            at_resistance &
            (df["close"] < df["high"]) &
            (df["rsi"] > 60) &
            (df["maxima"] == 1) &
            (df["volume"] > vol_mean10) &
            (df["close"] < trailing_stop_level) &
            (df["market_score"] > 0.3) &
            (df["consecutive_red"] >= 1) &
            (df['adx'] < 25)
        )

        long_exit_extreme_overbought = (
            (df["close"] > df["[7/8]P"]) &
            (df["rsi"] > 75) &
            (df["close"] < df["close"].shift(1)) &
            (df["maxima"] == 1) &
            (df['volume_fading'])
        )

        long_exit_volume_exhaustion = (
            at_resistance &
            (df['volume_fading']) &
            (df["rsi"] > 70) &
            (df["close"] < df["close"].shift(1)) &
            (~volatility_high)
        )

        # 2. Structure Breakdown
        long_exit_structure_breakdown = (
            (df["close"] < df["[4/8]P"]) &
            (df["close"].shift(1) >= df["[4/8]P"].shift(1)) &
            bullish_mml_prev &
            (df["close"] < df["[4/8]P"] * 0.995) &
            (df["close"] < df["close"].shift(1)) &
            (df["close"] < df["close"].shift(2)) &
            (df["rsi"] < 50) &
            (df["volume"] > vol_mean15 * 1.2) &
            (df["close"] < df["open"]) &
            (df["low"] < df["low"].shift(1)) &
            (df["close"] < df["close"].rolling(3).mean()) &
            (df["atr"] > atr_mean10) &
            (df["market_regime"] != 'bull_run') &
            (df['adx'] < 20)
        )

        # 3. Momentum Divergence
        long_exit_momentum_divergence = (
            at_resistance &
            (df["rsi"] < df["rsi"].shift(1)) &
            (df["rsi"].shift(1) < df["rsi"].shift(2)) &
            (df["rsi"] < df["rsi"].shift(3)) &
            (df["close"] >= df["close"].shift(1)) &
            (df["maxima"] == 1) &
            (df["rsi"] > 60) &
            (df["consecutive_red"].shift(1) >= 1)
        )

        # 4. Range Exit
        long_exit_range = (
            (df["close"] >= df["[2/8]P"]) &
            (df["close"] <= df["[6/8]P"]) &
            (df["high"] >= df["[6/8]P"]) &
            (df["close"] < df["[6/8]P"] * 0.995) &
            (df["rsi"] > 65) &
            (df["maxima"] == 1) &
            (df["volume"] > vol_mean10 * 1.2)
        )

        # 5. Emergency Exit
        no_signal = pd.Series(False, index=df.index)
        long_exit_emergency = (
            (df["close"] < df["[0/8]P"]) &
            (df["rsi"] < 25) &
            (df["volume"] > vol_mean20 * 2) &
            (df["close"] < df["close"].shift(1)) &
            (df["close"] < df["close"].shift(2)) &
            (df["close"] < df["open"]) &
            volatility_high
        ) if self.use_emergency_exits else no_signal

        any_long_exit = (
            long_exit_resistance_profit |
            long_exit_extreme_overbought |
            long_exit_volume_exhaustion |
            long_exit_structure_breakdown |
            long_exit_momentum_divergence |
            long_exit_range |
            long_exit_emergency
        )

        # ===========================================
        # SHORT EXIT SIGNALS
        # ===========================================
        short_exit_support_profit = short_exit_extreme_oversold = short_exit_volume_exhaustion = no_signal
        short_exit_structure_breakout = short_exit_momentum_divergence = short_exit_range = no_signal
        short_exit_emergency = no_signal

        if self.can_short:
            atr_adjusted_short = df["atr"].fillna(0).astype(float) * self.atr_multiplier.value
            trailing_stop_level_short = df["low"] + atr_adjusted_short
            short_exit_support_profit = (
                at_support &
                (df["close"] > df["low"]) &
                (df["rsi"] < 40) &
                (df["minima"] == 1) &
                (df["volume"] > vol_mean10) &
                (df["close"] > trailing_stop_level_short) &
                (df["market_score"] < 0.7) &
                (~(df["consecutive_red"] >= 1)) &
                (df['volume_fading'])
            )

            short_exit_extreme_oversold = (
                (df["close"] < df["[1/8]P"]) &
                (df["rsi"] < 25) &
                (df["close"] > df["close"].shift(1)) &
                (df["minima"] == 1) &
                (df['volume_fading'])
            )

            short_exit_volume_exhaustion = (
                at_support &
                (df['volume_fading']) &
                (df["rsi"] < 30) &
                (df["close"] > df["close"].shift(1)) &
                (~volatility_high)
            )

            short_exit_structure_breakout = (
                (df["close"] > df["[4/8]P"]) &
                (df["close"].shift(1) <= df["[4/8]P"].shift(1)) &
                bearish_mml_prev &
                (df["close"] > df["[4/8]P"] * 1.005) &
                (df["close"] > df["close"].shift(1)) &
                (df["close"] > df["close"].shift(2)) &
                (df["rsi"] > 50) &
                (df["volume"] > vol_mean15 * 1.2) &
                (df["close"] > df["open"]) &
                (df["high"] > df["high"].shift(1)) &
                (df["atr"] > atr_mean10) &
                (df["market_regime"] != 'bear_market') &
                (df['adx'] < 20)
            )

            short_exit_momentum_divergence = (
                at_support &
                (df["rsi"] > df["rsi"].shift(1)) &
                (df["rsi"].shift(1) > df["rsi"].shift(2)) &
                (df["rsi"] > df["rsi"].shift(3)) &
                (df["close"] <= df["close"].shift(1)) &
                (df["minima"] == 1) &
                (df["rsi"] < 40)
            )

            short_exit_range = (
                (df["close"] >= df["[2/8]P"]) &
                (df["close"] <= df["[6/8]P"]) &
                (df["low"] <= df["[2/8]P"]) &
                (df["close"] > df["[2/8]P"] * 1.005) &
                (df["rsi"] < 35) &
                (df["minima"] == 1) &
                (df["volume"] > vol_mean10 * 1.2)
            )

            short_exit_emergency = (
                (df["close"] > df["[8/8]P"]) &
                (df["rsi"] > 85) &
                (df["volume"] > vol_mean20 * 2) &
                (df["close"] > df["close"].shift(1)) &
                (df["close"] > df["close"].shift(2)) &
                (df["close"] > df["open"]) &
                volatility_high
            ) if self.use_emergency_exits else no_signal

        any_short_exit = (
            short_exit_support_profit |
            short_exit_extreme_oversold |
            short_exit_volume_exhaustion |
            short_exit_structure_breakout |
            short_exit_momentum_divergence |
            short_exit_range |
            short_exit_emergency
        )

        # Exit on opposite entry signals
        if 'enter_long' in df.columns and 'enter_short' in df.columns:
            df.loc[df['enter_short'] == 1, 'exit_long'] = 1
            df.loc[df['enter_long'] == 1, 'exit_short'] = 1

            df.loc[df['exit_long'] == 1, 'exit_tag'] = 'Reversal_Short_Entry'
            if self.can_short:
                df.loc[df['exit_short'] == 1, 'exit_tag'] = 'Reversal_Long_Entry'
        else:
            logger.warning("Missing 'enter_long' or 'enter_short' columns; skipping reversal-based exits")

        df.loc[any_long_exit, "exit_long"] = 1
        if self.can_short:
            df.loc[any_short_exit, "exit_short"] = 1

        df.loc[any_long_exit & long_exit_emergency, "exit_tag"] = "MML_Emergency_Long_Exit"
        df.loc[any_long_exit & long_exit_structure_breakdown & (df["exit_tag"] == ""), "exit_tag"] = "MML_Structure_Breakdown"
        df.loc[any_long_exit & long_exit_resistance_profit & (df["exit_tag"] == ""), "exit_tag"] = "MML_Resistance_Profit"
        df.loc[any_long_exit & long_exit_extreme_overbought & (df["exit_tag"] == ""), "exit_tag"] = "MML_Extreme_Overbought"
        df.loc[any_long_exit & long_exit_volume_exhaustion & (df["exit_tag"] == ""), "exit_tag"] = "MML_Volume_Exhaustion_Long"
        df.loc[any_long_exit & long_exit_momentum_divergence & (df["exit_tag"] == ""), "exit_tag"] = "MML_Momentum_Divergence_Long"
        df.loc[any_long_exit & long_exit_range & (df["exit_tag"] == ""), "exit_tag"] = "MML_Range_Exit_Long"

        if self.can_short:
            df.loc[any_short_exit & short_exit_emergency, "exit_tag"] = "MML_Emergency_Short_Exit"
            df.loc[any_short_exit & short_exit_structure_breakout & (df["exit_tag"] == ""), "exit_tag"] = "MML_Structure_Breakout"
            df.loc[any_short_exit & short_exit_support_profit & (df["exit_tag"] == ""), "exit_tag"] = "MML_Support_Profit"
            df.loc[any_short_exit & short_exit_extreme_oversold & (df["exit_tag"] == ""), "exit_tag"] = "MML_Extreme_Oversold"
            df.loc[any_short_exit & short_exit_volume_exhaustion & (df["exit_tag"] == ""), "exit_tag"] = "MML_Volume_Exhaustion_Short"
            df.loc[any_short_exit & short_exit_momentum_divergence & (df["exit_tag"] == ""), "exit_tag"] = "MML_Momentum_Divergence_Short"
            df.loc[any_short_exit & short_exit_range & (df["exit_tag"] == ""), "exit_tag"] = "MML_Range_Exit_Short"

        logger.info(f"{pair} 🚪 FINAL EXITS: Long={int(any_long_exit.sum())}, Short={int(any_short_exit.sum())}")
        return df

    def _populate_simple_exits(self, df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        SIMPLE OPPOSITE SIGNAL EXIT SYSTEM (V48 raised NameError: volatility_high undefined)
        """
        volatility_high = df['atr'] / df['close'] > 0.03
        volume_rising = df['volume_rising'] if 'volume_rising' in df.columns else \
            (df['volume'] > df['volume'].shift(1)) & (df['volume'] > df['volume'].rolling(5).mean() * 1.2)
        vol_mean20 = df['volume'].rolling(20).mean()
        atr_mean20 = df['atr'].rolling(20).mean()

        long_exit_on_short = (df["enter_short"] == 1) & volume_rising
        short_exit_on_long = (df["enter_long"] == 1) & volume_rising

        if self.use_emergency_exits:
            sb_down = df['structure_break_bear'] if 'structure_break_bear' in df.columns else pd.Series(False, index=df.index)
            sb_up = df['structure_break_bull'] if 'structure_break_bull' in df.columns else pd.Series(False, index=df.index)
            emergency_long_exit = (
                (df['rsi'] > 85) &
                (df['volume'] > vol_mean20 * 3) &
                (df['close'] < df['open']) &
                (df['close'] < df['low'].shift(1)) &
                volatility_high
            ) | (
                sb_down.astype(bool) &
                (df['volume'] > vol_mean20 * 2.5) &
                (df['atr'] > atr_mean20 * 2)
            )
            emergency_short_exit = (
                (df['rsi'] < 15) &
                (df['volume'] > vol_mean20 * 3) &
                (df['close'] > df['open']) &
                (df['close'] > df['high'].shift(1)) &
                volatility_high
            ) | (
                sb_up.astype(bool) &
                (df['volume'] > vol_mean20 * 2.5) &
                (df['atr'] > atr_mean20 * 2)
            )
        else:
            emergency_long_exit = pd.Series(False, index=df.index)
            emergency_short_exit = pd.Series(False, index=df.index)

        df.loc[long_exit_on_short | emergency_long_exit, 'exit_long'] = 1
        df.loc[short_exit_on_long | emergency_short_exit, 'exit_short'] = 1
        df.loc[emergency_long_exit, 'exit_tag'] = 'Emergency_Long_Exit'
        df.loc[emergency_short_exit, 'exit_tag'] = 'Emergency_Short_Exit'
        df.loc[long_exit_on_short & (df['exit_tag'] == ""), 'exit_tag'] = 'Reversal_Short_Entry'
        df.loc[short_exit_on_long & (df['exit_tag'] == ""), 'exit_tag'] = 'Reversal_Long_Entry'
        return df

    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                           rate: float, time_in_force: str, exit_reason: str,
                           current_time: datetime, **kwargs) -> bool:

        current_profit_ratio = trade.calc_profit_ratio(rate)
        time_in_trade = (current_time - trade.open_date_utc).total_seconds() / 3600

        logger.info(f"🚪 {pair} EXIT REQUEST: {exit_reason} | profit {current_profit_ratio:.2%} "
                    f"| {time_in_trade:.1f}h in trade")

        # Emergency exits fire when freqtrade cannot place/keep a stoploss order. Blocking them
        # (as V48 did) leaves the position unprotected - always allow.
        if exit_reason == "emergency_exit":
            logger.warning(f"{pair} ⚠️ EMERGENCY EXIT allowed ({current_profit_ratio:.2%})")
            return True

        # Manual /forceexit from Telegram/UI. V48 always refused it; now opt-in.
        if exit_reason in ("force_exit", "force_sell") and self.block_force_exits:
            logger.warning(f"{pair} 🛑 BLOCKED FORCE EXIT: {exit_reason} (Profit: {current_profit_ratio:.2%})")
            return False

        return True

    def should_exit_early_warning(self, pair: str, trade: Trade, current_time: datetime,
                                  current_rate: float, current_profit: float) -> bool:
        """
        ⚠️ EARLY WARNING: Exit before stop loss if conditions deteriorate (helper - not called)
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe is None or dataframe.empty:
                return False

            last_candle = dataframe.iloc[-1]
            time_in_trade = (current_time - trade.open_date_utc).total_seconds() / 3600
            if time_in_trade < 0.25:
                return False

            rsi = last_candle.get('rsi', 50)
            if trade.is_short:
                warning_signals = (
                    (rsi < 30 and current_profit < -0.02) or
                    (last_candle.get('close', 0) > last_candle.get('[6/8]P', np.inf)) or
                    (current_profit < -0.04 and time_in_trade > 2)
                )
            else:
                warning_signals = (
                    (rsi > 70 and current_profit < -0.02) or
                    (last_candle.get('close', 0) < last_candle.get('[2/8]P', -np.inf)) or
                    (current_profit < -0.04 and time_in_trade > 2)
                )

            if warning_signals:
                logger.warning(f"⚠️ {pair} EARLY WARNING: Deteriorating conditions detected")
                return True
            return False

        except Exception as e:
            logger.error(f"Early warning error for {pair}: {e}")
            return False

    def validate_backtest_conditions(self, dataframe: pd.DataFrame, metadata: dict) -> None:
        """
        📊 VALIDATION: Check if backtest conditions match live trading (helper - not called;
        V48 referenced an undefined `df` and raised NameError).
        """
        pair = metadata['pair']
        if 'btc_status' not in dataframe.columns:
            logger.error(f"{pair} 🚨 NO BTC CORRELATION DATA IN BACKTEST!")
            return

        btc_status_counts = dataframe['btc_status'].value_counts()
        total_candles = len(dataframe)
        logger.warning(f"{pair} 📊 BACKTEST VALIDATION REPORT: {total_candles} candles")
        for status, count in btc_status_counts.items():
            logger.warning(f"   BTC Status '{status}': {count} candles ({count / total_candles * 100:.1f}%)")

        no_data_percentage = btc_status_counts.get('NO_DATA', 0) / total_candles * 100 if total_candles else 0
        if no_data_percentage > 10:
            logger.error(f"{pair} 🚨 WARNING: {no_data_percentage:.1f}% of backtest has no BTC data!")

        if 'btc_correlation_ok' in dataframe.columns:
            blocked = (~dataframe['btc_correlation_ok'].fillna(True).astype(bool)).mean() * 100
            logger.info(f"{pair} ₿ BTC filter would block {blocked:.1f}% of candles")

    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs) -> Optional[str]:
        """
        🚪 Time-based exits so stale trades don't linger.
        """
        trade_duration_hours = (current_time - trade.open_date_utc).total_seconds() / 3600
        entry_tag = getattr(trade, 'enter_tag', '') or ''

        if 'MML_Breakout_75' in entry_tag:
            if trade_duration_hours > (7 * 24) and current_profit < -0.05:
                logger.info(f"{pair} 🚪 Smart exit MML_Breakout_75: {current_profit:.2%} after {trade_duration_hours/24:.1f} days")
                return "smart_exit_mml75"

        elif 'Bear_Breakdown_25' in entry_tag:
            if trade_duration_hours > (14 * 24) and current_profit < -0.02:
                logger.info(f"{pair} 🚪 Smart exit Bear_Breakdown_25: {current_profit:.2%} after {trade_duration_hours/24:.1f} days")
                return "smart_exit_bear25"

        if trade_duration_hours > (21 * 24):
            logger.info(f"{pair} 🚪 Smart exit long trade: {current_profit:.2%} after {trade_duration_hours/24:.1f} days")
            return "smart_exit_timeout"

        return None

    def bot_loop_start(self, current_time: Optional[datetime] = None, **kwargs) -> None:
        """
        🔍 Log trades that are old or deep in loss (live/dry-run only, throttled to once an hour
        per trade). V48 subtracted an aware datetime from a naive one, raised TypeError on every
        loop and swallowed it, so nothing was ever logged.
        """
        try:
            if self.dp.runmode.value in ('backtest', 'hyperopt'):
                return
            now = current_time or datetime.now(timezone.utc)
            if now.tzinfo is None:
                now = now.replace(tzinfo=timezone.utc)
            if not hasattr(self, '_loop_log_times'):
                self._loop_log_times = {}

            for trade in Trade.get_open_trades():
                last_logged = self._loop_log_times.get(trade.id)
                if last_logged and (now - last_logged) < timedelta(hours=1):
                    continue

                time_in_trade = (now - trade.open_date_utc).total_seconds() / 3600
                df, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
                rate = df['close'].iloc[-1] if df is not None and not df.empty else None
                current_profit = trade.calc_profit_ratio(rate) if rate else 0.0

                logged = False
                if time_in_trade > 48:
                    logger.warning(f"⚠️ OLD TRADE {trade.pair}: {time_in_trade:.1f}h old, Profit: {current_profit:.2%}")
                    logged = True
                if current_profit < -0.10:
                    logger.warning(f"⚠️ LARGE LOSS {trade.pair}: {current_profit:.2%}, Time: {time_in_trade:.1f}h")
                    logged = True
                if logged:
                    self._loop_log_times[trade.id] = now

        except Exception as e:
            logger.debug(f"bot_loop_start debug error: {e}")
