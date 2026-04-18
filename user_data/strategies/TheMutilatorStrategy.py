from __future__ import annotations

import logging
import math
from datetime import datetime, timedelta
from typing import Dict, List, Sequence, Tuple

import numpy as np
import talib.abstract as ta
from pandas import DataFrame, Series

import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.persistence import Trade
from freqtrade.strategy import DecimalParameter, IntParameter, IStrategy, stoploss_from_open
from freqtrade.vendor.qtpylib.indicators import awesome_oscillator, heikinashi, sma, tdi

logger = logging.getLogger(__name__)


def midnight_massacre_oscillator(dataframe: DataFrame, ema_length: int = 5, ema2_length: int = 35) -> Series:
    ema1 = ta.EMA(dataframe, timeperiod=ema_length)
    ema2 = ta.EMA(dataframe, timeperiod=ema2_length)
    close = dataframe['close'].replace(0, np.nan)
    return ((ema1 - ema2) / close) * 100


class TheMutilatorStrategyBase(IStrategy):
    INTERFACE_VERSION: int = 3
    can_short = True

    minimal_roi = {"0": 0.025, "30": 0.018, "90": 0.010, "240": 0.0}
    stoploss = -0.12
    timeframe = '5m'

    fast_ewo = 50
    slow_ewo = 200

    buy_params = {
        "base_nb_candles_buy": 12,
        "ewo_high": 3.023,
        "ewo_high_2": -3.585,
        "low_offset": 0.995,
        "low_offset_2": 0.942,
        "ewo_low": -9.606,
        "rsi_buy": 58,
        "scream_queen_ema_14_factor": 1.041,
        "knife_drop_rsi_14_limit": 44,
        "knife_drop_rsi_4_limit": 6,
    }
    sell_params = {"base_nb_candles_sell": 22, "high_offset": 1.014, "high_offset_2": 1.01}

    base_nb_candles_buy = IntParameter(8, 20, default=buy_params['base_nb_candles_buy'], space='buy', optimize=False)
    base_nb_candles_sell = IntParameter(
        8, 20, default=sell_params['base_nb_candles_sell'], space='sell', optimize=False
    )
    low_offset = DecimalParameter(0.985, 0.995, default=buy_params['low_offset'], space='buy', optimize=True)
    low_offset_2 = DecimalParameter(0.900, 0.990, default=buy_params['low_offset_2'], space='buy', optimize=False)
    high_offset = DecimalParameter(1.005, 1.015, default=sell_params['high_offset'], space='sell', optimize=True)
    high_offset_2 = DecimalParameter(1.010, 1.020, default=sell_params['high_offset_2'], space='sell', optimize=True)

    ewo_low = DecimalParameter(-20.0, -8.0, default=buy_params['ewo_low'], space='buy', optimize=True)
    ewo_high = DecimalParameter(3.0, 3.4, default=buy_params['ewo_high'], space='buy', optimize=True)
    ewo_high_2 = DecimalParameter(-6.0, 12.0, default=buy_params['ewo_high_2'], space='buy', optimize=False)
    rsi_buy = IntParameter(30, 70, default=buy_params['rsi_buy'], space='buy', optimize=False)

    scream_queen_ema_14_factor = DecimalParameter(
        0.8, 1.2, decimals=3, default=buy_params['scream_queen_ema_14_factor'], space='buy', optimize=True
    )
    knife_drop_rsi_4_limit = IntParameter(5, 60, default=buy_params['knife_drop_rsi_4_limit'], space='buy', optimize=True)
    knife_drop_rsi_14_limit = IntParameter(
        5, 60, default=buy_params['knife_drop_rsi_14_limit'], space='buy', optimize=True
    )

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.018

    process_only_new_candles = True
    startup_candle_count = 240

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_stoploss = True

    initial_safety_order_trigger = -0.020
    max_safety_orders = 6
    safety_order_step_scale = 1.2
    safety_order_volume_scale = 1.35
    position_adjustment_enable = True

    threshold = 0.35
    signal_lookback = 48
    signal_summary_window = 24
    snapshot_log_interval = 36
    active_regimes: Tuple[str, ...] = ("bull", "range", "bear")
    regime_name = "all"

    short_rsi_fast_limit = 68
    short_rsi_limit = 60
    short_blowoff_rsi_limit = 72
    short_trend_offset = 1.002

    slippage_protection = {'retries': 3, 'max_slippage': -0.02}
    _log_cache: Dict[str, str] = {}

    @property
    def protections(self) -> List[Dict[str, float]]:
        return [
            {"method": "CooldownPeriod", "stop_duration_candles": 5},
            {
                "method": "MaxDrawdown",
                "lookback_period_candles": 48,
                "trade_limit": 20,
                "stop_duration_candles": 4,
                "max_allowed_drawdown": 0.20,
            },
            {
                "method": "StoplossGuard",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "only_per_pair": False,
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 6,
                "trade_limit": 2,
                "stop_duration_candles": 60,
                "required_profit": 0.02,
            },
            {
                "method": "LowProfitPairs",
                "lookback_period_candles": 24,
                "trade_limit": 4,
                "stop_duration_candles": 2,
                "required_profit": 0.01,
            },
        ]

    def informative_pairs(self):
        return []

    @staticmethod
    def _false_mask(dataframe: DataFrame) -> Series:
        return dataframe['close'].gt(0) & False

    @staticmethod
    def _clean_mask(mask: Series) -> Series:
        return mask.fillna(False)

    @staticmethod
    def _fmt(value: float, digits: int = 4) -> str:
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "-"
        return f"{value:.{digits}f}"

    @classmethod
    def _campfire_table(cls, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
        rows_as_text = [[str(cell) for cell in row] for row in rows]
        widths = [len(str(header)) for header in headers]
        for row in rows_as_text:
            for index, cell in enumerate(row):
                widths[index] = max(widths[index], len(cell))

        border = "+" + "+".join("-" * (width + 2) for width in widths) + "+"
        header_line = "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |"
        lines = [border, header_line, border]
        for row in rows_as_text:
            line = "| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |"
            lines.append(line)
        lines.append(border)
        return "\n".join(lines)

    def _log_once(self, key: str, marker: str, message: str, *args) -> None:
        cache_key = f"{self.__class__.__name__}:{key}"
        if self._log_cache.get(cache_key) == marker:
            return
        self._log_cache[cache_key] = marker
        logger.info(message, *args)

    @staticmethod
    def _carve_tag(dataframe: DataFrame, mask: Series, column: str, tag: str) -> None:
        mask = mask.fillna(False)
        if not mask.any():
            return
        existing = dataframe.loc[mask, column].fillna("")
        dataframe.loc[mask, column] = existing.where(existing.eq(""), existing + " ") + tag

    def _wipe_the_blade(self, dataframe: DataFrame) -> None:
        for column in ('enter_long', 'enter_short', 'exit_long', 'exit_short'):
            dataframe[column] = 0
        dataframe['enter_tag'] = ""
        dataframe['exit_tag'] = ""

    def _regime_mask(self, dataframe: DataFrame, regime: str) -> Series:
        mask = dataframe['market_regime'].eq(regime)
        if regime not in self.active_regimes:
            return mask & False
        return mask

    def _blood_trail_stop_from_open(self, locked_profit: float, current_profit: float, trade: Trade) -> float:
        try:
            return float(stoploss_from_open(locked_profit, current_profit, is_short=trade.is_short))
        except TypeError:
            return float(stoploss_from_open(locked_profit, current_profit))

    def _opening_night_snapshot(self, dataframe: DataFrame, metadata: dict) -> None:
        if dataframe.empty or len(dataframe) % self.snapshot_log_interval != 0:
            return

        pair = metadata.get('pair', 'UNKNOWN')
        last = dataframe.iloc[-1]
        candle_marker = str(last['date']) if 'date' in dataframe.columns else str(len(dataframe))
        recent_regimes = dataframe['market_regime'].tail(self.signal_summary_window).value_counts()
        rows = [
            ["pair", pair, "regime", str(last['market_regime']).upper()],
            ["close", self._fmt(last['close']), "EWO", self._fmt(last['EWO'], 2)],
            ["ADX", self._fmt(last['adx'], 2), "RSI14", self._fmt(last['rsi_14'], 2)],
            ["bb_width", self._fmt(last['bb_width'], 4), "atr_pct", self._fmt(last['atr_pct'], 4)],
            ["regime_24", f"B:{recent_regimes.get('bull', 0)}", "range/bear", f"R:{recent_regimes.get('range', 0)} B:{recent_regimes.get('bear', 0)}"],
        ]
        self._log_once(
            f"snapshot:{pair}",
            candle_marker,
            "🔪 %s %s opening-night snapshot\n%s",
            f"{pair} [{self.regime_name}]",
            self.__class__.__name__,
            self._campfire_table(["Metric", "Value", "Metric", "Value"], rows),
        )

    def _body_count_summary(
        self,
        dataframe: DataFrame,
        metadata: dict,
        signal_specs: Sequence[Tuple[str, str, Series]],
        stage: str,
        emoji: str,
    ) -> None:
        if dataframe.empty:
            return

        last = dataframe.iloc[-1]
        pair = metadata.get('pair', 'UNKNOWN')
        candle_marker = str(last['date']) if 'date' in dataframe.columns else str(len(dataframe))
        rows = []
        triggered = False
        for signal_column, tag, mask in signal_specs:
            mask = self._clean_mask(mask)
            latest = bool(mask.iloc[-1])
            recent_hits = int(mask.tail(self.signal_summary_window).sum())
            if latest or recent_hits > 0:
                triggered = triggered or latest
                side = 'LONG' if signal_column.endswith('long') else 'SHORT'
                rows.append([side, tag, "✅" if latest else "", str(recent_hits)])

        if not triggered:
            return

        table = self._campfire_table(["Side", "Tag", "Now", f"Last{self.signal_summary_window}"], rows)
        self._log_once(
            f"{stage}:{pair}",
            candle_marker,
            "%s %s %s summary\n%s",
            emoji,
            f"{pair} [{self.regime_name}]",
            stage,
            table,
        )

    def _blood_trail_log(self, emoji: str, title: str, rows: Sequence[Sequence[object]], marker: str) -> None:
        self._log_once(
            title,
            marker,
            "%s %s\n%s",
            emoji,
            title,
            self._campfire_table(["Metric", "Value"], rows),
        )

    def pump_dump_protection(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        volume = dataframe['volume']
        dataframe['volume_mean_short'] = volume.rolling(4, min_periods=1).mean()
        dataframe['volume_mean_long'] = volume.shift(288).rolling(48, min_periods=12).mean()
        dataframe['volume_mean_base'] = volume.shift(432).rolling(288, min_periods=48).mean()

        volume_mean_long = dataframe['volume_mean_long'].replace(0, np.nan)
        volume_mean_base = dataframe['volume_mean_base'].replace(0, np.nan)
        dataframe['volume_change_percentage'] = volume_mean_long / volume_mean_base
        dataframe['rsi_mean'] = dataframe['rsi_14'].rolling(48, min_periods=12).mean()
        dataframe['pnd_volume_warn'] = np.where(
            (dataframe['volume_mean_short'] / volume_mean_long > 5.0),
            -1,
            0,
        )
        dataframe['pnd_volume_warn'] = dataframe['pnd_volume_warn'].fillna(0)
        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        heikin_ashi_df = heikinashi(dataframe)
        dataframe['ha_close'] = heikin_ashi_df['close']
        dataframe['ha_open'] = heikin_ashi_df['open']

        dataframe['ema20'] = ta.EMA(dataframe, timeperiod=20)
        dataframe['ema50'] = ta.EMA(dataframe, timeperiod=50)
        dataframe['ema100'] = ta.EMA(dataframe, timeperiod=100)
        dataframe['ema_8'] = ta.EMA(dataframe, timeperiod=8)
        dataframe['ema_14'] = ta.EMA(dataframe, timeperiod=14)
        dataframe['hma_50'] = qtpylib.hull_moving_average(dataframe['close'], window=50)

        for val in self.base_nb_candles_buy.range:
            dataframe[f'ma_buy_{val}'] = ta.EMA(dataframe, timeperiod=val)

        for val in self.base_nb_candles_sell.range:
            dataframe[f'ma_sell_{val}'] = ta.EMA(dataframe, timeperiod=val)

        macd = ta.MACD(dataframe)
        dataframe['macd'] = macd['macd']
        dataframe['macdsignal'] = macd['macdsignal']
        dataframe['macdhist'] = macd['macdhist']
        dataframe['macd_hist_slope'] = dataframe['macdhist'].diff()

        dataframe['adx'] = ta.ADX(dataframe, timeperiod=14)
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['atr_pct'] = dataframe['atr'] / dataframe['close'].replace(0, np.nan)

        dataframe['rsi_fast'] = ta.RSI(dataframe, timeperiod=4)
        dataframe['rsi_slow'] = ta.RSI(dataframe, timeperiod=20)
        dataframe['rsi_4'] = dataframe['rsi_fast']
        dataframe['rsi_14'] = ta.RSI(dataframe, timeperiod=14)

        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe['fastd'] = stoch_fast['fastd']
        dataframe['fastk'] = stoch_fast['fastk']

        bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger['lower']
        dataframe['bb_middleband2'] = bollinger['mid']
        dataframe['bb_upperband2'] = bollinger['upper']
        dataframe['bb_width'] = (
            (dataframe['bb_upperband2'] - dataframe['bb_lowerband2'])
            / dataframe['bb_middleband2'].replace(0, np.nan)
        )

        dataframe['buysignal'] = dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value
        dataframe['sellsignal'] = dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.high_offset.value

        buy_gap = dataframe['ha_close'] - dataframe['buysignal']
        sell_gap = dataframe['ha_close'] - dataframe[f'ma_sell_{self.base_nb_candles_sell.value}']
        buy_gap_std = buy_gap.rolling(self.signal_lookback, min_periods=12).std().replace(0, np.nan)
        sell_gap_std = sell_gap.rolling(self.signal_lookback, min_periods=12).std().replace(0, np.nan)

        dataframe['difference_signal'] = (
            (sell_gap - sell_gap.rolling(self.signal_lookback, min_periods=12).mean()) / sell_gap_std
        )
        dataframe['close_buy_signal'] = (
            (buy_gap - buy_gap.rolling(self.signal_lookback, min_periods=12).mean()) / buy_gap_std
        )
        dataframe['distance'] = buy_gap / dataframe['ha_close'].rolling(self.signal_lookback, min_periods=12).std().replace(0, np.nan)
        dataframe['buy_signal_distance'] = dataframe['distance'].abs() < self.threshold

        dataframe['EWO'] = midnight_massacre_oscillator(dataframe, self.fast_ewo, self.slow_ewo)

        tdi_df = tdi(dataframe['close'])
        dataframe['tdi_rsi'] = tdi_df.get('rsi', 0)
        dataframe['tdi_signal'] = tdi_df.get('rsi_signal', tdi_df.get('signal', 0))
        dataframe['ao'] = awesome_oscillator(dataframe)
        dataframe['ao_delta'] = dataframe['ao'].diff()
        dataframe['sma'] = sma(dataframe['close'], window=14)

        dataframe['trend_strength'] = (
            (dataframe['ema20'] - dataframe['ema50']) / dataframe['close'].replace(0, np.nan)
        ) * 100
        dataframe['ema_spread'] = (
            (dataframe['ema20'] - dataframe['ema100']) / dataframe['close'].replace(0, np.nan)
        ) * 100

        bull_regime = (
            (dataframe['ema20'] > dataframe['ema50'])
            & (dataframe['ema50'] > dataframe['ema100'])
            & (dataframe['close'] > dataframe['ema20'])
            & (dataframe['EWO'] > 0)
            & (dataframe['adx'] > 18)
        )
        bear_regime = (
            (dataframe['ema20'] < dataframe['ema50'])
            & (dataframe['ema50'] < dataframe['ema100'])
            & (dataframe['close'] < dataframe['ema20'])
            & (dataframe['EWO'] < 0)
            & (dataframe['adx'] > 18)
        )
        dataframe['market_regime'] = np.select([bull_regime, bear_regime], ['bull', 'bear'], default='range')

        dataframe = self.pump_dump_protection(dataframe, metadata)

        for column in (
            'difference_signal',
            'close_buy_signal',
            'distance',
            'atr_pct',
            'bb_width',
            'trend_strength',
            'ema_spread',
            'volume_change_percentage',
            'ao_delta',
            'macd_hist_slope',
        ):
            dataframe[column] = dataframe[column].replace([np.inf, -np.inf], np.nan).fillna(0.0)

        self._opening_night_snapshot(dataframe, metadata)
        return dataframe

    def _final_girl_entry_specs(self, dataframe: DataFrame) -> List[Tuple[str, str, Series]]:
        volume_ok = dataframe['volume'] > 0
        not_pnd = dataframe['pnd_volume_warn'].eq(0)

        bull_regime = self._regime_mask(dataframe, 'bull')
        range_regime = self._regime_mask(dataframe, 'range')
        bear_regime = self._regime_mask(dataframe, 'bear')

        bull_pullback = self._clean_mask(
            bull_regime
            & volume_ok
            & not_pnd
            & (dataframe['rsi_fast'] < 38)
            & (dataframe['rsi_14'] < self.rsi_buy.value)
            & (dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset.value)
            & (dataframe['EWO'] > self.ewo_high.value)
            & (dataframe['difference_signal'] < -0.35)
            & (dataframe['macd_hist_slope'] > 0)
        )

        bull_scream_queen = self._clean_mask(
            bull_regime
            & volume_ok
            & not_pnd
            & (dataframe['close'] < dataframe['ema_14'] * self.scream_queen_ema_14_factor.value)
            & (dataframe['rsi_4'] < int(self.knife_drop_rsi_4_limit.value))
            & (dataframe['rsi_14'] < int(self.knife_drop_rsi_14_limit.value))
            & (dataframe['ao_delta'] > 0)
        )

        range_reclaim = self._clean_mask(
            range_regime
            & volume_ok
            & not_pnd
            & qtpylib.crossed_above(dataframe['ha_close'], dataframe['bb_lowerband2'])
            & dataframe['rsi_14'].between(24, 48)
            & (dataframe['fastk'] < 40)
            & (dataframe['difference_signal'] < -0.90)
        )

        bear_capitulation = self._clean_mask(
            bear_regime
            & volume_ok
            & not_pnd
            & (dataframe['close'] < dataframe[f'ma_buy_{self.base_nb_candles_buy.value}'] * self.low_offset_2.value)
            & (dataframe['EWO'] < self.ewo_low.value)
            & (dataframe['rsi_fast'] < 30)
            & (dataframe['rsi_14'] < 34)
            & (dataframe['close'] < dataframe['bb_lowerband2'])
            & (dataframe['ao_delta'] > 0)
        )

        bull_blowoff_short = self._clean_mask(
            bull_regime
            & volume_ok
            & not_pnd
            & (dataframe['close'] > dataframe['bb_upperband2'])
            & (dataframe['rsi_fast'] > self.short_blowoff_rsi_limit)
            & (dataframe['rsi_14'] > self.short_rsi_limit + 6)
            & (dataframe['EWO'] > max(self.ewo_high.value + 1.0, 4.0))
            & (dataframe['macd_hist_slope'] < 0)
        )

        range_reject_short = self._clean_mask(
            range_regime
            & volume_ok
            & not_pnd
            & qtpylib.crossed_below(dataframe['ha_close'], dataframe['bb_upperband2'])
            & (dataframe['rsi_14'] > self.short_rsi_limit)
            & (dataframe['fastk'] > 75)
            & (dataframe['difference_signal'] > 1.0)
        )

        bear_rally_short = self._clean_mask(
            bear_regime
            & volume_ok
            & not_pnd
            & (dataframe['close'] > dataframe['ema20'])
            & (dataframe['close'] > dataframe[f'ma_sell_{self.base_nb_candles_sell.value}'] * self.short_trend_offset)
            & (dataframe['rsi_fast'] > self.short_rsi_fast_limit - 6)
            & (dataframe['rsi_14'] > self.short_rsi_limit - 4)
            & (dataframe['difference_signal'] > 0.60)
        )

        return [
            ('enter_long', 'L|bull|final_girl_pullback', bull_pullback),
            ('enter_long', 'L|bull|scream_queen_revival', bull_scream_queen),
            ('enter_long', 'L|range|cabin_reclaim', range_reclaim),
            ('enter_long', 'L|bear|last_reel_reversal', bear_capitulation),
            ('enter_short', 'S|bull|blood_moon_fade', bull_blowoff_short),
            ('enter_short', 'S|range|campfire_rejection', range_reject_short),
            ('enter_short', 'S|bear|masked_killer_reject', bear_rally_short),
        ]

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self._wipe_the_blade(dataframe)
        signal_specs = self._final_girl_entry_specs(dataframe)

        for signal_column, tag, mask in signal_specs:
            dataframe.loc[mask, signal_column] = 1
            self._carve_tag(dataframe, mask, 'enter_tag', tag)

        cabin_lockdown = self._clean_mask(
            dataframe['pnd_volume_warn'].eq(-1) & (dataframe['enter_long'].eq(1) | dataframe['enter_short'].eq(1))
        )
        if cabin_lockdown.any():
            dataframe.loc[cabin_lockdown, ['enter_long', 'enter_short']] = 0
            self._carve_tag(dataframe, cabin_lockdown, 'enter_tag', 'BLOCK|cabin_lockdown')

        self._body_count_summary(dataframe, metadata, signal_specs, "entry", "🔪")
        if cabin_lockdown.iloc[-1]:
            last = dataframe.iloc[-1]
            self._blood_trail_log(
                "🔪",
                f"{metadata.get('pair', 'UNKNOWN')} cabin-door lockdown",
                [
                    ["regime", str(last['market_regime']).upper()],
                    ["volume_short", self._fmt(last['volume_mean_short'], 2)],
                    ["volume_long", self._fmt(last['volume_mean_long'], 2)],
                    ["enter_tag", last['enter_tag']],
                ],
                marker=str(last['date']) if 'date' in dataframe.columns else str(len(dataframe)),
            )
        return dataframe

    def _final_cut_exit_specs(self, dataframe: DataFrame) -> List[Tuple[str, str, Series]]:
        bull_regime = self._regime_mask(dataframe, 'bull')
        range_regime = self._regime_mask(dataframe, 'range')
        bear_regime = self._regime_mask(dataframe, 'bear')

        exit_long_bull = self._clean_mask(
            bull_regime
            & (
                (dataframe['difference_signal'] > 1.60)
                | (dataframe['close'] > dataframe['bb_upperband2'] * self.high_offset.value)
            )
            & (dataframe['rsi_fast'] > 62)
        )

        exit_long_range = self._clean_mask(
            range_regime
            & (dataframe['close'] > dataframe['bb_middleband2'])
            & (dataframe['rsi_14'] > 56)
            & (dataframe['ao_delta'] < 0)
        )

        exit_long_bear = self._clean_mask(
            bear_regime
            & ((dataframe['close'] > dataframe['ema20']) | (dataframe['difference_signal'] > 0.70))
            & (dataframe['rsi_fast'] > 50)
        )

        exit_short_bull = self._clean_mask(
            bull_regime
            & ((dataframe['close'] < dataframe['ema20']) | (dataframe['difference_signal'] < -0.40))
            & (dataframe['rsi_fast'] < 45)
        )

        exit_short_range = self._clean_mask(
            range_regime
            & ((dataframe['close'] < dataframe['bb_middleband2']) | qtpylib.crossed_above(dataframe['ha_close'], dataframe['sma']))
            & (dataframe['rsi_14'] < 46)
        )

        exit_short_bear = self._clean_mask(
            bear_regime
            & ((dataframe['difference_signal'] < -1.20) | (dataframe['close'] < dataframe['bb_lowerband2']))
            & (dataframe['rsi_fast'] < 38)
        )

        return [
            ('exit_long', 'XL|bull|body_count_exhaustion', exit_long_bull),
            ('exit_long', 'XL|range|midnight_midband_takeprofit', exit_long_range),
            ('exit_long', 'XL|bear|final_reel_takeprofit', exit_long_bear),
            ('exit_short', 'XS|bull|blood_moon_cover', exit_short_bull),
            ('exit_short', 'XS|range|campfire_mean_reversion', exit_short_range),
            ('exit_short', 'XS|bear|graveyard_cover', exit_short_bear),
        ]

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ""
        exit_specs = self._final_cut_exit_specs(dataframe)
        for signal_column, tag, mask in exit_specs:
            dataframe.loc[mask, signal_column] = 1
            self._carve_tag(dataframe, mask, 'exit_tag', tag)

        self._body_count_summary(dataframe, metadata, exit_specs, "exit", "🩸")
        return dataframe

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> float:
        if current_profit >= 0.12:
            locked_profit = 0.060
            stage = "deep_cut_6.0%"
        elif current_profit >= 0.08:
            locked_profit = 0.035
            stage = "deep_cut_3.5%"
        elif current_profit >= 0.04:
            locked_profit = 0.015
            stage = "deep_cut_1.5%"
        else:
            return self.stoploss

        dynamic_stop = max(self.stoploss, self._blood_trail_stop_from_open(locked_profit, current_profit, trade))
        trade_id = getattr(trade, 'id', f"{pair}:{trade.open_date_utc.isoformat()}")
        self._blood_trail_log(
            "🩸",
            f"{pair} blood-trail tightened",
            [
                ["trade_id", trade_id],
                ["side", "SHORT" if trade.is_short else "LONG"],
                ["profit", self._fmt(current_profit, 4)],
                ["stage", stage],
                ["new_stop", self._fmt(dynamic_stop, 4)],
            ],
            marker=f"{trade_id}:{stage}",
        )
        return dynamic_stop

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ):
        trade_age = current_time - trade.open_date_utc
        loss_limit = -0.035 if trade.is_short else -0.030
        max_age = timedelta(days=2)

        if current_profit < loss_limit and trade_age >= max_age:
            reason = 'XS|sunrise_cleanup' if trade.is_short else 'XL|sunrise_cleanup'
            trade_id = getattr(trade, 'id', f"{pair}:{trade.open_date_utc.isoformat()}")
            self._blood_trail_log(
                "🩸",
                f"{pair} sunrise cleanup",
                [
                    ["trade_id", trade_id],
                    ["side", "SHORT" if trade.is_short else "LONG"],
                    ["age_hours", str(int(trade_age.total_seconds() // 3600))],
                    ["profit", self._fmt(current_profit, 4)],
                    ["reason", reason],
                ],
                marker=f"{trade_id}:{reason}",
            )
            return reason
        return None

    def confirm_trade_exit(
        self,
        pair: str,
        trade: Trade,
        order_type: str,
        amount: float,
        rate: float,
        time_in_force: str,
        exit_reason: str,
        current_time: datetime,
        **kwargs,
    ) -> bool:
        exit_reason = exit_reason or kwargs.get('sell_reason') or 'unknown'
        if not self.dp:
            return True

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        if dataframe is None or dataframe.empty:
            return True

        last_candle = dataframe.iloc[-1]
        candle_close = float(last_candle['close'])

        if exit_reason in {'exit_signal', 'sell_signal'}:
            if not trade.is_short:
                strong_uptrend = (
                    (last_candle['market_regime'] == 'bull')
                    and (last_candle['close'] > last_candle['ema20'])
                    and (last_candle['macdhist'] > 0)
                    and (last_candle['rsi_14'] < 72)
                )
                if strong_uptrend:
                    self._blood_trail_log(
                        "🔪",
                        f"{pair} final-girl hold",
                        [
                            ["side", "LONG"],
                            ["reason", exit_reason],
                            ["regime", str(last_candle['market_regime']).upper()],
                            ["close", self._fmt(last_candle['close'])],
                        ],
                        marker=f"{pair}:{current_time.isoformat()}:{exit_reason}:long",
                    )
                    return False
            else:
                strong_downtrend = (
                    (last_candle['market_regime'] == 'bear')
                    and (last_candle['close'] < last_candle['ema20'])
                    and (last_candle['macdhist'] < 0)
                    and (last_candle['rsi_14'] > 28)
                )
                if strong_downtrend:
                    self._blood_trail_log(
                        "🔪",
                        f"{pair} masked-killer hold",
                        [
                            ["side", "SHORT"],
                            ["reason", exit_reason],
                            ["regime", str(last_candle['market_regime']).upper()],
                            ["close", self._fmt(last_candle['close'])],
                        ],
                        marker=f"{pair}:{current_time.isoformat()}:{exit_reason}:short",
                    )
                    return False

        try:
            state = self.slippage_protection['__pair_retries']
        except KeyError:
            state = self.slippage_protection['__pair_retries'] = {}

        slippage = (candle_close / rate) - 1 if trade.is_short else (rate / candle_close) - 1
        if slippage < self.slippage_protection['max_slippage']:
            pair_retries = state.get(pair, 0)
            if pair_retries < self.slippage_protection['retries']:
                state[pair] = pair_retries + 1
                self._blood_trail_log(
                    "🩸",
                    f"{pair} missed-stab retry",
                    [
                        ["side", "SHORT" if trade.is_short else "LONG"],
                        ["slippage", self._fmt(slippage, 4)],
                        ["retry", f"{pair_retries + 1}/{self.slippage_protection['retries']}"],
                        ["reason", exit_reason],
                    ],
                    marker=f"{pair}:{current_time.isoformat()}:{pair_retries + 1}",
                )
                return False

        state[pair] = 0
        return True

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: float,
        max_stake: float,
        **kwargs,
    ):
        if current_profit > self.initial_safety_order_trigger or not self.dp:
            return None

        dataframe, _ = self.dp.get_analyzed_dataframe(trade.pair, self.timeframe)
        if dataframe is None or len(dataframe) < 2:
            return None

        last_candle = dataframe.iloc[-1]
        previous_candle = dataframe.iloc[-2]

        if trade.is_short:
            rebound_confirmed = (last_candle['close'] < previous_candle['close']) and (last_candle['rsi_fast'] < previous_candle['rsi_fast'])
            entry_side = 'sell'
        else:
            rebound_confirmed = (last_candle['close'] > previous_candle['close']) and (last_candle['rsi_fast'] > previous_candle['rsi_fast'])
            entry_side = 'buy'

        if not rebound_confirmed:
            return None

        completed_entries = 0
        for order in trade.orders:
            if order.ft_is_open or order.ft_order_side != entry_side:
                continue
            if order.status == "closed":
                completed_entries += 1

        if not 1 <= completed_entries <= self.max_safety_orders:
            return None

        safety_order_trigger = abs(self.initial_safety_order_trigger) + (
            abs(self.initial_safety_order_trigger)
            * self.safety_order_step_scale
            * (math.pow(self.safety_order_step_scale, completed_entries - 1) - 1)
            / (self.safety_order_step_scale - 1)
        )
        if current_profit > (-1 * abs(safety_order_trigger)):
            return None

        try:
            stake_amount = self.wallets.get_trade_stake_amount(trade.pair, None)
        except Exception as exception:
            logger.info("🩸 Fallback stake sizing for %s after wallet error: %s", trade.pair, str(exception))
            stake_amount = trade.stake_amount

        stake_amount = stake_amount * math.pow(self.safety_order_volume_scale, completed_entries - 1)
        if max_stake:
            stake_amount = min(stake_amount, max_stake)
        if min_stake and stake_amount < min_stake:
            return None

        trade_id = getattr(trade, 'id', f"{trade.pair}:{trade.open_date_utc.isoformat()}")
        self._blood_trail_log(
            "🔪",
            f"{trade.pair} sequel stake",
            [
                ["trade_id", trade_id],
                ["side", "SHORT" if trade.is_short else "LONG"],
                ["order_no", str(completed_entries)],
                ["trigger", self._fmt(-abs(safety_order_trigger), 4)],
                ["profit", self._fmt(current_profit, 4)],
                ["stake", self._fmt(stake_amount, 4)],
                ["regime", str(last_candle['market_regime']).upper()],
            ],
            marker=f"{trade_id}:so:{completed_entries}",
        )
        return stake_amount


class TheMutilatorStrategy(TheMutilatorStrategyBase):
    regime_name = "all"
    active_regimes = ("bull", "range", "bear")


class TheMutilatorStrategyBull(TheMutilatorStrategyBase):
    regime_name = "bull"
    active_regimes = ("bull",)
    minimal_roi = {"0": 0.032, "25": 0.022, "80": 0.014, "240": 0.0}
    stoploss = -0.10
    trailing_stop_positive = 0.004
    trailing_stop_positive_offset = 0.020
    max_safety_orders = 5


class TheMutilatorStrategyRange(TheMutilatorStrategyBase):
    regime_name = "range"
    active_regimes = ("range",)
    minimal_roi = {"0": 0.018, "40": 0.013, "120": 0.008, "300": 0.0}
    stoploss = -0.09
    trailing_stop_positive = 0.003
    trailing_stop_positive_offset = 0.014
    max_safety_orders = 4
    short_rsi_fast_limit = 64
    short_rsi_limit = 58


class TheMutilatorStrategyBear(TheMutilatorStrategyBase):
    regime_name = "bear"
    active_regimes = ("bear",)
    minimal_roi = {"0": 0.022, "30": 0.017, "90": 0.010, "240": 0.0}
    stoploss = -0.08
    trailing_stop_positive = 0.002
    trailing_stop_positive_offset = 0.012
    max_safety_orders = 3
    initial_safety_order_trigger = -0.022
