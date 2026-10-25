import logging
import os
import pickle
import re
import warnings
from datetime import datetime, timedelta, timezone
from importlib import metadata
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import talib.abstract as ta
from scipy.fft import fft, fftfreq
from freqtrade.strategy import (
    IStrategy,
    DecimalParameter,
    IntParameter,
    BooleanParameter,
    stoploss_from_open,
)
from freqtrade.persistence import Trade
from freqtrade.enums import RunMode
from freqtrade.exchange import timeframe_to_seconds

# Suppress deprecation warnings globally
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*pkg_resources.*")
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        GradientBoostingClassifier,
        ExtraTreesClassifier,
        AdaBoostClassifier,
    )
    from sklearn.base import clone
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        roc_auc_score,
        average_precision_score,
    )
    from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
    from sklearn.feature_selection import SelectKBest, f_classif

    SKLEARN_AVAILABLE = True

    # Check sklearn version using modern approach
    try:
        sklearn_version = metadata.version("scikit-learn")
        logger.info(f"Using scikit-learn version: {sklearn_version}")
    except Exception as e:
        logger.debug(f"Could not get sklearn version: {e}")

except ImportError as e:
    logger.warning(f"scikit-learn not available: {e}")
    SKLEARN_AVAILABLE = False

# Modern PyWavelets import
try:
    import pywt

    WAVELETS_AVAILABLE = True
    try:
        pywt_version = metadata.version("PyWavelets")
        logger.info(f"Using PyWavelets version: {pywt_version}")
    except Exception as e:
        logger.debug(f"Could not get PyWavelets version: {e}")
except ImportError as e:
    logger.warning(f"PyWavelets not available: {e}")
    WAVELETS_AVAILABLE = False

# Define Murrey Math level names for consistency
MML_LEVEL_NAMES = [
    "[-3/8]P",
    "[-2/8]P",
    "[-1/8]P",
    "[0/8]P",
    "[1/8]P",
    "[2/8]P",
    "[3/8]P",
    "[4/8]P",
    "[5/8]P",
    "[6/8]P",
    "[7/8]P",
    "[8/8]P",
    "[+1/8]P",
    "[+2/8]P",
    "[+3/8]P",
]


def calculate_minima_maxima(df, window):
    if df is None or df.empty:
        return np.zeros(0), np.zeros(0)

    minima = np.zeros(len(df))
    maxima = np.zeros(len(df))

    for i in range(window, len(df)):
        window_data = df["ha_close"].iloc[i - window : i + 1]
        if (
            df["ha_close"].iloc[i] == window_data.min()
            and (window_data == df["ha_close"].iloc[i]).sum() == 1
        ):
            minima[i] = -window
        if (
            df["ha_close"].iloc[i] == window_data.max()
            and (window_data == df["ha_close"].iloc[i]).sum() == 1
        ):
            maxima[i] = window

    return minima, maxima


def calc_slope_advanced(series, period):
    """
    Enhanced linear regression slope calculation with Wavelet Transform and FFT analysis
    for superior trend detection and noise filtering
    """
    if len(series) < period:
        return 0

    # Use only the last 'period' values for consistency
    y = series.values[-period:]

    # Enhanced data validation
    if np.isnan(y).any() or np.isinf(y).any():
        return 0

    # Check for constant values (no trend)
    if np.all(y == y[0]):
        return 0

    try:
        coeffs = None
        use_level = 0
        # === 1. WAVELET DENOISING ===
        if WAVELETS_AVAILABLE and len(y) >= 8:
            wavelet = "db4"
            try:
                w = pywt.Wavelet(wavelet)
                max_level = pywt.dwt_max_level(len(y), w.dec_len)
                use_level = min(3, max_level)  # cap at 3 but adapt if shorter series
            except Exception:
                use_level = 1
            if use_level >= 1:
                coeffs = pywt.wavedec(y, wavelet, level=use_level, mode="periodization")
                threshold = 0.1 * np.std(coeffs[-1]) if len(coeffs) > 1 else 0.0
                coeffs_thresh = list(coeffs)
                for i in range(1, len(coeffs_thresh)):
                    coeffs_thresh[i] = pywt.threshold(
                        coeffs_thresh[i], threshold, mode="soft"
                    )
                y_denoised = pywt.waverec(coeffs_thresh, wavelet, mode="periodization")
                if len(y_denoised) != len(y):
                    y_denoised = y_denoised[: len(y)]
            else:
                y_denoised = y
        else:
            y_denoised = y

        # === 2. FFT FREQUENCY ANALYSIS ===
        # Analyze dominant frequencies to identify trend components
        if len(y_denoised) >= 4:
            # Apply FFT
            fft_values = fft(y_denoised)
            freqs = fftfreq(len(y_denoised))

            # Get magnitude spectrum
            magnitude = np.abs(fft_values)

            # Find dominant frequency (excluding DC component)
            non_dc_indices = np.where(freqs != 0)[0]
            if len(non_dc_indices) > 0:
                dominant_freq_idx = non_dc_indices[np.argmax(magnitude[non_dc_indices])]
                dominant_freq = freqs[dominant_freq_idx]

                # Calculate trend strength based on frequency content
                trend_frequency_weight = 1.0 / (1.0 + abs(dominant_freq) * 10)
            else:
                trend_frequency_weight = 1.0
        else:
            trend_frequency_weight = 1.0

        # === 3. MULTI-SCALE SLOPE CALCULATION ===
        x = np.linspace(0, period - 1, period)

        # Original slope calculation
        slope_original = np.polyfit(x, y, 1)[0]

        # Wavelet-denoised slope calculation
        slope_denoised = np.polyfit(x, y_denoised, 1)[0]

        # === 4. WAVELET-BASED TREND DECOMPOSITION ===
        # Only when step 1 actually decomposed the series. The old code referenced
        # `coeffs` even when it was never created and always reconstructed at
        # level=3 regardless of the decomposition level actually used.
        if coeffs is not None and use_level >= 1:
            approx_coeffs = coeffs[0]  # Approximation coefficients (trend)
            trend_component = pywt.upcoef(
                "a", approx_coeffs, wavelet, level=use_level, take=len(y)
            )
            if len(trend_component) > len(y):
                trend_component = trend_component[: len(y)]
            elif len(trend_component) < len(y):
                # Pad with last value if needed
                pad_length = len(y) - len(trend_component)
                trend_component = np.pad(trend_component, (0, pad_length), mode="edge")

            # Calculate slope of trend component
            slope_trend = np.polyfit(x, trend_component, 1)[0]
        else:
            slope_trend = slope_denoised

        # === 5. FREQUENCY-WEIGHTED SLOPE COMBINATION ===
        # Weight slopes based on signal characteristics
        weights = {"original": 0.3, "denoised": 0.4, "trend": 0.3}

        # Adjust weights based on noise level
        noise_level = np.std(y - y_denoised) / np.std(y) if np.std(y) > 0 else 0
        if noise_level > 0.1:  # High noise
            weights = {"original": 0.2, "denoised": 0.5, "trend": 0.3}
        elif noise_level < 0.05:  # Low noise
            weights = {"original": 0.4, "denoised": 0.3, "trend": 0.3}

        # Combined slope calculation
        slope_combined = (
            slope_original * weights["original"]
            + slope_denoised * weights["denoised"]
            + slope_trend * weights["trend"]
        )

        # Apply frequency weighting
        final_slope = slope_combined * trend_frequency_weight

        # === 6. ENHANCED VALIDATION ===
        if np.isnan(final_slope) or np.isinf(final_slope):
            return (
                slope_original
                if not (np.isnan(slope_original) or np.isinf(slope_original))
                else 0
            )

        # Normalize extreme slopes
        max_reasonable_slope = np.std(y) / period
        if abs(final_slope) > max_reasonable_slope * 15:
            return np.sign(final_slope) * max_reasonable_slope * 15

        return final_slope

    except Exception:
        # Fallback to enhanced simple method if advanced processing fails
        try:
            # Apply simple moving average smoothing as fallback
            if len(y) >= 3:
                y_smooth = (
                    pd.Series(y)
                    .rolling(window=3, center=True)
                    .mean()
                    .bfill()
                    .ffill()
                    .values
                )
                x = np.linspace(0, period - 1, period)
                slope = np.polyfit(x, y_smooth, 1)[0]

                if not (np.isnan(slope) or np.isinf(slope)):
                    return slope

            # Ultimate fallback: simple difference
            simple_slope = (y[-1] - y[0]) / (period - 1)
            return (
                simple_slope
                if not (np.isnan(simple_slope) or np.isinf(simple_slope))
                else 0
            )

        except Exception:
            return 0


def calculate_advanced_trend_strength_with_wavelets(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Enhanced trend strength calculation using Wavelet Transform and FFT analysis
    """
    try:
        # === WAVELET-ENHANCED SLOPE CALCULATION ===
        dataframe["slope_5_advanced"] = (
            dataframe["close"]
            .rolling(5)
            .apply(lambda x: calc_slope_advanced(x, 5), raw=False)
        )
        dataframe["slope_10_advanced"] = (
            dataframe["close"]
            .rolling(10)
            .apply(lambda x: calc_slope_advanced(x, 10), raw=False)
        )
        dataframe["slope_20_advanced"] = (
            dataframe["close"]
            .rolling(20)
            .apply(lambda x: calc_slope_advanced(x, 20), raw=False)
        )

        # === WAVELET TREND DECOMPOSITION ===
        def wavelet_trend_analysis(series, window=20):
            """Analyze trend using adaptive wavelet (haar/db4), safe levels, symmetric mode, robust threshold."""
            if not WAVELETS_AVAILABLE or len(series) < window:
                return pd.Series([0.0] * len(series), index=series.index)
            results: list[float] = []
            for i in range(len(series)):
                if i < window:
                    results.append(0.0)
                    continue
                window_data = series.iloc[i - window + 1 : i + 1].values
                n = len(window_data)
                if n < 12:
                    results.append(0.0)
                    continue
                wavelet_name = "haar" if n < 24 else "db4"
                try:
                    w = pywt.Wavelet(wavelet_name)
                    max_level = pywt.dwt_max_level(n, w.dec_len)
                except Exception:
                    max_level = 1
                if n < 48:
                    max_level = min(max_level, 2)
                use_level = max(1, min(3, max_level))
                try:
                    coeffs = pywt.wavedec(
                        window_data, wavelet_name, level=use_level, mode="symmetric"
                    )
                    # Estimate sigma from finest detail
                    if len(coeffs) > 1 and len(coeffs[-1]):
                        detail = coeffs[-1]
                        sigma = np.median(np.abs(detail - np.median(detail))) / 0.6745
                        thr = sigma * np.sqrt(2 * np.log(n)) if sigma > 0 else 0.0
                    else:
                        thr = 0.0
                    for j in range(1, len(coeffs)):
                        coeffs[j] = pywt.threshold(coeffs[j], thr, mode="soft")
                    approx = coeffs[0]
                    trend_strength = np.std(approx) / (np.std(window_data) + 1e-9)
                    direction = 0
                    if len(approx) >= 2:
                        direction = 1 if approx[-1] > approx[0] else -1
                    score = trend_strength * direction
                    if not np.isfinite(score):
                        score = 0.0
                    # Clamp extreme outliers
                    results.append(float(np.clip(score, -5, 5)))
                except Exception:
                    results.append(0.0)
            return pd.Series(results, index=series.index)

        # Apply wavelet trend analysis
        dataframe["wavelet_trend_strength"] = wavelet_trend_analysis(dataframe["close"])

        # === FFT-BASED CYCLE DETECTION ===
        def fft_cycle_analysis(series, window=50):
            """Detect market cycles using FFT"""
            if len(series) < window:
                return (
                    pd.Series([0] * len(series), index=series.index),
                    pd.Series([0] * len(series), index=series.index),
                )

            cycle_strength = []
            dominant_period = []

            for i in range(len(series)):
                if i < window:
                    cycle_strength.append(0)
                    dominant_period.append(0)
                    continue

                # Get window data
                window_data = series.iloc[i - window + 1 : i + 1].values

                try:
                    # Remove linear trend
                    x = np.arange(len(window_data))
                    slope, intercept = np.polyfit(x, window_data, 1)
                    detrended = window_data - (slope * x + intercept)

                    # Apply FFT
                    fft_values = fft(detrended)
                    freqs = fftfreq(len(detrended))
                    magnitude = np.abs(fft_values)

                    # Find dominant cycle (excluding DC component)
                    positive_freqs = freqs[1 : len(freqs) // 2]
                    positive_magnitude = magnitude[1 : len(magnitude) // 2]

                    if len(positive_magnitude) > 0:
                        max_idx = np.argmax(positive_magnitude)
                        dominant_freq = positive_freqs[max_idx]
                        dominant_per = 1.0 / (abs(dominant_freq) + 1e-8)

                        # Cycle strength (normalized)
                        cycle_str = positive_magnitude[max_idx] / (
                            np.sum(positive_magnitude) + 1e-8
                        )
                    else:
                        dominant_per = 0
                        cycle_str = 0

                    cycle_strength.append(cycle_str)
                    dominant_period.append(dominant_per)

                except Exception:
                    cycle_strength.append(0)
                    dominant_period.append(0)

            return (
                pd.Series(cycle_strength, index=series.index),
                pd.Series(dominant_period, index=series.index),
            )

        # Apply FFT cycle analysis
        (dataframe["cycle_strength"], dataframe["dominant_cycle_period"]) = (
            fft_cycle_analysis(dataframe["close"])
        )

        # === ENHANCED TREND STRENGTH CALCULATION ===
        # Normalize advanced slopes by price
        dataframe["trend_strength_5_advanced"] = (
            dataframe["slope_5_advanced"] / dataframe["close"] * 100
        )
        dataframe["trend_strength_10_advanced"] = (
            dataframe["slope_10_advanced"] / dataframe["close"] * 100
        )
        dataframe["trend_strength_20_advanced"] = (
            dataframe["slope_20_advanced"] / dataframe["close"] * 100
        )

        # Wavelet-weighted combined trend strength
        dataframe["trend_strength_wavelet"] = (
            dataframe["trend_strength_5_advanced"] * 0.4
            + dataframe["trend_strength_10_advanced"] * 0.35
            + dataframe["trend_strength_20_advanced"] * 0.25
        )

        # Incorporate wavelet trend analysis
        dataframe["trend_strength_combined"] = (
            dataframe["trend_strength_wavelet"] * 0.7
            + dataframe["wavelet_trend_strength"] * 0.3
        )

        # === CYCLE-ADJUSTED TREND STRENGTH ===
        # Adjust trend strength based on cycle analysis
        dataframe["trend_strength_cycle_adjusted"] = dataframe[
            "trend_strength_combined"
        ].copy()

        # Boost trend strength when aligned with dominant cycle
        strong_cycle_mask = dataframe["cycle_strength"] > 0.3
        dataframe.loc[strong_cycle_mask, "trend_strength_cycle_adjusted"] *= (
            1 + dataframe.loc[strong_cycle_mask, "cycle_strength"]
        )

        # === FINAL TREND CLASSIFICATION WITH ADVANCED FEATURES ===
        strong_threshold = 0.02

        # Enhanced trend classification
        dataframe["strong_uptrend_advanced"] = (
            (dataframe["trend_strength_cycle_adjusted"] > strong_threshold)
            & (dataframe["wavelet_trend_strength"] > 0)
            & (dataframe["cycle_strength"] > 0.1)
        )

        dataframe["strong_downtrend_advanced"] = (
            (dataframe["trend_strength_cycle_adjusted"] < -strong_threshold)
            & (dataframe["wavelet_trend_strength"] < 0)
            & (dataframe["cycle_strength"] > 0.1)
        )

        dataframe["ranging_advanced"] = (
            dataframe["trend_strength_cycle_adjusted"].abs() < strong_threshold * 0.5
        ) | (
            dataframe["cycle_strength"] < 0.05
        )  # Very weak cycles indicate ranging

        # === TREND CONFIDENCE SCORE ===
        # Calculate confidence based on agreement between methods
        methods_agreement = (
            (
                np.sign(dataframe["trend_strength_5_advanced"])
                == np.sign(dataframe["trend_strength_10_advanced"])
            ).astype(int)
            + (
                np.sign(dataframe["trend_strength_10_advanced"])
                == np.sign(dataframe["trend_strength_20_advanced"])
            ).astype(int)
            + (
                np.sign(dataframe["trend_strength_wavelet"])
                == np.sign(dataframe["wavelet_trend_strength"])
            ).astype(int)
        )

        dataframe["trend_confidence"] = methods_agreement / 3.0

        # High confidence trends
        dataframe["high_confidence_trend"] = (
            (dataframe["trend_confidence"] >= 0.67)
            & (dataframe["cycle_strength"] > 0.2)
            & (
                dataframe["trend_strength_cycle_adjusted"].abs()
                > strong_threshold * 0.8
            )
        )

        return dataframe

    except Exception as e:
        logger.warning(f"Advanced trend analysis failed: {e}. Using fallback method.")
        # Return dataframe with fallback values
        fallback_columns = [
            "slope_5_advanced",
            "slope_10_advanced",
            "slope_20_advanced",
            "wavelet_trend_strength",
            "cycle_strength",
            "dominant_cycle_period",
            "trend_strength_5_advanced",
            "trend_strength_10_advanced",
            "trend_strength_20_advanced",
            "trend_strength_wavelet",
            "trend_strength_combined",
            "trend_strength_cycle_adjusted",
            "strong_uptrend_advanced",
            "strong_downtrend_advanced",
            "ranging_advanced",
            "trend_confidence",
            "high_confidence_trend",
        ]

        bool_columns = {
            "strong_uptrend_advanced",
            "strong_downtrend_advanced",
            "ranging_advanced",
            "high_confidence_trend",
        }
        for col in fallback_columns:
            dataframe[col] = False if col in bool_columns else 0.0

        return dataframe


# === ADVANCED PREDICTIVE ANALYSIS SYSTEM ===

# Columns that must never be used as ML features: raw OHLCV, signal columns, and
# non-stationary absolute price / volume levels (they make tree models memorise the
# price range of the training window instead of learning patterns).
FEATURE_EXCLUDE_EXACT = {
    "open",
    "high",
    "low",
    "close",
    "volume",
    "date",
    "enter_long",
    "enter_short",
    "exit_long",
    "exit_short",
    "enter_tag",
    "exit_tag",
    "entry_type",
    "btc_close",
    "ha_close",
    "ema50",
    "ema100",
    "atr",
    "minima_sort_threshold",
    "maxima_sort_threshold",
    "volatility_range",
    "avg_volatility",
    "avg_volume",
    "volume_sma20",
    "volume_sma50",
    "vpt",
    "market_impact",
    "slope_5",
    "slope_10",
    "slope_20",
    "slope_5_advanced",
    "slope_10_advanced",
    "slope_20_advanced",
    "heisenberg_uncertainty",
    *MML_LEVEL_NAMES,
}
FEATURE_EXCLUDE_PREFIXES = (
    "_ext_",  # per-candidate extrema columns precomputed for hyperopt
    "enter_",
    "exit_",
    "ml_",
    "quantum_",
    "neural_",
    "momentum_entanglement",
    "pattern_prediction",
)


def _utcnow() -> datetime:
    """Timezone-aware UTC now (datetime.utcnow() is deprecated since Python 3.12)."""
    return datetime.now(timezone.utc)


class AdvancedPredictiveEngine:
    """
    Machine learning engine for trade entry prediction.

    One model bundle is kept per (pair, direction). "long" models predict the
    probability of a profitable up-move, "short" models the probability of a
    profitable down-move. Bundles are persisted as a single pickle per key and
    loaded lazily the first time a pair is seen.
    """

    MODEL_VERSION = 2

    def __init__(self, models_dir: Optional[Path] = None):
        # Per-key state, key = "<pair>|<direction>"
        self.models: dict[str, dict] = {}
        self.model_weights: dict[str, dict[str, float]] = {}
        self.scalers: dict[str, object] = {}
        self.feature_columns: dict[str, list[str]] = {}
        self.metrics: dict[str, dict] = {}
        self.skill_factor: dict[str, float] = {}
        self.is_trained: dict[str, bool] = {}
        self.last_train_time: dict[str, datetime] = {}
        self._load_attempted: set[str] = set()
        self._wf_cache: dict[tuple, tuple[pd.Series, pd.Series, pd.Series]] = {}

        # Training configuration
        self.retrain_interval_hours: int = 48
        self.initial_train_candles: int = 1000  # rolling training window
        self.forward_periods: int = 5  # label horizon (candles)
        self.min_train_rows: int = 300
        self.max_features: int = 30
        self.live_fast_mode: bool = False  # True = skip hyper-parameter search live
        self.backtest_retrain_step: int = 720  # walk-forward retrain every N candles
        self.refit_on_full_window: bool = True  # refit on train+test after evaluation

        self.models_dir = (
            Path(models_dir) if models_dir else Path("user_data/strategies/ml_models")
        )

    # ----------------- PATHS / PERSISTENCE -----------------
    def set_models_dir(self, models_dir: Path) -> None:
        self.models_dir = Path(models_dir)

    @staticmethod
    def _key(pair: str, direction: str) -> str:
        return f"{pair}|{direction}"

    def _bundle_path(self, pair: str, direction: str) -> Path:
        safe_pair = re.sub(r"[^A-Za-z0-9]+", "_", pair).strip("_")
        return self.models_dir / f"{safe_pair}_{direction}_bundle_v{self.MODEL_VERSION}.pkl"

    def _save_bundle(self, pair: str, direction: str) -> None:
        key = self._key(pair, direction)
        try:
            self.models_dir.mkdir(parents=True, exist_ok=True)
            bundle = {
                "version": self.MODEL_VERSION,
                "pair": pair,
                "direction": direction,
                "models": self.models[key],
                "weights": self.model_weights[key],
                "scaler": self.scalers[key],
                "feature_columns": self.feature_columns[key],
                "metrics": self.metrics.get(key, {}),
                "skill_factor": self.skill_factor.get(key, 1.0),
                "timestamp": self.last_train_time[key].isoformat(),
            }
            path = self._bundle_path(pair, direction)
            tmp_path = path.with_suffix(".tmp")
            with open(tmp_path, "wb") as f:
                pickle.dump(bundle, f)
            os.replace(tmp_path, path)  # atomic: never leaves a half-written model
            logger.info(f"[ML] Saved {direction} model bundle for {pair} -> {path.name}")
        except Exception as e:
            logger.warning(f"[ML] Failed to save models for {key}: {e}")

    def _ensure_loaded(self, pair: str, direction: str) -> None:
        key = self._key(pair, direction)
        if key in self._load_attempted:
            return
        self._load_attempted.add(key)
        path = self._bundle_path(pair, direction)
        if not path.exists():
            return
        try:
            with open(path, "rb") as f:
                bundle = pickle.load(f)
            if bundle.get("version") != self.MODEL_VERSION:
                logger.info(f"[ML] Ignoring outdated model bundle {path.name}")
                return
            self.models[key] = bundle["models"]
            self.model_weights[key] = bundle["weights"]
            self.scalers[key] = bundle["scaler"]
            self.feature_columns[key] = bundle["feature_columns"]
            self.metrics[key] = bundle.get("metrics", {})
            self.skill_factor[key] = bundle.get("skill_factor", 1.0)
            ts = datetime.fromisoformat(bundle["timestamp"])
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            self.last_train_time[key] = ts
            self.is_trained[key] = True
            logger.info(f"[ML] Loaded {direction} model bundle for {pair} (trained {ts})")
        except Exception as e:
            logger.warning(f"[ML] Failed to load model bundle {path.name}: {e}")

    def cleanup_old_models(self, max_age_days: int = 7) -> None:
        """Remove model files older than max_age_days."""
        try:
            if not self.models_dir.exists():
                return
            cutoff = _utcnow().timestamp() - max_age_days * 86400
            for model_file in self.models_dir.glob("*.pkl"):
                if model_file.stat().st_mtime < cutoff:
                    model_file.unlink()
                    logger.info(f"[ML] Removed old model file: {model_file.name}")
        except Exception as e:
            logger.warning(f"[ML] Failed to cleanup old models: {e}")

    def has_model(self, pair: str, direction: str = "long") -> bool:
        self._ensure_loaded(pair, direction)
        return self.is_trained.get(self._key(pair, direction), False)

    def needs_training(self, pair: str, direction: str, now: Optional[datetime] = None) -> bool:
        """True if no model exists or the existing one is older than retrain_interval_hours."""
        key = self._key(pair, direction)
        self._ensure_loaded(pair, direction)
        if not self.is_trained.get(key, False):
            return True
        now = now or _utcnow()
        last = self.last_train_time.get(key)
        if last is None:
            return True
        return (now - last).total_seconds() / 3600.0 >= self.retrain_interval_hours

    # ----------------- FEATURE ENGINEERING -----------------
    def extract_advanced_features(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Extract features for ML prediction. All features use past data only."""
        df = dataframe.copy()
        new_cols: dict[str, pd.Series] = {}

        close = df["close"]
        returns = close.pct_change(1)

        # === 1. PRICE ACTION ===
        for period in [1, 2, 3, 5]:
            vel = close.pct_change(period)
            new_cols[f"price_velocity_{period}"] = vel
            new_cols[f"price_acceleration_{period}"] = vel.diff(1)
            new_cols[f"price_velocity_std_{period}"] = vel.rolling(20).std()
            new_cols[f"price_velocity_skew_{period}"] = vel.rolling(20).skew()

        vol_20 = returns.rolling(20).std()
        new_cols["vol_adjusted_momentum"] = returns / (vol_20 + 1e-10)
        if "atr" in df.columns:
            new_cols["atr_pct"] = df["atr"] / (close + 1e-10)

        for window in [10, 20, 50]:
            high_window = df["high"].rolling(window).max()
            low_window = df["low"].rolling(window).min()
            new_cols[f"price_position_{window}"] = (close - low_window) / (
                high_window - low_window + 1e-10
            )

        # Support / resistance distance. (The old version filled these with
        # np.random noise when the columns were missing, which made training and
        # prediction non-deterministic. They are now simply skipped.)
        if "minima_sort_threshold" in df.columns:
            support_distance = (df["low"] - df["minima_sort_threshold"]).abs() / (close + 1e-10)
            new_cols["sr_support_strength"] = (support_distance < 0.02).astype(int).rolling(20).mean()
            new_cols["support_distance_norm"] = support_distance
        if "maxima_sort_threshold" in df.columns:
            resistance_distance = (df["high"] - df["maxima_sort_threshold"]).abs() / (close + 1e-10)
            new_cols["sr_resistance_strength"] = (
                (resistance_distance < 0.02).astype(int).rolling(20).mean()
            )
            new_cols["resistance_distance_norm"] = resistance_distance

        # === 2. VOLUME DYNAMICS ===
        new_cols["volume_profile_score"] = self._calculate_volume_profile_score(df)
        new_cols["volume_imbalance"] = self._calculate_volume_imbalance(df)
        new_cols["smart_money_index"] = self._calculate_smart_money_index(df)
        new_cols["volume_price_correlation"] = df["volume"].rolling(20).corr(close)
        new_cols["volume_breakout_strength"] = self._calculate_volume_breakout_strength(df)

        # === 3. VOLATILITY CLUSTERING ===
        new_cols["volatility_regime"] = self._calculate_volatility_regime(df)
        new_cols["volatility_persistence"] = self._calculate_volatility_persistence(df)
        new_cols["volatility_mean_reversion"] = self._calculate_volatility_mean_reversion(df)

        # === 4. MOMENTUM DECOMPOSITION ===
        for period in [3, 5, 8, 13, 21]:
            mom = close.pct_change(period)
            new_cols[f"momentum_{period}"] = mom
            new_cols[f"momentum_strength_{period}"] = mom.abs()
            new_cols[f"momentum_consistency_{period}"] = np.sign(mom).rolling(5).mean()

        new_cols["momentum_regime"] = self._classify_momentum_regime(df)
        new_cols["momentum_divergence_strength"] = self._calculate_momentum_divergence(df)

        # === 5. MICROSTRUCTURE ===
        spread_proxy = (df["high"] - df["low"]) / close
        new_cols["spread_proxy"] = spread_proxy
        new_cols["order_flow_imbalance"] = self._calculate_order_flow_imbalance(df)
        new_cols["liquidity_index"] = self._calculate_liquidity_index(df)

        # === 6. STATISTICAL FEATURES (vectorised, much faster than rolling.apply) ===
        for window in [10, 20, 50]:
            new_cols[f"skewness_{window}"] = returns.rolling(window).skew()
            new_cols[f"kurtosis_{window}"] = returns.rolling(window).kurt()
            new_cols[f"entropy_{window}"] = self._calculate_entropy(close, window)

        # === 7. REGIME DETECTION ===
        regime = self._detect_market_regime(df)
        new_cols["market_regime"] = regime
        regime_changes = regime.diff(1).abs()
        new_cols["regime_stability"] = 1 / (regime_changes.rolling(20).mean() + 1e-3)
        new_cols["regime_transition_probability"] = (
            (regime_changes > 0.5).astype(int).rolling(20).mean()
        )

        # Assign all at once (avoids pandas fragmentation warnings and is faster)
        new_df = pd.DataFrame(new_cols, index=df.index)
        df = pd.concat([df.drop(columns=[c for c in new_df.columns if c in df.columns]), new_df], axis=1)
        return df

    def _calculate_volume_profile_score(self, df: pd.DataFrame, window: int = 50) -> pd.Series:
        """Distance of price from the rolling volume Point of Control (1 = at POC)."""
        prices_all = df["close"].to_numpy(dtype=float)
        volumes_all = df["volume"].to_numpy(dtype=float)
        result = np.full(len(df), 0.5)
        for i in range(window, len(df)):
            prices = prices_all[i - window + 1 : i + 1]
            volumes = volumes_all[i - window + 1 : i + 1]
            price_min, price_max = prices.min(), prices.max()
            if price_max == price_min:
                continue
            bins = np.linspace(price_min, price_max, 10)
            vol_at_price, _ = np.histogram(prices, bins=bins, weights=volumes)
            if vol_at_price.sum() == 0:
                continue
            poc_index = int(np.argmax(vol_at_price))
            poc_price = (bins[poc_index] + bins[poc_index + 1]) / 2
            distance_ratio = abs(prices[-1] - poc_price) / (price_max - price_min + 1e-10)
            result[i] = max(0.0, min(1.0, 1 - distance_ratio))
        return pd.Series(result, index=df.index)

    def _calculate_volume_imbalance(self, df: pd.DataFrame) -> pd.Series:
        up_volume = df["volume"].where(df["close"] > df["open"], 0)
        down_volume = df["volume"].where(df["close"] < df["open"], 0)
        imbalance = (up_volume - down_volume) / (up_volume + down_volume + 1e-10)
        return imbalance.rolling(10).mean()

    def _calculate_smart_money_index(self, df: pd.DataFrame) -> pd.Series:
        price_change = df["close"].pct_change(1).abs()
        volume_norm = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
        # log1p keeps the ratio from exploding on near-zero price changes
        smi = np.log1p(volume_norm / (price_change + 1e-4))
        return smi.rolling(10).mean()

    def _calculate_volume_breakout_strength(self, df: pd.DataFrame) -> pd.Series:
        volume_ratio = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
        price_breakout = (
            (df["close"] > df["close"].rolling(20).max().shift(1))
            | (df["close"] < df["close"].rolling(20).min().shift(1))
        ).astype(int)
        return (volume_ratio * price_breakout).rolling(5).mean()

    def _calculate_volatility_regime(self, df: pd.DataFrame) -> pd.Series:
        volatility = df["close"].pct_change(1).rolling(20).std()
        vol_ma = volatility.rolling(50).mean()
        regime = pd.Series(1, index=df.index)
        regime[volatility < vol_ma * 0.7] = 0
        regime[volatility > vol_ma * 1.5] = 2
        return regime

    def _calculate_volatility_persistence(self, df: pd.DataFrame) -> pd.Series:
        volatility = df["close"].pct_change(1).rolling(5).std()
        # Rolling lag-1 autocorrelation, vectorised
        return volatility.rolling(20).corr(volatility.shift(1))

    def _calculate_volatility_mean_reversion(self, df: pd.DataFrame) -> pd.Series:
        volatility = df["close"].pct_change(1).rolling(10).std()
        vol_ma = volatility.rolling(50).mean()
        return -(volatility - vol_ma) / (volatility.rolling(50).std() + 1e-10)

    def _classify_momentum_regime(self, df: pd.DataFrame) -> pd.Series:
        mom_3 = df["close"].pct_change(3)
        mom_8 = df["close"].pct_change(8)
        mom_21 = df["close"].pct_change(21)
        regime = pd.Series(0, index=df.index)
        strong_up = (mom_3 > 0.02) & (mom_8 > 0.05) & (mom_21 > 0.1)
        regime[(mom_3 > 0) & (mom_8 > 0) & (mom_21 > 0) & ~strong_up] = 1
        regime[strong_up] = 2
        strong_down = (mom_3 < -0.02) & (mom_8 < -0.05) & (mom_21 < -0.1)
        regime[(mom_3 < 0) & (mom_8 < 0) & (mom_21 < 0) & ~strong_down] = -1
        regime[strong_down] = -2
        return regime

    def _calculate_momentum_divergence(self, df: pd.DataFrame) -> pd.Series:
        price_momentum = df["close"].pct_change(10)
        rsi_momentum = df["rsi"].diff(10) if "rsi" in df.columns else pd.Series(0.0, index=df.index)
        volume_momentum = df["volume"].pct_change(10).replace([np.inf, -np.inf], np.nan)

        def zscore(s: pd.Series) -> pd.Series:
            return (s - s.rolling(50).mean()) / (s.rolling(50).std() + 1e-10)

        price_norm = zscore(price_momentum)
        divergence = ((price_norm - zscore(rsi_momentum)).abs() + (price_norm - zscore(volume_momentum)).abs()) / 2
        return divergence.rolling(5).mean()

    def _calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.Series:
        price_impact = (df["close"] - df["open"]) / df["open"]
        volume_impact = df["volume"] / (df["volume"].rolling(20).mean() + 1e-10)
        return (price_impact * volume_impact).rolling(5).mean()

    def _calculate_liquidity_index(self, df: pd.DataFrame) -> pd.Series:
        spread = (df["high"] - df["low"]) / df["close"]
        volume_norm = df["volume"] / (df["volume"].rolling(50).mean() + 1e-10)
        return np.log1p(volume_norm / (spread + 1e-4)).rolling(10).mean()

    def _calculate_entropy(self, series: pd.Series, window: int) -> pd.Series:
        def entropy(data: np.ndarray) -> float:
            if len(data) < 5:
                return 0.0
            rets = np.diff(data) / (data[:-1] + 1e-10)
            if not np.isfinite(rets).all() or np.ptp(rets) == 0:
                return 0.0
            hist, _ = np.histogram(rets, bins=10)
            probs = hist / (hist.sum() + 1e-10)
            probs = probs[probs > 0]
            return float(-np.sum(probs * np.log2(probs)))

        return series.rolling(window).apply(entropy, raw=True)

    def _detect_market_regime(self, df: pd.DataFrame) -> pd.Series:
        trend_regime = (
            np.sign(df["trend_strength"]) if "trend_strength" in df.columns else pd.Series(0, index=df.index)
        )
        vol_regime = self._calculate_volatility_regime(df) - 1
        momentum_regime = self._classify_momentum_regime(df)
        market_regime = trend_regime * 0.4 + vol_regime * 0.3 + momentum_regime * 0.3
        return market_regime.rolling(5).mean()

    def _candidate_feature_columns(self, feature_df: pd.DataFrame) -> list[str]:
        cols = []
        for col in feature_df.columns:
            if col in FEATURE_EXCLUDE_EXACT or col.startswith(FEATURE_EXCLUDE_PREFIXES):
                continue
            if not pd.api.types.is_numeric_dtype(feature_df[col]) or pd.api.types.is_bool_dtype(
                feature_df[col]
            ):
                continue
            cols.append(col)
        return cols

    # ----------------- TARGET -----------------
    def create_target_variable(
        self,
        df: pd.DataFrame,
        direction: str = "long",
        forward_periods: Optional[int] = None,
        profit_threshold: Optional[float] = None,
        dynamic: bool = True,
        quantile: float = 0.85,
        k_atr: float = 1.2,
        k_vol: float = 1.5,
        min_abs: float = 0.003,
        max_abs: float = 0.05,
    ) -> pd.Series:
        """Ensemble-vote target. Returns a float Series: 1/0, and NaN where the future
        is not yet known (last `forward_periods` rows). Those rows used to be filled
        with 0, i.e. silently labelled as "bad entry"."""
        fp = forward_periods or self.forward_periods
        close = df["close"]
        sign = 1.0 if direction == "long" else -1.0

        forward_returns = close.shift(-fp) / close - 1
        gain = sign * forward_returns
        forward_highs = df["high"].rolling(fp).max().shift(-fp)
        forward_lows = df["low"].rolling(fp).min().shift(-fp)
        if direction == "long":
            potential = (forward_highs - close) / close
            adverse = ((forward_lows - close) / close).abs()
        else:
            potential = (close - forward_lows) / close
            adverse = ((forward_highs - close) / close).abs()
        known = forward_returns.notna() & forward_highs.notna() & forward_lows.notna()

        if profit_threshold is None and dynamic:
            try:
                prev_close = close.shift(1)
                tr = np.maximum(
                    df["high"] - df["low"],
                    np.maximum((df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()),
                )
                atr_pct = (tr.rolling(14).mean() / close).clip(lower=0)
                vola20 = close.pct_change().rolling(20).std()
                base_series = (k_atr * atr_pct + k_vol * vola20).dropna()
                base_scalar = float(np.median(base_series.tail(300))) if len(base_series) else min_abs
                q_thr = float(gain[known].quantile(quantile)) if known.any() else min_abs
                if not np.isfinite(q_thr):
                    q_thr = min_abs
                profit_threshold = float(np.clip(0.5 * base_scalar + 0.5 * q_thr, min_abs, max_abs))
            except Exception as e:
                logger.warning(f"[ML] Dynamic threshold failed ({e}), using 0.015")
                profit_threshold = 0.015
        elif profit_threshold is None:
            profit_threshold = 0.015

        simple_target = (gain > profit_threshold).astype(int)
        profit_target = (potential > profit_threshold).astype(int)
        risk_adjusted_return = gain / (adverse + 1e-10)
        risk_target = ((gain > profit_threshold * 0.7) & (risk_adjusted_return > 0.5)).astype(int)
        returns_std = close.pct_change().rolling(20).std()
        vol_target = (gain > profit_threshold * (1 + returns_std)).astype(int)

        final_target = ((simple_target + profit_target + risk_target + vol_target) >= 2).astype(float)
        final_target[~known] = np.nan

        ratio = final_target.mean()
        logger.info(
            f"[ML] {direction} target (fwd={fp}) thr={profit_threshold:.4f} "
            f"positives={int(final_target.sum())}/{int(known.sum())} ratio={ratio:.3f}"
        )
        return final_target

    # ----------------- TRAINING -----------------
    def _make_models(self, fast: bool, n_train: int) -> dict:
        tscv = TimeSeriesSplit(n_splits=3)
        models: dict = {}
        if fast:
            models["random_forest"] = RandomForestClassifier(
                n_estimators=200, max_depth=10, min_samples_leaf=5, max_features="sqrt",
                class_weight="balanced_subsample", random_state=42, n_jobs=-1,
            )
            models["gradient_boosting"] = GradientBoostingClassifier(
                n_estimators=150, max_depth=3, learning_rate=0.05, subsample=0.8,
                min_samples_leaf=10, validation_fraction=0.1, n_iter_no_change=10, random_state=42,
            )
        else:
            models["random_forest"] = RandomizedSearchCV(
                RandomForestClassifier(random_state=42, n_jobs=-1, class_weight="balanced_subsample"),
                param_distributions={
                    "n_estimators": [150, 200, 300],
                    "max_depth": [6, 10, 15, 20],
                    "min_samples_split": [5, 10, 20],
                    "min_samples_leaf": [2, 5, 10],
                    "max_features": ["sqrt", "log2", 0.5],
                },
                n_iter=12, cv=tscv, scoring="average_precision", n_jobs=-1, random_state=42,
            )
            models["gradient_boosting"] = RandomizedSearchCV(
                GradientBoostingClassifier(
                    random_state=42, validation_fraction=0.1, n_iter_no_change=10, tol=1e-4
                ),
                param_distributions={
                    "n_estimators": [100, 150, 250],
                    "max_depth": [2, 3, 4, 6],
                    "learning_rate": [0.03, 0.05, 0.1],
                    "min_samples_leaf": [5, 10, 20],
                    "subsample": [0.7, 0.85, 1.0],
                },
                n_iter=12, cv=tscv, scoring="average_precision", n_jobs=-1, random_state=42,
            )
            if n_train < 2000:
                models["svm"] = SVC(
                    kernel="rbf", C=1.0, gamma="scale", probability=True,
                    random_state=42, class_weight="balanced",
                )
        models["extra_trees"] = ExtraTreesClassifier(
            n_estimators=150, max_depth=12, min_samples_split=8, min_samples_leaf=4,
            max_features="sqrt", random_state=42, n_jobs=-1, class_weight="balanced",
        )
        models["ada_boost"] = AdaBoostClassifier(n_estimators=80, learning_rate=0.8, random_state=42)
        models["logistic_regression"] = LogisticRegression(
            C=1.0, solver="liblinear", random_state=42, class_weight="balanced", max_iter=1000,
        )
        return models

    def train(
        self,
        feature_df: pd.DataFrame,
        df: pd.DataFrame,
        pair: str,
        direction: str = "long",
        fast: bool = False,
        persist: bool = True,
        key_suffix: str = "",
    ) -> dict:
        """Train the model ensemble for one pair/direction on the given window."""
        if not SKLEARN_AVAILABLE:
            return {"status": "sklearn_not_available"}
        key = self._key(pair, direction) + key_suffix
        fp = self.forward_periods
        try:
            target = self.create_target_variable(df, direction=direction)
            cols = self._candidate_feature_columns(feature_df)
            X = feature_df[cols].replace([np.inf, -np.inf], np.nan)

            # Drop indicator warm-up rows (mostly NaN) and rows with unknown future
            warm = X.isna().mean(axis=1) < 0.3
            valid = target.notna() & warm
            X = X[valid].fillna(0)
            y = target[valid].astype(int)

            if len(X) < self.min_train_rows:
                return {"status": "insufficient_data", "rows": len(X)}

            if y.sum() < 10:
                logger.warning(f"[ML] {key}: only {int(y.sum())} positives, relaxing target")
                relaxed = self.create_target_variable(
                    df, direction=direction, forward_periods=3, profit_threshold=0.01
                )
                y = relaxed.reindex(X.index)
                keep = y.notna()
                X, y = X[keep], y[keep].astype(int)
            if y.sum() < 5 or (len(y) - y.sum()) < 5:
                return {"status": "insufficient_positive_examples"}

            # Chronological split with a purge gap so test labels don't overlap train labels
            split_idx = int(len(X) * 0.7)
            X_train, y_train = X.iloc[: split_idx - fp], y.iloc[: split_idx - fp]
            X_test, y_test = X.iloc[split_idx:], y.iloc[split_idx:]
            if y_train.nunique() < 2:
                return {"status": "single_class_train"}

            # Remove constant features (measured on train only)
            variance = X_train.var()
            feats = variance[variance > 1e-10].index.tolist()
            if len(feats) < 5:
                return {"status": "insufficient_features"}

            # Feature selection fitted on TRAIN only (was fitted on all data -> leakage)
            if len(feats) > self.max_features:
                selector = SelectKBest(score_func=f_classif, k=self.max_features)
                selector.fit(X_train[feats], y_train)
                feats = [feats[i] for i in selector.get_support(indices=True)]

            scaler = RobustScaler()
            X_train_s = scaler.fit_transform(X_train[feats])
            X_test_s = scaler.transform(X_test[feats])

            candidates = self._make_models(fast, len(X_train))
            fitted: dict = {}
            results: dict = {}
            for name, model in candidates.items():
                try:
                    model.fit(X_train_s, y_train)
                    if isinstance(model, RandomizedSearchCV):
                        cv_score = float(model.best_score_)
                        model = model.best_estimator_
                    else:
                        cv_score = float("nan")
                    proba = model.predict_proba(X_test_s)[:, 1]
                    pred = (proba >= 0.5).astype(int)
                    two_classes = y_test.nunique() > 1
                    auc = roc_auc_score(y_test, proba) if two_classes else 0.5
                    results[name] = {
                        "accuracy": accuracy_score(y_test, pred),
                        "precision": precision_score(y_test, pred, zero_division=0),
                        "recall": recall_score(y_test, pred, zero_division=0),
                        "f1_score": f1_score(y_test, pred, zero_division=0),
                        "auc_roc": auc,
                        "auc_pr": average_precision_score(y_test, proba) if two_classes else 0.0,
                        "cv_score": cv_score,
                        "feature_importance": self._get_feature_importance(model, feats),
                    }
                    fitted[name] = model
                    logger.info(
                        f"[ML] {key} {name}: AUC={auc:.3f} AP={results[name]['auc_pr']:.3f} "
                        f"F1={results[name]['f1_score']:.3f} CV(AP)={cv_score:.3f}"
                    )
                except Exception as e:
                    logger.warning(f"[ML] {key} {name} failed: {e}")
            if not fitted:
                return {"status": "all_models_failed"}

            # Ensemble = top 3 models by out-of-sample AUC, weighted by their skill over random
            ranked = sorted(results, key=lambda n: results[n]["auc_roc"], reverse=True)[:3]
            weights = {n: max(results[n]["auc_roc"] - 0.5, 0.01) for n in ranked}
            best_auc = results[ranked[0]]["auc_roc"]
            # Shrink predictions toward 0.5 when even the best model is barely better
            # than a coin flip, so a useless model cannot trigger entries.
            skill = float(np.clip((best_auc - 0.5) / 0.1, 0.0, 1.0))
            if skill < 0.2:
                logger.warning(
                    f"[ML] {key}: best out-of-sample AUC {best_auc:.3f} ~ random; "
                    f"predictions will stay close to 0.5"
                )

            members = {n: fitted[n] for n in ranked}
            if self.refit_on_full_window:
                # Refit chosen models (with tuned params) on train+test so the newest
                # candles are used. Metrics above remain out-of-sample.
                scaler = RobustScaler()
                X_all_s = scaler.fit_transform(X[feats])
                refit = {}
                for n, m in members.items():
                    try:
                        refit[n] = clone(m).fit(X_all_s, y)
                    except Exception as e:
                        logger.warning(f"[ML] {key} refit {n} failed, keeping train fit: {e}")
                        refit[n] = m
                members = refit

            self.models[key] = members
            self.model_weights[key] = weights
            self.scalers[key] = scaler
            self.feature_columns[key] = feats
            self.metrics[key] = results
            self.skill_factor[key] = skill
            self.is_trained[key] = True
            self.last_train_time[key] = _utcnow()
            self._load_attempted.add(key)

            if persist:
                self._save_bundle(pair, direction)

            return {
                "status": "success",
                "best_model": ranked[0],
                "best_auc": best_auc,
                "skill": skill,
                "n_samples": len(X),
                "n_features": len(feats),
                "n_models": len(members),
            }
        except Exception as e:
            logger.warning(f"[ML] Model training failed for {key}: {e}")
            return {"status": "error", "message": str(e)}

    def _get_feature_importance(self, model, feature_columns):
        try:
            if hasattr(model, "feature_importances_"):
                return dict(zip(feature_columns, model.feature_importances_))
            if hasattr(model, "coef_"):
                return dict(zip(feature_columns, np.abs(model.coef_[0])))
        except Exception:
            pass
        return {}

    # ----------------- PREDICTION -----------------
    def predict(
        self, feature_df: pd.DataFrame, pair: str, direction: str = "long", key_suffix: str = ""
    ) -> tuple[pd.Series, pd.Series]:
        """Return (probability, model_disagreement) Series. Neutral 0.5 if no model."""
        key = self._key(pair, direction) + key_suffix
        neutral = pd.Series(0.5, index=feature_df.index)
        no_disagreement = pd.Series(0.0, index=feature_df.index)
        if not key_suffix:
            self._ensure_loaded(pair, direction)
        if not SKLEARN_AVAILABLE or not self.is_trained.get(key, False):
            return neutral, no_disagreement
        try:
            cols = self.feature_columns[key]
            missing = [c for c in cols if c not in feature_df.columns]
            if missing:
                logger.warning(f"[ML] {key}: {len(missing)} trained features missing {missing[:5]}; forcing retrain")
                self.is_trained[key] = False
                return neutral, no_disagreement

            X = feature_df[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
            X_s = self.scalers[key].transform(X)

            preds: dict[str, np.ndarray] = {}
            for name, model in self.models[key].items():
                try:
                    preds[name] = model.predict_proba(X_s)[:, 1]
                except Exception as e:
                    logger.warning(f"[ML] {key} {name} predict failed: {e}")
            if not preds:
                return neutral, no_disagreement

            weights = self.model_weights.get(key, {})
            total = sum(weights.get(n, 0.01) for n in preds)
            final = sum(preds[n] * weights.get(n, 0.01) for n in preds) / total

            # Per-row market condition (the old version used the LAST 20 candles for the
            # whole history, which leaked the future into backtests).
            close = feature_df["close"]
            vol = close.pct_change().rolling(20).std().to_numpy()
            chg = close.pct_change(19).abs().to_numpy()
            volatile = np.nan_to_num(vol) > 0.03
            trending = ~volatile & (np.nan_to_num(chg) > 0.05)
            ranging = ~volatile & ~trending
            if "gradient_boosting" in preds:
                final = np.where(trending, 0.5 * final + 0.5 * preds["gradient_boosting"], final)
            if "random_forest" in preds:
                final = np.where(volatile, 0.5 * final + 0.5 * preds["random_forest"], final)
            for name in ("svm", "logistic_regression"):
                if name in preds:
                    final = np.where(ranging, 0.6 * final + 0.4 * preds[name], final)
                    break

            # Shrink toward neutral when models disagree
            if len(preds) > 1:
                disagreement = np.std(np.vstack(list(preds.values())), axis=0)
                confidence = 1 - np.clip(disagreement * 2, 0, 0.3)
                final = final * confidence + 0.5 * (1 - confidence)
            else:
                disagreement = np.zeros(len(final))

            # Shrink toward neutral when the model has little out-of-sample skill
            skill = self.skill_factor.get(key, 1.0)
            final = 0.5 + (final - 0.5) * skill

            # Causal smoothing (the old rolling(center=True) looked one candle ahead)
            result = pd.Series(final, index=feature_df.index).ewm(span=3, adjust=False).mean()
            return result.clip(0, 1), pd.Series(disagreement, index=feature_df.index)
        except Exception as e:
            logger.warning(f"[ML] Prediction failed for {key}: {e}")
            return neutral, no_disagreement

    def walk_forward_predict(
        self, feature_df: pd.DataFrame, df: pd.DataFrame, pair: str, direction: str = "long"
    ) -> tuple[pd.Series, pd.Series, pd.Series]:
        """Backtest-safe prediction: every block of candles is predicted by a model
        trained only on candles before it. Returns (prob, disagreement, ready_flag)."""
        n = len(df)
        prob = pd.Series(0.5, index=df.index)
        disagreement = pd.Series(0.0, index=df.index)
        ready = pd.Series(0, index=df.index)

        first_date = df["date"].iloc[0] if "date" in df.columns else 0
        last_date = df["date"].iloc[-1] if "date" in df.columns else n
        cache_key = (pair, direction, n, str(first_date), str(last_date))
        if cache_key in self._wf_cache:
            p, d, r = self._wf_cache[cache_key]
            return p.copy(), d.copy(), r.copy()

        window = self.initial_train_candles
        step = max(50, self.backtest_retrain_step)
        start = window if n > window + step // 2 else max(self.min_train_rows + 50, n // 2)
        if start >= n:
            return prob, disagreement, ready

        suffix = "|walkforward"
        for s in range(start, n, step):
            lo = max(0, s - window)
            res = self.train(
                feature_df.iloc[lo:s], df.iloc[lo:s], pair, direction,
                fast=True, persist=False, key_suffix=suffix,
            )
            if res.get("status") != "success":
                continue
            e = min(s + step, n)
            ctx = max(0, s - 10)  # small context so the EWM smoothing is warmed up
            p, d = self.predict(feature_df.iloc[ctx:e], pair, direction, key_suffix=suffix)
            prob.iloc[s:e] = p.iloc[s - ctx :].to_numpy()
            disagreement.iloc[s:e] = d.iloc[s - ctx :].to_numpy()
            ready.iloc[s:e] = 1
        self._wf_cache[cache_key] = (prob.copy(), disagreement.copy(), ready.copy())
        return prob, disagreement, ready


# Global engine instance (models dir is re-pointed to user_data_dir in bot_start)
predictive_engine = AdvancedPredictiveEngine()

# Weight of the ML probability in ml_enhanced_score (rest = ultimate_score).
# The old code used the rolling std of the ML probability as the weight, which
# meant the ML signal contributed ~2-10% and less the more stable it was.
ML_SCORE_WEIGHT = 0.5


def calculate_advanced_predictive_signals(
    dataframe: pd.DataFrame,
    pair: str,
    backtest_mode: bool = False,
    train_short_model: bool = True,
) -> pd.DataFrame:
    """Add ML columns to the dataframe.

    Live / dry-run: train when no model exists or the model is older than
    retrain_interval_hours (checked per pair, so it no longer retrains on every
    candle once the bot has been up for 48h), then predict.

    Backtest / hyperopt: walk-forward training so no candle is ever predicted by a
    model that has seen it (or its future). Models from disk are NOT used, because
    they were trained on data that lies inside the backtest period.
    """
    engine = predictive_engine
    directions = ["long"] + (["short"] if train_short_model else [])
    try:
        feature_df = engine.extract_advanced_features(dataframe)
        outputs: dict[str, tuple[pd.Series, pd.Series, pd.Series]] = {}

        for direction in directions:
            if backtest_mode:
                outputs[direction] = engine.walk_forward_predict(feature_df, dataframe, pair, direction)
                continue

            if engine.needs_training(pair, direction) and len(dataframe) >= engine.min_train_rows + 100:
                window = engine.initial_train_candles
                logger.info(f"[ML] Training {direction} models for {pair} (len={len(dataframe)})")
                res = engine.train(
                    feature_df.iloc[-window:], dataframe.iloc[-window:], pair, direction,
                    fast=engine.live_fast_mode, persist=True,
                )
                if res.get("status") == "success":
                    logger.info(
                        f"🚀 {pair} {direction} ML trained: {res['n_models']} models, "
                        f"best {res['best_model']} AUC={res['best_auc']:.3f} skill={res['skill']:.2f}"
                    )
                else:
                    logger.warning(f"❌ {pair} {direction} ML training status: {res}")
            p, d = engine.predict(feature_df, pair, direction)
            ready = pd.Series(int(engine.is_trained.get(engine._key(pair, direction), False)), index=dataframe.index)
            outputs[direction] = (p, d, ready)

        long_prob, long_dis, long_ready = outputs["long"]
        dataframe["ml_entry_probability"] = long_prob
        dataframe["ml_model_ready"] = long_ready
        dataframe["ml_model_agreement"] = (1 - 2 * long_dis).clip(0, 1)

        if "short" in outputs and outputs["short"][2].any():
            short_prob, _, short_ready = outputs["short"]
            # Fall back to the mirrored long model where no short model was available
            dataframe["ml_short_probability"] = short_prob.where(short_ready == 1, 1 - long_prob)
        else:
            dataframe["ml_short_probability"] = 1 - long_prob

        # Confidence flags (regimes come from the feature frame; the old code looked
        # them up on the strategy dataframe where they never existed)
        momentum_regime = feature_df["momentum_regime"]
        volatility_regime = feature_df["volatility_regime"]
        dataframe["ml_high_confidence"] = (
            (long_prob > 0.8) & (momentum_regime > 0) & (volatility_regime < 2)
        ).astype(int)
        dataframe["ml_ultra_confidence"] = (long_prob > 0.93).astype(int)

        if "ultimate_score" in dataframe.columns:
            u = dataframe["ultimate_score"]
            dataframe["ml_enhanced_score"] = u * (1 - ML_SCORE_WEIGHT) + long_prob * ML_SCORE_WEIGHT
            dataframe["ml_enhanced_short_score"] = (1 - u) * (1 - ML_SCORE_WEIGHT) + dataframe[
                "ml_short_probability"
            ] * ML_SCORE_WEIGHT
        else:
            dataframe["ml_enhanced_score"] = long_prob
            dataframe["ml_enhanced_short_score"] = dataframe["ml_short_probability"]
        return dataframe

    except Exception as e:
        logger.warning(f"[ML] Advanced predictive analysis failed for {pair}: {e}")
        dataframe["ml_entry_probability"] = 0.5
        dataframe["ml_short_probability"] = 0.5
        dataframe["ml_model_ready"] = 0
        dataframe["ml_enhanced_score"] = dataframe.get("ultimate_score", 0.5)
        dataframe["ml_enhanced_short_score"] = 1 - dataframe.get("ultimate_score", 0.5)
        dataframe["ml_high_confidence"] = 0
        dataframe["ml_ultra_confidence"] = 0
        dataframe["ml_model_agreement"] = 0.5
        return dataframe


def calculate_quantum_momentum_analysis(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Quantum-inspired momentum analysis for ultra-precise predictions"""
    try:
        momentum_periods = [3, 5, 8, 13, 21, 34]
        momentum_matrix = pd.DataFrame()

        for period in momentum_periods:
            momentum_matrix[f"mom_{period}"] = dataframe["close"].pct_change(period)

        dataframe["quantum_momentum_coherence"] = momentum_matrix.std(axis=1) / (
            momentum_matrix.mean(axis=1).abs() + 1e-10
        )

        # Calculate momentum entanglement using correlation matrix
        def calculate_entanglement(window_data):
            if len(window_data) < 10:
                return 0
            try:
                corr_matrix = window_data.corr()
                if corr_matrix.empty or corr_matrix.isna().all().all():
                    return 0
                # Get upper triangular correlation values (excluding diagonal)
                upper_tri_indices = np.triu_indices_from(corr_matrix, k=1)
                correlations = corr_matrix.values[upper_tri_indices]
                # Remove NaN values and calculate mean
                valid_correlations = correlations[~np.isnan(correlations)]
                return valid_correlations.mean() if len(valid_correlations) > 0 else 0
            except Exception:
                return 0

        entanglement_values = []
        for i in range(len(momentum_matrix)):
            if i < 20:
                entanglement_values.append(0.5)
            else:
                window_data = momentum_matrix.iloc[i - 19 : i + 1]
                entanglement = calculate_entanglement(window_data)
                entanglement_values.append(entanglement)

        dataframe["momentum_entanglement"] = pd.Series(
            entanglement_values, index=dataframe.index
        )

        price_uncertainty = dataframe["close"].rolling(20).std()
        momentum_uncertainty = momentum_matrix["mom_8"].rolling(20).std()
        dataframe["heisenberg_uncertainty"] = price_uncertainty * momentum_uncertainty

        if "maxima_sort_threshold" in dataframe.columns:
            resistance_distance = (
                dataframe["maxima_sort_threshold"] - dataframe["close"]
            ) / dataframe["close"]
            dataframe["quantum_tunnel_up_prob"] = np.exp(
                -resistance_distance.abs() * 10
            )
        else:
            dataframe["quantum_tunnel_up_prob"] = 0.5

        if "minima_sort_threshold" in dataframe.columns:
            support_distance = (
                dataframe["close"] - dataframe["minima_sort_threshold"]
            ) / dataframe["close"]
            dataframe["quantum_tunnel_down_prob"] = np.exp(-support_distance.abs() * 10)
        else:
            dataframe["quantum_tunnel_down_prob"] = 0.5

        return dataframe

    except Exception as e:
        logger.warning(f"Quantum momentum analysis failed: {e}")
        dataframe["quantum_momentum_coherence"] = 0.5
        dataframe["momentum_entanglement"] = 0.5
        dataframe["heisenberg_uncertainty"] = 1.0
        dataframe["quantum_tunnel_up_prob"] = 0.5
        dataframe["quantum_tunnel_down_prob"] = 0.5
        return dataframe


def calculate_neural_pattern_recognition(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Neural pattern recognition for complex market patterns"""
    try:
        dataframe["body_size"] = (
            abs(dataframe["close"] - dataframe["open"]) / dataframe["close"]
        )
        dataframe["upper_shadow"] = (
            dataframe["high"] - np.maximum(dataframe["open"], dataframe["close"])
        ) / dataframe["close"]
        dataframe["lower_shadow"] = (
            np.minimum(dataframe["open"], dataframe["close"]) - dataframe["low"]
        ) / dataframe["close"]
        dataframe["candle_range"] = (dataframe["high"] - dataframe["low"]) / dataframe[
            "close"
        ]

        # L2 norm of the last 5 candles' shape vector (vectorised version of the
        # former Python loop; identical result, ~100x faster)
        sq = (
            dataframe["body_size"] ** 2
            + dataframe["upper_shadow"] ** 2
            + dataframe["lower_shadow"] ** 2
        )
        pattern_norm = np.sqrt(sq.rolling(5).sum())
        dataframe["neural_pattern_score"] = (
            (pattern_norm / 0.1).clip(upper=1.0).fillna(0)
        )
        dataframe["pattern_prediction_confidence"] = (
            dataframe["neural_pattern_score"].rolling(10).std()
        )

        return dataframe

    except Exception as e:
        logger.warning(f"Neural pattern recognition failed: {e}")
        dataframe["neural_pattern_score"] = 0.5
        dataframe["pattern_prediction_confidence"] = 0.5
        dataframe["body_size"] = 0.01
        dataframe["candle_range"] = 0.02
        return dataframe


def calculate_exit_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate advanced exit signals based on market deterioration"""
    # === MOMENTUM DETERIORATION ===
    dataframe["momentum_deteriorating"] = (
        (dataframe["momentum_quality"] < dataframe["momentum_quality"].shift(1))
        & (dataframe["momentum_acceleration"] < 0)
        & (dataframe["price_momentum"] < dataframe["price_momentum"].shift(1))
    ).astype(int)

    # === VOLUME DETERIORATION ===
    dataframe["volume_deteriorating"] = (
        (dataframe["volume_strength"] < 0.8)
        & (dataframe["selling_pressure"] > dataframe["buying_pressure"])
        & (dataframe["volume_pressure"] < 0)
    ).astype(int)

    # === STRUCTURE DETERIORATION ===
    dataframe["structure_deteriorating"] = (
        (dataframe["structure_score"] < -1)
        & (dataframe["bearish_structure"] > dataframe["bullish_structure"])
        & (dataframe["structure_break_down"] == 1)
    ).astype(int)

    # === CONFLUENCE BREAKDOWN ===
    dataframe["confluence_breakdown"] = (
        (dataframe["confluence_score"] < 2)
        & (dataframe["near_resistance"] == 1)
        & (dataframe["volume_spike"] == 0)
    ).astype(int)

    # === TREND WEAKNESS ===
    dataframe["trend_weakening"] = (
        (dataframe["trend_strength"] < 0)
        & (dataframe["close"] < dataframe["ema50"])
        & (dataframe["strong_downtrend"] == 1)
    ).astype(int)

    # === ULTIMATE EXIT SCORE ===
    dataframe["exit_pressure"] = (
        dataframe["momentum_deteriorating"] * 2
        + dataframe["volume_deteriorating"] * 2
        + dataframe["structure_deteriorating"] * 2
        + dataframe["confluence_breakdown"] * 1
        + dataframe["trend_weakening"] * 1
    )

    # === RSI OVERBOUGHT WITH DIVERGENCE ===
    dataframe["rsi_exit_signal"] = (
        (dataframe["rsi"] > 75)
        & (
            (dataframe["rsi_divergence_bear"] == 1)
            | (dataframe["rsi"] > dataframe["rsi"].shift(1))
            & (dataframe["close"] < dataframe["close"].shift(1))
        )
    ).astype(int)

    # === PROFIT TAKING LEVELS ===
    mml_resistance_levels = ["[6/8]P", "[8/8]P"]
    dataframe["near_resistance_level"] = 0

    for level in mml_resistance_levels:
        if level in dataframe.columns:
            near_level = (
                (dataframe["close"] >= dataframe[level] * 0.99)
                & (dataframe["close"] <= dataframe[level] * 1.02)
            ).astype(int)
            dataframe["near_resistance_level"] += near_level

    # === VOLATILITY SPIKE EXIT ===
    dataframe["volatility_spike"] = (
        dataframe["atr"] > dataframe["atr"].rolling(20).mean() * 1.5
    ).astype(int)

    # === EXHAUSTION SIGNALS ===
    dataframe["bullish_exhaustion"] = (
        (dataframe["consecutive_green"] >= 4)
        & (dataframe["rsi"] > 70)
        & (dataframe["volume"] < dataframe["avg_volume"] * 0.8)
        & (dataframe["momentum_acceleration"] < 0)
    ).astype(int)

    return dataframe


def calculate_dynamic_profit_targets(
    dataframe: pd.DataFrame, entry_type_col: str = "entry_type"
) -> pd.DataFrame:
    """Calculate dynamic profit targets based on entry quality and market conditions"""

    # Base profit targets based on ATR
    dataframe["base_profit_target"] = dataframe["atr"] * 2

    # Adjust based on entry type
    dataframe["profit_multiplier"] = 1.0
    if entry_type_col in dataframe.columns:
        dataframe.loc[dataframe[entry_type_col] == 3, "profit_multiplier"] = (
            2.0  # High quality
        )
        dataframe.loc[dataframe[entry_type_col] == 2, "profit_multiplier"] = (
            1.5  # Medium quality
        )
        dataframe.loc[dataframe[entry_type_col] == 1, "profit_multiplier"] = (
            1.2  # Backup
        )
        dataframe.loc[dataframe[entry_type_col] == 4, "profit_multiplier"] = (
            2.5  # Breakout
        )
        dataframe.loc[dataframe[entry_type_col] == 5, "profit_multiplier"] = (
            1.8  # Reversal
        )

    # Final profit target
    dataframe["dynamic_profit_target"] = (
        dataframe["base_profit_target"] * dataframe["profit_multiplier"]
    )

    return dataframe


def calculate_advanced_stop_loss(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Stop distance in price units, and final_stop_loss as a negative ratio.

    The old version clipped a *price distance* (always positive) to [-0.15, -0.01],
    so final_stop_loss was always -0.01 regardless of volatility.
    """
    dataframe["base_stop_loss"] = dataframe["atr"] * 1.5
    stop_distance = dataframe["base_stop_loss"]
    if "minima_sort_threshold" in dataframe.columns:
        dataframe["support_stop_loss"] = (
            dataframe["close"] - dataframe["minima_sort_threshold"]
        ).clip(dataframe["base_stop_loss"] * 0.5, dataframe["base_stop_loss"] * 1.5)
        stop_distance = np.minimum(
            dataframe["base_stop_loss"], dataframe["support_stop_loss"]
        )
    dataframe["final_stop_loss"] = -(stop_distance / dataframe["close"]).clip(
        0.01, 0.15
    )
    return dataframe


def calculate_confluence_score(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Multi-factor confluence analysis - much better than BTC correlation"""

    # Support/Resistance Confluence
    dataframe["near_support"] = (
        (dataframe["close"] <= dataframe["minima_sort_threshold"] * 1.02)
        & (dataframe["close"] >= dataframe["minima_sort_threshold"] * 0.98)
    ).astype(int)

    dataframe["near_resistance"] = (
        (dataframe["close"] <= dataframe["maxima_sort_threshold"] * 1.02)
        & (dataframe["close"] >= dataframe["maxima_sort_threshold"] * 0.98)
    ).astype(int)

    # MML Level Confluence
    mml_levels = ["[0/8]P", "[2/8]P", "[4/8]P", "[6/8]P", "[8/8]P"]
    dataframe["near_mml"] = 0

    for level in mml_levels:
        if level in dataframe.columns:
            near_level = (
                (dataframe["close"] <= dataframe[level] * 1.015)
                & (dataframe["close"] >= dataframe[level] * 0.985)
            ).astype(int)
            dataframe["near_mml"] += near_level

    # Volume Confluence
    dataframe["volume_spike"] = (
        dataframe["volume"] > dataframe["avg_volume"] * 1.5
    ).astype(int)

    # RSI Confluence Zones
    dataframe["rsi_oversold"] = (dataframe["rsi"] < 30).astype(int)
    dataframe["rsi_overbought"] = (dataframe["rsi"] > 70).astype(int)
    dataframe["rsi_neutral"] = (
        (dataframe["rsi"] >= 40) & (dataframe["rsi"] <= 60)
    ).astype(int)

    # EMA Confluence
    dataframe["above_ema"] = (dataframe["close"] > dataframe["ema50"]).astype(int)

    # CONFLUENCE SCORE (0-6)
    dataframe["confluence_score"] = (
        dataframe["near_support"]
        + dataframe["near_mml"].clip(0, 2)  # Max 2 points for MML
        + dataframe["volume_spike"]
        + dataframe["rsi_oversold"]
        + dataframe["above_ema"]
        + (dataframe["trend_strength"] > 0.01).astype(int)  # Positive trend
    )

    return dataframe


def calculate_smart_volume(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Advanced volume analysis - beats any external correlation"""

    # Volume-Price Trend (VPT)
    price_change_pct = (dataframe["close"] - dataframe["close"].shift(1)) / dataframe[
        "close"
    ].shift(1)
    dataframe["vpt"] = (dataframe["volume"] * price_change_pct).fillna(0).cumsum()

    # Volume moving averages
    dataframe["volume_sma20"] = dataframe["volume"].rolling(20).mean()
    dataframe["volume_sma50"] = dataframe["volume"].rolling(50).mean()

    # Volume strength
    dataframe["volume_strength"] = dataframe["volume"] / dataframe["volume_sma20"]

    # Smart money indicators
    dataframe["accumulation"] = (
        (dataframe["close"] > dataframe["open"])  # Green candle
        & (dataframe["volume"] > dataframe["volume_sma20"] * 1.2)  # High volume
        & (
            dataframe["close"] > (dataframe["high"] + dataframe["low"]) / 2
        )  # Close in upper half
    ).astype(int)

    dataframe["distribution"] = (
        (dataframe["close"] < dataframe["open"])  # Red candle
        & (dataframe["volume"] > dataframe["volume_sma20"] * 1.2)  # High volume
        & (
            dataframe["close"] < (dataframe["high"] + dataframe["low"]) / 2
        )  # Close in lower half
    ).astype(int)

    # Buying/Selling pressure
    dataframe["buying_pressure"] = dataframe["accumulation"].rolling(5).sum()
    dataframe["selling_pressure"] = dataframe["distribution"].rolling(5).sum()

    # Net volume pressure
    dataframe["volume_pressure"] = (
        dataframe["buying_pressure"] - dataframe["selling_pressure"]
    )

    # Volume trend
    dataframe["volume_trend"] = (
        dataframe["volume_sma20"] > dataframe["volume_sma50"]
    ).astype(int)

    # Money flow
    typical_price = (dataframe["high"] + dataframe["low"] + dataframe["close"]) / 3
    money_flow = typical_price * dataframe["volume"]
    positive_flow = money_flow.where(typical_price > typical_price.shift(1), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(1), 0)

    positive_flow_sum = positive_flow.rolling(14).sum()
    negative_flow_sum = negative_flow.rolling(14).sum()

    dataframe["money_flow_ratio"] = positive_flow_sum / (negative_flow_sum + 1e-10)
    dataframe["money_flow_index"] = 100 - (100 / (1 + dataframe["money_flow_ratio"]))

    return dataframe


def calculate_advanced_momentum(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Multi-timeframe momentum system - superior to BTC correlation"""

    # Multi-timeframe momentum
    dataframe["momentum_3"] = dataframe["close"].pct_change(6)
    dataframe["momentum_7"] = dataframe["close"].pct_change(14)
    dataframe["momentum_14"] = dataframe["close"].pct_change(28)
    dataframe["momentum_21"] = dataframe["close"].pct_change(21)

    # Momentum acceleration
    dataframe["momentum_acceleration"] = dataframe["momentum_3"] - dataframe[
        "momentum_3"
    ].shift(3)

    # Momentum consistency
    dataframe["momentum_consistency"] = (
        (dataframe["momentum_3"] > 0).astype(int)
        + (dataframe["momentum_7"] > 0).astype(int)
        + (dataframe["momentum_14"] > 0).astype(int)
    )

    # Momentum divergence with volume
    dataframe["price_momentum_rank"] = (
        dataframe["momentum_7"].rolling(20).rank(pct=True)
    )
    dataframe["volume_momentum_rank"] = (
        dataframe["volume_strength"].rolling(20).rank(pct=True)
    )

    dataframe["momentum_divergence"] = (
        dataframe["price_momentum_rank"] - dataframe["volume_momentum_rank"]
    ).abs()

    # Momentum strength
    dataframe["momentum_strength"] = (
        dataframe["momentum_3"].abs()
        + dataframe["momentum_7"].abs()
        + dataframe["momentum_14"].abs()
    ) / 3

    # Momentum quality score (0-5)
    dataframe["momentum_quality"] = (
        (dataframe["momentum_3"] > 0).astype(int)
        + (dataframe["momentum_7"] > 0).astype(int)
        + (dataframe["momentum_acceleration"] > 0).astype(int)
        + (dataframe["volume_strength"] > 1.1).astype(int)
        + (dataframe["momentum_divergence"] < 0.3).astype(int)
    )

    # Rate of Change
    dataframe["roc_5"] = dataframe["close"].pct_change(5) * 100
    dataframe["roc_10"] = dataframe["close"].pct_change(10) * 100
    dataframe["roc_20"] = dataframe["close"].pct_change(20) * 100

    # Momentum oscillator
    dataframe["momentum_oscillator"] = (
        dataframe["roc_5"] + dataframe["roc_10"] + dataframe["roc_20"]
    ) / 3

    return dataframe


def calculate_market_structure(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Market structure analysis - intrinsic trend recognition"""

    # Higher highs, higher lows detection
    dataframe["higher_high"] = (
        (dataframe["high"] > dataframe["high"].shift(1))
        & (dataframe["high"].shift(1) > dataframe["high"].shift(2))
    ).astype(int)

    dataframe["higher_low"] = (
        (dataframe["low"] > dataframe["low"].shift(1))
        & (dataframe["low"].shift(1) > dataframe["low"].shift(2))
    ).astype(int)

    dataframe["lower_high"] = (
        (dataframe["high"] < dataframe["high"].shift(1))
        & (dataframe["high"].shift(1) < dataframe["high"].shift(2))
    ).astype(int)

    dataframe["lower_low"] = (
        (dataframe["low"] < dataframe["low"].shift(1))
        & (dataframe["low"].shift(1) < dataframe["low"].shift(2))
    ).astype(int)

    # Market structure scores
    dataframe["bullish_structure"] = (
        dataframe["higher_high"].rolling(5).sum()
        + dataframe["higher_low"].rolling(5).sum()
    )

    dataframe["bearish_structure"] = (
        dataframe["lower_high"].rolling(5).sum()
        + dataframe["lower_low"].rolling(5).sum()
    )

    dataframe["structure_score"] = (
        dataframe["bullish_structure"] - dataframe["bearish_structure"]
    )

    # Swing highs and lows
    # Live-safe pivot detection without using future candles (no shift(-1))
    # Confirm swing at the PREVIOUS candle using only information up to current bar.
    # A swing high at t-1 is when high[t-1] > high[t-2] and high[t-1] > high[t].
    prev_high = dataframe["high"].shift(1)
    prev_low = dataframe["low"].shift(1)
    dataframe["swing_high"] = (
        (prev_high > dataframe["high"].shift(2)) & (prev_high > dataframe["high"])
    ).astype(int)

    # A swing low at t-1 is when low[t-1] < low[t-2] and low[t-1] < low[t].
    dataframe["swing_low"] = (
        (prev_low < dataframe["low"].shift(2)) & (prev_low < dataframe["low"])
    ).astype(int)

    # Market structure breaks
    # Use previous candle values where the swing was confirmed to avoid lookahead bias
    swing_highs = prev_high.where(dataframe["swing_high"] == 1)
    swing_lows = prev_low.where(dataframe["swing_low"] == 1)

    # Structure break detection
    dataframe["structure_break_up"] = (dataframe["close"] > swing_highs.ffill()).astype(
        int
    )

    dataframe["structure_break_down"] = (
        dataframe["close"] < swing_lows.ffill()
    ).astype(int)

    # Trend strength based on structure
    dataframe["structure_trend_strength"] = (
        dataframe["structure_score"] / 10  # Normalize
    ).clip(-1, 1)

    # Support and resistance strength
    dataframe["support_strength"] = dataframe["swing_low"].rolling(20).sum()
    dataframe["resistance_strength"] = dataframe["swing_high"].rolling(20).sum()

    return dataframe


def calculate_advanced_entry_signals(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Advanced entry signal generation"""

    # Multi-factor signal strength
    dataframe["signal_strength"] = 0

    # Confluence signals
    dataframe["confluence_signal"] = (dataframe["confluence_score"] >= 3).astype(int)
    dataframe["signal_strength"] += dataframe["confluence_signal"] * 2

    # Volume signals
    dataframe["volume_signal"] = (
        (dataframe["volume_pressure"] >= 2) & (dataframe["volume_strength"] > 1.2)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["volume_signal"] * 2

    # Momentum signals
    dataframe["momentum_signal"] = (
        (dataframe["momentum_quality"] >= 3) & (dataframe["momentum_acceleration"] > 0)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["momentum_signal"] * 2

    # Structure signals
    dataframe["structure_signal"] = (
        (dataframe["structure_score"] > 0) & (dataframe["structure_break_up"] == 1)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["structure_signal"] * 1

    # RSI position signal
    dataframe["rsi_signal"] = (
        (dataframe["rsi"] > 30) & (dataframe["rsi"] < 70)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["rsi_signal"] * 1

    # Trend alignment signal
    dataframe["trend_signal"] = (
        (dataframe["close"] > dataframe["ema50"]) & (dataframe["trend_strength"] > 0)
    ).astype(int)
    dataframe["signal_strength"] += dataframe["trend_signal"] * 1

    # Money flow signal
    dataframe["money_flow_signal"] = (dataframe["money_flow_index"] > 50).astype(int)
    dataframe["signal_strength"] += dataframe["money_flow_signal"] * 1

    return dataframe


class AlexNexusForgeV8AIV2(IStrategy):
    # General strategy parameters
    timeframe = "1h"
    startup_candle_count: int = 1000
    stoploss = -0.11
    trailing_stop = True
    trailing_stop_positive = 0.005  # Trail at 0.5% below peak profit
    trailing_stop_positive_offset = 0.03  # Start trailing only at 3% profit
    trailing_only_offset_is_reached = (
        True  # Ensure trailing only starts after offset is reached
    )
    use_custom_stoploss = True
    can_short = True
    use_exit_signal = True
    ignore_roi_if_entry_signal = (
        False  # CHANGED: Allow ROI to work even with entry signals
    )
    # True: indicators + ML only recompute when a new candle closes. False re-ran the
    # whole (very heavy) pipeline on every bot loop, i.e. every few seconds per pair.
    process_only_new_candles = True
    use_custom_exits_advanced = True
    use_emergency_exits = True

    # Position adjustment (DCA). max_entry_position_adjustment is the hard cap
    # (= top of the max_safety_orders range); the parameter decides the actual count.
    position_adjustment_enable = True
    max_entry_position_adjustment = 3

    # =====================================================================
    # HYPEROPT PARAMETERS
    # ---------------------------------------------------------------------
    # Freqtrade's hyperopt runs populate_indicators ONCE per pair and only
    # re-runs populate_entry_trend / populate_exit_trend and the callbacks
    # (custom_stoploss, custom_exit, leverage, adjust_trade_position, ...) for
    # each epoch. So a parameter can only be optimised if it is read outside
    # populate_indicators, or if every candidate value is precomputed there.
    #   optimize=True  -> genuinely changes results between epochs
    #   optimize=False -> feeds populate_indicators / the ML features; tune it
    #                     with separate backtests instead (load=True keeps it
    #                     loadable from a params JSON file).
    # ~50 parameters that were defined but never read anywhere were removed;
    # hyperopt was searching those dimensions for nothing.
    # =====================================================================

    # --- Regime protection (evaluated per epoch in populate_entry_trend + custom_exit)
    regime_change_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    regime_change_sensitivity = DecimalParameter(
        0.3, 0.8, default=0.5, decimals=2, space="sell", optimize=True, load=True
    )
    flash_move_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    flash_move_threshold = DecimalParameter(
        0.03, 0.08, default=0.05, decimals=3, space="sell", optimize=True, load=True
    )
    flash_move_candles = IntParameter(
        3, 10, default=5, space="sell", optimize=True, load=True
    )
    volume_spike_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    volume_spike_multiplier = DecimalParameter(
        2.0, 5.0, default=3.0, decimals=1, space="sell", optimize=True, load=True
    )
    sentiment_protection_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    sentiment_shift_threshold = DecimalParameter(
        0.2, 0.4, default=0.3, decimals=2, space="sell", optimize=True, load=True
    )
    emergency_exit_enabled = BooleanParameter(
        default=True, space="sell", optimize=True, load=True
    )
    emergency_exit_profit_threshold = DecimalParameter(
        0.005, 0.03, default=0.015, decimals=3, space="sell", optimize=True, load=True
    )

    # --- Trailing-stop exit control. Safety switch, deliberately not optimised:
    # letting hyperopt "discover" that blocking stop exits looks good in a
    # backtest is exactly how accounts get liquidated.
    allow_trailing_exit_when_negative = BooleanParameter(
        default=True, space="sell", optimize=False, load=True
    )
    trailing_exit_min_profit = DecimalParameter(
        -0.03, 0.02, default=-0.002, decimals=3, space="sell", optimize=False, load=True
    )

    # --- ATR stoploss (custom_stoploss). The widest stop is capped by the static
    # `stoploss` (-0.11), so atr_stoploss_maximum (-0.15..-0.30) only matters if
    # you loosen `stoploss`; it is therefore not optimised.
    atr_stoploss_multiplier = DecimalParameter(
        0.8, 2.0, default=1.0, decimals=1, space="sell", optimize=True, load=True
    )
    atr_stoploss_maximum = DecimalParameter(
        -0.30, -0.15, default=-0.18, decimals=2, space="sell", optimize=False, load=True
    )
    atr_stoploss_ceiling = DecimalParameter(
        -0.10, -0.06, default=-0.06, decimals=2, space="sell", optimize=True, load=True
    )

    # --- DCA / safety orders (adjust_trade_position + custom_stake_amount)
    dca_enabled = BooleanParameter(default=True, space="buy", optimize=True, load=True)
    # Loss from the FIRST entry (leveraged profit ratio, like current_profit) that
    # triggers safety order 1. Later orders are spaced by safety_order_step_scale.
    initial_safety_order_trigger = DecimalParameter(
        low=-0.02, high=-0.01, default=-0.018, decimals=3,
        space="buy", optimize=True, load=True,
    )
    max_safety_orders = IntParameter(1, 3, default=1, space="buy", optimize=True, load=True)
    safety_order_step_scale = DecimalParameter(
        low=1.05, high=1.5, default=1.25, decimals=2, space="buy", optimize=True, load=True,
    )
    safety_order_volume_scale = DecimalParameter(
        low=1.1, high=2.0, default=1.4, decimals=1, space="buy", optimize=True, load=True,
    )
    # Skip a safety order when the ML model for the trade's side has turned against it
    dca_ml_guard = DecimalParameter(
        0.35, 0.55, default=0.45, decimals=2, space="buy", optimize=True, load=True
    )

    # --- Dynamic leverage (leverage callback, evaluated per trade)
    leverage_window_size = IntParameter(
        20, 100, default=70, space="buy", optimize=True, load=True
    )
    leverage_base = DecimalParameter(
        5.0, 20.0, default=5.0, decimals=1, space="buy", optimize=True, load=True
    )
    leverage_rsi_low = DecimalParameter(
        20.0, 40.0, default=30.0, decimals=1, space="buy", optimize=True, load=True
    )
    leverage_rsi_high = DecimalParameter(
        60.0, 80.0, default=70.0, decimals=1, space="buy", optimize=True, load=True
    )
    leverage_long_increase_factor = DecimalParameter(
        1.1, 2.0, default=1.5, decimals=1, space="buy", optimize=True, load=True
    )
    leverage_long_decrease_factor = DecimalParameter(
        0.3, 0.9, default=0.5, decimals=1, space="buy", optimize=True, load=True
    )
    leverage_volatility_decrease_factor = DecimalParameter(
        0.5, 0.95, default=0.8, decimals=2, space="buy", optimize=True, load=True
    )
    leverage_atr_threshold_pct = DecimalParameter(
        0.01, 0.05, default=0.03, decimals=3, space="buy", optimize=True, load=True
    )

    # --- Extrema order: every candidate is precomputed in populate_indicators and
    # the active one is selected in populate_entry_trend, so this IS optimisable.
    indicator_extrema_order = IntParameter(
        3, 15, default=8, space="buy", optimize=True, load=True
    )

    # --- Indicator-stage parameters (feed populate_indicators and the ML model)
    h2 = IntParameter(20, 60, default=40, space="buy", optimize=False, load=True)
    h1 = IntParameter(10, 40, default=20, space="buy", optimize=False, load=True)
    h0 = IntParameter(5, 20, default=10, space="buy", optimize=False, load=True)
    cp = IntParameter(5, 20, default=10, space="buy", optimize=False, load=True)
    mml_const1 = DecimalParameter(
        1.0, 1.1, default=1.0699, decimals=4, space="buy", optimize=False, load=True
    )
    mml_const2 = DecimalParameter(
        0.99, 1.0, default=0.99875, decimals=5, space="buy", optimize=False, load=True
    )
    indicator_mml_window = IntParameter(
        50, 200, default=50, space="buy", optimize=False, load=True
    )
    indicator_rolling_window_threshold = IntParameter(
        20, 100, default=50, space="buy", optimize=False, load=True
    )
    indicator_rolling_check_window = IntParameter(
        5, 20, default=10, space="buy", optimize=False, load=True
    )
    confluence_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    volume_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    momentum_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    structure_analysis_enabled = BooleanParameter(default=True, space="buy", optimize=False, load=True)
    # Only drives the `high_quality_setup` plot column, which no entry uses
    ultimate_score_threshold = DecimalParameter(
        0.5, 3.0, default=1.5, space="buy", optimize=False, load=True
    )

    # --- Protections (only optimised with `--enable-protections`)
    cooldown_lookback = IntParameter(1, 48, default=1, space="protection", optimize=True)
    stop_duration = IntParameter(4, 200, default=4, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)

    # ✅ Replace your old ROI with this:
    minimal_roi = {
        "0": 0.06,
        "5": 0.055,
        "10": 0.04,
        "20": 0.03,
        "40": 0.025,
        "80": 0.02,
        "160": 0.015,
        "320": 0.01,
    }

    # Plot configuration for backtesting UI
    plot_config = {
        "main_plot": {
            # Trend indicators
            "ema50": {"color": "gray", "type": "line"},
            # Support/Resistance
            "minima_sort_threshold": {"color": "#4ae747", "type": "line"},
            "maxima_sort_threshold": {"color": "#5b5e4b", "type": "line"},
        },
        "subplots": {
            "extrema_analysis": {
                "s_extrema": {"color": "#f53580", "type": "line"},
                "maxima": {"color": "#a29db9", "type": "scatter"},
                "minima": {"color": "#aac7fc", "type": "scatter"},
            },
            "murrey_math_levels": {
                "[4/8]P": {"color": "blue", "type": "line"},  # 50% MML
                "[6/8]P": {"color": "green", "type": "line"},  # 75% MML
                "[2/8]P": {"color": "orange", "type": "line"},  # 25% MML
                "[8/8]P": {"color": "red", "type": "line"},  # 100% MML
                "[0/8]P": {"color": "red", "type": "line"},  # 0% MML
                "mmlextreme_oscillator": {"color": "purple", "type": "line"},
            },
            "rsi_analysis": {
                "rsi": {"color": "purple", "type": "line"},
                "rsi_divergence_bull": {"color": "green", "type": "scatter"},
                "rsi_divergence_bear": {"color": "red", "type": "scatter"},
            },
            "confluence_analysis": {
                "confluence_score": {"color": "gold", "type": "line"},
                "near_support": {"color": "green", "type": "scatter"},
                "near_resistance": {"color": "red", "type": "scatter"},
                "near_mml": {"color": "blue", "type": "line"},
                "volume_spike": {"color": "orange", "type": "scatter"},
            },
            "volume_analysis": {
                "volume_strength": {"color": "cyan", "type": "line"},
                "volume_pressure": {"color": "magenta", "type": "line"},
                "buying_pressure": {"color": "green", "type": "line"},
                "selling_pressure": {"color": "red", "type": "line"},
                "money_flow_index": {"color": "yellow", "type": "line"},
            },
            "momentum_analysis": {
                "momentum_quality": {"color": "brown", "type": "line"},
                "momentum_acceleration": {"color": "pink", "type": "line"},
                "momentum_consistency": {"color": "lime", "type": "line"},
                "momentum_oscillator": {"color": "navy", "type": "line"},
            },
            "structure_analysis": {
                "structure_score": {"color": "teal", "type": "line"},
                "bullish_structure": {"color": "green", "type": "line"},
                "bearish_structure": {"color": "red", "type": "line"},
                "structure_break_up": {"color": "lime", "type": "scatter"},
                "structure_break_down": {"color": "crimson", "type": "scatter"},
            },
            "trend_strength": {
                "trend_strength": {"color": "indigo", "type": "line"},
                "trend_strength_5": {"color": "lightblue", "type": "line"},
                "trend_strength_10": {"color": "mediumblue", "type": "line"},
                "trend_strength_20": {"color": "darkblue", "type": "line"},
            },
            "ultimate_signals": {
                "ultimate_score": {"color": "gold", "type": "line"},
                "signal_strength": {"color": "silver", "type": "line"},
                "high_quality_setup": {"color": "lime", "type": "scatter"},
                "entry_type": {"color": "white", "type": "line"},
            },
            "market_conditions": {
                "strong_uptrend": {"color": "green", "type": "scatter"},
                "strong_downtrend": {"color": "red", "type": "scatter"},
                "ranging": {"color": "yellow", "type": "scatter"},
                "strong_up_momentum": {"color": "lime", "type": "scatter"},
                "strong_down_momentum": {"color": "crimson", "type": "scatter"},
            },
            "di_analysis": {
                "DI_values": {"color": "orange", "type": "line"},
                "DI_catch": {"color": "red", "type": "scatter"},
                "plus_di": {"color": "green", "type": "line"},
                "minus_di": {"color": "red", "type": "line"},
            },
        },
    }

    def bot_start(self, **kwargs) -> None:
        """Store ML models under the configured user_data_dir instead of a path
        relative to whatever directory the bot happened to be started from."""
        user_data_dir = self.config.get("user_data_dir") if self.config else None
        if user_data_dir:
            predictive_engine.set_models_dir(
                Path(user_data_dir) / "strategies" / "ml_models"
            )
        logger.info(f"[ML] Model directory: {predictive_engine.models_dir}")

    def _is_backtest_like(self) -> bool:
        runmode = getattr(self.dp, "runmode", None) if self.dp else None
        return runmode in (RunMode.BACKTEST, RunMode.HYPEROPT, RunMode.PLOT)

    # Helper method to check if we have an active position in the opposite direction
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
    def _calculate_mml_core(
        mn: float, finalH: float, mx: float, finalL: float, mml_c1: float, mml_c2: float
    ) -> Dict[str, float]:
        dmml_calc = ((finalH - finalL) / 8.0) * mml_c1
        if (
            dmml_calc == 0
            or np.isinf(dmml_calc)
            or np.isnan(dmml_calc)
            or finalH == finalL
        ):
            return {key: finalL for key in MML_LEVEL_NAMES}
        mml_val = (mx * mml_c2) + (dmml_calc * 3)
        if np.isinf(mml_val) or np.isnan(mml_val):
            return {key: finalL for key in MML_LEVEL_NAMES}
        ml = [mml_val - (dmml_calc * i) for i in range(16)]
        return {
            "[-3/8]P": ml[14],
            "[-2/8]P": ml[13],
            "[-1/8]P": ml[12],
            "[0/8]P": ml[11],
            "[1/8]P": ml[10],
            "[2/8]P": ml[9],
            "[3/8]P": ml[8],
            "[4/8]P": ml[7],
            "[5/8]P": ml[6],
            "[6/8]P": ml[5],
            "[7/8]P": ml[4],
            "[8/8]P": ml[3],
            "[+1/8]P": ml[2],
            "[+2/8]P": ml[1],
            "[+3/8]P": ml[0],
        }

    def calculate_rolling_murrey_math_levels_optimized(
        self, df: pd.DataFrame, window_size: int
    ) -> Dict[str, pd.Series]:
        """
        OPTIMIZED Version - Calculate MML levels every 5 candles using only past data
        """
        murrey_levels_data: Dict[str, list] = {
            key: [np.nan] * len(df) for key in MML_LEVEL_NAMES
        }
        mml_c1 = self.mml_const1.value
        mml_c2 = self.mml_const2.value

        calculation_step = 5

        for i in range(0, len(df), calculation_step):
            if i < window_size:
                continue

            # Levels written at row `window_end` use candles up to and including it.
            # (iloc's end is exclusive, so the old slice silently dropped that candle.)
            window_end = i - 1
            window_start = max(0, window_end - window_size + 1)

            window_data = df.iloc[window_start : window_end + 1]
            mn_period = window_data["low"].min()
            mx_period = window_data["high"].max()
            current_close = df["close"].iloc[window_end]

            if pd.isna(mn_period) or pd.isna(mx_period) or mn_period == mx_period:
                for key in MML_LEVEL_NAMES:
                    murrey_levels_data[key][window_end] = current_close
                continue

            levels = self._calculate_mml_core(
                mn_period, mx_period, mx_period, mn_period, mml_c1, mml_c2
            )

            for key in MML_LEVEL_NAMES:
                murrey_levels_data[key][window_end] = levels.get(key, current_close)

        # Hold each computed level until the next calculation (forward-fill only).
        # The old code used expanding().mean(), i.e. the AVERAGE of every level since
        # the start of the data, so levels lagged further behind price over time and
        # differed between backtest and live (different history length).
        for key in MML_LEVEL_NAMES:
            series = pd.Series(murrey_levels_data[key], index=df.index).ffill()
            murrey_levels_data[key] = series.tolist()

        return {
            key: pd.Series(data, index=df.index)
            for key, data in murrey_levels_data.items()
        }

    def calculate_synthetic_market_breadth(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Calculate synthetic market breadth using technical indicators
        Simulates market sentiment based on multiple factors
        """
        try:
            # RSI component (30% weight)
            rsi_component = (dataframe["rsi"] - 50) / 50  # Normalize to -1 to 1

            # Volume component (25% weight)
            volume_ma = dataframe["volume"].rolling(20).mean()
            volume_component = (dataframe["volume"] / volume_ma - 1).clip(-1, 1)

            # Momentum component (25% weight)
            momentum_3 = dataframe["close"].pct_change(3)
            momentum_component = np.tanh(momentum_3 * 100)  # Smooth normalization

            # Volatility component (20% weight) - inverted (lower vol = higher breadth)
            atr_normalized = dataframe["atr"] / dataframe["close"]
            atr_ma = atr_normalized.rolling(20).mean()
            volatility_component = -(atr_normalized / atr_ma - 1).clip(-1, 1)

            # Combine components with weights
            synthetic_breadth = (
                rsi_component * 0.30
                + volume_component * 0.25
                + momentum_component * 0.25
                + volatility_component * 0.20
            )

            # Normalize to 0-1 range (market breadth percentage)
            synthetic_breadth = (synthetic_breadth + 1) / 2

            # Smooth with rolling average
            synthetic_breadth = synthetic_breadth.rolling(3).mean()

            return synthetic_breadth.fillna(0.5)

        except Exception as e:
            logger.warning(f"Synthetic market breadth calculation failed: {e}")
            return pd.Series(0.5, index=dataframe.index)

    def calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength to avoid entering against strong trends
        """

        # Linear regression slope
        def calc_slope(series, period=10):
            """Calculate linear regression slope"""
            if len(series) < period:
                return 0
            x = np.arange(period)
            y = series.iloc[-period:].values
            if np.isnan(y).any() or np.isinf(y).any():
                return 0
            slope = np.polyfit(x, y, 1)[0]
            return slope

        # Calculate trend strength using multiple timeframes
        df["slope_5"] = (
            df["close"].rolling(5).apply(lambda x: calc_slope(x, 5), raw=False)
        )
        df["slope_10"] = (
            df["close"].rolling(10).apply(lambda x: calc_slope(x, 10), raw=False)
        )
        df["slope_20"] = (
            df["close"].rolling(20).apply(lambda x: calc_slope(x, 20), raw=False)
        )

        df["trend_strength_5"] = df["slope_5"] / df["close"] * 100
        df["trend_strength_10"] = df["slope_10"] / df["close"] * 100
        df["trend_strength_20"] = df["slope_20"] / df["close"] * 100

        # Combined trend strength
        df["trend_strength"] = (
            df["trend_strength_5"] + df["trend_strength_10"] + df["trend_strength_20"]
        ) / 3

        # Trend classification
        strong_threshold = 0.02
        df["strong_uptrend"] = df["trend_strength"] > strong_threshold
        df["strong_downtrend"] = df["trend_strength"] < -strong_threshold
        df["ranging"] = df["trend_strength"].abs() < (strong_threshold * 0.5)

        return df

    @property
    def protections(self):
        prot = [
            {
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_lookback.value,
            }
        ]
        if self.use_stop_protection.value:
            prot.append(
                {
                    "method": "StoplossGuard",
                    "lookback_period_candles": 72,
                    "trade_limit": 2,
                    "stop_duration_candles": self.stop_duration.value,
                    "only_per_pair": False,
                }
            )
        return prot

    def informative_pairs(self):
        """
        Define additional pairs for correlation analysis
        """
        # (BTC used to be listed twice)
        return [
            ("BTC/USDT:USDT", self.timeframe),
            ("ETH/USDT:USDT", self.timeframe),
            ("BNB/USDT:USDT", self.timeframe),
        ]

    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool = False,
        **kwargs,
    ) -> Optional[float]:
        """ATR stop + tiered profit lock.

        Fixes vs. the previous version:
        * the profit lock used min() of two negative numbers, i.e. it picked the
          LOOSER stop, and was then overridden by the -6% "ceiling" anyway, so the
          stop could never trail tighter than 6% below price;
        * the ATR distance ignored leverage (stoploss in freqtrade is risk on the
          leveraged position: 1% price move at 5x = 5%);
        * profit locks are now converted with stoploss_from_open, because the
          return value of custom_stoploss is relative to current_rate, not entry.
        """
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty or "atr" not in dataframe.columns:
            return None  # keep the current stop

        atr = dataframe["atr"].iat[-1]
        if pd.isna(atr) or atr <= 0 or current_rate <= 0:
            return None

        leverage = trade.leverage or 1.0
        atr_percent = atr / current_rate

        if current_profit > 0.15:
            multiplier = 1.0
        elif current_profit > 0.08:
            multiplier = 1.2
        elif current_profit > 0.03:
            multiplier = 1.4
        else:
            multiplier = 1.6

        atr_distance = atr_percent * multiplier * self.atr_stoploss_multiplier.value * leverage
        # ceiling = closest allowed stop, maximum = widest allowed stop.
        # Never wider than the static stoploss: after a safety order fills,
        # freqtrade calls this with after_fill=True and lets the stop WIDEN, which
        # would otherwise jump to atr_stoploss_maximum on the now larger position.
        min_distance = abs(self.atr_stoploss_ceiling.value)
        max_distance = min(abs(self.atr_stoploss_maximum.value), abs(self.stoploss))
        distance = float(np.clip(atr_distance, min_distance, max_distance))

        # Tiered profit lock (profit to keep if price reverses)
        lock_profit = 0.0
        if current_profit > self.trailing_stop_positive_offset:
            lock_profit = current_profit - self.trailing_stop_positive
        elif current_profit > 0.02:
            lock_profit = current_profit - 0.01
        elif current_profit > 0.015:
            lock_profit = current_profit - 0.015

        if lock_profit > 0:
            lock_distance = stoploss_from_open(
                lock_profit, current_profit, is_short=trade.is_short, leverage=leverage
            )
            if lock_distance > 0:
                distance = min(distance, lock_distance)

        logger.debug(
            f"{pair} Custom SL: -{distance:.4f} | ATR%: {atr_percent:.4f} | "
            f"Lev: {leverage:.1f} | Profit: {current_profit:.4f} | Lock: {lock_profit:.4f}"
        )
        return -distance

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        side: str,
        **kwargs,
    ) -> float:
        window_size = self.leverage_window_size.value
        dataframe, _ = self.dp.get_analyzed_dataframe(
            pair=pair, timeframe=self.timeframe
        )
        if len(dataframe) < window_size:
            logger.warning(
                f"{pair} Not enough data ({len(dataframe)} candles) to calculate dynamic leverage (requires {window_size}). Using proposed: {proposed_leverage}"
            )
            return proposed_leverage
        # The old code passed single Series to talib.abstract, which returns plain
        # arrays (MACD returns a tuple), so the `isinstance(macd_output, DataFrame)`
        # check was always False and the MACD rule never applied. Passing the OHLCV
        # frame returns pandas objects consistently.
        tail = dataframe.tail(window_size)
        base_leverage = self.leverage_base.value

        def _last(values, default: float) -> float:
            arr = np.asarray(values, dtype=float)
            arr = arr[~np.isnan(arr)]
            return float(arr[-1]) if arr.size else default

        current_rsi = _last(ta.RSI(tail, timeperiod=14), 50.0)
        current_atr = _last(ta.ATR(tail, timeperiod=14), 0.0)
        current_sma = _last(ta.SMA(tail, timeperiod=20), current_rate)
        macd_output = ta.MACD(tail, fastperiod=12, slowperiod=26, signalperiod=9)
        current_macd_hist = (
            _last(macd_output["macdhist"], 0.0)
            if isinstance(macd_output, pd.DataFrame) and "macdhist" in macd_output
            else 0.0
        )

        high_volatility = (
            current_atr > 0
            and current_rate > 0
            and (current_atr / current_rate) > self.leverage_atr_threshold_pct.value
        )

        # Apply rules based on indicators (shorts mirror the long rules; previously
        # shorts always got the flat base leverage)
        if side == "short":
            if current_rsi > self.leverage_rsi_high.value:
                base_leverage *= self.leverage_long_increase_factor.value
            elif current_rsi < self.leverage_rsi_low.value:
                base_leverage *= self.leverage_long_decrease_factor.value
            if high_volatility:
                base_leverage *= self.leverage_volatility_decrease_factor.value
            if current_macd_hist < 0:
                base_leverage *= self.leverage_long_increase_factor.value
            if current_sma > 0 and current_rate > current_sma:
                base_leverage *= self.leverage_long_decrease_factor.value

        if side == "long":
            if current_rsi < self.leverage_rsi_low.value:
                base_leverage *= self.leverage_long_increase_factor.value
            elif current_rsi > self.leverage_rsi_high.value:
                base_leverage *= self.leverage_long_decrease_factor.value

            if current_atr > 0 and current_rate > 0:
                if (current_atr / current_rate) > self.leverage_atr_threshold_pct.value:
                    base_leverage *= self.leverage_volatility_decrease_factor.value

            if current_macd_hist > 0:
                base_leverage *= self.leverage_long_increase_factor.value

            if current_sma > 0 and current_rate < current_sma:
                base_leverage *= self.leverage_long_decrease_factor.value

        adjusted_leverage = round(max(1.0, min(base_leverage, max_leverage)), 2)
        logger.info(
            f"{pair} Dynamic Leverage: {adjusted_leverage:.2f} (Base: {base_leverage:.2f}, RSI: {current_rsi:.2f}, "
            f"ATR: {current_atr:.4f}, MACD Hist: {current_macd_hist:.4f}, SMA: {current_sma:.4f})"
        )
        return adjusted_leverage

    def populate_indicators(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        ULTIMATE indicator calculations with advanced market analysis
        """
        # === EXTERNAL DATA INTEGRATION ===
        try:
            # Add BTC close aligned by candle date. (The old code took the last N BTC
            # rows by position, which misaligns whenever the two series don't end on
            # the same candle or have gaps.)
            if metadata["pair"] != "BTC/USDT:USDT":
                btc_info = self.dp.get_pair_dataframe("BTC/USDT:USDT", self.timeframe)
                if btc_info is not None and not btc_info.empty:
                    btc_map = (
                        btc_info[["date", "close"]]
                        .drop_duplicates("date")
                        .rename(columns={"close": "btc_close"})
                    )
                    dataframe = dataframe.drop(
                        columns=["btc_close"], errors="ignore"
                    ).merge(btc_map, on="date", how="left")
                    dataframe["btc_close"] = (
                        dataframe["btc_close"].ffill().fillna(dataframe["close"])
                    )
                    logger.debug(
                        f"{metadata['pair']} BTC correlation data added successfully"
                    )
                else:
                    # Fallback: use current pair data
                    dataframe["btc_close"] = dataframe["close"]
                    logger.warning(
                        f"{metadata['pair']} BTC data unavailable, using pair data as fallback"
                    )
            else:
                # For BTC pairs, use own data
                dataframe["btc_close"] = dataframe["close"]
        except Exception as e:
            logger.warning(f"{metadata['pair']} BTC data integration failed: {e}")
            dataframe["btc_close"] = dataframe["close"]  # Safe fallback

        # === STANDARD INDICATORS ===
        dataframe["ema50"] = ta.EMA(dataframe["close"], timeperiod=50)
        dataframe["ema100"] = ta.EMA(
            dataframe["close"], timeperiod=100
        )  # Neu hinzufÃ¼gen
        dataframe["rsi"] = ta.RSI(dataframe["close"])
        dataframe["atr"] = ta.ATR(
            dataframe["high"], dataframe["low"], dataframe["close"], timeperiod=10
        )

        # === SYNTHETIC MARKET BREADTH CALCULATION ===
        try:
            # Calculate synthetic market breadth using multiple indicators
            # (after RSI and ATR are available)
            dataframe["market_breadth"] = self.calculate_synthetic_market_breadth(
                dataframe
            )
            logger.info(f"{metadata['pair']} Synthetic market breadth calculated")
        except Exception as e:
            logger.warning(f"{metadata['pair']} Market breadth calculation failed: {e}")
            dataframe["market_breadth"] = 0.5  # Neutral fallback
        dataframe["plus_di"] = ta.PLUS_DI(dataframe)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe)
        dataframe["DI_values"] = dataframe["plus_di"] - dataframe["minus_di"]
        dataframe["DI_cutoff"] = 0

        # === EXTREMA DETECTION ===
        # maxima = close makes a new N-candle high (live-safe local top),
        # minima = new N-candle low. The old test was `close == previous max`, an
        # exact float equality that is almost never true, so every exit requiring
        # maxima/minima == 1 was effectively dead.
        # Every candidate order is precomputed so hyperopt can select one per epoch
        # (`.range` is the full range in hyperopt, only the current value otherwise).
        close = dataframe["close"]
        extrema_cols = {}
        for order in self.indicator_extrema_order.range:
            extrema_cols[f"_ext_max_{order}"] = (
                close > close.shift(1).rolling(window=order).max()
            ).astype(int)
            extrema_cols[f"_ext_min_{order}"] = (
                close < close.shift(1).rolling(window=order).min()
            ).astype(int)
        dataframe = pd.concat(
            [dataframe.drop(columns=list(extrema_cols), errors="ignore"),
             pd.DataFrame(extrema_cols, index=dataframe.index)],
            axis=1,
        )
        self._select_extrema(dataframe)

        # === HEIKIN-ASHI ===
        dataframe["ha_close"] = (
            dataframe["open"]
            + dataframe["high"]
            + dataframe["low"]
            + dataframe["close"]
        ) / 4

        # === ROLLING EXTREMA ===
        dataframe["minh2"], dataframe["maxh2"] = calculate_minima_maxima(
            dataframe, self.h2.value
        )
        dataframe["minh1"], dataframe["maxh1"] = calculate_minima_maxima(
            dataframe, self.h1.value
        )
        dataframe["minh0"], dataframe["maxh0"] = calculate_minima_maxima(
            dataframe, self.h0.value
        )
        dataframe["mincp"], dataframe["maxcp"] = calculate_minima_maxima(
            dataframe, self.cp.value
        )

        # === MURREY MATH LEVELS ===
        mml_window = self.indicator_mml_window.value
        murrey_levels = self.calculate_rolling_murrey_math_levels_optimized(
            dataframe, window_size=mml_window
        )

        for level_name in MML_LEVEL_NAMES:
            if level_name in murrey_levels:
                dataframe[level_name] = murrey_levels[level_name]
            else:
                dataframe[level_name] = dataframe["close"]

        # === MML OSCILLATOR ===
        mml_4_8 = dataframe.get("[4/8]P")
        mml_plus_3_8 = dataframe.get("[+3/8]P")
        mml_minus_3_8 = dataframe.get("[-3/8]P")

        if (
            mml_4_8 is not None
            and mml_plus_3_8 is not None
            and mml_minus_3_8 is not None
        ):
            osc_denominator = (mml_plus_3_8 - mml_minus_3_8).replace(0, np.nan)
            dataframe["mmlextreme_oscillator"] = 100 * (
                (dataframe["close"] - mml_4_8) / osc_denominator
            )
        else:
            dataframe["mmlextreme_oscillator"] = np.nan

        # === DI CATCH ===
        dataframe["DI_catch"] = np.where(
            dataframe["DI_values"] > dataframe["DI_cutoff"], 0, 1
        )

        # === ROLLING THRESHOLDS ===
        rolling_window_threshold = self.indicator_rolling_window_threshold.value
        dataframe["minima_sort_threshold"] = (
            dataframe["close"]
            .rolling(window=rolling_window_threshold, min_periods=1)
            .min()
        )
        dataframe["maxima_sort_threshold"] = (
            dataframe["close"]
            .rolling(window=rolling_window_threshold, min_periods=1)
            .max()
        )

        # === EXTREMA CHECKS ===
        rolling_check_window = self.indicator_rolling_check_window.value
        dataframe["minima_check"] = (
            dataframe["minima"]
            .rolling(window=rolling_check_window, min_periods=1)
            .sum()
            == 0
        ).astype(int)
        dataframe["maxima_check"] = (
            dataframe["maxima"]
            .rolling(window=rolling_check_window, min_periods=1)
            .sum()
            == 0
        ).astype(int)

        # === VOLATILITY INDICATORS ===
        dataframe["volatility_range"] = dataframe["high"] - dataframe["low"]
        dataframe["avg_volatility"] = (
            dataframe["volatility_range"].rolling(window=50).mean()
        )
        dataframe["avg_volume"] = dataframe["volume"].rolling(window=50).mean()

        # === TREND STRENGTH INDICATORS ===
        # Use enhanced Wavelet+FFT method with fallback
        try:
            # Advanced wavelet & FFT method
            dataframe = calculate_advanced_trend_strength_with_wavelets(dataframe)

            # Use advanced trend strength as primary
            dataframe["trend_strength"] = dataframe["trend_strength_cycle_adjusted"]
            dataframe["strong_uptrend"] = dataframe["strong_uptrend_advanced"]
            dataframe["strong_downtrend"] = dataframe["strong_downtrend_advanced"]
            dataframe["ranging"] = dataframe["ranging_advanced"]

            logger.info(f"{metadata['pair']} Using advanced Wavelet+FFT trend analysis")

        except Exception as e:
            # Fallback to original enhanced method if advanced fails
            logger.warning(
                f"{metadata['pair']} Wavelet/FFT analysis failed: {e}. "
                "Using enhanced method."
            )

            def calc_slope(series, period):
                """Enhanced slope calculation as fallback"""
                if len(series) < period:
                    return 0
                y = series.values[-period:]
                if np.isnan(y).any() or np.isinf(y).any():
                    return 0
                if np.all(y == y[0]):
                    return 0
                x = np.linspace(0, period - 1, period)
                try:
                    coefficients = np.polyfit(x, y, 1)
                    slope = coefficients[0]
                    if np.isnan(slope) or np.isinf(slope):
                        return 0
                    max_reasonable_slope = np.std(y) / period
                    if abs(slope) > max_reasonable_slope * 10:
                        return np.sign(slope) * max_reasonable_slope * 10
                    return slope
                except Exception:
                    try:
                        simple_slope = (y[-1] - y[0]) / (period - 1)
                        return (
                            simple_slope
                            if not (np.isnan(simple_slope) or np.isinf(simple_slope))
                            else 0
                        )
                    except Exception:
                        return 0

            # Original slope calculations
            dataframe["slope_5"] = (
                dataframe["close"]
                .rolling(5)
                .apply(lambda x: calc_slope(x, 5), raw=False)
            )
            dataframe["slope_10"] = (
                dataframe["close"]
                .rolling(10)
                .apply(lambda x: calc_slope(x, 10), raw=False)
            )
            dataframe["slope_20"] = (
                dataframe["close"]
                .rolling(20)
                .apply(lambda x: calc_slope(x, 20), raw=False)
            )

            dataframe["trend_strength_5"] = (
                dataframe["slope_5"] / dataframe["close"] * 100
            )
            dataframe["trend_strength_10"] = (
                dataframe["slope_10"] / dataframe["close"] * 100
            )
            dataframe["trend_strength_20"] = (
                dataframe["slope_20"] / dataframe["close"] * 100
            )

            dataframe["trend_strength"] = (
                dataframe["trend_strength_5"]
                + dataframe["trend_strength_10"]
                + dataframe["trend_strength_20"]
            ) / 3

            strong_threshold = 0.02
            dataframe["strong_uptrend"] = dataframe["trend_strength"] > strong_threshold
            dataframe["strong_downtrend"] = (
                dataframe["trend_strength"] < -strong_threshold
            )
            dataframe["ranging"] = dataframe["trend_strength"].abs() < (
                strong_threshold * 0.5
            )

        # === MOMENTUM INDICATORS ===
        dataframe["price_momentum"] = dataframe["close"].pct_change(3)
        dataframe["momentum_increasing"] = dataframe["price_momentum"] > dataframe[
            "price_momentum"
        ].shift(1)
        dataframe["momentum_decreasing"] = dataframe["price_momentum"] < dataframe[
            "price_momentum"
        ].shift(1)

        dataframe["volume_momentum"] = (
            dataframe["volume"].rolling(3).mean()
            / dataframe["volume"].rolling(20).mean()
        )

        dataframe["rsi_divergence_bull"] = (
            dataframe["close"] < dataframe["close"].shift(5)
        ) & (dataframe["rsi"] > dataframe["rsi"].shift(5))
        dataframe["rsi_divergence_bear"] = (
            dataframe["close"] > dataframe["close"].shift(5)
        ) & (dataframe["rsi"] < dataframe["rsi"].shift(5))

        # === CANDLE PATTERNS ===
        dataframe["green_candle"] = dataframe["close"] > dataframe["open"]
        dataframe["red_candle"] = dataframe["close"] < dataframe["open"]
        dataframe["consecutive_green"] = dataframe["green_candle"].rolling(3).sum()
        dataframe["consecutive_red"] = dataframe["red_candle"].rolling(3).sum()

        # Define strong_threshold for momentum calculations
        strong_threshold = 0.02

        dataframe["strong_up_momentum"] = (
            (dataframe["consecutive_green"] >= 3)
            & (dataframe["volume"] > dataframe["avg_volume"])
            & (dataframe["trend_strength"] > strong_threshold)
        )
        dataframe["strong_down_momentum"] = (
            (dataframe["consecutive_red"] >= 3)
            & (dataframe["volume"] > dataframe["avg_volume"])
            & (dataframe["trend_strength"] < -strong_threshold)
        )

        # === ADVANCED ANALYSIS MODULES ===

        # 1. CONFLUENCE ANALYSIS
        if self.confluence_enabled.value:
            dataframe = calculate_confluence_score(dataframe)
        else:
            dataframe["confluence_score"] = 0

        # 2. SMART VOLUME ANALYSIS
        if self.volume_analysis_enabled.value:
            dataframe = calculate_smart_volume(dataframe)
        else:
            dataframe["volume_pressure"] = 0
            dataframe["volume_strength"] = 1.0
            dataframe["money_flow_index"] = 50

        # 3. ADVANCED MOMENTUM
        if self.momentum_analysis_enabled.value:
            dataframe = calculate_advanced_momentum(dataframe)
        else:
            dataframe["momentum_quality"] = 0
            dataframe["momentum_acceleration"] = 0

        # 4. MARKET STRUCTURE
        if self.structure_analysis_enabled.value:
            dataframe = calculate_market_structure(dataframe)
        else:
            dataframe["structure_score"] = 0
            dataframe["structure_break_up"] = 0

        # 5. ADVANCED ENTRY SIGNALS
        dataframe = calculate_advanced_entry_signals(dataframe)

        # === ULTIMATE MARKET SCORE ===
        dataframe["ultimate_score"] = (
            dataframe["confluence_score"] * 0.25  # 25% confluence
            + dataframe["volume_pressure"] * 0.2  # 20% volume pressure
            + dataframe["momentum_quality"] * 0.2  # 20% momentum quality
            + (dataframe["structure_score"] / 5) * 0.15  # 15% structure (normalized)
            + (dataframe["signal_strength"] / 10) * 0.2  # 20% signal strength
        )

        # Normalize ultimate score to 0-1 range
        dataframe["ultimate_score"] = dataframe["ultimate_score"].clip(0, 5) / 5

        # === FINAL QUALITY CHECKS ===
        # ultimate_score was normalised to 0..1 above (clip(0, 5) / 5) but compared to
        # a threshold in 0.5..3.0, so this could never be true. Scale the threshold.
        dataframe["high_quality_setup"] = (
            (dataframe["ultimate_score"] > self.ultimate_score_threshold.value / 5)
            & (dataframe["signal_strength"] >= 5)
            & (dataframe["volume_strength"] > 1.1)
            & (dataframe["rsi"] > 30)
            & (dataframe["rsi"] < 70)
        ).astype(int)

        # === DEBUG INFO ===
        if metadata["pair"] in [
            "BTC/USDT:USDT",
            "ETH/USDT:USDT",
        ]:  # Only log for major pairs
            latest_score = dataframe["ultimate_score"].iloc[-1]
            latest_signal = dataframe["signal_strength"].iloc[-1]
            logger.info(
                f"{metadata['pair']} Ultimate Score: {latest_score:.3f}, Signal Strength: {latest_signal}"
            )

        # === ADVANCED PREDICTIVE ANALYSIS ===
        try:
            pair = metadata.get("pair", "UNKNOWN")
            dataframe = calculate_advanced_predictive_signals(
                dataframe,
                pair,
                backtest_mode=self._is_backtest_like(),
                train_short_model=self.can_short,
            )
            dataframe = calculate_quantum_momentum_analysis(dataframe)
            dataframe = calculate_neural_pattern_recognition(dataframe)

            logger.info(f"{pair} Advanced predictive analysis completed")
        except Exception as e:
            logger.warning(f"Advanced predictive analysis failed: {e}")
            dataframe["ml_entry_probability"] = 0.5
            dataframe["ml_short_probability"] = 0.5
            dataframe["ml_enhanced_score"] = dataframe.get("ultimate_score", 0.5)
            dataframe["ml_enhanced_short_score"] = 1 - dataframe.get("ultimate_score", 0.5)
            dataframe["ml_high_confidence"] = 0
            dataframe["ml_ultra_confidence"] = 0
            dataframe["quantum_momentum_coherence"] = 0.5
            dataframe["momentum_entanglement"] = 0.5
            dataframe["quantum_tunnel_up_prob"] = 0.5
            dataframe["neural_pattern_score"] = 0.5

        return dataframe

    def _select_extrema(self, dataframe: pd.DataFrame) -> None:
        """Point maxima/minima/s_extrema at the precomputed columns for the
        current indicator_extrema_order value (works per hyperopt epoch)."""
        order = self.indicator_extrema_order.value
        if f"_ext_max_{order}" not in dataframe.columns:
            return
        dataframe["maxima"] = dataframe[f"_ext_max_{order}"]
        dataframe["minima"] = dataframe[f"_ext_min_{order}"]
        dataframe["s_extrema"] = 0
        dataframe.loc[dataframe["minima"] == 1, "s_extrema"] = -1
        dataframe.loc[dataframe["maxima"] == 1, "s_extrema"] = 1

    def _populate_regime_signals(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Regime-change columns used by custom_exit and adjust_trade_position.

        Moved out of populate_indicators: these are cheap and depend only on
        hyperopt parameters, and populate_indicators runs only once per hyperopt,
        so the flash/volume/sentiment/sensitivity parameters never changed
        between epochs. They are no longer ML inputs either, so tuning them can't
        silently change what the model learned.
        """
        # ===========================================
        # REGIME CHANGE DETECTION
        # ===========================================

        if self.regime_change_enabled.value:

            # ===========================================
            # FLASH MOVE DETECTION
            # ===========================================

            flash_candles = self.flash_move_candles.value
            flash_threshold = self.flash_move_threshold.value

            # Schnelle Preisbewegungen
            dataframe["price_change_fast"] = dataframe["close"].pct_change(
                flash_candles
            )
            dataframe["flash_pump"] = dataframe["price_change_fast"] > flash_threshold
            dataframe["flash_dump"] = dataframe["price_change_fast"] < -flash_threshold
            dataframe["flash_move"] = dataframe["flash_pump"] | dataframe["flash_dump"]

            # ===========================================
            # VOLUME SPIKE DETECTION
            # ===========================================

            volume_ma20 = dataframe["volume"].rolling(20).mean()
            volume_multiplier = self.volume_spike_multiplier.value
            # Separate column: this used to overwrite the 1.5x "volume_spike" from the
            # confluence module with a 3x boolean version.
            dataframe["regime_volume_spike"] = dataframe["volume"] > (
                volume_ma20 * volume_multiplier
            )

            # Volume + Bewegung kombiniert
            dataframe["volume_pump"] = (
                dataframe["regime_volume_spike"] & dataframe["flash_pump"]
            )
            dataframe["volume_dump"] = (
                dataframe["regime_volume_spike"] & dataframe["flash_dump"]
            )

            # ===========================================
            # MARKET SENTIMENT DETECTION
            # ===========================================

            # Market Breadth Change
            if "market_breadth" in dataframe.columns:
                dataframe["market_breadth_change"] = dataframe["market_breadth"].diff(3)
                sentiment_threshold = self.sentiment_shift_threshold.value
                dataframe["sentiment_shift_bull"] = (
                    dataframe["market_breadth_change"] > sentiment_threshold
                )
                dataframe["sentiment_shift_bear"] = (
                    dataframe["market_breadth_change"] < -sentiment_threshold
                )
            else:
                dataframe["sentiment_shift_bull"] = False
                dataframe["sentiment_shift_bear"] = False

            # ===========================================
            # BTC CORRELATION MONITORING
            # ===========================================

            # BTC Flash Moves
            if "btc_close" in dataframe.columns:
                dataframe["btc_change_fast"] = dataframe["btc_close"].pct_change(
                    flash_candles
                )
                dataframe["btc_flash_pump"] = (
                    dataframe["btc_change_fast"] > flash_threshold
                )
                dataframe["btc_flash_dump"] = (
                    dataframe["btc_change_fast"] < -flash_threshold
                )

                # Correlation Break
                pair_movement = dataframe["price_change_fast"].abs()
                btc_movement = dataframe["btc_change_fast"].abs()
                dataframe["correlation_break"] = (btc_movement > flash_threshold) & (
                    pair_movement < flash_threshold * 0.4
                )
            else:
                dataframe["btc_flash_pump"] = False
                dataframe["btc_flash_dump"] = False
                dataframe["correlation_break"] = False

            # ===========================================
            # REGIME CHANGE SCORE
            # ===========================================

            regime_signals = [
                "flash_move",
                "regime_volume_spike",
                "sentiment_shift_bull",
                "sentiment_shift_bear",
                "btc_flash_pump",
                "btc_flash_dump",
                "correlation_break",
            ]

            dataframe["regime_change_score"] = 0
            for signal in regime_signals:
                if signal in dataframe.columns:
                    dataframe["regime_change_score"] += dataframe[signal].astype(int)

            # Normalisiere auf 0-1
            max_signals = len(regime_signals)
            dataframe["regime_change_intensity"] = (
                dataframe["regime_change_score"] / max_signals
            )

            # Alert Level
            sensitivity = self.regime_change_sensitivity.value
            dataframe["regime_alert"] = (
                dataframe["regime_change_intensity"] >= sensitivity
            )

        else:
            # Falls Regime Detection deaktiviert
            dataframe["flash_pump"] = False
            dataframe["flash_dump"] = False
            dataframe["sentiment_shift_bull"] = False
            dataframe["sentiment_shift_bear"] = False
            dataframe["btc_flash_pump"] = False
            dataframe["btc_flash_dump"] = False
            dataframe["volume_pump"] = False
            dataframe["volume_dump"] = False
            dataframe["regime_alert"] = False
            dataframe["regime_change_intensity"] = 0.0

        return dataframe

    def populate_entry_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        AI-CENTRIC ENTRY LOGIC - Simplified and functional
        """

        # Hyperopt-dependent signals (re-evaluated every epoch)
        self._select_extrema(dataframe)
        dataframe = self._populate_regime_signals(dataframe)

        # ===========================================
        # INITIALIZE ENTRY COLUMNS
        # ===========================================
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0
        dataframe["enter_tag"] = ""
        dataframe["entry_type"] = 0

        # ===========================================
        # CORE AI SIGNALS (Primary Decision Makers)
        # ===========================================

        # Ensure AI probability exists and is valid
        ml_prob = dataframe.get(
            "ml_entry_probability", pd.Series(0.5, index=dataframe.index)
        )
        ml_enhanced = dataframe.get(
            "ml_enhanced_score", pd.Series(0.5, index=dataframe.index)
        )
        # Shorts use the dedicated short model (P(profitable down-move)). The old
        # logic shorted whenever the LONG model was unenthusiastic (ml_prob < 0.48),
        # which is not the same thing as expecting a drop. Without a short model this
        # falls back to exactly the old behaviour (1 - long probability).
        ml_short = dataframe.get("ml_short_probability", 1 - ml_prob)
        ml_enhanced_short = dataframe.get("ml_enhanced_short_score", 1 - ml_enhanced)

        # ===========================================
        # BASIC SAFETY FILTERS (Minimal Requirements)
        # ===========================================

        basic_safety = (
            (dataframe["rsi"] > 15)
            & (dataframe["rsi"] < 85)  # Not extreme RSI
            & (dataframe["volume"] > dataframe["avg_volume"] * 0.3)  # Some volume
            & (dataframe["atr"] < dataframe["close"] * 0.12)  # Reasonable volatility
        )

        # ===========================================
        # AI-DRIVEN LONG ENTRIES (Tiered System)
        # ===========================================

        # TIER 1: Ultra AI Confidence (Best threshold that actually works)
        ai_ultra_long = (
            (ml_prob > 0.7)  # Reduced from 0.9 to 0.7
            & (ml_enhanced > 0.65)  # Reduced from 0.8 to 0.65
            & basic_safety
        )

        # TIER 2: High AI Confidence
        ai_high_long = (
            (ml_prob > 0.6)  # Reduced from 0.75 to 0.6
            & (ml_enhanced > 0.55)  # Reduced from 0.7 to 0.55
            & basic_safety
            & ~ai_ultra_long  # Only if not ultra
        )

        # TIER 3: Standard AI Confidence
        ai_standard_long = (
            (ml_prob > 0.52)  # Reduced from 0.65 to 0.52
            & basic_safety
            & ~(ai_ultra_long | ai_high_long)  # Only if not higher tier
        )

        # TIER 4: Opportunistic AI (Market conditions favorable)
        ai_opportunistic_long = (
            (ml_prob > 0.48)  # Close to neutral but favorable
            & (dataframe["rsi"] < 35)  # Oversold opportunity
            & (dataframe["volume"] > dataframe["avg_volume"] * 1.5)  # High volume
            & basic_safety
            & ~(ai_ultra_long | ai_high_long | ai_standard_long)
        )

        # ===========================================
        # AI-DRIVEN SHORT ENTRIES (Mirror Logic)
        # ===========================================

        # TIER 1: Ultra AI Short Confidence
        ai_ultra_short = (
            (ml_short > 0.7)
            & (ml_enhanced_short > 0.65)
            & basic_safety
            & (dataframe["close"] < dataframe["ema50"])  # Below EMA for shorts
        )

        # TIER 2: High AI Short Confidence
        ai_high_short = (
            (ml_short > 0.6)
            & (ml_enhanced_short > 0.55)
            & basic_safety
            & (dataframe["close"] < dataframe["ema50"])
            & ~ai_ultra_short
        )

        # TIER 3: Standard AI Short
        ai_standard_short = (
            (ml_short > 0.52)
            & basic_safety
            & (dataframe["close"] < dataframe["ema50"])
            & ~(ai_ultra_short | ai_high_short)
        )

        # TIER 4: Opportunistic Short
        ai_opportunistic_short = (
            (ml_short > 0.48)
            & (dataframe["rsi"] > 65)  # Overbought opportunity
            & (dataframe["volume"] > dataframe["avg_volume"] * 1.5)
            & basic_safety
            & (dataframe["close"] < dataframe["ema50"])
            & ~(ai_ultra_short | ai_high_short | ai_standard_short)
        )

        # ===========================================
        # ENHANCED AI ENTRIES (Optional Boost)
        # ===========================================

        # Check for enhanced AI indicators safely
        quantum_coherence = dataframe.get(
            "quantum_momentum_coherence", pd.Series(0.5, index=dataframe.index)
        )
        neural_pattern = dataframe.get(
            "neural_pattern_score", pd.Series(0.5, index=dataframe.index)
        )
        ml_agreement = dataframe.get(
            "ml_model_agreement", pd.Series(0.7, index=dataframe.index)
        )

        # Enhanced long entries (when advanced AI agrees)
        ai_enhanced_long = (
            (ml_prob > 0.6)
            & (quantum_coherence > 0.6)  # Reduced threshold
            & (neural_pattern > 0.6)  # Reduced threshold
            & (ml_agreement > 0.6)  # Reduced threshold
            & basic_safety
        )

        # Enhanced short entries
        ai_enhanced_short = (
            (ml_short > 0.6)
            & (quantum_coherence < 0.4)  # Inverted for shorts
            & (neural_pattern < 0.4)  # Inverted for shorts
            & (ml_agreement > 0.6)  # Models agree on direction
            & basic_safety
            & (dataframe["close"] < dataframe["ema50"])
        )

        # ===========================================
        # FALLBACK TECHNICAL ENTRIES (If AI fails)
        # ===========================================

        # Simple technical long (backup when AI is neutral)
        technical_long = (
            (ml_prob.between(0.45, 0.55))  # AI neutral
            & (dataframe["rsi"] < 30)  # Oversold
            & (dataframe["close"] > dataframe["ema50"])  # Above EMA
            & (dataframe["volume"] > dataframe["avg_volume"] * 2)  # High volume
            & basic_safety
        )

        # Simple technical short (backup when AI is neutral)
        technical_short = (
            (ml_prob.between(0.45, 0.55))  # AI neutral
            & (dataframe["rsi"] > 70)  # Overbought
            & (dataframe["close"] < dataframe["ema50"])  # Below EMA
            & (dataframe["volume"] > dataframe["avg_volume"] * 2)  # High volume
            & basic_safety
        )

        # ===========================================
        # APPLY ENTRY SIGNALS (Hierarchical Priority)
        # ===========================================

        # LONG ENTRIES (Highest priority first)

        # Enhanced AI Long (Priority 1)
        dataframe.loc[ai_enhanced_long, "enter_long"] = 1
        dataframe.loc[ai_enhanced_long, "entry_type"] = 15
        dataframe.loc[ai_enhanced_long, "enter_tag"] = "ai_enhanced_long"

        # Ultra AI Long (Priority 2)
        mask = ai_ultra_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 14
        dataframe.loc[mask, "enter_tag"] = "ai_ultra_long"

        # High AI Long (Priority 3)
        mask = ai_high_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 13
        dataframe.loc[mask, "enter_tag"] = "ai_high_long"

        # Standard AI Long (Priority 4)
        mask = ai_standard_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 12
        dataframe.loc[mask, "enter_tag"] = "ai_standard_long"

        # Opportunistic AI Long (Priority 5)
        mask = ai_opportunistic_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 11
        dataframe.loc[mask, "enter_tag"] = "ai_opportunistic_long"

        # Technical Long (Priority 6 - Fallback)
        mask = technical_long & (dataframe["enter_long"] == 0)
        dataframe.loc[mask, "enter_long"] = 1
        dataframe.loc[mask, "entry_type"] = 10
        dataframe.loc[mask, "enter_tag"] = "technical_long"

        # SHORT ENTRIES (If shorting enabled)

        # Shorts are collected separately so a conflict can be resolved without the
        # short's entry_type/tag overwriting the long's. (Before, entry_type was
        # overwritten by the short, so the comparison always kept the LONG while the
        # tag still said "..._short".)
        short_signal = pd.Series(0, index=dataframe.index)
        short_type = pd.Series(0, index=dataframe.index)
        short_tag = pd.Series("", index=dataframe.index, dtype=object)
        if self.can_short:
            for cond, etype, tag in (
                (ai_enhanced_short, 25, "ai_enhanced_short"),
                (ai_ultra_short, 24, "ai_ultra_short"),
                (ai_high_short, 23, "ai_high_short"),
                (ai_standard_short, 22, "ai_standard_short"),
                (ai_opportunistic_short, 21, "ai_opportunistic_short"),
                (technical_short, 20, "technical_short"),
            ):
                mask = cond.fillna(False) & (short_signal == 0)
                short_signal[mask] = 1
                short_type[mask] = etype
                short_tag[mask] = tag

        # Resolve long/short conflicts by tier (x5 = enhanced ... x0 = technical);
        # ties go to the long.
        long_rank = (dataframe["entry_type"] % 10).where(dataframe["enter_long"] == 1, -1)
        short_rank = (short_type % 10).where(short_signal == 1, -1)
        conflict_mask = (dataframe["enter_long"] == 1) & (short_signal == 1)
        if conflict_mask.any():
            logger.debug(
                f"{metadata['pair']} Resolving {conflict_mask.sum()} conflicting signals"
            )
        take_short = (short_signal == 1) & (short_rank > long_rank)
        dataframe.loc[take_short, "enter_long"] = 0
        dataframe.loc[take_short, "enter_short"] = 1
        dataframe.loc[take_short, "entry_type"] = short_type[take_short]
        dataframe.loc[take_short, "enter_tag"] = short_tag[take_short]

        # ===========================================
        # ENTRY DEBUGGING & MONITORING
        # ===========================================

        if metadata["pair"] in ["BTC/USDT:USDT", "ETH/USDT:USDT"]:
            # Count total entries in last 10 candles
            recent_long_entries = dataframe["enter_long"].tail(10).sum()
            recent_short_entries = dataframe["enter_short"].tail(10).sum()

            if recent_long_entries > 0 or recent_short_entries > 0:
                latest_ml_prob = ml_prob.iloc[-1]
                latest_enhanced = ml_enhanced.iloc[-1]
                latest_entry_type = dataframe["entry_type"].iloc[-1]
                latest_tag = dataframe["enter_tag"].iloc[-1]

                entry_types = {
                    10: "Technical Long",
                    11: "AI Opportunistic Long",
                    12: "AI Standard Long",
                    13: "AI High Long",
                    14: "AI Ultra Long",
                    15: "AI Enhanced Long",
                    20: "Technical Short",
                    21: "AI Opportunistic Short",
                    22: "AI Standard Short",
                    23: "AI High Short",
                    24: "AI Ultra Short",
                    25: "AI Enhanced Short",
                }

                logger.info(f"🎯 {metadata['pair']} ENTRY SIGNAL DETECTED!")
                logger.info(
                    f"   Type: {entry_types.get(latest_entry_type, 'Unknown')} ({latest_tag})"
                )
                logger.info(f"   🤖 ML Probability: {latest_ml_prob:.3f}")
                logger.info(f"   📈 ML Enhanced Score: {latest_enhanced:.3f}")
                logger.info(f"   📊 RSI: {dataframe['rsi'].iloc[-1]:.1f}")
                logger.info(
                    f"   💧 Volume Strength: {dataframe.get('volume_strength', pd.Series([1.0])).iloc[-1]:.2f}"
                )
                logger.info(
                    f"   Recent Entries: {recent_long_entries} Long, {recent_short_entries} Short"
                )

                # Alert if no AI indicators available
                if latest_ml_prob == 0.5 and latest_enhanced == 0.5:
                    logger.warning(
                        f"⚠️  {metadata['pair']} Using fallback - AI indicators may not be working!"
                    )

        return dataframe

    def populate_exit_trend(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        UNIFIED EXIT SYSTEM - Choose between Custom MML Exits or Simple Opposite Signal Exits
        """
        # ===========================================
        # INITIALIZE EXIT COLUMNS
        # ===========================================
        dataframe["exit_long"] = 0
        dataframe["exit_short"] = 0
        dataframe["exit_tag"] = ""

        # ===========================================
        # CHOOSE EXIT SYSTEM
        # ===========================================
        if self.use_custom_exits_advanced:
            # Use Alex's Advanced MML-based Exit System
            return self._populate_custom_exits_advanced(dataframe, metadata)
        else:
            # Use Simple Opposite Signal Exit System
            return self._populate_simple_exits(dataframe, metadata)

    def _populate_custom_exits_advanced(
        self, df: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        ALEX'S ADVANCED MML-BASED EXIT SYSTEM
        Profit-protecting exit strategy with better signal coordination
        """

        # ===========================================
        # MML MARKET STRUCTURE FOR EXITS
        # ===========================================

        # Bullish/Bearish structure (same as entry)
        bullish_mml = (df["close"] > df["[6/8]P"]) | (
            (df["close"] > df["[4/8]P"])
            & (df["close"].shift(5) < df["[4/8]P"].shift(5))
        )

        bearish_mml = (df["close"] < df["[2/8]P"]) | (
            (df["close"] < df["[4/8]P"])
            & (df["close"].shift(5) > df["[4/8]P"].shift(5))
        )

        # MML resistance/support levels for exits
        at_resistance = (
            (df["high"] >= df["[6/8]P"])  # At 75%
            | (df["high"] >= df["[7/8]P"])  # At 87.5%
            | (df["high"] >= df["[8/8]P"])  # At 100%
        )

        at_support = (
            (df["low"] <= df["[2/8]P"])  # At 25%
            | (df["low"] <= df["[1/8]P"])  # At 12.5%
            | (df["low"] <= df["[0/8]P"])  # At 0%
        )

        # ===========================================
        # AI-BASED INTELLIGENT EXIT SIGNALS
        # ===========================================

        # Long and short AI signals are now kept SEPARATE. Previously
        # ai_degradation_signal = long_exit | short_exit was applied to BOTH sides:
        # "ml_prob > 0.55" (bad for shorts) therefore closed every healthy long, and
        # "ml_prob < 0.45" closed every healthy short.
        false_series = pd.Series(False, index=df.index)
        current_profit_signal = false_series.copy()
        long_stability_signal = false_series.copy()
        short_stability_signal = false_series.copy()
        long_degradation_signal = false_series.copy()
        short_degradation_signal = false_series.copy()

        ml_prob = df.get("ml_entry_probability", pd.Series(0.5, index=df.index))
        ml_enhanced = df.get("ml_enhanced_score", pd.Series(0.5, index=df.index))
        ml_short = df.get("ml_short_probability", 1 - ml_prob)
        ml_enhanced_short = df.get("ml_enhanced_short_score", 1 - ml_enhanced)

        try:
            # 1. AI STABILITY ANALYSIS

            # Calculate AI stability metrics
            ml_prob_sma_5 = ml_prob.rolling(5).mean().fillna(ml_prob)
            ml_prob_sma_10 = ml_prob.rolling(10).mean().fillna(ml_prob)
            ml_prob_std_5 = ml_prob.rolling(5).std().fillna(0.1)

            # For LONG positions: AI should maintain high probability (>0.6) for continuation
            ai_long_stable = (
                (ml_prob > 0.6)  # Current prediction supports long
                & (ml_prob_sma_5 > 0.6)  # Recent average supports long
                & (ml_prob_std_5 < 0.15)  # Low volatility in predictions (stable)
                & (ml_enhanced > 0.65)  # Enhanced score supports direction
                & (ml_prob > ml_prob.shift(1))  # Predictions improving or stable
            )

            # For SHORT positions: the short model should stay confident
            ml_short_sma_5 = ml_short.rolling(5).mean().fillna(ml_short)
            ml_short_sma_10 = ml_short.rolling(10).mean().fillna(ml_short)
            ml_short_std_5 = ml_short.rolling(5).std().fillna(0.1)
            ai_short_stable = (
                (ml_short > 0.6)
                & (ml_short_sma_5 > 0.6)
                & (ml_short_std_5 < 0.15)
                & (ml_enhanced_short > 0.65)
                & (ml_short > ml_short.shift(1))
            )

            # 2. AI DEGRADATION DETECTION
            # Exit signals when AI predictions deteriorate

            # For LONG positions: Exit when AI confidence drops significantly
            ai_long_degradation = (
                # Rapid confidence loss
                (ml_prob < 0.45)  # Prediction dropped below neutral
                | (ml_prob < ml_prob.shift(2) - 0.2)  # 20% drop in 2 candles
                | (ml_prob_sma_5 < ml_prob_sma_10 - 0.1)  # Trend deteriorating
                |
                # High prediction volatility (uncertainty)
                (ml_prob_std_5 > 0.25)
                |
                # Enhanced score contradiction
                ((ml_prob > 0.6) & (ml_enhanced < 0.4))  # Conflicting signals
            )

            # For SHORT positions: Exit when AI confidence drops for shorts
            ai_short_degradation = (
                (ml_short < 0.45)
                | (ml_short < ml_short.shift(2) - 0.2)
                | (ml_short_sma_5 < ml_short_sma_10 - 0.1)
                | (ml_short_std_5 > 0.25)
                | ((ml_short > 0.6) & (ml_enhanced_short < 0.4))
            )

            # 3. COMBINED AI EXIT LOGIC
            # Exit LONG when AI shows degradation AND no stability
            ai_long_exit = ai_long_degradation & (~ai_long_stable)

            # Exit SHORT when AI shows degradation AND no stability
            ai_short_exit = ai_short_degradation & (~ai_short_stable)

            long_stability_signal = ai_long_stable
            short_stability_signal = ai_short_stable
            long_degradation_signal = ai_long_exit
            short_degradation_signal = ai_short_exit

            # Simple profit conditions (backup to AI logic)
            rolling_high = df["high"].rolling(20).max()
            current_drawdown = (rolling_high - df["close"]) / rolling_high

            profit_exit_signal = (
                df["close"] > df["close"].shift(20) * 1.06
            ) & (  # 6%+ gain from 20 candles ago
                current_drawdown > 0.02
            )  # But now dropped 2%+ from recent high

            resistance_profit_exit = (
                at_resistance
                & (
                    df["close"] > df["close"].shift(10) * 1.04
                )  # 4%+ gain from 10 candles ago
                & (df["rsi"] > 65)  # Overbought
                & (df["close"] < df["high"])  # Didn't close at high
            )

            # Traditional long profit-taking, only when the long model isn't stable
            current_profit_signal = (
                profit_exit_signal | resistance_profit_exit
            ) & (~long_stability_signal)

        except Exception as e:
            # If any error, continue with normal exit logic
            logger.warning(
                f"AI exit logic error for {metadata.get('pair', 'unknown')}: {e}"
            )
            # Fallback to simple profit signals
            try:
                rolling_high = df["high"].rolling(20).max()
                current_drawdown = (rolling_high - df["close"]) / rolling_high
                current_profit_signal = (df["close"] > df["close"].shift(20) * 1.06) & (
                    current_drawdown > 0.02
                )
            except Exception:
                current_profit_signal = pd.Series([False] * len(df), index=df.index)

        # ===========================================
        # LONG EXIT SIGNALS (ADVANCED MML SYSTEM)
        # ===========================================

        # 1. Profit-Taking Exits
        long_exit_resistance_profit = (
            at_resistance
            & (df["close"] < df["high"])  # Failed to close at high
            & (df["rsi"] > 65)  # Overbought
            & (df["maxima"] == 1)  # Local top
            & (df["volume"] > df["volume"].rolling(10).mean())
        )

        long_exit_extreme_overbought = (
            (df["close"] > df["[7/8]P"])
            & (df["rsi"] > 75)
            & (df["close"] < df["close"].shift(1))  # Price turning down
            & (df["maxima"] == 1)
        )

        long_exit_volume_exhaustion = (
            at_resistance
            & (
                df["volume"] < df["volume"].rolling(20).mean() * 0.6
            )  # Tightened from 0.8
            & (df["rsi"] > 70)
            & (df["close"] < df["close"].shift(1))
            & (df["close"] < df["close"].rolling(3).mean())  # Added price confirmation
        )

        # 2. Structure Breakdown (Improved with strong filters)
        long_exit_structure_breakdown = (
            (df["close"] < df["[4/8]P"])
            & (df["close"].shift(1) >= df["[4/8]P"].shift(1))
            & bullish_mml.shift(1)
            & (df["close"] < df["[4/8]P"] * 0.995)
            & (df["close"] < df["close"].shift(1))
            & (df["close"] < df["close"].shift(2))
            & (df["rsi"] < 45)  # Tightened from 50
            & (
                df["volume"] > df["volume"].rolling(15).mean() * 2.0
            )  # Increased from 1.5
            & (df["close"] < df["open"])
            & (df["low"] < df["low"].shift(1))
            & (df["close"] < df["close"].rolling(3).mean())
            # momentum_quality is a 0..5 count, so the old "< 0" never fired and this
            # whole exit was dead.
            & (df["momentum_quality"] <= 1)
        )

        # 3. Momentum Divergence
        long_exit_momentum_divergence = (
            at_resistance
            & (df["rsi"] < df["rsi"].shift(1))  # RSI falling
            & (df["rsi"].shift(1) < df["rsi"].shift(2))  # RSI was falling
            & (df["rsi"] < df["rsi"].shift(3))  # 3-candle RSI decline
            & (df["close"] >= df["close"].shift(1))  # Price still up/flat
            & (df["maxima"] == 1)
            & (df["rsi"] > 60)  # Only in overbought territory
        )

        # 4. Range Exit
        long_exit_range = (
            (df["close"] >= df["[2/8]P"])
            & (df["close"] <= df["[6/8]P"])  # In range
            & (df["high"] >= df["[6/8]P"])  # HIGH touched 75%, not close
            & (df["close"] < df["[6/8]P"] * 0.995)  # But closed below
            & (df["rsi"] > 65)  # More conservative RSI
            & (df["maxima"] == 1)
            & (
                df["volume"] > df["volume"].rolling(10).mean() * 1.2
            )  # Volume confirmation
        )

        # 5. Emergency Exit
        long_exit_emergency = (
            (
                (df["close"] < df["[0/8]P"])
                & (df["rsi"] < 20)  # Changed from 15
                & (
                    df["volume"] > df["volume"].rolling(20).mean() * 2.5
                )  # Reduced from 3
                & (df["close"] < df["close"].shift(1))
                & (df["close"] < df["close"].shift(2))
                & (df["close"] < df["open"])
            )
            if self.use_emergency_exits
            else pd.Series([False] * len(df), index=df.index)
        )

        # ===========================================
        # AI-ENHANCED EXIT COMBINATION
        # ===========================================

        ml_prob_sma_20 = ml_prob.rolling(20).mean().fillna(ml_prob)
        ml_trend_strength = (ml_prob - ml_prob_sma_20).abs()
        ml_short_sma_20 = ml_short.rolling(20).mean().fillna(ml_short)
        ml_short_trend_strength = (ml_short - ml_short_sma_20).abs()

        try:
            # AI Override Logic: Don't exit if AI shows strong consistent signals
            ai_override_long = (
                long_stability_signal  # AI is stable
                & (ml_prob > 0.7)  # High confidence for long
                & (ml_trend_strength < 0.1)  # Low deviation from 20-period average
            )

            # Traditional MML exit signals
            traditional_mml_exits = (
                long_exit_resistance_profit
                | long_exit_extreme_overbought
                | long_exit_volume_exhaustion
                | long_exit_structure_breakdown
                | long_exit_momentum_divergence
                | long_exit_range
                | long_exit_emergency
            )

            # Final AI-enhanced exit decision
            any_long_exit = (
                # AI degradation signal (highest priority - always exit)
                long_degradation_signal
                |
                # Traditional profit-taking when AI not stable
                current_profit_signal
                |
                # Traditional MML exits unless AI strongly overrides
                (traditional_mml_exits & (~ai_override_long))
            )

            # Log AI decision for debugging (only when signal changes)
            if len(df) > 1:
                current_ai_exit = bool(long_degradation_signal.iloc[-1])
                current_ai_stable = bool(long_stability_signal.iloc[-1])
                current_ml_prob = ml_prob.iloc[-1] if len(ml_prob) > 0 else 0.5
                current_ai_override = (
                    ai_override_long.iloc[-1] if len(ai_override_long) > 0 else False
                )

                # Initialize logging state if needed
                if not hasattr(self, "_last_ai_state"):
                    self._last_ai_state = {}

                pair_key = metadata.get("pair", "UNKNOWN")
                last_state = self._last_ai_state.get(pair_key, {})

                # Log significant AI state changes
                if (
                    abs(current_ml_prob - last_state.get("ml_prob", 0.5)) > 0.1
                    or current_ai_exit != last_state.get("ai_exit", False)
                    or current_ai_stable != last_state.get("ai_stable", False)
                    or current_ai_override != last_state.get("ai_override", False)
                ):

                    logger.debug(
                        f"AI Exit Analysis {pair_key}: "
                        f"ML_Prob={current_ml_prob:.3f}, "
                        f"AI_Exit={current_ai_exit}, "
                        f"AI_Stable={current_ai_stable}, "
                        f"AI_Override={current_ai_override}"
                    )

                # Store current state
                self._last_ai_state[pair_key] = {
                    "ml_prob": current_ml_prob,
                    "ai_exit": current_ai_exit,
                    "ai_stable": current_ai_stable,
                    "ai_override": current_ai_override,
                }

        except Exception as e:
            logger.warning(
                f"AI exit combination error for {metadata.get('pair', 'unknown')}: {e}"
            )
            # Fallback to traditional logic
            any_long_exit = (
                current_profit_signal
                | long_exit_resistance_profit
                | long_exit_extreme_overbought
                | long_exit_volume_exhaustion
                | long_exit_structure_breakdown
                | long_exit_momentum_divergence
                | long_exit_range
                | long_exit_emergency
            )

        # ===========================================
        # SHORT EXIT SIGNALS (if enabled)
        # ===========================================

        if self.can_short:
            # 1. Profit-Taking Exits
            short_exit_support_profit = (
                at_support
                & (df["close"] > df["low"])  # Failed to close at low
                & (df["rsi"] < 35)  # Oversold
                & (df["minima"] == 1)  # Local bottom
                & (df["volume"] > df["volume"].rolling(10).mean())
            )

            short_exit_extreme_oversold = (
                (df["close"] < df["[1/8]P"])
                & (df["rsi"] < 25)
                & (df["close"] > df["close"].shift(1))  # Price turning up
                & (df["minima"] == 1)
            )

            short_exit_volume_exhaustion = (
                at_support
                & (
                    df["volume"] < df["volume"].rolling(20).mean() * 0.6
                )  # Tightened from 0.8
                & (df["rsi"] < 30)
                & (df["close"] > df["close"].shift(1))
                & (
                    df["close"] > df["close"].rolling(3).mean()
                )  # Added price confirmation
            )

            # 2. Structure Breakout
            short_exit_structure_breakout = (
                (df["close"] > df["[4/8]P"])
                & (df["close"].shift(1) <= df["[4/8]P"].shift(1))
                & bearish_mml.shift(1)
                & (df["close"] > df["[4/8]P"] * 1.005)
                & (df["close"] > df["close"].shift(1))
                & (df["close"] > df["close"].shift(2))
                & (df["rsi"] > 55)  # Tightened from 50
                & (
                    df["volume"] > df["volume"].rolling(15).mean() * 2.0
                )  # Increased from 1.5
                & (df["close"] > df["open"])
                & (df["high"] > df["high"].shift(1))
                & (df["momentum_quality"] >= 3)
            )

            # 3. Momentum Divergence
            short_exit_momentum_divergence = (
                at_support
                & (df["rsi"] > df["rsi"].shift(1))  # RSI rising
                & (df["rsi"].shift(1) > df["rsi"].shift(2))  # RSI was rising
                & (df["rsi"] > df["rsi"].shift(3))  # 3-candle RSI rise
                & (df["close"] <= df["close"].shift(1))  # Price still down/flat
                & (df["minima"] == 1)
                & (df["rsi"] < 40)  # Only in oversold territory
            )

            # 4. Range Exit
            short_exit_range = (
                (df["close"] >= df["[2/8]P"])
                & (df["close"] <= df["[6/8]P"])  # In range
                & (df["low"] <= df["[2/8]P"])  # LOW touched 25%
                & (df["close"] > df["[2/8]P"] * 1.005)  # But closed above
                & (df["rsi"] < 35)  # More conservative RSI
                & (df["minima"] == 1)
                & (
                    df["volume"] > df["volume"].rolling(10).mean() * 1.2
                )  # Volume confirmation
            )

            # 5. Emergency Exit
            short_exit_emergency = (
                (
                    (df["close"] > df["[8/8]P"])
                    & (df["rsi"] > 80)  # Changed from 85
                    & (
                        df["volume"] > df["volume"].rolling(20).mean() * 2.5
                    )  # Reduced from 3
                    & (df["close"] > df["close"].shift(1))
                    & (df["close"] > df["close"].shift(2))
                    & (df["close"] > df["open"])
                )
                if self.use_emergency_exits
                else pd.Series([False] * len(df), index=df.index)
            )

            # ===========================================
            # AI-ENHANCED SHORT EXIT COMBINATION
            # ===========================================

            try:
                # AI Override Logic for SHORT positions
                ai_override_short = (
                    short_stability_signal
                    & (ml_short > 0.7)  # High confidence for short
                    & (ml_short_trend_strength < 0.1)
                )

                # Traditional MML short exit signals
                traditional_short_exits = (
                    short_exit_support_profit
                    | short_exit_extreme_oversold
                    | short_exit_volume_exhaustion
                    | short_exit_structure_breakout
                    | short_exit_momentum_divergence
                    | short_exit_range
                    | short_exit_emergency
                )

                # Final AI-enhanced SHORT exit decision
                any_short_exit = (
                    short_degradation_signal
                    |
                    # Traditional short exits unless AI strongly overrides
                    (traditional_short_exits & (~ai_override_short))
                )

            except Exception as e:
                logger.warning(
                    f"AI short exit error for {metadata.get('pair', 'unknown')}: {e}"
                )
                # Fallback to traditional logic
                any_short_exit = (
                    short_exit_support_profit
                    | short_exit_extreme_oversold
                    | short_exit_volume_exhaustion
                    | short_exit_structure_breakout
                    | short_exit_momentum_divergence
                    | short_exit_range
                    | short_exit_emergency
                )
        else:
            any_short_exit = false_series.copy()

        any_long_exit = any_long_exit.fillna(False).astype(bool)
        any_short_exit = any_short_exit.fillna(False).astype(bool)

        # ===========================================
        # COORDINATION WITH ENTRY SIGNALS
        # ===========================================

        # If we have new Entry signals, they override Exit signals
        has_long_entry = "enter_long" in df.columns and (df["enter_long"] == 1).any()
        has_short_entry = "enter_short" in df.columns and (df["enter_short"] == 1).any()

        if has_long_entry:
            long_entry_mask = df["enter_long"] == 1
            any_long_exit = any_long_exit & (~long_entry_mask)

        if has_short_entry and self.can_short:
            short_entry_mask = df["enter_short"] == 1
            any_short_exit = any_short_exit & (~short_entry_mask)

        # ===========================================
        # SET FINAL EXIT SIGNALS AND TAGS
        # ===========================================

        # Long Exits
        df.loc[any_long_exit, "exit_long"] = 1

        # Tags for Long Exits (Priority: AI > Emergency > Structure > Profit)
        df.loc[any_long_exit & long_degradation_signal, "exit_tag"] = (
            "AI_Degradation_Exit"
        )
        df.loc[any_long_exit & long_exit_emergency, "exit_tag"] = (
            "MML_Emergency_Long_Exit"
        )
        df.loc[
            any_long_exit & current_profit_signal & (df["exit_tag"] == ""), "exit_tag"
        ] = "AI_Profit_Taking"
        df.loc[
            any_long_exit & long_exit_structure_breakdown & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Structure_Breakdown_Confirmed"
        df.loc[
            any_long_exit & long_exit_resistance_profit & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Resistance_Profit"
        df.loc[
            any_long_exit & long_exit_extreme_overbought & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Extreme_Overbought"
        df.loc[
            any_long_exit & long_exit_volume_exhaustion & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Volume_Exhaustion_Long"
        df.loc[
            any_long_exit & long_exit_momentum_divergence & (df["exit_tag"] == ""),
            "exit_tag",
        ] = "MML_Momentum_Divergence_Long"
        df.loc[any_long_exit & long_exit_range & (df["exit_tag"] == ""), "exit_tag"] = (
            "MML_Range_Exit_Long"
        )

        # Short Exits
        if self.can_short:
            df.loc[any_short_exit, "exit_short"] = 1

            # Tags for Short Exits (Priority: AI > Emergency > Structure > Profit)
            df.loc[any_short_exit & short_degradation_signal, "exit_tag"] = (
                "AI_Degradation_Exit"
            )
            df.loc[any_short_exit & short_exit_emergency, "exit_tag"] = (
                "MML_Emergency_Short_Exit"
            )
            df.loc[
                any_short_exit & short_exit_structure_breakout & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Structure_Breakout_Confirmed"
            df.loc[
                any_short_exit & short_exit_support_profit & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Support_Profit"
            df.loc[
                any_short_exit & short_exit_extreme_oversold & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Extreme_Oversold"
            df.loc[
                any_short_exit & short_exit_volume_exhaustion & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Volume_Exhaustion_Short"
            df.loc[
                any_short_exit
                & short_exit_momentum_divergence
                & (df["exit_tag"] == ""),
                "exit_tag",
            ] = "MML_Momentum_Divergence_Short"
            df.loc[
                any_short_exit & short_exit_range & (df["exit_tag"] == ""), "exit_tag"
            ] = "MML_Range_Exit_Short"

        return df

    def _populate_simple_exits(
        self, dataframe: pd.DataFrame, metadata: dict
    ) -> pd.DataFrame:
        """
        SIMPLE OPPOSITE SIGNAL EXIT SYSTEM - SYNTAX FIXED
        """

        # Exit LONG when any SHORT signal appears
        long_exit_on_short = dataframe["enter_short"] == 1

        # Exit SHORT when any LONG signal appears
        short_exit_on_long = dataframe["enter_long"] == 1

        # Emergency exits (if enabled)
        if self.use_emergency_exits:
            emergency_long_exit = (
                (dataframe["rsi"] > 85)
                & (dataframe["volume"] > dataframe["avg_volume"] * 3)
                & (dataframe["close"] < dataframe["open"])
                & (dataframe["close"] < dataframe["low"].shift(1))
            ) | (
                (dataframe.get("structure_break_down", 0) == 1)
                & (dataframe["volume"] > dataframe["avg_volume"] * 2.5)
                & (dataframe["atr"] > dataframe["atr"].rolling(20).mean() * 2)
            )

            emergency_short_exit = (
                (dataframe["rsi"] < 15)
                & (dataframe["volume"] > dataframe["avg_volume"] * 3)
                & (dataframe["close"] > dataframe["open"])
                & (dataframe["close"] > dataframe["high"].shift(1))
            ) | (
                (dataframe.get("structure_break_up", 0) == 1)
                & (dataframe["volume"] > dataframe["avg_volume"] * 2.5)
                & (dataframe["atr"] > dataframe["atr"].rolling(20).mean() * 2)
            )
        else:
            emergency_long_exit = pd.Series(
                [False] * len(dataframe), index=dataframe.index
            )
            emergency_short_exit = pd.Series(
                [False] * len(dataframe), index=dataframe.index
            )

        # Apply exits
        dataframe.loc[long_exit_on_short, "exit_long"] = 1
        dataframe.loc[long_exit_on_short, "exit_tag"] = "trend_reversal"

        dataframe.loc[short_exit_on_long, "exit_short"] = 1
        dataframe.loc[short_exit_on_long, "exit_tag"] = "trend_reversal"

        # Emergency exits
        dataframe.loc[emergency_long_exit & ~long_exit_on_short, "exit_long"] = 1
        dataframe.loc[emergency_long_exit & ~long_exit_on_short, "exit_tag"] = (
            "emergency_exit"
        )

        dataframe.loc[emergency_short_exit & ~short_exit_on_long, "exit_short"] = 1
        dataframe.loc[emergency_short_exit & ~short_exit_on_long, "exit_tag"] = (
            "emergency_exit"
        )

        # DEBUGGING (FIXED THE ERROR HERE)
        if metadata["pair"] in ["BTC/USDT:USDT", "ETH/USDT:USDT"]:
            recent_exits = (
                dataframe["exit_long"].tail(5).sum()
                + dataframe["exit_short"].tail(5).sum()
            )
            if recent_exits > 0:
                exit_tag = dataframe["exit_tag"].iloc[-1]
                logger.info(f"{metadata['pair']} EXIT SIGNAL - Tag: {exit_tag}")
                # âœ… FIXED: Use the correct attribute name
                logger.info(
                    f"  Exit System: {'Custom MML' if self.use_custom_exits_advanced else 'Simple Opposite'}"
                )
                logger.info(f"  RSI: {dataframe['rsi'].iloc[-1]:.1f}")

        return dataframe

    # ===========================================
    # DCA / SAFETY ORDERS
    # ===========================================
    def _dca_stake_multiplier(self) -> float:
        """Total stake of a fully-averaged trade, in units of the initial order."""
        vs = self.safety_order_volume_scale.value
        return 1.0 + sum(vs**n for n in range(1, self.max_safety_orders.value + 1))

    def _safety_order_triggers(self) -> list[float]:
        """Cumulative loss from the first entry at which each safety order fires.
        With the defaults (-1.8%, step 1.25): -1.8%, -4.05%, -6.86%, ..."""
        first = self.initial_safety_order_trigger.value
        step = self.safety_order_step_scale.value
        triggers, total = [], 0.0
        for n in range(self.max_safety_orders.value):
            total += first * step**n
            triggers.append(total)
        return triggers

    def custom_stake_amount(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_stake: float,
        min_stake: Optional[float],
        max_stake: float,
        leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        """Reserve budget for safety orders.

        The configured stake becomes the budget for the WHOLE averaged position, so
        DCA doesn't silently multiply your risk per trade. Example with one safety
        order at volume scale 1.4: 100 USDT -> 41.7 initial + 58.3 safety order.
        """
        if not self.dca_enabled.value:
            return proposed_stake
        stake = proposed_stake / self._dca_stake_multiplier()
        if min_stake and stake < min_stake:
            # Budget too small to split: enter with the minimum; safety orders will
            # then be skipped if they don't fit the remaining balance.
            stake = min_stake
        return min(stake, max_stake)

    def adjust_trade_position(
        self,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        min_stake: Optional[float],
        max_stake: float,
        current_entry_rate: float,
        current_exit_rate: float,
        current_entry_profit: float,
        current_exit_profit: float,
        **kwargs,
    ):
        """Place safety orders when the trade moves against us.

        The DCA parameters existed before but nothing used them (no
        adjust_trade_position, position_adjustment_enable off). Rules:
        * trigger N is measured from the FIRST entry price, as a leveraged loss
          ratio (same scale as current_profit), so averaging down does not reset
          the distance to the next order;
        * order N stake = first stake * volume_scale ** N;
        * at most one safety order per candle, never with an order still open;
        * skipped during an adverse flash move / volume dump (falling knife) or
          when the ML model for the trade's side has dropped below dca_ml_guard.
        """
        if not self.dca_enabled.value or trade.has_open_orders:
            return None

        filled_entries = trade.select_filled_orders(trade.entry_side)
        if not filled_entries:
            return None
        safety_done = trade.nr_of_successful_entries - 1
        if safety_done >= self.max_safety_orders.value:
            return None

        first_order = filled_entries[0]
        first_price = first_order.safe_price
        if not first_price or current_rate <= 0:
            return None
        leverage = trade.leverage or 1.0
        direction = -1.0 if trade.is_short else 1.0
        loss_from_first = (current_rate / first_price - 1.0) * direction * leverage

        trigger = self._safety_order_triggers()[safety_done]
        if loss_from_first > trigger:
            return None

        # One safety order per candle
        last_fill = filled_entries[-1].order_filled_utc
        candle = timedelta(seconds=timeframe_to_seconds(self.timeframe))
        if last_fill and current_time - last_fill < candle:
            return None

        # Market-condition guards
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
        if dataframe is not None and not dataframe.empty:
            last = dataframe.iloc[-1]
            adverse = ("flash_pump", "volume_pump") if trade.is_short else ("flash_dump", "volume_dump")
            if any(bool(last.get(col, False)) for col in adverse):
                logger.info(f"{trade.pair} DCA skipped: adverse flash move in progress")
                return None
            prob_col = "ml_short_probability" if trade.is_short else "ml_entry_probability"
            prob = last.get(prob_col, 0.5)
            if pd.notna(prob) and prob < self.dca_ml_guard.value:
                logger.info(f"{trade.pair} DCA skipped: {prob_col}={prob:.3f} < {self.dca_ml_guard.value}")
                return None

        stake = first_order.stake_amount_filled * self.safety_order_volume_scale.value ** (safety_done + 1)
        if min_stake:
            stake = max(stake, min_stake)
        if stake > max_stake:
            logger.info(f"{trade.pair} DCA skipped: stake {stake:.2f} > available {max_stake:.2f}")
            return None

        logger.info(
            f"{trade.pair} Safety order {safety_done + 1}/{self.max_safety_orders.value}: "
            f"loss from first entry {loss_from_first:.2%} <= {trigger:.2%}, stake {stake:.2f}"
        )
        return stake, f"safety_order_{safety_done + 1}"

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        """Regime-protection exits.

        The flash-move / volume-spike / sentiment / BTC / emergency parameters and
        indicator columns existed, and confirm_trade_exit even whitelisted their
        emoji tags, but nothing ever produced those exits. This wires them up:
        each one only fires against the trade's direction and only while the trade
        is in profit, so it locks gains instead of duplicating the stoploss.
        """
        if not self.regime_change_enabled.value:
            return None
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return None
        last = dataframe.iloc[-1]

        def flag(col: str) -> bool:
            val = last.get(col, False)
            return bool(val) if pd.notna(val) else False

        lock = self.emergency_exit_profit_threshold.value
        if trade.is_short:
            flash, vol, sentiment, btc = "flash_pump", "volume_pump", "sentiment_shift_bull", "btc_flash_pump"
            side = "Pump"
        else:
            flash, vol, sentiment, btc = "flash_dump", "volume_dump", "sentiment_shift_bear", "btc_flash_dump"
            side = "Dump"

        if self.flash_move_enabled.value and flag(flash) and current_profit > lock:
            return f"⚡ Flash {side} Protection"
        if self.volume_spike_enabled.value and flag(vol) and current_profit > 0:
            return f"🔊 Volume {side} Protection"
        if self.sentiment_protection_enabled.value and flag(sentiment) and current_profit > lock:
            return "🌊 Sentiment Shift Protection"
        if flag(btc) and current_profit > 0:
            return f"₿ BTC {side} Protection"
        if self.emergency_exit_enabled.value and flag("regime_alert") and current_profit > lock:
            return "🎯 Regime Alert Protection"
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
        """Never block stop-type exits.

        The old version rejected "trailing_stop_loss" whenever the trade was not in
        profit. With use_custom_stoploss, a stop that has been moved at least once is
        reported as trailing_stop_loss, so losing trades could not be stopped out
        and kept bleeding (freqtrade retries and keeps refusing). Blocking is now
        opt-in via allow_trailing_exit_when_negative=False and bounded by
        trailing_exit_min_profit.
        """
        if exit_reason == "trailing_stop_loss" and not self.allow_trailing_exit_when_negative.value:
            profit = trade.calc_profit_ratio(rate)
            if profit < self.trailing_exit_min_profit.value:
                logger.info(
                    f"{pair} Blocking trailing stop exit at {profit:.2%} "
                    f"(< {self.trailing_exit_min_profit.value:.2%})"
                )
                return False
        return True
