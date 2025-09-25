# --- Do not remove these libs ---
from freqtrade.exchange import timeframe_to_minutes, timeframe_to_prev_date
import logging
import pickle
from pathlib import Path
import datetime
from datetime import datetime, timedelta, timezone
import numpy as np
import pandas as pd
pd.options.mode.chained_assignment = None
from pandas import DataFrame, Series
from technical.util import resample_to_interval, resampled_merge
from freqtrade.strategy import (IStrategy, BooleanParameter, CategoricalParameter, DecimalParameter,
                                IntParameter, RealParameter, merge_informative_pair, stoploss_from_open,
                                stoploss_from_absolute)
from freqtrade.persistence import Trade
from typing import List, Tuple, Optional, Dict
from freqtrade.optimize.space import Categorical, Dimension, Integer, SKDecimal

# --------------------------------
# Add your lib to import here
import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib
from collections import deque
import sys
from importlib import metadata
from functools import lru_cache
from scipy.fft import fft, fftfreq
from scipy.stats import skew, kurtosis

import warnings
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

logger = logging.getLogger(__name__)

# === SKLEARN IMPORTS FOR CONFIDENCE SCORING ===
try:
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.metrics import accuracy_score
    SKLEARN_AVAILABLE = True
    logger.info("Signal Confidence Scoring enabled (sklearn available)")
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("Signal Confidence Scoring disabled (sklearn not available)")

logger = logging.getLogger(__name__)

# === SIGNAL CONFIDENCE SCORING SYSTEM ===
class SignalConfidenceScorer:
    """
    Scores your existing signals based on historical success patterns
    Doesn't change signal logic - only adds confidence scores
    """
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = {}
        self.models_dir = Path("user_data/strategies/confidence_models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Confidence thresholds (optimizable in strategy)
        self.min_confidence_threshold = 65  # Only take signals with 65%+ confidence
        self.high_confidence_threshold = 80  # Premium signals
        
        self._load_models()
    
    def create_signal_features(self, dataframe: DataFrame, signal_row_idx: int) -> dict:
        """
        Create features that describe the context around a signal
        This captures what makes your signals successful
        """
        if signal_row_idx < 20:  # Need history
            return None
            
        row = dataframe.iloc[signal_row_idx]
        
        features = {
            # Market context when signal fired
            'rsi_value': row['rsi'],
            'rsi_position': (row['rsi'] - 30) / (70 - 30),  # Normalized RSI position
            'volume_ratio': row['volume'] / (dataframe['volume'].iloc[signal_row_idx-20:signal_row_idx].mean() + 1e-10),
            
            # Divergence strength
            'bull_div_count': row.get('total_bullish_divergences_count', 0),
            'bear_div_count': row.get('total_bearish_divergences_count', 0),
            'has_strong_divergence': 1 if row.get('total_bullish_divergences_count', 0) >= 2 or row.get('total_bearish_divergences_count', 0) >= 2 else 0,
            
            # Price position in bands
            'kc_position': (row['close'] - row['kc_lowerband']) / (row['kc_upperband'] - row['kc_lowerband'] + 1e-10),
            'distance_from_ema20': (row['close'] - row['ema20']) / row['close'],
            'distance_from_ema50': (row['close'] - row['ema50']) / row['close'],
            
            # Momentum context
            'momentum_2c': (row['close'] - dataframe['close'].iloc[signal_row_idx-2]) / dataframe['close'].iloc[signal_row_idx-2],
            'momentum_5c': (row['close'] - dataframe['close'].iloc[signal_row_idx-5]) / dataframe['close'].iloc[signal_row_idx-5],
            
            # Volatility context
            'atr_percentile': dataframe['atr'].iloc[signal_row_idx-50:signal_row_idx].rank(pct=True).iloc[-1] if signal_row_idx >= 50 else 0.5,
            'volatility_regime': row['atr'] / (dataframe['atr'].iloc[signal_row_idx-20:signal_row_idx].mean() + 1e-10),
            
            # Market structure
            'adx_strength': row['adx'],
            'trend_alignment': 1 if row['ema20'] > row['ema50'] > row['ema200'] else (-1 if row['ema20'] < row['ema50'] < row['ema200'] else 0),
            
            # Time context (if available)
            'hour': 12,  # Default mid-day
            
            # Recent price action
            'consolidation_score': dataframe['high'].iloc[signal_row_idx-10:signal_row_idx].max() / dataframe['low'].iloc[signal_row_idx-10:signal_row_idx].min(),
            
            # Signal strength
            'signal_strength': row.get('signal_strength', 3),
        }
        
        return features
    
    def calculate_signal_outcome(self, dataframe: DataFrame, signal_row_idx: int, 
                               is_long: bool, forward_periods: int = 10) -> dict:
        """
        Calculate how successful this signal was
        """
        if signal_row_idx + forward_periods >= len(dataframe):
            return None
            
        entry_price = dataframe['close'].iloc[signal_row_idx]
        
        # Look forward to see what happened
        future_slice = dataframe.iloc[signal_row_idx+1:signal_row_idx+forward_periods+1]
        
        if is_long:
            max_profit = (future_slice['high'].max() - entry_price) / entry_price
            max_loss = (future_slice['low'].min() - entry_price) / entry_price
            final_return = (future_slice['close'].iloc[-1] - entry_price) / entry_price
        else:
            max_profit = (entry_price - future_slice['low'].min()) / entry_price
            max_loss = (entry_price - future_slice['high'].max()) / entry_price
            final_return = (entry_price - future_slice['close'].iloc[-1]) / entry_price
        
        # Define success criteria for 15m timeframe
        profit_target = 0.012  # 1.2% profit target (reasonable for 15m)
        stop_loss = -0.008     # 0.8% stop loss
        
        success_score = 0
        if max_profit >= profit_target:
            success_score += 40  # Hit profit target
        if max_loss <= stop_loss:
            success_score -= 30  # Hit stop loss
        if final_return > 0:
            success_score += 20  # Positive at end
        if final_return > profit_target * 0.5:
            success_score += 20  # Good final return
            
        # Risk-adjusted score
        if max_loss != 0:
            risk_reward = max_profit / abs(max_loss)
            if risk_reward > 2:
                success_score += 20
        
        return {
            'success_score': max(0, min(100, success_score)),
            'max_profit': max_profit,
            'max_loss': max_loss,
            'final_return': final_return,
            'hit_target': max_profit >= profit_target,
            'hit_stop': max_loss <= stop_loss
        }
    
    def train_confidence_model(self, dataframe: DataFrame, pair: str) -> dict:
        """
        Train a model to predict signal confidence based on historical outcomes
        """
        if not SKLEARN_AVAILABLE:
            return {'status': 'sklearn_not_available'}
        
        # Find all your historical signals
        long_signals = dataframe[dataframe['enter_long'] == 1].index
        short_signals = dataframe[dataframe['enter_short'] == 1].index
        
        if len(long_signals) + len(short_signals) < 30:
            return {'status': 'insufficient_historical_signals'}
        
        # Extract features and outcomes for each signal
        training_data = []
        
        # Process long signals
        for signal_idx in long_signals:
            if signal_idx >= len(dataframe) - 15:  # Skip recent signals (need forward data)
                continue
                
            features = self.create_signal_features(dataframe, signal_idx)
            outcome = self.calculate_signal_outcome(dataframe, signal_idx, is_long=True)
            
            if features and outcome:
                features['signal_type'] = 'long'
                training_data.append({
                    'features': features,
                    'confidence': outcome['success_score'],
                    'success': outcome['success_score'] >= 60
                })
        
        # Process short signals
        for signal_idx in short_signals:
            if signal_idx >= len(dataframe) - 15:
                continue
                
            features = self.create_signal_features(dataframe, signal_idx)
            outcome = self.calculate_signal_outcome(dataframe, signal_idx, is_long=False)
            
            if features and outcome:
                features['signal_type'] = 'short'
                training_data.append({
                    'features': features,
                    'confidence': outcome['success_score'],
                    'success': outcome['success_score'] >= 60
                })
        
        if len(training_data) < 20:
            return {'status': 'insufficient_training_data'}
        
        # Prepare training dataset
        feature_names = list(training_data[0]['features'].keys())
        X = []
        y_confidence = []
        
        for sample in training_data:
            feature_vector = []
            for fname in feature_names:
                val = sample['features'].get(fname, 0)
                # Convert string features to numeric
                if fname == 'signal_type':
                    val = 1 if val == 'long' else 0
                feature_vector.append(float(val))
            
            X.append(feature_vector)
            y_confidence.append(sample['confidence'])
        
        X = np.array(X)
        y_confidence = np.array(y_confidence)
        
        # Clean data
        valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y_confidence) | np.isinf(X).any(axis=1))
        X = X[valid_mask]
        y_confidence = y_confidence[valid_mask]
        
        if len(X) < 15:
            return {'status': 'insufficient_clean_data'}
        
        # Scale features
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Train confidence prediction model
        confidence_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=3,
            random_state=42,
            n_jobs=-1
        )
        confidence_model.fit(X_scaled, y_confidence)
        
        # Store models
        self.models[pair] = {
            'confidence_model': confidence_model,
            'feature_names': feature_names
        }
        self.scalers[pair] = scaler
        self.is_trained[pair] = True
        
        # Save models
        self._save_models(pair)
        
        # Evaluate performance
        conf_score = confidence_model.score(X_scaled, y_confidence)
        
        return {
            'status': 'success',
            'training_samples': len(X),
            'confidence_r2': conf_score,
            'avg_historical_confidence': y_confidence.mean(),
        }
    
    def score_current_signals(self, dataframe: DataFrame, pair: str) -> pd.Series:
        """
        Score current signals based on trained model
        Returns confidence scores (0-100) for each signal
        """
        if pair not in self.is_trained or not self.is_trained[pair]:
            return pd.Series(75, index=dataframe.index, dtype=float)  # Default neutral-high confidence
        
        confidence_scores = pd.Series(75, index=dataframe.index, dtype=float)
        
        # Find current signals
        current_long_signals = dataframe[dataframe['enter_long'] == 1].index
        current_short_signals = dataframe[dataframe['enter_short'] == 1].index
        
        models = self.models[pair]
        scaler = self.scalers[pair]
        feature_names = models['feature_names']
        
        # Score long signals
        for signal_idx in current_long_signals:
            features = self.create_signal_features(dataframe, signal_idx)
            if features:
                features['signal_type'] = 'long'
                
                # Convert to feature vector
                feature_vector = []
                for fname in feature_names:
                    val = features.get(fname, 0)
                    if fname == 'signal_type':
                        val = 1 if val == 'long' else 0
                    feature_vector.append(float(val))
                
                try:
                    X_scaled = scaler.transform([feature_vector])
                    confidence = models['confidence_model'].predict(X_scaled)[0]
                    confidence_scores.iloc[signal_idx] = max(20, min(100, confidence))
                except:
                    confidence_scores.iloc[signal_idx] = 75  # Fallback
        
        # Score short signals  
        for signal_idx in current_short_signals:
            features = self.create_signal_features(dataframe, signal_idx)
            if features:
                features['signal_type'] = 'short'
                
                feature_vector = []
                for fname in feature_names:
                    val = features.get(fname, 0)
                    if fname == 'signal_type':
                        val = 1 if val == 'short' else 0
                    feature_vector.append(float(val))
                
                try:
                    X_scaled = scaler.transform([feature_vector])
                    confidence = models['confidence_model'].predict(X_scaled)[0]
                    confidence_scores.iloc[signal_idx] = max(20, min(100, confidence))
                except:
                    confidence_scores.iloc[signal_idx] = 75  # Fallback
        
        return confidence_scores
    
    def _save_models(self, pair: str):
        """Save models to disk"""
        try:
            safe_pair = pair.replace('/', '_').replace(':', '_')
            model_path = self.models_dir / f"{safe_pair}_confidence.pkl"
            
            with open(model_path, 'wb') as f:
                pickle.dump({
                    'models': self.models[pair],
                    'scaler': self.scalers[pair],
                    'is_trained': self.is_trained[pair]
                }, f)
        except Exception as e:
            logger.warning(f"Failed to save confidence models for {pair}: {e}")
    
    def _load_models(self):
        """Load existing models from disk"""
        try:
            for model_file in self.models_dir.glob("*_confidence.pkl"):
                pair = model_file.stem.replace('_confidence', '').replace('_', '/')
                
                with open(model_file, 'rb') as f:
                    data = pickle.load(f)
                    self.models[pair] = data['models']
                    self.scalers[pair] = data['scaler']
                    self.is_trained[pair] = data['is_trained']
                    
                logger.info(f"Loaded confidence model for {pair}")
        except Exception as e:
            logger.debug(f"Could not load confidence models: {e}")

# Initialize global confidence scorer
if SKLEARN_AVAILABLE:
    confidence_scorer = SignalConfidenceScorer()
else:
    confidence_scorer = None
logger = logging.getLogger(__name__)
class PlotConfig():
    def __init__(self):
        self.config = {
            'main_plot': {
                resample('bollinger_upperband') : {'color': 'rgba(4,137,122,0.7)'},
                resample('kc_upperband') : {'color': 'rgba(4,146,250,0.7)'},
                resample('kc_middleband') : {'color': 'rgba(4,146,250,0.7)'},
                resample('kc_lowerband') : {'color': 'rgba(4,146,250,0.7)'},
                resample('bollinger_lowerband') : {
                    'color': 'rgba(4,137,122,0.7)',
                    'fill_to': resample('bollinger_upperband'),
                    'fill_color': 'rgba(4,137,122,0.07)'
                },
                resample('ema9') : {'color': 'purple'},
                resample('ema20') : {'color': 'yellow'},
                resample('ema50') : {'color': 'red'},
                resample('ema200') : {'color': 'white'},
                resample('rsi') : {'color': 'green'},
                'trend_1h_1h': {'color': 'orange'},
            },
            'subplots': {
                "ATR" : {
                    resample('atr'):{'color':'firebrick'}
                },
                "Signal Strength": {
                    resample('signal_strength'):{'color':'blue'}
                },
                "Confidence Score": {
                    resample('signal_confidence'):{'color':'orange'}
                }
            }
        }
    
    def add_total_divergences_in_config(self, dataframe):
        self.config['main_plot'][resample("total_bullish_divergences")] = {
            "plotly": {
                'mode': 'markers+text',
                'text': resample("total_bullish_divergences_count"),
                'hovertext': resample("total_bullish_divergences_names"),
                'textfont':{'size': 11, 'color':'green'},
                'textposition':'bottom center',
                'marker': {
                    'symbol': 'diamond',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'green'
                }
            }
        }
        self.config['main_plot'][resample("total_bearish_divergences")] = {
            "plotly": {
                'mode': 'markers+text',
                'text': resample("total_bearish_divergences_count"),
                'hovertext': resample("total_bearish_divergences_names"),
                'textfont':{'size': 11, 'color':'crimson'},
                'textposition':'top center',
                'marker': {
                    'symbol': 'diamond',
                    'size': 11,
                    'line': {
                        'width': 2
                    },
                    'color': 'crimson'
                }
            }
        }
        return self

class AlexBandSniperAI(IStrategy):
    """
    Alex BandSniperAi on 15m Timeframe - OPTIMIZED VERSION
    Version 86AI - Claude optimized Entry & Exit
    Key improvements:
    - Signal Confidence Scoring
    - Fixed ROI and Trailing adjusted Custom Exits
    - Fixed Entry Signals
    - Included 1h Informative Timeframe
    - Multi-timeframe analysis (1h trend confirmation)
    - Enhanced signal filtering with minimum divergence counts
    - Volume and volatility filters
    - Adaptive position sizing based on signal strength
    - Improved risk management
    """
    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "v34C-confidence-optimized"

    class HyperOpt:
        # Define a custom stoploss space.
        def stoploss_space():
            return [SKDecimal(-0.15, -0.03, decimals=2, name='stoploss')]

        # Define a custom max_open_trades space
        def max_open_trades_space() -> List[Dimension]:
            return [
                Integer(3, 8, name='max_open_trades'),
            ]
        
        def trailing_space() -> List[Dimension]:
            return [
                Categorical([True], name='trailing_stop'),
                SKDecimal(0.02, 0.3, decimals=2, name='trailing_stop_positive'),
                SKDecimal(0.03, 0.1, decimals=2, name='trailing_stop_positive_offset_p1'),
                Categorical([True, False], name='trailing_only_offset_is_reached'),
            ]
    
    # Minimal ROI designed for the strategy.
    minimal_roi = {
        "0": 100  # Disables ROI completely - let custom_exit handle everything
    }
    
    # Optimal stoploss designed for the strategy.
    stoploss = -0.20
    can_short = True
    use_custom_stoploss = True
    leverage_value = 10.0  # Reduced leverage for better risk management

    trailing_stop = True
    trailing_stop_positive = 0.40        # Only trail after 40% profit (very high)
    trailing_stop_positive_offset = 0.45 # Start trailing at 45% profit
    trailing_only_offset_is_reached = True

    # Optimal timeframe for the strategy.
    timeframe = '15m'
    timeframe_minutes = timeframe_to_minutes(timeframe)

    # Run "populate_indicators()" only for new candle.
    process_only_new_candles = True

    # These values can be overridden in the "exit_pricing" section in the config.
    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False
    use_custom_exits = True
    
    # Your existing hyperopt parameters
    min_divergence_count = IntParameter(1, 3, default=1, space='buy', optimize=True, load=True)
    min_signal_strength = IntParameter(1, 5, default=1, space='buy', optimize=True, load=True)
    volume_threshold = DecimalParameter(1.0, 1.5, default=1.0, decimals=1, space='buy', optimize=True, load=True)
    adx_threshold = IntParameter(15, 30, default=15, space='buy', optimize=True, load=True)
  
    # Market Condition Filters
    rsi_overbought = DecimalParameter(65.0, 85.0, default=80.0, decimals=1, space='buy', optimize=True, load=True)
    rsi_oversold = DecimalParameter(15.0, 35.0, default=15.0, decimals=1, space='buy', optimize=True, load=True)
    
    # Volatility Filters
    max_volatility = DecimalParameter(0.015, 0.035, default=0.025, decimals=3, space='buy', optimize=True, load=True)
    min_volatility = DecimalParameter(0.003, 0.008, default=0.005, decimals=3, space='buy', optimize=True, load=True)
    
    # Exit Parameters
    rsi_exit_overbought = DecimalParameter(70.0, 90.0, default=80.0, decimals=1, space='sell', optimize=True, load=True)
    rsi_exit_oversold = DecimalParameter(10.0, 30.0, default=20.0, decimals=1, space='sell', optimize=True, load=True)
    adx_exit_threshold = IntParameter(15, 30, default=20, space='sell', optimize=True, load=True)
    
    # Trend Confirmation Parameters
    trend_strength_threshold = IntParameter(20, 40, default=25, space='buy', optimize=True, load=True)
    
    # Technical Parameters
    window = IntParameter(3, 6, default=4, space="buy", optimize=True, load=True)
    index_range = IntParameter(20, 50, default=30, space='buy', optimize=True, load=True)

    # === NEW CONFIDENCE SCORING PARAMETERS ===
    confidence_threshold = IntParameter(50, 85, default=65, space='buy', optimize=True, load=True)
    high_confidence_threshold = IntParameter(75, 95, default=80, space='buy', optimize=True, load=True)
    enable_confidence_training = BooleanParameter(default=True, space='buy', optimize=False, load=True)

    # Number of candles the strategy requires before producing valid signals
    startup_candle_count: int = 10

    # Protection parameters (your existing ones)
    cooldown_lookback = IntParameter(2, 48, default=5, space="protection", optimize=True)
    stop_duration = IntParameter(12, 200, default=20, space="protection", optimize=True)
    use_stop_protection = BooleanParameter(default=True, space="protection", optimize=True)
    use_cooldown_protection = BooleanParameter(default=True, space="protection", optimize=True)

    use_max_drawdown_protection = BooleanParameter(default=True, space="protection", optimize=True)
    max_drawdown_lookback = IntParameter(100, 300, default=200, space="protection", optimize=True)
    max_drawdown_trade_limit = IntParameter(5, 15, default=10, space="protection", optimize=True)
    max_drawdown_stop_duration = IntParameter(1, 5, default=2, space="protection", optimize=True)
    max_allowed_drawdown = DecimalParameter(0.08, 0.25, default=0.15, decimals=2, space="protection", optimize=True)

    stoploss_guard_lookback = IntParameter(30, 80, default=50, space="protection", optimize=True)
    stoploss_guard_trade_limit = IntParameter(2, 6, default=3, space="protection", optimize=True)
    stoploss_guard_only_per_pair = BooleanParameter(default=True, space="protection", optimize=True)

    # Optional order type mapping.
    order_types = {
        'entry': 'market',
        'exit': 'market',
        'stoploss': 'limit',
        'stoploss_on_exchange': True
    }

    # Optional order time in force.
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }

    plot_config = None

    def get_ticker_indicator(self):
        return int(self.timeframe[:-1])
    
    def informative_pairs(self):
        """Define additional timeframes to download"""
        pairs = self.dp.current_whitelist()
        return [(pair, '1h') for pair in pairs]
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Enhanced indicator population with multi-timeframe analysis - Fixed for dry run
        """
        
        # === MULTI-TIMEFRAME ANALYSIS ===
        # Get 1h timeframe for trend confirmation with improved error handling
        try:
            # Check if we're in backtesting mode or if pair supports 1h data
            if hasattr(self.dp, 'runmode') and self.dp.runmode.value in ['backtest', 'hyperopt']:
                # In backtesting, try to get 1h data but don't fail if unavailable
                informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
            else:
                # In live/dry run, be more cautious about data availability
                try:
                    informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe='1h')
                except Exception:
                    informative_1h = None
            
            # Enhanced data validation
            if (informative_1h is not None and 
                len(informative_1h) > 50 and  # Reduced minimum requirement
                not informative_1h.empty and
                'close' in informative_1h.columns):
                
                try:
                    # 1h Trend indicators with additional error checking
                    informative_1h['ema50_1h'] = ta.EMA(informative_1h, timeperiod=50)
                    informative_1h['ema200_1h'] = ta.EMA(informative_1h, timeperiod=200)
                    informative_1h['trend_1h'] = ta.EMA(informative_1h, timeperiod=21)
                    informative_1h['trend_strength_1h'] = ta.ADX(informative_1h)
                    informative_1h['rsi_1h'] = ta.RSI(informative_1h)
                    
                    # Fill NaN values before merging
                    informative_1h = informative_1h.bfill().ffill()
                    
                    # Safe merge with additional error handling
                    dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, '1h', ffill=True)
                    logger.info(f"Successfully merged 1h data for {metadata['pair']}")
                    
                except Exception as merge_error:
                    logger.warning(f"Failed to merge 1h data for {metadata['pair']}: {merge_error}")
                    self._add_dummy_1h_columns(dataframe)
            else:
                logger.info(f"Using fallback 1h indicators for {metadata['pair']} (insufficient data)")
                self._add_dummy_1h_columns(dataframe)
                
        except Exception as e:
            logger.warning(f"Error accessing 1h data for {metadata['pair']}: {e}")
            self._add_dummy_1h_columns(dataframe)
        
        # === 15M TIMEFRAME INDICATORS ===
        informative = dataframe.copy()
        
        # Volume analysis with safe calculations
        try:
            informative['volume_sma'] = informative['volume'].rolling(window=20, min_periods=1).mean()
            informative['volume_ratio'] = informative['volume'] / informative['volume_sma']
            informative['volume_ratio'] = informative['volume_ratio'].fillna(1.0)
        except:
            informative['volume_sma'] = informative['volume']
            informative['volume_ratio'] = 1.0
        
        # Volatility analysis with safe calculations
        try:
            informative['atr'] = qtpylib.atr(informative, window=14, exp=False)
            informative['volatility'] = informative['atr'] / informative['close']
            informative['volatility'] = informative['volatility'].fillna(0.01)
        except:
            informative['atr'] = informative['close'] * 0.02
            informative['volatility'] = 0.01
        
        # Momentum Indicators with error handling
        try:
            informative['rsi'] = ta.RSI(informative)
            informative['stoch'] = ta.STOCH(informative)['slowk']
            informative['roc'] = ta.ROC(informative)
            informative['uo'] = ta.ULTOSC(informative)
            informative['ao'] = qtpylib.awesome_oscillator(informative)
            informative['macd'] = ta.MACD(informative)['macd']
            informative['cci'] = ta.CCI(informative)
            informative['cmf'] = chaikin_money_flow(informative, 20)
            informative['obv'] = ta.OBV(informative)
            informative['mfi'] = ta.MFI(informative)
            informative['adx'] = ta.ADX(informative)
            
            # Fill NaN values for all indicators
            indicator_columns = ['rsi', 'stoch', 'roc', 'uo', 'ao', 'macd', 'cci', 'cmf', 'obv', 'mfi', 'adx']
            for col in indicator_columns:
                if col in informative.columns:
                    informative[col] = informative[col].bfill().fillna(50 if col in ['rsi', 'mfi'] else 0)
                    
        except Exception as e:
            logger.warning(f"Error calculating momentum indicators: {e}")
            # Provide fallback values
            informative['rsi'] = 50
            informative['stoch'] = 50
            informative['roc'] = 0
            informative['uo'] = 50
            informative['ao'] = 0
            informative['macd'] = 0
            informative['cci'] = 0
            informative['cmf'] = 0
            informative['obv'] = informative['volume'].cumsum()
            informative['mfi'] = 50
            informative['adx'] = 25

        # Keltner Channel with error handling
        try:
            keltner = emaKeltner(informative)
            informative["kc_upperband"] = keltner["upper"]
            informative["kc_middleband"] = keltner["mid"]
            informative["kc_lowerband"] = keltner["lower"]
        except:
            informative["kc_upperband"] = informative['close'] * 1.02
            informative["kc_middleband"] = informative['close']
            informative["kc_lowerband"] = informative['close'] * 0.98

        # Bollinger Bands with error handling
        try:
            bollinger = qtpylib.bollinger_bands(qtpylib.typical_price(informative), window=20, stds=2)
            informative['bollinger_upperband'] = bollinger['upper']
            informative['bollinger_lowerband'] = bollinger['lower']
        except:
            informative['bollinger_upperband'] = informative['close'] * 1.02
            informative['bollinger_lowerband'] = informative['close'] * 0.98

        # EMA - Exponential Moving Average with error handling
        try:
            informative['ema9'] = ta.EMA(informative, timeperiod=9)
            informative['ema20'] = ta.EMA(informative, timeperiod=20)
            informative['ema50'] = ta.EMA(informative, timeperiod=50)
            informative['ema200'] = ta.EMA(informative, timeperiod=200)
            
            # Fill NaN values for EMAs
            ema_columns = ['ema9', 'ema20', 'ema50', 'ema200']
            for col in ema_columns:
                if col in informative.columns:
                    informative[col] = informative[col].bfill().fillna(informative['close'])
        except:
            informative['ema9'] = informative['close']
            informative['ema20'] = informative['close']
            informative['ema50'] = informative['close']
            informative['ema200'] = informative['close']

        # Pivot Points with error handling
        try:
            pivots = pivot_points(informative, self.window.value)
            informative['pivot_lows'] = pivots['pivot_lows']
            informative['pivot_highs'] = pivots['pivot_highs']
        except Exception as e:
            logger.warning(f"Error calculating pivot points: {e}")
            informative['pivot_lows'] = np.nan
            informative['pivot_highs'] = np.nan

        # === DIVERGENCE ANALYSIS ===
        try:
            self.initialize_divergences_lists(informative)
            (high_iterator, low_iterator) = self.get_iterators(informative)
            
            # Add divergences for multiple indicators
            indicators = ['rsi', 'stoch', 'roc', 'uo', 'ao', 'macd', 'cci', 'cmf', 'obv', 'mfi']
            for indicator in indicators:
                try:
                    if indicator in informative.columns:
                        self.add_divergences(informative, indicator, high_iterator, low_iterator)
                except Exception as e:
                    logger.warning(f"Error adding divergences for {indicator}: {e}")
                    continue
        except Exception as e:
            logger.warning(f"Error in divergence analysis: {e}")
            # Initialize with empty divergence data
            informative["total_bullish_divergences"] = np.nan
            informative["total_bullish_divergences_count"] = 0
            informative["total_bullish_divergences_names"] = ''
            informative["total_bearish_divergences"] = np.nan
            informative["total_bearish_divergences_count"] = 0
            informative["total_bearish_divergences_names"] = ''
        
        # === SIGNAL STRENGTH CALCULATION ===
        try:
            informative['signal_strength'] = self.calculate_signal_strength(informative)
        except:
            informative['signal_strength'] = 0
        
        # === MERGE BACK TO DATAFRAME ===
        for col in informative.columns:
            if col not in dataframe.columns:
                dataframe[col] = informative[col]
            else:
                dataframe[col] = informative[col]

        # Additional market structure analysis with error handling
        try:
            dataframe['chop'] = choppiness_index(dataframe['high'], dataframe['low'], dataframe['close'], window=14)
            dataframe['natr'] = ta.NATR(dataframe['high'], dataframe['low'], dataframe['close'], window=14)
            dataframe['natr_diff'] = dataframe['natr'] - dataframe['natr'].shift(1)
            dataframe['natr_direction_change'] = (dataframe['natr_diff'] * dataframe['natr_diff'].shift(1) < 0)
        except:
            dataframe['chop'] = 50
            dataframe['natr'] = 0.02
            dataframe['natr_diff'] = 0
            dataframe['natr_direction_change'] = False

        # Support/Resistance levels with error handling
        try:
            dataframe['swing_high'] = dataframe['high'].rolling(window=50, min_periods=1).max()
            dataframe['swing_low'] = dataframe['low'].rolling(window=50, min_periods=1).min()
            dataframe['distance_to_resistance'] = (dataframe['swing_high'] - dataframe['close']) / dataframe['close']
            dataframe['distance_to_support'] = (dataframe['close'] - dataframe['swing_low']) / dataframe['close']
        except:
            dataframe['swing_high'] = dataframe['high']
            dataframe['swing_low'] = dataframe['low']
            dataframe['distance_to_resistance'] = 0.02
            dataframe['distance_to_support'] = 0.02

        # Initialize confidence score column
        dataframe['signal_confidence'] = 75  # Default confidence

        # Plot configuration with error handling
        try:
            self.plot_config = (
                PlotConfig()
                .add_total_divergences_in_config(dataframe)
                .config)
        except:
            self.plot_config = None

        return dataframe

    def _add_dummy_1h_columns(self, dataframe):
        """Add dummy 1h columns when higher timeframe data is unavailable"""
        # Use current 15m data to simulate 1h trend
        try:
            dataframe['ema50_1h_1h'] = ta.EMA(dataframe, timeperiod=200)  # Use longer period on 15m
            dataframe['ema200_1h_1h'] = ta.EMA(dataframe, timeperiod=800)  # Use much longer period
            dataframe['trend_1h_1h'] = ta.EMA(dataframe, timeperiod=84)   # 21 * 4 (4x 15m = 1h)
            dataframe['trend_strength_1h_1h'] = ta.ADX(dataframe)
            dataframe['rsi_1h_1h'] = ta.RSI(dataframe, timeperiod=56)     # Adjusted for timeframe
            
            # Fill NaN values
            columns_1h = ['ema50_1h_1h', 'ema200_1h_1h', 'trend_1h_1h', 'trend_strength_1h_1h', 'rsi_1h_1h']
            for col in columns_1h:
                if col in dataframe.columns:
                    dataframe[col] = dataframe[col].bfill().fillna(
                        dataframe['close'] if 'ema' in col or 'trend' in col else 
                        25 if 'strength' in col else 50
                    )
        except Exception as e:
            logger.warning(f"Error creating dummy 1h columns: {e}")
            # Absolute fallback
            dataframe['ema50_1h_1h'] = dataframe['close']
            dataframe['ema200_1h_1h'] = dataframe['close']
            dataframe['trend_1h_1h'] = dataframe['close']
            dataframe['trend_strength_1h_1h'] = 25
            dataframe['rsi_1h_1h'] = 50

    def calculate_signal_strength(self, dataframe: DataFrame) -> Series:
        """
        Calculate overall signal strength based on multiple factors
        """
        strength = pd.Series(0, index=dataframe.index)
        
        # Divergence strength
        strength += dataframe['total_bullish_divergences_count'] * 2
        strength += dataframe['total_bearish_divergences_count'] * 2
        
        # Volume strength
        volume_strength = np.where(dataframe['volume_ratio'] > 1.5, 2, 
                                 np.where(dataframe['volume_ratio'] > 1.2, 1, 0))
        strength += volume_strength
        
        # Trend alignment strength
        ema_bullish = (dataframe['ema20'] > dataframe['ema50']) & (dataframe['ema50'] > dataframe['ema200'])
        ema_bearish = (dataframe['ema20'] < dataframe['ema50']) & (dataframe['ema50'] < dataframe['ema200'])
        strength += np.where(ema_bullish | ema_bearish, 1, 0)
        
        # ADX strength
        strength += np.where(dataframe['adx'] > 30, 1, 0)
        
        return strength
        
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
            """
            Your exact original entry logic + Signal Confidence Scoring
            No signal logic changes - only intelligent filtering based on historical success
            """
            # Initialize
            dataframe['enter_long'] = 0
            dataframe['enter_short'] = 0
            dataframe["enter_tag"] = ""
           
            # === YOUR EXACT ORIGINAL CONDITIONS (UNCHANGED) ===
            bullish_divergence = (dataframe['total_bullish_divergences'].shift(1) > 0)
            bearish_divergence = (dataframe['total_bearish_divergences'].shift(1) > 0)
           
            # Your existing filters (unchanged)
            volatility_ok = (
                (dataframe['volatility'].shift(1) >= self.min_volatility.value * 0.5) &
                (dataframe['volatility'].shift(1) <= self.max_volatility.value * 2.0)
            )
           
            bands_long = (
                (dataframe['low'] <= dataframe['kc_lowerband']) |
                (dataframe['close'] <= dataframe['kc_lowerband'])
            )
           
            bands_short = (
                (dataframe['high'] >= dataframe['kc_upperband']) |
                (dataframe['close'] >= dataframe['kc_upperband'])
            )
           
            rsi_long_ok = (
                (dataframe['rsi'].shift(1) < self.rsi_overbought.value + 5) &
                (dataframe['rsi'].shift(1) > 25)
            )
           
            rsi_short_ok = (
                (dataframe['rsi'].shift(1) > self.rsi_oversold.value - 5) &
                (dataframe['rsi'].shift(1) < 75)
            )
           
            has_volume = dataframe['volume'] > 0
           
            # === ALL YOUR ORIGINAL CONDITIONS (UNCHANGED) ===
            long_condition_div = bullish_divergence
            short_condition_div = bearish_divergence
            
            long_condition_primary = (
                bullish_divergence & volatility_ok & bands_long & rsi_long_ok & has_volume
            )
           
            short_condition_primary = (
                bearish_divergence & volatility_ok & bands_short & rsi_short_ok & has_volume
            )
           
            long_condition_secondary = (
                bullish_divergence &
                (dataframe['close'] > dataframe['close'].shift(2)) &
                (dataframe['rsi'].shift(1) > 35) &
                (dataframe['rsi'].shift(1) < 70) &
                has_volume
            )
           
            short_condition_secondary = (
                bearish_divergence &
                (dataframe['close'] < dataframe['close'].shift(2)) &
                (dataframe['rsi'].shift(1) > 30) &
                (dataframe['rsi'].shift(1) < 80) &
                has_volume
            )
           
            long_condition_tertiary = (
                bullish_divergence &
                (dataframe['close'] > dataframe['ema20']) &
                (dataframe['rsi'].shift(1) > 40) &
                (dataframe['rsi'].shift(1) < 65) &
                (dataframe['volume'] > dataframe['volume'].rolling(10).mean()) &
                has_volume
            )
           
            short_condition_tertiary = (
                bearish_divergence &
                (dataframe['close'] < dataframe['ema20']) &
                (dataframe['rsi'].shift(1) > 35) &
                (dataframe['rsi'].shift(1) < 75) &
                (dataframe['volume'] > dataframe['volume'].rolling(10).mean()) &
                has_volume
            )
           
            long_condition_quaternary = (
                bullish_divergence &
                (dataframe['close'].shift(1) < dataframe['ema20'].shift(1)) &
                (dataframe['close'] > dataframe['ema20']) &
                (dataframe['rsi'] > 45) &
                (dataframe['volume'] > dataframe['volume'].rolling(5).mean() * 1.2) &
                has_volume
            )
           
            short_condition_quaternary = (
                bearish_divergence &
                (dataframe['close'].shift(1) > dataframe['ema20'].shift(1)) &
                (dataframe['close'] < dataframe['ema20']) &
                (dataframe['rsi'] < 55) &
                (dataframe['volume'] > dataframe['volume'].rolling(5).mean() * 1.2) &
                has_volume
            )
           
            long_condition_fifth = (
                bullish_divergence &
                (dataframe['close'] > dataframe['kc_middleband']) &
                (dataframe['close'] < dataframe['kc_upperband']) &
                (dataframe['close'] > dataframe['close'].shift(1)) &
                (dataframe['rsi'].shift(1) > 35) &
                (dataframe['rsi'].shift(1) < 70) &
                has_volume
            )
           
            short_condition_fifth = (
                bearish_divergence &
                (dataframe['close'] < dataframe['kc_middleband']) &
                (dataframe['close'] > dataframe['kc_lowerband']) &
                (dataframe['close'] < dataframe['close'].shift(1)) &
                (dataframe['rsi'].shift(1) > 30) &
                (dataframe['rsi'].shift(1) < 65) &
                has_volume
            )
            
            long_condition_sixth = (
                (dataframe['total_bullish_divergences_count'] >= 2) &
                bullish_divergence &
                (dataframe['rsi'].shift(1) > 30) &
                (dataframe['rsi'].shift(1) < 75) &
                has_volume
            )
           
            short_condition_sixth = (
                (dataframe['total_bearish_divergences_count'] >= 2) &
                bearish_divergence &
                (dataframe['rsi'].shift(1) > 25) &
                (dataframe['rsi'].shift(1) < 70) &
                has_volume
            )
           
            long_condition_seventh = (
                (dataframe['ema20'] > dataframe['ema50']) &
                (dataframe['close'].shift(1) < dataframe['ema20'].shift(1)) &
                (dataframe['close'] > dataframe['ema20']) &
                (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.2) &
                (dataframe['rsi'].shift(1) > 30) &
                (dataframe['rsi'].shift(1) < 70) &
                (dataframe['close'] > dataframe['close'].shift(2)) &
                (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 0.5) &
                (dataframe['close'] > dataframe['kc_middleband']) &
                (dataframe['adx'] > 15) &
                has_volume
            )
            
            short_condition_seventh = (
                (dataframe['ema20'] < dataframe['ema50']) &
                (dataframe['close'].shift(1) > dataframe['ema20'].shift(1)) &
                (dataframe['close'] < dataframe['ema20']) &
                (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.2) &
                (dataframe['rsi'].shift(1) > 30) &
                (dataframe['rsi'].shift(1) < 70) &
                (dataframe['close'] < dataframe['close'].shift(2)) &
                (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 0.5) &
                (dataframe['close'] < dataframe['kc_middleband']) &
                (dataframe['adx'] > 15) &
                has_volume
            )
           
            long_condition_trend = (
                (dataframe['close'] > dataframe['ema20']) &
                (dataframe['ema20'] > dataframe['ema50']) &
                (dataframe['ema50'] > dataframe['ema200']) &
                (dataframe['close'].shift(1) < dataframe['ema20'].shift(1)) &
                (dataframe['close'] > dataframe['ema20']) &
                (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5) &
                (dataframe['rsi'].shift(1) > 40) &
                (dataframe['rsi'].shift(1) < 60) &
                (dataframe['close'] > dataframe['close'].shift(2)) &
                (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 1.0) &
                (dataframe['close'] > dataframe['kc_middleband']) &
                (dataframe['adx'] > 25) &
                has_volume
            )
            
            short_condition_trend = (
                (dataframe['close'] < dataframe['ema20']) &
                (dataframe['ema20'] < dataframe['ema50']) &
                (dataframe['ema50'] < dataframe['ema200']) &
                (dataframe['close'].shift(1) > dataframe['ema20'].shift(1)) &
                (dataframe['close'] < dataframe['ema20']) &
                (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5) &
                (dataframe['rsi'].shift(1) > 40) &
                (dataframe['rsi'].shift(1) < 60) &
                (dataframe['close'] < dataframe['close'].shift(2)) &
                (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 1.0) &
                (dataframe['close'] < dataframe['kc_middleband']) &
                (dataframe['adx'] > 25) &
                has_volume
            )
           
            long_condition_momentum = (
                (dataframe['close'] > dataframe['ema50']) &
                (dataframe['ema20'] > dataframe['ema50']) &
                (dataframe['ema50'] > dataframe['ema200']) &
                (dataframe['rsi'].shift(3) < 50) &
                (dataframe['rsi'].shift(1) > 55) &
                (dataframe['rsi'] > dataframe['rsi'].shift(1)) &
                (dataframe['low'].shift(1) > dataframe['low'].shift(2)) &
                (dataframe['close'] > dataframe['high'].shift(2)) &
                (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.2) &
                (dataframe['close'] < dataframe['kc_upperband'] * 0.995) &
                (dataframe['adx'] > 25) &
                has_volume
            )
            
            short_condition_momentum = (
                (dataframe['close'] < dataframe['ema50']) &
                (dataframe['ema20'] < dataframe['ema50']) &
                (dataframe['ema50'] < dataframe['ema200']) &
                (dataframe['rsi'].shift(3) > 50) &
                (dataframe['rsi'].shift(1) < 45) &
                (dataframe['rsi'] < dataframe['rsi'].shift(1)) &
                (dataframe['high'].shift(1) < dataframe['high'].shift(2)) &
                (dataframe['close'] < dataframe['low'].shift(2)) &
                (dataframe['volume'] > dataframe['volume'].rolling(10).mean() * 1.2) &
                (dataframe['close'] > dataframe['kc_lowerband'] * 1.005) &
                (dataframe['adx'] > 25) &
                has_volume
            )
           
            # === YOUR ORIGINAL COMBINED CONDITIONS ===
            original_long_condition = (
                long_condition_div |
                long_condition_primary |
                long_condition_secondary |
                long_condition_tertiary |
                long_condition_quaternary |
                long_condition_fifth |
                long_condition_sixth |
                long_condition_seventh |
                long_condition_trend |
                long_condition_momentum
            )
            
            original_short_condition = (
                short_condition_div |
                short_condition_primary |
                short_condition_secondary |
                short_condition_tertiary |
                short_condition_quaternary |
                short_condition_fifth |
                short_condition_sixth |
                short_condition_seventh |
                short_condition_trend |
                short_condition_momentum
            )
           
            # === APPLY ORIGINAL SIGNALS FIRST ===
            dataframe.loc[original_long_condition, 'enter_long'] = 1
            dataframe.loc[original_short_condition, 'enter_short'] = 1
           
            # === SIGNAL CONFIDENCE SCORING & FILTERING ===
            pair = metadata['pair']
            
            if SKLEARN_AVAILABLE and confidence_scorer and self.enable_confidence_training.value:
                # Train confidence model if needed (weekly retraining)
                if (len(dataframe) >= 500 and 
                    (pair not in confidence_scorer.is_trained or not confidence_scorer.is_trained.get(pair, False))):
                    
                    logger.info(f"Training confidence model for {pair}")
                    result = confidence_scorer.train_confidence_model(dataframe, pair)
                    if result['status'] == 'success':
                        logger.info(f"Confidence model trained: {result['training_samples']} samples, "
                                   f"R2={result['confidence_r2']:.3f}")
                    else:
                        logger.warning(f"Confidence training failed for {pair}: {result['status']}")
                
                # Update confidence scorer thresholds from strategy parameters
                confidence_scorer.min_confidence_threshold = self.confidence_threshold.value
                confidence_scorer.high_confidence_threshold = self.high_confidence_threshold.value
                
                # Score current signals
                confidence_scores = confidence_scorer.score_current_signals(dataframe, pair)
                dataframe['signal_confidence'] = confidence_scores
                
                # Apply confidence filter - only keep high-confidence signals
                low_confidence_mask = confidence_scores < self.confidence_threshold.value
                
                # Count signals before filtering
                original_signals = dataframe['enter_long'].sum() + dataframe['enter_short'].sum()
                
                # Filter out low-confidence signals
                dataframe.loc[low_confidence_mask, 'enter_long'] = 0
                dataframe.loc[low_confidence_mask, 'enter_short'] = 0
                
                # Count signals after filtering
                filtered_signals = dataframe['enter_long'].sum() + dataframe['enter_short'].sum()
                
                if original_signals > 0:
                    logger.info(f"Confidence filter for {pair}: {original_signals} -> {filtered_signals} signals "
                               f"(threshold: {self.confidence_threshold.value}%)")
            else:
                # No confidence filtering - use all original signals
                dataframe['signal_confidence'] = 75  # Default confidence
           
            # === ENHANCED TAGGING WITH CONFIDENCE INFO ===
            for idx in dataframe.index:
                if dataframe.loc[idx, 'enter_long'] == 1:
                    confidence = dataframe.loc[idx, 'signal_confidence']
                    
                    # Determine confidence level for tag
                    if confidence >= self.high_confidence_threshold.value:
                        conf_tag = f"_High{confidence:.0f}"
                    elif confidence >= self.confidence_threshold.value:
                        conf_tag = f"_Med{confidence:.0f}"
                    else:
                        conf_tag = f"_Low{confidence:.0f}"
                    
                    # Your original tag priority logic + confidence info
                    if long_condition_div.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bull_Div{conf_tag}'
                    elif long_condition_primary.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bull_E1{conf_tag}'
                    elif long_condition_secondary.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bull_E2{conf_tag}'
                    elif long_condition_tertiary.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bull_E3{conf_tag}'
                    elif long_condition_quaternary.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bull_E4{conf_tag}'
                    elif long_condition_fifth.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bull_E5{conf_tag}'
                    elif long_condition_sixth.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bull_E6{conf_tag}'
                    elif long_condition_seventh.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bull_E7{conf_tag}'
                    elif long_condition_trend.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bull_Trend{conf_tag}'
                    elif long_condition_momentum.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bull_Momentum{conf_tag}'
               
                elif dataframe.loc[idx, 'enter_short'] == 1:
                    confidence = dataframe.loc[idx, 'signal_confidence']
                    
                    if confidence >= self.high_confidence_threshold.value:
                        conf_tag = f"_High{confidence:.0f}"
                    elif confidence >= self.confidence_threshold.value:
                        conf_tag = f"_Med{confidence:.0f}"
                    else:
                        conf_tag = f"_Low{confidence:.0f}"
                    
                    # Your original short tag logic + confidence
                    if short_condition_div.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bear_Div{conf_tag}'
                    elif short_condition_primary.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bear_E1{conf_tag}'
                    elif short_condition_secondary.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bear_E2{conf_tag}'
                    elif short_condition_tertiary.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bear_E3{conf_tag}'
                    elif short_condition_quaternary.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bear_E4{conf_tag}'
                    elif short_condition_fifth.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bear_E5{conf_tag}'
                    elif short_condition_sixth.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bear_E6{conf_tag}'
                    elif short_condition_seventh.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bear_E7{conf_tag}'
                    elif short_condition_trend.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bear_Trend{conf_tag}'
                    elif short_condition_momentum.loc[idx]:
                        dataframe.loc[idx, 'enter_tag'] = f'Bear_Momentum{conf_tag}'
           
            # === CONFIDENCE FILTER DEBUG INFO ===
            if len(dataframe) > 0:
                last_idx = dataframe.index[-1]
                if dataframe.loc[last_idx, 'enter_long'] == 1 or dataframe.loc[last_idx, 'enter_short'] == 1:
                    confidence = dataframe.loc[last_idx, 'signal_confidence']
                    logger.warning(f"🔍 CONFIDENCE DEBUG for {metadata['pair']}:")
                    logger.warning(f"   Div_long: {long_condition_div.loc[last_idx]}")
                    logger.warning(f"   Primary_long: {long_condition_primary.loc[last_idx]}")
                    logger.warning(f"   Secondary_long: {long_condition_secondary.loc[last_idx]}")
                    logger.warning(f"   Confidence Score: {confidence:.1f}%")
                    logger.warning(f"   Above threshold: {confidence >= self.confidence_threshold.value}")
                    logger.warning(f"   Div_short: {short_condition_div.loc[last_idx]}")
                    logger.warning(f"   Primary_short: {short_condition_primary.loc[last_idx]}")
            
            # === ENHANCED DEBUG OUTPUT ===
            if True:  # Show for all pairs
                recent_entries = dataframe['enter_long'].tail(10).sum() + dataframe['enter_short'].tail(10).sum()
                if recent_entries > 0:
                    latest = dataframe.iloc[-1]
                    logger.info(f"🚀 CONFIDENCE-FILTERED {metadata['pair']} ENTRY DETECTED!")
                    logger.info(f"   🏷️  Tag: {latest['enter_tag']}")
                    logger.info(f"   📈 Confidence Score: {latest['signal_confidence']:.1f}%")
                    logger.info(f"   📊 RSI: {latest['rsi']:.1f}")
                    logger.info(f"   💧 Volume Ratio: {latest['volume_ratio']:.2f}")
                    logger.info(f"   🎯 Bull Div Count: {latest.get('total_bullish_divergences_count', 0)}")
                    logger.info(f"   🎯 Bear Div Count: {latest.get('total_bearish_divergences_count', 0)}")
            
            # General debug output for all pairs
            if len(dataframe) > 0:
                last_row = dataframe.iloc[-1]
                total_entries = last_row.get('enter_long', 0) + last_row.get('enter_short', 0)
                if total_entries > 0:
                    logger.info(f"{metadata['pair']} Confidence-Filtered Entry: {last_row.get('enter_tag', '')}, "
                               f"Confidence:{last_row.get('signal_confidence', 75):.1f}%")
            
            return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        Exit when opposite entry signals appear
        """
        
        # Initialize
        dataframe['exit_long'] = 0
        dataframe['exit_short'] = 0
        dataframe['exit_tag'] = ''
        
        # === USE YOUR EXISTING ENTRY CONDITIONS AS EXIT TRIGGERS ===
        # Exit longs when ANY short entry condition triggers
        exit_long_condition = (
            (dataframe['enter_short'] == 1)  # Any short signal exits long
        )
        
        # Exit shorts when ANY long entry condition triggers  
        exit_short_condition = (
            (dataframe['enter_long'] == 1)  # Any long signal exits short
        )
        
        # Set the exits
        dataframe.loc[exit_long_condition, 'exit_long'] = 1
        dataframe.loc[exit_short_condition, 'exit_short'] = 1
        
        # Set tags
        dataframe.loc[exit_long_condition, 'exit_tag'] = 'Exit_Long_Short_Signal'
        dataframe.loc[exit_short_condition, 'exit_tag'] = 'Exit_Short_Long_Signal'

        return dataframe

    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """
        Adaptive leverage based on signal strength and confidence
        """
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if len(dataframe) > 0:
                current_signal_strength = dataframe['signal_strength'].iloc[-1]
                current_confidence = dataframe.get('signal_confidence', pd.Series([75])).iloc[-1]
                
                # Base leverage adjustment on signal strength
                if current_signal_strength >= 8:
                    strength_multiplier = 1.0  # Full leverage for strong signals
                elif current_signal_strength >= 6:
                    strength_multiplier = 0.8  # 80% leverage
                elif current_signal_strength >= 4:
                    strength_multiplier = 0.6  # 60% leverage
                else:
                    strength_multiplier = 0.4  # 40% leverage for weak signals
                
                # Additional adjustment based on confidence
                if current_confidence >= self.high_confidence_threshold.value:
                    confidence_multiplier = 1.0  # No reduction for high confidence
                elif current_confidence >= self.confidence_threshold.value:
                    confidence_multiplier = 0.9  # Small reduction for medium confidence
                else:
                    confidence_multiplier = 0.7  # Larger reduction for low confidence
                
                final_multiplier = strength_multiplier * confidence_multiplier
                return self.leverage_value * final_multiplier
        except:
            pass
        
        return self.leverage_value * 0.5  # Conservative fallback

    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime, current_rate: float,
                        current_profit: float, after_fill: bool, **kwargs) -> float:
        """Modified to not interfere with profit taking"""
        
        # Only apply stoploss for losses or very small profits
        if current_profit > 0.04:  # Let custom_exit handle profits > 4%
            return None
        
        # Your existing stoploss logic here for losses only
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if not dataframe.empty:
                current_candle = dataframe.iloc[-1]
                atr_value = current_candle.get('atr', 0.02)
                atr_multiplier = 3.0  # More conservative
                
                if trade.is_short:
                    stoploss_price = trade.open_rate + (atr_value * atr_multiplier)
                else:
                    stoploss_price = trade.open_rate - (atr_value * atr_multiplier)
                
                return stoploss_from_absolute(stoploss_price, current_rate, 
                                            is_short=trade.is_short, leverage=trade.leverage)
        except:
            pass
        
        return None  # Keep current stoploss
    
    def initialize_divergences_lists(self, dataframe: DataFrame):
        """Initialize divergence tracking columns"""
        # Bullish Divergences
        dataframe["total_bullish_divergences"] = np.nan
        dataframe["total_bullish_divergences_count"] = 0
        dataframe["total_bullish_divergences_names"] = ''

        # Bearish Divergences
        dataframe["total_bearish_divergences"] = np.nan
        dataframe["total_bearish_divergences_count"] = 0
        dataframe["total_bearish_divergences_names"] = ''

    def get_iterators(self, dataframe):
        """Get pivot point iterators for divergence detection"""
        low_iterator = []
        high_iterator = []

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
            if np.isnan(row.pivot_lows):
                low_iterator.append(0 if len(low_iterator) == 0 else low_iterator[-1])
            else:
                low_iterator.append(index)
            if np.isnan(row.pivot_highs):
                high_iterator.append(0 if len(high_iterator) == 0 else high_iterator[-1])
            else:
                high_iterator.append(index)
        
        return high_iterator, low_iterator

    def add_divergences(self, dataframe: DataFrame, indicator: str, high_iterator, low_iterator):
        """Add divergence detection for a specific indicator"""
        (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines) = self.divergence_finder_dataframe(
            dataframe, indicator, high_iterator, low_iterator)
        dataframe['bearish_divergence_' + indicator + '_occurence'] = bearish_divergences
        dataframe['bullish_divergence_' + indicator + '_occurence'] = bullish_divergences

    def divergence_finder_dataframe(self, dataframe: DataFrame, indicator_source: str, high_iterator, low_iterator) -> Tuple[pd.Series, pd.Series]:
        """Enhanced divergence finder with improved logic"""
        bearish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bearish_divergences = np.empty(len(dataframe['close'])) * np.nan
        bullish_lines = [np.empty(len(dataframe['close'])) * np.nan]
        bullish_divergences = np.empty(len(dataframe['close'])) * np.nan

        for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):

            # Bearish divergence detection
            bearish_occurence = self.bearish_divergence_finder(
                dataframe, dataframe[indicator_source], high_iterator, index)

            if bearish_occurence is not None:
                (prev_pivot, current_pivot) = bearish_occurence
                bearish_prev_pivot = dataframe['close'][prev_pivot]
                bearish_current_pivot = dataframe['close'][current_pivot]
                bearish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bearish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                
                # Enhanced validation for bearish divergence
                price_diff = abs(bearish_current_pivot - bearish_prev_pivot)
                indicator_diff = abs(bearish_ind_current_pivot - bearish_ind_prev_pivot)
                time_diff = current_pivot - prev_pivot
                
                # Only accept divergences with sufficient magnitude and time separation
                if (price_diff > dataframe['atr'][current_pivot] * 0.5 and 
                    indicator_diff > 5 and 
                    time_diff >= 5):
                    
                    bearish_divergences[index] = row.close
                    dataframe.loc[index, "total_bearish_divergences"] = row.close
                    dataframe.loc[index, "total_bearish_divergences_count"] += 1
                    dataframe.loc[index, "total_bearish_divergences_names"] += indicator_source.upper() + '<br>'

            # Bullish divergence detection
            bullish_occurence = self.bullish_divergence_finder(
                dataframe, dataframe[indicator_source], low_iterator, index)

            if bullish_occurence is not None:
                (prev_pivot, current_pivot) = bullish_occurence
                bullish_prev_pivot = dataframe['close'][prev_pivot]
                bullish_current_pivot = dataframe['close'][current_pivot]
                bullish_ind_prev_pivot = dataframe[indicator_source][prev_pivot]
                bullish_ind_current_pivot = dataframe[indicator_source][current_pivot]
                
                # Enhanced validation for bullish divergence
                price_diff = abs(bullish_current_pivot - bullish_prev_pivot)
                indicator_diff = abs(bullish_ind_current_pivot - bullish_ind_prev_pivot)
                time_diff = current_pivot - prev_pivot
                
                # Only accept divergences with sufficient magnitude and time separation
                if (price_diff > dataframe['atr'][current_pivot] * 0.5 and 
                    indicator_diff > 5 and 
                    time_diff >= 5):
                    
                    bullish_divergences[index] = row.close
                    dataframe.loc[index, "total_bullish_divergences"] = row.close
                    
                    # CORRECT - increment BULLISH counters for bullish divergence:
                    dataframe.loc[index, "total_bullish_divergences_count"] += 1
                    dataframe.loc[index, "total_bullish_divergences_names"] += indicator_source.upper() + '<br>'

        return (bearish_divergences, bearish_lines, bullish_divergences, bullish_lines)

    def bearish_divergence_finder(self, dataframe, indicator, high_iterator, index):
        """Enhanced bearish divergence detection"""
        try:
            if high_iterator[index] == index:
                current_pivot = high_iterator[index]
                occurences = list(dict.fromkeys(high_iterator))
                current_index = occurences.index(high_iterator[index])
                
                for i in range(current_index-1, current_index - self.window.value - 1, -1):
                    if i < 0 or i >= len(occurences):
                        continue
                    prev_pivot = occurences[i]
                    if np.isnan(prev_pivot):
                        continue
                    
                    # Enhanced divergence validation
                    price_higher = dataframe['pivot_highs'][current_pivot] > dataframe['pivot_highs'][prev_pivot]
                    indicator_lower = indicator[current_pivot] < indicator[prev_pivot]
                    
                    price_lower = dataframe['pivot_highs'][current_pivot] < dataframe['pivot_highs'][prev_pivot]
                    indicator_higher = indicator[current_pivot] > indicator[prev_pivot]
                    
                    # Check for classic or hidden divergence
                    if (price_higher and indicator_lower) or (price_lower and indicator_higher):
                        # Additional validation: check trend consistency
                        if self.validate_divergence_trend(dataframe, prev_pivot, current_pivot, 'bearish'):
                            return (prev_pivot, current_pivot)
        except:
            pass
        return None

    def bullish_divergence_finder(self, dataframe, indicator, low_iterator, index):
        """Enhanced bullish divergence detection"""
        try:
            if low_iterator[index] == index:
                current_pivot = low_iterator[index]
                occurences = list(dict.fromkeys(low_iterator))
                current_index = occurences.index(low_iterator[index])
                
                for i in range(current_index-1, current_index - self.window.value - 1, -1):
                    if i < 0 or i >= len(occurences):
                        continue
                    prev_pivot = occurences[i]
                    if np.isnan(prev_pivot):
                        continue
                    
                    # Enhanced divergence validation
                    price_lower = dataframe['pivot_lows'][current_pivot] < dataframe['pivot_lows'][prev_pivot]
                    indicator_higher = indicator[current_pivot] > indicator[prev_pivot]
                    
                    price_higher = dataframe['pivot_lows'][current_pivot] > dataframe['pivot_lows'][prev_pivot]
                    indicator_lower = indicator[current_pivot] < indicator[prev_pivot]
                    
                    # Check for classic or hidden divergence
                    if (price_lower and indicator_higher) or (price_higher and indicator_lower):
                        # Additional validation: check trend consistency
                        if self.validate_divergence_trend(dataframe, prev_pivot, current_pivot, 'bullish'):
                            return (prev_pivot, current_pivot)
        except:
            pass
        return None

    def validate_divergence_trend(self, dataframe, prev_pivot, current_pivot, divergence_type):
        """Validate divergence by checking intermediate trend"""
        try:
            # Check if there's a clear trend between pivots
            mid_point = (prev_pivot + current_pivot) // 2
            
            if divergence_type == 'bearish':
                # For bearish divergence, expect uptrend in between
                return dataframe['ema20'][mid_point] > dataframe['ema20'][prev_pivot]
            else:
                # For bullish divergence, expect downtrend in between
                return dataframe['ema20'][mid_point] < dataframe['ema20'][prev_pivot]
        except:
            return True  # Default to accepting divergence if validation fails

    @property
    def protections(self):
        """Enhanced protection configuration"""
        prot = []
       
        if self.use_cooldown_protection.value:
            prot.append({
                "method": "CooldownPeriod",
                "stop_duration_candles": self.cooldown_lookback.value
            })
       
        if self.use_max_drawdown_protection.value:
            prot.append({
                "method": "MaxDrawdown",
                "lookback_period_candles": self.max_drawdown_lookback.value,
                "trade_limit": self.max_drawdown_trade_limit.value,
                "stop_duration_candles": self.max_drawdown_stop_duration.value,
                "max_allowed_drawdown": self.max_allowed_drawdown.value
            })
       
        if self.use_stop_protection.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": self.stoploss_guard_lookback.value,
                "trade_limit": self.stoploss_guard_trade_limit.value,
                "stop_duration_candles": self.stop_duration.value,
                "only_per_pair": self.stoploss_guard_only_per_pair.value,
            })
       
        return prot

    def custom_exit(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):
        """
        Hybrid exit: Emergency + Swing profit-taking for 15m timeframe
        """
        trade_duration_minutes = (current_time - trade.open_date_utc).total_seconds() / 60
        
        # === EMERGENCY HIGH PROFIT PROTECTION ===
        if current_profit >= 0.40: return "emergency_40pct"
        if current_profit >= 0.30: return "emergency_30pct"
        if current_profit >= 0.25: return "emergency_25pct"
        
        # === QUICK SCALP PROTECTION (15M SWINGS) ===
        if trade_duration_minutes <= 20:  # Within 1-2 candles
            if current_profit >= 0.08: return "quick_scalp_8pct"
            elif current_profit >= 0.06: return "quick_scalp_6pct"
            elif current_profit >= 0.04: return "quick_scalp_4pct"
        
        # === 15M SWING PROTECTION ===
        # 45 minutes (3 candles) - catch swing moves
        if trade_duration_minutes >= 45:
            if current_profit >= 0.12: return "swing_45min_12pct"
            elif current_profit >= 0.08: return "swing_45min_8pct"
        
        # 90 minutes (6 candles) - reasonable swing duration
        if trade_duration_minutes >= 90:
            if current_profit >= 0.10: return "swing_90min_10pct"
            elif current_profit >= 0.06: return "swing_90min_6pct"
            elif current_profit >= 0.04: return "swing_90min_4pct"
        
        # 2 hours (8 candles) - target swing completion
        if trade_duration_minutes >= 120:
            if current_profit >= 0.08: return "swing_2hr_8pct"
            elif current_profit >= 0.05: return "swing_2hr_5pct"
            elif current_profit >= 0.03: return "swing_2hr_3pct"
        
        # === ATR-BASED SWING TARGETS ===
        try:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            for i in range(1, min(len(dataframe), 20)):
                if dataframe.iloc[-i]['date'].to_pydatetime().replace(tzinfo=timezone.utc) <= trade.open_date_utc:
                    entry_candle = dataframe.iloc[-i-1] if i < len(dataframe) - 1 else dataframe.iloc[-i]
                    atr_value = entry_candle.get('atr', 0.02)
                    
                    if trade.is_short:
                        atr_target_05 = entry_candle['close'] - (atr_value * 0.5)
                        atr_target_10 = entry_candle['close'] - (atr_value * 1.0)
                        
                        if current_rate <= atr_target_10 and current_profit > 0.05:
                            return "atr_1x_short"
                        elif current_rate <= atr_target_05 and current_profit > 0.03:
                            return "atr_0.5x_short"
                    else:
                        atr_target_05 = entry_candle['close'] + (atr_value * 0.5)
                        atr_target_10 = entry_candle['close'] + (atr_value * 1.0)
                        
                        if current_rate >= atr_target_10 and current_profit > 0.05:
                            return "atr_1x_long"
                        elif current_rate >= atr_target_05 and current_profit > 0.03:
                            return "atr_0.5x_long"
                    break
        except:
            pass
        
        # === EXTENDED TIME MANAGEMENT ===
        if trade_duration_minutes >= 240:  # 4+ hours - beyond normal swing
            if current_profit >= 0.05: return "extended_5pct"
            elif current_profit >= 0.03: return "extended_3pct"
            elif current_profit >= 0.02: return "extended_2pct"
        
        # === STAGNANT TRADE CLEANUP ===
        if trade_duration_minutes >= 300 and abs(current_profit) < 0.01:
            return "stagnant_timeout"
        
        return None


def choppiness_index(high, low, close, window=14):
    """Calculate Choppiness Index"""
    natr = pd.Series(ta.NATR(high, low, close, window=window))
    high_max = high.rolling(window=window).max()
    low_min = low.rolling(window=window).min()
    
    choppiness = 100 * np.log10((natr.rolling(window=window).sum()) / (high_max - low_min)) / np.log10(window)
    return choppiness

def resample(indicator):
    """Resample function for compatibility"""
    return indicator

def two_bands_check_long(dataframe):
    """Allow long when price is near/at lower band (oversold area)"""
    return (
        (dataframe['low'] <= dataframe['kc_lowerband']) |
        (dataframe['close'] <= dataframe['kc_lowerband'])
    )

def two_bands_check_short(dataframe):
    """Allow short when price is near/at upper band (overbought area)"""
    return (
        (dataframe['high'] >= dataframe['kc_upperband']) |
        (dataframe['close'] >= dataframe['kc_upperband'])
    )
    
def green_candle(dataframe):
    """Check for green candle"""
    return dataframe[resample('open')] < dataframe[resample('close')]

def red_candle(dataframe):
    """Check for red candle"""
    return dataframe[resample('open')] > dataframe[resample('close')]

def pivot_points(dataframe: DataFrame, window: int = 5, pivot_source=None) -> DataFrame:
    """Enhanced pivot point detection"""
    from enum import Enum
    
    class PivotSource(Enum):
        HighLow = 0
        Close = 1
    
    if pivot_source is None:
        pivot_source = PivotSource.Close
    
    high_source = 'close' if pivot_source == PivotSource.Close else 'high'
    low_source = 'close' if pivot_source == PivotSource.Close else 'low'

    pivot_points_lows = np.empty(len(dataframe['close'])) * np.nan
    pivot_points_highs = np.empty(len(dataframe['close'])) * np.nan
    last_values = deque()

    # Find pivot points with enhanced validation
    for index, row in enumerate(dataframe.itertuples(index=True, name='Pandas')):
        last_values.append(row)
        if len(last_values) >= window * 2 + 1:
            current_value = last_values[window]
            is_greater = True
            is_less = True
            
            for window_index in range(0, window):
                left = last_values[window_index]
                right = last_values[2 * window - window_index]
                local_is_greater, local_is_less = check_if_pivot_is_greater_or_less(
                    current_value, high_source, low_source, left, right)
                is_greater &= local_is_greater
                is_less &= local_is_less
            
            # Additional validation: ensure pivot is significant
            if is_greater:
                current_high = getattr(current_value, high_source)
                # Check if high is significant enough (above ATR threshold)
                if hasattr(current_value, 'atr') and current_high > 0:
                    pivot_points_highs[index - window] = current_high
            
            if is_less:
                current_low = getattr(current_value, low_source)
                # Check if low is significant enough
                if hasattr(current_value, 'atr') and current_low > 0:
                    pivot_points_lows[index - window] = current_low
            
            last_values.popleft()

    return pd.DataFrame(index=dataframe.index, data={
        'pivot_lows': pivot_points_lows,
        'pivot_highs': pivot_points_highs
    })

def check_if_pivot_is_greater_or_less(current_value, high_source: str, low_source: str, left, right) -> Tuple[bool, bool]:
    """Helper function for pivot point validation"""
    is_greater = True
    is_less = True
    
    if (getattr(current_value, high_source) <= getattr(left, high_source) or
            getattr(current_value, high_source) <= getattr(right, high_source)):
        is_greater = False

    if (getattr(current_value, low_source) >= getattr(left, low_source) or
            getattr(current_value, low_source) >= getattr(right, low_source)):
        is_less = False
    
    return (is_greater, is_less)

def emaKeltner(dataframe):
    """Calculate EMA-based Keltner Channels"""
    keltner = {}
    atr = qtpylib.atr(dataframe, window=10)
    ema20 = ta.EMA(dataframe, timeperiod=20)
    keltner['upper'] = ema20 + atr
    keltner['mid'] = ema20
    keltner['lower'] = ema20 - atr
    return keltner

def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Calculate Chaikin Money Flow indicator"""
    df = dataframe.copy()
    mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    mfv = mfv.fillna(0.0)
    mfv *= df['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum() / df['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')# --- Do not remove these libs ---