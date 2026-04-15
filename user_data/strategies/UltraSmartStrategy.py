import numpy as np
import pandas as pd
from scipy import stats
from datetime import datetime, timezone
from typing import Optional, Dict, List, Tuple, Any
import freqtrade.vendor.qtpylib.indicators as qtpylib
from freqtrade.strategy import IStrategy, merge_informative_pair
# Hyperopt dependencies removed for fixed parameter strategy
from pandas import DataFrame
import talib.abstract as ta
from freqtrade.persistence import Trade
from freqtrade.exchange import timeframe_to_prev_date
import logging

logger = logging.getLogger(__name__)

# Removed the StrategyDecisionLogger class - simplified the logging system

class TradingStyleManager:
    """Trading style manager - automatically switches between stable/sideways/aggressive modes based on market state"""
    
    def __init__(self):
        self.current_style = "stable"  # default stable mode
        self.style_switch_cooldown = 0
        self.min_switch_interval = 0.5  # minimum 30 minutes between switches (improves responsiveness)
        
        # === Stable Modeconfiguration ===
        self.STABLE_CONFIG = {
            'name': 'Stable Mode',
            'leverage_range': (2, 5),  # raise base leverage from 1-3 to 2-5
            'position_range': (0.08, 0.20),  # safe position size 8-20%
            'entry_threshold': 6.5,  # moderately loosen entry requirements
            'exit_threshold': 5.5,   # more sensitive exit signal
            'risk_per_trade': 0.015,  # raise risk from 1% to 1.5%
            'max_trades': 4,         # increase concurrent trades from 3 to 4
            'description': 'balanced and steady, combining stable returns with moderate risk'
        }
        
        # === Sideways Modeconfiguration ===  
        self.SIDEWAYS_CONFIG = {
            'name': 'Sideways Mode',
            'leverage_range': (4, 8),  # raise leverage from 2-5 to 4-8
            'position_range': (0.10, 0.25),  # safe position size 10-25%
            'entry_threshold': 5.0,  # moderately loosen entry requirements
            'exit_threshold': 4.0,   # more sensitive exit signal
            'risk_per_trade': 0.02, # raise risk from 1.5% to 2%
            'max_trades': 5,         # increase concurrent trades from 4 to 5
            'description': 'active range trading with quick entries and exits, medium-high risk and reward'
        }
        
        # === Aggressive Modeconfiguration ===
        self.AGGRESSIVE_CONFIG = {
            'name': 'Aggressive Mode',
            'leverage_range': (5, 10),  # optimize leverage from 3-10 to 5-10 for better capital usage
            'position_range': (0.12, 0.30),  # safe position size 12-30%
            'entry_threshold': 3.5,  # more flexible entry requirements
            'exit_threshold': 2.5,   # extremely sensitive exit signal  
            'risk_per_trade': 0.015,  # reduce risk to 1.5%
            'max_trades': 8,         # increase concurrent trades from 6 to 8
            'description': 'aggressive and ambitious, pursuing high returns with high risk/high reward'
        }
        
        self.style_configs = {
            'stable': self.STABLE_CONFIG,
            'sideways': self.SIDEWAYS_CONFIG,
            'aggressive': self.AGGRESSIVE_CONFIG
        }
        
    def get_current_config(self) -> dict:
        """Get the current style config"""
        return self.style_configs[self.current_style]
    
    def classify_market_regime(self, dataframe: DataFrame) -> str:
        """Identify the current market regime to choose the appropriate trading style"""
        
        if dataframe.empty or len(dataframe) < 50:
            return "stable"  # use stable mode when data is insufficient
            
        try:
            # get recent data for analysis
            recent_data = dataframe.tail(50)
            current_data = dataframe.iloc[-1]
            
            # === market feature calculation ===
            
            # 1. trend strength analysis
            trend_strength = current_data.get('trend_strength', 50)
            adx_value = current_data.get('adx', 20)
            
            # 2. volatility analysis
            volatility_state = current_data.get('volatility_state', 50)
            atr_recent = recent_data['atr_p'].mean() if 'atr_p' in recent_data.columns else 0.02
            
            # 3. price action analysis  
            price_range = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].mean()
            
            # 4. volume behavior analysis
            volume_consistency = recent_data['volume_ratio'].std() if 'volume_ratio' in recent_data.columns else 1
            
            # === market regime decision logic ===
            
            # aggressive-mode condition: strong trend + high volatility + clear direction
            if (trend_strength > 75 and adx_value > 30 and 
                volatility_state > 60 and atr_recent > 0.025):
                return "aggressive"
            
            # sideways-mode condition: weak trend + medium volatility + range-bound movement
            elif (trend_strength < 50 and adx_value < 20 and 
                  volatility_state < 40 and price_range < 0.15):
                return "sideways"
            
            # stable mode: all other or uncertain conditions
            else:
                return "stable"
                
        except Exception as e:
            logger.warning(f"market regime classification failed, using stable mode: {e}")
            return "stable"
    
    def should_switch_style(self, dataframe: DataFrame) -> tuple[bool, str]:
        """Determine whether the trading style should switch"""
        
        # check cooldown period
        if self.style_switch_cooldown > 0:
            self.style_switch_cooldown -= 1
            return False, self.current_style
        
        # analyze current market state
        suggested_regime = self.classify_market_regime(dataframe)
        
        # do not switch if the suggested state matches the current one
        if suggested_regime == self.current_style:
            return False, self.current_style
        
        # switch needed, set cooldown
        return True, suggested_regime
    
    def switch_style(self, new_style: str, reason: str = "") -> bool:
        """Switch trading style"""
        
        if new_style not in self.style_configs:
            logger.error(f"unknown trading style: {new_style}")
            return False
        
        old_style = self.current_style
        self.current_style = new_style
        self.style_switch_cooldown = self.min_switch_interval
        
        logger.info(f"🔄 Trading style switch: {old_style} → {new_style} | Reason: {reason}")
        
        return True
    
    def get_dynamic_leverage_range(self) -> tuple[int, int]:
        """Get the leverage range for the current style"""
        config = self.get_current_config()
        return config['leverage_range']
    
    def get_dynamic_position_range(self) -> tuple[float, float]:
        """Get the position range for the current style"""
        config = self.get_current_config()
        return config['position_range']
    
    # Removed get_dynamic_stoploss_range - simplified stoploss logic
    
    def get_risk_per_trade(self) -> float:
        """Get per-trade risk for the current style"""
        config = self.get_current_config()
        return config['risk_per_trade']
    
    def get_signal_threshold(self, signal_type: str = 'entry') -> float:
        """Get signal threshold for the current style"""
        config = self.get_current_config()
        return config.get(f'{signal_type}_threshold', 5.0)
    
    def get_max_concurrent_trades(self) -> int:
        """Get max concurrent trades for the current style"""
        config = self.get_current_config()
        return config['max_trades']
    
    def get_style_summary(self) -> dict:
        """Get a full summary of the current style"""
        config = self.get_current_config()
        
        return {
            'current_style': self.current_style,
            'style_name': config['name'],
            'description': config['description'],
            'leverage_range': config['leverage_range'],
            'position_range': [f"{p*100:.0f}%" for p in config['position_range']], 
            'risk_per_trade': f"{config['risk_per_trade']*100:.1f}%",
            'max_trades': config['max_trades'],
            'switch_cooldown': self.style_switch_cooldown
        }

class UltraSmartStrategy(IStrategy):
 

    INTERFACE_VERSION = 3
    
    # Strategy core parameters
    timeframe = '15m'  # 15 minutes - balances noise filtering and reaction speed
    can_short: bool = True
    
    # Removed informative timeframes to eliminate data sync issues and noise
    
    # Enhanced indicator calculation: supports all advanced technical analysis features
    startup_candle_count: int = 150  # Reduced from 350 for efficiency
    
    # Smart trading mode: optimized configuration after precise entries
    position_adjustment_enable = True
    max_dca_orders = 4  # reduce DCA dependence after precise entries to improve capital efficiency
    
    # === Scientific fixed-parameter configuration ===
    # Remove HYPEROPT dependency and use fixed parameters based on market behavior
    
    # Price-position filter (scientific asymmetric design)
    price_percentile_long_max = 0.50    # long: below the 50th percentile (more opportunities)
    price_percentile_long_best = 0.35   # best long zone: below the 35th percentile
    price_percentile_short_min = 0.65   # short: above the 65th percentile (moderately strict)
    price_percentile_short_best = 0.75  # best short zone: above the 75th percentile
    
    # RSI parameters (looser range to generate more trade opportunities)
    rsi_long_min = 15        # Long RSI lower bound (relaxed oversold requirement)
    rsi_long_max = 55        # Long RSI upper bound (allow more opportunities)
    rsi_short_min = 45       # Short RSI lower bound (relaxed overbought requirement)  
    rsi_short_max = 85       # Short RSI upper bound (keep it high)
    
    # Volume confirmation parameters
    volume_long_threshold = 1.2     # long volume requirement (moderate is enough)
    volume_short_threshold = 1.5    # short volume requirement (clear volume expansion)
    volume_spike_threshold = 2.0    # abnormal volume-spike threshold
    
    # Trend-strength requirements (relaxed)
    adx_long_min = 15        # long ADX requirement (more relaxed)
    adx_short_min = 15       # short ADX requirement (more relaxed)
    trend_strength_threshold = 30    # strong-trend threshold (lowered)
    
    # Technical indicator parameters (fixed classic values)
    macd_fast = 12           # MACD fast line
    macd_slow = 26           # MACD slow line  
    macd_signal = 9          # MACD signal line
    bb_period = 20           # Bollinger Band period
    bb_std = 2.0             # Bollinger Band standard deviation
    
    # Simplified risk management - use a fixed stoploss
    # Removed complex dynamic stoploss logic and use a simple reliable fixed value
    
    # === Optimized ROI settings - widen take-profit targets to capture more profit ===
    # Perpetual/futures trading is volatile, so widen the ROI range to capture larger moves
    minimal_roi = {
        #"0": 0.25,      # 25% capture large volatility with immediate take profit
        #"20": 0.15,     # 20take 15% profit after 20 minutes
        "40": 0.10,     # 40take 10% profit after 40 minutes
        "60": 0.06,     # take 6% profit after 1 hour
        "120": 0.03,    # take 3% profit after 2 hours
        "240": 0.02,    # take 2% profit after 4 hours
        "720": 0.01,    # take 1% profit after 12 hours
        "1440": 0.005   # protect breakeven at 0.5% after 24 hours
    }
    
    # Fully disable stoploss (set a very large value so it never triggers)
    stoploss = -0.99

    # Trailing-stop configuration (larger trailing-stop values)
    trailing_stop = True  # enable trailing stop
    trailing_stop_positive = 0.03  # start trailing stop after 5% profit
    trailing_stop_positive_offset = 0.13  # only start trailing stop after 13% profit
    trailing_only_offset_is_reached = True  # only start trailing after the offset is reached
    
    # Enable smart exit signals
    use_exit_signal = True
    exit_profit_only = True  # allow exit signals to trigger even while losing
    exit_profit_offset = 0.0  # do not set a profit offset
    ignore_roi_if_entry_signal = False  # do not ignore ROI

    # Order type configuration
    order_types = {
        'entry': 'limit',
        'exit': 'limit',
        'stoploss': 'market',
        'stoploss_on_exchange': True,
        'stoploss_on_exchange_interval': 15,
        'stoploss_on_exchange_market_ratio': 0.99
    }
    
    # Chart configuration - ensure all key indicators are visible in FreqUI
    plot_config = {
        'main_plot': {
            'ema_5': {'color': 'yellow', 'type': 'line'},
            'ema_13': {'color': 'orange', 'type': 'line'},
            'ema_34': {'color': 'red', 'type': 'line'},
            'bb_lower': {'color': 'lightblue', 'type': 'line'},
            'bb_middle': {'color': 'gray', 'type': 'line'},
            'bb_upper': {'color': 'lightblue', 'type': 'line'},
            'supertrend': {'color': 'green', 'type': 'line'},
            'vwap': {'color': 'purple', 'type': 'line'}
        },
        'subplots': {
            "RSI": {
                'rsi_14': {'color': 'purple', 'type': 'line'}
            },
            "MACD": {
                'macd': {'color': 'blue', 'type': 'line'},
                'macd_signal': {'color': 'red', 'type': 'line'},
                'macd_hist': {'color': 'gray', 'type': 'bar'}
            },
            "ADX": {
                'adx': {'color': 'orange', 'type': 'line'}
            },
            "Volume": {
                'volume_ratio': {'color': 'cyan', 'type': 'line'}
            },
            "Trend": {
                'trend_strength': {'color': 'magenta', 'type': 'line'},
                'momentum_score': {'color': 'lime', 'type': 'line'}
            }
        }
    }
    
    # Order fill timeout
    order_time_in_force = {
        'entry': 'gtc',
        'exit': 'gtc'
    }
    
    # === Dynamic strategy core parameters (auto-adjusted by trading style) ===
    # Note: the parameters below are overridden by dynamic properties after initialization
    _base_leverage_multiplier = 2  # default base leverage
    _base_max_leverage = 10        # default max leverage (user requested 10x)
    _base_position_size = 0.08     # default base position size
    _base_max_position_size = 0.25 # default max position size
    
    # === Technical indicator parameters (fixed classic values) ===
    @property
    def rsi_period(self):
        return 14  # keep the RSI period fixed
        
    atr_period = 14
    adx_period = 14
    
    # === Simplified market-state parameters ===
    volatility_threshold = 0.025     # slightly raise the volatility threshold
    trend_strength_min = 50          # raise trend-strength requirements
    volume_spike_threshold = 1.5     # lower the volume-spike threshold
    
    # === Optimized DCA parameters ===
    dca_multiplier = 1.3        # reduce the DCA multiplier
    dca_price_deviation = 0.025  # reduce trigger deviation (2.5%)
    
    # === Strict risk-management parameters ===
    max_risk_per_trade = 0.015  # reduce per-trade risk to 1.5%
    kelly_lookback = 50         # shorten the lookback period to improve responsiveness
    drawdown_protection = 0.12  # lower the drawdown-protection threshold
    
    # Advanced capital-management parameters
    var_confidence_level = 0.95    # VaR confidence level
    cvar_confidence_level = 0.99   # CVaR confidence level
    max_portfolio_heat = 0.3       # maximum portfolio heat
    correlation_threshold = 0.7    # correlation threshold
    rebalance_threshold = 0.1      # rebalance threshold
    portfolio_optimization_method = 'kelly'  # 'kelly', 'markowitz', 'risk_parity'
    
    def bot_start(self, **kwargs) -> None:
        """Strategy initialization"""
        self.custom_info = {}
        self.trade_count = 0
        self.total_profit = 0
        self.consecutive_losses = 0
        self.consecutive_wins = 0
        self.max_consecutive_losses = 3
        self.initial_balance = None
        self.peak_balance = None
        self.current_drawdown = 0
        self.trade_history = []
        self.leverage_adjustment_factor = 1.0
        self.profit_taking_tracker = {}  # track staged take-profit status for each trade
        
        # DCA performance tracking system
        self.dca_performance_tracker = {
            'total_dca_count': 0,
            'successful_dca_count': 0,
            'dca_success_rate': 0.0,
            'dca_type_performance': {},  # success rate of each DCA type
            'avg_dca_profit': 0.0,
            'dca_history': []
        }
        
        # advanced capital-management data structures
        self.portfolio_returns = []       # portfolio return history
        self.pair_returns_history = {}    # pair return history
        self.position_correlation_matrix = {}  # position correlation matrix
        self.risk_metrics_history = []    # risk metric history
        self.allocation_history = []      # allocation history
        self.var_cache = {}              # VaR calculation cache
        self.optimal_f_cache = {}        # optimal-f cache
        self.last_rebalance_time = None  # last rebalance time
        self.kelly_coefficients = {}     # Kelly coefficient cache
        
        # initialize account balance
        try:
            if hasattr(self, 'wallets') and self.wallets:
                self.initial_balance = self.wallets.get_total_stake_amount()
                self.peak_balance = self.initial_balance
        except Exception:
            pass
            
        # === performance optimization initialization ===
        self.initialize_performance_optimization()
        
        # === logging system initialization ===
        # Removed StrategyDecisionLogger - use the standard logger
        logger.info("🔥 Strategy started - UltraSmartStrategy v2")
        
        # === trading style management system initialization ===
        self.style_manager = TradingStyleManager()
        logger.info(f"🎯 Trading style management system started - current mode: {self.style_manager.current_style}")
        
        # initialize style-switch tracking
        self.last_style_check = datetime.now(timezone.utc)
        self.style_check_interval = 300  # check for style switches every 5 minutes
        
    def initialize_performance_optimization(self):
        """Initialize the performance optimization system"""
        
        # cache system
        self.indicator_cache = {}  
        self.signal_cache = {}     
        self.market_state_cache = {}  
        self.cache_ttl = 300  # 5-minute cache
        self.last_cache_cleanup = datetime.now(timezone.utc)
        
        # performance statistics
        self.calculation_stats = {
            'indicator_calls': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'avg_calculation_time': 0
        }
        
        # precompute common thresholds
        self.precomputed_thresholds = {
            'rsi_oversold': 35,
            'rsi_overbought': 65, 
            'adx_strong': 25,
            'volume_spike': 1.2,
            'atr_high_vol': 0.03,
            'atr_low_vol': 0.015
        }
        
        # batch calculation optimization
        self.batch_size = 50
        self.optimize_calculations = True
    
    def get_cached_indicators(self, pair: str, dataframe_len: int) -> Optional[DataFrame]:
        """Get cached indicator data"""
        cache_key = f"{pair}_{dataframe_len}"
        
        if cache_key in self.indicator_cache:
            cache_data = self.indicator_cache[cache_key]
            # check whether the cache has expired
            if (datetime.now(timezone.utc) - cache_data['timestamp']).seconds < self.cache_ttl:
                self.calculation_stats['cache_hits'] += 1
                return cache_data['indicators']
        
        self.calculation_stats['cache_misses'] += 1
        return None
    
    def cache_indicators(self, pair: str, dataframe_len: int, indicators: DataFrame):
        """Cache indicator data"""
        cache_key = f"{pair}_{dataframe_len}"
        self.indicator_cache[cache_key] = {
            'indicators': indicators.copy(),
            'timestamp': datetime.now(timezone.utc)
        }
        
        # periodically clean expired cache entries
        if (datetime.now(timezone.utc) - self.last_cache_cleanup).seconds > self.cache_ttl * 2:
            self.cleanup_expired_cache()
    
    def cleanup_expired_cache(self):
        """Clean expired cache entries"""
        current_time = datetime.now(timezone.utc)
        expired_keys = []
        
        for key, data in self.indicator_cache.items():
            if (current_time - data['timestamp']).seconds > self.cache_ttl:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.indicator_cache[key]
        
        # clean other caches as well
        for cache_dict in [self.signal_cache, self.market_state_cache]:
            expired_keys = []
            for key, data in cache_dict.items():
                if (current_time - data.get('timestamp', current_time)).seconds > self.cache_ttl:
                    expired_keys.append(key)
            for key in expired_keys:
                del cache_dict[key]
        
        self.last_cache_cleanup = current_time
    
    # ===== Dynamic trading style system =====
    
    @property  
    def leverage_multiplier(self) -> int:
        """Dynamic leverage multiplier - based on current trading style"""
        leverage_range = self.style_manager.get_dynamic_leverage_range()
        return leverage_range[0]  # use the lower bound of the range as the base multiplier
    
    @property
    def max_leverage(self) -> int:
        """Dynamic max leverage - based on current trading style"""
        leverage_range = self.style_manager.get_dynamic_leverage_range()
        return leverage_range[1]  # use the upper bound of the range as the max multiplier
    
    @property
    def base_position_size(self) -> float:
        """Dynamic base position size - based on current trading style"""
        position_range = self.style_manager.get_dynamic_position_range()
        return position_range[0]  # use the lower bound of the range as the base position size
    
    @property  
    def max_position_size(self) -> float:
        """Dynamic max position size - based on current trading style"""
        position_range = self.style_manager.get_dynamic_position_range()
        return position_range[1]  # use the upper bound of the range as the max position size
    
    @property
    def max_risk_per_trade(self) -> float:
        """Dynamic max risk per trade - based on current trading style"""
        return self.style_manager.get_risk_per_trade()
    
    # Removed dynamic_stoploss - simplified stoploss logic
    
    def check_and_switch_trading_style(self, dataframe: DataFrame) -> None:
        """check andSwitch trading style"""
        
        current_time = datetime.now(timezone.utc)
        
        # check whether it is time to evaluate style switching
        if (current_time - self.last_style_check).seconds < self.style_check_interval:
            return
            
        self.last_style_check = current_time
        
        # check whether a style switch is needed
        should_switch, new_style = self.style_manager.should_switch_style(dataframe)
        
        if should_switch:
            old_config = self.style_manager.get_current_config()
            
            # execute style switch
            market_regime = self.style_manager.classify_market_regime(dataframe)
            reason = f"market state changed: {market_regime}"
            
            if self.style_manager.switch_style(new_style, reason):
                new_config = self.style_manager.get_current_config()
                
                # record style-switch log
                self._log_style_switch(old_config, new_config, reason, dataframe)
    
    def _log_style_switch(self, old_config: dict, new_config: dict, 
                         reason: str, dataframe: DataFrame) -> None:
        """record style-switch details"""
        
        try:
            current_data = dataframe.iloc[-1] if not dataframe.empty else {}
            
            switch_log = f"""
==================== Trading style switch ====================
time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
translatedReason: {reason}

📊 market state analysis:
├─ trend strength: {current_data.get('trend_strength', 0):.0f}/100
├─ ADXvalue: {current_data.get('adx', 0):.1f}  
├─ volatility state: {current_data.get('volatility_state', 0):.0f}/100
├─ ATRvolatility: {(current_data.get('atr_p', 0) * 100):.2f}%

🔄 style change details:
├─ old style: {old_config['name']} → new style: {new_config['name']}
├─ leverage adjustment: {old_config['leverage_range']} → {new_config['leverage_range']}
├─ position adjustment: {[f"{p*100:.0f}%" for p in old_config['position_range']]} → {[f"{p*100:.0f}%" for p in new_config['position_range']]}
├─ risk adjustment: {old_config['risk_per_trade']*100:.1f}% → {new_config['risk_per_trade']*100:.1f}%

🎯 new style features:
├─ description: {new_config['description']}
├─ entry threshold: {new_config['entry_threshold']:.1f}
├─ max concurrent: {new_config['max_trades']}count
├─ cooldown period: {self.style_manager.style_switch_cooldown}hours

=================================================="""
            
            logger.info(switch_log)
            
            # record
            style_summary = self.style_manager.get_style_summary()
            logger.info(f"🔄 style switch completed: {style_summary}")
            
        except Exception as e:
            logger.error(f"failed to record style-switch log: {e}")
    
    def get_current_trading_style_info(self) -> dict:
        """get detailed information for the current trading style"""
        return self.style_manager.get_style_summary()
        
    # Removed informative_pairs() method - no longer needed without informative timeframes
    
    def get_market_orderbook(self, pair: str) -> Dict:
        """get orderbook data"""
        try:
            orderbook = self.dp.orderbook(pair, 10)  # get10translated
            if orderbook:
                bids = np.array([[float(bid[0]), float(bid[1])] for bid in orderbook['bids']])
                asks = np.array([[float(ask[0]), float(ask[1])] for ask in orderbook['asks']])
                
                # calculate orderbook metrics
                bid_volume = np.sum(bids[:, 1]) if len(bids) > 0 else 0
                ask_volume = np.sum(asks[:, 1]) if len(asks) > 0 else 0
                
                volume_ratio = bid_volume / (ask_volume + 1e-10)
                
                # calculate spread
                spread = ((asks[0][0] - bids[0][0]) / bids[0][0] * 100) if len(asks) > 0 and len(bids) > 0 else 0
                
                # calculate depth imbalance
                imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume + 1e-10)
                
                # calculate buy/sell pressure metrics (0-1range)
                buy_pressure = bid_volume / (bid_volume + ask_volume + 1e-10)
                sell_pressure = ask_volume / (bid_volume + ask_volume + 1e-10)
                
                # calculate market quality (0-1range)
                total_volume = bid_volume + ask_volume
                spread_quality = max(0, 1 - spread / 1.0)  # smaller spread means higher quality
                volume_quality = min(1, total_volume / 10000)  # higher volume means higher quality
                balance_quality = 1 - abs(imbalance)  # better balance means higher quality
                market_quality = (spread_quality + volume_quality + balance_quality) / 3
                
                return {
                    'volume_ratio': volume_ratio,
                    'spread_pct': spread,
                    'depth_imbalance': imbalance,
                    'market_quality': market_quality,
                    'bid_volume': bid_volume,
                    'ask_volume': ask_volume,
                    'buy_pressure': buy_pressure,
                    'sell_pressure': sell_pressure,
                    'liquidity_score': market_quality  # usemarket_qualityasliquidity_score
                }
        except Exception as e:
            logger.warning(f"failed to get orderbook: {e}")
            
        return {
            'volume_ratio': 1.0,
            'spread_pct': 0.1,
            'depth_imbalance': 0.0,
            'market_quality': 0.5,
            'bid_volume': 0,
            'ask_volume': 0,
            'buy_pressure': 0.5,
            'sell_pressure': 0.5,
            'liquidity_score': 0.5
        }
    
    def calculate_technical_indicators(self, dataframe: DataFrame) -> DataFrame:
        """optimized technical-indicator calculation - avoidDataFrametranslated"""
        
        # use a dictionary to batch-store all new columns
        new_columns = {}
        
        # === optimized sensitive moving-average system - translated，fast ===
        new_columns['ema_5'] = ta.EMA(dataframe, timeperiod=5)    # ultra-short-term：quickly capture changes
        new_columns['ema_8'] = ta.EMA(dataframe, timeperiod=8)    # ultra-short-term enhancement
        new_columns['ema_13'] = ta.EMA(dataframe, timeperiod=13)  # short-term：trend confirmation
        new_columns['ema_21'] = ta.EMA(dataframe, timeperiod=21)  # medium-short-term transition
        new_columns['ema_34'] = ta.EMA(dataframe, timeperiod=34)  # mid stage：main-trend filter
        new_columns['ema_50'] = ta.EMA(dataframe, timeperiod=50)  # long-term trend
        new_columns['sma_20'] = ta.SMA(dataframe, timeperiod=20)  # translatedSMA20as
        
        # === Bollinger Bands (translated，high indicator) ===
        bb = qtpylib.bollinger_bands(dataframe['close'], window=self.bb_period, stds=self.bb_std)
        new_columns['bb_lower'] = bb['lower']
        new_columns['bb_middle'] = bb['mid']
        new_columns['bb_upper'] = bb['upper']
        new_columns['bb_width'] = np.where(bb['mid'] > 0, 
                                        (bb['upper'] - bb['lower']) / bb['mid'], 
                                        0)
        new_columns['bb_position'] = (dataframe['close'] - bb['lower']) / (bb['upper'] - bb['lower'])
        
        # === RSI (most has14translated) ===
        new_columns['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        
        # === MACD (translated，classic trend indicator) ===
        macd = ta.MACD(dataframe, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal)
        new_columns['macd'] = macd['macd']
        new_columns['macd_signal'] = macd['macdsignal'] 
        new_columns['macd_hist'] = macd['macdhist']
        
        # === ADX trend strength (translated，important trend indicator) ===
        new_columns['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period)
        new_columns['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=self.adx_period)
        new_columns['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=self.adx_period)
        
        # === ATR volatility (translated，required for risk management) ===
        new_columns['atr'] = ta.ATR(dataframe, timeperiod=self.atr_period)
        new_columns['atr_p'] = new_columns['atr'] / dataframe['close']
        
        # === volume indicators (simplified) ===
        new_columns['volume_sma'] = ta.SMA(dataframe['volume'], timeperiod=20)
        new_columns['volume_ratio'] = np.where(new_columns['volume_sma'] > 0, 
                                            dataframe['volume'] / new_columns['volume_sma'], 
                                            1.0)
        
        # === momentum indicators ===
        new_columns['mom_10'] = ta.MOM(dataframe, timeperiod=10)
        new_columns['roc_10'] = ta.ROC(dataframe, timeperiod=10)
        
        # === new leading-indicator set - solve lagging issues ===
        
        # 1. fastRSI - translatedRSItranslated
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=3, fastd_period=3)
        new_columns['stoch_rsi_k'] = stoch_rsi['fastk']
        new_columns['stoch_rsi_d'] = stoch_rsi['fastd']
        
        # 2. Williams indicator - fast reversal signal
        new_columns['williams_r'] = ta.WILLR(dataframe, timeperiod=14)
        
        # 3. CCItranslated - sensitive overbought/oversold indicator  
        new_columns['cci'] = ta.CCI(dataframe, timeperiod=20)
        
        # 4. price action analysis - translatedKtranslated
        new_columns['candle_body'] = abs(dataframe['close'] - dataframe['open'])
        new_columns['candle_upper_shadow'] = dataframe['high'] - np.maximum(dataframe['close'], dataframe['open'])
        new_columns['candle_lower_shadow'] = np.minimum(dataframe['close'], dataframe['open']) - dataframe['low']
        new_columns['candle_total_range'] = dataframe['high'] - dataframe['low']
        
        # 6. abnormal-volume detection - lead price changes
        new_columns['volume_spike'] = (dataframe['volume'] > new_columns['volume_sma'] * 2).astype(int)
        new_columns['volume_dry'] = (dataframe['volume'] < new_columns['volume_sma'] * 0.5).astype(int)
        
        # 8. support/resistance breakout strength
        new_columns['resistance_strength'] = (
            dataframe['close'] / dataframe['high'].rolling(20).max() - 1
        ) * 100  # translated20day most high
        
        new_columns['support_strength'] = (
            1 - dataframe['close'] / dataframe['low'].rolling(20).min()
        ) * 100  # translated20day most low
        
        # === VWAP (important institutional trading reference) ===
        new_columns['vwap'] = qtpylib.rolling_vwap(dataframe)
        
        # === supertrend (efficient trend following) ===
        new_columns['supertrend'] = self.supertrend(dataframe, 10, 3)
        
        # 1 times will has new todataframe，useconcatavoid
        if new_columns:
            new_df = pd.DataFrame(new_columns, index=dataframe.index)
            dataframe = pd.concat([dataframe, new_df], axis=1)
        
        # === optimized composite indicators (replace many single-purpose indicators) ===
        dataframe = self.calculate_optimized_composite_indicators(dataframe)
        
        # === advanced momentum indicators ===
        dataframe = self.calculate_advanced_momentum_indicators(dataframe)
        
        # === volume indicators ===
        dataframe = self.calculate_advanced_volume_indicators(dataframe)
        
        # === Ichimokucloud indicators ===
        dataframe = self.ichimoku(dataframe)
        
        # === market-structure indicators (including price-action patterns) ===
        dataframe = self.calculate_market_structure_indicators(dataframe)
        
        # === market-state indicators (simplified version) ===
        dataframe = self.calculate_market_regime_simple(dataframe)
        
        # === indicator validation and calibration ===
        dataframe = self.validate_and_calibrate_indicators(dataframe)
        
        # === final indicator-integrity check ===
        required_indicators = ['rsi_14', 'adx', 'atr_p', 'macd', 'macd_signal', 'volume_ratio', 'trend_strength', 'momentum_score', 
                              'ema_5', 'ema_8', 'ema_13', 'ema_21', 'ema_34', 'ema_50', 'mom_10', 'roc_10']
        missing_indicators = [indicator for indicator in required_indicators if indicator not in dataframe.columns or dataframe[indicator].isnull().all()]
        
        if missing_indicators:
            logger.error(f"critical indicator calculation failed: {missing_indicators}")
            # as indicator default value，use update avoid
            default_values = {}
            for indicator in missing_indicators:
                if indicator == 'rsi_14':
                    default_values[indicator] = 50.0
                elif indicator == 'adx':
                    default_values[indicator] = 25.0
                elif indicator == 'atr_p':
                    default_values[indicator] = 0.02
                elif indicator in ['macd', 'macd_signal']:
                    default_values[indicator] = 0.0
                elif indicator == 'volume_ratio':
                    default_values[indicator] = 1.0
                elif indicator == 'trend_strength':
                    default_values[indicator] = 50.0
                elif indicator == 'momentum_score':
                    default_values[indicator] = 0.0
                elif indicator in ['ema_5', 'ema_13', 'ema_34']:
                    # translatedEMAindicator，recalculate
                    if indicator == 'ema_5':
                        default_values[indicator] = ta.EMA(dataframe, timeperiod=5)
                    elif indicator == 'ema_13':
                        default_values[indicator] = ta.EMA(dataframe, timeperiod=13)
                    elif indicator == 'ema_34':
                        default_values[indicator] = ta.EMA(dataframe, timeperiod=34)
            
            # add all default values at once
            if default_values:
                defaults_df = pd.DataFrame(default_values, index=dataframe.index)
                dataframe = pd.concat([dataframe, defaults_df], axis=1)
        else:
            logger.info("✅ all indicators calculated successfully")
        
        # === translatedEMAindicator ===
        # checkEMAindicator hasNaNvalue
        for ema_col in ['ema_8', 'ema_21', 'ema_50']:
            if ema_col in dataframe.columns:
                nan_count = dataframe[ema_col].isnull().sum()
                total_count = len(dataframe)
                if nan_count > total_count * 0.1:  # translated10%value asNaN
                    logger.warning(f"{ema_col} has too many missing values ({nan_count}/{total_count}), recalculate")
                    if ema_col == 'ema_8':
                        dataframe[ema_col] = ta.EMA(dataframe, timeperiod=8)
                    elif ema_col == 'ema_21':
                        dataframe[ema_col] = ta.EMA(dataframe, timeperiod=21)
                    elif ema_col == 'ema_50':
                        dataframe[ema_col] = ta.EMA(dataframe, timeperiod=50)
        
        return dataframe
    
    def calculate_optimized_composite_indicators(self, dataframe: DataFrame) -> DataFrame:
        """optimized composite indicators - avoidDataFrametranslated"""
        
        # use a dictionary to batch-store all new columns
        new_columns = {}
        
        # === revolutionary trend-strength scoring system - momentum，before2-3translatedKtranslated ===
        
        # 1. price-momentum slope analysis（early warning） - useEMA(5,13,34)
        ema5_slope = np.where(dataframe['ema_5'].shift(2) > 0,
                             (dataframe['ema_5'] - dataframe['ema_5'].shift(2)) / dataframe['ema_5'].shift(2),
                             0) * 100  # short，fast
        ema13_slope = np.where(dataframe['ema_13'].shift(3) > 0,
                              (dataframe['ema_13'] - dataframe['ema_13'].shift(3)) / dataframe['ema_13'].shift(3),
                              0) * 100
        
        # 2. moving-average divergence analysis（trend-acceleration signal）
        ema_spread = np.where(dataframe['ema_34'] > 0,
                             (dataframe['ema_5'] - dataframe['ema_34']) / dataframe['ema_34'] * 100,
                             0)
        ema_spread_series = self._safe_series(ema_spread, len(dataframe))
        ema_spread_change = ema_spread - ema_spread_series.shift(3)  # divergence change
        
        # 3. ADXtranslated（trend-strengthening signal）
        adx_slope = dataframe['adx'] - dataframe['adx'].shift(3)  # ADXtranslated
        adx_acceleration = adx_slope - adx_slope.shift(2)  # ADXtranslated
        
        # 4. volume trend confirmation
        volume_20_mean = dataframe['volume'].rolling(20).mean()
        volume_trend = np.where(volume_20_mean != 0,
                               dataframe['volume'].rolling(5).mean() / volume_20_mean,
                               1.0)  # translated20day as0，translated1.0（mid）
        volume_trend_series = self._safe_series(volume_trend, len(dataframe))
        volume_momentum = volume_trend_series - volume_trend_series.shift(2).fillna(0)
        
        # 5. price acceleration（second derivative）
        close_shift_3 = dataframe['close'].shift(3)
        price_velocity = np.where(close_shift_3 != 0,
                                 (dataframe['close'] / close_shift_3 - 1) * 100,
                                 0)  # first derivative
        price_velocity_series = self._safe_series(price_velocity, len(dataframe))
        price_acceleration = price_velocity_series - price_velocity_series.shift(2).fillna(0)
        
        # === composite trend-strength score ===
        trend_score = (
            ema5_slope * 0.30 +        # ultra-short-term momentum（most，high weight）
            ema13_slope * 0.20 +       # short-term momentum confirmation
            ema_spread_change * 0.15 + # trend-divergence change
            adx_slope * 0.15 +         # trend-strength change
            volume_momentum * 0.10 +   # volume support
            price_acceleration * 0.10  # price acceleration
        )
        
        # useADXas trend confirmation
        adx_multiplier = np.where(dataframe['adx'] > 30, 1.5,
                                 np.where(dataframe['adx'] > 20, 1.2,
                                         np.where(dataframe['adx'] > 15, 1.0, 0.7)))
        
        # final trend strength
        new_columns['trend_strength'] = (trend_score * adx_multiplier).clip(-100, 100)
        new_columns['price_acceleration'] = price_acceleration
        
        # === momentum composite indicators ===
        rsi_normalized = (dataframe['rsi_14'] - 50) / 50  # -1 to 1
        macd_normalized = np.where(dataframe['atr_p'] > 0, 
                                 dataframe['macd_hist'] / (dataframe['atr_p'] * dataframe['close']), 
                                 0)  # normalize
        price_momentum = (dataframe['close'] / dataframe['close'].shift(5) - 1) * 10  # 5period price change
        
        new_columns['momentum_score'] = (rsi_normalized + macd_normalized + price_momentum) / 3
        new_columns['price_velocity'] = price_velocity_series
        
        # === volatility-state indicator ===  
        atr_percentile = dataframe['atr_p'].rolling(50).rank(pct=True)
        bb_squeeze = np.where(dataframe['bb_width'] < dataframe['bb_width'].rolling(20).quantile(0.3), 1, 0)
        volume_spike = np.where(dataframe['volume_ratio'] > 1.5, 1, 0)
        
        new_columns['volatility_state'] = atr_percentile * 50 + bb_squeeze * 25 + volume_spike * 25
        
        # === support/resistance strength ===
        bb_position_score = np.abs(dataframe['bb_position'] - 0.5) * 2  # 0-1, the closer to the edge, the higher the score
        vwap_distance = np.where(dataframe['vwap'] > 0, 
                                np.abs((dataframe['close'] - dataframe['vwap']) / dataframe['vwap']) * 100, 
                                0)
        
        new_columns['sr_strength'] = (bb_position_score + np.minimum(vwap_distance, 5)) / 2  # normalize to a reasonable range
        
        # === trend sustainability indicator ===
        adx_sustainability = np.where(dataframe['adx'] > 25, 1, 0)
        volume_sustainability = np.where(dataframe['volume_ratio'] > 0.8, 1, 0)
        volatility_sustainability = np.where(dataframe['atr_p'] < dataframe['atr_p'].rolling(20).quantile(0.8), 1, 0)
        new_columns['trend_sustainability'] = (
            (adx_sustainability * 0.5 + volume_sustainability * 0.3 + volatility_sustainability * 0.2) * 2 - 1
        ).clip(-1, 1)  # normalize to[-1, 1]
        
        # === RSIdivergence strength indicator ===
        price_high_10 = dataframe['high'].rolling(10).max()
        price_low_10 = dataframe['low'].rolling(10).min()
        rsi_high_10 = dataframe['rsi_14'].rolling(10).max()
        rsi_low_10 = dataframe['rsi_14'].rolling(10).min()
        
        # bearish divergence：price new high butRSInew high
        bearish_divergence = np.where(
            (dataframe['high'] >= price_high_10) & (dataframe['rsi_14'] < rsi_high_10),
            -(dataframe['high'] / price_high_10 - dataframe['rsi_14'] / rsi_high_10),
            0
        )
        
        # bullish divergence：price new low butRSInew low
        bullish_divergence = np.where(
            (dataframe['low'] <= price_low_10) & (dataframe['rsi_14'] > rsi_low_10),
            (dataframe['low'] / price_low_10 - dataframe['rsi_14'] / rsi_low_10),
            0
        )
        
        new_columns['rsi_divergence_strength'] = (bearish_divergence + bullish_divergence).clip(-2, 2)
        
        # === new：predictive indicator system ===
        
        # 1. translatedRSIdivergence
        price_higher_5 = dataframe['close'] > dataframe['close'].shift(5)
        rsi_lower_5 = dataframe['rsi_14'] < dataframe['rsi_14'].shift(5)
        new_columns['bearish_divergence'] = (price_higher_5 & rsi_lower_5).astype(int)
        
        price_lower_5 = dataframe['close'] < dataframe['close'].shift(5)
        rsi_higher_5 = dataframe['rsi_14'] > dataframe['rsi_14'].shift(5)
        new_columns['bullish_divergence'] = (price_lower_5 & rsi_higher_5).astype(int)
        
        # 2. volume-exhaustion detection
        volume_decreasing = (
            (dataframe['volume'] < dataframe['volume'].shift(1)) &
            (dataframe['volume'].shift(1) < dataframe['volume'].shift(2)) &
            (dataframe['volume'].shift(2) < dataframe['volume'].shift(3))
        )
        new_columns['volume_exhaustion'] = volume_decreasing.astype(int)
        
        # 3. price-acceleration change（predict turning points）
        price_roc_3 = dataframe['close'].pct_change(3)
        price_acceleration_new = price_roc_3 - price_roc_3.shift(3)
        new_columns['price_acceleration_rate'] = price_acceleration_new
        new_columns['price_decelerating'] = (np.abs(price_acceleration_new) < np.abs(price_acceleration_new.shift(3))).astype(int)
        
        # 4. composite momentum-exhaustion score
        momentum_exhaustion = (
            (new_columns['bearish_divergence'] * 0.3) +
            (volume_decreasing.astype(int) * 0.3) +
            (new_columns['price_decelerating'] * 0.2) +
            ((dataframe['adx'] < dataframe['adx'].shift(3)).astype(int) * 0.2)
        )
        new_columns['momentum_exhaustion_score'] = momentum_exhaustion
        
        # 5. trend phase detection（predictive）
        # early stage：breakout+translated
        trend_early = (
            (dataframe['adx'] > dataframe['adx'].shift(1)) &
            (dataframe['adx'] > 20) &
            (dataframe['volume_ratio'] > 1.2)
        ).astype(int)
        # mid stage：stable trend
        trend_middle = (
            (dataframe['adx'] > 25) &
            (np.abs(price_acceleration_new) < 0.02) &
            (~volume_decreasing)
        ).astype(int)
        # late stage：translated+divergence
        trend_late = (
            (np.abs(price_acceleration_new) > 0.03) |
            (new_columns['bearish_divergence'] == 1) |
            (new_columns['bullish_divergence'] == 1) |
            (momentum_exhaustion > 0.6)
        ).astype(int)
        
        new_columns['trend_phase'] = trend_late * 3 + trend_middle * 2 + trend_early * 1
        
        # === market sentiment indicator ===
        rsi_sentiment = (dataframe['rsi_14'] - 50) / 50  # normalizeRSI
        volatility_sentiment = np.where(dataframe['atr_p'] > 0, 
                                       -(dataframe['atr_p'] / dataframe['atr_p'].rolling(20).mean() - 1), 
                                       0)  # high volatility=translated，low volatility=translated
        volume_sentiment = np.where(dataframe['volume_ratio'] > 1.5, -0.5,  # abnormal volume spike=translated
                                   np.where(dataframe['volume_ratio'] < 0.7, 0.5, 0))  # translated=translated
        new_columns['market_sentiment'] = ((rsi_sentiment + volatility_sentiment + volume_sentiment) / 3).clip(-1, 1)
        
        # === translated4reversal ===
        reversal_warnings = self.detect_reversal_warnings_system(dataframe)
        new_columns['reversal_warning_level'] = reversal_warnings['level']
        new_columns['reversal_probability'] = reversal_warnings['probability']
        new_columns['reversal_signal_strength'] = reversal_warnings['signal_strength']
        
        # 1 times will has new todataframe，useconcatavoid
        if new_columns:
            new_df = pd.DataFrame(new_columns, index=dataframe.index)
            dataframe = pd.concat([dataframe, new_df], axis=1)
        
        # === add a breakout-validity verification system ===
        breakout_validation = self.validate_breakout_effectiveness(dataframe)
        dataframe['breakout_validity_score'] = breakout_validation['validity_score']
        dataframe['breakout_confidence'] = breakout_validation['confidence']
        dataframe['breakout_type'] = breakout_validation['breakout_type']
        
        return dataframe
    
    def detect_reversal_warnings_system(self, dataframe: DataFrame) -> dict:
        """🚨 translated4reversal - before2-5translatedKtrend"""
        
        # === 1level warning：momentum-decay detection ===
        # detect whether trend momentum has started to decay（earliest signal）
        momentum_decay_long = (
            # price gains are shrinking
            (dataframe['close'] - dataframe['close'].shift(3) < 
             dataframe['close'].shift(3) - dataframe['close'].shift(6)) &
            # but price is still rising
            (dataframe['close'] > dataframe['close'].shift(3)) &
            # ADXfall
            (dataframe['adx'] < dataframe['adx'].shift(2)) &
            # volume starts shrinking
            (dataframe['volume_ratio'] < dataframe['volume_ratio'].shift(3))
        )
        
        momentum_decay_short = (
            # price declines are shrinking  
            (dataframe['close'] - dataframe['close'].shift(3) > 
             dataframe['close'].shift(3) - dataframe['close'].shift(6)) &
            # but price is still falling
            (dataframe['close'] < dataframe['close'].shift(3)) &
            # ADXfall
            (dataframe['adx'] < dataframe['adx'].shift(2)) &
            # volume starts shrinking
            (dataframe['volume_ratio'] < dataframe['volume_ratio'].shift(3))
        )
        
        # === Fixed RSI Divergence Detection (increased lookback for reliability) ===
        # Price new high but RSI not making new high (fixed 25-period lookback)
        price_higher_high = (
            (dataframe['high'] > dataframe['high'].shift(25)) &
            (dataframe['high'].shift(25) > dataframe['high'].shift(50))
        )
        rsi_lower_high = (
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(25)) &
            (dataframe['rsi_14'].shift(25) < dataframe['rsi_14'].shift(50))
        )
        bearish_rsi_divergence = price_higher_high & rsi_lower_high & (dataframe['rsi_14'] > 65)
        
        # Price new low but RSI not making new low
        price_lower_low = (
            (dataframe['low'] < dataframe['low'].shift(25)) &
            (dataframe['low'].shift(25) < dataframe['low'].shift(50))
        )
        rsi_higher_low = (
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(25)) &
            (dataframe['rsi_14'].shift(25) > dataframe['rsi_14'].shift(50))
        )
        bullish_rsi_divergence = price_lower_low & rsi_higher_low & (dataframe['rsi_14'] < 35)
        
        # === 3level warning：abnormal volume distribution（capital-flow shift） ===
        # heavy selling appears within a bullish trend
        distribution_volume = (
            (dataframe['close'] > dataframe['ema_13']) &  # still in an uptrend
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5) &  # abnormal volume spike
            (dataframe['close'] < dataframe['open']) &  # but closes bearish
            (dataframe['close'] < (dataframe['high'] + dataframe['low']) / 2)  # atKdown
        )
        
        # heavy buying appears within a bearish trend
        accumulation_volume = (
            (dataframe['close'] < dataframe['ema_13']) &  # still in a downtrend
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5) &  # abnormal volume spike
            (dataframe['close'] > dataframe['open']) &  # but closes bullish
            (dataframe['close'] > (dataframe['high'] + dataframe['low']) / 2)  # atKup
        )
        
        # === 4level warning：translated+volatility ===
        # moving averages begin to converge（the trend is about to end）
        ema_convergence = (
            abs(dataframe['ema_5'] - dataframe['ema_13']) < dataframe['atr'] * 0.8
        )
        
        # abnormal volatility compression（the calm before the storm）
        volatility_squeeze = (
            dataframe['atr_p'] < dataframe['atr_p'].rolling(20).quantile(0.3)
        ) & (
            dataframe['bb_width'] < dataframe['bb_width'].rolling(20).quantile(0.2)
        )
        
        # === calculate the composite warning level ===
        warning_level = self._safe_series(0, len(dataframe))
        
        # bullish reversal warning
        bullish_reversal_signals = (
            momentum_decay_short.astype(int) +
            bullish_rsi_divergence.astype(int) +
            accumulation_volume.astype(int) +
            (ema_convergence & volatility_squeeze).astype(int)
        )
        
        # bearish reversal warning  
        bearish_reversal_signals = (
            momentum_decay_long.astype(int) +
            bearish_rsi_divergence.astype(int) +  
            distribution_volume.astype(int) +
            (ema_convergence & volatility_squeeze).astype(int)
        )
        
        # warning level：1-4translated，the higher the level, the greater the reversal probability
        warning_level = np.maximum(bullish_reversal_signals, bearish_reversal_signals)
        
        # === reversal-probability calculation ===
        # probability model based on historical statistics
        reversal_probability = np.where(
            warning_level >= 3, 0.75,  # 3-4level warning：75%probability
            np.where(warning_level == 2, 0.55,  # 2level warning：55%probability
                    np.where(warning_level == 1, 0.35, 0.1))  # 1level warning：35%probability
        )
        
        # === signal-strength score ===
        signal_strength = (
            bullish_reversal_signals * 25 -  # bullish signals are positive
            bearish_reversal_signals * 25    # bearish signals are negative
        ).clip(-100, 100)
        
        return {
            'level': warning_level,
            'probability': reversal_probability,
            'signal_strength': signal_strength,
            'bullish_signals': bullish_reversal_signals,
            'bearish_signals': bearish_reversal_signals
        }
    
    def validate_breakout_effectiveness(self, dataframe: DataFrame) -> dict:
        """🔍 breakout-validity verification system - breakoutvsbreakout"""
        
        # === 1. volume breakout confirmation ===
        # breakouts must be accompanied by expanding volume
        volume_breakout_score = np.where(
            dataframe['volume_ratio'] > 2.0, 3,  # abnormal volume spike：3translated
            np.where(dataframe['volume_ratio'] > 1.5, 2,  # significant volume expansion：2translated
                    np.where(dataframe['volume_ratio'] > 1.2, 1, 0))  # moderate volume expansion：1translated，no volume expansion：0translated
        )
        
        # === 2. price-strength validation ===
        # score breakout magnitude and strength
        atr_current = dataframe['atr']
        
        # upward breakout strength
        upward_strength = np.where(
            # break above the upper Bollinger Band + translated1countATR
            (dataframe['close'] > dataframe['bb_upper']) & 
            ((dataframe['close'] - dataframe['bb_upper']) > atr_current), 3,
            np.where(
                # break above the upper Bollinger Band but1countATR
                dataframe['close'] > dataframe['bb_upper'], 2,
                np.where(
                    # break above the Bollinger middle band
                    dataframe['close'] > dataframe['bb_middle'], 1, 0
                )
            )
        )
        
        # downward breakout strength  
        downward_strength = np.where(
            # break below the lower Bollinger Band + translated1countATR
            (dataframe['close'] < dataframe['bb_lower']) & 
            ((dataframe['bb_lower'] - dataframe['close']) > atr_current), -3,
            np.where(
                # break below the lower Bollinger Band but1countATR
                dataframe['close'] < dataframe['bb_lower'], -2,
                np.where(
                    # Bollinger Bands mid
                    dataframe['close'] < dataframe['bb_middle'], -1, 0
                )
            )
        )
        
        price_strength = upward_strength + downward_strength  # combined score
        
        # === 3. time-persistence validation ===
        # follow-through confirmation after the breakout（after2-3translatedKtranslated）
        breakout_persistence = self._safe_series(0, len(dataframe))
        
        # upward-breakout persistence
        upward_persistence = (
            (dataframe['close'] > dataframe['bb_middle']) &  # currently above the middle band
            (dataframe['close'].shift(-1) > dataframe['bb_middle'].shift(-1)) &  # the next candle is also there
            (dataframe['low'].shift(-1) > dataframe['bb_middle'].shift(-1) * 0.995)  # and the pullback is shallow
        ).astype(int) * 2
        
        # downward-breakout persistence
        downward_persistence = (
            (dataframe['close'] < dataframe['bb_middle']) &  # currently below the middle band
            (dataframe['close'].shift(-1) < dataframe['bb_middle'].shift(-1)) &  # the next candle is also there
            (dataframe['high'].shift(-1) < dataframe['bb_middle'].shift(-1) * 1.005)  # and the bounce is limited
        ).astype(int) * -2
        
        breakout_persistence = upward_persistence + downward_persistence
        
        # === 4. fake-breakout filter ===
        # detect common fake-breakout patterns
        false_breakout_penalty = self._safe_series(0, len(dataframe))
        
        # fake breakout with an overly long upper wick（pushes up and then fades）
        long_upper_shadow = (
            (dataframe['high'] - dataframe['close']) > (dataframe['close'] - dataframe['open']) * 2
        ) & (dataframe['close'] > dataframe['open'])  # bullish candle but upper wick is too long
        false_breakout_penalty -= long_upper_shadow.astype(int) * 2
        
        # fake breakout with an overly long lower wick（dips and then rebounds）
        long_lower_shadow = (
            (dataframe['close'] - dataframe['low']) > (dataframe['open'] - dataframe['close']) * 2
        ) & (dataframe['close'] < dataframe['open'])  # bearish candle but lower wick is too long
        false_breakout_penalty -= long_lower_shadow.astype(int) * 2
        
        # === 5. technical-indicator confirmation ===
        # RSItranslatedMACDconfirm
        technical_confirmation = self._safe_series(0, len(dataframe))
        
        # bullish breakout confirmation
        bullish_tech_confirm = (
            (dataframe['rsi_14'] > 50) &  # RSItranslated
            (dataframe['macd_hist'] > 0) &  # MACDas
            (dataframe['trend_strength'] > 0)  # trend strength is positive
        ).astype(int) * 2
        
        # bearish breakout confirmation
        bearish_tech_confirm = (
            (dataframe['rsi_14'] < 50) &  # RSItranslated
            (dataframe['macd_hist'] < 0) &  # MACDas
            (dataframe['trend_strength'] < 0)  # trend strength is negative
        ).astype(int) * -2
        
        technical_confirmation = bullish_tech_confirm + bearish_tech_confirm
        
        # === 6. has score ===
        # weight allocation
        validity_score = (
            volume_breakout_score * 0.30 +      # volume confirmation：30%
            price_strength * 0.25 +             # price strength：25%
            breakout_persistence * 0.20 +       # translated：20%
            technical_confirmation * 0.15 +     # confirm：15%
            false_breakout_penalty * 0.10       # breakout：10%
        ).clip(-10, 10)
        
        # === 7. calculate ===
        # score calculate breakout
        confidence = np.where(
            abs(validity_score) >= 6, 0.85,  # high confidence：85%
            np.where(abs(validity_score) >= 4, 0.70,  # medium confidence：70%
                    np.where(abs(validity_score) >= 2, 0.55,  # low confidence：55%
                            0.30))  # very low confidence：30%
        )
        
        # === 8. breakout ===
        breakout_type = self._safe_series('NONE', len(dataframe), 'NONE')
        
        # strong breakout
        strong_breakout_up = (validity_score >= 5) & (price_strength > 0)
        strong_breakout_down = (validity_score <= -5) & (price_strength < 0)
        
        # moderate breakout
        mild_breakout_up = (validity_score >= 2) & (validity_score < 5) & (price_strength > 0)
        mild_breakout_down = (validity_score <= -2) & (validity_score > -5) & (price_strength < 0)
        
        # possible fake breakout
        false_breakout = (abs(validity_score) < 2) & (abs(price_strength) > 0)
        
        breakout_type.loc[strong_breakout_up] = 'STRONG_BULLISH'
        breakout_type.loc[strong_breakout_down] = 'STRONG_BEARISH'
        breakout_type.loc[mild_breakout_up] = 'MILD_BULLISH'
        breakout_type.loc[mild_breakout_down] = 'MILD_BEARISH'
        breakout_type.loc[false_breakout] = 'LIKELY_FALSE'
        
        return {
            'validity_score': validity_score,
            'confidence': confidence,
            'breakout_type': breakout_type,
            'volume_score': volume_breakout_score,
            'price_strength': price_strength,
            'persistence': breakout_persistence,
            'tech_confirmation': technical_confirmation
        }
    
    def calculate_market_regime_simple(self, dataframe: DataFrame) -> DataFrame:
        """simplified market-state detection - optimizeDataFrametranslated"""
        
        # 1 times calculate has，avoidDataFrametranslated
        new_columns = {}
        
        # determine market type based on trend strength and volatility state
        conditions = [
            (dataframe['trend_strength'] > 75) & (dataframe['adx'] > 25),  # strong trend
            (dataframe['trend_strength'] > 50) & (dataframe['adx'] > 20),  # medium trend  
            (dataframe['volatility_state'] > 75),  # high volatility
            (dataframe['adx'] < 20) & (dataframe['volatility_state'] < 30)  # consolidation
        ]
        
        choices = ['strong_trend', 'medium_trend', 'volatile', 'consolidation']
        new_columns['market_regime'] = np.select(conditions, choices, default='neutral')
        
        # market sentiment indicator (simplified version)
        price_vs_ma = np.where(dataframe['ema_21'] > 0, 
                              (dataframe['close'] - dataframe['ema_21']) / dataframe['ema_21'], 
                              0)
        volume_sentiment = np.where(dataframe['volume_ratio'] > 1.2, 1, 
                                  np.where(dataframe['volume_ratio'] < 0.8, -1, 0))
        
        new_columns['market_sentiment'] = (price_vs_ma * 10 + volume_sentiment) / 2
        
        # use value has new，avoidconcattranslated
        if new_columns:
            for col_name, value in new_columns.items():
                if isinstance(value, pd.Series):
                    # translatedSerieslong anddataframetranslated
                    if len(value) == len(dataframe):
                        dataframe[col_name] = value.values
                    else:
                        dataframe[col_name] = value
                else:
                    dataframe[col_name] = value
        
        return dataframe
    
    def ichimoku(self, dataframe: DataFrame, tenkan=9, kijun=26, senkou_b=52) -> DataFrame:
        """Ichimoku cloud indicators - optimizeDataFrametranslated"""
        # batch-calculate all indicators
        new_columns = {}
        
        new_columns['tenkan'] = (dataframe['high'].rolling(tenkan).max() + dataframe['low'].rolling(tenkan).min()) / 2
        new_columns['kijun'] = (dataframe['high'].rolling(kijun).max() + dataframe['low'].rolling(kijun).min()) / 2
        new_columns['senkou_a'] = ((new_columns['tenkan'] + new_columns['kijun']) / 2).shift(kijun)
        new_columns['senkou_b'] = ((dataframe['high'].rolling(senkou_b).max() + dataframe['low'].rolling(senkou_b).min()) / 2).shift(kijun)
        new_columns['chikou'] = dataframe['close'].shift(-kijun)
        
        # use value has new，avoidconcattranslated
        if new_columns:
            for col_name, value in new_columns.items():
                if isinstance(value, pd.Series):
                    # translatedSerieslong anddataframetranslated
                    if len(value) == len(dataframe):
                        dataframe[col_name] = value.values
                    else:
                        dataframe[col_name] = value
                else:
                    dataframe[col_name] = value
        
        return dataframe
    
    def supertrend(self, dataframe: DataFrame, period=10, multiplier=3) -> pd.Series:
        """Super Trend indicator"""
        hl2 = (dataframe['high'] + dataframe['low']) / 2
        atr = ta.ATR(dataframe, timeperiod=period)
        
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        supertrend = dataframe['close'] * 0  # initialize
        direction = self._safe_series(0.0, len(dataframe))
        
        for i in range(1, len(dataframe)):
            if dataframe['close'].iloc[i] > upper_band.iloc[i-1]:
                direction.iloc[i] = 1
            elif dataframe['close'].iloc[i] < lower_band.iloc[i-1]:
                direction.iloc[i] = -1
            else:
                direction.iloc[i] = direction.iloc[i-1]
            
            if direction.iloc[i] == 1:
                supertrend.iloc[i] = lower_band.iloc[i]
            else:
                supertrend.iloc[i] = upper_band.iloc[i]
                
        return supertrend
    
    def calculate_advanced_volatility_indicators(self, dataframe: DataFrame) -> DataFrame:
        """calculate advanced volatility indicators"""
        
        # Keltner translated（translatedATRtranslated）
        kc_period = 20
        kc_multiplier = 2
        kc_middle = ta.EMA(dataframe, timeperiod=kc_period)
        kc_range = ta.ATR(dataframe, timeperiod=kc_period) * kc_multiplier
        dataframe['kc_upper'] = kc_middle + kc_range
        dataframe['kc_lower'] = kc_middle - kc_range
        dataframe['kc_middle'] = kc_middle
        dataframe['kc_width'] = np.where(dataframe['kc_middle'] > 0, 
                                        (dataframe['kc_upper'] - dataframe['kc_lower']) / dataframe['kc_middle'], 
                                        0)
        dataframe['kc_position'] = (dataframe['close'] - dataframe['kc_lower']) / (dataframe['kc_upper'] - dataframe['kc_lower'])
        
        # Donchian translated（breakout）
        dc_period = 20
        dataframe['dc_upper'] = dataframe['high'].rolling(dc_period).max()
        dataframe['dc_lower'] = dataframe['low'].rolling(dc_period).min()
        dataframe['dc_middle'] = (dataframe['dc_upper'] + dataframe['dc_lower']) / 2
        dataframe['dc_width'] = np.where(dataframe['dc_middle'] > 0, 
                                        (dataframe['dc_upper'] - dataframe['dc_lower']) / dataframe['dc_middle'], 
                                        0)
        
        # Bollinger Bandwidth（volatility）
        dataframe['bb_bandwidth'] = dataframe['bb_width']  # at indicator mid calculate
        dataframe['bb_squeeze'] = (dataframe['bb_bandwidth'] < dataframe['bb_bandwidth'].rolling(20).quantile(0.2)).astype(int)
        
        # Chaikin Volatility（volume volatility）
        cv_period = 10
        hl_ema = ta.EMA(dataframe['high'] - dataframe['low'], timeperiod=cv_period)
        dataframe['chaikin_volatility'] = ((hl_ema - hl_ema.shift(cv_period)) / hl_ema.shift(cv_period)) * 100
        
        # volatility（VIXtranslated）
        returns = dataframe['close'].pct_change()
        dataframe['volatility_index'] = returns.rolling(20).std() * np.sqrt(365) * 100  # volatility
        
        return dataframe
    
    def calculate_advanced_momentum_indicators(self, dataframe: DataFrame) -> DataFrame:
        """calculate advanced momentum indicators"""
        
        # Fisher Transform（price）
        dataframe = self.fisher_transform(dataframe)
        
        # KSTindicator（translatedROCtranslated）
        dataframe = self.kst_indicator(dataframe)
        
        # Coppocktranslated（long momentum indicators）
        dataframe = self.coppock_curve(dataframe)
        
        # Vortexindicator（trend strength）
        dataframe = self.vortex_indicator(dataframe)
        
        # Stochastic Momentum Index（SMI）
        dataframe = self.stochastic_momentum_index(dataframe)
        
        # True Strength Index（TSI）
        dataframe = self.true_strength_index(dataframe)
        
        return dataframe
    
    def fisher_transform(self, dataframe: DataFrame, period: int = 10) -> DataFrame:
        """calculateFisher Transformindicator"""
        hl2 = (dataframe['high'] + dataframe['low']) / 2
        
        # calculate price most large value most small value
        high_n = hl2.rolling(period).max()
        low_n = hl2.rolling(period).min()
        
        # price to-1to1translated
        normalized_price = 2 * ((hl2 - low_n) / (high_n - low_n) - 0.5)
        normalized_price = normalized_price.clip(-0.999, 0.999)  # translated
        
        # Fisher Transform
        fisher = self._safe_series(0.0, len(dataframe))
        fisher[0] = 0
        
        for i in range(1, len(dataframe)):
            if not pd.isna(normalized_price.iloc[i]):
                raw_fisher = 0.5 * np.log((1 + normalized_price.iloc[i]) / (1 - normalized_price.iloc[i]))
                fisher.iloc[i] = 0.5 * fisher.iloc[i-1] + 0.5 * raw_fisher
            else:
                fisher.iloc[i] = fisher.iloc[i-1]
        
        dataframe['fisher'] = fisher
        dataframe['fisher_signal'] = fisher.shift(1)
        
        return dataframe
    
    def kst_indicator(self, dataframe: DataFrame) -> DataFrame:
        """calculateKST (Know Sure Thing) indicator"""
        # 4 countROCtranslated
        roc1 = ta.ROC(dataframe, timeperiod=10)
        roc2 = ta.ROC(dataframe, timeperiod=15)
        roc3 = ta.ROC(dataframe, timeperiod=20)
        roc4 = ta.ROC(dataframe, timeperiod=30)
        
        # translatedROCtranslated
        roc1_ma = ta.SMA(roc1, timeperiod=10)
        roc2_ma = ta.SMA(roc2, timeperiod=10)
        roc3_ma = ta.SMA(roc3, timeperiod=10)
        roc4_ma = ta.SMA(roc4, timeperiod=15)
        
        # KSTcalculate（translated）
        dataframe['kst'] = (roc1_ma * 1) + (roc2_ma * 2) + (roc3_ma * 3) + (roc4_ma * 4)
        dataframe['kst_signal'] = ta.SMA(dataframe['kst'], timeperiod=9)
        
        return dataframe
    
    def coppock_curve(self, dataframe: DataFrame, wma_period: int = 10) -> DataFrame:
        """calculateCoppocktranslated"""
        # Coppock ROCcalculate
        roc11 = ta.ROC(dataframe, timeperiod=11)
        roc14 = ta.ROC(dataframe, timeperiod=14)
        
        # countROCtranslated
        roc_sum = roc11 + roc14
        
        # translated
        dataframe['coppock'] = ta.WMA(roc_sum, timeperiod=wma_period)
        
        return dataframe
    
    def vortex_indicator(self, dataframe: DataFrame, period: int = 14) -> DataFrame:
        """calculateVortexindicator"""
        # True Range
        tr = ta.TRANGE(dataframe)
        
        # translated
        vm_plus = abs(dataframe['high'] - dataframe['low'].shift(1))
        vm_minus = abs(dataframe['low'] - dataframe['high'].shift(1))
        
        # translated
        vm_plus_sum = vm_plus.rolling(period).sum()
        vm_minus_sum = vm_minus.rolling(period).sum()
        tr_sum = tr.rolling(period).sum()
        
        # VIcalculate
        dataframe['vi_plus'] = vm_plus_sum / tr_sum
        dataframe['vi_minus'] = vm_minus_sum / tr_sum
        dataframe['vi_diff'] = dataframe['vi_plus'] - dataframe['vi_minus']
        
        return dataframe
    
    def stochastic_momentum_index(self, dataframe: DataFrame, k_period: int = 10, d_period: int = 3) -> DataFrame:
        """calculate momentum (SMI)"""
        # price mid
        mid_point = (dataframe['high'].rolling(k_period).max() + dataframe['low'].rolling(k_period).min()) / 2
        
        # calculateSMI
        numerator = (dataframe['close'] - mid_point).rolling(k_period).sum()
        denominator = (dataframe['high'].rolling(k_period).max() - dataframe['low'].rolling(k_period).min()).rolling(k_period).sum() / 2
        
        smi_k = (numerator / denominator) * 100
        dataframe['smi_k'] = smi_k
        dataframe['smi_d'] = smi_k.rolling(d_period).mean()
        
        return dataframe
    
    def true_strength_index(self, dataframe: DataFrame, r: int = 25, s: int = 13) -> DataFrame:
        """calculate strength (TSI)"""
        # price
        price_change = dataframe['close'].diff()
        
        # times price
        first_smooth_pc = price_change.ewm(span=r).mean()
        double_smooth_pc = first_smooth_pc.ewm(span=s).mean()
        
        # times value price
        first_smooth_abs_pc = abs(price_change).ewm(span=r).mean()
        double_smooth_abs_pc = first_smooth_abs_pc.ewm(span=s).mean()
        
        # TSIcalculate
        dataframe['tsi'] = 100 * (double_smooth_pc / double_smooth_abs_pc)
        dataframe['tsi_signal'] = dataframe['tsi'].ewm(span=7).mean()
        
        return dataframe
    
    def calculate_advanced_volume_indicators(self, dataframe: DataFrame) -> DataFrame:
        """calculate high volume indicators"""
        
        # Accumulation/Distribution Line（A/Dtranslated）
        dataframe['ad_line'] = ta.AD(dataframe)
        dataframe['ad_line_ma'] = ta.SMA(dataframe['ad_line'], timeperiod=20)
        
        # Money Flow Index（MFI - volumeRSI）
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        # Force Index（translated）
        force_index = (dataframe['close'] - dataframe['close'].shift(1)) * dataframe['volume']
        dataframe['force_index'] = force_index.ewm(span=13).mean()
        dataframe['force_index_ma'] = force_index.rolling(20).mean()
        
        # Ease of Movement（translated）
        high_low_avg = (dataframe['high'] + dataframe['low']) / 2
        high_low_avg_prev = high_low_avg.shift(1)
        distance_moved = high_low_avg - high_low_avg_prev
        
        high_low_diff = dataframe['high'] - dataframe['low']
        box_ratio = (dataframe['volume'] / 1000000) / (high_low_diff + 1e-10)
        
        emv_1 = distance_moved / (box_ratio + 1e-10)
        dataframe['emv'] = emv_1.rolling(14).mean()
        
        # Chaikin Money Flow（CMF）
        money_flow_multiplier = ((dataframe['close'] - dataframe['low']) - 
                               (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'] + 1e-10)
        money_flow_volume = money_flow_multiplier * dataframe['volume']
        dataframe['cmf'] = money_flow_volume.rolling(20).sum() / (dataframe['volume'].rolling(20).sum() + 1e-10)
        
        # Volume Price Trend（VPT）
        vpt = (dataframe['volume'] * ((dataframe['close'] - dataframe['close'].shift(1)) / (dataframe['close'].shift(1) + 1e-10)))
        dataframe['vpt'] = vpt.cumsum()
        dataframe['vpt_ma'] = dataframe['vpt'].rolling(20).mean()
        
        return dataframe
    
    def calculate_market_structure_indicators(self, dataframe: DataFrame) -> DataFrame:
        """calculate market-structure indicators"""
        
        # Price Actionindicator
        dataframe = self.calculate_price_action_indicators(dataframe)
        
        # support/resistance
        dataframe = self.identify_support_resistance(dataframe)
        
        # translated
        dataframe = self.calculate_wave_analysis(dataframe)
        
        # price
        dataframe = self.calculate_price_density(dataframe)
        
        return dataframe
    
    def calculate_price_action_indicators(self, dataframe: DataFrame) -> DataFrame:
        """calculate price as indicator"""
        # large small
        dataframe['real_body'] = abs(dataframe['close'] - dataframe['open'])
        dataframe['real_body_pct'] = dataframe['real_body'] / (dataframe['close'] + 1e-10) * 100
        
        # up down
        dataframe['upper_shadow'] = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        dataframe['lower_shadow'] = dataframe[['open', 'close']].min(axis=1) - dataframe['low']
        
        # Ktranslated
        dataframe['is_doji'] = (dataframe['real_body_pct'] < 0.1).astype(int)
        dataframe['is_hammer'] = ((dataframe['lower_shadow'] > dataframe['real_body'] * 2) & 
                                 (dataframe['upper_shadow'] < dataframe['real_body'] * 0.5)).astype(int)
        dataframe['is_shooting_star'] = ((dataframe['upper_shadow'] > dataframe['real_body'] * 2) & 
                                        (dataframe['lower_shadow'] < dataframe['real_body'] * 0.5)).astype(int)
        
        # Pin Bar translated
        # Pin Bar Bullish: long down，small，short up，signal
        dataframe['is_pin_bar_bullish'] = ((dataframe['lower_shadow'] > dataframe['real_body'] * 2) & 
                                          (dataframe['upper_shadow'] < dataframe['real_body']) &
                                          (dataframe['real_body_pct'] < 2.0) &  # small
                                          (dataframe['close'] > dataframe['open'])).astype(int)  # translated
        
        # Pin Bar Bearish: long up，small，short down，signal
        dataframe['is_pin_bar_bearish'] = ((dataframe['upper_shadow'] > dataframe['real_body'] * 2) & 
                                          (dataframe['lower_shadow'] < dataframe['real_body']) &
                                          (dataframe['real_body_pct'] < 2.0) &  # small
                                          (dataframe['close'] < dataframe['open'])).astype(int)  # translated
        
        # translated
        # before get before 1Ktranslated
        prev_open = dataframe['open'].shift(1)
        prev_close = dataframe['close'].shift(1)
        prev_high = dataframe['high'].shift(1)
        prev_low = dataframe['low'].shift(1)
        
        # translated：before before 1
        dataframe['is_bullish_engulfing'] = ((dataframe['close'] > dataframe['open']) &  # before as
                                           (prev_close < prev_open) &  # before 1 as
                                           (dataframe['open'] < prev_close) &  # before low before 1
                                           (dataframe['close'] > prev_open) &  # before high before 1
                                           (dataframe['real_body'] > dataframe['real_body'].shift(1) * 1.2)).astype(int)  # before large
        
        # translated：before before 1
        dataframe['is_bearish_engulfing'] = ((dataframe['close'] < dataframe['open']) &  # before as
                                           (prev_close > prev_open) &  # before 1 as
                                           (dataframe['open'] > prev_close) &  # before high before 1
                                           (dataframe['close'] < prev_open) &  # before low before 1
                                           (dataframe['real_body'] > dataframe['real_body'].shift(1) * 1.2)).astype(int)  # before large
        
        return dataframe
    
    def identify_support_resistance(self, dataframe: DataFrame, window: int = 20) -> DataFrame:
        """support resistance"""
        # calculate has support resistance indicator，1 times avoid
        sr_columns = {
            'local_max': dataframe['high'].rolling(window, center=True).max() == dataframe['high'],
            'local_min': dataframe['low'].rolling(window, center=True).min() == dataframe['low'],
            'resistance_distance': np.where(dataframe['close'] > 0, 
                                           (dataframe['high'].rolling(50).max() - dataframe['close']) / dataframe['close'], 
                                           0),
            'support_distance': np.where(dataframe['close'] > 0, 
                                        (dataframe['close'] - dataframe['low'].rolling(50).min()) / dataframe['close'], 
                                        0)
        }
        
        sr_df = pd.DataFrame(sr_columns, index=dataframe.index)
        return pd.concat([dataframe, sr_df], axis=1)
    
    def calculate_wave_analysis(self, dataframe: DataFrame) -> DataFrame:
        """calculate indicator"""
        # Elliott Waveindicator，1 times calculate avoid
        returns = dataframe['close'].pct_change()
        
        wave_columns = {
            'wave_strength': abs(dataframe['close'] - dataframe['close'].shift(5)) / (dataframe['close'].shift(5) + 1e-10),
            'normalized_returns': returns / (returns.rolling(20).std() + 1e-10),
            'momentum_dispersion': dataframe['mom_10'].rolling(10).std() / (abs(dataframe['mom_10']).rolling(10).mean() + 1e-10)
        }
        
        wave_df = pd.DataFrame(wave_columns, index=dataframe.index)
        return pd.concat([dataframe, wave_df], axis=1)
    
    def calculate_price_density(self, dataframe: DataFrame) -> DataFrame:
        """calculate price indicator - optimizeDataFrametranslated"""
        # 1 times calculate has
        new_columns = {}
        
        # price
        price_range = dataframe['high'] - dataframe['low']
        new_columns['price_range_pct'] = price_range / (dataframe['close'] + 1e-10) * 100
        
        # simplified price calculate
        new_columns['price_density'] = 1 / (new_columns['price_range_pct'] + 0.1)  # price small high
        
        # use value has new，avoidconcattranslated
        if new_columns:
            for col_name, value in new_columns.items():
                if isinstance(value, pd.Series):
                    # translatedSerieslong anddataframetranslated
                    if len(value) == len(dataframe):
                        dataframe[col_name] = value.values
                    else:
                        dataframe[col_name] = value
                else:
                    dataframe[col_name] = value
        
        return dataframe
    
    def calculate_composite_indicators(self, dataframe: DataFrame) -> DataFrame:
        """calculate indicator - optimizeDataFrametranslated"""
        
        # 1 times calculate has
        new_columns = {}
        
        # momentum score
        new_columns['momentum_score'] = self.calculate_momentum_score(dataframe)
        
        # trend strength score
        new_columns['trend_strength_score'] = self.calculate_trend_strength_score(dataframe)
        
        # volatility score
        new_columns['volatility_regime'] = self.calculate_volatility_regime(dataframe)
        
        # market state score
        new_columns['market_regime'] = self.calculate_market_regime(dataframe)
        
        # risk adjustment indicator
        new_columns['risk_adjusted_return'] = self.calculate_risk_adjusted_returns(dataframe)
        
        # translated
        new_columns['technical_health'] = self.calculate_technical_health(dataframe)
        
        # use value has new，avoidconcattranslated
        if new_columns:
            for col_name, value in new_columns.items():
                if isinstance(value, pd.Series):
                    # translatedSerieslong anddataframetranslated
                    if len(value) == len(dataframe):
                        dataframe[col_name] = value.values
                    else:
                        dataframe[col_name] = value
                else:
                    dataframe[col_name] = value
        
        return dataframe
    
    def calculate_momentum_score(self, dataframe: DataFrame) -> pd.Series:
        """calculate momentum score"""
        # count momentum indicators
        momentum_indicators = {}
        
        # momentum indicators
        if 'rsi_14' in dataframe.columns:
            momentum_indicators['rsi_14'] = (dataframe['rsi_14'] - 50) / 50  # translatedRSI
        if 'mom_10' in dataframe.columns:
            momentum_indicators['mom_10'] = np.where(dataframe['close'] > 0, 
                                                     dataframe['mom_10'] / dataframe['close'] * 100, 
                                                     0)  # momentum
        if 'roc_10' in dataframe.columns:
            momentum_indicators['roc_10'] = dataframe['roc_10'] / 100  # ROC
        if 'macd' in dataframe.columns:
            momentum_indicators['macd_normalized'] = np.where(dataframe['close'] > 0, 
                                                             dataframe['macd'] / dataframe['close'] * 1000, 
                                                             0)  # translatedMACD
        
        # advanced momentum indicators
        if 'kst' in dataframe.columns:
            momentum_indicators['kst_normalized'] = dataframe['kst'] / abs(dataframe['kst']).rolling(20).mean()  # translatedKST
        if 'fisher' in dataframe.columns:
            momentum_indicators['fisher'] = dataframe['fisher']  # Fisher Transform
        if 'tsi' in dataframe.columns:
            momentum_indicators['tsi'] = dataframe['tsi'] / 100  # TSI
        if 'vi_diff' in dataframe.columns:
            momentum_indicators['vi_diff'] = dataframe['vi_diff']  # Vortexvalue
        
        # translated
        weights = {
            'rsi_14': 0.15, 'mom_10': 0.10, 'roc_10': 0.10, 'macd_normalized': 0.15,
            'kst_normalized': 0.15, 'fisher': 0.15, 'tsi': 0.10, 'vi_diff': 0.10
        }
        
        momentum_score = self._safe_series(0.0, len(dataframe))
        
        for indicator, weight in weights.items():
            if indicator in momentum_indicators:
                normalized_indicator = momentum_indicators[indicator].fillna(0)
                # at-1to1translated
                normalized_indicator = normalized_indicator.clip(-3, 3) / 3
                momentum_score += normalized_indicator * weight
        
        return momentum_score.clip(-1, 1)
    
    def calculate_trend_strength_score(self, dataframe: DataFrame) -> pd.Series:
        """calculate trend strength score"""
        # trend indicator
        trend_indicators = {}
        
        if 'adx' in dataframe.columns:
            trend_indicators['adx'] = dataframe['adx'] / 100  # ADXtranslated
        
        # EMAtranslated
        trend_indicators['ema_trend'] = self.calculate_ema_trend_score(dataframe)
        
        # SuperTrend
        trend_indicators['supertrend_trend'] = self.calculate_supertrend_score(dataframe)
        
        # Ichimoku
        trend_indicators['ichimoku_trend'] = self.calculate_ichimoku_score(dataframe)
        
        # trend
        trend_indicators['linear_reg_trend'] = self.calculate_linear_regression_trend(dataframe)
        
        weights = {
            'adx': 0.3, 'ema_trend': 0.25, 'supertrend_trend': 0.2,
            'ichimoku_trend': 0.15, 'linear_reg_trend': 0.1
        }
        
        trend_score = self._safe_series(0.0, len(dataframe))
        
        for indicator, weight in weights.items():
            if indicator in trend_indicators:
                normalized_indicator = trend_indicators[indicator].fillna(0)
                trend_score += normalized_indicator * weight
        
        return trend_score.clip(-1, 1)
    
    def calculate_ema_trend_score(self, dataframe: DataFrame) -> pd.Series:
        """calculateEMAtrend score"""
        score = self._safe_series(0.0, len(dataframe))
        
        # EMAtranslated
        if all(col in dataframe.columns for col in ['ema_8', 'ema_21', 'ema_50']):
            # bullish: EMA8 > EMA21 > EMA50
            score += (dataframe['ema_8'] > dataframe['ema_21']).astype(int) * 0.4
            score += (dataframe['ema_21'] > dataframe['ema_50']).astype(int) * 0.3
            score += (dataframe['close'] > dataframe['ema_8']).astype(int) * 0.3
            
            # bearish：translated
            score -= (dataframe['ema_8'] < dataframe['ema_21']).astype(int) * 0.4
            score -= (dataframe['ema_21'] < dataframe['ema_50']).astype(int) * 0.3
            score -= (dataframe['close'] < dataframe['ema_8']).astype(int) * 0.3
        
        return score.clip(-1, 1)
    
    def calculate_supertrend_score(self, dataframe: DataFrame) -> pd.Series:
        """calculateSuperTrendscore"""
        if 'supertrend' not in dataframe.columns:
            return self._safe_series(0.0, len(dataframe))
        
        # SuperTrendtranslated
        trend_score = ((dataframe['close'] > dataframe['supertrend']).astype(int) * 2 - 1)
        
        # translated
        distance_factor = np.where(dataframe['close'] > 0, 
                                  abs(dataframe['close'] - dataframe['supertrend']) / dataframe['close'], 
                                  0)
        distance_factor = distance_factor.clip(0, 0.1) / 0.1  # most10%translated
        
        return trend_score * distance_factor
    
    def calculate_ichimoku_score(self, dataframe: DataFrame) -> pd.Series:
        """calculateIchimokuscore"""
        score = self._safe_series(0.0, len(dataframe))
        
        # Ichimokusignal
        if all(col in dataframe.columns for col in ['tenkan', 'kijun', 'senkou_a', 'senkou_b']):
            # price at up
            above_cloud = ((dataframe['close'] > dataframe['senkou_a']) & 
                          (dataframe['close'] > dataframe['senkou_b'])).astype(int)
            
            # price at down
            below_cloud = ((dataframe['close'] < dataframe['senkou_a']) & 
                          (dataframe['close'] < dataframe['senkou_b'])).astype(int)
            
            # Tenkan-Kijuntranslated
            tenkan_above_kijun = (dataframe['tenkan'] > dataframe['kijun']).astype(int)
            
            score = (above_cloud * 0.5 + tenkan_above_kijun * 0.3 + 
                    (dataframe['close'] > dataframe['tenkan']).astype(int) * 0.2 - 
                    below_cloud * 0.5)
        
        return score.clip(-1, 1)
    
    def calculate_linear_regression_trend(self, dataframe: DataFrame, period: int = 20) -> pd.Series:
        """calculate trend"""
        def linear_reg_slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            from scipy import stats
            slope, _, r_value, _, _ = stats.linregress(x, y)
            return slope * r_value ** 2  # toRtranslated
        
        # calculate
        reg_slope = dataframe['close'].rolling(period).apply(linear_reg_slope, raw=False)
        
        # translated
        normalized_slope = np.where(dataframe['close'] > 0, 
                                   reg_slope / dataframe['close'] * 1000, 
                                   0)  # large
        
        return normalized_slope.fillna(0).clip(-1, 1)
    
    def calculate_volatility_regime(self, dataframe: DataFrame) -> pd.Series:
        """calculate volatility"""
        # before volatility
        current_vol = dataframe['atr_p']
        
        # volatility
        vol_percentile = current_vol.rolling(100).rank(pct=True)
        
        # volatility
        regime = self._safe_series(0, len(dataframe))  # 0: mid
        regime[vol_percentile < 0.2] = -1  # low volatility
        regime[vol_percentile > 0.8] = 1   # high volatility
        
        return regime
    
    def calculate_market_regime(self, dataframe: DataFrame) -> pd.Series:
        """calculate market state score"""
        # count
        regime_factors = {}
        
        if 'trend_strength_score' in dataframe.columns:
            regime_factors['trend_strength'] = dataframe['trend_strength_score']
        if 'momentum_score' in dataframe.columns:
            regime_factors['momentum'] = dataframe['momentum_score']
        if 'volatility_regime' in dataframe.columns:
            regime_factors['volatility'] = dataframe['volatility_regime'] / 2  # translated
        if 'volume_ratio' in dataframe.columns:
            regime_factors['volume_trend'] = (dataframe['volume_ratio'] - 1).clip(-1, 1)
        
        weights = {'trend_strength': 0.4, 'momentum': 0.3, 'volatility': 0.2, 'volume_trend': 0.1}
        
        market_regime = self._safe_series(0.0, len(dataframe))
        for factor, weight in weights.items():
            if factor in regime_factors:
                market_regime += regime_factors[factor].fillna(0) * weight
        
        return market_regime.clip(-1, 1)
    
    # translated calculate_risk_adjusted_returns - simplified
    def calculate_risk_adjusted_returns(self, dataframe: DataFrame, window: int = 20) -> pd.Series:
        """calculate risk adjustment"""
        # calculate
        returns = dataframe['close'].pct_change()
        
        # translatedSharpetranslated
        rolling_returns = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        risk_adjusted = rolling_returns / (rolling_std + 1e-6)  # avoid
        
        return risk_adjusted.fillna(0)
    
    def identify_coin_risk_tier(self, pair: str, dataframe: DataFrame) -> str:
        """🎯 risk level - features"""
        
        try:
            if dataframe.empty or len(dataframe) < 96:  # translated
                return 'medium_risk'  # default medium risk
                
            current_idx = -1
            
            # === features1: pricevolatility analysis ===
            volatility = dataframe['atr_p'].iloc[current_idx] if 'atr_p' in dataframe.columns else 0.05
            volatility_24h = dataframe['close'].rolling(96).std().iloc[current_idx] / dataframe['close'].iloc[current_idx]
            
            # === features2: translated ===
            volume_series = dataframe['volume'].rolling(24)
            volume_mean = volume_series.mean().iloc[current_idx]
            volume_std = volume_series.std().iloc[current_idx]
            volume_cv = (volume_std / volume_mean) if volume_mean > 0 else 5  # translated
            
            # === features3: price as features ===
            current_price = dataframe['close'].iloc[current_idx]
            price_24h_ago = dataframe['close'].iloc[-96] if len(dataframe) >= 96 else dataframe['close'].iloc[0]
            price_change_24h = abs((current_price / price_24h_ago) - 1) if price_24h_ago > 0 else 0
            
            # === features4: price ===
            is_micro_price = current_price < 0.001  # small price（translatedmemefeatures）
            is_low_price = current_price < 0.1      # low price
            
            # === features5: indicator abnormal ===
            rsi = dataframe['rsi_14'].iloc[current_idx] if 'rsi_14' in dataframe.columns else 50
            is_extreme_rsi = rsi > 80 or rsi < 20  # translatedRSIvalue
            
            # === features6: price ===
            recent_pumps = 0
            if len(dataframe) >= 24:
                for i in range(1, min(24, len(dataframe))):
                    hour_change = (dataframe['close'].iloc[-i] / dataframe['close'].iloc[-i-1]) - 1
                    if hour_change > 0.15:  # hours15%
                        recent_pumps += 1
            
            # === score ===
            risk_score = 0
            risk_factors = []
            
            # volatility score (0-40translated)
            if volatility > 0.20:  # high volatility
                risk_score += 40
                risk_factors.append(f"high volatility({volatility*100:.1f}%)")
            elif volatility > 0.10:
                risk_score += 25
                risk_factors.append(f"high volatility({volatility*100:.1f}%)")
            elif volatility > 0.05:
                risk_score += 10
                risk_factors.append(f"mid({volatility*100:.1f}%)")
            
            # not score (0-25translated)
            if volume_cv > 3:  # not
                risk_score += 25
                risk_factors.append(f"not(CV:{volume_cv:.1f})")
            elif volume_cv > 1.5:
                risk_score += 15
                risk_factors.append(f"not(CV:{volume_cv:.1f})")
            
            # short-term price abnormal score (0-20translated)
            if price_change_24h > 0.50:  # 24hours50%
                risk_score += 20
                risk_factors.append(f"24htranslated({price_change_24h*100:.1f}%)")
            elif price_change_24h > 0.20:
                risk_score += 10
                risk_factors.append(f"24hlarge({price_change_24h*100:.1f}%)")
            
            # price score (0-10translated)
            if is_micro_price:
                risk_score += 10
                risk_factors.append(f"price(${current_price:.6f})")
            elif is_low_price:
                risk_score += 5
                risk_factors.append(f"low price(${current_price:.3f})")
            
            # Pumpas score (0-15translated)
            if recent_pumps >= 3:
                risk_score += 15
                risk_factors.append(f"translatedpump({recent_pumps}times)")
            elif recent_pumps >= 1:
                risk_score += 8
                risk_factors.append(f"haspumpas({recent_pumps}times)")
            
            # === risk level ===
            if risk_score >= 70:
                risk_tier = 'high_risk'    # high risk（translated/memetranslated）
                tier_name = "⚠️ high risk"
            elif risk_score >= 40:
                risk_tier = 'medium_risk'  # medium risk
                tier_name = "⚡ medium risk"
            else:
                risk_tier = 'low_risk'     # low risk（translated）
                tier_name = "✅ low risk"
            
            # day
            logger.info(f"""
🎯 risk - {pair}:
├─ risk level: {tier_name} (score: {risk_score}/100)
├─ before price: ${current_price:.6f}
├─ volatility: {volatility*100:.2f}% | 24htranslated: {price_change_24h*100:.1f}%
├─ translatedCV: {volume_cv:.2f} | translatedPump: {recent_pumps}times
├─ translated: {' | '.join(risk_factors) if risk_factors else 'features'}
└─ translated: {'small position size to small large' if risk_tier == 'high_risk' else 'configuration' if risk_tier == 'low_risk' else 'translated'}
""")
            
            return risk_tier
            
        except Exception as e:
            logger.error(f"risk {pair}: {e}")
            return 'medium_risk'  # medium risk
    
    def calculate_technical_health(self, dataframe: DataFrame) -> pd.Series:
        """calculate"""
        health_components = {}
        
        # 1. trend 1（count indicator）
        trend_signals = []
        if 'ema_21' in dataframe.columns:
            trend_signals.append((dataframe['close'] > dataframe['ema_21']).astype(int))
        if 'macd' in dataframe.columns and 'macd_signal' in dataframe.columns:
            trend_signals.append((dataframe['macd'] > dataframe['macd_signal']).astype(int))
        if 'rsi_14' in dataframe.columns:
            trend_signals.append((dataframe['rsi_14'] > 50).astype(int))
        if 'momentum_score' in dataframe.columns:
            trend_signals.append((dataframe['momentum_score'] > 0).astype(int))
        
        if trend_signals:
            health_components['trend_consistency'] = (sum(trend_signals) / len(trend_signals) - 0.5) * 2
        
        # 2. volatility（not high not low）
        if 'volatility_regime' in dataframe.columns:
            vol_score = 1 - abs(dataframe['volatility_regime']) * 0.5  # mid most
            health_components['volatility_health'] = vol_score
        
        # 3. volume confirmation
        if 'volume_ratio' in dataframe.columns:
            volume_health = ((dataframe['volume_ratio'] > 0.8).astype(float) * 0.5 + 
                           (dataframe['volume_ratio'] < 2.0).astype(float) * 0.5)  # translated
            health_components['volume_health'] = volume_health
        
        # 4. indicator（translated/translated）
        overbought_signals = []
        oversold_signals = []
        
        if 'rsi_14' in dataframe.columns:
            overbought_signals.append((dataframe['rsi_14'] > 80).astype(int))
            oversold_signals.append((dataframe['rsi_14'] < 20).astype(int))
        if 'mfi' in dataframe.columns:
            overbought_signals.append((dataframe['mfi'] > 80).astype(int))
            oversold_signals.append((dataframe['mfi'] < 20).astype(int))
        if 'stoch_k' in dataframe.columns:
            overbought_signals.append((dataframe['stoch_k'] > 80).astype(int))
            oversold_signals.append((dataframe['stoch_k'] < 20).astype(int))
        
        if overbought_signals and oversold_signals:
            extreme_condition = ((sum(overbought_signals) >= 2).astype(int) + 
                               (sum(oversold_signals) >= 2).astype(int))
            health_components['balance_health'] = 1 - extreme_condition * 0.5
        
        # score
        weights = {
            'trend_consistency': 0.3, 'volatility_health': 0.25,
            'volume_health': 0.25, 'balance_health': 0.2
        }
        
        technical_health = self._safe_series(0.0, len(dataframe))
        for component, weight in weights.items():
            if component in health_components:
                technical_health += health_components[component].fillna(0) * weight
        
        return technical_health.clip(-1, 1)
    
    def detect_market_state(self, dataframe: DataFrame) -> str:
        """market state - translated"""
        current_idx = -1
        
        # get indicator
        adx = dataframe['adx'].iloc[current_idx]
        atr_p = dataframe['atr_p'].iloc[current_idx]
        rsi = dataframe['rsi_14'].iloc[current_idx]
        volume_ratio = dataframe['volume_ratio'].iloc[current_idx]
        price = dataframe['close'].iloc[current_idx]
        ema_8 = dataframe['ema_8'].iloc[current_idx] if 'ema_8' in dataframe.columns else price
        ema_21 = dataframe['ema_21'].iloc[current_idx]
        ema_50 = dataframe['ema_50'].iloc[current_idx]
        
        # getMACDindicator
        macd = dataframe['macd'].iloc[current_idx] if 'macd' in dataframe.columns else 0
        macd_signal = dataframe['macd_signal'].iloc[current_idx] if 'macd_signal' in dataframe.columns else 0
        
        # === translated ===
        # calculate high low
        high_20 = dataframe['high'].rolling(20).max().iloc[current_idx]
        low_20 = dataframe['low'].rolling(20).min().iloc[current_idx]
        price_position = (price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
        
        # at（avoid at）
        is_at_top = (
            price_position > 0.90 and  # price at20day high
            rsi > 70 and  # RSIoverbought
            macd < macd_signal  # MACDtranslated
        )
        
        # at（avoid at）
        is_at_bottom = (
            price_position < 0.10 and  # price at20day low
            rsi < 30 and  # RSIoversold
            macd > macd_signal  # MACDtranslated
        )
        
        # === trend strength analysis ===
        # timeEMAtranslated
        ema_bullish = ema_8 > ema_21 > ema_50
        ema_bearish = ema_8 < ema_21 < ema_50
        
        # === market state ===
        if is_at_top:
            return "market_top"  # translated，avoid
        elif is_at_bottom:
            return "market_bottom"  # translated，avoid
        elif adx > 40 and atr_p > self.volatility_threshold:
            if ema_bullish and not is_at_top:
                return "strong_uptrend"
            elif ema_bearish and not is_at_bottom:
                return "strong_downtrend"
            else:
                return "volatile"
        elif adx > 25:
            if price > ema_21 and not is_at_top:
                return "mild_uptrend"
            elif price < ema_21 and not is_at_bottom:
                return "mild_downtrend"
            else:
                return "sideways"
        elif atr_p < self.volatility_threshold * 0.5:
            return "consolidation"
        else:
            return "sideways"
    
    def calculate_var(self, returns: List[float], confidence_level: float = 0.05) -> float:
        """calculateVaR (Value at Risk)"""
        if len(returns) < 20:
            return 0.05  # default5%risk
        
        returns_array = np.array(returns)
        # use
        var = np.percentile(returns_array, confidence_level * 100)
        return abs(var)
    
    def calculate_cvar(self, returns: List[float], confidence_level: float = 0.05) -> float:
        """calculateCVaR (Conditional Value at Risk)"""
        if len(returns) < 20:
            return 0.08  # default8%risk
        
        returns_array = np.array(returns)
        var = np.percentile(returns_array, confidence_level * 100)
        # CVaRtranslatedVaRvalue
        tail_losses = returns_array[returns_array <= var]
        if len(tail_losses) > 0:
            cvar = np.mean(tail_losses)
            return abs(cvar)
        return abs(var)
    
    def calculate_portfolio_correlation(self, pair: str) -> float:
        """calculate"""
        if pair not in self.pair_returns_history:
            return 0.0
        
        current_returns = self.pair_returns_history[pair]
        if len(current_returns) < 20:
            return 0.0
        
        # calculate and
        correlations = []
        for other_pair, other_returns in self.pair_returns_history.items():
            if other_pair != pair and len(other_returns) >= 20:
                try:
                    # count long
                    min_length = min(len(current_returns), len(other_returns))
                    corr = np.corrcoef(
                        current_returns[-min_length:], 
                        other_returns[-min_length:]
                    )[0, 1]
                    if not np.isnan(corr):
                        correlations.append(abs(corr))
                except:
                    continue
        
        return np.mean(correlations) if correlations else 0.0
    
    def calculate_kelly_fraction(self, pair: str) -> float:
        """translatedKellycalculate"""
        if pair not in self.pair_performance or self.trade_count < 20:
            return 0.25  # default value
        
        try:
            pair_trades = self.pair_performance[pair]
            wins = [t for t in pair_trades if t > 0]
            losses = [t for t in pair_trades if t < 0]
            
            if len(wins) == 0 or len(losses) == 0:
                return 0.25
            
            win_prob = len(wins) / len(pair_trades)
            avg_win = np.mean(wins)
            avg_loss = abs(np.mean(losses))
            
            # Kellytranslated: f = (bp - q) / b
            # mid b = avg_win/avg_loss, p = win_prob, q = 1-win_prob
            b = avg_win / avg_loss
            kelly = (b * win_prob - (1 - win_prob)) / b
            
            # translated：useKellytranslated1/4to1/2
            kelly_adjusted = max(0.05, min(0.4, kelly * 0.25))
            return kelly_adjusted
            
        except:
            return 0.25
    
    def calculate_position_size(self, current_price: float, market_state: str, pair: str) -> float:
        """position size - configuration market state + risk"""
        
        # === 🎯 get risk level ===
        try:
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if not dataframe.empty:
                coin_risk_tier = self.identify_coin_risk_tier(pair, dataframe)
            else:
                coin_risk_tier = 'medium_risk'
        except Exception as e:
            logger.warning(f"get risk level {pair}: {e}")
            coin_risk_tier = 'medium_risk'
        
        # === risk（small position size to small large）===
        coin_risk_multipliers = {
            'low_risk': 1.0,        # low risk：position size
            'medium_risk': 0.8,     # medium risk：80%position size
            'high_risk': 0.3        # high risk（translated）：30%position size，to small large
        }
        coin_risk_multiplier = coin_risk_multipliers.get(coin_risk_tier, 0.8)
        
        # === use configuration position size range mid value as ===
        base_position = (self.base_position_size + self.max_position_size) / 2
        
        # === winning streak/losing streak ===
        streak_multiplier = 1.0
        if self.consecutive_wins >= 5:
            streak_multiplier = 1.5      # winning streak5times：position size1.5translated
        elif self.consecutive_wins >= 3:
            streak_multiplier = 1.3      # winning streak3times：position size1.3translated
        elif self.consecutive_wins >= 1:
            streak_multiplier = 1.1      # winning streak1times：position size1.1translated
        elif self.consecutive_losses >= 3:
            streak_multiplier = 0.6      # loss streak3times：position size to60%
        elif self.consecutive_losses >= 1:
            streak_multiplier = 0.8      # loss streak1times：position size to80%
            
        # === market state（simplified） ===
        market_multipliers = {
            "strong_uptrend": 1.25,      # strong trend：translated
            "strong_downtrend": 1.25,    # strong trend：translated
            "mild_uptrend": 1.2,        # medium trend
            "mild_downtrend": 1.2,      # medium trend
            "sideways": 1.0,            # translated：translated
            "volatile": 0.8,            # high volatility：translated
            "consolidation": 0.9        # translated：translated
        }
        market_multiplier = market_multipliers.get(market_state, 1.0)
        
        # === time ===
        time_multiplier = self.get_time_session_position_boost()
        
        # === translated ===
        equity_multiplier = 1.0
        if self.current_drawdown < -0.10:  # drawdown10%
            equity_multiplier = 0.6
        elif self.current_drawdown < -0.05:  # drawdown5%
            equity_multiplier = 0.8
        elif self.current_drawdown == 0:     # none drawdown，profit
            equity_multiplier = 1.15
            
        # === leverage ===
        # get before leverage
        current_leverage = getattr(self, '_current_leverage', {}).get(pair, 20)
        # leverage high，position size to low（as risk）
        leverage_adjustment = 1.0
        if current_leverage >= 75:
            leverage_adjustment = 0.8    # high leverage low position size
        elif current_leverage >= 50:
            leverage_adjustment = 0.9
        else:
            leverage_adjustment = 1.1    # low leverage to high position size
            
        # === 🚀translated（translated）===
        compound_multiplier = self.get_compound_accelerator_multiplier()
            
        # === 🎯 risk to ===
        total_multiplier = (streak_multiplier * market_multiplier * 
                          time_multiplier * equity_multiplier * 
                          leverage_adjustment * compound_multiplier * 
                          coin_risk_multiplier)  # new risk
        
        # risk level most large
        max_multiplier_limits = {
            'low_risk': 1.8,        # low risk：most1.8translated
            'medium_risk': 1.5,     # medium risk：most1.5translated
            'high_risk': 1.2        # high risk（translated）：most1.2translated，risk
        }
        max_multiplier = max_multiplier_limits.get(coin_risk_tier, 1.5)
        total_multiplier = min(total_multiplier, max_multiplier)
        
        # === most position size calculate ===
        calculated_position = base_position * total_multiplier
        
        # === position size（leverage）===
        if current_leverage >= 75:
            max_allowed_position = 0.15  # high leverage most15%
        elif current_leverage >= 50:
            max_allowed_position = 0.20  # mid high leverage most20%
        elif current_leverage >= 20:
            max_allowed_position = 0.30  # mid leverage most30%
        else:
            max_allowed_position = self.max_position_size  # low leverage configuration up
        
        # translated
        final_position = max(self.base_position_size * 0.8, 
                           min(calculated_position, max_allowed_position))
        
        # risk level
        risk_tier_names = {
            'low_risk': '✅ low risk',
            'medium_risk': '⚡ medium risk', 
            'high_risk': '⚠️ high risk'
        }
        
        logger.info(f"""
💰 position size calculate - {pair}:
├─ 🔍 risk level: {risk_tier_names.get(coin_risk_tier, coin_risk_tier)}
├─ 📊 position size: {base_position*100:.0f}%
├─ 🏆 winning streak: {streak_multiplier:.1f}x (translated{self.consecutive_wins}/translated{self.consecutive_losses})
├─ 📈 translated: {market_multiplier:.1f}x ({market_state})
├─ ⏰ time: {time_multiplier:.1f}x
├─ 💰 translated: {equity_multiplier:.1f}x
├─ ⚖️ leverage adjustment: {leverage_adjustment:.1f}x ({current_leverage}xleverage)
├─ 🚀 translated: {compound_multiplier:.1f}x
├─ 🎯 risk adjustment: {coin_risk_multiplier:.1f}x ({coin_risk_tier})
├─ 📐 translated: {max_multiplier:.1f}x (risk level)
├─ 🧮 calculate position size: {calculated_position*100:.1f}%
└─ 🎉 most position size: {final_position*100:.1f}%
""")
        
        return final_position
    
    def get_time_session_position_boost(self) -> float:
        """get time position size"""
        current_time = datetime.now(timezone.utc)
        hour = current_time.hour
        
        # position adjustment
        if 14 <= hour <= 16:       # translated：most
            return 1.2
        elif 8 <= hour <= 10:      # translated：translated  
            return 1.1
        elif 0 <= hour <= 2:       # translated：mid
            return 1.0
        elif 3 <= hour <= 7:       # translated：low
            return 0.9
        else:
            return 1.0
    
    def get_compound_accelerator_multiplier(self) -> float:
        """🚀translated - day position size"""
        
        # get day
        daily_profit = self.get_daily_profit_percentage()
        
        # translated
        if daily_profit >= 0.20:      # day > 20%
            multiplier = 1.5          # times day position size1.5translated（translated）
            mode = "🚀translated"
        elif daily_profit >= 0.10:    # day 10-20%
            multiplier = 1.5          # times day position size1.5translated
            mode = "⚡high"
        elif daily_profit >= 0.05:    # day 5-10%
            multiplier = 1.2          # times day position size1.2translated
            mode = "📈translated"
        elif daily_profit >= 0:       # day 0-5%
            multiplier = 1.0          # position size
            mode = "📊translated"
        elif daily_profit >= -0.05:   # day loss 0-5%
            multiplier = 0.8          # translated
            mode = "🔄translated"
        else:                         # day loss > 5%
            multiplier = 0.5          # times day position size（translated）
            mode = "❄️translated"
            
        # profit day
        consecutive_profit_days = self.get_consecutive_profit_days()
        if consecutive_profit_days >= 3:
            multiplier *= min(1.3, 1 + consecutive_profit_days * 0.05)  # most high30%translated
            
        # loss day
        consecutive_loss_days = self.get_consecutive_loss_days()
        if consecutive_loss_days >= 2:
            multiplier *= max(0.3, 1 - consecutive_loss_days * 0.15)   # most low30%
            
        # translated：0.3x - 2.5x
        final_multiplier = max(0.3, min(multiplier, 2.5))
        
        logger.info(f"""
🚀 translated:
├─ day: {daily_profit*100:+.2f}%
├─ translated: {mode}
├─ translated: {multiplier:.2f}x
├─ profit: {consecutive_profit_days}translated
├─ loss: {consecutive_loss_days}translated
└─ most: {final_multiplier:.2f}x
""")
        
        return final_multiplier
    
    def get_daily_profit_percentage(self) -> float:
        """get day"""
        try:
            # simplified version：before
            if hasattr(self, 'total_profit'):
                # to day calculate
                # use value
                return self.total_profit * 0.1  # day10%
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def get_consecutive_profit_days(self) -> int:
        """get profit"""
        try:
            # simplified，to after optimize as day
            if self.consecutive_wins >= 5:
                return min(7, self.consecutive_wins // 2)  # as large
            else:
                return 0
        except Exception:
            return 0
    
    def get_consecutive_loss_days(self) -> int:
        """get loss"""
        try:
            # simplified，to after optimize as day
            if self.consecutive_losses >= 3:
                return min(5, self.consecutive_losses // 1)  # as large
            else:
                return 0
        except Exception:
            return 0
    
    def update_portfolio_performance(self, pair: str, return_pct: float):
        """update record"""
        # update
        if pair not in self.pair_returns_history:
            self.pair_returns_history[pair] = []
        
        self.pair_returns_history[pair].append(return_pct)
        
        # most500count record
        if len(self.pair_returns_history[pair]) > 500:
            self.pair_returns_history[pair] = self.pair_returns_history[pair][-500:]
        
        # update record
        if pair not in self.pair_performance:
            self.pair_performance[pair] = []
        
        self.pair_performance[pair].append(return_pct)
        if len(self.pair_performance[pair]) > 200:
            self.pair_performance[pair] = self.pair_performance[pair][-200:]
        
        # update
        self.update_correlation_matrix()
    
    def update_correlation_matrix(self):
        """update"""
        try:
            pairs = list(self.pair_returns_history.keys())
            if len(pairs) < 2:
                return
            
            # translated
            n = len(pairs)
            correlation_matrix = np.zeros((n, n))
            
            for i, pair1 in enumerate(pairs):
                for j, pair2 in enumerate(pairs):
                    if i == j:
                        correlation_matrix[i][j] = 1.0
                    else:
                        returns1 = self.pair_returns_history[pair1]
                        returns2 = self.pair_returns_history[pair2]
                        
                        if len(returns1) >= 20 and len(returns2) >= 20:
                            min_length = min(len(returns1), len(returns2))
                            corr = np.corrcoef(
                                returns1[-min_length:], 
                                returns2[-min_length:]
                            )[0, 1]
                            
                            if not np.isnan(corr):
                                correlation_matrix[i][j] = corr
            
            self.correlation_matrix = correlation_matrix
            self.correlation_pairs = pairs
            
        except Exception as e:
            pass
    
    def get_portfolio_risk_metrics(self) -> Dict[str, float]:
        """calculate risk indicator"""
        try:
            total_var = 0.0
            total_cvar = 0.0
            portfolio_correlation = 0.0
            
            active_pairs = [pair for pair, returns in self.pair_returns_history.items() 
                          if len(returns) >= 20]
            
            if not active_pairs:
                return {
                    'portfolio_var': 0.05,
                    'portfolio_cvar': 0.08,
                    'avg_correlation': 0.0,
                    'diversification_ratio': 1.0
                }
            
            # calculateVaRtranslatedCVaR
            var_values = []
            cvar_values = []
            
            for pair in active_pairs:
                returns = self.pair_returns_history[pair]
                var_values.append(self.calculate_var(returns))
                cvar_values.append(self.calculate_cvar(returns))
            
            total_var = np.mean(var_values)
            total_cvar = np.mean(cvar_values)
            
            # calculate
            correlations = []
            for i, pair1 in enumerate(active_pairs):
                for j, pair2 in enumerate(active_pairs):
                    if i < j:  # avoid calculate
                        corr = self.calculate_portfolio_correlation(pair1)
                        if corr > 0:
                            correlations.append(corr)
            
            portfolio_correlation = np.mean(correlations) if correlations else 0.0
            
            # translated
            diversification_ratio = len(active_pairs) * (1 - portfolio_correlation)
            
            return {
                'portfolio_var': total_var,
                'portfolio_cvar': total_cvar,
                'avg_correlation': portfolio_correlation,
                'diversification_ratio': max(1.0, diversification_ratio)
            }
            
        except Exception as e:
            return {
                'portfolio_var': 0.05,
                'portfolio_cvar': 0.08,
                'avg_correlation': 0.0,
                'diversification_ratio': 1.0
            }
    
    def calculate_leverage(self, market_state: str, volatility: float, pair: str, current_time: datetime = None) -> int:
        """🚀leverage - volatility calculate + risk"""
        
        # === 🎯 get risk level（translated） ===
        try:
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if not dataframe.empty:
                coin_risk_tier = self.identify_coin_risk_tier(pair, dataframe)
            else:
                coin_risk_tier = 'medium_risk'  # default medium risk
        except Exception as e:
            logger.warning(f"get risk level {pair}: {e}")
            coin_risk_tier = 'medium_risk'
        
        # === risk leverage ===
        coin_leverage_limits = {
            'low_risk': (10, 100),      # low risk：10-100translated（not）
            'medium_risk': (5, 50),     # medium risk：5-50translated
            'high_risk': (1, 10)        # high risk（translated）：1-10translated（translated）
        }
        
        # get before leverage
        min_allowed, max_allowed = coin_leverage_limits.get(coin_risk_tier, (5, 50))
        
        # === translated：volatility leverage ===
        volatility_percent = volatility * 100  # as
        
        # leverage（volatility）
        if volatility_percent < 0.5:
            base_leverage = 100  # low volatility = high leverage
        elif volatility_percent < 1.0:
            base_leverage = 75   # low volatility
        elif volatility_percent < 1.5:
            base_leverage = 50   # mid low volatility
        elif volatility_percent < 2.0:
            base_leverage = 30   # mid
        elif volatility_percent < 2.5:
            base_leverage = 20   # mid high volatility
        else:
            base_leverage = 10   # high volatility，leverage
            
        # === winning streak/losing streak ===
        streak_multiplier = 1.0
        if self.consecutive_wins >= 5:
            streak_multiplier = 2.0      # winning streak5times：leverage
        elif self.consecutive_wins >= 3:
            streak_multiplier = 1.5      # winning streak3times：leverage1.5translated
        elif self.consecutive_wins >= 1:
            streak_multiplier = 1.2      # winning streak1times：leverage1.2translated
        elif self.consecutive_losses >= 3:
            streak_multiplier = 0.5      # loss streak3times：leverage
        elif self.consecutive_losses >= 1:
            streak_multiplier = 0.8      # loss streak1times：leverage8translated
            
        # === time optimize ===
        time_multiplier = self.get_time_session_leverage_boost(current_time)
        
        # === market state（simplified） ===
        market_multipliers = {
            "strong_uptrend": 1.3,
            "strong_downtrend": 1.3,
            "mild_uptrend": 1.1,
            "mild_downtrend": 1.1,
            "sideways": 1.0,
            "volatile": 0.8,
            "consolidation": 0.9
        }
        market_multiplier = market_multipliers.get(market_state, 1.0)
        
        # === translated ===
        equity_multiplier = 1.0
        if self.current_drawdown < -0.05:  # drawdown5%
            equity_multiplier = 0.7
        elif self.current_drawdown < -0.02:  # drawdown2%
            equity_multiplier = 0.85
        elif self.current_drawdown == 0:     # none drawdown
            equity_multiplier = 1.2
            
        # === most leverage calculate ===
        calculated_leverage = base_leverage * streak_multiplier * time_multiplier * market_multiplier * equity_multiplier
        
        # original：10-100translated
        pre_risk_leverage = max(10, min(int(calculated_leverage), 100))
        
        # === 🎯 risk leverage（translated） ===
        final_leverage = max(min_allowed, min(pre_risk_leverage, max_allowed))
        
        # === translated ===
        # day loss3%，low leverage
        if hasattr(self, 'daily_loss') and self.daily_loss < -0.03:
            final_leverage = min(final_leverage, 20)
            
        # loss
        if self.consecutive_losses >= 5:
            final_leverage = min(final_leverage, 15)
            
        # risk level
        risk_tier_names = {
            'low_risk': '✅ low risk',
            'medium_risk': '⚡ medium risk', 
            'high_risk': '⚠️ high risk'
        }
        
        logger.info(f"""
⚡ leverage calculate - {pair}:
├─ 🔍 risk level: {risk_tier_names.get(coin_risk_tier, coin_risk_tier)}
├─ 🎯 risk: {min_allowed}-{max_allowed}translated
├─ 📊 volatility: {volatility_percent:.2f}% → leverage: {base_leverage}x
├─ 🏆 winning streak: {self.consecutive_wins}translated{self.consecutive_losses}translated → translated: {streak_multiplier:.1f}x
├─ ⏰ time: {time_multiplier:.1f}x
├─ 📈 translated: {market_multiplier:.1f}x  
├─ 💰 translated: {equity_multiplier:.1f}x
├─ 🧮 calculate leverage: {calculated_leverage:.1f}x
├─ 🔒 leverage: {pre_risk_leverage}x (translated: 10-100x)
└─ 🎉 most leverage: {final_leverage}x ({coin_risk_tier}translated: {min_allowed}-{max_allowed}x)
""")
        
        return final_leverage
    
    def get_time_session_leverage_boost(self, current_time: datetime = None) -> float:
        """get time leverage"""
        if not current_time:
            current_time = datetime.now(timezone.utc)
            
        hour = current_time.hour
        
        # leverage optimize
        if 0 <= hour <= 2:      # translated 00:00-02:00
            return 1.2
        elif 8 <= hour <= 10:   # translated 08:00-10:00
            return 1.3
        elif 14 <= hour <= 16:  # translated 14:00-16:00
            return 1.5          # most high
        elif 20 <= hour <= 22:  # translated 20:00-22:00
            return 1.2
        elif 3 <= hour <= 7:    # translated 03:00-07:00
            return 0.8          # low leverage
        elif 11 <= hour <= 13:  # translated 11:00-13:00
            return 0.9
        else:
            return 1.0          # translated
    
    # translated calculate_dynamic_stoploss - use
    
    def calculate_dynamic_takeprofit(self, pair: str, current_rate: float, trade: Trade, current_profit: float) -> Optional[float]:
        """calculate price"""
        try:
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if dataframe.empty:
                return None
            
            current_data = dataframe.iloc[-1]
            current_atr = current_data.get('atr_p', 0.02)
            adx = current_data.get('adx', 25)
            trend_strength = current_data.get('trend_strength', 50)
            momentum_score = current_data.get('momentum_score', 0)
            
            # translatedATRtranslated
            base_profit_multiplier = 2.5  # ATRtranslated2.5translated
            
            # trend strength
            if abs(trend_strength) > 70:  # strong trend
                trend_multiplier = 1.5
            elif abs(trend_strength) > 40:  # medium trend
                trend_multiplier = 1.2
            else:  # trend
                trend_multiplier = 1.0
            
            # momentum
            momentum_multiplier = 1.0
            if abs(momentum_score) > 0.3:
                momentum_multiplier = 1.3
            elif abs(momentum_score) > 0.1:
                momentum_multiplier = 1.1
            
            # translated
            profit_multiplier = base_profit_multiplier * trend_multiplier * momentum_multiplier
            
            # calculate
            profit_distance = current_atr * profit_multiplier
            
            # range：8%-80%
            profit_distance = max(0.08, min(0.80, profit_distance))
            
            # calculate price
            if trade.is_short:
                target_price = trade.open_rate * (1 - profit_distance)
            else:
                target_price = trade.open_rate * (1 + profit_distance)
            
            logger.info(f"""
🎯 calculate - {pair}:
├─ price: ${trade.open_rate:.6f}
├─ before price: ${current_rate:.6f}
├─ before: {current_profit:.2%}
├─ ATRtranslated: {profit_multiplier:.2f}
├─ translated: {profit_distance:.2%}
├─ price: ${target_price:.6f}
└─ translated: {'bearish' if trade.is_short else 'bullish'}
""")
            
            return target_price
            
        except Exception as e:
            logger.error(f"calculate {pair}: {e}")
            return None
    
    # translated get_smart_trailing_stop - simplified
    
    def validate_and_calibrate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """indicator"""
        try:
            logger.info(f"indicator validation and calibration，translated: {len(dataframe)}")
            
            # === RSI indicator ===
            if 'rsi_14' in dataframe.columns:
                # translatedRSIabnormal value value
                original_rsi_nulls = dataframe['rsi_14'].isnull().sum()
                dataframe['rsi_14'] = dataframe['rsi_14'].clip(0, 100)
                dataframe['rsi_14'] = dataframe['rsi_14'].fillna(50)
                
                # RSItranslated（translated）
                dataframe['rsi_14'] = dataframe['rsi_14'].ewm(span=2).mean()
                
                logger.info(f"RSItranslated - original value: {original_rsi_nulls}, range: 0-100")
            
            # === MACD indicator ===
            if 'macd' in dataframe.columns:
                # MACDindicator
                original_macd_nulls = dataframe['macd'].isnull().sum()
                dataframe['macd'] = dataframe['macd'].fillna(0)
                dataframe['macd'] = dataframe['macd'].ewm(span=3).mean()
                
                if 'macd_signal' in dataframe.columns:
                    dataframe['macd_signal'] = dataframe['macd_signal'].fillna(0)
                    dataframe['macd_signal'] = dataframe['macd_signal'].ewm(span=3).mean()
                
                logger.info(f"MACDtranslated - original value: {original_macd_nulls}, translated3translated")
            
            # === ATR indicator ===
            if 'atr_p' in dataframe.columns:
                # ATRabnormal value
                atr_median = dataframe['atr_p'].median()
                atr_std = dataframe['atr_p'].std()
                
                # translatedATRat range（mid ± 5translated）
                lower_bound = max(0.001, atr_median - 5 * atr_std)
                upper_bound = min(0.5, atr_median + 5 * atr_std)
                
                original_atr_outliers = ((dataframe['atr_p'] < lower_bound) | 
                                       (dataframe['atr_p'] > upper_bound)).sum()
                
                dataframe['atr_p'] = dataframe['atr_p'].clip(lower_bound, upper_bound)
                dataframe['atr_p'] = dataframe['atr_p'].fillna(atr_median)
                
                logger.info(f"ATRtranslated - abnormal value: {original_atr_outliers}, range: {lower_bound:.4f}-{upper_bound:.4f}")
            
            # === ADX indicator ===
            if 'adx' in dataframe.columns:
                dataframe['adx'] = dataframe['adx'].clip(0, 100)
                dataframe['adx'] = dataframe['adx'].fillna(25)  # ADXdefault value25
                logger.info("ADXtranslated - range: 0-100, default value: 25")
            
            # === volume ===
            if 'volume_ratio' in dataframe.columns:
                # volume at range
                dataframe['volume_ratio'] = dataframe['volume_ratio'].clip(0.1, 20)
                dataframe['volume_ratio'] = dataframe['volume_ratio'].fillna(1.0)
                logger.info("volume - range: 0.1-20, default value: 1.0")
            
            # === trend strength ===
            if 'trend_strength' in dataframe.columns:
                dataframe['trend_strength'] = dataframe['trend_strength'].clip(-100, 100)
                dataframe['trend_strength'] = dataframe['trend_strength'].fillna(50)
                logger.info("trend strength - range: -100to100, default value: 50")
            
            # === momentum score ===
            if 'momentum_score' in dataframe.columns:
                dataframe['momentum_score'] = dataframe['momentum_score'].clip(-3, 3)
                dataframe['momentum_score'] = dataframe['momentum_score'].fillna(0)
                logger.info("momentum score - range: -3to3, default value: 0")
            
            # === EMA indicator ===
            # translatedEMAindicator not，original calculate
            for ema_col in ['ema_8', 'ema_21', 'ema_50']:
                if ema_col in dataframe.columns:
                    # abnormal value value，not
                    null_count = dataframe[ema_col].isnull().sum()
                    if null_count > 0:
                        # use before value
                        dataframe[ema_col] = dataframe[ema_col].ffill().bfill()
                        logger.info(f"{ema_col} value - original value: {null_count}")
                    
                    # check has abnormalEMAvalue（price10to up）
                    if 'close' in dataframe.columns:
                        price_ratio = dataframe[ema_col] / dataframe['close']
                        outliers = ((price_ratio > 10) | (price_ratio < 0.1)).sum()
                        if outliers > 0:
                            logger.warning(f"{ema_col} translated {outliers} count abnormal value，recalculate")
                            # recalculateEMA
                            if ema_col == 'ema_8':
                                dataframe[ema_col] = ta.EMA(dataframe, timeperiod=8)
                            elif ema_col == 'ema_21':
                                dataframe[ema_col] = ta.EMA(dataframe, timeperiod=21)
                            elif ema_col == 'ema_50':
                                dataframe[ema_col] = ta.EMA(dataframe, timeperiod=50)
            
            # === indicator check ===
            self._log_indicator_health(dataframe)
            
            return dataframe
            
        except Exception as e:
            logger.error(f"indicator validation and calibration: {e}")
            return dataframe
    
    def _log_indicator_health(self, dataframe: DataFrame):
        """record indicator day"""
        try:
            health_report = []
            
            # check count indicator
            indicators_to_check = ['rsi_14', 'macd', 'atr_p', 'adx', 'volume_ratio', 'trend_strength', 'momentum_score', 'ema_8', 'ema_21', 'ema_50']
            
            for indicator in indicators_to_check:
                if indicator in dataframe.columns:
                    series = dataframe[indicator].dropna()
                    if len(series) > 0:
                        null_count = dataframe[indicator].isnull().sum()
                        null_pct = null_count / len(dataframe) * 100
                        
                        health_status = "translated" if null_pct < 5 else "translated" if null_pct < 15 else "translated"
                        
                        health_report.append(f"├─ {indicator}: {health_status} (value: {null_pct:.1f}%)")
            
            if health_report:
                logger.info(f"""
📊 indicator:
{chr(10).join(health_report)}
└─ translated: {'translated' if all('translated' in line for line in health_report) else 'translated' if any('translated' in line for line in health_report) else 'translated'}
""")
        except Exception as e:
            logger.error(f"indicator check: {e}")
    
    def validate_real_data_quality(self, dataframe: DataFrame, pair: str) -> bool:
        """as"""
        try:
            if len(dataframe) < 10:
                logger.warning(f"not {pair}: {len(dataframe)} translated")
                return False
            
            # check price
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in dataframe.columns:
                    if dataframe[col].isnull().all():
                        logger.error(f"price as value {pair}: {col}")
                        return False
                    
                    # check price has
                    price_std = dataframe[col].std()
                    price_mean = dataframe[col].mean()
                    if price_std / price_mean < 0.001:  # low0.1%
                        logger.warning(f"price abnormal small {pair}: {col} std/mean = {price_std/price_mean:.6f}")
            
            # check volume
            if 'volume' in dataframe.columns:
                if dataframe['volume'].sum() == 0:
                    logger.warning(f"volume as0 {pair}")
                else:
                    # check volume has
                    volume_std = dataframe['volume'].std()
                    volume_mean = dataframe['volume'].mean()
                    if volume_mean > 0 and volume_std / volume_mean < 0.1:
                        logger.warning(f"volume abnormal small {pair}: std/mean = {volume_std/volume_mean:.6f}")
            
            # check time
            if 'date' in dataframe.columns or dataframe.index.name == 'date':
                time_diff = dataframe.index.to_series().diff().dropna()
                if len(time_diff) > 0:
                    # calculate time，use most time as value
                    expected_interval = time_diff.mode().iloc[0] if len(time_diff.mode()) > 0 else pd.Timedelta(minutes=5)
                    abnormal_intervals = (time_diff != expected_interval).sum()
                    if abnormal_intervals > len(time_diff) * 0.1:  # translated10%time abnormal
                        logger.warning(f"time abnormal {pair}: {abnormal_intervals}/{len(time_diff)} count abnormal (translated: {expected_interval})")
            
            logger.info(f"✅ translated {pair}: {len(dataframe)} has")
            return True
            
        except Exception as e:
            logger.error(f"translated {pair}: {e}")
            return False
    
    # translated _log_detailed_exit_decision translated - simplified day
    
    def _log_risk_calculation_details(self, pair: str, input_params: dict, result: dict):
        """record risk calculate"""
        try:
            # day record
            pass
        except Exception as e:
            logger.error(f"risk calculate day record {pair}: {e}")
    
    def _calculate_risk_rating(self, risk_percentage: float) -> str:
        """calculate risk level"""
        try:
            if risk_percentage < 0.01:  # small1%
                return "low risk"
            elif risk_percentage < 0.02:  # 1-2%
                return "mid low risk"
            elif risk_percentage < 0.03:  # 2-3%
                return "medium risk"
            elif risk_percentage < 0.05:  # 3-5%
                return "mid high risk"
            else:  # large5%
                return "high risk"
        except Exception:
            return "risk"
    
    def get_equity_performance_factor(self) -> float:
        """get"""
        if self.initial_balance is None:
            return 1.0
            
        try:
            current_balance = self.wallets.get_total_stake_amount()
            
            if current_balance <= 0:
                return 0.5
                
            # calculate
            returns = (current_balance - self.initial_balance) / self.initial_balance
            
            # update value
            if self.peak_balance is None or current_balance > self.peak_balance:
                self.peak_balance = current_balance
                self.current_drawdown = 0
            else:
                self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            
            # drawdown calculate weight
            if returns > 0.5:  # translated50%
                return 1.5
            elif returns > 0.2:  # translated20-50%
                return 1.3
            elif returns > 0:
                return 1.1
            elif returns > -0.1:
                return 0.9
            elif returns > -0.2:
                return 0.7
            else:
                return 0.5
                
        except Exception:
            return 1.0
    
    def get_streak_factor(self) -> float:
        """get winning streak losing streak"""
        if self.consecutive_wins >= 5:
            return 1.4  # winning streak5times to up，leverage
        elif self.consecutive_wins >= 3:
            return 1.2  # winning streak3-4times
        elif self.consecutive_wins >= 1:
            return 1.1  # winning streak1-2times
        elif self.consecutive_losses >= 5:
            return 0.4  # losing streak5times to up，large low leverage
        elif self.consecutive_losses >= 3:
            return 0.6  # losing streak3-4times
        elif self.consecutive_losses >= 1:
            return 0.8  # losing streak1-2times
        else:
            return 1.0  # has winning streak losing streak record
    
    def get_time_session_factor(self, current_time: datetime) -> float:
        """get weight"""
        if current_time is None:
            return 1.0
            
        # getUTCtime hours
        hour_utc = current_time.hour
        
        # weight
        if 8 <= hour_utc <= 16:  # translated (translated)
            return 1.3
        elif 13 <= hour_utc <= 21:  # translated (most)
            return 1.5
        elif 22 <= hour_utc <= 6:  # translated (translated)
            return 0.8
        else:  # translated
            return 1.0
    
    def get_position_diversity_factor(self) -> float:
        """get"""
        try:
            open_trades = Trade.get_open_trades()
            open_count = len(open_trades)
            
            if open_count == 0:
                return 1.0
            elif open_count <= 2:
                return 1.2  # translated，leverage
            elif open_count <= 5:
                return 1.0  # mid
            elif open_count <= 8:
                return 0.8  # translated，low leverage
            else:
                return 0.6  # translated，large low
                
        except Exception:
            return 1.0
    
    def get_win_rate(self) -> float:
        """get"""
        if len(self.trade_history) < 10:
            return 0.55  # default
            
        wins = sum(1 for trade in self.trade_history if trade.get('profit', 0) > 0)
        return wins / len(self.trade_history)
    
    def get_avg_win_loss_ratio(self) -> float:
        """get"""
        if len(self.trade_history) < 10:
            return 1.5  # default
            
        wins = [trade['profit'] for trade in self.trade_history if trade.get('profit', 0) > 0]
        losses = [abs(trade['profit']) for trade in self.trade_history if trade.get('profit', 0) < 0]
        
        if not wins or not losses:
            return 1.5
            
        avg_win = sum(wins) / len(wins)
        avg_loss = sum(losses) / len(losses)
        
        return avg_win / avg_loss if avg_loss > 0 else 1.5
    
    # translated analyze_multi_timeframe - simplified
    def analyze_multi_timeframe(self, dataframe: DataFrame, metadata: dict) -> Dict:
        """Simplified single timeframe analysis - removed multi-timeframe complexity"""
        
        # Return simple analysis based on current 5m timeframe only
        if dataframe.empty or len(dataframe) < 50:
            return {
                '5m': {
                    'trend': 'unknown',
                    'trend_direction': 'neutral', 
                    'trend_strength': 'unknown',
                    'rsi': 50,
                    'adx': 25
                }
            }
        
        current_data = dataframe.iloc[-1]
        
        # Simple trend analysis using current timeframe
        rsi = current_data.get('rsi_14', 50)
        adx = current_data.get('adx', 25) 
        close = current_data.get('close', 0)
        ema_21 = current_data.get('ema_21', close)
        
        if close > ema_21 and rsi > 50:
            trend_direction = 'bullish'
            trend = 'up'
        elif close < ema_21 and rsi < 50:
            trend_direction = 'bearish' 
            trend = 'down'
        else:
            trend_direction = 'neutral'
            trend = 'sideways'
            
        trend_strength = 'strong' if adx > 25 else 'weak'
        
        return {
            '5m': {
                'trend': trend,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'rsi': rsi,
                'adx': adx,
                'price_position': 0.5,
                'is_top': False,
                'is_bottom': False,
                'momentum': 'neutral',
                'ema_alignment': trend_direction
            }
        }
    
    def get_dataframe_with_indicators(self, pair: str, timeframe: str = None) -> DataFrame:
        """get indicatordataframe"""
        if timeframe is None:
            timeframe = self.timeframe
            
        try:
            # get original
            dataframe = self.dp.get_pair_dataframe(pair, timeframe)
            if dataframe.empty:
                return dataframe
            
            # check calculate indicator
            required_indicators = ['rsi_14', 'adx', 'atr_p', 'macd', 'macd_signal', 'volume_ratio', 'trend_strength', 'momentum_score']
            missing_indicators = [indicator for indicator in required_indicators if indicator not in dataframe.columns]
            
            if missing_indicators:
                # recalculate indicator
                metadata = {'pair': pair}
                dataframe = self.populate_indicators(dataframe, metadata)
                
            return dataframe
            
        except Exception as e:
            logger.error(f"get indicator {pair}: {e}")
            return DataFrame()

    def _safe_series(self, data, length: int, fill_value=0) -> pd.Series:
        """translatedSeries，avoid"""
        if isinstance(data, (int, float)):
            return pd.Series([data] * length, index=range(length))
        elif hasattr(data, '__len__') and len(data) == length:
            return pd.Series(data, index=range(length))
        else:
            return pd.Series([fill_value] * length, index=range(length))
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """optimize indicator - fix indicator calculate"""

        pair = metadata['pair']

        # fix - 1
        if dataframe.index.duplicated().any():
            logger.warning(f"to，at and: {pair}")
            dataframe = dataframe[~dataframe.index.duplicated(keep='first')].reset_index(drop=True)

        # original
        original_index = dataframe.index.copy()
        
        # has indicator calculate
        if len(dataframe) < 50:
            logger.warning(f"long not {pair}: {len(dataframe)} < 50")
            # still calculate indicator，but hasNaNvalue
        
        # translated
        data_quality_ok = self.validate_real_data_quality(dataframe, pair)
        if not data_quality_ok:
            logger.warning(f"translated {pair}, but")
        
        # to indicator calculate
        # cached_indicators = self.get_cached_indicators(pair, len(dataframe))
        # if cached_indicators is not None and len(cached_indicators) == len(dataframe):
        #     # indicator
        #     required_indicators = ['rsi_14', 'adx', 'atr_p', 'macd', 'macd_signal', 'volume_ratio', 'trend_strength', 'momentum_score']
        #     if all(indicator in cached_indicators.columns for indicator in required_indicators):
        #         return cached_indicators
        
        # calculate indicator
        start_time = datetime.now(timezone.utc)
        dataframe = self.calculate_technical_indicators(dataframe)
        
        # recordperformance statistics
        calculation_time = (datetime.now(timezone.utc) - start_time).total_seconds()
        self.calculation_stats['indicator_calls'] += 1
        self.calculation_stats['avg_calculation_time'] = (
            (self.calculation_stats['avg_calculation_time'] * (self.calculation_stats['indicator_calls'] - 1) + 
             calculation_time) / self.calculation_stats['indicator_calls']
        )
        
        # to
        # self.cache_indicators(pair, len(dataframe), dataframe)
        
        # === checkTrading style switch ===
        try:
            self.check_and_switch_trading_style(dataframe)
        except Exception as e:
            logger.warning(f"check: {e}")
        
        # get orderbook data
        pair = metadata['pair']
        try:
            orderbook_data = self.get_market_orderbook(pair)
            if not orderbook_data:
                orderbook_data = {}
        except Exception as e:
            logger.warning(f"get orderbook data {pair}: {e}")
            orderbook_data = {}
        
        # at
        required_ob_fields = {
            'volume_ratio': 1.0,
            'spread_pct': 0.1,
            'depth_imbalance': 0.0,
            'market_quality': 0.5,
            'bid_volume': 0,
            'ask_volume': 0,
            'strong_resistance': 0.0,
            'strong_support': 0.0,
            'large_ask_orders': 0.0,
            'large_bid_orders': 0.0,
            'liquidity_score': 0.5,
            'buy_pressure': 0.5,  # indicator
            'sell_pressure': 0.5   # indicator
        }
        
        # translated，avoidDataFrametranslated
        ob_columns = {}
        for key, default_value in required_ob_fields.items():
            value = orderbook_data.get(key, default_value)
            if isinstance(value, (int, float, np.number)):
                ob_columns[f'ob_{key}'] = value
            else:
                # value，use default value
                ob_columns[f'ob_{key}'] = default_value
        
        # 1 times has，useconcatavoidDataFrametranslated
        if ob_columns:
            ob_df = pd.DataFrame(ob_columns, index=dataframe.index)
            dataframe = pd.concat([dataframe, ob_df], axis=1)
        
        # market state
        if len(dataframe) > 50:
            dataframe['market_state'] = dataframe.apply(
                lambda row: self.detect_market_state(dataframe.loc[:row.name]), axis=1
            )
        else:
            dataframe['market_state'] = 'sideways'
        
        # time - to mid
        mtf_analysis = self.analyze_multi_timeframe(dataframe, metadata)
        
        # will time todataframe
        dataframe = self.apply_mtf_analysis_to_dataframe(dataframe, mtf_analysis, metadata)
        
        # signal strength（translated）
        dataframe['signal_strength'] = self.calculate_enhanced_signal_strength(dataframe)

        # most check
        if dataframe.index.duplicated().any():
            logger.warning(f"most check，at: {pair}")
            dataframe = dataframe[~dataframe.index.duplicated(keep='first')]

        # optimize：translatedDataFrameto avoidPerformanceWarning
        dataframe = dataframe.copy()

        return dataframe
    
    def convert_trend_strength_to_numeric(self, trend_strength):
        """will trend strength as value"""
        if isinstance(trend_strength, (int, float)):
            return trend_strength
        
        strength_mapping = {
            'strong': 80,
            'moderate': 60,
            'weak': 30,
            'reversing': 20,
            'unknown': 0
        }
        
        if isinstance(trend_strength, str):
            return strength_mapping.get(trend_strength.lower(), 0)
        
        return 0
    
    def apply_mtf_analysis_to_dataframe(self, dataframe: DataFrame, mtf_analysis: dict, metadata: dict) -> DataFrame:
        """will time todataframe - translatedMTF"""
        
        # === 1. time trend 1 score ===
        mtf_trend_score = 0
        mtf_strength_score = 0
        mtf_risk_score = 0
        
        # time weight：long weight large
        tf_weights = {'1m': 0.1, '15m': 0.15, '1h': 0.25, '4h': 0.3, '1d': 0.2}
        
        for tf, analysis in mtf_analysis.items():
            if tf in tf_weights and analysis:
                weight = tf_weights[tf]
                
                # trend score
                if analysis.get('trend_direction') == 'bullish':
                    mtf_trend_score += weight * 1
                elif analysis.get('trend_direction') == 'bearish':
                    mtf_trend_score -= weight * 1
                
                # strength score - fix
                trend_strength_raw = analysis.get('trend_strength', 0)
                trend_strength_numeric = self.convert_trend_strength_to_numeric(trend_strength_raw)
                mtf_strength_score += weight * trend_strength_numeric / 100
                
                # risk score（RSIvalue）
                rsi = analysis.get('rsi', 50)
                if rsi > 70:
                    mtf_risk_score += weight * (rsi - 70) / 30  # overbought risk
                elif rsi < 30:
                    mtf_risk_score -= weight * (30 - rsi) / 30  # oversold
        
        # === 2. time ===
        # get1hours4hours price
        h1_data = mtf_analysis.get('1h', {})
        h4_data = mtf_analysis.get('4h', {})
        
        # === 3. time signal ===
        # long-term trend - asSeriestranslated
        mtf_long_condition = (
            (mtf_trend_score > 0.3) &  # time
            (mtf_risk_score > -0.5)    # risk
        )
        
        mtf_short_condition = (
            (mtf_trend_score < -0.3) &  # time
            (mtf_risk_score < 0.5)     # risk
        )
        
        # === 4. time confirm signal ===
        # long confirm：4hours+day
        h4_trend = h4_data.get('trend_direction', 'neutral')
        d1_trend = mtf_analysis.get('1d', {}).get('trend_direction', 'neutral')
        
        mtf_strong_bull_condition = (
            (h4_trend == 'bullish') & (d1_trend == 'bullish') &
            (mtf_strength_score > 0.6)
        )
        
        mtf_strong_bear_condition = (
            (h4_trend == 'bearish') & (d1_trend == 'bearish') &
            (mtf_strength_score > 0.6)
        )
        
        # has time，avoidDataFrametranslated
        h1_support = h1_data.get('support_level', dataframe['close'] * 0.99)
        h1_resistance = h1_data.get('resistance_level', dataframe['close'] * 1.01)
        h4_support = h4_data.get('support_level', dataframe['close'] * 0.98)
        h4_resistance = h4_data.get('resistance_level', dataframe['close'] * 1.02)
        
        mtf_columns = {
            # score indicator
            'mtf_trend_score': mtf_trend_score,  # [-1, 1] trend 1
            'mtf_strength_score': mtf_strength_score,  # [0, 1] trend strength
            'mtf_risk_score': mtf_risk_score,  # [-1, 1] risk/score
            
            # price
            'h1_support': h1_support,
            'h1_resistance': h1_resistance,
            'h4_support': h4_support,
            'h4_resistance': h4_resistance,
            
            # price and
            'near_h1_support': (abs(dataframe['close'] - h1_support) / dataframe['close'] < 0.005).astype(int),
            'near_h1_resistance': (abs(dataframe['close'] - h1_resistance) / dataframe['close'] < 0.005).astype(int),
            'near_h4_support': (abs(dataframe['close'] - h4_support) / dataframe['close'] < 0.01).astype(int),
            'near_h4_resistance': (abs(dataframe['close'] - h4_resistance) / dataframe['close'] < 0.01).astype(int),
            
            # signal
            'mtf_long_filter': self._safe_series(1 if mtf_long_condition else 0, len(dataframe)),
            'mtf_short_filter': self._safe_series(1 if mtf_short_condition else 0, len(dataframe)),
            
            # confirm signal
            'mtf_strong_bull': self._safe_series(1 if mtf_strong_bull_condition else 0, len(dataframe)),
            'mtf_strong_bear': self._safe_series(1 if mtf_strong_bear_condition else 0, len(dataframe))
        }
        
        # 1 times has time，useconcatavoidDataFrametranslated
        if mtf_columns:
            # translatedSeriesvalue
            processed_columns = {}
            for col_name, value in mtf_columns.items():
                if isinstance(value, pd.Series):
                    # translatedSerieslong anddataframetranslated
                    if len(value) == len(dataframe):
                        processed_columns[col_name] = value.values
                    else:
                        processed_columns[col_name] = value
                else:
                    processed_columns[col_name] = value
            
            mtf_df = pd.DataFrame(processed_columns, index=dataframe.index)
            dataframe = pd.concat([dataframe, mtf_df], axis=1)
        
        return dataframe
    
    def calculate_enhanced_signal_strength(self, dataframe: DataFrame) -> pd.Series:
        """calculate signal strength"""
        signal_strength = self._safe_series(0.0, len(dataframe))
        
        # 1. indicator signal (40%weight)
        traditional_signals = self.calculate_traditional_signals(dataframe) * 0.4
        
        # 2. momentum signal (25%weight)
        momentum_signals = self._safe_series(0.0, len(dataframe))
        if 'momentum_score' in dataframe.columns:
            momentum_signals = dataframe['momentum_score'] * 2.5 * 0.25  # large to[-2.5, 2.5]
        
        # 3. trend strength signal (20%weight)
        trend_signals = self._safe_series(0.0, len(dataframe))
        if 'trend_strength_score' in dataframe.columns:
            trend_signals = dataframe['trend_strength_score'] * 2 * 0.2  # large to[-2, 2]
        
        # 4. signal (15%weight)
        health_signals = self._safe_series(0.0, len(dataframe))
        if 'technical_health' in dataframe.columns:
            health_signals = dataframe['technical_health'] * 1.5 * 0.15  # large to[-1.5, 1.5]
        
        # signal strength
        signal_strength = traditional_signals + momentum_signals + trend_signals + health_signals
        
        return signal_strength.fillna(0).clip(-10, 10)  # at[-10, 10]range
    
    def calculate_traditional_signals(self, dataframe: DataFrame) -> pd.Series:
        """calculate indicator signal"""
        signals = self._safe_series(0.0, len(dataframe))
        
        # RSI signal (-3 to +3)
        rsi_signals = self._safe_series(0.0, len(dataframe))
        if 'rsi_14' in dataframe.columns:
            rsi_signals[dataframe['rsi_14'] < 30] = 2
            rsi_signals[dataframe['rsi_14'] > 70] = -2
            rsi_signals[(dataframe['rsi_14'] > 40) & (dataframe['rsi_14'] < 60)] = 1
        
        # MACD signal (-2 to +2)
        macd_signals = self._safe_series(0.0, len(dataframe))
        if 'macd' in dataframe.columns and 'macd_signal' in dataframe.columns:
            macd_signals = ((dataframe['macd'] > dataframe['macd_signal']).astype(int) * 2 - 1)
            if 'macd_hist' in dataframe.columns:
                macd_hist_signals = (dataframe['macd_hist'] > 0).astype(int) * 2 - 1
                macd_signals = (macd_signals + macd_hist_signals) / 2
        
        # trend EMA signal (-3 to +3)
        ema_signals = self._safe_series(0.0, len(dataframe))
        if all(col in dataframe.columns for col in ['ema_8', 'ema_21', 'ema_50']):
            bullish_ema = ((dataframe['ema_8'] > dataframe['ema_21']) & 
                          (dataframe['ema_21'] > dataframe['ema_50']))
            bearish_ema = ((dataframe['ema_8'] < dataframe['ema_21']) & 
                          (dataframe['ema_21'] < dataframe['ema_50']))
            ema_signals[bullish_ema] = 3
            ema_signals[bearish_ema] = -3
        
        # volume signal (-1 to +2)
        volume_signals = self._safe_series(0.0, len(dataframe))
        if 'volume_ratio' in dataframe.columns:
            volume_signals[dataframe['volume_ratio'] > 1.5] = 2
            volume_signals[dataframe['volume_ratio'] < 0.7] = -1
        
        # ADX trend strength signal (0 to +2)
        adx_signals = self._safe_series(0.0, len(dataframe))
        if 'adx' in dataframe.columns:
            adx_signals[dataframe['adx'] > 25] = 1
            adx_signals[dataframe['adx'] > 40] = 2
        
        # high indicator signal
        advanced_signals = self._safe_series(0.0, len(dataframe))
        
        # Fisher Transform signal
        if 'fisher' in dataframe.columns and 'fisher_signal' in dataframe.columns:
            fisher_cross_up = ((dataframe['fisher'] > dataframe['fisher_signal']) & 
                              (dataframe['fisher'].shift(1) <= dataframe['fisher_signal'].shift(1)))
            fisher_cross_down = ((dataframe['fisher'] < dataframe['fisher_signal']) & 
                                (dataframe['fisher'].shift(1) >= dataframe['fisher_signal'].shift(1)))
            advanced_signals[fisher_cross_up] += 1.5
            advanced_signals[fisher_cross_down] -= 1.5
        
        # KST signal
        if 'kst' in dataframe.columns and 'kst_signal' in dataframe.columns:
            kst_bullish = dataframe['kst'] > dataframe['kst_signal']
            advanced_signals[kst_bullish] += 1
            advanced_signals[~kst_bullish] -= 1
        
        # MFI signal
        if 'mfi' in dataframe.columns:
            advanced_signals[dataframe['mfi'] < 30] += 1  # oversold
            advanced_signals[dataframe['mfi'] > 70] -= 1  # overbought
        
        # signal
        total_signals = (rsi_signals + macd_signals + ema_signals + 
                        volume_signals + adx_signals + advanced_signals)
        
        return total_signals.fillna(0).clip(-10, 10)
    
    def _calculate_signal_quality(self, dataframe: DataFrame) -> pd.Series:
        """calculate signal score"""
        quality_score = self._safe_series(0.5, len(dataframe))  # default mid
        
        # signal strength 1 calculate
        if 'signal_strength' in dataframe.columns:
            # signal strength value large high
            abs_strength = abs(dataframe['signal_strength'])
            quality_score = abs_strength / 10.0  # to0-1
        
        # indicator 1
        consistency_factors = []
        
        # RSI1
        if 'rsi_14' in dataframe.columns:
            rsi_consistency = 1 - abs(dataframe['rsi_14'] - 50) / 50  # 0-1
            consistency_factors.append(rsi_consistency)
        
        # MACD1
        if 'macd' in dataframe.columns and 'macd_signal' in dataframe.columns:
            macd_diff = abs(dataframe['macd'] - dataframe['macd_signal'])
            macd_consistency = 1 / (1 + macd_diff)  # 0-1
            consistency_factors.append(macd_consistency)
        
        # trend strength 1
        if 'trend_strength' in dataframe.columns:
            trend_consistency = abs(dataframe['trend_strength']) / 100  # 0-1
            consistency_factors.append(trend_consistency)
        
        # volume confirmation
        if 'volume_ratio' in dataframe.columns:
            volume_quality = np.minimum(dataframe['volume_ratio'] / 2, 1.0)  # 0-1
            consistency_factors.append(volume_quality)
        
        # score
        if consistency_factors:
            avg_consistency = np.mean(consistency_factors, axis=0)
            quality_score = (quality_score + avg_consistency) / 2
        
        return quality_score.fillna(0.5).clip(0, 1)
    
    def _calculate_position_weight(self, dataframe: DataFrame) -> pd.Series:
        """calculate position size weight"""
        base_weight = self._safe_series(1.0, len(dataframe))  # weight100%
        
        # signal weight
        if 'signal_quality_score' in dataframe.columns:
            quality_multiplier = 0.5 + dataframe['signal_quality_score'] * 1.5  # 0.5-2.0translated
            base_weight = base_weight * quality_multiplier
        
        # volatility
        if 'atr_p' in dataframe.columns:
            # high volatility low weight
            volatility_factor = 1 / (1 + dataframe['atr_p'] * 10)  # 0.09-1.0
            base_weight = base_weight * volatility_factor
        
        # trend strength
        if 'trend_strength' in dataframe.columns:
            trend_factor = 0.8 + abs(dataframe['trend_strength']) / 500  # 0.8-1.0
            base_weight = base_weight * trend_factor
        
        return base_weight.fillna(1.0).clip(0.1, 3.0)  # 10%-300%
    
    def _calculate_leverage_multiplier(self, dataframe: DataFrame) -> pd.Series:
        """calculate leverage"""
        base_leverage = self._safe_series(1.0, len(dataframe))  # translated1leverage
        
        # signal leverage
        if 'signal_quality_score' in dataframe.columns:
            # high signal to use high leverage
            quality_leverage = 1.0 + dataframe['signal_quality_score'] * 2.0  # 1.0-3.0translated
            base_leverage = base_leverage * quality_leverage
        
        # volatility leverage
        if 'atr_p' in dataframe.columns:
            # high volatility use low leverage
            volatility_factor = 1 / (1 + dataframe['atr_p'] * 5)  # 0.17-1.0
            base_leverage = base_leverage * volatility_factor
        
        # translatedADXtrend strength
        if 'adx' in dataframe.columns:
            # strong trend to use high leverage
            adx_factor = 1.0 + (dataframe['adx'] - 25) / 100  # 0.75-1.75
            adx_factor = np.maximum(adx_factor, 0.5)  # most low0.5translated
            base_leverage = base_leverage * adx_factor
        
        return base_leverage.fillna(1.0).clip(0.5, 5.0)  # 0.5-5leverage
    
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """translated - translated"""
        
        pair = metadata['pair']
        
        # === translated ===
        # calculate price（20translatedKtranslated）
        highest_20 = dataframe['high'].rolling(20).max()
        lowest_20 = dataframe['low'].rolling(20).min()
        price_position = (dataframe['close'] - lowest_20) / (highest_20 - lowest_20 + 0.0001)
        
        # 🚨 fix：price - avoid
        not_at_top = price_position < 0.80  # to80%，allow at high
        # at  
        not_at_bottom = price_position > 0.20  # to20%，allow at low
        
        # === momentum（translated）===
        # translatedRSImomentum（translated）
        rsi_momentum_strong = (
            (dataframe['rsi_14'] - dataframe['rsi_14'].shift(3) > -10) &  # translatedRSIdown
            (dataframe['rsi_14'] < 80) & (dataframe['rsi_14'] > 20)  # translatedRSIvalue range
        )
        
        # volume support（translated）
        volume_support = (
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 0.6) &  # volume
            (dataframe['volume'] > dataframe['volume'].shift(1) * 0.7)  # volume
        )
        
        # simplified breakout（translated）
        no_fake_breakout = ~(
            # long（translated）
            ((dataframe['high'] - dataframe['close']) > (dataframe['close'] - dataframe['open']) * 3) |  # high to3translated
            ((dataframe['open'] - dataframe['low']) > (dataframe['close'] - dataframe['open']) * 3)       # high to3translated
            # translated - translated
        )
        
        # translated（ADX < 20 none trend）
        is_trending = dataframe['adx'] > 20
        is_sideways = dataframe['adx'] < 20
        
        # translated（translated）
        sideways_filter = ~is_sideways | (dataframe['atr_p'] > 0.02)  # large
        
        # translated
        basic_env = (
            (dataframe['volume_ratio'] > 0.8) &  # volume not low
            (dataframe['atr_p'] > 0.001) &       # volatility
            sideways_filter &                     # translated
            rsi_momentum_strong &                # RSImomentum
            volume_support                       # volume support
        )
        
        # 🚨 fix：translated（before60+signal）
        # long has：trend not + not
        long_favourable_environment = (
            basic_env &  # translated
            (dataframe['trend_strength'] > -40) &  # trend not（translated）
            (dataframe.get('market_sentiment', 0) > -0.8) &  # not（translated）
            (dataframe['rsi_14'] > 25)  # RSInot at oversold（avoid）
        )
        
        # short has：trend not + not  
        short_favourable_environment = (
            basic_env &  # translated
            (dataframe['trend_strength'] < 40) &   # trend not（translated）
            (dataframe.get('market_sentiment', 0) < 0.8) &   # not（translated）
            (dataframe['rsi_14'] < 75)  # RSInot at overbought（avoid at short）
        )
        
        # === 🌍 market state ===
        market_regime_data = self._enhanced_market_regime_detection(dataframe)
        current_regime = market_regime_data['regime']
        regime_confidence = market_regime_data['confidence']
        signals_advice = market_regime_data['signals_advice']
        
        # record market state todataframe（after）
        dataframe.loc[:, 'market_regime'] = current_regime
        dataframe.loc[:, 'regime_confidence'] = regime_confidence
        
        logger.info(
            f"📊 market state {metadata.get('pair', '')}: "
            f"{current_regime} (translated:{regime_confidence:.1%}) | "
            f"signal:{signals_advice.get('recommended_signals', [])} | "
            f"avoid signal:{signals_advice.get('avoid_signals', [])}"
        )
        
        # === 💰 signal ===
        
        # 🎯 Signal 1: RSIoversold（translated）
        # === translatedRSIvalue calculate ===
        # volatilityRSIvalue，high volatility avoid signal
        base_oversold = 30
        volatility_percentile = dataframe['atr_p'].rolling(50).rank(pct=True)
        dynamic_oversold = base_oversold - (volatility_percentile * 8)  # 20-30range
        
        # === confirm ===
        rsi_condition = (dataframe['rsi_14'] < dynamic_oversold)
        rsi_momentum = (dataframe['rsi_14'] > dataframe['rsi_14'].shift(2))  # translated2rise
        price_confirmation = (dataframe['close'] > dataframe['close'].shift(1))
        
        # === trend confirmation：at rise trend or mid long ===
        trend_confirmation = (
            (dataframe['ema_8'] >= dataframe['ema_21']) |  # bullish
            (dataframe['adx'] < 25)  # or
        )
        
        # === volume confirmation：breakout volume support ===
        volume_confirmation = (
            dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.1
        )
        
        # === strength confirm：ADXtrend ===
        strength_confirmation = (
            (dataframe['adx'] > 20) &  # most low strength
            (dataframe['adx'] > dataframe['adx'].shift(2))  # ADXrise
        )
        
        # === divergence：avoid at bearish divergence ===
        no_bearish_divergence = ~dataframe.get('bearish_divergence', False).astype(bool)
        
        rsi_oversold_bounce = (
            rsi_condition &
            rsi_momentum &
            price_confirmation &
            trend_confirmation &
            volume_confirmation &
            strength_confirmation &
            no_bearish_divergence &
            not_at_top &  # at
            basic_env
        )
        dataframe.loc[rsi_oversold_bounce, 'enter_long'] = 1
        dataframe.loc[rsi_oversold_bounce, 'enter_tag'] = 'RSI_Oversold_Bounce'
        
        # 🎯 Signal 2: EMAafter（translated）
        ema_golden_cross = (
            (dataframe['ema_8'] > dataframe['ema_21']) &     # translated
            (dataframe['ema_8'].shift(3) <= dataframe['ema_21'].shift(3)) &  # 3translatedKbefore
            (dataframe['close'] <= dataframe['ema_8'] * 1.01) &  # price toEMA8translated
            (dataframe['close'] > dataframe['ema_21']) &     # but still atEMA21up
            (dataframe['volume_ratio'] > 1.0) &              # volume
            # new：momentum
            (dataframe['momentum_exhaustion_score'] < 0.5) &  # momentum
            (dataframe['trend_phase'] <= 2) &  # not at trend late stage
            (~dataframe['bearish_divergence'].astype(bool)) &  # none bearish divergence
            basic_env
        )
        dataframe.loc[ema_golden_cross, 'enter_long'] = 1
        dataframe.loc[ema_golden_cross, 'enter_tag'] = 'EMA_Golden_Cross'
        
        # 🎯 Signal 3: MACDup breakout（fix：translated）
        macd_bullish = (
            (
                # MACDtranslated - at trend signal
                ((dataframe['macd'] > dataframe['macd_signal']) & 
                 (dataframe['macd'].shift(1) <= dataframe['macd_signal'].shift(1))) |
                # or from（confirm）
                ((dataframe['macd_hist'] > 0) & 
                 (dataframe['macd_hist'].shift(1) <= 0))
            ) &
            basic_env
        )
        dataframe.loc[macd_bullish, 'enter_long'] = 1
        dataframe.loc[macd_bullish, 'enter_tag'] = 'MACD_Bullish'
        
        # 🎯 Signal 4: Bollinger Bands down（confirm）
        bb_lower_bounce = (
            (dataframe['close'] <= dataframe['bb_lower'] * 1.005) &  # down
            (dataframe['close'] > dataframe['close'].shift(1)) &     # price
            (dataframe['close'].shift(1) > dataframe['close'].shift(2)) &  # 2 times confirm：translated
            (dataframe['rsi_14'] < 50) &                             # RSIlow
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(1)) &  # RSIrise
            (dataframe['volume_ratio'] > 1.1) &                     # volume
            not_at_top &  # high
            no_fake_breakout &  # none breakout risk
            basic_env
        )
        dataframe.loc[bb_lower_bounce, 'enter_long'] = 1
        dataframe.loc[bb_lower_bounce, 'enter_tag'] = 'BB_Lower_Bounce'
        
        # Signal 5 translated - Simple_Breakoutbreakout signal
        
        # === 📉 simplified short signal ===
        
        # 🎯 Signal 1: RSIoverbought（translated）
        # === translatedRSIvalue calculate ===
        # volatilityRSIvalue，high volatility avoid signal
        base_overbought = 70
        volatility_percentile = dataframe['atr_p'].rolling(50).rank(pct=True)
        dynamic_overbought = base_overbought + (volatility_percentile * 8)  # 70-78range
        
        # === confirm ===
        rsi_condition = (dataframe['rsi_14'] > dynamic_overbought)
        rsi_momentum = (dataframe['rsi_14'] < dataframe['rsi_14'].shift(2))  # translated2fall
        price_confirmation = (dataframe['close'] < dataframe['close'].shift(1))
        
        # === trend confirmation：at fall trend or mid short ===
        trend_confirmation = (
            (dataframe['ema_8'] <= dataframe['ema_21']) |  # bearish
            (dataframe['adx'] < 25)  # or
        )
        
        # === volume confirmation：breakout volume support ===
        volume_confirmation = (
            dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.1
        )
        
        # === strength confirm：ADXtrend ===
        strength_confirmation = (
            (dataframe['adx'] > 20) &  # most low strength
            (dataframe['adx'] > dataframe['adx'].shift(2))  # ADXrise
        )
        
        # === divergence：avoid at bullish divergence ===
        no_bullish_divergence = ~dataframe.get('bullish_divergence', False).astype(bool)
        
        rsi_overbought_fall = (
            rsi_condition &
            rsi_momentum &
            price_confirmation &
            trend_confirmation &
            volume_confirmation &
            strength_confirmation &
            no_bullish_divergence &
            not_at_bottom &  # at
            basic_env
        )
        # === 📊 signal score ===
        rsi_long_score = self._calculate_signal_quality_score(
            dataframe, rsi_oversold_bounce, 'RSI_Oversold_Bounce'
        )
        rsi_short_score = self._calculate_signal_quality_score(
            dataframe, rsi_overbought_fall, 'RSI_Overbought_Fall'
        )
        
        # === 📊 market state signal ===
        # has high+market state signal
        
        # RSIlong signal
        rsi_long_regime_ok = 'RSI_Oversold_Bounce' not in signals_advice.get('avoid_signals', [])
        high_quality_long = rsi_oversold_bounce & (rsi_long_score >= 6) & rsi_long_regime_ok
        
        # RSIshort signal  
        rsi_short_regime_ok = 'RSI_Overbought_Fall' not in signals_advice.get('avoid_signals', [])
        high_quality_short = rsi_overbought_fall & (rsi_short_score >= 6) & rsi_short_regime_ok
        
        # market state：at mid low
        if 'RSI_Oversold_Bounce' in signals_advice.get('recommended_signals', []):
            regime_bonus_long = rsi_oversold_bounce & (rsi_long_score >= 5)  # low1translated
            high_quality_long = high_quality_long | regime_bonus_long
            
        if 'RSI_Overbought_Fall' in signals_advice.get('recommended_signals', []):
            regime_bonus_short = rsi_overbought_fall & (rsi_short_score >= 5)  # low1translated  
            high_quality_short = high_quality_short | regime_bonus_short
        
        dataframe.loc[high_quality_long, 'enter_long'] = 1
        dataframe.loc[high_quality_long, 'enter_tag'] = 'RSI_Oversold_Bounce'
        dataframe.loc[high_quality_long, 'signal_quality'] = rsi_long_score
        dataframe.loc[high_quality_long, 'market_regime_bonus'] = 'RSI_Oversold_Bounce' in signals_advice.get('recommended_signals', [])
        
        dataframe.loc[high_quality_short, 'enter_short'] = 1
        dataframe.loc[high_quality_short, 'enter_tag'] = 'RSI_Overbought_Fall'
        dataframe.loc[high_quality_short, 'signal_quality'] = rsi_short_score
        dataframe.loc[high_quality_short, 'market_regime_bonus'] = 'RSI_Overbought_Fall' in signals_advice.get('recommended_signals', [])
        
        # 🎯 Signal 2: EMAafter（translated）
        ema_death_cross = (
            (dataframe['ema_8'] < dataframe['ema_21']) &     # translated
            (dataframe['ema_8'].shift(3) >= dataframe['ema_21'].shift(3)) &  # 3translatedKbefore
            (dataframe['close'] >= dataframe['ema_8'] * 0.99) &  # price toEMA8translated
            (dataframe['close'] < dataframe['ema_21']) &     # but still atEMA21down
            (dataframe['volume_ratio'] > 1.0) &              # volume
            # new：momentum
            (dataframe['momentum_exhaustion_score'] < 0.5) &  # momentum
            (dataframe['trend_phase'] <= 2) &  # not at trend late stage
            (~dataframe['bullish_divergence'].astype(bool)) &  # none bullish divergence
            basic_env
        )
        dataframe.loc[ema_death_cross, 'enter_short'] = 1
        dataframe.loc[ema_death_cross, 'enter_tag'] = 'EMA_Death_Cross'
        
        # 🎯 Signal 3: MACDsignal（translated）
        # === MACDsignal ===
        macd_death_cross = (
            (dataframe['macd'] < dataframe['macd_signal']) & 
            (dataframe['macd'].shift(1) >= dataframe['macd_signal'].shift(1))
        )
        macd_hist_negative = (
            (dataframe['macd_hist'] < 0) & 
            (dataframe['macd_hist'].shift(1) >= 0)
        )
        macd_basic_signal = macd_death_cross | macd_hist_negative
        
        # === 🛡️ translated - signal ===
        
        # 1. trend confirm：avoid at rise trend mid short
        trend_bearish = (
            (dataframe['ema_8'] < dataframe['ema_21']) &  # EMAbearish
            (dataframe['ema_21'] < dataframe['ema_50']) & # mid long-term trend down
            (dataframe['close'] < dataframe['ema_21'])     # price at trend down
        )
        
        # 2. momentum confirm：down momentum at
        momentum_confirmation = (
            (dataframe['rsi_14'] < 55) &                  # RSItranslated
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(2)) &  # RSIdown
            (dataframe['close'] < dataframe['close'].shift(2))      # price down
        )
        
        # 3. volume confirmation：down volume support
        volume_confirmation = (
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.1) &
            (dataframe['volume'] > dataframe['volume'].shift(1))  # volume
        )
        
        # 4. strength confirm：ADXtrend
        strength_confirmation = (
            (dataframe['adx'] > 25) &                     # has 1 trend strength
            (dataframe['adx'] > dataframe['adx'].shift(3)) # ADXrise trend
        )
        
        # 5. translated：avoid at mid
        not_sideways = (dataframe['adx'] > 20)            # not at
        
        # 6. confirm：at high short
        position_confirmation = (
            dataframe['close'] > dataframe['close'].rolling(20).mean() * 1.02  # price high
        )
        
        # 7. divergence：avoid at bullish divergence short
        no_bullish_divergence = ~dataframe.get('bullish_divergence', False).astype(bool)
        
        # === mostMACDsignal ===
        macd_bearish = (
            macd_basic_signal &
            trend_bearish &
            momentum_confirmation &
            volume_confirmation &
            strength_confirmation &
            not_sideways &
            position_confirmation &
            no_bullish_divergence &
            not_at_bottom &  # at
            basic_env
        )
        
        # === 📊 MACDsignal score ===
        macd_score = self._calculate_macd_signal_quality(dataframe, macd_bearish, 'MACD_Bearish')
        
        # === 📊 MACDmarket state ===
        # MACDsignal market state confirm
        macd_regime_ok = 'MACD_Bearish' not in signals_advice.get('avoid_signals', [])
        high_quality_macd = macd_bearish & (macd_score >= 7) & macd_regime_ok  # MACDhigh+confirm
        
        # market state：at down trend mid lowMACDtranslated
        if 'MACD_Bearish' in signals_advice.get('recommended_signals', []):
            regime_bonus_macd = macd_bearish & (macd_score >= 6) & macd_regime_ok  # low1translated
            high_quality_macd = high_quality_macd | regime_bonus_macd
        
        dataframe.loc[high_quality_macd, 'enter_short'] = 1
        dataframe.loc[high_quality_macd, 'enter_tag'] = 'MACD_Bearish'
        dataframe.loc[high_quality_macd, 'signal_quality'] = macd_score
        dataframe.loc[high_quality_macd, 'market_regime_bonus'] = 'MACD_Bearish' in signals_advice.get('recommended_signals', [])
        
        # 🎯 Signal 4: Bollinger Bands up
        bb_upper_rejection = (
            (dataframe['close'] >= dataframe['bb_upper'] * 0.995) &  # up
            (dataframe['close'] < dataframe['close'].shift(1)) &     # price
            (dataframe['rsi_14'] > 50) &                             # RSIhigh
            (dataframe['volume_ratio'] > 1.1) &                     # volume
            basic_env
        )
        dataframe.loc[bb_upper_rejection, 'enter_short'] = 1
        dataframe.loc[bb_upper_rejection, 'enter_tag'] = 'BB_Upper_Rejection'
        
        # Signal 5 translated - Simple_Breakdownbreakout signal
        
        # ==============================
        # 🚨 new：position size weight - signal
        # ==============================
        
        # 1. signal score
        dataframe['signal_quality_score'] = self._calculate_signal_quality(dataframe)
        dataframe['position_weight'] = self._calculate_position_weight(dataframe)
        dataframe['leverage_multiplier'] = self._calculate_leverage_multiplier(dataframe)
        
        # signal
        total_long_signals = dataframe['enter_long'].sum()
        total_short_signals = dataframe['enter_short'].sum()
        
        # translated
        env_basic_rate = basic_env.sum() / len(dataframe) * 100
        env_long_rate = long_favourable_environment.sum() / len(dataframe) * 100  
        env_short_rate = short_favourable_environment.sum() / len(dataframe) * 100
        
        # has signal
        if total_long_signals > 0 or total_short_signals > 0:
            logger.info(f"""
🔥 fix - {metadata['pair']}:
📊 signal:
   └─ long signal: {total_long_signals} count
   └─ short signal: {total_short_signals} count
   └─ signal: {total_long_signals + total_short_signals} count

🌍 translated:
   └─ translated: {env_basic_rate:.1f}%
   └─ long: {env_long_rate:.1f}%  
   └─ short: {env_short_rate:.1f}%

✅ fix: translated，60+signal new！
""")
        
        # has signal，translated
        if total_long_signals == 0 and total_short_signals == 0:
            logger.warning(f"""
⚠️  none signal - {metadata['pair']}:
🔍 translatedReason:
   └─ translated: {100-env_basic_rate:.1f}% Ktranslated
   └─ long: {100-env_long_rate:.1f}% Knot long
   └─ short: {100-env_short_rate:.1f}% Knot short
   
💡 translated: checkRSI({dataframe['rsi_14'].iloc[-1]:.1f}), trend strength({dataframe.get('trend_strength', [0]).iloc[-1]:.1f})
""")
        
        return dataframe
    
    def _legacy_populate_entry_trend_backup(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """old signal（translated）"""
        
        # 0A. price as - most fast reversal signal
        price_action_bottom = (
            # translatedPin Bar：long down reversal signal
            (dataframe['is_pin_bar_bullish'] == 1) &
            # or：reversal
            ((dataframe['is_bullish_engulfing'] == 1) | 
             # StochRSIoversold after
             ((dataframe['stoch_rsi_k'] < 20) & (dataframe['stoch_rsi_k'] > dataframe['stoch_rsi_k'].shift(1)))) &
            # Williams indicator confirm reversal
            (dataframe['williams_r'] < -70) & (dataframe['williams_r'] > dataframe['williams_r'].shift(1)) &
            # CCIfrom oversold
            (dataframe['cci'] < -100) & (dataframe['cci'] > dataframe['cci'].shift(1)) &
            # volume abnormal large（translated）
            (dataframe['volume_spike'] == 1) &
            # price support（not）
            (dataframe['support_strength'] < 8) &
            # price：use predictive，not
            (price_percentile_20 > 0.15) & in_favorable_long_zone &
            # time confirm：long-term trend
            (dataframe['mtf_long_filter'] == 1) &
            long_favourable_environment
        )
        dataframe.loc[price_action_bottom, 'enter_long'] = 1
        dataframe.loc[price_action_bottom, 'enter_tag'] = 'price as'
        
        # 0B. [translated] reversal - translated
        acceleration_reversal = (
            False &  # count signal
            # price down（signal）
            (dataframe['price_velocity'] < 0) &  # still at down
            (dataframe['price_acceleration'] > 0) &  # but down at
            (dataframe['price_acceleration'] > dataframe['price_acceleration'].shift(1)) &  # fast
            # RSIdivergence：price new low butRSInew low
            (dataframe['rsi_divergence_strength'] > 0) &
            # StochRSIfast from low
            (dataframe['stoch_rsi_k'] < 30) & 
            (dataframe['stoch_rsi_k'] > dataframe['stoch_rsi_k'].shift(1) + 3) &
            # volume abnormal（after）
            (dataframe['volume_ratio'] > 1.3) &
            # price：predictive most
            (price_percentile_20 > 0.20) & in_favorable_long_zone &
            # time confirm
            (dataframe['mtf_long_filter'] == 1) &
            long_favourable_environment
        )
        dataframe.loc[acceleration_reversal, 'enter_long'] = 1
        dataframe.loc[acceleration_reversal, 'enter_tag'] = 'reversal'
        
        # === 🚀 short signal (time: 10translated-30minutes) ===
        
        # 1. short momentum - translated（avoid）
        ultra_short_momentum = (
            # EMA5fast upEMA8but not at high
            (dataframe['ema_5'] > dataframe['ema_8']) & 
            (dataframe['ema_5'].shift(1) <= dataframe['ema_8'].shift(1)) &
            # RSIfrom low rise，but high（Hyperoptoptimize）
            (dataframe['rsi_14'] > self.rsi_buy_min.value) & (dataframe['rsi_14'] < self.rsi_buy_max.value) &
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(2) + 5) &
            (dataframe['rsi_14'].shift(2) < self.rsi_buy_max.value) &  # from low
            # volume but not（avoid）
            (dataframe['volume_ratio'] > self.volume_spike_min.value) & (dataframe['volume_ratio'] < self.volume_spike_max.value) &
            # price breakout but has before
            (dataframe['close'] > dataframe['high'].rolling(5).max().shift(1)) &
            # not at high（breakout）
            (dataframe['close'] < dataframe['high'].rolling(20).max() * 0.95) &
            # trend confirmation：has
            (dataframe['ema_8'] > dataframe['ema_21']) &  # mid stage trend up
            # translated + predictive
            not_chasing_high & long_favourable_environment &
            in_favorable_long_zone  # use predictive most
        )
        dataframe.loc[ultra_short_momentum, 'enter_long'] = 1
        dataframe.loc[ultra_short_momentum, 'enter_tag'] = 'short momentum'
        
        # 2. translated - low
        scalping_opportunity = (
            # price at Bollinger Bands mid down（low）
            (dataframe['bb_position'] > 0.3) & (dataframe['bb_position'] < 0.55) &
            # MACDand has
            (dataframe['macd_hist'] > 0) & (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)) &
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(2)) &  # translated
            # volume mid but not
            (dataframe['volume_ratio'] > 1.1) & (dataframe['volume_ratio'] < 2.0) &
            # RSIfrom low，but avoid high
            (dataframe['rsi_14'] > 48) & (dataframe['rsi_14'] < 62) &
            (dataframe['rsi_14'].shift(3) < 55) &  # from low
            # trend strength mid and up
            (dataframe['trend_strength'] > 5) & (dataframe['trend_strength'] < 25) &
            (dataframe['trend_strength'] > dataframe['trend_strength'].shift(2)) &  # trend
            # price：predictive most
            in_favorable_long_zone &  # use predictive
            long_favourable_environment
        )
        dataframe.loc[scalping_opportunity, 'enter_long'] = 1
        dataframe.loc[scalping_opportunity, 'enter_tag'] = 'long'
        
        # === 📈 mid signal (time: 30minutes-4hours) ===
        
        # 3. confirm - but predictive
        golden_cross_confirmed = (
            # EMA8upEMA21，but predictive
            (dataframe['ema_8'] > dataframe['ema_21']) & 
            (dataframe['ema_8'].shift(1) <= dataframe['ema_21'].shift(1)) &
            # translatedEMA5atEMA8up
            (dataframe['ema_5'] > dataframe['ema_8']) &
            # MACDconfirm and has
            (dataframe['macd'] > dataframe['macd_signal']) &
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)) &
            (dataframe['macd'] > dataframe['macd'].shift(2)) &  # MACDrise
            # RSIrise but not high
            (dataframe['rsi_14'] > 40) & (dataframe['rsi_14'] < 65) &
            (dataframe['rsi_14'].shift(5) < 50) &  # from low
            # ADXtrend and has before
            (dataframe['adx'] > 20) & (dataframe['adx'] > dataframe['adx'].shift(2)) &
            (dataframe['adx'] < 45) &  # avoid trend（reversal）
            # volume confirmation but
            (dataframe['volume_ratio'] > 1.2) & (dataframe['volume_ratio'] < 3.0) &
            # price：predictive
            in_favorable_long_zone &  # predictive most
            # has support confirm
            (dataframe['close'] > dataframe['ema_34']) &  # at long up
            # translated
            not_chasing_high & long_favourable_environment
        )
        dataframe.loc[golden_cross_confirmed, 'enter_long'] = 1
        dataframe.loc[golden_cross_confirmed, 'enter_tag'] = 'confirm'
        
        # 4. support - translated
        support_bounce = (
            # priceEMA34support
            (dataframe['close'] > dataframe['ema_34'] * 0.995) & 
            (dataframe['close'] < dataframe['ema_34'] * 1.01) &
            # RSIoversold
            (dataframe['rsi_14'] < 35) & (dataframe['rsi_14'] > dataframe['rsi_14'].shift(2)) &
            # Bollinger Bands down
            (dataframe['close'] > dataframe['bb_lower']) & 
            (dataframe['close'].shift(1) <= dataframe['bb_lower'].shift(1)) &
            # volume large
            (dataframe['volume_ratio'] > 1.3) &
            # trend strength
            (dataframe['trend_strength'] > dataframe['trend_strength'].shift(3)) &
            # not at low
            not_chasing_low & long_favourable_environment
        )
        dataframe.loc[support_bounce, 'enter_long'] = 1
        dataframe.loc[support_bounce, 'enter_tag'] = 'support'
        
        # === 📊 long signal (time: 4hours-24hours) ===
        
        # 5. 🚨 new：trend - optimize after avoid
        trend_acceleration = (
            # new but
            # translated
            (dataframe['ema_5'] > dataframe['ema_13']) & (dataframe['ema_13'] > dataframe['ema_34']) &
            (dataframe['ema_34'] > dataframe['ema_50']) &
            # trend strength but not
            (dataframe['trend_strength'] > 25) & (dataframe['trend_strength'] < 65) &  # low up
            (dataframe['trend_strength'] - dataframe['trend_strength'].shift(10) > 10) &  # low
            # ADXstrong trend confirm but has up
            (dataframe['adx'] > 25) & (dataframe['adx'] < 50) &  # avoid trend
            (dataframe['adx'] > dataframe['adx'].shift(5)) &
            # MACDbut not
            (dataframe['macd'] > dataframe['macd_signal']) & (dataframe['macd'] > 0) &
            # price atVWAPup but not high
            (dataframe['close'] > dataframe['vwap']) &
            (dataframe['close'] < dataframe['vwap'] * 1.08) &  # notVWAP 8%
            # RSIbut overbought
            (dataframe['rsi_14'] > 50) & (dataframe['rsi_14'] < 70) &  # low up
            (dataframe['rsi_14'].shift(10) < 55) &  # from low
            # volume large
            (dataframe['volume_ratio'] > 1.15) & (dataframe['volume_ratio'] < 2.5) &
            # translated：predictive
            in_favorable_long_zone &  # predictive most
            not_chasing_high & long_favourable_environment
        )
        dataframe.loc[trend_acceleration, 'enter_long'] = 1
        dataframe.loc[trend_acceleration, 'enter_tag'] = 'trend'
        
        # 6. breakout confirm - high probability
        breakout_retest = (
            # price new up resistance
            (dataframe['close'] > dataframe['bb_middle']) & 
            (dataframe['close'] > dataframe['vwap']) &
            # before has not
            (dataframe['low'].rolling(3).min() > dataframe['ema_21'] * 0.99) &
            # volume confirmation
            (dataframe['volume_ratio'] > 1.25) &
            # indicator
            (dataframe['rsi_14'] > 50) & (dataframe['macd_hist'] > 0) &
            # ADXtrend
            (dataframe['adx'] > 25) &
            # momentum score
            (dataframe['momentum_score'] > 0.1) &
            long_favourable_environment
        )
        dataframe.loc[breakout_retest, 'enter_long'] = 1
        dataframe.loc[breakout_retest, 'enter_tag'] = 'breakout confirm'
        
        # === 🎯 signal ===
        
        # 7. reversal - high risk high
        reversal_bottom = (
            # price at low
            (price_percentile_20 < 0.25) &
            # RSIoversold after
            (dataframe['rsi_14'] < 25) & (dataframe['rsi_14'] > dataframe['rsi_14'].shift(3) + 5) &
            # volume abnormal large（after）
            (dataframe['volume_ratio'] > 2.0) &
            # reversal signal
            (dataframe.get('reversal_signal_strength', 0) > 25) &
            # MACDbullish divergence
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)) &
            long_favourable_environment
        )
        dataframe.loc[reversal_bottom, 'enter_long'] = 1
        dataframe.loc[reversal_bottom, 'enter_tag'] = 'reversal'
        
        # === 🎯 time signal - high ===
        
        # 8A. MTFstrong breakout - has time 1
        mtf_strong_breakout = (
            # time 1
            (dataframe['mtf_strong_bull'] == 1) &
            # price breakout1hours resistance
            (dataframe['close'] > dataframe['h1_resistance']) &
            (dataframe['close'].shift(1) <= dataframe['h1_resistance'].shift(1)) &
            # volume confirmation
            (dataframe['volume_spike'] == 1) &
            # 5minutes indicator
            (dataframe['rsi_14'] > 50) & (dataframe['rsi_14'] < 70) &
            (dataframe['macd_hist'] > 0) &
            # not at high：predictive
            in_favorable_long_zone &
            long_favourable_environment
        )
        dataframe.loc[mtf_strong_breakout, 'enter_long'] = 1
        dataframe.loc[mtf_strong_breakout, 'enter_tag'] = 'time breakout'
        
        # 8B. MTFsupport - at support
        mtf_support_bounce = (
            # price at1hours or4hours support
            ((dataframe['near_h1_support'] == 1) | (dataframe['near_h4_support'] == 1)) &
            # time trend score
            (dataframe['mtf_trend_score'] > 0.2) &
            # fast reversal signal
            ((dataframe['is_pin_bar_bullish'] == 1) | (dataframe['is_bullish_engulfing'] == 1)) &
            # StochRSIoversold
            (dataframe['stoch_rsi_k'] < 30) & (dataframe['stoch_rsi_k'] > dataframe['stoch_rsi_k'].shift(1)) &
            # volume confirmation
            (dataframe['volume_ratio'] > 1.2) &
            long_favourable_environment
        )
        dataframe.loc[mtf_support_bounce, 'enter_long'] = 1
        dataframe.loc[mtf_support_bounce, 'enter_tag'] = 'time support'
        
        # === 🎯 new predictive long signal - translated，long ===
        
        # 10A. volume divergence long - high
        volume_divergence_long = (
            # price new low but volume（translated）
            (dataframe['close'] < dataframe['close'].rolling(10).min().shift(1)) &
            (dataframe['volume_ratio'] < dataframe['volume_ratio'].rolling(10).mean() * 0.8) &
            # RSIbullish divergence confirm
            (dataframe['rsi_divergence_strength'] > 0.5) &
            # MACDmomentum
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1)) &
            (dataframe['macd_hist'] > dataframe['macd_hist'].shift(2)) &
            # translated
            (dataframe['market_sentiment'] < -0.5) &
            # at predictive long
            in_favorable_long_zone & long_favourable_environment
        )
        dataframe.loc[volume_divergence_long, 'enter_long'] = 1
        dataframe.loc[volume_divergence_long, 'enter_tag'] = 'volume divergence long'
        
        # 10B. [momentum long - nonemomentum_exhaustion]
        momentum_recovery_long = (
            False &  # translated
            # trend
            (dataframe['trend_sustainability'] > 0.3) &
            # StochRSIfrom oversold fast rise
            (dataframe['stoch_rsi_k'] < 25) &
            (dataframe['stoch_rsi_k'] > dataframe['stoch_rsi_k'].shift(1) + 5) &
            # Williams indicator oversold after
            (dataframe['williams_r'] < -80) &
            (dataframe['williams_r'] > dataframe['williams_r'].shift(2) + 10) &
            # CCIoversold after
            (dataframe['cci'] < -100) & (dataframe['cci'] > dataframe['cci'].shift(1)) &
            # at predictive long
            in_favorable_long_zone & long_favourable_environment
        )
        # [translated momentum_recovery_long signal]
        
        # 10C. reversal long - before
        early_reversal_long = (
            # reversal signal（signal）
            (dataframe['reversal_probability'] > 0.6) &
            # price reversal value（oversold）
            (dataframe['market_sentiment'] < -0.7) &
            # volatility but up（translated）
            (dataframe['volatility_state'] > 70) &
            # trend strength from low
            (dataframe['trend_strength'] > dataframe['trend_strength'].rolling(5).mean() + 10) &
            # ADXrise（new trend）
            (dataframe['adx'] > dataframe['adx'].shift(2)) &
            # Bollinger Bands（translated）
            (dataframe['bb_upper'] - dataframe['bb_lower'] > (dataframe['bb_upper'] - dataframe['bb_lower']).rolling(10).mean() * 1.1) &
            # RSIfrom
            (dataframe['rsi_14'] < 40) & (dataframe['rsi_14'] > dataframe['rsi_14'].shift(3) + 3) &
            # at predictive long
            in_favorable_long_zone & long_favourable_environment
        )
        dataframe.loc[early_reversal_long, 'enter_long'] = 1
        dataframe.loc[early_reversal_long, 'enter_tag'] = 'reversal long'
        
        # 10D. long - confirm
        smart_bottom_long = (
            # price support
            (dataframe['close'] > dataframe['vwap'] * 0.98) & (dataframe['close'] < dataframe['vwap'] * 1.01) &
            # oversold confirm
            (dataframe['rsi_14'] < 35) & (dataframe['stoch_rsi_k'] < 30) & (dataframe['williams_r'] < -75) &
            # volume abnormal large（translated）
            (dataframe['volume_ratio'] > 1.4) &
            # Bollinger Bands down support
            (dataframe['close'] > dataframe['bb_lower']) & 
            (dataframe['close'].shift(1) <= dataframe['bb_lower'].shift(1)) &
            # divergence signal
            (dataframe['rsi_divergence_strength'] > 0.3) &
            # trend strength not
            (dataframe['trend_strength'] > -50) &
            # at predictive long
            in_favorable_long_zone & long_favourable_environment
        )
        dataframe.loc[smart_bottom_long, 'enter_long'] = 1
        dataframe.loc[smart_bottom_long, 'enter_tag'] = 'long'
        
        # === 🔻 bearish signal - use indicator solve lagging issues ===
        
        # 0A. price as - most fast reversal signal
        price_action_top = (
            # translatedPin Bar：long up reversal signal
            (dataframe['is_pin_bar_bearish'] == 1) &
            # or：reversal
            ((dataframe['is_bearish_engulfing'] == 1) | 
             # StochRSIoverbought after down
             ((dataframe['stoch_rsi_k'] > 80) & (dataframe['stoch_rsi_k'] < dataframe['stoch_rsi_k'].shift(1)))) &
            # Williams indicator confirm reversal
            (dataframe['williams_r'] > -30) & (dataframe['williams_r'] < dataframe['williams_r'].shift(1)) &
            # CCIfrom overbought
            (dataframe['cci'] > 100) & (dataframe['cci'] < dataframe['cci'].shift(1)) &
            # volume abnormal large（translated）
            (dataframe['volume_spike'] == 1) &
            # price resistance（not short）
            (dataframe['resistance_strength'] > -8) &
            # price：predictive most short
            in_favorable_short_zone & (price_percentile_20 < 0.85) &
            # time confirm：long-term trend short
            (dataframe['mtf_short_filter'] == 1) &
            short_favourable_environment
        )
        dataframe.loc[price_action_top, 'enter_short'] = 1
        dataframe.loc[price_action_top, 'enter_tag'] = 'price as'
        
        # 0B. reversal short - most short
        acceleration_reversal_short = (
            # price up（signal）
            (dataframe['price_velocity'] > 0) &  # still at up
            (dataframe['price_acceleration'] < 0) &  # but up at
            (dataframe['price_acceleration'] < dataframe['price_acceleration'].shift(1)) &  # fast
            # RSIbearish divergence：price new high butRSInew high
            (dataframe['rsi_divergence_strength'] < 0) &
            # StochRSIfast from high
            (dataframe['stoch_rsi_k'] > 70) & 
            (dataframe['stoch_rsi_k'] < dataframe['stoch_rsi_k'].shift(1) - 3) &
            # volume abnormal（after）
            (dataframe['volume_ratio'] > 1.3) &
            # price：predictive most short
            in_favorable_short_zone & (price_percentile_20 < 0.80) &
            # time confirm
            (dataframe['mtf_short_filter'] == 1) &
            short_favourable_environment
        )
        dataframe.loc[acceleration_reversal_short, 'enter_short'] = 1
        dataframe.loc[acceleration_reversal_short, 'enter_tag'] = 'reversal short'
        
        # === 🔻 bearish signal（but optimize） ===
        
        # 1. short reversal - translated
        ultra_short_bear = (
            # EMA5fast downEMA8 + divergence confirm
            (dataframe['ema_5'] < dataframe['ema_8']) & 
            (dataframe['ema_5'].shift(1) >= dataframe['ema_8'].shift(1)) &
            # RSIdivergence：price new high butRSInot new high
            (dataframe['rsi_14'] < 70) & (dataframe['rsi_14'] > 40) &
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(2) - 5) &
            (dataframe['close'] > dataframe['close'].shift(2)) &  # price still at up
            (dataframe['rsi_14'] < dataframe['rsi_14'].rolling(5).max().shift(3)) &  # RSIdivergence
            # volume but price
            (dataframe['volume_ratio'] > 1.5) &
            (dataframe['close'] < dataframe['high'].rolling(3).max()) &  # new high
            # avoid at low short：use predictive
            not_chasing_low & short_favourable_environment &
            in_favorable_short_zone  # predictive most short
        )
        dataframe.loc[ultra_short_bear, 'enter_short'] = 1
        dataframe.loc[ultra_short_bear, 'enter_tag'] = 'short divergence'
        
        # 2. short - high
        scalping_short = (
            # price at Bollinger Bands up
            (dataframe['bb_position'] > 0.7) & (dataframe['bb_position'] < 0.95) &
            # MACDtranslated
            (dataframe['macd_hist'] < 0) & (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1)) &
            # volume mid but not
            (dataframe['volume_ratio'] > 1.1) & (dataframe['volume_ratio'] < 2.0) &
            # RSIhigh but overbought
            (dataframe['rsi_14'] > 55) & (dataframe['rsi_14'] < 75) &
            # trend strength
            (dataframe['trend_strength'] < 70) & (dataframe['trend_strength'] > -10) &
            # price：predictive short
            in_favorable_short_zone & short_favourable_environment
        )
        dataframe.loc[scalping_short, 'enter_short'] = 1
        dataframe.loc[scalping_short, 'enter_tag'] = 'short'
        
        # 3. confirm short - but optimize
        death_cross_confirmed = (
            # EMA8downEMA21
            (dataframe['ema_8'] < dataframe['ema_21']) & 
            (dataframe['ema_8'].shift(1) >= dataframe['ema_21'].shift(1)) &
            # translatedEMA5atEMA8down
            (dataframe['ema_5'] < dataframe['ema_8']) &
            # MACDconfirm
            (dataframe['macd'] < dataframe['macd_signal']) &
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1)) &
            # RSIfall
            (dataframe['rsi_14'] < 60) & (dataframe['rsi_14'] > 30) &
            # ADXtrend
            (dataframe['adx'] > 20) & (dataframe['adx'] > dataframe['adx'].shift(2)) &
            # volume confirmation
            (dataframe['volume_ratio'] > 1.2) &
            # translated - predictive short
            not_chasing_low & short_favourable_environment & in_favorable_short_zone
        )
        dataframe.loc[death_cross_confirmed, 'enter_short'] = 1
        dataframe.loc[death_cross_confirmed, 'enter_tag'] = 'confirm'
        
        # 4. resistance - short
        resistance_rejection = (
            # priceEMA34resistance but none breakout
            (dataframe['close'] < dataframe['ema_34'] * 1.005) & 
            (dataframe['close'] > dataframe['ema_34'] * 0.99) &
            # RSIoverbought（Hyperoptoptimize）
            (dataframe['rsi_14'] > self.rsi_sell_max.value) & (dataframe['rsi_14'] < dataframe['rsi_14'].shift(2)) &
            # Bollinger Bands up
            (dataframe['close'] < dataframe['bb_upper']) & 
            (dataframe['close'].shift(1) >= dataframe['bb_upper'].shift(1)) &
            # volume large but not
            (dataframe['volume_ratio'] > 1.3) &
            # trend strength fall
            (dataframe['trend_strength'] < dataframe['trend_strength'].shift(3)) &
            # not at high：predictive short
            not_chasing_high & short_favourable_environment & in_favorable_short_zone
        )
        dataframe.loc[resistance_rejection, 'enter_short'] = 1
        dataframe.loc[resistance_rejection, 'enter_tag'] = 'resistance'
        
        # 5. trend short - mid long has
        trend_exhaustion = (
            # translated
            (dataframe['ema_5'] < dataframe['ema_13']) & (dataframe['ema_13'] < dataframe['ema_34']) &
            (dataframe['ema_34'] < dataframe['ema_50']) &
            # trend strength fall
            (dataframe['trend_strength'] < -20) & 
            (dataframe['trend_strength'] - dataframe['trend_strength'].shift(10) < -15) &
            # ADXstrong trend confirm fall
            (dataframe['adx'] > 30) & (dataframe['adx'] > dataframe['adx'].shift(5)) &
            # MACDtranslated
            (dataframe['macd'] < dataframe['macd_signal']) & (dataframe['macd'] < 0) &
            # price atVWAPdown
            (dataframe['close'] < dataframe['vwap']) &
            # RSIbut
            (dataframe['rsi_14'] < 45) & (dataframe['rsi_14'] > 20) &
            # volume large（translated）
            (dataframe['volume_ratio'] > 1.15) &
            # translated：predictive short
            not_chasing_low & short_favourable_environment & in_favorable_short_zone
        )
        dataframe.loc[trend_exhaustion, 'enter_short'] = 1
        dataframe.loc[trend_exhaustion, 'enter_tag'] = 'trend'
        
        # 6. breakout short - high probability
        false_breakout_short = (
            # price support after fast
            (dataframe['close'] < dataframe['bb_middle']) & 
            (dataframe['close'] < dataframe['vwap']) &
            # before has breakout
            (dataframe['high'].rolling(3).max() < dataframe['ema_21'] * 1.01) &
            # volume confirmation but not
            (dataframe['volume_ratio'] > 1.25) &
            # indicator
            (dataframe['rsi_14'] < 50) & (dataframe['macd_hist'] < 0) &
            # ADXtrend
            (dataframe['adx'] > 25) &
            # momentum score
            (dataframe['momentum_score'] < -0.1) &
            short_favourable_environment & in_favorable_short_zone
        )
        dataframe.loc[false_breakout_short, 'enter_short'] = 1
        dataframe.loc[false_breakout_short, 'enter_tag'] = 'breakout short'
        
        # 7. reversal - high risk high
        reversal_top = (
            # price at high
            (price_percentile_20 > 0.75) &
            # RSIoverbought after
            (dataframe['rsi_14'] > 75) & (dataframe['rsi_14'] < dataframe['rsi_14'].shift(3) - 5) &
            # volume abnormal large（translated）
            (dataframe['volume_ratio'] > 2.0) &
            # reversal signal
            (dataframe.get('reversal_signal_strength', 0) < -25) &
            # MACDbearish divergence
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1)) &
            # price new high but indicator divergence
            (dataframe['close'] > dataframe['close'].shift(5)) &
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(5)) &
            short_favourable_environment
        )
        dataframe.loc[reversal_top, 'enter_short'] = 1
        dataframe.loc[reversal_top, 'enter_tag'] = 'reversal'
        
        # === 🎯 time bearish signal ===
        
        # 8A. MTFtranslated - has time 1
        mtf_strong_breakdown = (
            # time 1
            (dataframe['mtf_strong_bear'] == 1) &
            # price1hours support
            (dataframe['close'] < dataframe['h1_support']) &
            (dataframe['close'].shift(1) >= dataframe['h1_support'].shift(1)) &
            # volume confirmation
            (dataframe['volume_spike'] == 1) &
            # 5minutes indicator
            (dataframe['rsi_14'] < 50) & (dataframe['rsi_14'] > 30) &
            (dataframe['macd_hist'] < 0) &
            # not at low：predictive short
            in_favorable_short_zone &
            short_favourable_environment
        )
        dataframe.loc[mtf_strong_breakdown, 'enter_short'] = 1
        dataframe.loc[mtf_strong_breakdown, 'enter_tag'] = 'time'
        
        # 8B. MTFresistance - at resistance short
        mtf_resistance_rejection = (
            # price at1hours or4hours resistance
            ((dataframe['near_h1_resistance'] == 1) | (dataframe['near_h4_resistance'] == 1)) &
            # time trend score
            (dataframe['mtf_trend_score'] < -0.2) &
            # fast reversal signal
            ((dataframe['is_pin_bar_bearish'] == 1) | (dataframe['is_bearish_engulfing'] == 1)) &
            # StochRSIoverbought
            (dataframe['stoch_rsi_k'] > 70) & (dataframe['stoch_rsi_k'] < dataframe['stoch_rsi_k'].shift(1)) &
            # volume confirmation
            (dataframe['volume_ratio'] > 1.2) &
            short_favourable_environment
        )
        dataframe.loc[mtf_resistance_rejection, 'enter_short'] = 1
        dataframe.loc[mtf_resistance_rejection, 'enter_tag'] = 'time resistance'
        
        # === 🎯 new predictive short signal - short ===
        
        # 9A. volume divergence short - high
        volume_divergence_short = (
            # price new high but volume
            (dataframe['close'] > dataframe['close'].rolling(10).max().shift(1)) &
            (dataframe['volume_ratio'] < dataframe['volume_ratio'].rolling(10).mean() * 0.8) &
            # RSIdivergence confirm
            (dataframe['rsi_divergence_strength'] < -0.5) &
            # MACDmomentum
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(1)) &
            (dataframe['macd_hist'] < dataframe['macd_hist'].shift(2)) &
            # translated
            (dataframe['market_sentiment'] > 0.5) &
            # at predictive short
            in_favorable_short_zone & short_favourable_environment
        )
        dataframe.loc[volume_divergence_short, 'enter_short'] = 1
        dataframe.loc[volume_divergence_short, 'enter_tag'] = 'volume divergence short'
        
        # 9B. [momentum short - nonemomentum_exhaustion]
        momentum_exhaustion_short = (
            False &  # translated
            # trend not
            (dataframe['trend_sustainability'] < -0.3) &
            # StochRSIfrom overbought fast fall
            (dataframe['stoch_rsi_k'] > 75) &
            (dataframe['stoch_rsi_k'] < dataframe['stoch_rsi_k'].shift(1) - 5) &
            # Williams indicator overbought after
            (dataframe['williams_r'] > -20) &
            (dataframe['williams_r'] < dataframe['williams_r'].shift(2) - 10) &
            # CCIoverbought after
            (dataframe['cci'] > 100) & (dataframe['cci'] < dataframe['cci'].shift(1)) &
            # at predictive short
            in_favorable_short_zone & short_favourable_environment
        )
        # [translated momentum_exhaustion_short signal]
        
        # 9C. reversal short - before
        early_reversal_short = (
            # reversal signal
            (dataframe['reversal_probability'] > 0.6) &
            # price reversal value
            (dataframe['market_sentiment'] > 0.7) &
            # volatility（not）
            (dataframe['volatility_state'] > 70) &
            # trend strength
            (dataframe['trend_strength'] < dataframe['trend_strength'].rolling(5).mean() - 10) &
            # ADXfall（trend）
            (dataframe['adx'] < dataframe['adx'].shift(2)) &
            # Bollinger Bands（before）
            (dataframe['bb_upper'] - dataframe['bb_lower'] < (dataframe['bb_upper'] - dataframe['bb_lower']).rolling(10).mean() * 0.9) &
            # at predictive short
            in_favorable_short_zone & short_favourable_environment
        )
        dataframe.loc[early_reversal_short, 'enter_short'] = 1
        dataframe.loc[early_reversal_short, 'enter_tag'] = 'reversal short'
        
        # 9D. short - confirm short (long)
        smart_top_short = (
            # price resistance
            (dataframe['close'] < dataframe['vwap'] * 1.02) & (dataframe['close'] > dataframe['vwap'] * 0.99) &
            # overbought confirm
            (dataframe['rsi_14'] > 65) & (dataframe['stoch_rsi_k'] > 70) & (dataframe['williams_r'] > -25) &
            # volume abnormal large（translated）
            (dataframe['volume_ratio'] > 1.4) &
            # Bollinger Bands up resistance
            (dataframe['close'] < dataframe['bb_upper']) & 
            (dataframe['close'].shift(1) >= dataframe['bb_upper'].shift(1)) &
            # bearish divergence signal
            (dataframe['rsi_divergence_strength'] < -0.3) &
            # trend strength not
            (dataframe['trend_strength'] < 50) &
            # at predictive short
            in_favorable_short_zone & short_favourable_environment
        )
        dataframe.loc[smart_top_short, 'enter_short'] = 1
        dataframe.loc[smart_top_short, 'enter_tag'] = 'short'
        
        # === signal（translated） ===
        # has count signal，most
        signal_priority = {
            # short signal - most high（most fast）
            'ULTRA_SHORT_MOMENTUM': 10, 'ULTRA_SHORT_BEAR_DIVERGENCE': 10,
            # predictive signal - high（predictive，translated）
            'VOLUME_DIVERGENCE_SHORT': 10, 'MOMENTUM_EXHAUSTION_SHORT': 10, 'EARLY_REVERSAL_SHORT': 10, 'SMART_TOP_SHORT': 10,
            'VOLUME_DIVERGENCE_LONG': 10, 'MOMENTUM_RECOVERY_LONG': 10, 'EARLY_REVERSAL_LONG': 10, 'SMART_BOTTOM_LONG': 10,
            # reversal signal - high（predictive）
            'REVERSAL_TOP': 9, 'REVERSAL_BOTTOM': 9,
            # signal - high（translated）
            'GOLDEN_CROSS_CONFIRMED': 8, 'DEATH_CROSS_CONFIRMED': 8,
            # resistance support signal - mid high
            'RESISTANCE_REJECTION': 7, 'SUPPORT_BOUNCE': 7,
            # trend signal - mid
            'TREND_ACCELERATION': 6, 'TREND_EXHAUSTION': 6,
            # breakout signal - mid
            'BREAKOUT_RETEST': 5, 'FALSE_BREAKOUT_SHORT': 5,
            # signal - low（high but small）
            'SCALPING_LONG': 4, 'SCALPING_SHORT': 4
        }
        
        # record signal
        signal_counts = {}
        # short signal（translated15count）
        short_signals = ['ULTRA_SHORT_BEAR_DIVERGENCE', 'SCALPING_SHORT', 'DEATH_CROSS_CONFIRMED', 
                        'RESISTANCE_REJECTION', 'TREND_EXHAUSTION', 'FALSE_BREAKOUT_SHORT', 'REVERSAL_TOP',
                        'VOLUME_DIVERGENCE_SHORT', 'MOMENTUM_EXHAUSTION_SHORT', 'EARLY_REVERSAL_SHORT', 'SMART_TOP_SHORT',
                        'PRICE_ACTION_TOP', 'ACCELERATION_REVERSAL_SHORT', 'MTF_STRONG_BREAKDOWN', 'MTF_RESISTANCE_REJECTION']
        
        # long signal（after15count）
        long_signals = ['ULTRA_SHORT_MOMENTUM', 'SCALPING_LONG', 'GOLDEN_CROSS_CONFIRMED',
                       'SUPPORT_BOUNCE', 'TREND_ACCELERATION', 'BREAKOUT_RETEST', 'REVERSAL_BOTTOM',
                       'VOLUME_DIVERGENCE_LONG', 'MOMENTUM_RECOVERY_LONG', 'EARLY_REVERSAL_LONG', 'SMART_BOTTOM_LONG',
                       'PRICE_ACTION_BOTTOM', 'ACCELERATION_REVERSAL', 'MTF_STRONG_BREAKOUT', 'MTF_SUPPORT_BOUNCE']
        
        for tag in signal_priority.keys():
            count = (dataframe['enter_tag'] == tag).sum() if 'enter_tag' in dataframe.columns else 0
            if count > 0:
                if tag in short_signals:
                    signal_counts[f"bearish-{tag}"] = count
                elif tag in long_signals:
                    signal_counts[f"bullish-{tag}"] = count
                    
        # signal
        total_long_signals = sum([count for key, count in signal_counts.items() if key.startswith("bullish")])
        total_short_signals = sum([count for key, count in signal_counts.items() if key.startswith("bearish")])
        signal_balance_ratio = total_long_signals / (total_short_signals + 1e-6)  # avoid
        
        logger.info(f"""
🎯 signal - {pair} (optimize after):
{'='*60}
📊 signal:
├─ bullish signal: {total_long_signals}
├─ bearish signal: {total_short_signals}
├─ translated: {signal_balance_ratio:.2f} {'✅translated' if 0.5 <= signal_balance_ratio <= 2.0 else '⚠️translated'}
└─ translated: {signal_counts if signal_counts else 'before none signal'}

📈 before market state:
├─ price: {price_percentile_20.iloc[-1] if len(price_percentile_20) > 0 else 0:.1%}translated ({price_percentile_50.iloc[-1] if len(price_percentile_50) > 0 else 0:.1%}long)
├─ bullish: {'✅' if (price_percentile_20.iloc[-1] if len(price_percentile_20) > 0 else 0) < 0.55 else '❌'}long most
├─ bearish: {'✅' if (price_percentile_20.iloc[-1] if len(price_percentile_20) > 0 else 0) > 0.45 else '❌'}short most
├─ RSI: {dataframe['rsi_14'].iloc[-1] if 'rsi_14' in dataframe.columns and len(dataframe) > 0 else 50:.1f}
├─ ADXtrend strength: {dataframe['adx'].iloc[-1] if 'adx' in dataframe.columns and len(dataframe) > 0 else 25:.1f}
├─ volume: {dataframe['volume_ratio'].iloc[-1] if 'volume_ratio' in dataframe.columns and len(dataframe) > 0 else 1:.2f}x
├─ trend score: {dataframe['trend_strength'].iloc[-1] if 'trend_strength' in dataframe.columns and len(dataframe) > 0 else 50:.0f}/100
├─ momentum score: {dataframe['momentum_score'].iloc[-1] if 'momentum_score' in dataframe.columns and len(dataframe) > 0 else 0:.3f}
├─ translated: {dataframe['market_sentiment'].iloc[-1] if 'market_sentiment' in dataframe.columns and len(dataframe) > 0 else 0:.3f}
└─ divergence strength: {dataframe['rsi_divergence_strength'].iloc[-1] if 'rsi_divergence_strength' in dataframe.columns and len(dataframe) > 0 else 0:.3f}

🎯 predictive signal:
├─ bullish signal: 4count high signal (volume divergence/momentum/reversal/translated)
├─ bearish signal: 4count high signal (volume divergence/momentum/reversal/translated)
└─ signal: bullish15count vs bearish15count (translated)
{'='*60}
""")
        
        return dataframe
    
    def _log_enhanced_entry_decision(self, pair: str, dataframe: DataFrame, current_data, direction: str):
        """record"""
        
        # get
        entry_tag = current_data.get('enter_tag', 'UNKNOWN_SIGNAL')
        
        # signal
        signal_explanations = {
            'GOLDEN_CROSS_BREAKOUT': 'breakout - EMA8upEMA21，confirm rise trend',
            'MACD_MOMENTUM_CONFIRMED': 'MACDmomentum confirm - MACDand long，momentum',
            'OVERSOLD_SUPPORT_BOUNCE': 'oversold support - RSIoversold after，support confirm has',
            'BREAKOUT_RETEST_HOLD': 'breakout confirm - breakout after not，trend',
            'INSTITUTIONAL_ACCUMULATION': 'translated - large，translated',
            'DEATH_CROSS_BREAKDOWN': 'translated - EMA8downEMA21，confirm fall trend',
            'MACD_MOMENTUM_BEARISH': 'MACDmomentum confirm - MACDand fall，momentum',
            'OVERBOUGHT_RESISTANCE_REJECT': 'overbought resistance - RSIoverbought after，resistance has',
            'BREAKDOWN_RETEST_FAIL': 'translated - support after none',
            'INSTITUTIONAL_DISTRIBUTION': 'translated - large，translated'
        }
        
        signal_type = signal_explanations.get(entry_tag, f'signal confirm - {entry_tag}')
        
        # translated
        technical_analysis = {
            'rsi_14': current_data.get('rsi_14', 50),
            'macd': current_data.get('macd', 0),
            'macd_signal': current_data.get('macd_signal', 0),
            'macd_hist': current_data.get('macd_hist', 0),
            'ema_8': current_data.get('ema_8', 0),
            'ema_21': current_data.get('ema_21', 0),
            'ema_50': current_data.get('ema_50', 0),
            'adx': current_data.get('adx', 25),
            'volume_ratio': current_data.get('volume_ratio', 1),
            'bb_position': current_data.get('bb_position', 0.5),
            'trend_strength': current_data.get('trend_strength', 50),
            'momentum_score': current_data.get('momentum_score', 0),
            'ob_depth_imbalance': current_data.get('ob_depth_imbalance', 0),
            'ob_market_quality': current_data.get('ob_market_quality', 0.5)
        }
        
        # reason
        entry_reasoning = self._build_entry_reasoning(entry_tag, technical_analysis, direction)
        
        signal_details = {
            'signal_strength': current_data.get('signal_strength', 0),
            'entry_tag': entry_tag,
            'signal_explanation': signal_type,
            'entry_reasoning': entry_reasoning,
            'trend_confirmed': technical_analysis['trend_strength'] > 30 if direction == 'LONG' else technical_analysis['trend_strength'] < -30,
            'momentum_support': technical_analysis['momentum_score'] > 0.1 if direction == 'LONG' else technical_analysis['momentum_score'] < -0.1,
            'volume_confirmed': technical_analysis['volume_ratio'] > 1.1,
            'market_favorable': technical_analysis['ob_market_quality'] > 0.4,
            'decision_reason': f"{signal_type}"
        }
        
        risk_analysis = {
            'planned_stoploss': abs(self.stoploss) * 100,
            'risk_percentage': self.max_risk_per_trade * 100,
            'suggested_position': self.base_position_size * 100,
            'suggested_leverage': self.leverage_multiplier,
            'risk_budget_remaining': 80,
            'risk_level': self._assess_entry_risk_level(technical_analysis)
        }
        
        # translated decision_logger day record
        pass
    
    def _build_entry_reasoning(self, entry_tag: str, tech: dict, direction: str) -> str:
        """reason"""
        
        reasoning_templates = {
            'GOLDEN_CROSS_BREAKOUT': f"EMA8({tech['ema_8']:.2f})upEMA21({tech['ema_21']:.2f})translated，price breakoutEMA50({tech['ema_50']:.2f})confirm trend，ADX({tech['adx']:.1f})trend strength，volume large{tech['volume_ratio']:.1f}confirm breakout has",
            
            'MACD_MOMENTUM_CONFIRMED': f"MACD({tech['macd']:.4f})up signal({tech['macd_signal']:.4f})translated，translated({tech['macd_hist']:.4f})as and long，momentum score{tech['momentum_score']:.3f}rise，price upVWAPconfirm",
            
            'OVERSOLD_SUPPORT_BOUNCE': f"RSI({tech['rsi_14']:.1f})from oversold，Bollinger Bands({tech['bb_position']:.2f})price down after，volume{tech['volume_ratio']:.1f}large confirm，translated({tech['ob_depth_imbalance']:.2f})translated",
            
            'BREAKOUT_RETEST_HOLD': f"price breakout supertrend Bollinger Bands mid after，translatedEMA21support has，ADX({tech['adx']:.1f})confirm trend，volatility at range，volume{tech['volume_ratio']:.1f}support breakout",
            
            'INSTITUTIONAL_ACCUMULATION': f"translated({tech['ob_depth_imbalance']:.2f})large，abnormal volume spike{tech['volume_ratio']:.1f}translated，price upVWAP，trend strength({tech['trend_strength']:.0f})translated",
            
            'DEATH_CROSS_BREAKDOWN': f"EMA8({tech['ema_8']:.2f})downEMA21({tech['ema_21']:.2f})translated，priceEMA50({tech['ema_50']:.2f})confirm trend，ADX({tech['adx']:.1f})down trend strength，translated{tech['volume_ratio']:.1f}confirm",
            
            'MACD_MOMENTUM_BEARISH': f"MACD({tech['macd']:.4f})down signal({tech['macd_signal']:.4f})translated，translated({tech['macd_hist']:.4f})as and fall，momentum score{tech['momentum_score']:.3f}down，priceVWAPconfirm",
            
            'OVERBOUGHT_RESISTANCE_REJECT': f"RSI({tech['rsi_14']:.1f})from overbought，Bollinger Bands({tech['bb_position']:.2f})price at up，volume{tech['volume_ratio']:.1f}confirm，resistance has",
            
            'BREAKDOWN_RETEST_FAIL': f"price supertrend Bollinger Bands mid after，translatedEMA21resistance，ADX({tech['adx']:.1f})confirm down trend，volume{tech['volume_ratio']:.1f}support",
            
            'INSTITUTIONAL_DISTRIBUTION': f"translated({tech['ob_depth_imbalance']:.2f})large，abnormal volume spike{tech['volume_ratio']:.1f}translated，priceVWAP，trend strength({tech['trend_strength']:.0f})translated"
        }
        
        return reasoning_templates.get(entry_tag, f"translated{entry_tag}signal confirm，indicator{direction}translated")
    
    def _assess_entry_risk_level(self, tech: dict) -> str:
        """risk level"""
        risk_score = 0
        
        # ADXrisk
        if tech['adx'] > 30:
            risk_score += 1  # strong trend low risk
        elif tech['adx'] < 20:
            risk_score -= 1  # trend risk
            
        # volume risk
        if tech['volume_ratio'] > 1.5:
            risk_score += 1  # low risk
        elif tech['volume_ratio'] < 0.8:
            risk_score -= 1  # risk
            
        # risk
        if tech['ob_market_quality'] > 0.6:
            risk_score += 1  # high low risk
        elif tech['ob_market_quality'] < 0.3:
            risk_score -= 1  # low risk
            
        # volatility risk (translatedRSIvalue)
        if 25 < tech['rsi_14'] < 75:
            risk_score += 1  # low risk
        else:
            risk_score -= 1  # value risk
        
        if risk_score >= 2:
            return "low risk"
        elif risk_score >= 0:
            return "medium risk"
        else:
            return "high risk"
    
    def _log_short_entry_decision(self, pair: str, dataframe: DataFrame, current_data):
        """record bearish"""
        
        signal_type = self._determine_short_signal_type(current_data)
        
        signal_details = {
            'signal_strength': current_data.get('signal_strength', 0),
            'trend_confirmed': current_data.get('trend_strength', 0) > 60,
            'momentum_support': current_data.get('momentum_score', 0) < -0.1,
            'volume_confirmed': current_data.get('volume_ratio', 1) > 1.1,
            'market_favorable': current_data.get('volatility_state', 50) < 90,
            'decision_reason': f"{signal_type} - signal strength{current_data.get('signal_strength', 0):.1f}"
        }
        
        risk_analysis = {
            'planned_stoploss': abs(self.stoploss) * 100,
            'risk_percentage': self.max_risk_per_trade * 100,
            'suggested_position': self.base_position_size * 100,
            'suggested_leverage': self.leverage_multiplier,
            'risk_budget_remaining': 80,  # value
            'risk_level': 'mid'
        }
        
        # translated decision_logger day record
        pass
    
    def _determine_long_signal_type(self, current_data) -> str:
        """bullish signal"""
        if (current_data.get('trend_strength', 0) > 60 and 
            current_data.get('momentum_score', 0) > 0.1):
            return "trend confirmation+momentum support"
        elif current_data.get('rsi_14', 50) < 35:
            return "oversold"
        elif (current_data.get('close', 0) > current_data.get('supertrend', 0)):
            return "breakout confirm signal"
        else:
            return "signal"
    
    def _determine_short_signal_type(self, current_data) -> str:
        """bearish signal"""
        if (current_data.get('trend_strength', 0) > 60 and 
            current_data.get('momentum_score', 0) < -0.1):
            return "trend confirmation+momentum support(bearish)"
        elif current_data.get('rsi_14', 50) > 65:
            return "overbought"
        elif (current_data.get('close', 0) < current_data.get('supertrend', 0)):
            return "breakout confirm signal(bearish)"
        else:
            return "signal(bearish)"
    
    def calculate_signal_strength(self, dataframe: DataFrame) -> DataFrame:
        """signal strength calculate - score"""
        
        # === 1. trend signal strength (weight35%) ===
        # translatedADXconfirm trend strength
        trend_signal = np.where(
            (dataframe['trend_strength'] > 70) & (dataframe['adx'] > 30), 3,  # strong trend
            np.where(
                (dataframe['trend_strength'] > 50) & (dataframe['adx'] > 25), 2,  # strong trend
                np.where(
                    (dataframe['trend_strength'] > 30) & (dataframe['adx'] > 20), 1,  # medium trend
                    np.where(
                        (dataframe['trend_strength'] < -70) & (dataframe['adx'] > 30), -3,  # down
                        np.where(
                            (dataframe['trend_strength'] < -50) & (dataframe['adx'] > 25), -2,  # down
                            np.where(
                                (dataframe['trend_strength'] < -30) & (dataframe['adx'] > 20), -1, 0  # mid down
                            )
                        )
                    )
                )
            )
        ) * 0.35
        
        # === 2. momentum signal strength (weight30%) ===
        # MACD + RSI + price momentum
        macd_momentum = np.where(
            (dataframe['macd'] > dataframe['macd_signal']) & (dataframe['macd_hist'] > 0), 1,
            np.where((dataframe['macd'] < dataframe['macd_signal']) & (dataframe['macd_hist'] < 0), -1, 0)
        )
        
        rsi_momentum = np.where(
            dataframe['rsi_14'] > 60, 1,
            np.where(dataframe['rsi_14'] < 40, -1, 0)
        )
        
        price_momentum = np.where(
            dataframe['momentum_score'] > 0.2, 2,
            np.where(
                dataframe['momentum_score'] > 0.1, 1,
                np.where(
                    dataframe['momentum_score'] < -0.2, -2,
                    np.where(dataframe['momentum_score'] < -0.1, -1, 0)
                )
            )
        )
        
        momentum_signal = (macd_momentum + rsi_momentum + price_momentum) * 0.30
        
        # === 3. volume confirmation signal (weight20%) ===
        volume_signal = np.where(
            dataframe['volume_ratio'] > 2.0, 2,  # abnormal volume spike
            np.where(
                dataframe['volume_ratio'] > 1.5, 1,  # translated
                np.where(
                    dataframe['volume_ratio'] < 0.6, -1,  # translated
                    0
                )
            )
        ) * 0.20
        
        # === 4. signal (weight10%) ===
        microstructure_signal = np.where(
            (dataframe['ob_depth_imbalance'] > 0.2) & (dataframe['ob_market_quality'] > 0.5), 1,  # translated
            np.where(
                (dataframe['ob_depth_imbalance'] < -0.2) & (dataframe['ob_market_quality'] > 0.5), -1,  # translated
                0
            )
        ) * 0.10
        
        # === 5. breakout confirm (weight5%) ===
        breakout_signal = np.where(
            (dataframe['close'] > dataframe['supertrend']) & (dataframe['bb_position'] > 0.6), 1,  # up breakout
            np.where(
                (dataframe['close'] < dataframe['supertrend']) & (dataframe['bb_position'] < 0.4), -1,  # down breakout
                0
            )
        ) * 0.05
        
        # === signal strength ===
        dataframe['signal_strength'] = (trend_signal + momentum_signal + volume_signal + 
                                      microstructure_signal + breakout_signal)
        
        # === signal ===
        # confirm signal high
        confirmation_count = (
            (np.abs(trend_signal) > 0).astype(int) +
            (np.abs(momentum_signal) > 0).astype(int) +
            (np.abs(volume_signal) > 0).astype(int) +
            (np.abs(microstructure_signal) > 0).astype(int)
        )
        
        # signal
        quality_multiplier = np.where(
            confirmation_count >= 3, 1.3,  # 3 confirm
            np.where(confirmation_count >= 2, 1.1, 0.8)  # confirm
        )
        
        dataframe['signal_strength'] = dataframe['signal_strength'] * quality_multiplier
        
        # optimize：translatedDataFrameto avoidPerformanceWarning
        dataframe = dataframe.copy()
        
        return dataframe
    
    # ===== and =====
    
    def initialize_monitoring_system(self):
        """initialize"""
        self.monitoring_enabled = True
        self.performance_window = 100  # translated
        self.adaptation_threshold = 0.1  # value
        self.last_monitoring_time = datetime.now(timezone.utc)
        self.monitoring_interval = 300  # 5minutes
        
        # indicator
        self.performance_metrics = {
            'win_rate': [],
            'profit_factor': [],
            'sharpe_ratio': [],
            'max_drawdown': [],
            'avg_trade_duration': [],
            'volatility': []
        }
        
        # market state
        self.market_regime_history = []
        self.volatility_regime_history = []
        
        # record
        self.parameter_adjustments = []
        
        # risk value
        self.risk_thresholds = {
            'max_daily_loss': -0.05,  # day most large loss5%
            'max_drawdown': -0.15,    # most large drawdown15%
            'min_win_rate': 0.35,     # most low35%
            'max_volatility': 0.25,   # most large volatility25%
            'max_correlation': 0.8    # most large80%
        }
        
    def monitor_real_time_performance(self) -> Dict[str, Any]:
        """translated"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # check
            if (current_time - self.last_monitoring_time).seconds < self.monitoring_interval:
                return {}
            
            self.last_monitoring_time = current_time
            
            # get before indicator
            current_metrics = self.calculate_current_performance_metrics()
            
            # update
            self.update_performance_history(current_metrics)
            
            # risk check
            risk_alerts = self.check_risk_thresholds(current_metrics)
            
            # market state
            market_state = self.monitor_market_regime()
            
            # check
            adaptation_needed = self.check_adaptation_requirements(current_metrics)
            
            monitoring_report = {
                'timestamp': current_time,
                'performance_metrics': current_metrics,
                'risk_alerts': risk_alerts,
                'market_state': market_state,
                'adaptation_needed': adaptation_needed,
                'monitoring_status': 'active'
            }
            
            # translated，translated
            if adaptation_needed:
                self.execute_adaptive_adjustments(current_metrics, market_state)
            
            return monitoring_report
            
        except Exception as e:
            return {'error': f'translated: {str(e)}', 'monitoring_status': 'error'}
    
    def calculate_current_performance_metrics(self) -> Dict[str, float]:
        """calculate before indicator"""
        try:
            # get most record
            recent_trades = self.get_recent_trades(self.performance_window)
            
            if not recent_trades:
                return {
                    'win_rate': 0.0,
                    'profit_factor': 0.0,
                    'sharpe_ratio': 0.0,
                    'max_drawdown': 0.0,
                    'avg_trade_duration': 0.0,
                    'volatility': 0.0,
                    'total_trades': 0
                }
            
            # calculate
            profitable_trades = [t for t in recent_trades if t['profit'] > 0]
            win_rate = len(profitable_trades) / len(recent_trades)
            
            # calculate profit
            total_profit = sum([t['profit'] for t in profitable_trades])
            total_loss = abs(sum([t['profit'] for t in recent_trades if t['profit'] < 0]))
            profit_factor = total_profit / total_loss if total_loss > 0 else 0
            
            # calculate
            returns = [t['profit'] for t in recent_trades]
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = avg_return / std_return if std_return > 0 else 0
            
            # calculate most large drawdown
            cumulative_returns = np.cumsum(returns)
            running_max = np.maximum.accumulate(cumulative_returns)
            drawdown = cumulative_returns - running_max
            max_drawdown = np.min(drawdown)
            
            # time
            durations = [t.get('duration_hours', 0) for t in recent_trades]
            avg_trade_duration = np.mean(durations)
            
            # volatility
            volatility = std_return
            
            return {
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'avg_trade_duration': avg_trade_duration,
                'volatility': volatility,
                'total_trades': len(recent_trades)
            }
            
        except Exception:
            return {
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'avg_trade_duration': 0.0,
                'volatility': 0.0,
                'total_trades': 0
            }
    
    def get_recent_trades(self, window_size: int) -> List[Dict]:
        """get most record"""
        try:
            # from mid get
            # translated
            return []
        except Exception:
            return []
    
    def update_performance_history(self, metrics: Dict[str, float]):
        """update record"""
        try:
            for key, value in metrics.items():
                if key in self.performance_metrics:
                    self.performance_metrics[key].append(value)
                    
                    # record at long
                    if len(self.performance_metrics[key]) > 1000:
                        self.performance_metrics[key] = self.performance_metrics[key][-500:]
        except Exception:
            pass
    
    def check_risk_thresholds(self, metrics: Dict[str, float]) -> List[Dict[str, Any]]:
        """check risk value"""
        alerts = []
        
        try:
            # check
            if metrics['win_rate'] < self.risk_thresholds['min_win_rate']:
                alerts.append({
                    'type': 'low_win_rate',
                    'severity': 'warning',
                    'current_value': metrics['win_rate'],
                    'threshold': self.risk_thresholds['min_win_rate'],
                    'message': f"low: {metrics['win_rate']:.1%} < {self.risk_thresholds['min_win_rate']:.1%}"
                })
            
            # check most large drawdown
            if metrics['max_drawdown'] < self.risk_thresholds['max_drawdown']:
                alerts.append({
                    'type': 'high_drawdown',
                    'severity': 'critical',
                    'current_value': metrics['max_drawdown'],
                    'threshold': self.risk_thresholds['max_drawdown'],
                    'message': f"drawdown large: {metrics['max_drawdown']:.1%} < {self.risk_thresholds['max_drawdown']:.1%}"
                })
            
            # check volatility
            if metrics['volatility'] > self.risk_thresholds['max_volatility']:
                alerts.append({
                    'type': 'high_volatility',
                    'severity': 'warning',
                    'current_value': metrics['volatility'],
                    'threshold': self.risk_thresholds['max_volatility'],
                    'message': f"volatility high: {metrics['volatility']:.1%} > {self.risk_thresholds['max_volatility']:.1%}"
                })
                
        except Exception:
            pass
        
        return alerts
    
    def monitor_market_regime(self) -> Dict[str, Any]:
        """market state changed"""
        try:
            # get before indicator
            current_regime = {
                'trend_strength': 0.0,
                'volatility_level': 0.0,
                'market_state': 'unknown',
                'regime_stability': 0.0
            }
            
            # get
            # default
            
            return current_regime
            
        except Exception:
            return {
                'trend_strength': 0.0,
                'volatility_level': 0.0,
                'market_state': 'unknown',
                'regime_stability': 0.0
            }
    
    def check_adaptation_requirements(self, metrics: Dict[str, float]) -> bool:
        """check"""
        try:
            # fall
            if len(self.performance_metrics['win_rate']) > 50:
                recent_win_rate = np.mean(self.performance_metrics['win_rate'][-20:])
                historical_win_rate = np.mean(self.performance_metrics['win_rate'][-50:-20])
                
                if historical_win_rate > 0 and (recent_win_rate / historical_win_rate) < 0.8:
                    return True
            
            # translated
            if len(self.performance_metrics['sharpe_ratio']) > 50:
                recent_sharpe = np.mean(self.performance_metrics['sharpe_ratio'][-20:])
                if recent_sharpe < 0.5:  # low
                    return True
            
            # drawdown large
            if metrics['max_drawdown'] < -0.12:  # translated12%drawdown
                return True
            
            return False
            
        except Exception:
            return False
    
    def execute_adaptive_adjustments(self, metrics: Dict[str, float], market_state: Dict[str, Any]):
        """translated"""
        try:
            adjustments = []
            
            # translated
            if metrics['win_rate'] < 0.4:
                # low position size large small
                self.base_position_size *= 0.8
                adjustments.append('reduced_position_size')
                
                # translated
                self.stoploss *= 1.1
                adjustments.append('tightened_stoploss')
            
            # volatility
            if metrics['volatility'] > 0.2:
                # low most large leverage
                self.leverage_multiplier = max(3, self.leverage_multiplier - 1)
                adjustments.append('reduced_leverage')
            
            # drawdown
            if metrics['max_drawdown'] < -0.1:
                # risk
                self.drawdown_protection *= 0.8
                adjustments.append('enhanced_drawdown_protection')
            
            # record
            adjustment_record = {
                'timestamp': datetime.now(timezone.utc),
                'trigger_metrics': metrics,
                'market_state': market_state,
                'adjustments': adjustments
            }
            
            self.parameter_adjustments.append(adjustment_record)
            
            # at long
            if len(self.parameter_adjustments) > 100:
                self.parameter_adjustments = self.parameter_adjustments[-50:]
                
        except Exception:
            pass
    
    def get_monitoring_status(self) -> Dict[str, Any]:
        """get"""
        try:
            return {
                'monitoring_enabled': self.monitoring_enabled,
                'last_monitoring_time': self.last_monitoring_time,
                'performance_metrics_count': len(self.performance_metrics.get('win_rate', [])),
                'total_adjustments': len(self.parameter_adjustments),
                'current_parameters': {
                    'base_position_size': self.base_position_size,
                    'leverage_multiplier': self.leverage_multiplier,
                    'stoploss': self.stoploss,
                    'drawdown_protection': self.drawdown_protection
                }
            }
        except Exception:
            return {'error': 'none get'}
    
    # ===== translated =====
    
    def initialize_risk_control_system(self):
        """initialize"""
        # translated
        self.risk_control_enabled = True
        self.emergency_mode = False
        self.circuit_breaker_active = False
        
        # risk
        self.risk_budgets = {
            'daily_var_budget': 0.02,      # dayVaRtranslated2%
            'weekly_var_budget': 0.05,     # translatedVaRtranslated5%
            'monthly_var_budget': 0.12,    # translatedVaRtranslated12%
            'position_var_limit': 0.01,    # translatedVaRtranslated1%
            'correlation_limit': 0.7,      # translated70%
            'sector_exposure_limit': 0.3   # translated30%
        }
        
        # risk use
        self.risk_utilization = {
            'current_daily_var': 0.0,
            'current_weekly_var': 0.0,
            'current_monthly_var': 0.0,
            'used_correlation_capacity': 0.0,
            'sector_exposures': {}
        }
        
        # value
        self.circuit_breakers = {
            'daily_loss_limit': -0.08,      # day loss8%
            'hourly_loss_limit': -0.03,     # hours loss3%
            'consecutive_loss_limit': 6,     # loss
            'drawdown_limit': -0.20,        # most large drawdown20%
            'volatility_spike_limit': 5.0,  # volatility
            'correlation_spike_limit': 0.9  # translated
        }
        
        # risk record
        self.risk_events = []
        self.emergency_actions = []
        
        # risk
        self.last_risk_check_time = datetime.now(timezone.utc)
        self.risk_check_interval = 60  # check60translated
        
    def comprehensive_risk_check(self, pair: str, current_price: float, 
                               proposed_position_size: float, 
                               proposed_leverage: int) -> Dict[str, Any]:
        """risk check - translated"""
        
        risk_status = {
            'approved': True,
            'adjusted_position_size': proposed_position_size,
            'adjusted_leverage': proposed_leverage,
            'risk_warnings': [],
            'risk_violations': [],
            'emergency_action': None
        }
        
        try:
            current_time = datetime.now(timezone.utc)
            
            # 1. check
            circuit_breaker_result = self.check_circuit_breakers()
            if circuit_breaker_result['triggered']:
                risk_status['approved'] = False
                risk_status['emergency_action'] = 'circuit_breaker_halt'
                risk_status['risk_violations'].append(circuit_breaker_result)
                return risk_status
            
            # 2. VaRcheck
            var_check_result = self.check_var_budget_limits(pair, proposed_position_size)
            if not var_check_result['within_limits']:
                risk_status['adjusted_position_size'] *= var_check_result['adjustment_factor']
                risk_status['risk_warnings'].append(var_check_result)
            
            # 3. check
            correlation_check_result = self.check_correlation_limits(pair, proposed_position_size)
            if not correlation_check_result['within_limits']:
                risk_status['adjusted_position_size'] *= correlation_check_result['adjustment_factor']
                risk_status['risk_warnings'].append(correlation_check_result)
            
            # 4. mid risk check
            concentration_check_result = self.check_concentration_risk(pair, proposed_position_size)
            if not concentration_check_result['within_limits']:
                risk_status['adjusted_position_size'] *= concentration_check_result['adjustment_factor']
                risk_status['risk_warnings'].append(concentration_check_result)
            
            # 5. risk check
            liquidity_check_result = self.check_liquidity_risk(pair, proposed_position_size)
            if not liquidity_check_result['sufficient_liquidity']:
                risk_status['adjusted_position_size'] *= liquidity_check_result['adjustment_factor']
                risk_status['risk_warnings'].append(liquidity_check_result)
            
            # 6. leverage risk check
            leverage_check_result = self.check_leverage_risk(pair, proposed_leverage)
            if not leverage_check_result['within_limits']:
                risk_status['adjusted_leverage'] = leverage_check_result['max_allowed_leverage']
                risk_status['risk_warnings'].append(leverage_check_result)
            
            # 7. time risk check
            time_risk_result = self.check_time_based_risk(current_time)
            if time_risk_result['high_risk_period']:
                risk_status['adjusted_position_size'] *= time_risk_result['adjustment_factor']
                risk_status['risk_warnings'].append(time_risk_result)
            
            # most not most small/most large
            risk_status['adjusted_position_size'] = max(
                0.005, 
                min(risk_status['adjusted_position_size'], self.max_position_size * 0.8)
            )
            
            # record risk check
            self.record_risk_event('risk_check', risk_status)
            
        except Exception as e:
            risk_status['approved'] = False
            risk_status['emergency_action'] = 'system_error'
            risk_status['risk_violations'].append({
                'type': 'system_error',
                'message': f'translated: {str(e)}'
            })
        
        return risk_status
    
    def check_circuit_breakers(self) -> Dict[str, Any]:
        """check"""
        try:
            current_time = datetime.now(timezone.utc)
            
            # get before
            current_equity = getattr(self, 'current_equity', 100000)  # default value
            daily_pnl = getattr(self, 'daily_pnl', 0)
            hourly_pnl = getattr(self, 'hourly_pnl', 0)
            
            # 1. day loss
            daily_loss_pct = daily_pnl / current_equity if current_equity > 0 else 0
            if daily_loss_pct < self.circuit_breakers['daily_loss_limit']:
                return {
                    'triggered': True,
                    'type': 'daily_loss_circuit_breaker',
                    'current_value': daily_loss_pct,
                    'limit': self.circuit_breakers['daily_loss_limit'],
                    'message': f'day loss: {daily_loss_pct:.2%}'
                }
            
            # 2. hours loss
            hourly_loss_pct = hourly_pnl / current_equity if current_equity > 0 else 0
            if hourly_loss_pct < self.circuit_breakers['hourly_loss_limit']:
                return {
                    'triggered': True,
                    'type': 'hourly_loss_circuit_breaker',
                    'current_value': hourly_loss_pct,
                    'limit': self.circuit_breakers['hourly_loss_limit'],
                    'message': f'hours loss: {hourly_loss_pct:.2%}'
                }
            
            # 3. loss
            if self.consecutive_losses >= self.circuit_breakers['consecutive_loss_limit']:
                return {
                    'triggered': True,
                    'type': 'consecutive_loss_circuit_breaker',
                    'current_value': self.consecutive_losses,
                    'limit': self.circuit_breakers['consecutive_loss_limit'],
                    'message': f'loss: {self.consecutive_losses}times'
                }
            
            # 4. most large drawdown
            max_drawdown = getattr(self, 'current_max_drawdown', 0)
            if max_drawdown < self.circuit_breakers['drawdown_limit']:
                return {
                    'triggered': True,
                    'type': 'drawdown_circuit_breaker',
                    'current_value': max_drawdown,
                    'limit': self.circuit_breakers['drawdown_limit'],
                    'message': f'drawdown: {max_drawdown:.2%}'
                }
            
            return {'triggered': False, 'type': None, 'message': 'translated'}
            
        except Exception:
            return {
                'triggered': True,
                'type': 'circuit_breaker_error',
                'message': 'check'
            }
    
    def check_var_budget_limits(self, pair: str, position_size: float) -> Dict[str, Any]:
        """VaRcheck"""
        try:
            # calculate new position sizeVaRtranslated
            position_var = self.calculate_position_var(pair, position_size)
            
            # checkVaRtranslated
            current_daily_var = self.risk_utilization['current_daily_var']
            new_daily_var = current_daily_var + position_var
            
            if new_daily_var > self.risk_budgets['daily_var_budget']:
                # calculate allow most large position size
                available_var_budget = self.risk_budgets['daily_var_budget'] - current_daily_var
                max_allowed_position = available_var_budget / position_var * position_size if position_var > 0 else position_size
                
                adjustment_factor = max(0.1, max_allowed_position / position_size)
                
                return {
                    'within_limits': False,
                    'type': 'var_budget_exceeded',
                    'adjustment_factor': adjustment_factor,
                    'current_utilization': new_daily_var,
                    'budget_limit': self.risk_budgets['daily_var_budget'],
                    'message': f'VaRtranslated，position adjustment as{adjustment_factor:.1%}'
                }
            
            return {
                'within_limits': True,
                'type': 'var_budget_check',
                'utilization': new_daily_var / self.risk_budgets['daily_var_budget'],
                'message': 'VaRcheck'
            }
            
        except Exception:
            return {
                'within_limits': False,
                'adjustment_factor': 0.5,
                'message': 'VaRcheck，position size'
            }
    
    def calculate_position_var(self, pair: str, position_size: float) -> float:
        """calculate position sizeVaRtranslated"""
        try:
            if pair in self.pair_returns_history and len(self.pair_returns_history[pair]) >= 20:
                returns = self.pair_returns_history[pair]
                position_var = self.calculate_var(returns) * position_size
                return min(position_var, self.risk_budgets['position_var_limit'])
            else:
                # default risk
                return position_size * 0.02  # translated2%defaultVaR
        except Exception:
            return position_size * 0.03  # translated
    
    def check_correlation_limits(self, pair: str, position_size: float) -> Dict[str, Any]:
        """check"""
        try:
            current_correlation = self.calculate_portfolio_correlation(pair)
            
            if current_correlation > self.risk_budgets['correlation_limit']:
                # position size
                excess_correlation = current_correlation - self.risk_budgets['correlation_limit']
                adjustment_factor = max(0.2, 1 - (excess_correlation * 2))
                
                return {
                    'within_limits': False,
                    'type': 'correlation_limit_exceeded',
                    'adjustment_factor': adjustment_factor,
                    'current_correlation': current_correlation,
                    'limit': self.risk_budgets['correlation_limit'],
                    'message': f'translated({current_correlation:.1%})，position adjustment as{adjustment_factor:.1%}'
                }
            
            return {
                'within_limits': True,
                'type': 'correlation_check',
                'current_correlation': current_correlation,
                'message': 'check'
            }
            
        except Exception:
            return {
                'within_limits': False,
                'adjustment_factor': 0.7,
                'message': 'check，translated'
            }
    
    def check_concentration_risk(self, pair: str, position_size: float) -> Dict[str, Any]:
        """mid risk check"""
        try:
            # check 1 mid
            current_positions = getattr(self, 'portfolio_positions', {})
            total_exposure = sum([abs(pos) for pos in current_positions.values()])
            
            if pair in current_positions:
                new_exposure = current_positions[pair] + position_size
            else:
                new_exposure = position_size
            
            if total_exposure > 0:
                concentration_ratio = abs(new_exposure) / (total_exposure + position_size)
            else:
                concentration_ratio = 1.0
            
            max_single_position_ratio = 0.4  # 1 most large40%
            
            if concentration_ratio > max_single_position_ratio:
                adjustment_factor = max_single_position_ratio / concentration_ratio
                
                return {
                    'within_limits': False,
                    'type': 'concentration_risk_exceeded',
                    'adjustment_factor': adjustment_factor,
                    'concentration_ratio': concentration_ratio,
                    'limit': max_single_position_ratio,
                    'message': f'mid risk({concentration_ratio:.1%})，position size'
                }
            
            return {
                'within_limits': True,
                'type': 'concentration_check',
                'concentration_ratio': concentration_ratio,
                'message': 'mid risk check'
            }
            
        except Exception:
            return {
                'within_limits': False,
                'adjustment_factor': 0.6,
                'message': 'mid check，translated'
            }
    
    def check_liquidity_risk(self, pair: str, position_size: float) -> Dict[str, Any]:
        """risk check"""
        try:
            # get indicator
            market_data = getattr(self, 'current_market_data', {})
            
            if pair in market_data:
                volume_ratio = market_data[pair].get('volume_ratio', 1.0)
                spread = market_data[pair].get('spread', 0.001)
            else:
                volume_ratio = 1.0  # default value
                spread = 0.002
            
            # risk
            liquidity_risk_score = 0.0
            
            # volume risk
            if volume_ratio < 0.5:  # volume low
                liquidity_risk_score += 0.3
            elif volume_ratio < 0.8:
                liquidity_risk_score += 0.1
            
            # risk
            if spread > 0.005:  # large
                liquidity_risk_score += 0.4
            elif spread > 0.003:
                liquidity_risk_score += 0.2
            
            if liquidity_risk_score > 0.5:  # risk high
                adjustment_factor = max(0.3, 1 - liquidity_risk_score)
                
                return {
                    'sufficient_liquidity': False,
                    'type': 'liquidity_risk_high',
                    'adjustment_factor': adjustment_factor,
                    'risk_score': liquidity_risk_score,
                    'volume_ratio': volume_ratio,
                    'spread': spread,
                    'message': f'risk high({liquidity_risk_score:.1f})，position size'
                }
            
            return {
                'sufficient_liquidity': True,
                'type': 'liquidity_check',
                'risk_score': liquidity_risk_score,
                'message': 'risk check'
            }
            
        except Exception:
            return {
                'sufficient_liquidity': False,
                'adjustment_factor': 0.5,
                'message': 'check，translated'
            }
    
    def check_leverage_risk(self, pair: str, proposed_leverage: int) -> Dict[str, Any]:
        """leverage risk check"""
        try:
            # market state volatility leverage
            market_volatility = getattr(self, 'current_market_volatility', {}).get(pair, 0.02)
            
            # leverage
            if market_volatility > 0.05:  # high volatility
                max_allowed_leverage = min(5, self.leverage_multiplier)
            elif market_volatility > 0.03:  # mid
                max_allowed_leverage = min(8, self.leverage_multiplier)
            else:  # low volatility
                max_allowed_leverage = self.leverage_multiplier
            
            if proposed_leverage > max_allowed_leverage:
                return {
                    'within_limits': False,
                    'type': 'leverage_risk_exceeded',
                    'max_allowed_leverage': max_allowed_leverage,
                    'proposed_leverage': proposed_leverage,
                    'market_volatility': market_volatility,
                    'message': f'leverage risk high，as{max_allowed_leverage}translated'
                }
            
            return {
                'within_limits': True,
                'type': 'leverage_check',
                'approved_leverage': proposed_leverage,
                'message': 'leverage risk check'
            }
            
        except Exception:
            return {
                'within_limits': False,
                'max_allowed_leverage': min(3, proposed_leverage),
                'message': 'leverage check，translated'
            }
    
    def check_time_based_risk(self, current_time: datetime) -> Dict[str, Any]:
        """time risk check"""
        try:
            hour = current_time.hour
            weekday = current_time.weekday()
            
            high_risk_periods = [
                (weekday >= 5),  # translated
                (hour <= 6 or hour >= 22),  # translated
                (11 <= hour <= 13),  # translated
            ]
            
            if any(high_risk_periods):
                adjustment_factor = 0.7  # high risk small position size
                
                return {
                    'high_risk_period': True,
                    'type': 'time_based_risk',
                    'adjustment_factor': adjustment_factor,
                    'hour': hour,
                    'weekday': weekday,
                    'message': 'high risk，position size'
                }
            
            return {
                'high_risk_period': False,
                'type': 'time_check',
                'adjustment_factor': 1.0,
                'message': 'time risk check'
            }
            
        except Exception:
            return {
                'high_risk_period': True,
                'adjustment_factor': 0.8,
                'message': 'time check，translated'
            }
    
    def record_risk_event(self, event_type: str, event_data: Dict[str, Any]):
        """record risk"""
        try:
            risk_event = {
                'timestamp': datetime.now(timezone.utc),
                'event_type': event_type,
                'event_data': event_data,
                'severity': self.determine_event_severity(event_data)
            }
            
            self.risk_events.append(risk_event)
            
            # record at long
            if len(self.risk_events) > 1000:
                self.risk_events = self.risk_events[-500:]
                
        except Exception:
            pass
    
    def determine_event_severity(self, event_data: Dict[str, Any]) -> str:
        """translated"""
        try:
            if not event_data.get('approved', True):
                return 'critical'
            elif event_data.get('emergency_action'):
                return 'high'
            elif len(event_data.get('risk_violations', [])) > 0:
                return 'medium'
            elif len(event_data.get('risk_warnings', [])) > 2:
                return 'medium'
            elif len(event_data.get('risk_warnings', [])) > 0:
                return 'low'
            else:
                return 'info'
        except Exception:
            return 'unknown'
    
    def emergency_risk_shutdown(self, reason: str):
        """translated"""
        try:
            self.emergency_mode = True
            self.circuit_breaker_active = True
            
            emergency_action = {
                'timestamp': datetime.now(timezone.utc),
                'reason': reason,
                'action': 'emergency_shutdown',
                'open_positions_count': len(getattr(self, 'portfolio_positions', {})),
                'total_exposure': sum([abs(pos) for pos in getattr(self, 'portfolio_positions', {}).values()])
            }
            
            self.emergency_actions.append(emergency_action)
            
            # translated
            # record
            
        except Exception:
            pass
    
    def get_risk_control_status(self) -> Dict[str, Any]:
        """get"""
        try:
            return {
                'risk_control_enabled': self.risk_control_enabled,
                'emergency_mode': self.emergency_mode,
                'circuit_breaker_active': self.circuit_breaker_active,
                'risk_budgets': self.risk_budgets,
                'risk_utilization': self.risk_utilization,
                'recent_risk_events': len(self.risk_events[-24:]) if self.risk_events else 0,
                'emergency_actions_count': len(self.emergency_actions),
                'last_risk_check': self.last_risk_check_time
            }
        except Exception:
            return {'error': 'none get'}
    
    # ===== and =====
    
    def initialize_execution_system(self):
        """initialize"""
        # configuration
        self.execution_algorithms = {
            'twap': {'enabled': True, 'weight': 0.3},      # time price
            'vwap': {'enabled': True, 'weight': 0.4},      # volume price
            'implementation_shortfall': {'enabled': True, 'weight': 0.3}  # most small
        }
        
        # translated
        self.slippage_control = {
            'max_allowed_slippage': 0.002,    # most large allow0.2%
            'slippage_prediction_window': 50,  # translated
            'adaptive_threshold': 0.001,      # value0.1%
            'emergency_threshold': 0.005      # value0.5%
        }
        
        # translated
        self.order_splitting = {
            'min_split_size': 0.01,           # most small large small1%
            'max_split_count': 10,            # most large
            'split_interval_seconds': 30,     # translated30translated
            'adaptive_splitting': True        # translated
        }
        
        # translated
        self.execution_metrics = {
            'realized_slippage': [],
            'market_impact': [],
            'execution_time': [],
            'fill_ratio': [],
            'cost_basis_deviation': []
        }
        
        # translated
        self.market_impact_model = {
            'temporary_impact_factor': 0.5,   # translated
            'permanent_impact_factor': 0.3,   # translated
            'nonlinear_factor': 1.5,          # translated
            'decay_factor': 0.1               # translated
        }
        
        # translated
        self.active_executions = {}
        self.execution_history = []
        
    def smart_order_execution(self, pair: str, order_size: float, order_side: str, 
                            current_price: float, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """translated"""
        
        execution_plan = {
            'original_size': order_size,
            'execution_strategy': None,
            'split_orders': [],
            'expected_slippage': 0.0,
            'estimated_execution_time': 0,
            'risk_level': 'normal'
        }
        
        try:
            # 1. risk
            execution_risk = self.assess_execution_risk(pair, order_size, market_conditions)
            execution_plan['risk_level'] = execution_risk['level']
            
            # 2. translated
            predicted_slippage = self.predict_slippage(pair, order_size, order_side, market_conditions)
            execution_plan['expected_slippage'] = predicted_slippage
            
            # 3. translated
            optimal_algorithm = self.select_execution_algorithm(pair, order_size, market_conditions, execution_risk)
            execution_plan['execution_strategy'] = optimal_algorithm
            
            # 4. optimize
            if order_size > self.order_splitting['min_split_size'] and execution_risk['level'] != 'low':
                split_plan = self.optimize_order_splitting(pair, order_size, market_conditions, optimal_algorithm)
                execution_plan['split_orders'] = split_plan['orders']
                execution_plan['estimated_execution_time'] = split_plan['total_time']
            else:
                execution_plan['split_orders'] = [{'size': order_size, 'delay': 0, 'priority': 'high'}]
                execution_plan['estimated_execution_time'] = 30  # translated30translated
            
            # 5. optimize
            execution_timing = self.optimize_execution_timing(pair, market_conditions)
            execution_plan['optimal_timing'] = execution_timing
            
            # 6. translated
            execution_instructions = self.generate_execution_instructions(execution_plan, pair, order_side, current_price)
            execution_plan['instructions'] = execution_instructions
            
            return execution_plan
            
        except Exception as e:
            # to
            return {
                'original_size': order_size,
                'execution_strategy': 'immediate',
                'split_orders': [{'size': order_size, 'delay': 0, 'priority': 'high'}],
                'expected_slippage': 0.002,  # translated
                'estimated_execution_time': 30,
                'risk_level': 'unknown',
                'error': str(e)
            }
    
    def assess_execution_risk(self, pair: str, order_size: float, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """risk"""
        try:
            risk_score = 0.0
            risk_factors = []
            
            # 1. large small risk
            avg_volume = market_conditions.get('avg_volume', 1.0)
            order_volume_ratio = order_size / avg_volume if avg_volume > 0 else 1.0
            
            if order_volume_ratio > 0.1:  # translated10%volume
                risk_score += 0.4
                risk_factors.append('large_order_size')
            elif order_volume_ratio > 0.05:
                risk_score += 0.2
                risk_factors.append('medium_order_size')
            
            # 2. risk
            volatility = market_conditions.get('volatility', 0.02)
            if volatility > 0.05:
                risk_score += 0.3
                risk_factors.append('high_volatility')
            elif volatility > 0.03:
                risk_score += 0.15
                risk_factors.append('medium_volatility')
            
            # 3. risk
            bid_ask_spread = market_conditions.get('spread', 0.001)
            if bid_ask_spread > 0.003:
                risk_score += 0.2
                risk_factors.append('wide_spread')
            
            # 4. time risk
            if self.is_high_volatility_session(datetime.now(timezone.utc)):
                risk_score += 0.1
                risk_factors.append('high_volatility_session')
            
            # risk level
            if risk_score < 0.3:
                risk_level = 'low'
            elif risk_score < 0.6:
                risk_level = 'medium'
            else:
                risk_level = 'high'
            
            return {
                'level': risk_level,
                'score': risk_score,
                'factors': risk_factors,
                'order_volume_ratio': order_volume_ratio
            }
            
        except Exception:
            return {
                'level': 'medium',
                'score': 0.5,
                'factors': ['assessment_error'],
                'order_volume_ratio': 0.1
            }
    
    def predict_slippage(self, pair: str, order_size: float, order_side: str, 
                        market_conditions: Dict[str, Any]) -> float:
        """translated"""
        try:
            # translated
            base_slippage = market_conditions.get('spread', 0.001) / 2  # count
            
            # translated
            avg_volume = market_conditions.get('avg_volume', 1.0)
            volume_ratio = order_size / avg_volume if avg_volume > 0 else 0.1
            
            # translated
            temporary_impact = (
                self.market_impact_model['temporary_impact_factor'] * 
                (volume_ratio ** self.market_impact_model['nonlinear_factor'])
            )
            
            # translated
            permanent_impact = (
                self.market_impact_model['permanent_impact_factor'] * 
                (volume_ratio ** 0.5)
            )
            
            # volatility
            volatility = market_conditions.get('volatility', 0.02)
            volatility_adjustment = min(1.0, volatility * 10)  # volatility high large
            
            # time
            time_adjustment = 1.0
            if self.is_high_volatility_session(datetime.now(timezone.utc)):
                time_adjustment = 1.2
            elif self.is_low_liquidity_session(datetime.now(timezone.utc)):
                time_adjustment = 1.3
            
            # translated
            historical_slippage = self.get_historical_slippage(pair)
            historical_adjustment = max(0.5, min(2.0, historical_slippage / 0.001))
            
            # translated
            predicted_slippage = (
                base_slippage + temporary_impact + permanent_impact
            ) * volatility_adjustment * time_adjustment * historical_adjustment
            
            # at range
            predicted_slippage = min(predicted_slippage, self.slippage_control['emergency_threshold'])
            
            return max(0.0001, predicted_slippage)  # most small0.01%
            
        except Exception:
            return 0.002  # translated0.2%
    
    def get_historical_slippage(self, pair: str) -> float:
        """get"""
        try:
            if len(self.execution_metrics['realized_slippage']) > 0:
                recent_slippage = self.execution_metrics['realized_slippage'][-20:]  # most20times
                return np.mean(recent_slippage)
            else:
                return 0.001  # default0.1%
        except Exception:
            return 0.001
    
    def select_execution_algorithm(self, pair: str, order_size: float, 
                                 market_conditions: Dict[str, Any], 
                                 execution_risk: Dict[str, Any]) -> str:
        """most"""
        try:
            algorithm_scores = {}
            
            # TWAPscore
            if self.execution_algorithms['twap']['enabled']:
                twap_score = 0.5  # translated
                
                # time low
                if execution_risk['level'] == 'low':
                    twap_score += 0.2
                
                # translated
                if market_conditions.get('volatility', 0.02) < 0.025:
                    twap_score += 0.1
                
                algorithm_scores['twap'] = twap_score * self.execution_algorithms['twap']['weight']
            
            # VWAPscore
            if self.execution_algorithms['vwap']['enabled']:
                vwap_score = 0.6  # translated
                
                # volume
                if market_conditions.get('volume_ratio', 1.0) > 1.0:
                    vwap_score += 0.2
                
                # medium risk most
                if execution_risk['level'] == 'medium':
                    vwap_score += 0.15
                
                algorithm_scores['vwap'] = vwap_score * self.execution_algorithms['vwap']['weight']
            
            # Implementation Shortfallscore
            if self.execution_algorithms['implementation_shortfall']['enabled']:
                is_score = 0.4  # translated
                
                # high risk
                if execution_risk['level'] == 'high':
                    is_score += 0.3
                
                # large
                if execution_risk.get('order_volume_ratio', 0.1) > 0.05:
                    is_score += 0.2
                
                # high volatility
                if market_conditions.get('volatility', 0.02) > 0.03:
                    is_score += 0.1
                
                algorithm_scores['implementation_shortfall'] = is_score * self.execution_algorithms['implementation_shortfall']['weight']
            
            # most high
            if algorithm_scores:
                optimal_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])[0]
                return optimal_algorithm
            else:
                return 'twap'  # default
                
        except Exception:
            return 'twap'  # toTWAP
    
    def optimize_order_splitting(self, pair: str, order_size: float, 
                               market_conditions: Dict[str, Any], 
                               algorithm: str) -> Dict[str, Any]:
        """optimize"""
        try:
            split_plan = {
                'orders': [],
                'total_time': 0,
                'expected_total_slippage': 0.0
            }
            
            # translated
            avg_volume = market_conditions.get('avg_volume', 1.0)
            volume_ratio = order_size / avg_volume if avg_volume > 0 else 0.1
            
            if volume_ratio > 0.2:  # large
                split_count = min(self.order_splitting['max_split_count'], 8)
            elif volume_ratio > 0.1:  # large
                split_count = min(self.order_splitting['max_split_count'], 5)
            elif volume_ratio > 0.05:  # mid
                split_count = min(self.order_splitting['max_split_count'], 3)
            else:
                split_count = 1  # small not
            
            if split_count == 1:
                split_plan['orders'] = [{'size': order_size, 'delay': 0, 'priority': 'high'}]
                split_plan['total_time'] = 30
                return split_plan
            
            # translated
            if algorithm == 'twap':
                # time
                sub_order_size = order_size / split_count
                base_delay = self.order_splitting['split_interval_seconds']
                
                for i in range(split_count):
                    split_plan['orders'].append({
                        'size': sub_order_size,
                        'delay': i * base_delay,
                        'priority': 'medium' if i > 0 else 'high'
                    })
                
                split_plan['total_time'] = (split_count - 1) * base_delay + 30
                
            elif algorithm == 'vwap':
                # volume
                volume_distribution = self.get_volume_distribution_forecast()
                cumulative_size = 0
                
                for i, volume_weight in enumerate(volume_distribution[:split_count]):
                    sub_order_size = order_size * volume_weight
                    cumulative_size += sub_order_size
                    
                    split_plan['orders'].append({
                        'size': sub_order_size,
                        'delay': i * 60,  # minutes 1 count
                        'priority': 'high' if volume_weight > 0.2 else 'medium'
                    })
                
                # translated
                if cumulative_size < order_size:
                    remaining = order_size - cumulative_size
                    split_plan['orders'][-1]['size'] += remaining
                
                split_plan['total_time'] = len(split_plan['orders']) * 60
                
            else:  # implementation_shortfall
                # translated，translated
                remaining_size = order_size
                time_offset = 0
                urgency_factor = min(1.5, market_conditions.get('volatility', 0.02) * 20)
                
                for i in range(split_count):
                    if i == split_count - 1:
                        # most after 1 count has
                        sub_order_size = remaining_size
                    else:
                        # large small
                        base_portion = 1.0 / (split_count - i)
                        urgency_adjustment = base_portion * urgency_factor
                        sub_order_size = min(remaining_size, order_size * urgency_adjustment)
                    
                    split_plan['orders'].append({
                        'size': sub_order_size,
                        'delay': time_offset,
                        'priority': 'high' if i < 2 else 'medium'
                    })
                    
                    remaining_size -= sub_order_size
                    time_offset += max(15, int(45 / urgency_factor))  # translated
                    
                    if remaining_size <= 0:
                        break
                
                split_plan['total_time'] = time_offset + 30
            
            # calculate
            total_slippage = 0.0
            for order in split_plan['orders']:
                sub_slippage = self.predict_slippage(pair, order['size'], 'buy', market_conditions)
                total_slippage += sub_slippage * (order['size'] / order_size)
            
            split_plan['expected_total_slippage'] = total_slippage
            
            return split_plan
            
        except Exception:
            return {
                'orders': [{'size': order_size, 'delay': 0, 'priority': 'high'}],
                'total_time': 30,
                'expected_total_slippage': 0.002
            }
    
    def get_volume_distribution_forecast(self) -> List[float]:
        """get volume"""
        try:
            # simplified day volume
            # translated
            typical_distribution = [
                0.05, 0.08, 0.12, 0.15, 0.18, 0.15, 0.12, 0.08, 0.05, 0.02
            ]
            return typical_distribution
        except Exception:
            return [0.1] * 10  # translated
    
    def optimize_execution_timing(self, pair: str, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """optimize"""
        try:
            current_time = datetime.now(timezone.utc)
            hour = current_time.hour
            
            timing_score = 0.5  # translated
            timing_factors = []
            
            # score
            if 13 <= hour <= 16:  # translated
                timing_score += 0.3
                timing_factors.append('high_liquidity_session')
            elif 8 <= hour <= 11 or 17 <= hour <= 20:  # 1
                timing_score += 0.1
                timing_factors.append('medium_liquidity_session')
            else:  # low
                timing_score -= 0.2
                timing_factors.append('low_liquidity_session')
            
            # volatility score
            volatility = market_conditions.get('volatility', 0.02)
            if 0.02 <= volatility <= 0.04:  # mid volatility
                timing_score += 0.1
                timing_factors.append('optimal_volatility')
            elif volatility > 0.05:  # high volatility
                timing_score -= 0.15
                timing_factors.append('high_volatility_risk')
            
            # volume score
            volume_ratio = market_conditions.get('volume_ratio', 1.0)
            if volume_ratio > 1.2:
                timing_score += 0.1
                timing_factors.append('high_volume')
            elif volume_ratio < 0.8:
                timing_score -= 0.1
                timing_factors.append('low_volume')
            
            # translated
            if timing_score > 0.7:
                recommendation = 'execute_immediately'
            elif timing_score > 0.4:
                recommendation = 'execute_normal'
            else:
                recommendation = 'delay_execution'
            
            return {
                'timing_score': timing_score,
                'recommendation': recommendation,
                'factors': timing_factors,
                'optimal_delay_minutes': max(0, int((0.6 - timing_score) * 30))
            }
            
        except Exception:
            return {
                'timing_score': 0.5,
                'recommendation': 'execute_normal',
                'factors': ['timing_analysis_error'],
                'optimal_delay_minutes': 0
            }
    
    def generate_execution_instructions(self, execution_plan: Dict[str, Any], 
                                      pair: str, order_side: str, 
                                      current_price: float) -> List[Dict[str, Any]]:
        """translated"""
        try:
            instructions = []
            
            for i, order in enumerate(execution_plan['split_orders']):
                instruction = {
                    'instruction_id': f"{pair}_{order_side}_{i}_{int(datetime.now(timezone.utc).timestamp())}",
                    'pair': pair,
                    'side': order_side,
                    'size': order['size'],
                    'order_type': self.determine_order_type(order, execution_plan),
                    'price_limit': self.calculate_price_limit(current_price, order_side, order['size'], execution_plan),
                    'delay_seconds': order['delay'],
                    'priority': order['priority'],
                    'timeout_seconds': 300,  # 5minutes
                    'max_slippage': self.slippage_control['max_allowed_slippage'],
                    'execution_strategy': execution_plan['execution_strategy'],
                    'created_at': datetime.now(timezone.utc)
                }
                
                instructions.append(instruction)
            
            return instructions
            
        except Exception:
            # translated
            return [{
                'instruction_id': f"{pair}_{order_side}_simple_{int(datetime.now(timezone.utc).timestamp())}",
                'pair': pair,
                'side': order_side,
                'size': execution_plan['original_size'],
                'order_type': 'market',
                'delay_seconds': 0,
                'priority': 'high',
                'timeout_seconds': 180,
                'max_slippage': 0.003,
                'created_at': datetime.now(timezone.utc)
            }]
    
    def determine_order_type(self, order: Dict[str, Any], execution_plan: Dict[str, Any]) -> str:
        """translated"""
        try:
            if order['priority'] == 'high' or execution_plan.get('risk_level') == 'high':
                return 'market'
            elif execution_plan['expected_slippage'] < self.slippage_control['adaptive_threshold']:
                return 'limit'
            else:
                return 'market_with_protection'  # translated
        except Exception:
            return 'market'
    
    def calculate_price_limit(self, current_price: float, side: str, 
                            order_size: float, execution_plan: Dict[str, Any]) -> float:
        """calculate price"""
        try:
            expected_slippage = execution_plan['expected_slippage']
            
            # translated
            slippage_buffer = expected_slippage * 1.2  # 20%translated
            
            if side.lower() == 'buy':
                return current_price * (1 + slippage_buffer)
            else:
                return current_price * (1 - slippage_buffer)
                
        except Exception:
            # price
            if side.lower() == 'buy':
                return current_price * 1.005
            else:
                return current_price * 0.995
    
    def track_execution_performance(self, execution_id: str, execution_result: Dict[str, Any]):
        """translated"""
        try:
            # calculate
            expected_price = execution_result.get('expected_price', 0)
            actual_price = execution_result.get('actual_price', 0)
            
            if expected_price > 0 and actual_price > 0:
                realized_slippage = abs(actual_price - expected_price) / expected_price
                self.execution_metrics['realized_slippage'].append(realized_slippage)
            
            # calculate
            pre_trade_price = execution_result.get('pre_trade_price', 0)
            post_trade_price = execution_result.get('post_trade_price', 0)
            
            if pre_trade_price > 0 and post_trade_price > 0:
                market_impact = abs(post_trade_price - pre_trade_price) / pre_trade_price
                self.execution_metrics['market_impact'].append(market_impact)
            
            # record indicator
            execution_time = execution_result.get('execution_time_seconds', 0)
            if execution_time > 0:
                self.execution_metrics['execution_time'].append(execution_time)
            
            fill_ratio = execution_result.get('fill_ratio', 1.0)
            self.execution_metrics['fill_ratio'].append(fill_ratio)
            
            # indicator long
            for metric in self.execution_metrics.values():
                if len(metric) > 500:
                    metric[:] = metric[-250:]  # most250count record
                    
        except Exception:
            pass
    
    def get_execution_quality_report(self) -> Dict[str, Any]:
        """get"""
        try:
            if not any(self.execution_metrics.values()):
                return {'error': 'none'}
            
            report = {}
            
            # translated
            if self.execution_metrics['realized_slippage']:
                slippage_data = self.execution_metrics['realized_slippage']
                report['slippage'] = {
                    'avg': np.mean(slippage_data),
                    'median': np.median(slippage_data),
                    'std': np.std(slippage_data),
                    'p95': np.percentile(slippage_data, 95),
                    'samples': len(slippage_data)
                }
            
            # translated
            if self.execution_metrics['market_impact']:
                impact_data = self.execution_metrics['market_impact']
                report['market_impact'] = {
                    'avg': np.mean(impact_data),
                    'median': np.median(impact_data),
                    'std': np.std(impact_data),
                    'p95': np.percentile(impact_data, 95),
                    'samples': len(impact_data)
                }
            
            # time
            if self.execution_metrics['execution_time']:
                time_data = self.execution_metrics['execution_time']
                report['execution_time'] = {
                    'avg_seconds': np.mean(time_data),
                    'median_seconds': np.median(time_data),
                    'p95_seconds': np.percentile(time_data, 95),
                    'samples': len(time_data)
                }
            
            # translated
            if self.execution_metrics['fill_ratio']:
                fill_data = self.execution_metrics['fill_ratio']
                report['fill_ratio'] = {
                    'avg': np.mean(fill_data),
                    'median': np.median(fill_data),
                    'samples_below_95pct': sum(1 for x in fill_data if x < 0.95),
                    'samples': len(fill_data)
                }
            
            return report
            
        except Exception:
            return {'error': 'none'}
    
    # ===== and =====
    
    def initialize_sentiment_system(self):
        """initialize"""
        # market sentiment indicator configuration
        self.sentiment_indicators = {
            'fear_greed_index': {'enabled': True, 'weight': 0.25},
            'vix_equivalent': {'enabled': True, 'weight': 0.20},
            'news_sentiment': {'enabled': True, 'weight': 0.15},
            'social_sentiment': {'enabled': True, 'weight': 0.10},
            'positioning_data': {'enabled': True, 'weight': 0.15},
            'intermarket_sentiment': {'enabled': True, 'weight': 0.15}
        }
        
        # value
        self.sentiment_thresholds = {
            'extreme_fear': 20,      # translated
            'fear': 35,              # translated
            'neutral': 50,           # mid
            'greed': 65,             # translated
            'extreme_greed': 80      # translated
        }
        
        # configuration
        self.external_data_sources = {
            'economic_calendar': {'enabled': True, 'impact_threshold': 'medium'},
            'central_bank_policy': {'enabled': True, 'lookback_days': 30},
            'geopolitical_events': {'enabled': True, 'risk_threshold': 'medium'},
            'seasonal_patterns': {'enabled': True, 'historical_years': 5},
            'intermarket_correlations': {'enabled': True, 'correlation_threshold': 0.6}
        }
        
        # translated
        self.sentiment_history = {
            'composite_sentiment': [],
            'market_regime': [],
            'sentiment_extremes': [],
            'contrarian_signals': []
        }
        
        # translated
        self.external_events = []
        self.event_impact_history = []
        
        # translated
        self.seasonal_patterns = {}
        self.intermarket_data = {}
        
    # translated analyze_market_sentiment - simplified
    def analyze_market_sentiment(self) -> Dict[str, Any]:
        """translated"""
        try:
            sentiment_components = {}
            
            # 1. translated
            if self.sentiment_indicators['fear_greed_index']['enabled']:
                fear_greed = self.calculate_fear_greed_index()
                sentiment_components['fear_greed'] = fear_greed
            
            # 2. volatility
            if self.sentiment_indicators['vix_equivalent']['enabled']:
                vix_sentiment = self.analyze_volatility_sentiment()
                sentiment_components['volatility_sentiment'] = vix_sentiment
            
            # 3. new
            if self.sentiment_indicators['news_sentiment']['enabled']:
                news_sentiment = self.analyze_news_sentiment()
                sentiment_components['news_sentiment'] = news_sentiment
            
            # 4. translated
            if self.sentiment_indicators['social_sentiment']['enabled']:
                social_sentiment = self.analyze_social_sentiment()
                sentiment_components['social_sentiment'] = social_sentiment
            
            # 5. translated
            if self.sentiment_indicators['positioning_data']['enabled']:
                positioning_sentiment = self.analyze_positioning_data()
                sentiment_components['positioning_sentiment'] = positioning_sentiment
            
            # 6. translated
            if self.sentiment_indicators['intermarket_sentiment']['enabled']:
                intermarket_sentiment = self.analyze_intermarket_sentiment()
                sentiment_components['intermarket_sentiment'] = intermarket_sentiment
            
            # calculate
            composite_sentiment = self.calculate_composite_sentiment(sentiment_components)
            
            # translated
            sentiment_state = self.determine_sentiment_state(composite_sentiment)
            
            # signal
            sentiment_adjustment = self.generate_sentiment_adjustment(sentiment_state, sentiment_components)
            
            sentiment_analysis = {
                'composite_sentiment': composite_sentiment,
                'sentiment_state': sentiment_state,
                'components': sentiment_components,
                'trading_adjustment': sentiment_adjustment,
                'contrarian_opportunity': self.detect_contrarian_opportunity(composite_sentiment),
                'timestamp': datetime.now(timezone.utc)
            }
            
            # update
            self.update_sentiment_history(sentiment_analysis)
            
            return sentiment_analysis
            
        except Exception as e:
            return {
                'composite_sentiment': 50,  # mid
                'sentiment_state': 'neutral',
                'error': f'translated: {str(e)}',
                'timestamp': datetime.now(timezone.utc)
            }
    
    def calculate_fear_greed_index(self) -> Dict[str, Any]:
        """calculate"""
        try:
            components = {}
            
            # price momentum (25%)
            price_momentum = self.calculate_price_momentum_sentiment()
            components['price_momentum'] = price_momentum
            
            # volatility (25%) - andVIXtranslated
            volatility_fear = self.calculate_volatility_fear()
            components['volatility_fear'] = volatility_fear
            
            # translated (15%) - up down
            market_breadth = self.calculate_market_breadth_sentiment()
            components['market_breadth'] = market_breadth
            
            # translated (15%) - translated
            safe_haven_demand = self.calculate_safe_haven_sentiment()
            components['safe_haven_demand'] = safe_haven_demand
            
            # translated (10%) - risk indicator  
            junk_bond_demand = self.calculate_junk_bond_sentiment()
            components['junk_bond_demand'] = junk_bond_demand
            
            # translated (10%)
            put_call_ratio = self.calculate_put_call_sentiment()
            components['put_call_ratio'] = put_call_ratio
            
            # calculate
            weights = [0.25, 0.25, 0.15, 0.15, 0.10, 0.10]
            values = [price_momentum, volatility_fear, market_breadth, 
                     safe_haven_demand, junk_bond_demand, put_call_ratio]
            
            fear_greed_index = sum(w * v for w, v in zip(weights, values) if v is not None)
            
            return {
                'index_value': fear_greed_index,
                'components': components,
                'interpretation': self.interpret_fear_greed_index(fear_greed_index)
            }
            
        except Exception:
            return {
                'index_value': 50,
                'components': {},
                'interpretation': 'neutral'
            }
    
    def calculate_price_momentum_sentiment(self) -> float:
        """calculate price momentum"""
        try:
            # price calculate
            # simplified：price
            
            # translated125day up
            stocks_above_ma125 = 0.6  # 60%at125day up
            
            # as0-100value
            momentum_sentiment = stocks_above_ma125 * 100
            
            return min(100, max(0, momentum_sentiment))
            
        except Exception:
            return 50
    
    def calculate_volatility_fear(self) -> float:
        """calculate volatility"""
        try:
            # before volatility value
            current_volatility = getattr(self, 'current_market_volatility', {})
            avg_vol = sum(current_volatility.values()) / len(current_volatility) if current_volatility else 0.02
            
            # volatility（value）
            historical_avg_vol = 0.025
            
            # volatility
            vol_ratio = avg_vol / historical_avg_vol if historical_avg_vol > 0 else 1.0
            
            # as（volatility high，large，low）
            volatility_fear = max(0, min(100, 100 - (vol_ratio - 1) * 50))
            
            return volatility_fear
            
        except Exception:
            return 50
    
    def calculate_market_breadth_sentiment(self) -> float:
        """calculate"""
        try:
            # translated
            # up down
            
            # translated：up
            advancing_stocks_ratio = 0.55  # 55%up
            
            # as
            breadth_sentiment = advancing_stocks_ratio * 100
            
            return min(100, max(0, breadth_sentiment))
            
        except Exception:
            return 50
    
    def calculate_safe_haven_sentiment(self) -> float:
        """calculate"""
        try:
            # translated
            # translated、translated
            
            # translated（value high）
            safe_haven_performance = -0.02  # -2%translated
            
            # as（high，low）
            safe_haven_sentiment = max(0, min(100, 50 - safe_haven_performance * 1000))
            
            return safe_haven_sentiment
            
        except Exception:
            return 50
    
    def calculate_junk_bond_sentiment(self) -> float:
        """calculate"""
        try:
            # and
            # high
            
            # translated（bp）
            credit_spread_bp = 350  # 350count
            historical_avg_spread = 400  # translated400bp
            
            # as
            spread_ratio = credit_spread_bp / historical_avg_spread
            junk_bond_sentiment = max(0, min(100, 100 - (spread_ratio - 1) * 100))
            
            return junk_bond_sentiment
            
        except Exception:
            return 50
    
    def calculate_put_call_sentiment(self) -> float:
        """calculate"""
        try:
            # translated/translated
            # translated
            
            # translated/translated
            put_call_ratio = 0.8  # 0.8translated
            historical_avg_ratio = 1.0
            
            # as（low，high）
            put_call_sentiment = max(0, min(100, 100 - (put_call_ratio / historical_avg_ratio - 1) * 100))
            
            return put_call_sentiment
            
        except Exception:
            return 50
    
    def interpret_fear_greed_index(self, index_value: float) -> str:
        """translated"""
        if index_value <= self.sentiment_thresholds['extreme_fear']:
            return 'extreme_fear'
        elif index_value <= self.sentiment_thresholds['fear']:
            return 'fear'
        elif index_value <= self.sentiment_thresholds['neutral']:
            return 'neutral_fear'
        elif index_value <= self.sentiment_thresholds['greed']:
            return 'neutral_greed'
        elif index_value <= self.sentiment_thresholds['extreme_greed']:
            return 'greed'
        else:
            return 'extreme_greed'
    
    # translated analyze_volatility_sentiment - simplified
    def analyze_volatility_sentiment(self) -> Dict[str, Any]:
        """volatility"""
        try:
            current_volatility = getattr(self, 'current_market_volatility', {})
            
            if not current_volatility:
                return {
                    'volatility_level': 'normal',
                    'sentiment_signal': 'neutral',
                    'volatility_percentile': 50
                }
            
            avg_vol = sum(current_volatility.values()) / len(current_volatility)
            
            # volatility（simplified calculate）
            vol_percentile = min(95, max(5, avg_vol * 2000))  # simplified
            
            # signal
            if vol_percentile > 80:
                sentiment_signal = 'high_fear'
                volatility_level = 'high'
            elif vol_percentile > 60:
                sentiment_signal = 'moderate_fear'
                volatility_level = 'elevated'
            elif vol_percentile < 20:
                sentiment_signal = 'complacency'
                volatility_level = 'low'
            else:
                sentiment_signal = 'neutral'
                volatility_level = 'normal'
            
            return {
                'volatility_level': volatility_level,
                'sentiment_signal': sentiment_signal,
                'volatility_percentile': vol_percentile,
                'average_volatility': avg_vol
            }
            
        except Exception:
            return {
                'volatility_level': 'normal',
                'sentiment_signal': 'neutral',
                'volatility_percentile': 50
            }
    
    # translated analyze_news_sentiment - simplified
    def analyze_news_sentiment(self) -> Dict[str, Any]:
        """new"""
        try:
            # new
            # newAPItranslatedNLPtranslated
            
            # new (-1to1)
            news_sentiment_score = 0.1  # translated
            
            # new
            news_volume = 1.2  # 120%new
            
            # translated
            sentiment_keywords = {
                'positive': ['growth', 'opportunity', 'bullish'],
                'negative': ['uncertainty', 'risk', 'volatile'],
                'neutral': ['stable', 'unchanged', 'maintain']
            }
            
            # as signal
            if news_sentiment_score > 0.3:
                trading_signal = 'bullish'
            elif news_sentiment_score < -0.3:
                trading_signal = 'bearish'
            else:
                trading_signal = 'neutral'
            
            return {
                'sentiment_score': news_sentiment_score,
                'trading_signal': trading_signal,
                'news_volume': news_volume,
                'sentiment_keywords': sentiment_keywords,
                'confidence_level': min(1.0, abs(news_sentiment_score) + 0.5)
            }
            
        except Exception:
            return {
                'sentiment_score': 0.0,
                'trading_signal': 'neutral',
                'news_volume': 1.0,
                'confidence_level': 0.5
            }
    
    # translated analyze_social_sentiment - simplified
    def analyze_social_sentiment(self) -> Dict[str, Any]:
        """translated"""
        try:
            # translated
            # translatedTwitter/ReddittranslatedAPI
            
            # and
            mention_volume = 1.3  # 130%and
            
            # translated
            sentiment_distribution = {
                'bullish': 0.4,   # 40%translated
                'bearish': 0.3,   # 30%translated
                'neutral': 0.3    # 30%mid
            }
            
            # translated（weight high）
            influencer_sentiment = 0.2  # translated
            
            # trend strength
            trend_strength = abs(sentiment_distribution['bullish'] - sentiment_distribution['bearish'])
            
            # translated
            social_score = (
                sentiment_distribution['bullish'] * 1 + 
                sentiment_distribution['bearish'] * (-1) + 
                sentiment_distribution['neutral'] * 0
            )
            
            # weight
            adjusted_score = social_score * 0.7 + influencer_sentiment * 0.3
            
            return {
                'sentiment_score': adjusted_score,
                'mention_volume': mention_volume,
                'sentiment_distribution': sentiment_distribution,
                'influencer_sentiment': influencer_sentiment,
                'trend_strength': trend_strength,
                'social_signal': 'bullish' if adjusted_score > 0.1 else 'bearish' if adjusted_score < -0.1 else 'neutral'
            }
            
        except Exception:
            return {
                'sentiment_score': 0.0,
                'mention_volume': 1.0,
                'social_signal': 'neutral',
                'trend_strength': 0.0
            }
    
    # translated analyze_positioning_data - simplified
    def analyze_positioning_data(self) -> Dict[str, Any]:
        """translated"""
        try:
            # translated
            # translatedCOTtranslated
            
            # large
            large_trader_net_long = 0.15  # 15%bullish
            
            # translated
            retail_sentiment = -0.1  # translated
            
            # translated
            institutional_flow = 0.05  # 5%translated
            
            # translated
            positioning_extreme = max(
                abs(large_trader_net_long),
                abs(retail_sentiment),
                abs(institutional_flow)
            )
            
            # indicator（translated）
            contrarian_signal = 'bullish' if retail_sentiment < -0.15 else 'bearish' if retail_sentiment > 0.15 else 'neutral'
            
            return {
                'large_trader_positioning': large_trader_net_long,
                'retail_sentiment': retail_sentiment,
                'institutional_flow': institutional_flow,
                'positioning_extreme': positioning_extreme,
                'contrarian_signal': contrarian_signal,
                'positioning_risk': 'high' if positioning_extreme > 0.2 else 'medium' if positioning_extreme > 0.1 else 'low'
            }
            
        except Exception:
            return {
                'large_trader_positioning': 0.0,
                'retail_sentiment': 0.0,
                'institutional_flow': 0.0,
                'contrarian_signal': 'neutral',
                'positioning_risk': 'low'
            }
    
    # translated analyze_intermarket_sentiment - simplified
    def analyze_intermarket_sentiment(self) -> Dict[str, Any]:
        """translated"""
        try:
            # translated
            # translated、translated、translated、translated
            
            # translated
            stock_bond_correlation = -0.3  # as
            
            # strength
            dollar_strength = 0.02  # translated2%
            
            # translated
            commodity_performance = -0.01  # down
            
            # translated
            safe_haven_flows = 0.5  # mid
            
            # indicator
            intermarket_stress = abs(stock_bond_correlation + 0.5) + abs(dollar_strength) * 10
            
            # risk indicator
            risk_appetite = 0.6 - safe_haven_flows
            
            return {
                'stock_bond_correlation': stock_bond_correlation,
                'dollar_strength': dollar_strength,
                'commodity_performance': commodity_performance,
                'safe_haven_flows': safe_haven_flows,
                'intermarket_stress': intermarket_stress,
                'risk_appetite': risk_appetite,
                'regime': 'risk_on' if risk_appetite > 0.3 else 'risk_off' if risk_appetite < -0.3 else 'mixed'
            }
            
        except Exception:
            return {
                'stock_bond_correlation': -0.5,
                'dollar_strength': 0.0,
                'commodity_performance': 0.0,
                'safe_haven_flows': 0.5,
                'risk_appetite': 0.0,
                'regime': 'mixed'
            }
    
    def calculate_composite_sentiment(self, components: Dict[str, Any]) -> float:
        """calculate"""
        try:
            sentiment_values = []
            weights = []
            
            # translated
            if 'fear_greed' in components:
                sentiment_values.append(components['fear_greed']['index_value'])
                weights.append(self.sentiment_indicators['fear_greed_index']['weight'])
            
            # volatility
            if 'volatility_sentiment' in components:
                vol_sentiment = 100 - components['volatility_sentiment']['volatility_percentile']
                sentiment_values.append(vol_sentiment)
                weights.append(self.sentiment_indicators['vix_equivalent']['weight'])
            
            # new
            if 'news_sentiment' in components:
                news_score = (components['news_sentiment']['sentiment_score'] + 1) * 50
                sentiment_values.append(news_score)
                weights.append(self.sentiment_indicators['news_sentiment']['weight'])
            
            # translated
            if 'social_sentiment' in components:
                social_score = (components['social_sentiment']['sentiment_score'] + 1) * 50
                sentiment_values.append(social_score)
                weights.append(self.sentiment_indicators['social_sentiment']['weight'])
            
            # translated
            if 'positioning_sentiment' in components:
                pos_score = 50  # mid value，translated
                sentiment_values.append(pos_score)
                weights.append(self.sentiment_indicators['positioning_data']['weight'])
            
            # translated
            if 'intermarket_sentiment' in components:
                inter_score = (components['intermarket_sentiment']['risk_appetite'] + 1) * 50
                sentiment_values.append(inter_score)
                weights.append(self.sentiment_indicators['intermarket_sentiment']['weight'])
            
            # translated
            if sentiment_values and weights:
                total_weight = sum(weights)
                composite_sentiment = sum(s * w for s, w in zip(sentiment_values, weights)) / total_weight
            else:
                composite_sentiment = 50  # default mid
            
            return max(0, min(100, composite_sentiment))
            
        except Exception:
            return 50  # mid
    
    def determine_sentiment_state(self, composite_sentiment: float) -> str:
        """translated"""
        if composite_sentiment <= self.sentiment_thresholds['extreme_fear']:
            return 'extreme_fear'
        elif composite_sentiment <= self.sentiment_thresholds['fear']:
            return 'fear'
        elif composite_sentiment <= self.sentiment_thresholds['neutral']:
            return 'neutral_bearish'
        elif composite_sentiment <= self.sentiment_thresholds['greed']:
            return 'neutral_bullish'
        elif composite_sentiment <= self.sentiment_thresholds['extreme_greed']:
            return 'greed'
        else:
            return 'extreme_greed'
    
    def generate_sentiment_adjustment(self, sentiment_state: str, components: Dict[str, Any]) -> Dict[str, Any]:
        """translated"""
        try:
            adjustment = {
                'position_size_multiplier': 1.0,
                'leverage_multiplier': 1.0,
                'risk_tolerance_adjustment': 0.0,
                'entry_threshold_adjustment': 0.0,
                'sentiment_signal': 'neutral'
            }
            
            # translated
            if sentiment_state == 'extreme_fear':
                adjustment.update({
                    'position_size_multiplier': 0.8,    # small position size
                    'leverage_multiplier': 0.7,         # low leverage
                    'risk_tolerance_adjustment': -0.1,   # translated
                    'entry_threshold_adjustment': -0.05, # low（translated）
                    'sentiment_signal': 'contrarian_bullish'
                })
            elif sentiment_state == 'fear':
                adjustment.update({
                    'position_size_multiplier': 0.9,
                    'leverage_multiplier': 0.85,
                    'risk_tolerance_adjustment': -0.05,
                    'entry_threshold_adjustment': -0.02,
                    'sentiment_signal': 'cautious_bullish'
                })
            elif sentiment_state == 'extreme_greed':
                adjustment.update({
                    'position_size_multiplier': 0.7,    # large small position size
                    'leverage_multiplier': 0.6,         # large low leverage
                    'risk_tolerance_adjustment': -0.15,  # translated
                    'entry_threshold_adjustment': 0.1,   # high
                    'sentiment_signal': 'contrarian_bearish'
                })
            elif sentiment_state == 'greed':
                adjustment.update({
                    'position_size_multiplier': 0.85,
                    'leverage_multiplier': 0.8,
                    'risk_tolerance_adjustment': -0.08,
                    'entry_threshold_adjustment': 0.03,
                    'sentiment_signal': 'cautious_bearish'
                })
            
            # translated
            if 'volatility_sentiment' in components:
                vol_signal = components['volatility_sentiment']['sentiment_signal']
                if vol_signal == 'high_fear':
                    adjustment['position_size_multiplier'] *= 0.9
                elif vol_signal == 'complacency':
                    adjustment['risk_tolerance_adjustment'] -= 0.05
            
            return adjustment
            
        except Exception:
            return {
                'position_size_multiplier': 1.0,
                'leverage_multiplier': 1.0,
                'risk_tolerance_adjustment': 0.0,
                'entry_threshold_adjustment': 0.0,
                'sentiment_signal': 'neutral'
            }
    
    def detect_contrarian_opportunity(self, composite_sentiment: float) -> Dict[str, Any]:
        """translated"""
        try:
            # translated
            contrarian_opportunity = {
                'opportunity_detected': False,
                'opportunity_type': None,
                'strength': 0.0,
                'recommended_action': 'hold'
            }
            
            # translated
            if composite_sentiment <= 25:  # translated
                contrarian_opportunity.update({
                    'opportunity_detected': True,
                    'opportunity_type': 'extreme_fear_buying',
                    'strength': (25 - composite_sentiment) / 25,
                    'recommended_action': 'aggressive_buy'
                })
            elif composite_sentiment >= 75:  # translated
                contrarian_opportunity.update({
                    'opportunity_detected': True,
                    'opportunity_type': 'extreme_greed_selling',
                    'strength': (composite_sentiment - 75) / 25,
                    'recommended_action': 'reduce_exposure'
                })
            
            # fast
            if len(self.sentiment_history['composite_sentiment']) >= 5:
                recent_sentiments = self.sentiment_history['composite_sentiment'][-5:]
                sentiment_velocity = recent_sentiments[-1] - recent_sentiments[0]
                
                if abs(sentiment_velocity) > 20:  # fast
                    contrarian_opportunity.update({
                        'opportunity_detected': True,
                        'opportunity_type': 'sentiment_reversal',
                        'strength': min(1.0, abs(sentiment_velocity) / 30),
                        'recommended_action': 'fade_the_move'
                    })
            
            return contrarian_opportunity
            
        except Exception:
            return {
                'opportunity_detected': False,
                'opportunity_type': None,
                'strength': 0.0,
                'recommended_action': 'hold'
            }
    
    def update_sentiment_history(self, sentiment_analysis: Dict[str, Any]):
        """update record"""
        try:
            # update
            self.sentiment_history['composite_sentiment'].append(sentiment_analysis['composite_sentiment'])
            
            # update
            self.sentiment_history['sentiment_state'].append(sentiment_analysis['sentiment_state'])
            
            # record value
            if sentiment_analysis['composite_sentiment'] <= 25 or sentiment_analysis['composite_sentiment'] >= 75:
                extreme_record = {
                    'timestamp': sentiment_analysis['timestamp'],
                    'sentiment_value': sentiment_analysis['composite_sentiment'],
                    'sentiment_state': sentiment_analysis['sentiment_state']
                }
                self.sentiment_history['sentiment_extremes'].append(extreme_record)
            
            # record signal
            if sentiment_analysis.get('contrarian_opportunity', {}).get('opportunity_detected'):
                contrarian_record = {
                    'timestamp': sentiment_analysis['timestamp'],
                    'opportunity_type': sentiment_analysis['contrarian_opportunity']['opportunity_type'],
                    'strength': sentiment_analysis['contrarian_opportunity']['strength']
                }
                self.sentiment_history['contrarian_signals'].append(contrarian_record)
            
            # record long
            for key, history in self.sentiment_history.items():
                if len(history) > 500:
                    self.sentiment_history[key] = history[-250:]
                    
        except Exception:
            pass
    
    def get_sentiment_analysis_report(self) -> Dict[str, Any]:
        """get"""
        try:
            if not self.sentiment_history['composite_sentiment']:
                return {'error': 'none'}
            
            recent_sentiment = self.sentiment_history['composite_sentiment'][-1]
            recent_state = self.sentiment_history['sentiment_state'][-1]
            
            # translated
            sentiment_stats = {
                'current_sentiment': recent_sentiment,
                'current_state': recent_state,
                'avg_sentiment_30d': np.mean(self.sentiment_history['composite_sentiment'][-30:]) if len(self.sentiment_history['composite_sentiment']) >= 30 else recent_sentiment,
                'sentiment_volatility': np.std(self.sentiment_history['composite_sentiment'][-30:]) if len(self.sentiment_history['composite_sentiment']) >= 30 else 0,
                'extreme_events_30d': len([x for x in self.sentiment_history['sentiment_extremes'] if (datetime.now(timezone.utc) - x['timestamp']).days <= 30]),
                'contrarian_signals_30d': len([x for x in self.sentiment_history['contrarian_signals'] if (datetime.now(timezone.utc) - x['timestamp']).days <= 30])
            }
            
            return {
                'sentiment_stats': sentiment_stats,
                'sentiment_trend': 'improving' if len(self.sentiment_history['composite_sentiment']) >= 2 and self.sentiment_history['composite_sentiment'][-1] > self.sentiment_history['composite_sentiment'][-2] else 'deteriorating',
                'market_regime': 'fear_dominated' if recent_sentiment < 40 else 'greed_dominated' if recent_sentiment > 60 else 'neutral',
                'last_update': datetime.now(timezone.utc)
            }
            
        except Exception:
            return {'error': 'none'}
    
    # === 🛡️ ATRtranslated ===
    
    def _get_trade_entry_atr(self, trade: Trade, dataframe: DataFrame) -> float:
        """
        getATRvalue - as calculate
        avoid or
        """
        try:
            # use time toKtranslated
            from freqtrade.misc import timeframe_to_prev_date
            
            entry_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            entry_candles = dataframe[dataframe.index <= entry_date]
            
            if not entry_candles.empty and 'atr_p' in entry_candles.columns:
                entry_atr = entry_candles['atr_p'].iloc[-1]
                # range check
                if 0.005 <= entry_atr <= 0.20:
                    return entry_atr
                    
        except Exception as e:
            logger.warning(f"getATRtranslated: {e}")
            
        # translated：use most20translatedATRmid
        if 'atr_p' in dataframe.columns and len(dataframe) >= 20:
            return dataframe['atr_p'].tail(20).median()
        
        # most after：value
        if 'BTC' in trade.pair or 'ETH' in trade.pair:
            return 0.02  # translated
        else:
            return 0.035  # large
    
    def _calculate_atr_multiplier(self, entry_atr_p: float, current_candle: dict, enter_tag: str) -> float:
        """
        calculateATRtranslated - translated，translated
        signal
        """
        # translated：translated2.5-3.5as most range
        base_multiplier = 2.8
        
        # === 1. signal ===
        signal_adjustments = {
            'RSI_Oversold_Bounce': 2.5,    # RSIsignal，1
            'RSI_Overbought_Fall': 2.5,    
            'MACD_Bearish': 3.2,           # MACDsignal breakout，translated
            'MACD_Bullish': 3.2,
            'EMA_Golden_Cross': 2.6,       # trend signal，mid
            'EMA_Death_Cross': 2.6,
        }
        
        multiplier = signal_adjustments.get(enter_tag, base_multiplier)
        
        # === 2. volatility ===
        current_atr_p = current_candle.get('atr_p', entry_atr_p)
        volatility_ratio = current_atr_p / entry_atr_p
        
        if volatility_ratio > 1.5:      # before high50%
            multiplier *= 1.2           # translated20%
        elif volatility_ratio < 0.7:    # before low30%
            multiplier *= 0.9           # translated10%
        
        # === 3. trend strength ===
        adx = current_candle.get('adx', 25)
        if adx > 35:                    # strong trend
            multiplier *= 1.15          # trend
        elif adx < 20:                  # translated
            multiplier *= 0.85          # avoid
        
        # translated
        return max(1.5, min(4.0, multiplier))
    
    def _calculate_time_decay(self, hours_held: float, current_profit: float) -> float:
        """
        time - long
        time long，translated
        """
        # profit，time
        if current_profit > 0.02:       # profit2%to up
            decay_start_hours = 72      # 3after
        elif current_profit > -0.02:    # small loss
            decay_start_hours = 48      # 2after  
        else:                           # large loss
            decay_start_hours = 24      # 1after
        
        if hours_held <= decay_start_hours:
            return 1.0                  # none
            
        # translated：translated24hours10%
        excess_hours = hours_held - decay_start_hours
        decay_periods = excess_hours / 24
        
        # most to original50%
        min_factor = 0.5
        decay_factor = max(min_factor, 1.0 - (decay_periods * 0.1))
        
        return decay_factor
    
    def _calculate_profit_protection(self, current_profit: float) -> Optional[float]:
        """
        profit - translated，profit
        """
        if current_profit > 0.15:      # profit15%+，translated75%translated
            return -0.0375              # allow3.75%drawdown
        elif current_profit > 0.10:    # profit10%+，translated60%translated  
            return -0.04                # allow4%drawdown
        elif current_profit > 0.08:    # profit8%+，translated50%translated
            return -0.04                # allow4%drawdown
        elif current_profit > 0.05:    # profit5%+，translated+
            return -0.01                # allow1%drawdown
        elif current_profit > 0.03:    # profit3%+，translated
            return 0.001                # translated+translated
        
        return None                     # none profit，useATRtranslated
    
    def _calculate_trend_adjustment(self, current_candle: dict, is_short: bool, entry_atr_p: float) -> float:
        """
        trend strength - translated，translated
        """
        # get trend indicator
        ema_8 = current_candle.get('ema_8', 0)
        ema_21 = current_candle.get('ema_21', 0)
        adx = current_candle.get('adx', 25)
        current_price = current_candle.get('close', 0)
        
        # trend
        is_uptrend = ema_8 > ema_21 and adx > 25
        is_downtrend = ema_8 < ema_21 and adx > 25
        
        # trend 1 check
        if is_short and is_downtrend:      # short+down trend，translated
            return 1.2                     # translated20%
        elif not is_short and is_uptrend:  # long+up trend，translated
            return 1.2                     # translated20%
        elif is_short and is_uptrend:      # short+up trend，translated
            return 0.8                     # translated20%
        elif not is_short and is_downtrend: # long+down trend，translated  
            return 0.8                     # translated20%
        else:                              # or not
            return 1.0                     # none
    
    def _log_stoploss_calculation(self, pair: str, trade: Trade, current_profit: float,
                                 entry_atr_p: float, base_atr_multiplier: float,
                                 time_decay_factor: float, trend_adjustment: float,
                                 final_stoploss: float):
        """
        record calculate - optimize
        """
        hours_held = (datetime.now(timezone.utc) - trade.open_date_utc).total_seconds() / 3600
        
        logger.info(
            f"🛡️ ATRtranslated {pair} [{trade.enter_tag}]: "
            f"profit{current_profit:.1%} | "
            f"translated{hours_held:.1f}h | "
            f"translatedATR{entry_atr_p:.3f} | "
            f"ATRtranslated{base_atr_multiplier:.1f} | "
            f"time{time_decay_factor:.2f} | " 
            f"trend{trend_adjustment:.2f} | "
            f"most{final_stoploss:.3f}"
        )
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, 
                       after_fill: bool = False, **kwargs) -> Optional[float]:
        """
        🚀 translatedATRtranslated
        - translatedATRtranslated
        - time long
        - profit
        - trend strength
        """
        try:
            # get most new
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty or len(dataframe) < 50:
                return None
                
            current_candle = dataframe.iloc[-1]
            
            # === 1. getATR (translated) ===
            entry_atr_p = self._get_trade_entry_atr(trade, dataframe)
            current_atr_p = current_candle.get('atr_p', 0.02)
            
            # === 2. calculateATRtranslated ===
            # translated2.5-3translatedATRas most，translated
            base_atr_multiplier = self._calculate_atr_multiplier(
                entry_atr_p, current_candle, trade.enter_tag
            )
            base_stop_distance = entry_atr_p * base_atr_multiplier
            
            # === 3. time ===
            # translated，translated，long
            hours_held = (current_time - trade.open_date_utc).total_seconds() / 3600
            time_decay_factor = self._calculate_time_decay(hours_held, current_profit)
            
            # === 4. profit ===
            profit_protection = self._calculate_profit_protection(current_profit)
            if profit_protection is not None:
                return profit_protection
                
            # === 5. trend strength ===
            trend_adjustment = self._calculate_trend_adjustment(
                current_candle, trade.is_short, entry_atr_p
            )
            
            # === 6. most calculate ===
            final_stop_distance = (base_stop_distance * time_decay_factor * trend_adjustment)
            
            # translated：most small1%，most large8%
            final_stop_distance = max(0.01, min(0.08, final_stop_distance))
            
            # short
            final_stoploss = -final_stop_distance if not trade.is_short else final_stop_distance
            
            # === 7. day record ===
            if self.config.get('verbosity', 0) > 1:
                self._log_stoploss_calculation(
                    pair, trade, current_profit, entry_atr_p, base_atr_multiplier,
                    time_decay_factor, trend_adjustment, final_stoploss
                )
                
            return final_stoploss
            
        except Exception as e:
            logger.error(f"ATRcalculate {pair}: {e}")
            # use
            return -0.03 if not trade.is_short else 0.03
    
    def _calculate_signal_quality_score(self, dataframe: DataFrame, signal_mask: pd.Series, signal_type: str) -> pd.Series:
        """
        🎯 signal score (1-10translated)
        signal，as risk
        """
        # initialize score
        scores = pd.Series(0.0, index=dataframe.index)
        
        # has signal calculate score
        signal_indices = signal_mask[signal_mask].index
        
        for idx in signal_indices:
            try:
                score = 3.0  # translated
                current_data = dataframe.loc[idx]
                
                # === 1. indicator 1 (0-2translated) ===
                rsi = current_data.get('rsi_14', 50)
                if signal_type in ['RSI_Oversold_Bounce']:
                    if rsi < 25:
                        score += 2    # oversold，large
                    elif rsi < 30:
                        score += 1.5  # oversold
                elif signal_type in ['RSI_Overbought_Fall']:
                    if rsi > 75:
                        score += 2    # overbought，risk large
                    elif rsi > 70:
                        score += 1.5  # overbought
                
                # === 2. trend strength and (0-2translated) ===
                adx = current_data.get('adx', 25)
                ema_8 = current_data.get('ema_8', 0)
                ema_21 = current_data.get('ema_21', 0)
                
                if adx > 30:  # strong trend
                    if signal_type in ['RSI_Oversold_Bounce'] and ema_8 > ema_21:
                        score += 2  # rise trend mid oversold，high
                    elif signal_type in ['RSI_Overbought_Fall'] and ema_8 < ema_21:
                        score += 2  # down trend mid overbought，high
                    else:
                        score += 0.5  # signal，1
                elif 20 < adx <= 30:  # medium trend
                    score += 1
                
                # === 3. volume confirmation (0-1.5translated) ===
                volume_ratio = current_data.get('volume_ratio', 1.0)
                if volume_ratio > 1.5:
                    score += 1.5  # volume
                elif volume_ratio > 1.2:
                    score += 1.0  # volume large
                elif volume_ratio > 1.0:
                    score += 0.5  # volume
                
                # === 4. volatility (0-1translated) ===
                atr_percentile = dataframe['atr_p'].rolling(50).rank(pct=True).loc[idx]
                if 0.2 <= atr_percentile <= 0.8:  # translated
                    score += 1
                elif atr_percentile > 0.9:  # high volatility，risk large
                    score -= 0.5
                
                # === 5. divergence signal (0-1translated) ===
                no_bearish_div = not current_data.get('bearish_divergence', False)
                no_bullish_div = not current_data.get('bullish_divergence', False)
                
                if signal_type in ['RSI_Oversold_Bounce'] and no_bearish_div:
                    score += 1
                elif signal_type in ['RSI_Overbought_Fall'] and no_bullish_div:
                    score += 1
                
                # === 6. translated (0-0.5translated) ===
                price_position = current_data.get('price_position', 0.5)
                if signal_type in ['RSI_Oversold_Bounce'] and 0.2 < price_position < 0.7:
                    score += 0.5  # not at long
                elif signal_type in ['RSI_Overbought_Fall'] and 0.3 < price_position < 0.8:
                    score += 0.5  # not at short
                
                # score range
                scores.loc[idx] = max(1.0, min(10.0, score))
                
            except Exception as e:
                scores.loc[idx] = 3.0  # default score
                logger.warning(f"signal score calculate {signal_type}: {e}")
        
        return scores
    
    def _calculate_macd_signal_quality(self, dataframe: DataFrame, signal_mask: pd.Series, signal_type: str) -> pd.Series:
        """
        🎯 MACDsignal score (1-10translated)
        translatedMACDsignal，score
        """
        # initialize score
        scores = pd.Series(0.0, index=dataframe.index)
        
        # has signal calculate score
        signal_indices = signal_mask[signal_mask].index
        
        for idx in signal_indices:
            try:
                score = 2.0  # MACDsignal low，confirm
                current_data = dataframe.loc[idx]
                
                # === 1. MACDsignal strength (0-2.5translated) ===
                macd = current_data.get('macd', 0)
                macd_signal = current_data.get('macd_signal', 0)
                macd_hist = current_data.get('macd_hist', 0)
                
                # MACDlarge，signal
                cross_magnitude = abs(macd - macd_signal)
                if cross_magnitude > 0.002:  # translated
                    score += 2.5
                elif cross_magnitude > 0.001:  # translated
                    score += 1.5
                elif cross_magnitude > 0.0005:  # translated
                    score += 1.0
                
                # === 2. trend 1 (0-2translated) ===
                ema_8 = current_data.get('ema_8', 0)
                ema_21 = current_data.get('ema_21', 0)
                ema_50 = current_data.get('ema_50', 0)
                
                if ema_8 < ema_21 < ema_50:  # bearish
                    score += 2
                elif ema_8 < ema_21:  # bearish
                    score += 1
                
                # === 3. momentum confirm (0-2translated) ===
                rsi = current_data.get('rsi_14', 50)
                rsi_prev = dataframe['rsi_14'].iloc[max(0, idx-2):idx].mean()
                
                if rsi < 45 and rsi < rsi_prev:  # RSIdown
                    score += 2
                elif rsi < 50:  # RSItranslated
                    score += 1
                
                # === 4. volume (0-1.5translated) ===
                volume_ratio = current_data.get('volume_ratio', 1.0)
                volume_trend = dataframe['volume'].iloc[max(0, idx-3):idx+1].iloc[-1] > \
                              dataframe['volume'].iloc[max(0, idx-3):idx+1].iloc[0]
                
                if volume_ratio > 1.5 and volume_trend:  # volume and
                    score += 1.5
                elif volume_ratio > 1.2:  # volume large
                    score += 1.0
                
                # === 5. ADXtrend strength (0-1.5translated) ===
                adx = current_data.get('adx', 25)
                adx_trend = current_data.get('adx', 25) > dataframe['adx'].iloc[max(0, idx-3)]
                
                if adx > 35 and adx_trend:  # strong trend and
                    score += 1.5
                elif adx > 25:  # medium trend
                    score += 1.0
                
                # === 6. translated (0-1translated) ===
                # MACDmost at mid signal
                if adx > 25:  # not at
                    score += 1
                else:
                    score -= 1  # translated
                
                # === 7. translated (0-0.5translated) ===
                price_position = current_data.get('price_position', 0.5)
                if 0.4 < price_position < 0.8:  # at short
                    score += 0.5
                
                # === 8. divergence (0-0.5translated) ===
                no_bullish_div = not current_data.get('bullish_divergence', False)
                if no_bullish_div:
                    score += 0.5
                
                # score range
                scores.loc[idx] = max(1.0, min(10.0, score))
                
            except Exception as e:
                scores.loc[idx] = 2.0  # MACDdefault score low
                logger.warning(f"MACDsignal score calculate: {e}")
        
        return scores
    
    def _enhanced_market_regime_detection(self, dataframe: DataFrame) -> Dict[str, Any]:
        """
        🌍 market state
        as signal risk
        """
        try:
            if dataframe.empty or len(dataframe) < 50:
                return {'regime': 'UNKNOWN', 'confidence': 0.0, 'characteristics': {}}
            
            current_data = dataframe.iloc[-1]
            recent_data = dataframe.tail(30)
            
            # === 1. trend ===
            adx = current_data.get('adx', 25)
            ema_8 = current_data.get('ema_8', 0)
            ema_21 = current_data.get('ema_21', 0)
            ema_50 = current_data.get('ema_50', 0)
            
            # trend strength
            if adx > 35:
                trend_strength = 'STRONG'
            elif adx > 25:
                trend_strength = 'MODERATE' 
            elif adx > 15:
                trend_strength = 'WEAK'
            else:
                trend_strength = 'SIDEWAYS'
            
            # trend
            if ema_8 > ema_21 > ema_50:
                trend_direction = 'UPTREND'
            elif ema_8 < ema_21 < ema_50:
                trend_direction = 'DOWNTREND'
            else:
                trend_direction = 'SIDEWAYS'
            
            # === 2. volatility ===
            atr_p = current_data.get('atr_p', 0.02)
            atr_percentile = dataframe['atr_p'].rolling(50).rank(pct=True).iloc[-1]
            
            if atr_percentile > 0.8:
                volatility_regime = 'HIGH'
            elif atr_percentile > 0.6:
                volatility_regime = 'ELEVATED'
            elif atr_percentile > 0.3:
                volatility_regime = 'NORMAL'
            else:
                volatility_regime = 'LOW'
            
            # === 3. volume ===
            volume_ratio = current_data.get('volume_ratio', 1.0)
            avg_volume_ratio = recent_data['volume_ratio'].mean()
            
            if avg_volume_ratio > 1.3:
                volume_regime = 'HIGH_ACTIVITY'
            elif avg_volume_ratio > 1.1:
                volume_regime = 'ACTIVE'
            elif avg_volume_ratio > 0.8:
                volume_regime = 'NORMAL'
            else:
                volume_regime = 'LOW'
            
            # === 4. price ===
            high_20 = dataframe['high'].rolling(20).max().iloc[-1]
            low_20 = dataframe['low'].rolling(20).min().iloc[-1]
            current_price = current_data.get('close', 0)
            price_position = (current_price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
            
            if price_position > 0.8:
                position_regime = 'NEAR_HIGH'
            elif price_position > 0.6:
                position_regime = 'UPPER_RANGE'
            elif price_position > 0.4:
                position_regime = 'MIDDLE_RANGE'
            elif price_position > 0.2:
                position_regime = 'LOWER_RANGE'
            else:
                position_regime = 'NEAR_LOW'
            
            # === 5. market state ===
            regime_score = 0
            confidence_factors = []
            
            # strong trend
            if trend_strength in ['STRONG', 'MODERATE'] and trend_direction != 'SIDEWAYS':
                if volatility_regime in ['NORMAL', 'ELEVATED']:
                    regime = f"TRENDING_{trend_direction}"
                    regime_score += 3
                    confidence_factors.append("strong_trend")
                else:
                    regime = f"VOLATILE_{trend_direction}"
                    regime_score += 2
                    confidence_factors.append("volatile_trend")
            
            # translated
            elif trend_strength in ['WEAK', 'SIDEWAYS']:
                if volatility_regime in ['HIGH', 'ELEVATED']:
                    regime = "CHOPPY_SIDEWAYS"
                    regime_score += 1
                    confidence_factors.append("high_vol_sideways")
                else:
                    regime = "QUIET_SIDEWAYS"
                    regime_score += 2
                    confidence_factors.append("low_vol_sideways")
            
            # not
            else:
                regime = "TRANSITIONAL"
                regime_score += 1
                confidence_factors.append("uncertain")
            
            # === 6. translated ===
            special_conditions = []
            
            # translated
            if atr_p > 0.06:
                special_conditions.append("EXTREME_VOLATILITY")
                regime_score -= 1
            
            # volume abnormal
            if volume_ratio > 2.0:
                special_conditions.append("VOLUME_SPIKE")
                regime_score += 1
            elif volume_ratio < 0.5:
                special_conditions.append("VOLUME_DRYING")
                regime_score -= 1
            
            # translated
            if position_regime in ['NEAR_HIGH', 'NEAR_LOW']:
                special_conditions.append(f"EXTREME_POSITION_{position_regime}")
            
            # === 7. calculate ===
            base_confidence = min(0.9, regime_score / 5.0)
            
            # translated
            data_quality = min(1.0, len(dataframe) / 100)
            final_confidence = base_confidence * data_quality
            
            return {
                'regime': regime,
                'confidence': max(0.1, final_confidence),
                'characteristics': {
                    'trend_strength': trend_strength,
                    'trend_direction': trend_direction,
                    'volatility_regime': volatility_regime,
                    'volume_regime': volume_regime,
                    'position_regime': position_regime,
                    'special_conditions': special_conditions,
                    'adx': adx,
                    'atr_percentile': atr_percentile,
                    'price_position': price_position,
                    'volume_ratio': volume_ratio
                },
                'signals_advice': self._get_regime_trading_advice(regime, volatility_regime, position_regime),
                'confidence_factors': confidence_factors
            }
            
        except Exception as e:
            logger.error(f"market state: {e}")
            return {
                'regime': 'ERROR',
                'confidence': 0.0,
                'characteristics': {},
                'signals_advice': {'recommended_signals': [], 'avoid_signals': []},
                'confidence_factors': []
            }
    
    def _get_regime_trading_advice(self, regime: str, volatility_regime: str, position_regime: str) -> Dict[str, list]:
        """
        market state
        """
        advice = {
            'recommended_signals': [],
            'avoid_signals': [],
            'risk_adjustment': 1.0,
            'position_size_multiplier': 1.0
        }
        
        # not market state
        if 'TRENDING_UPTREND' in regime:
            advice['recommended_signals'] = ['RSI_Oversold_Bounce', 'EMA_Golden_Cross']
            advice['avoid_signals'] = ['RSI_Overbought_Fall'] 
            advice['position_size_multiplier'] = 1.2
            
        elif 'TRENDING_DOWNTREND' in regime:
            advice['recommended_signals'] = ['RSI_Overbought_Fall', 'MACD_Bearish']
            advice['avoid_signals'] = ['RSI_Oversold_Bounce']
            advice['position_size_multiplier'] = 1.2
            
        elif 'SIDEWAYS' in regime:
            if volatility_regime == 'LOW':
                advice['recommended_signals'] = ['RSI_Oversold_Bounce', 'RSI_Overbought_Fall']
                advice['avoid_signals'] = ['MACD_Bearish']
            else:
                advice['avoid_signals'] = ['MACD_Bearish', 'RSI_Overbought_Fall', 'RSI_Oversold_Bounce']
            advice['position_size_multiplier'] = 0.7
            
        elif 'VOLATILE' in regime:
            advice['avoid_signals'] = ['MACD_Bearish']
            advice['risk_adjustment'] = 1.5
            advice['position_size_multiplier'] = 0.6
            
        # translated
        if position_regime in ['NEAR_HIGH']:
            advice['avoid_signals'].extend(['RSI_Oversold_Bounce'])
            advice['position_size_multiplier'] *= 0.8
        elif position_regime in ['NEAR_LOW']:
            advice['avoid_signals'].extend(['RSI_Overbought_Fall', 'MACD_Bearish'])
            advice['position_size_multiplier'] *= 0.8
        
        return advice
    
    # === 🎯 leverage ===
    
    def _calculate_signal_quality_leverage_bonus(self, entry_tag: str, current_data: dict, 
                                               regime: str, signals_advice: dict) -> float:
        """
        signal calculate leverage
        high signal allow high leverage
        """
        if not entry_tag:
            return 1.0
        
        # get signal score（has）
        signal_quality = current_data.get('signal_quality', 5.0)
        
        # translated：5-10to0.8-1.5translated
        quality_bonus = 0.8 + (signal_quality - 5.0) / 5.0 * 0.7
        quality_bonus = max(0.8, min(1.5, quality_bonus))
        
        # market state：signal
        regime_bonus = 1.0
        if entry_tag in signals_advice.get('recommended_signals', []):
            regime_bonus = 1.2  # signal+20%leverage
        elif entry_tag in signals_advice.get('avoid_signals', []):
            regime_bonus = 0.6  # not signal-40%leverage
        
        return quality_bonus * regime_bonus
    
    def _get_regime_leverage_multiplier(self, regime: str, confidence: float) -> float:
        """
        market state calculate leverage
        """
        base_multiplier = 1.0
        
        # market state
        if 'TRENDING' in regime:
            if 'UPTREND' in regime or 'DOWNTREND' in regime:
                base_multiplier = 1.3  # trend+30%leverage
            else:
                base_multiplier = 1.1  # 1 trend+10%leverage
                
        elif 'SIDEWAYS' in regime:
            if 'QUIET' in regime:
                base_multiplier = 1.1  # translated+10%leverage
            else:
                base_multiplier = 0.8  # translated-20%leverage
                
        elif 'VOLATILE' in regime:
            base_multiplier = 0.7  # high volatility-30%leverage
            
        elif 'TRANSITIONAL' in regime:
            base_multiplier = 0.9  # translated-10%leverage
        
        # translated：high confidence
        confidence_multiplier = 0.8 + confidence * 0.4  # 0.8-1.2range
        
        return base_multiplier * confidence_multiplier
    
    def _get_signal_leverage_multiplier(self, entry_tag: str, signals_advice: dict) -> float:
        """
        signal calculate leverage
        """
        if not entry_tag:
            return 1.0
        
        # signal
        signal_reliability = {
            'RSI_Oversold_Bounce': 1.2,    # RSIsignal
            'RSI_Overbought_Fall': 1.2,
            'EMA_Golden_Cross': 1.3,       # trend signal most
            'EMA_Death_Cross': 1.3,
            'MACD_Bearish': 1.0,           # MACDsignal
            'MACD_Bullish': 1.0,
        }
        
        base_multiplier = signal_reliability.get(entry_tag, 1.0)
        
        # translated
        if entry_tag in signals_advice.get('recommended_signals', []):
            base_multiplier *= 1.1  # translated+10%
        elif entry_tag in signals_advice.get('avoid_signals', []):
            base_multiplier *= 0.7  # translated-30%
        
        return base_multiplier
    
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """
        signal - translated
        notROI，signal optimize
        """

        # initialize signal
        dataframe.loc[:, 'exit_long'] = 0
        dataframe.loc[:, 'exit_short'] = 0

        # ==============================
        # 1. reversal signal（most high）
        # ==============================

        # long reversal（translated：translated）
        strong_reversal_long_exit = (
            # RSIoverbought or divergence
            ((dataframe['rsi_14'] > 70) |  # original75，translated
             (dataframe['bearish_divergence'] == 1) |  # new：bearish divergence
             # MACDtranslated
             ((dataframe['macd_hist'].shift(1) > 0) & (dataframe['macd_hist'] < 0))) &
            # volume abnormal or
            ((dataframe['volume_ratio'] > 1.8) | (dataframe['volume_exhaustion'] == 1)) &  # original2.0，translated
            # price at Bollinger Bands up
            (dataframe['bb_position'] > 0.85) &  # original0.9，translated
            # new：momentum or trend late stage
            ((dataframe['momentum_exhaustion_score'] > 0.4) | (dataframe['trend_phase'] >= 3))
        )

        # short reversal（translated：translated）
        strong_reversal_short_exit = (
            # RSIoversold or divergence
            ((dataframe['rsi_14'] < 30) |  # original25，translated
             (dataframe['bullish_divergence'] == 1) |  # new：bullish divergence
             # MACDtranslated
             ((dataframe['macd_hist'].shift(1) < 0) & (dataframe['macd_hist'] > 0))) &
            # volume abnormal or
            ((dataframe['volume_ratio'] > 1.8) | (dataframe['volume_exhaustion'] == 1)) &  # original2.0，translated
            # price at Bollinger Bands down
            (dataframe['bb_position'] < 0.15) &  # original0.1，translated
            # new：momentum or trend late stage
            ((dataframe['momentum_exhaustion_score'] > 0.4) | (dataframe['trend_phase'] >= 3))
        )

        # ==============================
        # 2. trend signal
        # ==============================

        # long trend
        trend_exhaustion_long = (
            # ADXfall and low value
            (dataframe['adx'] < 20) &
            (dataframe['adx'] < dataframe['adx'].shift(1)) &
            # price Bollinger Bands mid
            (abs(dataframe['bb_position'] - 0.5) < 0.15) &
            # momentum indicators
            (dataframe['mom_10'] < 0) &
            # DI-rise
            (dataframe['plus_di'] < dataframe['minus_di'])
        )

        # short trend
        trend_exhaustion_short = (
            # ADXfall and low value
            (dataframe['adx'] < 20) &
            (dataframe['adx'] < dataframe['adx'].shift(1)) &
            # price Bollinger Bands mid
            (abs(dataframe['bb_position'] - 0.5) < 0.15) &
            # momentum indicators
            (dataframe['mom_10'] > 0) &
            # DI+rise
            (dataframe['plus_di'] > dataframe['minus_di'])
        )

        # ==============================
        # 3. translated
        # ==============================

        # long
        technical_exit_long = (
            # priceEMA13and short-term
            (dataframe['close'] < dataframe['ema_13']) &
            (dataframe['ema_5'] < dataframe['ema_8']) &
            # MACDtranslated
            (dataframe['macd'] < dataframe['macd_signal']) &
            # volume
            (dataframe['volume_ratio'] < 0.8)
        )

        # short
        technical_exit_short = (
            # price breakoutEMA13and short-term
            (dataframe['close'] > dataframe['ema_13']) &
            (dataframe['ema_5'] > dataframe['ema_8']) &
            # MACDtranslated
            (dataframe['macd'] > dataframe['macd_signal']) &
            # volume
            (dataframe['volume_ratio'] < 0.8)
        )

        # ==============================
        # 4. signal
        # ==============================

        # long
        microstructure_exit_long = (
            # translated（large）
            (dataframe['ob_depth_imbalance'] < -0.3) &
            # translated
            (dataframe['ob_liquidity_score'] < 0.3) &
            # translated
            (dataframe['ob_buy_pressure'] < 0.3)
        )

        # short
        microstructure_exit_short = (
            # translated（large）
            (dataframe['ob_depth_imbalance'] > 0.3) &
            # translated
            (dataframe['ob_liquidity_score'] < 0.3) &
            # translated
            (dataframe['ob_sell_pressure'] < 0.3)
        )

        # ==============================
        # 5. volatility
        # ==============================

        # ATRlarge（has）
        volatility_protection = (
            # ATRlarge value2translated
            (dataframe['atr'] > dataframe['atr'].rolling(20).mean() * 2) |
            # orATRtranslated5%
            (dataframe['atr_p'] > 0.05)
        )

        # ==============================
        # 6. market state
        # ==============================

        # get market state
        is_bull_market = dataframe['market_state'] == 'bullish'
        is_bear_market = dataframe['market_state'] == 'bearish'
        is_sideways = dataframe['market_state'] == 'sideways'

        # ==============================
        # has
        # ==============================

        # long signal
        dataframe.loc[
            (
                strong_reversal_long_exit |  # reversal
                trend_exhaustion_long |      # trend
                technical_exit_long |        # translated
                microstructure_exit_long |   # translated
                (volatility_protection & is_bear_market)  # mid
            ),
            'exit_long'
        ] = 1

        # short signal
        dataframe.loc[
            (
                strong_reversal_short_exit |  # reversal
                trend_exhaustion_short |       # trend
                technical_exit_short |         # translated
                microstructure_exit_short |    # translated
                (volatility_protection & is_bull_market)  # mid
            ),
            'exit_short'
        ] = 1

        # to
        dataframe.loc[strong_reversal_long_exit, 'exit_tag'] = 'strong_reversal'
        dataframe.loc[trend_exhaustion_long, 'exit_tag'] = 'trend_exhaustion'
        dataframe.loc[technical_exit_long, 'exit_tag'] = 'technical_exit'
        dataframe.loc[microstructure_exit_long, 'exit_tag'] = 'microstructure'
        dataframe.loc[volatility_protection, 'exit_tag'] = 'volatility_protection'

        # ==============================
        # 🚨 fix：translated - avoidKsignal
        # ==============================
        
        # 1Ksignal
        signal_conflict = (dataframe['enter_long'] == 1) & (dataframe['enter_short'] == 1)
        
        # translated：signal strength or trend
        conflict_resolution_favor_long = (
            signal_conflict &
            (
                (dataframe['trend_strength'] > 0) |  # trend long
                (dataframe['rsi_14'] < 50) |         # RSIlow long
                (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1))  # MACDlong
            )
        )
        
        # translated：high signal，low signal
        dataframe.loc[conflict_resolution_favor_long, 'enter_short'] = 0
        dataframe.loc[signal_conflict & ~conflict_resolution_favor_long, 'enter_long'] = 0
        
        # recalculate after signal
        clean_enter_long = dataframe['enter_long'] == 1
        clean_enter_short = dataframe['enter_short'] == 1
        
        # translated：at
        # original：translated，translated
        strong_bullish_signal = (
            clean_enter_long &
            (dataframe['rsi_14'] > 30) &  # avoid at oversold
            (dataframe['volume_ratio'] > 1.1)  # volume support
        )
        
        strong_bearish_signal = (
            clean_enter_short &
            (dataframe['rsi_14'] < 70) &  # avoid at overbought
            (dataframe['volume_ratio'] > 1.1)  # volume support
        )
        
        # translated（avoid）
        dataframe.loc[strong_bullish_signal, 'exit_short'] = 1
        dataframe.loc[strong_bearish_signal, 'exit_long'] = 1
        
        # updateexit_tagto（translated）
        dataframe.loc[
            strong_bullish_signal & (dataframe['exit_short'] == 1),
            'exit_tag'
        ] = 'smart_cross_exit_bullish'
        
        dataframe.loc[
            strong_bearish_signal & (dataframe['exit_long'] == 1),
            'exit_tag' 
        ] = 'smart_cross_exit_bearish'

        # record signal
        exit_long_count = dataframe['exit_long'].sum()
        exit_short_count = dataframe['exit_short'].sum()

        if exit_long_count > 0 or exit_short_count > 0:
            logger.info(f"""
📤 signal - {metadata['pair']}:
├─ long signal: {exit_long_count}count
├─ short signal: {exit_short_count}count
└─ time range: {dataframe.index[0]} - {dataframe.index[-1]}
""")

        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """position size large small"""
        
        try:
            # get most new
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if dataframe.empty:
                return proposed_stake
            
            # get market state
            market_state = dataframe['market_state'].iloc[-1] if 'market_state' in dataframe.columns else 'sideways'
            volatility = dataframe['atr_p'].iloc[-1] if 'atr_p' in dataframe.columns else 0.02
            
            # === 🎯 risk ===
            coin_risk_tier = self.identify_coin_risk_tier(pair, dataframe)
            
            # risk（small position size to small large）
            coin_risk_multipliers = {
                'low_risk': 1.0,        # low risk：position size
                'medium_risk': 0.7,     # medium risk：70%position size
                'high_risk': 0.25       # high risk（translated）：25%position size，to small large
            }
            
            # get risk
            coin_risk_multiplier = coin_risk_multipliers.get(coin_risk_tier, 0.7)
            
            # calculate position size large small
            position_size_ratio = self.calculate_position_size(current_rate, market_state, pair)
            
            # get
            available_balance = self.wallets.get_free(self.config['stake_currency'])
            
            # === risk to position size calculate ===
            # position size calculate
            base_calculated_stake = available_balance * position_size_ratio
            
            # risk（small position size）
            calculated_stake = base_calculated_stake * coin_risk_multiplier
            
            # calculate leverage
            dynamic_leverage = self.calculate_leverage(market_state, volatility, pair, current_time)
            
            # translated：atFreqtrademid，leverageleverage()translated，calculate position size
            # leverage，not to leverage
            # leveraged_stake = calculated_stake * dynamic_leverage  # translated
            leveraged_stake = calculated_stake  # position size
            
            # record leverage
            base_position_value = calculated_stake
            
            # at range
            final_stake = max(min_stake or 0, min(leveraged_stake, max_stake))
            
            # leverage day
            risk_tier_names = {
                'low_risk': '✅ low risk',
                'medium_risk': '⚡ medium risk', 
                'high_risk': '⚠️ high risk'
            }
            
            logger.info(f"""
🎯 position size calculate - {pair}:
├─ market state: {market_state}
├─ 🔍 risk level: {risk_tier_names.get(coin_risk_tier, coin_risk_tier)}
├─ 📊 position size: ${base_calculated_stake:.2f} ({position_size_ratio:.2%})
├─ 🎯 risk adjustment: {coin_risk_multiplier:.2f}x ({coin_risk_tier})
├─ 💰 after position size: ${calculated_stake:.2f}
├─ ⚡ calculate leverage: {dynamic_leverage}x (translatedleverage()translated)
├─ 🎉 most: ${final_stake:.2f}
├─ 📈 translated: {final_stake / current_rate:.6f}
└─ ⏰ time: {current_time}
""")
            
            # translated：before leverage（translatedFreqtradeuse）
            if hasattr(self, '_current_leverage'):
                self._current_leverage[pair] = dynamic_leverage
            else:
                self._current_leverage = {pair: dynamic_leverage}
            
            # record risk calculate day
            self._log_risk_calculation_details(pair, {
                'current_price': current_rate,
                'planned_position': position_size_ratio,
                'stoploss_level': abs(self.stoploss),
                'leverage': dynamic_leverage,
                'market_state': market_state,
                'volatility': volatility
            }, {
                'risk_amount': final_stake * abs(self.stoploss),
                'risk_percentage': (final_stake * abs(self.stoploss)) / available_balance,
                'max_loss': final_stake * abs(self.stoploss),
                'adjusted_position': position_size_ratio,
                'suggested_leverage': dynamic_leverage,
                'risk_rating': self._calculate_risk_rating(final_stake * abs(self.stoploss) / available_balance),
                'rating_reason': f'translated{market_state}market state{volatility*100:.1f}%volatility'
            })
            
            return final_stake
            
        except Exception as e:
            logger.error(f"position size calculate: {e}")
            return proposed_stake
    
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                            current_rate: float, current_profit: float,
                            min_stake: Optional[float], max_stake: float,
                            current_entry_rate: float, current_exit_rate: float,
                            current_entry_profit: float, current_exit_profit: float,
                            **kwargs) -> Optional[float]:
        """translatedDCAtranslated - confirm and risk"""
        
        # check allowDCA
        if trade.nr_of_successful_entries >= self.max_dca_orders:
            logger.info(f"DCAtranslated {trade.pair}: most large times {self.max_dca_orders}")
            return None
            
        # get indicator
        dataframe = self.get_dataframe_with_indicators(trade.pair, self.timeframe)
        if dataframe.empty:
            logger.warning(f"DCAcheck {trade.pair}: none")
            return None
            
        # most check indicator at
        required_indicators = ['rsi_14', 'adx', 'atr_p', 'macd', 'macd_signal', 'volume_ratio', 'trend_strength', 'momentum_score']
        missing_indicators = [indicator for indicator in required_indicators if indicator not in dataframe.columns]
        
        if missing_indicators:
            logger.warning(f"DCAcheck {trade.pair}: indicator still {missing_indicators}，translatedDCA")
            return None
            
        # get indicator
        current_data = dataframe.iloc[-1]
        prev_data = dataframe.iloc[-2] if len(dataframe) > 1 else current_data
        
        current_rsi = current_data.get('rsi_14', 50)
        current_adx = current_data.get('adx', 25)
        current_atr_p = current_data.get('atr_p', 0.02)
        trend_strength = current_data.get('trend_strength', 50)
        momentum_score = current_data.get('momentum_score', 0)
        volume_ratio = current_data.get('volume_ratio', 1)
        signal_strength = current_data.get('signal_strength', 0)
        bb_position = current_data.get('bb_position', 0.5)
        market_state = current_data.get('market_state', 'sideways')
        
        # calculate
        entry_price = trade.open_rate
        price_deviation = abs(current_rate - entry_price) / entry_price
        hold_time = current_time - trade.open_date_utc
        hold_hours = hold_time.total_seconds() / 3600
        
        # === translatedDCAtranslated ===
        
        dca_decision = self._analyze_dca_opportunity(
            trade, current_rate, current_profit, price_deviation,
            current_data, prev_data, hold_hours, market_state
        )
        
        if dca_decision['should_dca']:
            # calculateDCAtranslated
            dca_amount = self._calculate_smart_dca_amount(
                trade, dca_decision, current_data, market_state
            )
            
            # most risk check
            risk_check = self._dca_risk_validation(trade, dca_amount, current_data)
            
            if risk_check['approved']:
                final_dca_amount = risk_check['adjusted_amount']
                
                # recordDCAday
                self._log_dca_decision(
                    trade, current_rate, current_profit, price_deviation,
                    dca_decision, final_dca_amount, current_data
                )
                
                # translatedDCAtranslated
                self.track_dca_performance(trade, dca_decision['dca_type'], final_dca_amount)
                
                return final_dca_amount
            else:
                logger.warning(f"DCArisk check {trade.pair}: {risk_check['reason']}")
                return None
        
        return None
    
    # translated _analyze_dca_opportunity - simplified
    def _analyze_dca_opportunity(self, trade: Trade, current_rate: float, 
                               current_profit: float, price_deviation: float,
                               current_data: dict, prev_data: dict, 
                               hold_hours: float, market_state: str) -> dict:
        """translatedDCAtranslated - translated"""
        
        decision = {
            'should_dca': False,
            'dca_type': None,
            'confidence': 0.0,
            'risk_level': 'high',
            'technical_reasons': [],
            'market_conditions': {}
        }
        
        try:
            # === translatedDCAtranslated ===
            basic_trigger_met = (
                price_deviation > self.dca_price_deviation and  # price
                current_profit < -0.03 and  # translated3%to up（low）
                hold_hours > 0.5  # translated30minutes
            )
            
            if not basic_trigger_met:
                return decision
            
            # === translatedDCAtranslated ===
            
            if not trade.is_short:
                # === longDCAtranslated ===
                
                # 1. oversoldDCA - mostDCAtranslated
                oversold_dca = (
                    current_rate < trade.open_rate and  # price down
                    current_data.get('rsi_14', 50) < 35 and  # RSIoversold
                    current_data.get('bb_position', 0.5) < 0.2 and  # Bollinger Bands down
                    current_data.get('momentum_score', 0) > prev_data.get('momentum_score', 0)  # momentum
                )
                
                if oversold_dca:
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'OVERSOLD_REVERSAL_DCA',
                        'confidence': 0.8,
                        'risk_level': 'low'
                    })
                    decision['technical_reasons'].append(f"RSI{current_data.get('rsi_14', 50):.1f}oversold")
                
                # 2. supportDCA - at support
                elif (current_data.get('close', 0) > current_data.get('ema_50', 0) and  # still at long-term trend up
                      abs(current_rate - current_data.get('ema_21', 0)) / current_rate < 0.02 and  # translatedEMA21support
                      current_data.get('adx', 25) > 20):  # trend still has
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'SUPPORT_LEVEL_DCA',
                        'confidence': 0.7,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append("EMA21support")
                
                # 3. trendDCA - trend
                elif (current_data.get('trend_strength', 50) > 30 and  # trend still up
                      current_data.get('adx', 25) > 25 and  # ADXconfirm trend
                      current_data.get('signal_strength', 0) > 0):  # signal still
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'TREND_CONTINUATION_DCA',
                        'confidence': 0.6,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append(f"trend，trend strength{current_data.get('trend_strength', 50):.0f}")
                
                # 4. volume confirmationDCA - has volume support
                elif (current_data.get('volume_ratio', 1) > 1.2 and  # volume large
                      current_data.get('ob_depth_imbalance', 0) > 0.1):  # translated
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'VOLUME_CONFIRMED_DCA',
                        'confidence': 0.5,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append(f"volume{current_data.get('volume_ratio', 1):.1f}confirm")
                
            else:
                # === shortDCAtranslated ===
                
                # 1. overboughtDCA - most bearishDCAtranslated
                overbought_dca = (
                    current_rate > trade.open_rate and  # price up
                    current_data.get('rsi_14', 50) > 65 and  # RSIoverbought
                    current_data.get('bb_position', 0.5) > 0.8 and  # Bollinger Bands up
                    current_data.get('momentum_score', 0) < prev_data.get('momentum_score', 0)  # momentum
                )
                
                if overbought_dca:
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'OVERBOUGHT_REJECTION_DCA',
                        'confidence': 0.8,
                        'risk_level': 'low'
                    })
                    decision['technical_reasons'].append(f"RSI{current_data.get('rsi_14', 50):.1f}overbought")
                
                # 2. resistanceDCA - at resistance
                elif (current_data.get('close', 0) < current_data.get('ema_50', 0) and  # still at long-term trend down
                      abs(current_rate - current_data.get('ema_21', 0)) / current_rate < 0.02 and  # translatedEMA21resistance
                      current_data.get('adx', 25) > 20):  # trend still has
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'RESISTANCE_LEVEL_DCA',
                        'confidence': 0.7,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append("EMA21resistance")
                
                # 3. trendDCA - trend down
                elif (current_data.get('trend_strength', 50) < -30 and  # trend still down
                      current_data.get('adx', 25) > 25 and  # ADXconfirm trend
                      current_data.get('signal_strength', 0) < 0):  # signal still
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'TREND_CONTINUATION_DCA_SHORT',
                        'confidence': 0.6,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append(f"down trend，trend strength{current_data.get('trend_strength', 50):.0f}")
            
            # === translated ===
            decision['market_conditions'] = {
                'market_state': market_state,
                'volatility_acceptable': current_data.get('atr_p', 0.02) < 0.06,  # volatility not high
                'liquidity_sufficient': current_data.get('ob_market_quality', 0.5) > 0.3,  # translated
                'spread_reasonable': current_data.get('ob_spread_pct', 0.1) < 0.4,  # translated
                'trend_not_reversing': abs(current_data.get('trend_strength', 50)) > 20  # trend reversal
            }
            
            # not low orDCA
            unfavorable_conditions = sum([
                not decision['market_conditions']['volatility_acceptable'],
                not decision['market_conditions']['liquidity_sufficient'], 
                not decision['market_conditions']['spread_reasonable'],
                not decision['market_conditions']['trend_not_reversing']
            ])
            
            if unfavorable_conditions >= 2:
                decision['should_dca'] = False
                decision['risk_level'] = 'too_high'
            elif unfavorable_conditions == 1:
                decision['confidence'] *= 0.7  # low
                decision['risk_level'] = 'high'
                
        except Exception as e:
            logger.error(f"DCAtranslated {trade.pair}: {e}")
            decision['should_dca'] = False
            
        return decision
    
    def _calculate_smart_dca_amount(self, trade: Trade, dca_decision: dict, 
                                  current_data: dict, market_state: str) -> float:
        """calculateDCAtranslated - risk"""
        
        try:
            # translatedDCAtranslated
            base_amount = trade.stake_amount
            entry_count = trade.nr_of_successful_entries + 1
            
            # === translatedDCAtranslated ===
            dca_type_multipliers = {
                'OVERSOLD_REVERSAL_DCA': 1.5,  # oversold，translated
                'OVERBOUGHT_REJECTION_DCA': 1.5,  # overbought，translated
                'SUPPORT_LEVEL_DCA': 1.3,  # support，mid
                'RESISTANCE_LEVEL_DCA': 1.3,  # resistance，mid
                'TREND_CONTINUATION_DCA': 1.2,  # trend，translated
                'TREND_CONTINUATION_DCA_SHORT': 1.2,  # bearish trend
                'VOLUME_CONFIRMED_DCA': 1.1  # volume confirmation，translated
            }
            
            type_multiplier = dca_type_multipliers.get(dca_decision['dca_type'], 1.0)
            
            # === translated ===
            confidence_multiplier = 0.5 + (dca_decision['confidence'] * 0.8)  # 0.5-1.3translated
            
            # === market state ===
            market_multipliers = {
                'strong_uptrend': 1.4,  # strong trend midDCAtranslated
                'strong_downtrend': 1.4,
                'mild_uptrend': 1.2,
                'mild_downtrend': 1.2,
                'sideways': 1.0,
                'volatile': 0.7,  # translatedDCA
                'consolidation': 1.1
            }
            market_multiplier = market_multipliers.get(market_state, 1.0)
            
            # === times ===
            # after
            entry_decay = max(0.6, 1.0 - (entry_count - 1) * 0.15)
            
            # === calculateDCAtranslated ===
            total_multiplier = (type_multiplier * confidence_multiplier * 
                              market_multiplier * entry_decay)
            
            calculated_dca = base_amount * total_multiplier
            
            # === translated ===
            available_balance = self.wallets.get_free(self.config['stake_currency'])
            
            # most largeDCAtranslated
            max_dca_ratio = {
                'low': 0.15,      # low risk most15%translated
                'medium': 0.10,   # medium risk10%translated  
                'high': 0.05      # high risk5%translated
            }
            
            max_ratio = max_dca_ratio.get(dca_decision['risk_level'], 0.05)
            max_dca_amount = available_balance * max_ratio
            
            final_dca = min(calculated_dca, max_dca_amount, max_stake or float('inf'))
            
            return max(min_stake or 10, final_dca)
            
        except Exception as e:
            logger.error(f"DCAcalculate {trade.pair}: {e}")
            return trade.stake_amount * 0.5  # default value
    
    def _dca_risk_validation(self, trade: Trade, dca_amount: float, current_data: dict) -> dict:
        """DCArisk - most check"""
        
        risk_check = {
            'approved': True,
            'adjusted_amount': dca_amount,
            'reason': 'DCArisk check',
            'risk_factors': []
        }
        
        try:
            # 1. position size risk check
            available_balance = self.wallets.get_free(self.config['stake_currency'])
            total_exposure = trade.stake_amount + dca_amount
            exposure_ratio = total_exposure / available_balance
            
            if exposure_ratio > 0.4:  # 1 not40%translated
                adjustment = 0.4 / exposure_ratio
                risk_check['adjusted_amount'] = dca_amount * adjustment
                risk_check['risk_factors'].append(f'position size large，as{adjustment:.1%}')
            
            # 2. translatedDCArisk check
            if trade.nr_of_successful_entries >= 3:  # translatedDCA 3times to up
                risk_check['adjusted_amount'] *= 0.7  # afterDCAtranslated
                risk_check['risk_factors'].append('timesDCArisk')
            
            # 3. risk check
            if current_data.get('atr_p', 0.02) > 0.05:  # high volatility
                risk_check['adjusted_amount'] *= 0.8
                risk_check['risk_factors'].append('high volatility risk adjustment')
            
            # 4. drawdown
            if hasattr(self, 'current_drawdown') and self.current_drawdown > 0.08:
                risk_check['adjusted_amount'] *= 0.6
                risk_check['risk_factors'].append('drawdown')
            
            # 5. most small check
            min_meaningful_dca = trade.stake_amount * 0.2  # DCAoriginal position size20%
            if risk_check['adjusted_amount'] < min_meaningful_dca:
                risk_check['approved'] = False
                risk_check['reason'] = f'DCAsmall，low most small has${min_meaningful_dca:.2f}'
            
        except Exception as e:
            risk_check['approved'] = False
            risk_check['reason'] = f'DCArisk check: {e}'
            
        return risk_check
    
    def _log_dca_decision(self, trade: Trade, current_rate: float, current_profit: float,
                         price_deviation: float, dca_decision: dict, dca_amount: float,
                         current_data: dict):
        """recordDCAday"""
        
        try:
            hold_time = datetime.now(timezone.utc) - trade.open_date_utc
            hold_hours = hold_time.total_seconds() / 3600
            
            dca_log = f"""
==================== DCAtranslated ====================
time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | translated: {trade.pair}
times: level{trade.nr_of_successful_entries + 1}times / most large{self.max_dca_orders}times

📊 before:
├─ price: ${trade.open_rate:.6f}
├─ before price: ${current_rate:.6f}
├─ price: {price_deviation:.2%}
├─ before: {current_profit:.2%}
├─ time: {hold_hours:.1f}hours
├─ translated: {'🔻short' if trade.is_short else '🔹long'}
├─ original position size: ${trade.stake_amount:.2f}

🎯 DCAtranslated:
├─ DCAtranslated: {dca_decision['dca_type']}
├─ translated: {dca_decision['confidence']:.1%}
├─ risk level: {dca_decision['risk_level']}
├─ reason: {' | '.join(dca_decision['technical_reasons'])}

📋 indicator:
├─ RSI(14): {current_data.get('rsi_14', 50):.1f}
├─ trend strength: {current_data.get('trend_strength', 50):.0f}/100
├─ momentum score: {current_data.get('momentum_score', 0):.3f}
├─ ADX: {current_data.get('adx', 25):.1f}
├─ volume: {current_data.get('volume_ratio', 1):.1f}x
├─ Bollinger Bands: {current_data.get('bb_position', 0.5):.2f}
├─ signal strength: {current_data.get('signal_strength', 0):.1f}

💰 DCAcalculate:
├─ translated: ${trade.stake_amount:.2f}
├─ calculate: ${dca_amount:.2f}
├─ new: {(dca_amount/trade.stake_amount)*100:.0f}%
├─ position size: ${trade.stake_amount + dca_amount:.2f}

🌊 translated:
├─ market state: {dca_decision['market_conditions'].get('market_state', 'translated')}
├─ volatility: {'✅translated' if dca_decision['market_conditions'].get('volatility_acceptable', False) else '⚠️high'}
├─ translated: {'✅translated' if dca_decision['market_conditions'].get('liquidity_sufficient', False) else '⚠️not'}
├─ translated: {'✅translated' if dca_decision['market_conditions'].get('spread_reasonable', False) else '⚠️large'}

=================================================="""
            
            logger.info(dca_log)
            
        except Exception as e:
            logger.error(f"DCAday record {trade.pair}: {e}")
    
    def track_dca_performance(self, trade: Trade, dca_type: str, dca_amount: float):
        """translatedDCAtranslated"""
        try:
            # recordDCAtranslated
            self.dca_performance_tracker['total_dca_count'] += 1
            
            dca_record = {
                'trade_id': f"{trade.pair}_{trade.open_date_utc.timestamp()}",
                'pair': trade.pair,
                'dca_type': dca_type,
                'dca_amount': dca_amount,
                'execution_time': datetime.now(timezone.utc),
                'entry_number': trade.nr_of_successful_entries + 1,
                'price_at_dca': trade.open_rate  # will at update
            }
            
            self.dca_performance_tracker['dca_history'].append(dca_record)
            
            # updateDCAtranslatedperformance statistics
            if dca_type not in self.dca_performance_tracker['dca_type_performance']:
                self.dca_performance_tracker['dca_type_performance'][dca_type] = {
                    'count': 0,
                    'successful': 0,
                    'success_rate': 0.0,
                    'avg_profit_contribution': 0.0
                }
            
            self.dca_performance_tracker['dca_type_performance'][dca_type]['count'] += 1
            
        except Exception as e:
            logger.error(f"DCAtranslated: {e}")
    
    def get_dca_performance_report(self) -> dict:
        """getDCAtranslated"""
        try:
            tracker = self.dca_performance_tracker
            
            return {
                'total_dca_executions': tracker['total_dca_count'],
                'overall_success_rate': tracker['dca_success_rate'],
                'type_performance': tracker['dca_type_performance'],
                'avg_profit_contribution': tracker['avg_dca_profit'],
                'recent_dca_count_30d': len([
                    dca for dca in tracker['dca_history'] 
                    if (datetime.now(timezone.utc) - dca['execution_time']).days <= 30
                ]),
                'best_performing_dca_type': max(
                    tracker['dca_type_performance'].items(),
                    key=lambda x: x[1]['success_rate'],
                    default=('none', {'success_rate': 0})
                )[0] if tracker['dca_type_performance'] else 'none'
            }
        except Exception:
            return {'error': 'noneDCAtranslated'}
    
    # translated custom_stoploss - use
    
    # translated _analyze_smart_stoploss_conditions - simplified
    
    # translated _log_smart_stoploss_decision - simplified day
    
    def calculate_smart_takeprofit_levels(self, pair: str, trade: Trade, current_rate: float,
                                        current_profit: float) -> dict:
        """calculate - AItranslated"""
        
        try:
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if dataframe.empty:
                return {'error': 'none'}
            
            current_data = dataframe.iloc[-1]
            current_atr = current_data.get('atr_p', 0.02)
            trend_strength = current_data.get('trend_strength', 50)
            momentum_score = current_data.get('momentum_score', 0)
            current_adx = current_data.get('adx', 25)
            
            # === calculate ===
            base_multiplier = 3.0  # translatedATRtranslated
            
            # trend strength
            if abs(trend_strength) > 80:
                trend_mult = 2.5
            elif abs(trend_strength) > 60:
                trend_mult = 2.0
            else:
                trend_mult = 1.5
            
            # calculate
            total_mult = base_multiplier * trend_mult
            base_distance = current_atr * total_mult
            
            # 4translated
            targets = {
                'level_1': {'target': base_distance * 0.6, 'close': 0.25, 'desc': 'fast'},
                'level_2': {'target': base_distance * 1.0, 'close': 0.35, 'desc': 'translated'},
                'level_3': {'target': base_distance * 1.6, 'close': 0.25, 'desc': 'trend'},
                'level_4': {'target': base_distance * 2.5, 'close': 0.15, 'desc': 'translated'}
            }
            
            # calculate price
            for level_data in targets.values():
                if not trade.is_short:
                    level_data['price'] = trade.open_rate * (1 + level_data['target'])
                else:
                    level_data['price'] = trade.open_rate * (1 - level_data['target'])
                level_data['profit_pct'] = level_data['target'] * 100
            
            return {
                'targets': targets,
                'trend_strength': trend_strength,
                'momentum_score': momentum_score,
                'atr_percent': current_atr * 100,
                'analysis_time': datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"translated {pair}: {e}")
            return {'error': f'translated: {e}'}
    
    # translated get_smart_stoploss_takeprofit_status
    def should_protect_strong_trend(self, pair: str, trade: Trade, 
                                  dataframe: DataFrame, current_rate: float) -> bool:
        """strong trend - trend mid"""
        
        if dataframe.empty:
            return False
            
        try:
            current_data = dataframe.iloc[-1]
            
            # get trend indicator
            trend_strength = current_data.get('trend_strength', 0)
            adx = current_data.get('adx', 0)
            momentum_score = current_data.get('momentum_score', 0)
            
            # check price and
            ema_21 = current_data.get('ema_21', current_rate)
            ema_50 = current_data.get('ema_50', current_rate)
            
            # === bullish trend ===
            if not trade.is_short:
                trend_protection = (
                    trend_strength > 70 and          # trend strength
                    adx > 25 and                     # ADXconfirm trend
                    current_rate > ema_21 and        # price still at up
                    momentum_score > -0.2 and        # momentum has
                    current_rate > ema_50 * 0.98     # price has support
                )
                
            # === bearish trend ===
            else:
                trend_protection = (
                    trend_strength > 70 and          # trend strength
                    adx > 25 and                     # ADXconfirm trend
                    current_rate < ema_21 and        # price still at down
                    momentum_score < 0.2 and         # momentum has  
                    current_rate < ema_50 * 1.02     # price has breakout resistance
                )
            
            return trend_protection
            
        except Exception as e:
            logger.warning(f"trend check: {e}")
            return False
    
    def detect_false_breakout(self, dataframe: DataFrame, current_rate: float, 
                            trade: Trade) -> bool:
        """breakout - at breakout after fast reversal mid"""
        
        if dataframe.empty or len(dataframe) < 10:
            return False
            
        try:
            # get most10translatedKtranslated
            recent_data = dataframe.tail(10)
            current_data = dataframe.iloc[-1]
            
            # get
            supertrend = current_data.get('supertrend', current_rate)
            bb_upper = current_data.get('bb_upper', current_rate * 1.02)
            bb_lower = current_data.get('bb_lower', current_rate * 0.98)
            
            # === bullish breakout ===
            if not trade.is_short:
                # check support after fast
                recent_low = recent_data['low'].min()
                current_recovery = (current_rate - recent_low) / recent_low
                
                # breakout after fast50%as breakout
                if (recent_low < supertrend and 
                    current_rate > supertrend and 
                    current_recovery > 0.005):  # 0.5%translated
                    return True
                    
                # Bollinger Bands breakout
                if (recent_data['low'].min() < bb_lower and 
                    current_rate > bb_lower and
                    current_rate > recent_data['close'].iloc[-3]):  # translated3translatedKbefore high
                    return True
            
            # === bearish breakout ===
            else:
                # check breakout resistance after fast
                recent_high = recent_data['high'].max()
                current_pullback = (recent_high - current_rate) / recent_high
                
                # breakout after fast50%as breakout
                if (recent_high > supertrend and 
                    current_rate < supertrend and 
                    current_pullback > 0.005):  # 0.5%translated
                    return True
                
                # Bollinger Bands breakout
                if (recent_data['high'].max() > bb_upper and 
                    current_rate < bb_upper and
                    current_rate < recent_data['close'].iloc[-3]):  # translated3translatedKbefore low
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"breakout: {e}")
            return False
    
    # translated confirm_stoploss_signal
    
    def _log_trend_protection(self, pair: str, trade: Trade, current_rate: float, 
                            current_profit: float, dataframe: DataFrame):
        """record trend"""
        
        try:
            current_data = dataframe.iloc[-1]
            
            protection_details = {
                'current_rate': current_rate,
                'current_profit': current_profit,
                'trend_strength': current_data.get('trend_strength', 0),
                'adx': current_data.get('adx', 0),
                'momentum_score': current_data.get('momentum_score', 0),
                'trend_protection': True,
                'time_decay': False,
                'profit_protection': False,
                'atr_percent': current_data.get('atr_p', 0),
                'volatility_state': current_data.get('volatility_state', 0),
                'atr_multiplier': 1.0
            }
            
            # calculate new value（before market state）
            suggested_new_stoploss = self.stoploss
            
            # translated decision_logger day record
            pass
            
        except Exception as e:
            logger.warning(f"trend day record: {e}")
    
    def _log_false_breakout_protection(self, pair: str, trade: Trade, 
                                     current_rate: float, dataframe: DataFrame):
        """record breakout"""
        
        try:
            logger.info(f"🚫 breakout - {pair} to breakout，translated50%")
            
        except Exception as e:
            logger.warning(f"breakout day record: {e}")
    
    # ===== new =====
    
    # translated _calculate_structure_based_stop 
    # translated calculate_atr_stop_multiplier - simplified
    
    # translated calculate_trend_stop_adjustment - simplified
    
    # translated calculate_volatility_cluster_stop - simplified
    
    # translated calculate_time_decay_stop - simplified
    
    # translated calculate_profit_protection_stop - simplified
    
    # translated calculate_volume_stop_adjustment - simplified
    
    # translated calculate_microstructure_stop - simplified
    
    # translated apply_stoploss_limits - simplified
    
    # translated get_enhanced_technical_stoploss - simplified
    
    # translated custom_exit translated - useROItranslated
    
    # translated _get_detailed_exit_reason translated - simplified
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, current_time: datetime,
                          entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """confirm"""
        
        try:
            # most check
            
            # 1. time check (avoid large)
            # to time
            
            # 2. check
            orderbook_data = self.get_market_orderbook(pair)
            if orderbook_data['spread_pct'] > 0.3:  # large
                logger.warning(f"large，translated: {pair}")
                return False
            
            # 3. check
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if not dataframe.empty:
                current_atr_p = dataframe['atr_p'].iloc[-1] if 'atr_p' in dataframe.columns else 0.02
                if current_atr_p > 0.06:  # high volatility
                    logger.warning(f"volatility high，translated: {pair}")
                    return False
            
            
            logger.info(f"confirm: {pair} {side} {amount} @ {rate}")
            return True
            
        except Exception as e:
            logger.error(f"confirm: {e}")
            return False
    
    def confirm_trade_exit(self, pair: str, trade: Trade, order_type: str, amount: float,
                          rate: float, time_in_force: str, exit_reason: str,
                          current_time: datetime, **kwargs) -> bool:
        """confirm - update winning streak losing streak"""
        try:
            # calculate
            profit_ratio = trade.calc_profit_ratio(rate)
            
            # update winning streak losing streak
            if profit_ratio > 0:
                self.consecutive_wins += 1
                self.consecutive_losses = 0
                logger.info(f"🏆 {pair} profit，winning streak: {self.consecutive_wins}")
            else:
                self.consecutive_wins = 0
                self.consecutive_losses += 1
                logger.info(f"❌ {pair} loss，losing streak: {self.consecutive_losses}")
                
            # update record
            trade_record = {
                'pair': pair,
                'profit': profit_ratio,
                'exit_reason': exit_reason,
                'timestamp': current_time,
                'entry_rate': trade.open_rate,
                'exit_rate': rate
            }
            
            self.trade_history.append(trade_record)
            
            # record at range
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-500:]
                
        except Exception as e:
            logger.warning(f"update winning streak: {e}")
            
        return True  # allow
    
    def check_entry_timeout(self, pair: str, trade: Trade, order: Dict,
                           current_time: datetime, **kwargs) -> bool:
        """check"""
        return True  # default allow
    
    def check_exit_timeout(self, pair: str, trade: Trade, order: Dict,
                          current_time: datetime, **kwargs) -> bool:
        """check"""  
        return True  # default allow
    
    def leverage(self, pair: str, current_time: datetime, current_rate: float,
                 proposed_leverage: float, max_leverage: float, entry_tag: Optional[str],
                 side: str, **kwargs) -> float:
        """🧠 leverage - signal market state"""
        
        try:
            # get
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if dataframe.empty:
                logger.warning(f"leverage calculate，none {pair}")
                return min(2.0, max_leverage)
            
            # get
            current_data = dataframe.iloc[-1]
            volatility = current_data.get('atr_p', 0.02)
            
            # === 1. get market state analysis ===
            market_regime_data = self._enhanced_market_regime_detection(dataframe)
            regime = market_regime_data['regime']
            regime_confidence = market_regime_data['confidence']
            signals_advice = market_regime_data['signals_advice']
            
            # === 2. signal ===
            signal_quality_bonus = self._calculate_signal_quality_leverage_bonus(
                entry_tag, current_data, regime, signals_advice
            )
            
            # === 3. leverage calculate ===
            base_leverage = self.calculate_leverage('sideways', volatility, pair, current_time)
            
            # === 4. market state ===
            regime_multiplier = self._get_regime_leverage_multiplier(regime, regime_confidence)
            
            # === 5. signal ===
            signal_multiplier = self._get_signal_leverage_multiplier(entry_tag, signals_advice)
            
            # === 6. calculate ===
            calculated_leverage = (
                base_leverage * 
                regime_multiplier * 
                signal_multiplier * 
                signal_quality_bonus
            )
            
            # === 7. translated ===
            # not
            safe_leverage = min(calculated_leverage, max_leverage)
            
            # translated
            if volatility > 0.08:  # 8%to up，low leverage
                safe_leverage = min(safe_leverage, 5)
            elif volatility > 0.05:  # 5%to up，leverage
                safe_leverage = min(safe_leverage, 15)
            
            # market state
            if 'VOLATILE' in regime or regime_confidence < 0.3:
                safe_leverage = min(safe_leverage, 10)
            
            final_leverage = max(1.0, safe_leverage)  # most low1leverage
            
            # === 8. day ===
            logger.info(
                f"🎯 leverage {pair} [{entry_tag}]: "
                f"translated{base_leverage:.1f}x × "
                f"translated{regime_multiplier:.2f} × "
                f"signal{signal_multiplier:.2f} × " 
                f"translated{signal_quality_bonus:.2f} = "
                f"{calculated_leverage:.1f}x → {final_leverage:.1f}x | "
                f"translated:{regime} ({regime_confidence:.1%})"
            )
            
            return final_leverage
            
        except Exception as e:
            logger.error(f"leverage calculate {pair}: {e}")
            return min(2.0, max_leverage)  # leverage
    
    def leverage_update_callback(self, trade: Trade, **kwargs):
        """leverage update"""
        # count at mid，leverage
        pass
    
    def update_trade_results(self, trade: Trade, profit: float, exit_reason: str):
        """update"""
        try:
            # update
            trade_record = {
                'pair': trade.pair,
                'profit': profit,
                'exit_reason': exit_reason,
                'hold_time': (trade.close_date_utc - trade.open_date_utc).total_seconds() / 3600,
                'timestamp': trade.close_date_utc
            }
            
            self.trade_history.append(trade_record)
            
            # record at range
            if len(self.trade_history) > 1000:
                self.trade_history = self.trade_history[-500:]
            
            # winning streak losing streak at confirm_trade_exit mid update
            
            # translated
            trade_id = f"{trade.pair}_{trade.open_date_utc.timestamp()}"
            if trade_id in self.profit_taking_tracker:
                del self.profit_taking_tracker[trade_id]
                
        except Exception as e:
            logger.error(f"update: {e}")
    
    # translated get_intelligent_exit_signal - not use
    
    # translated calculate_emergency_stoploss_triggers - simplified
