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
from NewsSentimentEngine import NewsSentimentEngine

logger = logging.getLogger(__name__)

# Removed the StrategyDecisionLogger class - simplified the logging system

class TradingStyleManager:
    """Trading style manager - automatically switches between stable/sideways/aggressive modes based on market state"""
    
    def __init__(self):
        self.current_style = "stable"  # default stable mode
        self.style_switch_cooldown = 0
        self.min_switch_interval = 0.5  # minimum 30 minutes between switches (improves responsiveness)
        
        # === Stable Mode Configuration ===
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
        
        # === Sideways Mode Configuration ===
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
        
        # === Aggressive Mode Configuration ===
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
    news_sentiment_enabled = True
    news_prefetch_interval_seconds = 300
    
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
        self.last_news_prefetch = None
        self._news_signal_cache = {}
        self._init_news_sentiment_engine()

    def _init_news_sentiment_engine(self) -> None:
        news_config = self.config.get('news_sentiment', {}) if hasattr(self, 'config') else {}
        self.news_sentiment_enabled = bool(
            news_config.get('enabled', self.news_sentiment_enabled)
        )
        if not self.news_sentiment_enabled:
            self.news_sentiment_engine = None
            logger.info("News sentiment integration disabled.")
            return

        api_key = str(news_config.get('api_key') or "").strip()
        backend = str(news_config.get('backend', 'hybrid')).strip()
        source_weights = news_config.get('source_weights')
        trusted_positive_sources = news_config.get('trusted_positive_sources')
        self.news_prefetch_interval_seconds = max(
            60,
            int(
                news_config.get(
                    'prefetch_interval_seconds',
                    self.news_prefetch_interval_seconds,
                )
            ),
        )

        self.news_sentiment_engine = NewsSentimentEngine(
            api_key=api_key,
            logger=logger,
            backend=backend,
            roberta_model=str(
                news_config.get(
                    'roberta_model',
                    "cardiffnlp/twitter-roberta-base-sentiment-latest",
                )
            ),
            lookback_hours=int(news_config.get('lookback_hours', 2)),
            cache_minutes=int(news_config.get('cache_minutes', 15)),
            page_size=int(news_config.get('page_size', 10)),
            min_articles_to_block=int(news_config.get('min_articles_to_block', 2)),
            min_articles_to_reduce=int(news_config.get('min_articles_to_reduce', 1)),
            block_abs_sentiment=float(news_config.get('block_abs_sentiment', 0.30)),
            reduce_abs_sentiment=float(news_config.get('reduce_abs_sentiment', 0.18)),
            min_impact_to_block=float(news_config.get('min_impact_to_block', 0.25)),
            min_stake_multiplier=float(news_config.get('min_stake_multiplier', 0.40)),
            max_stake_multiplier=float(news_config.get('max_stake_multiplier', 1.10)),
            stake_reduction_scale=float(news_config.get('stake_reduction_scale', 0.50)),
            positive_boost_threshold=float(news_config.get('positive_boost_threshold', 0.45)),
            min_confidence_to_act=float(news_config.get('min_confidence_to_act', 0.35)),
            max_fallback_minutes=int(news_config.get('max_fallback_minutes', 60)),
            request_timeout=int(news_config.get('request_timeout', 10)),
            language=str(news_config.get('language', 'en')),
            source_weights=source_weights if isinstance(source_weights, dict) else None,
            trusted_positive_sources=(
                trusted_positive_sources
                if isinstance(trusted_positive_sources, list)
                else None
            ),
        )

        status = self.news_sentiment_engine.backend_status()
        logger.info(
            "News sentiment engine initialized: enabled=%s backend=%s ready=%s",
            self.news_sentiment_enabled,
            status['backend'],
            status['ready'],
        )

    def _extract_base_asset(self, pair: str) -> str:
        base = str(pair or "").split("/")[0]
        return base.split(":")[0].upper()

    def _default_news_asset_names(self) -> dict[str, list[str]]:
        return {
            'BTC': ['Bitcoin'],
            'ETH': ['Ethereum', 'Ether'],
            'BNB': ['BNB', 'Binance Coin'],
            'SOL': ['Solana'],
            'XRP': ['XRP', 'Ripple'],
            'ADA': ['Cardano'],
            'DOGE': ['Dogecoin'],
            'AVAX': ['Avalanche'],
            'DOT': ['Polkadot'],
            'LINK': ['Chainlink'],
            'MATIC': ['Polygon', 'POL'],
            'POL': ['Polygon', 'POL'],
            'ATOM': ['Cosmos'],
            'NEAR': ['Near Protocol'],
            'ONE': ['Harmony'],
            'UNI': ['Uniswap'],
            'AAVE': ['Aave'],
            'SUI': ['Sui'],
            'APT': ['Aptos'],
            'ARB': ['Arbitrum'],
            'OP': ['Optimism'],
            'INJ': ['Injective'],
            'TIA': ['Celestia'],
            'SEI': ['Sei'],
            'TAO': ['Bittensor'],
            'PEPE': ['Pepe'],
            'SHIB': ['Shiba Inu'],
            'WIF': ['dogwifhat'],
            'BONK': ['Bonk'],
            'FET': ['Fetch.ai', 'Artificial Superintelligence Alliance', 'ASI'],
            'RENDER': ['Render'],
            'RNDR': ['Render'],
            'TRX': ['Tron'],
            'ETC': ['Ethereum Classic'],
            'BCH': ['Bitcoin Cash'],
            'LTC': ['Litecoin'],
            'XLM': ['Stellar'],
            'HBAR': ['Hedera'],
            'FIL': ['Filecoin'],
        }

    def _dedupe_news_aliases(self, aliases: list[str]) -> list[str]:
        deduped = []
        seen = set()
        for alias in aliases:
            cleaned = str(alias).strip()
            if not cleaned:
                continue
            key = cleaned.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(cleaned)
        return deduped

    def _expand_news_aliases(self, base_asset: str, aliases: list[str]) -> list[str]:
        expanded = list(aliases)
        ambiguity_sensitive = {
            'ONE', 'NEAR', 'LINK', 'ATOM', 'OP', 'TIA', 'TAO', 'SEI', 'SUI', 'APT',
            'INJ', 'ARB', 'UNI', 'FIL', 'ETC', 'BCH', 'TRX', 'XLM', 'HBAR', 'FET',
        }

        for alias in list(aliases):
            alias_clean = str(alias).strip()
            if not alias_clean:
                continue
            expanded.append(f"{alias_clean} crypto")
            expanded.append(f"{alias_clean} token")
            if len(alias_clean) > 3:
                expanded.append(f"{alias_clean} blockchain")

        if base_asset:
            expanded.append(base_asset)
            expanded.append(f"{base_asset} crypto")
            expanded.append(f"{base_asset} token")
            if base_asset in ambiguity_sensitive:
                expanded.append(f"{base_asset} cryptocurrency")

        return self._dedupe_news_aliases(expanded)

    def _get_news_aliases(self, pair: str) -> list[str]:
        news_config = self.config.get('news_sentiment', {}) if hasattr(self, 'config') else {}
        alias_map = news_config.get('aliases', {})
        asset_name_map = news_config.get('asset_names', {})
        if not isinstance(alias_map, dict):
            alias_map = {}
        if not isinstance(asset_name_map, dict):
            asset_name_map = {}

        base_asset = self._extract_base_asset(pair)
        aliases = []
        for key in (pair, base_asset):
            value = alias_map.get(key)
            if isinstance(value, list):
                aliases.extend(str(alias).strip() for alias in value if str(alias).strip())

        configured_asset_names = asset_name_map.get(base_asset)
        if isinstance(configured_asset_names, list):
            aliases.extend(str(alias).strip() for alias in configured_asset_names if str(alias).strip())
        elif isinstance(configured_asset_names, str) and configured_asset_names.strip():
            aliases.append(configured_asset_names.strip())

        aliases.extend(self._default_news_asset_names().get(base_asset, []))

        auto_expand = bool(news_config.get('auto_expand_aliases', True))
        aliases = self._dedupe_news_aliases(aliases)
        if auto_expand:
            aliases = self._expand_news_aliases(base_asset, aliases)
        elif base_asset and base_asset not in aliases:
            aliases.append(base_asset)

        return aliases

    def _get_news_signal(self, pair: str, current_time: datetime):
        if not getattr(self, 'news_sentiment_engine', None):
            return None

        base_asset = self._extract_base_asset(pair)
        aliases = self._get_news_aliases(pair)
        signal = self.news_sentiment_engine.get_signal(
            asset=base_asset,
            current_time=current_time,
            aliases=aliases,
        )
        self._news_signal_cache[base_asset] = signal
        return signal

    def bot_loop_start(self, current_time: datetime, **kwargs) -> None:
        if not getattr(self, 'news_sentiment_engine', None):
            return

        current_time = (
            current_time.astimezone(timezone.utc)
            if current_time.tzinfo
            else current_time.replace(tzinfo=timezone.utc)
        )
        if (
            self.last_news_prefetch
            and (current_time - self.last_news_prefetch).total_seconds()
            < self.news_prefetch_interval_seconds
        ):
            return

        pairs = []
        try:
            if hasattr(self, 'dp') and self.dp:
                pairs = list(self.dp.current_whitelist())
        except Exception:
            pairs = []

        if not pairs:
            exchange_config = self.config.get('exchange', {}) if hasattr(self, 'config') else {}
            pairs = list(exchange_config.get('pair_whitelist', []))

        for pair in pairs:
            try:
                self._get_news_signal(pair, current_time)
            except Exception as exc:
                logger.warning("News prefetch failed for %s: %s", pair, exc)

        self.last_news_prefetch = current_time
        
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
        """Check whether the trading style should switch."""
        
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
        """Record detailed information about a trading-style switch."""
        
        try:
            current_data = dataframe.iloc[-1] if not dataframe.empty else {}
            
            switch_log = f"""
==================== Trading Style Switch ====================
Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')}
Reason: {reason}

📊 Market regime analysis:
├─ Trend strength: {current_data.get('trend_strength', 0):.0f}/100
├─ ADX value: {current_data.get('adx', 0):.1f}
├─ Volatility state: {current_data.get('volatility_state', 0):.0f}/100
├─ ATR volatility: {(current_data.get('atr_p', 0) * 100):.2f}%

🔄 Style change details:
├─ Old style: {old_config['name']} → New style: {new_config['name']}
├─ Leverage adjustment: {old_config['leverage_range']} → {new_config['leverage_range']}
├─ Position adjustment: {[f"{p*100:.0f}%" for p in old_config['position_range']]} → {[f"{p*100:.0f}%" for p in new_config['position_range']]}
├─ Risk adjustment: {old_config['risk_per_trade']*100:.1f}% → {new_config['risk_per_trade']*100:.1f}%

🎯 New style settings:
├─ Description: {new_config['description']}
├─ Entry threshold: {new_config['entry_threshold']:.1f}
├─ Max concurrent trades: {new_config['max_trades']}
├─ Cooldown period: {self.style_manager.style_switch_cooldown} hours

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
        """Get orderbook data and derive market microstructure metrics."""
        try:
            orderbook = self.dp.orderbook(pair, 10)  # fetch top 10 bid/ask levels
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
                
                # calculate buy/sell pressure metrics (0-1 range)
                buy_pressure = bid_volume / (bid_volume + ask_volume + 1e-10)
                sell_pressure = ask_volume / (bid_volume + ask_volume + 1e-10)
                
                # calculate market quality (0-1 range)
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
        """Calculate the core technical indicator set with batched dataframe writes."""
        
        # use a dictionary to batch-store all new columns
        new_columns = {}
        
        # === Optimized responsive moving-average system ===
        new_columns['ema_5'] = ta.EMA(dataframe, timeperiod=5)    # ultra-short-term: capture rapid changes
        new_columns['ema_8'] = ta.EMA(dataframe, timeperiod=8)    # ultra-short-term enhancement
        new_columns['ema_13'] = ta.EMA(dataframe, timeperiod=13)  # short-term: trend confirmation
        new_columns['ema_21'] = ta.EMA(dataframe, timeperiod=21)  # medium-short-term transition
        new_columns['ema_34'] = ta.EMA(dataframe, timeperiod=34)  # medium-term main trend filter
        new_columns['ema_50'] = ta.EMA(dataframe, timeperiod=50)  # long-term trend
        new_columns['sma_20'] = ta.SMA(dataframe, timeperiod=20)  # baseline SMA reference
        
        # === Bollinger Bands ===
        bb = qtpylib.bollinger_bands(dataframe['close'], window=self.bb_period, stds=self.bb_std)
        new_columns['bb_lower'] = bb['lower']
        new_columns['bb_middle'] = bb['mid']
        new_columns['bb_upper'] = bb['upper']
        new_columns['bb_width'] = np.where(bb['mid'] > 0, 
                                        (bb['upper'] - bb['lower']) / bb['mid'], 
                                        0)
        new_columns['bb_position'] = (dataframe['close'] - bb['lower']) / (bb['upper'] - bb['lower'])
        
        # === RSI (standard 14-period) ===
        new_columns['rsi_14'] = ta.RSI(dataframe, timeperiod=14)
        
        # === MACD (classic trend indicator) ===
        macd = ta.MACD(dataframe, fastperiod=self.macd_fast, slowperiod=self.macd_slow, signalperiod=self.macd_signal)
        new_columns['macd'] = macd['macd']
        new_columns['macd_signal'] = macd['macdsignal'] 
        new_columns['macd_hist'] = macd['macdhist']
        
        # === ADX trend strength ===
        new_columns['adx'] = ta.ADX(dataframe, timeperiod=self.adx_period)
        new_columns['plus_di'] = ta.PLUS_DI(dataframe, timeperiod=self.adx_period)
        new_columns['minus_di'] = ta.MINUS_DI(dataframe, timeperiod=self.adx_period)
        
        # === ATR volatility (used in risk management) ===
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
        
        # 1. Fast RSI variant for earlier turning-point signals
        stoch_rsi = ta.STOCHRSI(dataframe, timeperiod=14, fastk_period=3, fastd_period=3)
        new_columns['stoch_rsi_k'] = stoch_rsi['fastk']
        new_columns['stoch_rsi_d'] = stoch_rsi['fastd']
        
        # 2. Williams indicator - fast reversal signal
        new_columns['williams_r'] = ta.WILLR(dataframe, timeperiod=14)
        
        # 3. CCI - responsive overbought/oversold indicator
        new_columns['cci'] = ta.CCI(dataframe, timeperiod=20)
        
        # 4. Price action analysis
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
        ) * 100  # distance from the 20-period high
        
        new_columns['support_strength'] = (
            1 - dataframe['close'] / dataframe['low'].rolling(20).min()
        ) * 100  # distance from the 20-period low
        
        # === VWAP (important institutional trading reference) ===
        new_columns['vwap'] = qtpylib.rolling_vwap(dataframe)
        
        # === supertrend (efficient trend following) ===
        new_columns['supertrend'] = self.supertrend(dataframe, 10, 3)
        
        # Add new columns in one batch to avoid repeated dataframe mutations
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
            # Fill critical indicators with safe defaults when calculation fails
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
                    # Recalculate missing EMA indicators directly
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
        
        # === EMA integrity check ===
        # Rebuild EMA columns if they contain too many missing values
        for ema_col in ['ema_8', 'ema_21', 'ema_50']:
            if ema_col in dataframe.columns:
                nan_count = dataframe[ema_col].isnull().sum()
                total_count = len(dataframe)
                if nan_count > total_count * 0.1:  # more than 10% missing values
                    logger.warning(f"{ema_col} has too many missing values ({nan_count}/{total_count}), recalculate")
                    if ema_col == 'ema_8':
                        dataframe[ema_col] = ta.EMA(dataframe, timeperiod=8)
                    elif ema_col == 'ema_21':
                        dataframe[ema_col] = ta.EMA(dataframe, timeperiod=21)
                    elif ema_col == 'ema_50':
                        dataframe[ema_col] = ta.EMA(dataframe, timeperiod=50)
        
        return dataframe
    
    def calculate_optimized_composite_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate composite indicators while minimizing dataframe churn."""
        
        # use a dictionary to batch-store all new columns
        new_columns = {}
        
        # === Composite trend-strength scoring system ===
        
        # 1. Price-momentum slope analysis (early warning) using EMA(5, 13, 34)
        ema5_slope = np.where(dataframe['ema_5'].shift(2) > 0,
                             (dataframe['ema_5'] - dataframe['ema_5'].shift(2)) / dataframe['ema_5'].shift(2),
                             0) * 100  # short-term, fast response
        ema13_slope = np.where(dataframe['ema_13'].shift(3) > 0,
                              (dataframe['ema_13'] - dataframe['ema_13'].shift(3)) / dataframe['ema_13'].shift(3),
                              0) * 100
        
        # 2. Moving-average divergence analysis (trend acceleration signal)
        ema_spread = np.where(dataframe['ema_34'] > 0,
                             (dataframe['ema_5'] - dataframe['ema_34']) / dataframe['ema_34'] * 100,
                             0)
        ema_spread_series = self._safe_series(ema_spread, len(dataframe))
        ema_spread_change = ema_spread - ema_spread_series.shift(3)  # divergence change
        
        # 3. ADX slope and acceleration (trend strengthening signal)
        adx_slope = dataframe['adx'] - dataframe['adx'].shift(3)  # ADX slope
        adx_acceleration = adx_slope - adx_slope.shift(2)  # ADX acceleration
        
        # 4. volume trend confirmation
        volume_20_mean = dataframe['volume'].rolling(20).mean()
        volume_trend = np.where(volume_20_mean != 0,
                               dataframe['volume'].rolling(5).mean() / volume_20_mean,
                               1.0)  # default to neutral if the 20-period mean is zero
        volume_trend_series = self._safe_series(volume_trend, len(dataframe))
        volume_momentum = volume_trend_series - volume_trend_series.shift(2).fillna(0)
        
        # 5. Price acceleration (second derivative)
        close_shift_3 = dataframe['close'].shift(3)
        price_velocity = np.where(close_shift_3 != 0,
                                 (dataframe['close'] / close_shift_3 - 1) * 100,
                                 0)  # first derivative
        price_velocity_series = self._safe_series(price_velocity, len(dataframe))
        price_acceleration = price_velocity_series - price_velocity_series.shift(2).fillna(0)
        
        # === Composite trend-strength score ===
        trend_score = (
            ema5_slope * 0.30 +        # ultra-short-term momentum with the highest weight
            ema13_slope * 0.20 +       # short-term momentum confirmation
            ema_spread_change * 0.15 + # trend-divergence change
            adx_slope * 0.15 +         # trend-strength change
            volume_momentum * 0.10 +   # volume support
            price_acceleration * 0.10  # price acceleration
        )
        
        # Use ADX as a trend-confirmation multiplier
        adx_multiplier = np.where(dataframe['adx'] > 30, 1.5,
                                 np.where(dataframe['adx'] > 20, 1.2,
                                         np.where(dataframe['adx'] > 15, 1.0, 0.7)))
        
        # final trend strength
        new_columns['trend_strength'] = (trend_score * adx_multiplier).clip(-100, 100)
        new_columns['price_acceleration'] = price_acceleration
        
        # === Momentum composite indicators ===
        rsi_normalized = (dataframe['rsi_14'] - 50) / 50  # -1 to 1
        macd_normalized = np.where(dataframe['atr_p'] > 0, 
                                 dataframe['macd_hist'] / (dataframe['atr_p'] * dataframe['close']), 
                                 0)  # volatility-normalized MACD histogram
        price_momentum = (dataframe['close'] / dataframe['close'].shift(5) - 1) * 10  # 5-period price change
        
        new_columns['momentum_score'] = (rsi_normalized + macd_normalized + price_momentum) / 3
        new_columns['price_velocity'] = price_velocity_series
        
        # === Volatility-state indicator ===
        atr_percentile = dataframe['atr_p'].rolling(50).rank(pct=True)
        bb_squeeze = np.where(dataframe['bb_width'] < dataframe['bb_width'].rolling(20).quantile(0.3), 1, 0)
        volume_spike = np.where(dataframe['volume_ratio'] > 1.5, 1, 0)
        
        new_columns['volatility_state'] = atr_percentile * 50 + bb_squeeze * 25 + volume_spike * 25
        
        # === Support/resistance strength ===
        bb_position_score = np.abs(dataframe['bb_position'] - 0.5) * 2  # 0-1, the closer to the edge, the higher the score
        vwap_distance = np.where(dataframe['vwap'] > 0, 
                                np.abs((dataframe['close'] - dataframe['vwap']) / dataframe['vwap']) * 100, 
                                0)
        
        new_columns['sr_strength'] = (bb_position_score + np.minimum(vwap_distance, 5)) / 2  # normalize to a reasonable range
        
        # === Trend sustainability indicator ===
        adx_sustainability = np.where(dataframe['adx'] > 25, 1, 0)
        volume_sustainability = np.where(dataframe['volume_ratio'] > 0.8, 1, 0)
        volatility_sustainability = np.where(dataframe['atr_p'] < dataframe['atr_p'].rolling(20).quantile(0.8), 1, 0)
        new_columns['trend_sustainability'] = (
            (adx_sustainability * 0.5 + volume_sustainability * 0.3 + volatility_sustainability * 0.2) * 2 - 1
        ).clip(-1, 1)  # normalize to [-1, 1]
        
        # === RSI divergence strength indicator ===
        price_high_10 = dataframe['high'].rolling(10).max()
        price_low_10 = dataframe['low'].rolling(10).min()
        rsi_high_10 = dataframe['rsi_14'].rolling(10).max()
        rsi_low_10 = dataframe['rsi_14'].rolling(10).min()
        
        # Bearish divergence: price makes a new high while RSI does not
        bearish_divergence = np.where(
            (dataframe['high'] >= price_high_10) & (dataframe['rsi_14'] < rsi_high_10),
            -(dataframe['high'] / price_high_10 - dataframe['rsi_14'] / rsi_high_10),
            0
        )
        
        # Bullish divergence: price makes a new low while RSI does not
        bullish_divergence = np.where(
            (dataframe['low'] <= price_low_10) & (dataframe['rsi_14'] > rsi_low_10),
            (dataframe['low'] / price_low_10 - dataframe['rsi_14'] / rsi_low_10),
            0
        )
        
        new_columns['rsi_divergence_strength'] = (bearish_divergence + bullish_divergence).clip(-2, 2)
        
        # === Predictive indicator system ===
        
        # 1. Simple RSI divergence flags
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
        
        # 3. Price-acceleration change (used to predict turning points)
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
        
        # 5. Trend phase detection
        # early stage: breakout with strengthening participation
        trend_early = (
            (dataframe['adx'] > dataframe['adx'].shift(1)) &
            (dataframe['adx'] > 20) &
            (dataframe['volume_ratio'] > 1.2)
        ).astype(int)
        # mid stage: stable trend
        trend_middle = (
            (dataframe['adx'] > 25) &
            (np.abs(price_acceleration_new) < 0.02) &
            (~volume_decreasing)
        ).astype(int)
        # late stage: exhaustion and divergence
        trend_late = (
            (np.abs(price_acceleration_new) > 0.03) |
            (new_columns['bearish_divergence'] == 1) |
            (new_columns['bullish_divergence'] == 1) |
            (momentum_exhaustion > 0.6)
        ).astype(int)
        
        new_columns['trend_phase'] = trend_late * 3 + trend_middle * 2 + trend_early * 1
        
        # === market sentiment indicator ===
        rsi_sentiment = (dataframe['rsi_14'] - 50) / 50  # normalized RSI
        volatility_sentiment = np.where(dataframe['atr_p'] > 0, 
                                       -(dataframe['atr_p'] / dataframe['atr_p'].rolling(20).mean() - 1), 
                                       0)  # high volatility is risk-off, low volatility is supportive
        volume_sentiment = np.where(dataframe['volume_ratio'] > 1.5, -0.5,  # volume spike can indicate instability
                                   np.where(dataframe['volume_ratio'] < 0.7, 0.5, 0))  # lighter volume is treated as calmer
        new_columns['market_sentiment'] = ((rsi_sentiment + volatility_sentiment + volume_sentiment) / 3).clip(-1, 1)
        
        # === Four-stage reversal warning system ===
        reversal_warnings = self.detect_reversal_warnings_system(dataframe)
        new_columns['reversal_warning_level'] = reversal_warnings['level']
        new_columns['reversal_probability'] = reversal_warnings['probability']
        new_columns['reversal_signal_strength'] = reversal_warnings['signal_strength']
        
        # Add new columns in one batch to avoid repeated dataframe mutations
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
        """Detect a four-stage reversal warning profile 2-5 candles before a trend reversal."""
        
        # === Level 1 warning: momentum decay detection ===
        # Detect whether trend momentum has started to fade (earliest warning)
        momentum_decay_long = (
            # price gains are shrinking
            (dataframe['close'] - dataframe['close'].shift(3) < 
             dataframe['close'].shift(3) - dataframe['close'].shift(6)) &
            # but price is still rising
            (dataframe['close'] > dataframe['close'].shift(3)) &
            # ADX is falling
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
            # ADX is falling
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
        
        # === Level 3 warning: abnormal volume distribution (capital rotation) ===
        # heavy selling appears within a bullish trend
        distribution_volume = (
            (dataframe['close'] > dataframe['ema_13']) &  # still in an uptrend
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5) &  # abnormal volume spike
            (dataframe['close'] < dataframe['open']) &  # but closes bearish
            (dataframe['close'] < (dataframe['high'] + dataframe['low']) / 2)  # closes in the lower half of the candle
        )
        
        # heavy buying appears within a bearish trend
        accumulation_volume = (
            (dataframe['close'] < dataframe['ema_13']) &  # still in a downtrend
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.5) &  # abnormal volume spike
            (dataframe['close'] > dataframe['open']) &  # but closes bullish
            (dataframe['close'] > (dataframe['high'] + dataframe['low']) / 2)  # closes in the upper half of the candle
        )
        
        # === Level 4 warning: trend compression and volatility squeeze ===
        # Moving averages begin to converge (the trend may be ending)
        ema_convergence = (
            abs(dataframe['ema_5'] - dataframe['ema_13']) < dataframe['atr'] * 0.8
        )
        
        # Abnormal volatility compression (the calm before the storm)
        volatility_squeeze = (
            dataframe['atr_p'] < dataframe['atr_p'].rolling(20).quantile(0.3)
        ) & (
            dataframe['bb_width'] < dataframe['bb_width'].rolling(20).quantile(0.2)
        )
        
        # === Calculate the composite warning level ===
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
        
        # Warning levels range from 1-4; higher levels imply higher reversal probability
        warning_level = np.maximum(bullish_reversal_signals, bearish_reversal_signals)
        
        # === Reversal-probability calculation ===
        # Probability model based on historical observations
        reversal_probability = np.where(
            warning_level >= 3, 0.75,  # level 3-4 warning: 75% probability
            np.where(warning_level == 2, 0.55,  # level 2 warning: 55% probability
                    np.where(warning_level == 1, 0.35, 0.1))  # level 1 warning: 35% probability
        )
        
        # === Signal-strength score ===
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
        """Validate whether a breakout is likely to be genuine or false."""
        
        # === 1. Volume breakout confirmation ===
        # breakouts must be accompanied by expanding volume
        volume_breakout_score = np.where(
            dataframe['volume_ratio'] > 2.0, 3,  # abnormal volume spike: score 3
            np.where(dataframe['volume_ratio'] > 1.5, 2,  # significant volume expansion: score 2
                    np.where(dataframe['volume_ratio'] > 1.2, 1, 0))  # moderate volume expansion: score 1, otherwise 0
        )
        
        # === 2. Price-strength validation ===
        # score breakout magnitude and strength
        atr_current = dataframe['atr']
        
        # upward breakout strength
        upward_strength = np.where(
            # break above the upper Bollinger Band by more than one ATR
            (dataframe['close'] > dataframe['bb_upper']) & 
            ((dataframe['close'] - dataframe['bb_upper']) > atr_current), 3,
            np.where(
                # break above the upper Bollinger Band but by less than one ATR
                dataframe['close'] > dataframe['bb_upper'], 2,
                np.where(
                    # break above the Bollinger middle band
                    dataframe['close'] > dataframe['bb_middle'], 1, 0
                )
            )
        )
        
        # downward breakout strength  
        downward_strength = np.where(
            # break below the lower Bollinger Band by more than one ATR
            (dataframe['close'] < dataframe['bb_lower']) & 
            ((dataframe['bb_lower'] - dataframe['close']) > atr_current), -3,
            np.where(
                # break below the lower Bollinger Band but by less than one ATR
                dataframe['close'] < dataframe['bb_lower'], -2,
                np.where(
                    # below the Bollinger middle band
                    dataframe['close'] < dataframe['bb_middle'], -1, 0
                )
            )
        )
        
        price_strength = upward_strength + downward_strength  # combined score
        
        # === 3. Time-persistence validation ===
        # Confirm follow-through after the breakout in the next 1-3 candles
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
        
        # fake breakout with an overly long upper wick (pushes up and then fades)
        long_upper_shadow = (
            (dataframe['high'] - dataframe['close']) > (dataframe['close'] - dataframe['open']) * 2
        ) & (dataframe['close'] > dataframe['open'])  # bullish candle but upper wick is too long
        false_breakout_penalty -= long_upper_shadow.astype(int) * 2
        
        # fake breakout with an overly long lower wick (dips and then rebounds)
        long_lower_shadow = (
            (dataframe['close'] - dataframe['low']) > (dataframe['open'] - dataframe['close']) * 2
        ) & (dataframe['close'] < dataframe['open'])  # bearish candle but lower wick is too long
        false_breakout_penalty -= long_lower_shadow.astype(int) * 2
        
        # === 5. Technical-indicator confirmation ===
        # Confirm with RSI and MACD alignment
        technical_confirmation = self._safe_series(0, len(dataframe))
        
        # bullish breakout confirmation
        bullish_tech_confirm = (
            (dataframe['rsi_14'] > 50) &  # RSI supports the breakout
            (dataframe['macd_hist'] > 0) &  # MACD histogram is positive
            (dataframe['trend_strength'] > 0)  # trend strength is positive
        ).astype(int) * 2
        
        # bearish breakout confirmation
        bearish_tech_confirm = (
            (dataframe['rsi_14'] < 50) &  # RSI supports the breakout
            (dataframe['macd_hist'] < 0) &  # MACD histogram is negative
            (dataframe['trend_strength'] < 0)  # trend strength is negative
        ).astype(int) * -2
        
        technical_confirmation = bullish_tech_confirm + bearish_tech_confirm
        
        # === 6. Composite scoring ===
        # Weight allocation
        validity_score = (
            volume_breakout_score * 0.30 +      # volume confirmation: 30%
            price_strength * 0.25 +             # price strength: 25%
            breakout_persistence * 0.20 +       # follow-through persistence: 20%
            technical_confirmation * 0.15 +     # technical confirmation: 15%
            false_breakout_penalty * 0.10       # false-breakout penalty: 10%
        ).clip(-10, 10)
        
        # === 7. Confidence calculation ===
        # Map the score to a breakout-confidence estimate
        confidence = np.where(
            abs(validity_score) >= 6, 0.85,  # high confidence: 85%
            np.where(abs(validity_score) >= 4, 0.70,  # medium confidence: 70%
                    np.where(abs(validity_score) >= 2, 0.55,  # low confidence: 55%
                            0.30))  # very low confidence: 30%
        )
        
        # === 8. Breakout classification ===
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
        """Simplified market-regime detection with low-overhead dataframe writes."""
        
        # Collect new columns first to avoid repeated dataframe mutations
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
        
        # Assign the new values directly once they are prepared
        if new_columns:
            for col_name, value in new_columns.items():
                if isinstance(value, pd.Series):
                    # Align the series length with the dataframe when possible
                    if len(value) == len(dataframe):
                        dataframe[col_name] = value.values
                    else:
                        dataframe[col_name] = value
                else:
                    dataframe[col_name] = value
        
        return dataframe
    
    def ichimoku(self, dataframe: DataFrame, tenkan=9, kijun=26, senkou_b=52) -> DataFrame:
        """Calculate Ichimoku cloud indicators with batched writes."""
        # Batch-calculate all indicators
        new_columns = {}
        
        new_columns['tenkan'] = (dataframe['high'].rolling(tenkan).max() + dataframe['low'].rolling(tenkan).min()) / 2
        new_columns['kijun'] = (dataframe['high'].rolling(kijun).max() + dataframe['low'].rolling(kijun).min()) / 2
        new_columns['senkou_a'] = ((new_columns['tenkan'] + new_columns['kijun']) / 2).shift(kijun)
        new_columns['senkou_b'] = ((dataframe['high'].rolling(senkou_b).max() + dataframe['low'].rolling(senkou_b).min()) / 2).shift(kijun)
        new_columns['chikou'] = dataframe['close'].shift(-kijun)
        
        # Add the generated columns in one batch
        if new_columns:
            for col_name, value in new_columns.items():
                if isinstance(value, pd.Series):
                    # Align the series length with the dataframe when possible
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
        """Calculate advanced volatility indicators."""
        
        # Keltner Channel (ATR-based envelope)
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
        
        # Donchian Channel (breakout range)
        dc_period = 20
        dataframe['dc_upper'] = dataframe['high'].rolling(dc_period).max()
        dataframe['dc_lower'] = dataframe['low'].rolling(dc_period).min()
        dataframe['dc_middle'] = (dataframe['dc_upper'] + dataframe['dc_lower']) / 2
        dataframe['dc_width'] = np.where(dataframe['dc_middle'] > 0, 
                                        (dataframe['dc_upper'] - dataframe['dc_lower']) / dataframe['dc_middle'], 
                                        0)
        
        # Bollinger Bandwidth (volatility compression/expansion)
        dataframe['bb_bandwidth'] = dataframe['bb_width']  # already calculated in the core indicator block
        dataframe['bb_squeeze'] = (dataframe['bb_bandwidth'] < dataframe['bb_bandwidth'].rolling(20).quantile(0.2)).astype(int)
        
        # Chaikin Volatility
        cv_period = 10
        hl_ema = ta.EMA(dataframe['high'] - dataframe['low'], timeperiod=cv_period)
        dataframe['chaikin_volatility'] = ((hl_ema - hl_ema.shift(cv_period)) / hl_ema.shift(cv_period)) * 100
        
        # Realized volatility proxy (VIX-style approximation)
        returns = dataframe['close'].pct_change()
        dataframe['volatility_index'] = returns.rolling(20).std() * np.sqrt(365) * 100  # volatility
        
        return dataframe
    
    def calculate_advanced_momentum_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate advanced momentum indicators."""
        
        # Fisher Transform (price normalization)
        dataframe = self.fisher_transform(dataframe)
        
        # KST indicator (multi-horizon ROC blend)
        dataframe = self.kst_indicator(dataframe)
        
        # Coppock Curve (longer-horizon momentum)
        dataframe = self.coppock_curve(dataframe)
        
        # Vortex indicator (trend strength)
        dataframe = self.vortex_indicator(dataframe)
        
        # Stochastic Momentum Index (SMI)
        dataframe = self.stochastic_momentum_index(dataframe)
        
        # True Strength Index (TSI)
        dataframe = self.true_strength_index(dataframe)
        
        return dataframe
    
    def fisher_transform(self, dataframe: DataFrame, period: int = 10) -> DataFrame:
        """Calculate the Fisher Transform indicator."""
        hl2 = (dataframe['high'] + dataframe['low']) / 2
        
        # Rolling high/low range for normalization
        high_n = hl2.rolling(period).max()
        low_n = hl2.rolling(period).min()
        
        # Normalize price into the [-1, 1] range
        normalized_price = 2 * ((hl2 - low_n) / (high_n - low_n) - 0.5)
        normalized_price = normalized_price.clip(-0.999, 0.999)  # prevent log singularities
        
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
        """Calculate the KST (Know Sure Thing) indicator."""
        # Four rate-of-change components
        roc1 = ta.ROC(dataframe, timeperiod=10)
        roc2 = ta.ROC(dataframe, timeperiod=15)
        roc3 = ta.ROC(dataframe, timeperiod=20)
        roc4 = ta.ROC(dataframe, timeperiod=30)
        
        # Smooth each ROC series
        roc1_ma = ta.SMA(roc1, timeperiod=10)
        roc2_ma = ta.SMA(roc2, timeperiod=10)
        roc3_ma = ta.SMA(roc3, timeperiod=10)
        roc4_ma = ta.SMA(roc4, timeperiod=15)
        
        # Build the weighted KST line
        dataframe['kst'] = (roc1_ma * 1) + (roc2_ma * 2) + (roc3_ma * 3) + (roc4_ma * 4)
        dataframe['kst_signal'] = ta.SMA(dataframe['kst'], timeperiod=9)
        
        return dataframe
    
    def coppock_curve(self, dataframe: DataFrame, wma_period: int = 10) -> DataFrame:
        """Calculate the Coppock Curve."""
        # Coppock ROC inputs
        roc11 = ta.ROC(dataframe, timeperiod=11)
        roc14 = ta.ROC(dataframe, timeperiod=14)
        
        # Combine the ROC components
        roc_sum = roc11 + roc14
        
        # Weighted moving average of the combined ROC
        dataframe['coppock'] = ta.WMA(roc_sum, timeperiod=wma_period)
        
        return dataframe
    
    def vortex_indicator(self, dataframe: DataFrame, period: int = 14) -> DataFrame:
        """Calculate the Vortex indicator."""
        # True Range
        tr = ta.TRANGE(dataframe)
        
        # Positive and negative vortex movement
        vm_plus = abs(dataframe['high'] - dataframe['low'].shift(1))
        vm_minus = abs(dataframe['low'] - dataframe['high'].shift(1))
        
        # Rolling sums used to form VI+/VI-
        vm_plus_sum = vm_plus.rolling(period).sum()
        vm_minus_sum = vm_minus.rolling(period).sum()
        tr_sum = tr.rolling(period).sum()
        
        # Final Vortex values
        dataframe['vi_plus'] = vm_plus_sum / tr_sum
        dataframe['vi_minus'] = vm_minus_sum / tr_sum
        dataframe['vi_diff'] = dataframe['vi_plus'] - dataframe['vi_minus']
        
        return dataframe
    
    def stochastic_momentum_index(self, dataframe: DataFrame, k_period: int = 10, d_period: int = 3) -> DataFrame:
        """Calculate the Stochastic Momentum Index (SMI)."""
        # Midpoint of the recent trading range
        mid_point = (dataframe['high'].rolling(k_period).max() + dataframe['low'].rolling(k_period).min()) / 2
        
        # Build the SMI numerator and denominator
        numerator = (dataframe['close'] - mid_point).rolling(k_period).sum()
        denominator = (dataframe['high'].rolling(k_period).max() - dataframe['low'].rolling(k_period).min()).rolling(k_period).sum() / 2
        
        smi_k = (numerator / denominator) * 100
        dataframe['smi_k'] = smi_k
        dataframe['smi_d'] = smi_k.rolling(d_period).mean()
        
        return dataframe
    
    def true_strength_index(self, dataframe: DataFrame, r: int = 25, s: int = 13) -> DataFrame:
        """Calculate the True Strength Index (TSI)."""
        # Price change
        price_change = dataframe['close'].diff()
        
        # Double-smoothed price change
        first_smooth_pc = price_change.ewm(span=r).mean()
        double_smooth_pc = first_smooth_pc.ewm(span=s).mean()
        
        # Double-smoothed absolute price change
        first_smooth_abs_pc = abs(price_change).ewm(span=r).mean()
        double_smooth_abs_pc = first_smooth_abs_pc.ewm(span=s).mean()
        
        # Final TSI values
        dataframe['tsi'] = 100 * (double_smooth_pc / double_smooth_abs_pc)
        dataframe['tsi_signal'] = dataframe['tsi'].ewm(span=7).mean()
        
        return dataframe
    
    def calculate_advanced_volume_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate advanced volume indicators."""
        
        # Accumulation/Distribution Line (A/D)
        dataframe['ad_line'] = ta.AD(dataframe)
        dataframe['ad_line_ma'] = ta.SMA(dataframe['ad_line'], timeperiod=20)
        
        # Money Flow Index (MFI, volume-weighted RSI-like oscillator)
        dataframe['mfi'] = ta.MFI(dataframe, timeperiod=14)
        
        # Force Index
        force_index = (dataframe['close'] - dataframe['close'].shift(1)) * dataframe['volume']
        dataframe['force_index'] = force_index.ewm(span=13).mean()
        dataframe['force_index_ma'] = force_index.rolling(20).mean()
        
        # Ease of Movement
        high_low_avg = (dataframe['high'] + dataframe['low']) / 2
        high_low_avg_prev = high_low_avg.shift(1)
        distance_moved = high_low_avg - high_low_avg_prev
        
        high_low_diff = dataframe['high'] - dataframe['low']
        box_ratio = (dataframe['volume'] / 1000000) / (high_low_diff + 1e-10)
        
        emv_1 = distance_moved / (box_ratio + 1e-10)
        dataframe['emv'] = emv_1.rolling(14).mean()
        
        # Chaikin Money Flow (CMF)
        money_flow_multiplier = ((dataframe['close'] - dataframe['low']) - 
                               (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'] + 1e-10)
        money_flow_volume = money_flow_multiplier * dataframe['volume']
        dataframe['cmf'] = money_flow_volume.rolling(20).sum() / (dataframe['volume'].rolling(20).sum() + 1e-10)
        
        # Volume Price Trend (VPT)
        vpt = (dataframe['volume'] * ((dataframe['close'] - dataframe['close'].shift(1)) / (dataframe['close'].shift(1) + 1e-10)))
        dataframe['vpt'] = vpt.cumsum()
        dataframe['vpt_ma'] = dataframe['vpt'].rolling(20).mean()
        
        return dataframe
    
    def calculate_market_structure_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate market-structure indicators."""
        
        # Price action indicators
        dataframe = self.calculate_price_action_indicators(dataframe)
        
        # support/resistance
        dataframe = self.identify_support_resistance(dataframe)
        
        # Wave structure analysis
        dataframe = self.calculate_wave_analysis(dataframe)
        
        # Price-density analysis
        dataframe = self.calculate_price_density(dataframe)
        
        return dataframe
    
    def calculate_price_action_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate price-action indicators."""
        # Candle body size
        dataframe['real_body'] = abs(dataframe['close'] - dataframe['open'])
        dataframe['real_body_pct'] = dataframe['real_body'] / (dataframe['close'] + 1e-10) * 100
        
        # Upper and lower shadows
        dataframe['upper_shadow'] = dataframe['high'] - dataframe[['open', 'close']].max(axis=1)
        dataframe['lower_shadow'] = dataframe[['open', 'close']].min(axis=1) - dataframe['low']
        
        # Basic candlestick classifications
        dataframe['is_doji'] = (dataframe['real_body_pct'] < 0.1).astype(int)
        dataframe['is_hammer'] = ((dataframe['lower_shadow'] > dataframe['real_body'] * 2) & 
                                 (dataframe['upper_shadow'] < dataframe['real_body'] * 0.5)).astype(int)
        dataframe['is_shooting_star'] = ((dataframe['upper_shadow'] > dataframe['real_body'] * 2) & 
                                        (dataframe['lower_shadow'] < dataframe['real_body'] * 0.5)).astype(int)
        
        # Pin bars
        # Bullish pin bar: long lower shadow, small body, short upper shadow
        dataframe['is_pin_bar_bullish'] = ((dataframe['lower_shadow'] > dataframe['real_body'] * 2) & 
                                          (dataframe['upper_shadow'] < dataframe['real_body']) &
                                          (dataframe['real_body_pct'] < 2.0) &  # small body
                                          (dataframe['close'] > dataframe['open'])).astype(int)  # bullish close
        
        # Bearish pin bar: long upper shadow, small body, short lower shadow
        dataframe['is_pin_bar_bearish'] = ((dataframe['upper_shadow'] > dataframe['real_body'] * 2) & 
                                          (dataframe['lower_shadow'] < dataframe['real_body']) &
                                          (dataframe['real_body_pct'] < 2.0) &  # small body
                                          (dataframe['close'] < dataframe['open'])).astype(int)  # bearish close
        
        # Engulfing patterns
        # Capture the previous candle for comparison
        prev_open = dataframe['open'].shift(1)
        prev_close = dataframe['close'].shift(1)
        prev_high = dataframe['high'].shift(1)
        prev_low = dataframe['low'].shift(1)
        
        # Bullish engulfing: current bullish body fully engulfs the previous bearish body
        dataframe['is_bullish_engulfing'] = ((dataframe['close'] > dataframe['open']) &  # current candle is bullish
                                           (prev_close < prev_open) &  # previous candle is bearish
                                           (dataframe['open'] < prev_close) &  # open below prior close
                                           (dataframe['close'] > prev_open) &  # close above prior open
                                           (dataframe['real_body'] > dataframe['real_body'].shift(1) * 1.2)).astype(int)  # stronger body
        
        # Bearish engulfing: current bearish body fully engulfs the previous bullish body
        dataframe['is_bearish_engulfing'] = ((dataframe['close'] < dataframe['open']) &  # current candle is bearish
                                           (prev_close > prev_open) &  # previous candle is bullish
                                           (dataframe['open'] > prev_close) &  # open above prior close
                                           (dataframe['close'] < prev_open) &  # close below prior open
                                           (dataframe['real_body'] > dataframe['real_body'].shift(1) * 1.2)).astype(int)  # stronger body
        
        return dataframe
    
    def identify_support_resistance(self, dataframe: DataFrame, window: int = 20) -> DataFrame:
        """Identify basic support and resistance structure."""
        # Build support/resistance columns in one pass
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
        """Calculate simple wave-structure indicators."""
        # Lightweight wave-proxy indicators collected in one pass
        returns = dataframe['close'].pct_change()
        
        wave_columns = {
            'wave_strength': abs(dataframe['close'] - dataframe['close'].shift(5)) / (dataframe['close'].shift(5) + 1e-10),
            'normalized_returns': returns / (returns.rolling(20).std() + 1e-10),
            'momentum_dispersion': dataframe['mom_10'].rolling(10).std() / (abs(dataframe['mom_10']).rolling(10).mean() + 1e-10)
        }
        
        wave_df = pd.DataFrame(wave_columns, index=dataframe.index)
        return pd.concat([dataframe, wave_df], axis=1)
    
    def calculate_price_density(self, dataframe: DataFrame) -> DataFrame:
        """Calculate price-density indicators with minimal dataframe churn."""
        # Collect new columns before assignment
        new_columns = {}
        
        # Candle range as a percentage of price
        price_range = dataframe['high'] - dataframe['low']
        new_columns['price_range_pct'] = price_range / (dataframe['close'] + 1e-10) * 100
        
        # Narrower ranges imply denser price action
        new_columns['price_density'] = 1 / (new_columns['price_range_pct'] + 0.1)  # small ranges yield higher density
        
        # Assign the prepared values directly
        if new_columns:
            for col_name, value in new_columns.items():
                if isinstance(value, pd.Series):
                    # Align the series length with the dataframe when possible
                    if len(value) == len(dataframe):
                        dataframe[col_name] = value.values
                    else:
                        dataframe[col_name] = value
                else:
                    dataframe[col_name] = value
        
        return dataframe
    
    def calculate_composite_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Calculate higher-level composite indicators."""
        
        # Collect new columns before assignment
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
        
        # Overall technical health summary
        new_columns['technical_health'] = self.calculate_technical_health(dataframe)
        
        # Assign the prepared values directly
        if new_columns:
            for col_name, value in new_columns.items():
                if isinstance(value, pd.Series):
                    # Align the series length with the dataframe when possible
                    if len(value) == len(dataframe):
                        dataframe[col_name] = value.values
                    else:
                        dataframe[col_name] = value
                else:
                    dataframe[col_name] = value
        
        return dataframe
    
    def calculate_momentum_score(self, dataframe: DataFrame) -> pd.Series:
        """Calculate a composite momentum score."""
        # Collect normalized momentum indicators
        momentum_indicators = {}
        
        # Core momentum indicators
        if 'rsi_14' in dataframe.columns:
            momentum_indicators['rsi_14'] = (dataframe['rsi_14'] - 50) / 50  # normalized RSI
        if 'mom_10' in dataframe.columns:
            momentum_indicators['mom_10'] = np.where(dataframe['close'] > 0, 
                                                     dataframe['mom_10'] / dataframe['close'] * 100, 
                                                     0)  # momentum
        if 'roc_10' in dataframe.columns:
            momentum_indicators['roc_10'] = dataframe['roc_10'] / 100  # ROC
        if 'macd' in dataframe.columns:
            momentum_indicators['macd_normalized'] = np.where(dataframe['close'] > 0, 
                                                             dataframe['macd'] / dataframe['close'] * 1000, 
                                                             0)  # normalized MACD
        
        # Advanced momentum indicators
        if 'kst' in dataframe.columns:
            momentum_indicators['kst_normalized'] = dataframe['kst'] / abs(dataframe['kst']).rolling(20).mean()  # normalized KST
        if 'fisher' in dataframe.columns:
            momentum_indicators['fisher'] = dataframe['fisher']  # Fisher Transform
        if 'tsi' in dataframe.columns:
            momentum_indicators['tsi'] = dataframe['tsi'] / 100  # TSI
        if 'vi_diff' in dataframe.columns:
            momentum_indicators['vi_diff'] = dataframe['vi_diff']  # Vortexvalue
        
        # Indicator weights
        weights = {
            'rsi_14': 0.15, 'mom_10': 0.10, 'roc_10': 0.10, 'macd_normalized': 0.15,
            'kst_normalized': 0.15, 'fisher': 0.15, 'tsi': 0.10, 'vi_diff': 0.10
        }
        
        momentum_score = self._safe_series(0.0, len(dataframe))
        
        for indicator, weight in weights.items():
            if indicator in momentum_indicators:
                normalized_indicator = momentum_indicators[indicator].fillna(0)
                # Clip each indicator into a bounded range before weighting
                normalized_indicator = normalized_indicator.clip(-3, 3) / 3
                momentum_score += normalized_indicator * weight
        
        return momentum_score.clip(-1, 1)
    
    def calculate_trend_strength_score(self, dataframe: DataFrame) -> pd.Series:
        """Calculate a composite trend-strength score."""
        # Collect normalized trend indicators
        trend_indicators = {}
        
        if 'adx' in dataframe.columns:
            trend_indicators['adx'] = dataframe['adx'] / 100  # normalized ADX
        
        # EMA alignment score
        trend_indicators['ema_trend'] = self.calculate_ema_trend_score(dataframe)
        
        # SuperTrend
        trend_indicators['supertrend_trend'] = self.calculate_supertrend_score(dataframe)
        
        # Ichimoku
        trend_indicators['ichimoku_trend'] = self.calculate_ichimoku_score(dataframe)
        
        # Linear regression trend estimate
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
        """Calculate a trend score from EMA alignment."""
        score = self._safe_series(0.0, len(dataframe))
        
        # EMA stack alignment
        if all(col in dataframe.columns for col in ['ema_8', 'ema_21', 'ema_50']):
            # bullish: EMA8 > EMA21 > EMA50
            score += (dataframe['ema_8'] > dataframe['ema_21']).astype(int) * 0.4
            score += (dataframe['ema_21'] > dataframe['ema_50']).astype(int) * 0.3
            score += (dataframe['close'] > dataframe['ema_8']).astype(int) * 0.3
            
            # bearish EMA alignment
            score -= (dataframe['ema_8'] < dataframe['ema_21']).astype(int) * 0.4
            score -= (dataframe['ema_21'] < dataframe['ema_50']).astype(int) * 0.3
            score -= (dataframe['close'] < dataframe['ema_8']).astype(int) * 0.3
        
        return score.clip(-1, 1)
    
    def calculate_supertrend_score(self, dataframe: DataFrame) -> pd.Series:
        """Calculate a trend score from SuperTrend positioning."""
        if 'supertrend' not in dataframe.columns:
            return self._safe_series(0.0, len(dataframe))
        
        # Price relative to SuperTrend
        trend_score = ((dataframe['close'] > dataframe['supertrend']).astype(int) * 2 - 1)
        
        # Scale conviction by distance from the SuperTrend line
        distance_factor = np.where(dataframe['close'] > 0, 
                                  abs(dataframe['close'] - dataframe['supertrend']) / dataframe['close'], 
                                  0)
        distance_factor = distance_factor.clip(0, 0.1) / 0.1  # cap the effect at 10%
        
        return trend_score * distance_factor
    
    def calculate_ichimoku_score(self, dataframe: DataFrame) -> pd.Series:
        """Calculate a trend score from Ichimoku signals."""
        score = self._safe_series(0.0, len(dataframe))
        
        # Ichimoku cloud signals
        if all(col in dataframe.columns for col in ['tenkan', 'kijun', 'senkou_a', 'senkou_b']):
            # Price above the cloud
            above_cloud = ((dataframe['close'] > dataframe['senkou_a']) & 
                          (dataframe['close'] > dataframe['senkou_b'])).astype(int)
            
            # Price below the cloud
            below_cloud = ((dataframe['close'] < dataframe['senkou_a']) & 
                          (dataframe['close'] < dataframe['senkou_b'])).astype(int)
            
            # Tenkan/Kijun alignment
            tenkan_above_kijun = (dataframe['tenkan'] > dataframe['kijun']).astype(int)
            
            score = (above_cloud * 0.5 + tenkan_above_kijun * 0.3 + 
                    (dataframe['close'] > dataframe['tenkan']).astype(int) * 0.2 - 
                    below_cloud * 0.5)
        
        return score.clip(-1, 1)
    
    def calculate_linear_regression_trend(self, dataframe: DataFrame, period: int = 20) -> pd.Series:
        """Calculate a linear-regression-based trend score."""
        def linear_reg_slope(y):
            if len(y) < 2:
                return 0
            x = np.arange(len(y))
            from scipy import stats
            slope, _, r_value, _, _ = stats.linregress(x, y)
            return slope * r_value ** 2  # scale slope by fit quality
        
        # Rolling regression slope
        reg_slope = dataframe['close'].rolling(period).apply(linear_reg_slope, raw=False)
        
        # Normalize the slope relative to price
        normalized_slope = np.where(dataframe['close'] > 0, 
                                   reg_slope / dataframe['close'] * 1000, 
                                   0)  # scale into a usable range
        
        return normalized_slope.fillna(0).clip(-1, 1)
    
    def calculate_volatility_regime(self, dataframe: DataFrame) -> pd.Series:
        """Calculate a simple volatility regime classifier."""
        # Current ATR-based volatility
        current_vol = dataframe['atr_p']
        
        # Volatility percentile over a rolling window
        vol_percentile = current_vol.rolling(100).rank(pct=True)
        
        # Regime buckets
        regime = self._safe_series(0, len(dataframe))  # 0: medium volatility
        regime[vol_percentile < 0.2] = -1  # low volatility
        regime[vol_percentile > 0.8] = 1   # high volatility
        
        return regime
    
    def calculate_market_regime(self, dataframe: DataFrame) -> pd.Series:
        """Calculate a composite market-regime score."""
        # Collect regime factors
        regime_factors = {}
        
        if 'trend_strength_score' in dataframe.columns:
            regime_factors['trend_strength'] = dataframe['trend_strength_score']
        if 'momentum_score' in dataframe.columns:
            regime_factors['momentum'] = dataframe['momentum_score']
        if 'volatility_regime' in dataframe.columns:
            regime_factors['volatility'] = dataframe['volatility_regime'] / 2  # reduce volatility weight into the same scale
        if 'volume_ratio' in dataframe.columns:
            regime_factors['volume_trend'] = (dataframe['volume_ratio'] - 1).clip(-1, 1)
        
        weights = {'trend_strength': 0.4, 'momentum': 0.3, 'volatility': 0.2, 'volume_trend': 0.1}
        
        market_regime = self._safe_series(0.0, len(dataframe))
        for factor, weight in weights.items():
            if factor in regime_factors:
                market_regime += regime_factors[factor].fillna(0) * weight
        
        return market_regime.clip(-1, 1)
    
    # Simplified risk-adjusted return calculation
    def calculate_risk_adjusted_returns(self, dataframe: DataFrame, window: int = 20) -> pd.Series:
        """Calculate a simplified risk-adjusted return series."""
        # Price returns
        returns = dataframe['close'].pct_change()
        
        # Rolling Sharpe-like estimate
        rolling_returns = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        
        risk_adjusted = rolling_returns / (rolling_std + 1e-6)  # avoid
        
        return risk_adjusted.fillna(0)
    
    def identify_coin_risk_tier(self, pair: str, dataframe: DataFrame) -> str:
        """Classify the pair into a simple coin risk tier."""
        
        try:
            if dataframe.empty or len(dataframe) < 96:  # need roughly 24 hours of 15m data
                return 'medium_risk'  # default medium risk
                
            current_idx = -1
            
            # === Factor 1: price volatility analysis ===
            volatility = dataframe['atr_p'].iloc[current_idx] if 'atr_p' in dataframe.columns else 0.05
            volatility_24h = dataframe['close'].rolling(96).std().iloc[current_idx] / dataframe['close'].iloc[current_idx]
            
            # === Factor 2: volume stability ===
            volume_series = dataframe['volume'].rolling(24)
            volume_mean = volume_series.mean().iloc[current_idx]
            volume_std = volume_series.std().iloc[current_idx]
            volume_cv = (volume_std / volume_mean) if volume_mean > 0 else 5  # coefficient of variation
            
            # === Factor 3: 24h price displacement ===
            current_price = dataframe['close'].iloc[current_idx]
            price_24h_ago = dataframe['close'].iloc[-96] if len(dataframe) >= 96 else dataframe['close'].iloc[0]
            price_change_24h = abs((current_price / price_24h_ago) - 1) if price_24h_ago > 0 else 0
            
            # === Factor 4: nominal price level ===
            is_micro_price = current_price < 0.001  # micro-priced assets are often more unstable
            is_low_price = current_price < 0.1      # low price
            
            # === Factor 5: extreme oscillator readings ===
            rsi = dataframe['rsi_14'].iloc[current_idx] if 'rsi_14' in dataframe.columns else 50
            is_extreme_rsi = rsi > 80 or rsi < 20  # extreme RSI reading
            
            # === Factor 6: pump-like hourly price bursts ===
            recent_pumps = 0
            if len(dataframe) >= 24:
                for i in range(1, min(24, len(dataframe))):
                    hour_change = (dataframe['close'].iloc[-i] / dataframe['close'].iloc[-i-1]) - 1
                    if hour_change > 0.15:  # >15% in one hour
                        recent_pumps += 1
            
            # === Aggregate risk score ===
            risk_score = 0
            risk_factors = []
            
            # Volatility score (0-40)
            if volatility > 0.20:  # high volatility
                risk_score += 40
                risk_factors.append(f"high volatility({volatility*100:.1f}%)")
            elif volatility > 0.10:
                risk_score += 25
                risk_factors.append(f"high volatility({volatility*100:.1f}%)")
            elif volatility > 0.05:
                risk_score += 10
                risk_factors.append(f"moderate volatility({volatility*100:.1f}%)")
            
            # Volume-instability score (0-25)
            if volume_cv > 3:  # highly unstable volume
                risk_score += 25
                risk_factors.append(f"unstable volume(CV:{volume_cv:.1f})")
            elif volume_cv > 1.5:
                risk_score += 15
                risk_factors.append(f"elevated volume instability(CV:{volume_cv:.1f})")
            
            # Short-term price dislocation score (0-20)
            if price_change_24h > 0.50:  # 24hours50%
                risk_score += 20
                risk_factors.append(f"24h move({price_change_24h*100:.1f}%)")
            elif price_change_24h > 0.20:
                risk_score += 10
                risk_factors.append(f"large 24h move({price_change_24h*100:.1f}%)")
            
            # Price-level score (0-10)
            if is_micro_price:
                risk_score += 10
                risk_factors.append(f"micro price(${current_price:.6f})")
            elif is_low_price:
                risk_score += 5
                risk_factors.append(f"low price(${current_price:.3f})")
            
            # Pump-pattern score (0-15)
            if recent_pumps >= 3:
                risk_score += 15
                risk_factors.append(f"repeated pumps({recent_pumps} times)")
            elif recent_pumps >= 1:
                risk_score += 8
                risk_factors.append(f"recent pumps({recent_pumps} times)")

            if is_extreme_rsi:
                risk_factors.append(f"extreme RSI({rsi:.1f})")
            
            # === Risk tier ===
            if risk_score >= 70:
                risk_tier = 'high_risk'    # aggressive risk profile
                tier_name = "⚠️ high risk"
            elif risk_score >= 40:
                risk_tier = 'medium_risk'  # medium risk
                tier_name = "⚡ medium risk"
            else:
                risk_tier = 'low_risk'     # lower-risk profile
                tier_name = "✅ low risk"
            
            # Diagnostic summary
            logger.info(f"""
🎯 Coin risk assessment - {pair}:
├─ Risk tier: {tier_name} (score: {risk_score}/100)
├─ Current price: ${current_price:.6f}
├─ ATR volatility: {volatility*100:.2f}% | 24h move: {price_change_24h*100:.1f}%
├─ Volume CV: {volume_cv:.2f} | Recent pumps: {recent_pumps} times
├─ Factors: {' | '.join(risk_factors) if risk_factors else 'none'}
└─ Guidance: {'use smaller size and tighter leverage' if risk_tier == 'high_risk' else 'standard risk controls are acceptable' if risk_tier == 'low_risk' else 'keep position sizing moderate'}
""")
            
            return risk_tier
            
        except Exception as e:
            logger.error(f"Coin risk classification failed for {pair}: {e}")
            return 'medium_risk'  # medium risk
    
    def calculate_technical_health(self, dataframe: DataFrame) -> pd.Series:
        """Calculate an overall technical-health score."""
        health_components = {}
        
        # 1. Trend consistency across multiple indicators
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
        
        # 2. Prefer moderate volatility over extremes
        if 'volatility_regime' in dataframe.columns:
            vol_score = 1 - abs(dataframe['volatility_regime']) * 0.5  # neutral volatility scores highest
            health_components['volatility_health'] = vol_score
        
        # 3. volume confirmation
        if 'volume_ratio' in dataframe.columns:
            volume_health = ((dataframe['volume_ratio'] > 0.8).astype(float) * 0.5 + 
                           (dataframe['volume_ratio'] < 2.0).astype(float) * 0.5)  # avoid both weak and extreme volume
            health_components['volume_health'] = volume_health
        
        # 4. Penalize multi-indicator overbought or oversold extremes
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
        
        # Weighted composite score
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
        """Detect the current market state."""
        current_idx = -1
        
        # Read core indicators
        adx = dataframe['adx'].iloc[current_idx]
        atr_p = dataframe['atr_p'].iloc[current_idx]
        rsi = dataframe['rsi_14'].iloc[current_idx]
        volume_ratio = dataframe['volume_ratio'].iloc[current_idx]
        price = dataframe['close'].iloc[current_idx]
        ema_8 = dataframe['ema_8'].iloc[current_idx] if 'ema_8' in dataframe.columns else price
        ema_21 = dataframe['ema_21'].iloc[current_idx]
        ema_50 = dataframe['ema_50'].iloc[current_idx]
        
        # MACD values
        macd = dataframe['macd'].iloc[current_idx] if 'macd' in dataframe.columns else 0
        macd_signal = dataframe['macd_signal'].iloc[current_idx] if 'macd_signal' in dataframe.columns else 0
        
        # === Price position within the recent range ===
        high_20 = dataframe['high'].rolling(20).max().iloc[current_idx]
        low_20 = dataframe['low'].rolling(20).min().iloc[current_idx]
        price_position = (price - low_20) / (high_20 - low_20) if high_20 > low_20 else 0.5
        
        # Potential local market top
        is_at_top = (
            price_position > 0.90 and  # price near the 20-period high
            rsi > 70 and  # RSI overbought
            macd < macd_signal  # MACD momentum weakening
        )
        
        # Potential local market bottom
        is_at_bottom = (
            price_position < 0.10 and  # price near the 20-period low
            rsi < 30 and  # RSI oversold
            macd > macd_signal  # MACD momentum improving
        )
        
        # === trend strength analysis ===
        # EMA stack alignment
        ema_bullish = ema_8 > ema_21 > ema_50
        ema_bearish = ema_8 < ema_21 < ema_50
        
        # === market state ===
        if is_at_top:
            return "market_top"  # avoid fresh longs into exhaustion
        elif is_at_bottom:
            return "market_bottom"  # avoid fresh shorts into exhaustion
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
        """Calculate Value at Risk (VaR)."""
        if len(returns) < 20:
            return 0.05  # default 5% risk
        
        returns_array = np.array(returns)
        # Historical percentile estimate
        var = np.percentile(returns_array, confidence_level * 100)
        return abs(var)
    
    def calculate_cvar(self, returns: List[float], confidence_level: float = 0.05) -> float:
        """Calculate Conditional Value at Risk (CVaR)."""
        if len(returns) < 20:
            return 0.08  # default 8% risk
        
        returns_array = np.array(returns)
        var = np.percentile(returns_array, confidence_level * 100)
        # CVaR is the average loss beyond the VaR threshold
        tail_losses = returns_array[returns_array <= var]
        if len(tail_losses) > 0:
            cvar = np.mean(tail_losses)
            return abs(cvar)
        return abs(var)
    
    def calculate_portfolio_correlation(self, pair: str) -> float:
        """Calculate the average correlation of a pair against the rest of the portfolio."""
        if pair not in self.pair_returns_history:
            return 0.0
        
        current_returns = self.pair_returns_history[pair]
        if len(current_returns) < 20:
            return 0.0
        
        # Compare the pair with other tracked pairs
        correlations = []
        for other_pair, other_returns in self.pair_returns_history.items():
            if other_pair != pair and len(other_returns) >= 20:
                try:
                    # Use the overlapping history window
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
        """Calculate a conservative Kelly fraction for the pair."""
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
            
            # Kelly formula: f = (bp - q) / b
            # where b = avg_win / avg_loss, p = win_prob, q = 1 - win_prob
            b = avg_win / avg_loss
            kelly = (b * win_prob - (1 - win_prob)) / b
            
            # Use a fractional Kelly approach for safety
            kelly_adjusted = max(0.05, min(0.4, kelly * 0.25))
            return kelly_adjusted
            
        except:
            return 0.25
    
    def calculate_position_size(self, current_price: float, market_state: str, pair: str) -> float:
        """Calculate position size from market state and portfolio risk controls."""
        
        # === 🎯 get risk level ===
        try:
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if not dataframe.empty:
                coin_risk_tier = self.identify_coin_risk_tier(pair, dataframe)
            else:
                coin_risk_tier = 'medium_risk'
        except Exception as e:
            logger.warning(f"Failed to get coin risk tier for {pair}: {e}")
            coin_risk_tier = 'medium_risk'
        
        # === Risk-tier position multipliers ===
        coin_risk_multipliers = {
            'low_risk': 1.0,        # standard position size
            'medium_risk': 0.8,     # reduce to 80%
            'high_risk': 0.3        # reduce to 30% for high-risk assets
        }
        coin_risk_multiplier = coin_risk_multipliers.get(coin_risk_tier, 0.8)
        
        # === Use the midpoint of the configured position-size range ===
        base_position = (self.base_position_size + self.max_position_size) / 2
        
        # === winning streak/losing streak ===
        streak_multiplier = 1.0
        if self.consecutive_wins >= 5:
            streak_multiplier = 1.5
        elif self.consecutive_wins >= 3:
            streak_multiplier = 1.3
        elif self.consecutive_wins >= 1:
            streak_multiplier = 1.1
        elif self.consecutive_losses >= 3:
            streak_multiplier = 0.6
        elif self.consecutive_losses >= 1:
            streak_multiplier = 0.8
            
        # === Market-state multiplier ===
        market_multipliers = {
            "strong_uptrend": 1.25,
            "strong_downtrend": 1.25,
            "mild_uptrend": 1.2,        # medium trend
            "mild_downtrend": 1.2,      # medium trend
            "sideways": 1.0,
            "volatile": 0.8,
            "consolidation": 0.9
        }
        market_multiplier = market_multipliers.get(market_state, 1.0)
        
        # === time ===
        time_multiplier = self.get_time_session_position_boost()
        
        # === Equity-state multiplier ===
        equity_multiplier = 1.0
        if self.current_drawdown < -0.10:  # drawdown10%
            equity_multiplier = 0.6
        elif self.current_drawdown < -0.05:  # drawdown5%
            equity_multiplier = 0.8
        elif self.current_drawdown == 0:     # none drawdown，profit
            equity_multiplier = 1.15
            
        # === Leverage-aware position scaling ===
        # get the currently assigned leverage
        current_leverage = getattr(self, '_current_leverage', {}).get(pair, 20)
        # reduce position size when leverage is already high
        leverage_adjustment = 1.0
        if current_leverage >= 75:
            leverage_adjustment = 0.8    # high leverage low position size
        elif current_leverage >= 50:
            leverage_adjustment = 0.9
        else:
            leverage_adjustment = 1.1    # slightly larger size at lower leverage
            
        # === Compound-growth accelerator ===
        compound_multiplier = self.get_compound_accelerator_multiplier()
            
        # === Total multiplier ===
        total_multiplier = (streak_multiplier * market_multiplier * 
                          time_multiplier * equity_multiplier * 
                          leverage_adjustment * compound_multiplier * 
                          coin_risk_multiplier)
        
        # Cap the multiplier by risk tier
        max_multiplier_limits = {
            'low_risk': 1.8,
            'medium_risk': 1.5,
            'high_risk': 1.2
        }
        max_multiplier = max_multiplier_limits.get(coin_risk_tier, 1.5)
        total_multiplier = min(total_multiplier, max_multiplier)
        
        # === Raw position-size calculation ===
        calculated_position = base_position * total_multiplier
        
        # === Position-size cap based on leverage ===
        if current_leverage >= 75:
            max_allowed_position = 0.15
        elif current_leverage >= 50:
            max_allowed_position = 0.20
        elif current_leverage >= 20:
            max_allowed_position = 0.30
        else:
            max_allowed_position = self.max_position_size
        
        # Keep size within a reasonable floor and cap
        final_position = max(self.base_position_size * 0.8, 
                           min(calculated_position, max_allowed_position))
        
        # risk level
        risk_tier_names = {
            'low_risk': '✅ low risk',
            'medium_risk': '⚡ medium risk', 
            'high_risk': '⚠️ high risk'
        }
        
        logger.info(f"""
💰 Position size calculation - {pair}:
├─ Risk tier: {risk_tier_names.get(coin_risk_tier, coin_risk_tier)}
├─ Base position: {base_position*100:.0f}%
├─ Streak multiplier: {streak_multiplier:.1f}x (wins: {self.consecutive_wins}, losses: {self.consecutive_losses})
├─ Market multiplier: {market_multiplier:.1f}x ({market_state})
├─ Time-session multiplier: {time_multiplier:.1f}x
├─ Equity multiplier: {equity_multiplier:.1f}x
├─ Leverage adjustment: {leverage_adjustment:.1f}x ({current_leverage}x leverage)
├─ Compound multiplier: {compound_multiplier:.1f}x
├─ Risk-tier adjustment: {coin_risk_multiplier:.1f}x ({coin_risk_tier})
├─ Max multiplier cap: {max_multiplier:.1f}x
├─ Raw position size: {calculated_position*100:.1f}%
└─ Final position size: {final_position*100:.1f}%
""")
        
        return final_position
    
    def get_time_session_position_boost(self) -> float:
        """Return a time-of-day position-size multiplier."""
        current_time = datetime.now(timezone.utc)
        hour = current_time.hour
        
        # Session-based position adjustments
        if 14 <= hour <= 16:       # strongest session
            return 1.2
        elif 8 <= hour <= 10:      # active session
            return 1.1
        elif 0 <= hour <= 2:       # neutral session
            return 1.0
        elif 3 <= hour <= 7:       # quieter session
            return 0.9
        else:
            return 1.0
    
    def get_compound_accelerator_multiplier(self) -> float:
        """Return a position-size multiplier based on recent daily performance."""
        
        # Estimated daily performance
        daily_profit = self.get_daily_profit_percentage()
        
        # Base accelerator mode
        if daily_profit >= 0.20:      # day > 20%
            multiplier = 1.5
            mode = "rocket"
        elif daily_profit >= 0.10:    # day 10-20%
            multiplier = 1.5
            mode = "high_gain"
        elif daily_profit >= 0.05:    # day 5-10%
            multiplier = 1.2
            mode = "positive"
        elif daily_profit >= 0:       # day 0-5%
            multiplier = 1.0
            mode = "neutral"
        elif daily_profit >= -0.05:   # day loss 0-5%
            multiplier = 0.8
            mode = "cooldown"
        else:                         # day loss > 5%
            multiplier = 0.5
            mode = "defensive"
            
        # Boost after multiple profitable days
        consecutive_profit_days = self.get_consecutive_profit_days()
        if consecutive_profit_days >= 3:
            multiplier *= min(1.3, 1 + consecutive_profit_days * 0.05)  # cap the boost at +30%
            
        # Reduce after multiple losing days
        consecutive_loss_days = self.get_consecutive_loss_days()
        if consecutive_loss_days >= 2:
            multiplier *= max(0.3, 1 - consecutive_loss_days * 0.15)   # floor at 30%
            
        # Final safety bounds: 0.3x - 2.5x
        final_multiplier = max(0.3, min(multiplier, 2.5))
        
        logger.info(f"""
🚀 Compound accelerator:
├─ Daily profit estimate: {daily_profit*100:+.2f}%
├─ Mode: {mode}
├─ Raw multiplier: {multiplier:.2f}x
├─ Consecutive profit days: {consecutive_profit_days}
├─ Consecutive loss days: {consecutive_loss_days}
└─ Final multiplier: {final_multiplier:.2f}x
""")
        
        return final_multiplier
    
    def get_daily_profit_percentage(self) -> float:
        """Return an estimated daily profit percentage."""
        try:
            # Simplified placeholder based on total profit
            if hasattr(self, 'total_profit'):
                return self.total_profit * 0.1  # approximate day-level PnL
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def get_consecutive_profit_days(self) -> int:
        """Return an approximate count of consecutive profit days."""
        try:
            # Simplified approximation from win streaks
            if self.consecutive_wins >= 5:
                return min(7, self.consecutive_wins // 2)
            else:
                return 0
        except Exception:
            return 0
    
    def get_consecutive_loss_days(self) -> int:
        """Return an approximate count of consecutive loss days."""
        try:
            # Simplified approximation from loss streaks
            if self.consecutive_losses >= 3:
                return min(5, self.consecutive_losses // 1)
            else:
                return 0
        except Exception:
            return 0
    
    def update_portfolio_performance(self, pair: str, return_pct: float):
        """Update stored performance history for the pair."""
        # Update return history
        if pair not in self.pair_returns_history:
            self.pair_returns_history[pair] = []
        
        self.pair_returns_history[pair].append(return_pct)
        
        # Keep at most 500 return samples
        if len(self.pair_returns_history[pair]) > 500:
            self.pair_returns_history[pair] = self.pair_returns_history[pair][-500:]
        
        # Update pair-level trade performance history
        if pair not in self.pair_performance:
            self.pair_performance[pair] = []
        
        self.pair_performance[pair].append(return_pct)
        if len(self.pair_performance[pair]) > 200:
            self.pair_performance[pair] = self.pair_performance[pair][-200:]
        
        # Refresh the correlation matrix
        self.update_correlation_matrix()
    
    def update_correlation_matrix(self):
        """Update the internal correlation matrix for tracked pairs."""
        try:
            pairs = list(self.pair_returns_history.keys())
            if len(pairs) < 2:
                return
            
            # Initialize the square correlation matrix
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
        """Calculate aggregate portfolio risk metrics."""
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
            
            # Aggregate VaR and CVaR across active pairs
            var_values = []
            cvar_values = []
            
            for pair in active_pairs:
                returns = self.pair_returns_history[pair]
                var_values.append(self.calculate_var(returns))
                cvar_values.append(self.calculate_cvar(returns))
            
            total_var = np.mean(var_values)
            total_cvar = np.mean(cvar_values)
            
            # Estimate average pairwise correlation
            correlations = []
            for i, pair1 in enumerate(active_pairs):
                for j, pair2 in enumerate(active_pairs):
                    if i < j:  # avoid duplicate pair calculations
                        corr = self.calculate_portfolio_correlation(pair1)
                        if corr > 0:
                            correlations.append(corr)
            
            portfolio_correlation = np.mean(correlations) if correlations else 0.0
            
            # Higher values imply better diversification
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
        """Calculate leverage from volatility, market state, and risk controls."""
        
        # === Determine the pair risk tier ===
        try:
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if not dataframe.empty:
                coin_risk_tier = self.identify_coin_risk_tier(pair, dataframe)
            else:
                coin_risk_tier = 'medium_risk'  # default medium risk
        except Exception as e:
            logger.warning(f"Failed to get coin risk tier for {pair}: {e}")
            coin_risk_tier = 'medium_risk'
        
        # === risk leverage ===
        coin_leverage_limits = {
            'low_risk': (10, 100),      # wider leverage range
            'medium_risk': (5, 50),     # moderate leverage range
            'high_risk': (1, 10)        # tightly capped leverage
        }
        
        # get before leverage
        min_allowed, max_allowed = coin_leverage_limits.get(coin_risk_tier, (5, 50))
        
        # === Map volatility to a base leverage ===
        volatility_percent = volatility * 100
        
        # Lower volatility allows higher base leverage
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
            base_leverage = 10   # high volatility
            
        # === winning streak/losing streak ===
        streak_multiplier = 1.0
        if self.consecutive_wins >= 5:
            streak_multiplier = 2.0      # winning streak5times：leverage
        elif self.consecutive_wins >= 3:
            streak_multiplier = 1.5
        elif self.consecutive_wins >= 1:
            streak_multiplier = 1.2
        elif self.consecutive_losses >= 3:
            streak_multiplier = 0.5      # loss streak3times：leverage
        elif self.consecutive_losses >= 1:
            streak_multiplier = 0.8
            
        # === time optimize ===
        time_multiplier = self.get_time_session_leverage_boost(current_time)
        
        # === Market-state multiplier ===
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
        
        # === Equity-state multiplier ===
        equity_multiplier = 1.0
        if self.current_drawdown < -0.05:  # drawdown5%
            equity_multiplier = 0.7
        elif self.current_drawdown < -0.02:  # drawdown2%
            equity_multiplier = 0.85
        elif self.current_drawdown == 0:     # none drawdown
            equity_multiplier = 1.2
            
        # === Raw leverage calculation ===
        calculated_leverage = base_leverage * streak_multiplier * time_multiplier * market_multiplier * equity_multiplier
        
        # Clamp to the global leverage window first
        pre_risk_leverage = max(10, min(int(calculated_leverage), 100))
        
        # Apply pair-specific leverage limits
        final_leverage = max(min_allowed, min(pre_risk_leverage, max_allowed))
        
        # Reduce leverage further after significant daily losses
        if hasattr(self, 'daily_loss') and self.daily_loss < -0.03:
            final_leverage = min(final_leverage, 20)
            
        # Reduce leverage after persistent losses
        if self.consecutive_losses >= 5:
            final_leverage = min(final_leverage, 15)
            
        # risk level
        risk_tier_names = {
            'low_risk': '✅ low risk',
            'medium_risk': '⚡ medium risk', 
            'high_risk': '⚠️ high risk'
        }
        
        logger.info(f"""
⚡ Leverage calculation - {pair}:
├─ Risk tier: {risk_tier_names.get(coin_risk_tier, coin_risk_tier)}
├─ Allowed leverage range: {min_allowed}-{max_allowed}x
├─ Volatility: {volatility_percent:.2f}% → base leverage: {base_leverage}x
├─ Streak multiplier: {streak_multiplier:.1f}x (wins: {self.consecutive_wins}, losses: {self.consecutive_losses})
├─ Time-session multiplier: {time_multiplier:.1f}x
├─ Market multiplier: {market_multiplier:.1f}x
├─ Equity multiplier: {equity_multiplier:.1f}x
├─ Raw leverage: {calculated_leverage:.1f}x
├─ Globally clamped leverage: {pre_risk_leverage}x (range: 10-100x)
└─ Final leverage: {final_leverage}x ({coin_risk_tier}: {min_allowed}-{max_allowed}x)
""")
        
        return final_leverage
    
    def get_time_session_leverage_boost(self, current_time: datetime = None) -> float:
        """Return a time-of-day leverage multiplier."""
        if not current_time:
            current_time = datetime.now(timezone.utc)
            
        hour = current_time.hour
        
        # Session-based leverage adjustments
        if 0 <= hour <= 2:      # early UTC session
            return 1.2
        elif 8 <= hour <= 10:   # active session
            return 1.3
        elif 14 <= hour <= 16:  # strongest session
            return 1.5
        elif 20 <= hour <= 22:  # evening session
            return 1.2
        elif 3 <= hour <= 7:    # quieter session
            return 0.8
        elif 11 <= hour <= 13:  # middling session
            return 0.9
        else:
            return 1.0
    
    # Dynamic stoploss helper removed in the current implementation
    
    def calculate_dynamic_takeprofit(self, pair: str, current_rate: float, trade: Trade, current_profit: float) -> Optional[float]:
        """Calculate an adaptive take-profit target price."""
        try:
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if dataframe.empty:
                return None
            
            current_data = dataframe.iloc[-1]
            current_atr = current_data.get('atr_p', 0.02)
            adx = current_data.get('adx', 25)
            trend_strength = current_data.get('trend_strength', 50)
            momentum_score = current_data.get('momentum_score', 0)
            
            # Base profit distance in ATR terms
            base_profit_multiplier = 2.5  # 2.5x ATR baseline
            
            # Trend-strength adjustment
            if abs(trend_strength) > 70:  # strong trend
                trend_multiplier = 1.5
            elif abs(trend_strength) > 40:  # medium trend
                trend_multiplier = 1.2
            else:  # weak or neutral trend
                trend_multiplier = 1.0
            
            # Momentum adjustment
            momentum_multiplier = 1.0
            if abs(momentum_score) > 0.3:
                momentum_multiplier = 1.3
            elif abs(momentum_score) > 0.1:
                momentum_multiplier = 1.1
            
            # Combined multiplier
            profit_multiplier = base_profit_multiplier * trend_multiplier * momentum_multiplier
            
            # Distance from entry expressed as a price fraction
            profit_distance = current_atr * profit_multiplier
            
            # range：8%-80%
            profit_distance = max(0.08, min(0.80, profit_distance))
            
            # Convert the target distance into an absolute target price
            if trade.is_short:
                target_price = trade.open_rate * (1 - profit_distance)
            else:
                target_price = trade.open_rate * (1 + profit_distance)
            
            logger.info(f"""
🎯 Dynamic take-profit - {pair}:
├─ Entry price: ${trade.open_rate:.6f}
├─ Current price: ${current_rate:.6f}
├─ Current profit: {current_profit:.2%}
├─ Profit multiplier: {profit_multiplier:.2f}x ATR
├─ Profit distance: {profit_distance:.2%}
├─ Target price: ${target_price:.6f}
└─ Direction: {'short' if trade.is_short else 'long'}
""")
            
            return target_price
            
        except Exception as e:
            logger.error(f"Dynamic take-profit calculation failed for {pair}: {e}")
            return None
    
    # Smart trailing-stop helper removed in the current implementation
    
    def validate_and_calibrate_indicators(self, dataframe: DataFrame) -> DataFrame:
        """Validate and lightly smooth key indicators."""
        try:
            logger.info(f"Validating and calibrating indicators for {len(dataframe)} rows")
            
            # === RSI indicator ===
            if 'rsi_14' in dataframe.columns:
                # Clamp RSI to a valid range and fill missing values
                original_rsi_nulls = dataframe['rsi_14'].isnull().sum()
                dataframe['rsi_14'] = dataframe['rsi_14'].clip(0, 100)
                dataframe['rsi_14'] = dataframe['rsi_14'].fillna(50)
                
                # Light RSI smoothing
                dataframe['rsi_14'] = dataframe['rsi_14'].ewm(span=2).mean()
                
                logger.info(f"RSI validated - original null count: {original_rsi_nulls}, clipped to 0-100")
            
            # === MACD indicator ===
            if 'macd' in dataframe.columns:
                # Fill missing values and lightly smooth MACD
                original_macd_nulls = dataframe['macd'].isnull().sum()
                dataframe['macd'] = dataframe['macd'].fillna(0)
                dataframe['macd'] = dataframe['macd'].ewm(span=3).mean()
                
                if 'macd_signal' in dataframe.columns:
                    dataframe['macd_signal'] = dataframe['macd_signal'].fillna(0)
                    dataframe['macd_signal'] = dataframe['macd_signal'].ewm(span=3).mean()
                
                logger.info(f"MACD validated - original null count: {original_macd_nulls}, smoothed with span=3")
            
            # === ATR indicator ===
            if 'atr_p' in dataframe.columns:
                # Detect and clip ATR outliers
                atr_median = dataframe['atr_p'].median()
                atr_std = dataframe['atr_p'].std()
                
                # Keep ATR within a wide median +/- 5 std envelope
                lower_bound = max(0.001, atr_median - 5 * atr_std)
                upper_bound = min(0.5, atr_median + 5 * atr_std)
                
                original_atr_outliers = ((dataframe['atr_p'] < lower_bound) | 
                                       (dataframe['atr_p'] > upper_bound)).sum()
                
                dataframe['atr_p'] = dataframe['atr_p'].clip(lower_bound, upper_bound)
                dataframe['atr_p'] = dataframe['atr_p'].fillna(atr_median)
                
                logger.info(f"ATR validated - outliers clipped: {original_atr_outliers}, range: {lower_bound:.4f}-{upper_bound:.4f}")
            
            # === ADX indicator ===
            if 'adx' in dataframe.columns:
                dataframe['adx'] = dataframe['adx'].clip(0, 100)
                dataframe['adx'] = dataframe['adx'].fillna(25)  # ADX default
                logger.info("ADX validated - range: 0-100, default: 25")
            
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
            # Forward/backfill EMA gaps and recalculate extreme outliers if needed
            for ema_col in ['ema_8', 'ema_21', 'ema_50']:
                if ema_col in dataframe.columns:
                    # Fill missing values
                    null_count = dataframe[ema_col].isnull().sum()
                    if null_count > 0:
                        # Fill from nearby values
                        dataframe[ema_col] = dataframe[ema_col].ffill().bfill()
                        logger.info(f"{ema_col} filled missing values: {null_count}")
                    
                    # Recalculate EMA if it becomes wildly detached from price
                    if 'close' in dataframe.columns:
                        price_ratio = dataframe[ema_col] / dataframe['close']
                        outliers = ((price_ratio > 10) | (price_ratio < 0.1)).sum()
                        if outliers > 0:
                            logger.warning(f"{ema_col} has {outliers} extreme values relative to price, recalculating")
                            # Recalculate the affected EMA
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
        """Log a compact health summary for the core indicators."""
        try:
            health_report = []
            
            # Core indicators to monitor
            indicators_to_check = ['rsi_14', 'macd', 'atr_p', 'adx', 'volume_ratio', 'trend_strength', 'momentum_score', 'ema_8', 'ema_21', 'ema_50']
            
            for indicator in indicators_to_check:
                if indicator in dataframe.columns:
                    series = dataframe[indicator].dropna()
                    if len(series) > 0:
                        null_count = dataframe[indicator].isnull().sum()
                        null_pct = null_count / len(dataframe) * 100
                        
                        health_status = "healthy" if null_pct < 5 else "warning" if null_pct < 15 else "critical"
                        
                        health_report.append(f"├─ {indicator}: {health_status} (nulls: {null_pct:.1f}%)")
            
            if health_report:
                overall_status = (
                    "healthy" if all("healthy" in line for line in health_report)
                    else "warning" if any("warning" in line for line in health_report)
                    else "critical"
                )
                logger.info(f"""
📊 Indicator health:
{chr(10).join(health_report)}
└─ Overall status: {overall_status}
""")
        except Exception as e:
            logger.error(f"Indicator health logging failed: {e}")
    
    def validate_real_data_quality(self, dataframe: DataFrame, pair: str) -> bool:
        """Validate that the dataframe contains plausible market data."""
        try:
            if len(dataframe) < 10:
                logger.warning(f"Insufficient data for {pair}: only {len(dataframe)} rows")
                return False
            
            # check price
            price_cols = ['open', 'high', 'low', 'close']
            for col in price_cols:
                if col in dataframe.columns:
                    if dataframe[col].isnull().all():
                        logger.error(f"price as value {pair}: {col}")
                        return False
                    
                    # Warn if the price barely changes across the sample
                    price_std = dataframe[col].std()
                    price_mean = dataframe[col].mean()
                    if price_std / price_mean < 0.001:  # low0.1%
                        logger.warning(f"price abnormal small {pair}: {col} std/mean = {price_std/price_mean:.6f}")
            
            # check volume
            if 'volume' in dataframe.columns:
                if dataframe['volume'].sum() == 0:
                    logger.warning(f"Volume is zero for {pair}")
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
                    # Use the most common interval as the expected spacing
                    expected_interval = time_diff.mode().iloc[0] if len(time_diff.mode()) > 0 else pd.Timedelta(minutes=5)
                    abnormal_intervals = (time_diff != expected_interval).sum()
                    if abnormal_intervals > len(time_diff) * 0.1:  # more than 10% abnormal intervals
                        logger.warning(f"Time spacing anomalies for {pair}: {abnormal_intervals}/{len(time_diff)} intervals differ from expected {expected_interval}")
            
            logger.info(f"✅ Data quality passed for {pair}: {len(dataframe)} rows checked")
            return True
            
        except Exception as e:
            logger.error(f"Data quality validation failed for {pair}: {e}")
            return False
    
    # `_log_detailed_exit_decision` removed in this simplified version
    
    def _log_risk_calculation_details(self, pair: str, input_params: dict, result: dict):
        """Log risk-calculation details."""
        try:
            # Reserved for a more detailed risk decision logger
            pass
        except Exception as e:
            logger.error(f"Risk-calculation logging failed for {pair}: {e}")
    
    def _calculate_risk_rating(self, risk_percentage: float) -> str:
        """Map a numeric risk percentage to a qualitative label."""
        try:
            if risk_percentage < 0.01:  # small1%
                return "low risk"
            elif risk_percentage < 0.02:  # 1-2%
                return "moderately low risk"
            elif risk_percentage < 0.03:  # 2-3%
                return "medium risk"
            elif risk_percentage < 0.05:  # 3-5%
                return "moderately high risk"
            else:  # large5%
                return "high risk"
        except Exception:
            return "unknown risk"
    
    def get_equity_performance_factor(self) -> float:
        """Return a multiplier based on equity growth or drawdown."""
        if self.initial_balance is None:
            return 1.0
            
        try:
            current_balance = self.wallets.get_total_stake_amount()
            
            if current_balance <= 0:
                return 0.5
                
            # Portfolio return since initialization
            returns = (current_balance - self.initial_balance) / self.initial_balance
            
            # Update peak balance and drawdown tracking
            if self.peak_balance is None or current_balance > self.peak_balance:
                self.peak_balance = current_balance
                self.current_drawdown = 0
            else:
                self.current_drawdown = (self.peak_balance - current_balance) / self.peak_balance
            
            # Convert equity performance into a multiplier
            if returns > 0.5:  # above 50%
                return 1.5
            elif returns > 0.2:  # 20% to 50%
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
        """Return a multiplier based on recent win/loss streaks."""
        if self.consecutive_wins >= 5:
            return 1.4
        elif self.consecutive_wins >= 3:
            return 1.2  # winning streak3-4times
        elif self.consecutive_wins >= 1:
            return 1.1  # winning streak1-2times
        elif self.consecutive_losses >= 5:
            return 0.4
        elif self.consecutive_losses >= 3:
            return 0.6  # losing streak3-4times
        elif self.consecutive_losses >= 1:
            return 0.8  # losing streak1-2times
        else:
            return 1.0
    
    def get_time_session_factor(self, current_time: datetime) -> float:
        """Return a session-based weighting factor."""
        if current_time is None:
            return 1.0
            
        # UTC hour
        hour_utc = current_time.hour
        
        # Session weights
        if 8 <= hour_utc <= 16:  # active session
            return 1.3
        elif 13 <= hour_utc <= 21:  # peak session overlap
            return 1.5
        elif 22 <= hour_utc <= 6:  # quieter session
            return 0.8
        else:
            return 1.0
    
    def get_position_diversity_factor(self) -> float:
        """Return a multiplier based on the number of open trades."""
        try:
            open_trades = Trade.get_open_trades()
            open_count = len(open_trades)
            
            if open_count == 0:
                return 1.0
            elif open_count <= 2:
                return 1.2
            elif open_count <= 5:
                return 1.0  # mid
            elif open_count <= 8:
                return 0.8
            else:
                return 0.6
                
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
    
    # Simplified single-timeframe analysis kept under the old MTF interface
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
        """Fetch a pair dataframe and ensure key indicators are populated."""
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
                # Recalculate indicators if they are missing
                metadata = {'pair': pair}
                dataframe = self.populate_indicators(dataframe, metadata)
                
            return dataframe
            
        except Exception as e:
            logger.error(f"Failed to get indicator dataframe for {pair}: {e}")
            return DataFrame()

    def _safe_series(self, data, length: int, fill_value=0) -> pd.Series:
        """Return a series of the requested length, falling back to a fill value when needed."""
        if isinstance(data, (int, float)):
            return pd.Series([data] * length, index=range(length))
        elif hasattr(data, '__len__') and len(data) == length:
            return pd.Series(data, index=range(length))
        else:
            return pd.Series([fill_value] * length, index=range(length))
    
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Populate the strategy indicator set for the current pair."""

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
        
        # Validate raw market data quality before indicator generation
        data_quality_ok = self.validate_real_data_quality(dataframe, pair)
        if not data_quality_ok:
            logger.warning(f"Data quality warnings detected for {pair}; continuing with indicator calculation")
        
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
        
        # Required orderbook-derived fields
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
        
        # Build orderbook columns in a temporary dict before concatenation
        ob_columns = {}
        for key, default_value in required_ob_fields.items():
            value = orderbook_data.get(key, default_value)
            if isinstance(value, (int, float, np.number)):
                ob_columns[f'ob_{key}'] = value
            else:
                # Fall back to the default value when the orderbook metric is invalid
                ob_columns[f'ob_{key}'] = default_value
        
        # Add orderbook columns in one concat to avoid repeated dataframe mutation
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
        
        # Apply the simplified MTF analysis back onto the dataframe
        dataframe = self.apply_mtf_analysis_to_dataframe(dataframe, mtf_analysis, metadata)
        
        # Composite signal strength
        dataframe['signal_strength'] = self.calculate_enhanced_signal_strength(dataframe)

        # most check
        if dataframe.index.duplicated().any():
            logger.warning(f"most check，at: {pair}")
            dataframe = dataframe[~dataframe.index.duplicated(keep='first')]

        # Return a copy to avoid chained-assignment warnings downstream
        dataframe = dataframe.copy()

        return dataframe
    
    def convert_trend_strength_to_numeric(self, trend_strength):
        """Convert textual trend-strength labels to numeric values."""
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
        """Project simplified multi-timeframe analysis back into the dataframe."""
        
        # === 1. Aggregate MTF trend, strength, and risk scores ===
        mtf_trend_score = 0
        mtf_strength_score = 0
        mtf_risk_score = 0
        
        # Timeframe weights: longer frames carry more influence
        tf_weights = {'1m': 0.1, '15m': 0.15, '1h': 0.25, '4h': 0.3, '1d': 0.2}
        
        for tf, analysis in mtf_analysis.items():
            if tf in tf_weights and analysis:
                weight = tf_weights[tf]
                
                # Trend direction score
                if analysis.get('trend_direction') == 'bullish':
                    mtf_trend_score += weight * 1
                elif analysis.get('trend_direction') == 'bearish':
                    mtf_trend_score -= weight * 1
                
                # Strength score
                trend_strength_raw = analysis.get('trend_strength', 0)
                trend_strength_numeric = self.convert_trend_strength_to_numeric(trend_strength_raw)
                mtf_strength_score += weight * trend_strength_numeric / 100
                
                # Risk score based on RSI extremes
                rsi = analysis.get('rsi', 50)
                if rsi > 70:
                    mtf_risk_score += weight * (rsi - 70) / 30  # overbought risk
                elif rsi < 30:
                    mtf_risk_score -= weight * (30 - rsi) / 30  # oversold
        
        # === 2. Pull support/resistance references from higher timeframes ===
        h1_data = mtf_analysis.get('1h', {})
        h4_data = mtf_analysis.get('4h', {})
        
        # === 3. Build higher-timeframe directional filters ===
        mtf_long_condition = (
            (mtf_trend_score > 0.3) &  # time
            (mtf_risk_score > -0.5)    # risk
        )
        
        mtf_short_condition = (
            (mtf_trend_score < -0.3) &  # time
            (mtf_risk_score < 0.5)     # risk
        )
        
        # === 4. Strong directional confirmation from 4h + 1d ===
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
        
        # Default higher-timeframe support/resistance levels when unavailable
        h1_support = h1_data.get('support_level', dataframe['close'] * 0.99)
        h1_resistance = h1_data.get('resistance_level', dataframe['close'] * 1.01)
        h4_support = h4_data.get('support_level', dataframe['close'] * 0.98)
        h4_resistance = h4_data.get('resistance_level', dataframe['close'] * 1.02)
        
        mtf_columns = {
            # Score indicators
            'mtf_trend_score': mtf_trend_score,  # [-1, 1]
            'mtf_strength_score': mtf_strength_score,  # [0, 1]
            'mtf_risk_score': mtf_risk_score,  # [-1, 1]
            
            # price
            'h1_support': h1_support,
            'h1_resistance': h1_resistance,
            'h4_support': h4_support,
            'h4_resistance': h4_resistance,
            
            # Proximity to higher-timeframe levels
            'near_h1_support': (abs(dataframe['close'] - h1_support) / dataframe['close'] < 0.005).astype(int),
            'near_h1_resistance': (abs(dataframe['close'] - h1_resistance) / dataframe['close'] < 0.005).astype(int),
            'near_h4_support': (abs(dataframe['close'] - h4_support) / dataframe['close'] < 0.01).astype(int),
            'near_h4_resistance': (abs(dataframe['close'] - h4_resistance) / dataframe['close'] < 0.01).astype(int),
            
            # Directional filters
            'mtf_long_filter': self._safe_series(1 if mtf_long_condition else 0, len(dataframe)),
            'mtf_short_filter': self._safe_series(1 if mtf_short_condition else 0, len(dataframe)),
            
            # Strong confirmation filters
            'mtf_strong_bull': self._safe_series(1 if mtf_strong_bull_condition else 0, len(dataframe)),
            'mtf_strong_bear': self._safe_series(1 if mtf_strong_bear_condition else 0, len(dataframe))
        }
        
        # Add the generated MTF columns in one batch
        if mtf_columns:
            # Normalize series-like values before dataframe construction
            processed_columns = {}
            for col_name, value in mtf_columns.items():
                if isinstance(value, pd.Series):
                    # Align the series length with the dataframe when possible
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
        """Populate long and short entry signals."""
        
        pair = metadata['pair']
        
        # === Price position within the recent 20-candle range ===
        highest_20 = dataframe['high'].rolling(20).max()
        lowest_20 = dataframe['low'].rolling(20).min()
        price_position = (dataframe['close'] - lowest_20) / (highest_20 - lowest_20 + 0.0001)
        
        # Avoid chasing extreme range edges
        not_at_top = price_position < 0.80
        not_at_bottom = price_position > 0.20
        
        # === Momentum and participation filters ===
        # Allow RSI pullbacks without requiring perfect upward momentum
        rsi_momentum_strong = (
            (dataframe['rsi_14'] - dataframe['rsi_14'].shift(3) > -10) &
            (dataframe['rsi_14'] < 80) & (dataframe['rsi_14'] > 20)
        )
        
        # Volume support
        volume_support = (
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 0.6) &
            (dataframe['volume'] > dataframe['volume'].shift(1) * 0.7)
        )
        
        # Basic fake-breakout filter
        no_fake_breakout = ~(
            ((dataframe['high'] - dataframe['close']) > (dataframe['close'] - dataframe['open']) * 3) |
            ((dataframe['open'] - dataframe['low']) > (dataframe['close'] - dataframe['open']) * 3)
        )
        
        # Trend/range state
        is_trending = dataframe['adx'] > 20
        is_sideways = dataframe['adx'] < 20
        
        # Allow sideways conditions only when volatility is still meaningful
        sideways_filter = ~is_sideways | (dataframe['atr_p'] > 0.02)
        
        # Baseline environment filter shared by long and short entries
        basic_env = (
            (dataframe['volume_ratio'] > 0.8) &
            (dataframe['atr_p'] > 0.001) &
            sideways_filter &
            rsi_momentum_strong &
            volume_support
        )
        
        # Long environment: avoid hostile trend and sentiment extremes
        long_favourable_environment = (
            basic_env &
            (dataframe['trend_strength'] > -40) &
            (dataframe.get('market_sentiment', 0) > -0.8) &
            (dataframe['rsi_14'] > 25)
        )
        
        # Short environment: avoid hostile trend and sentiment extremes
        short_favourable_environment = (
            basic_env &
            (dataframe['trend_strength'] < 40) &
            (dataframe.get('market_sentiment', 0) < 0.8) &
            (dataframe['rsi_14'] < 75)
        )
        
        # === 🌍 market state ===
        market_regime_data = self._enhanced_market_regime_detection(dataframe)
        current_regime = market_regime_data['regime']
        regime_confidence = market_regime_data['confidence']
        signals_advice = market_regime_data['signals_advice']
        
        # Persist the detected market regime into the dataframe
        dataframe.loc[:, 'market_regime'] = current_regime
        dataframe.loc[:, 'regime_confidence'] = regime_confidence
        
        logger.info(
            f"📊 Market regime {metadata.get('pair', '')}: "
            f"{current_regime} (confidence: {regime_confidence:.1%}) | "
            f"recommended signals: {signals_advice.get('recommended_signals', [])} | "
            f"avoid signals: {signals_advice.get('avoid_signals', [])}"
        )
        
        # === Long signal set ===
        
        # Signal 1: RSI oversold bounce
        # Dynamic oversold threshold lowers in higher-volatility regimes
        base_oversold = 30
        volatility_percentile = dataframe['atr_p'].rolling(50).rank(pct=True)
        dynamic_oversold = base_oversold - (volatility_percentile * 8)  # 20-30 range
        
        # Confirmation layers
        rsi_condition = (dataframe['rsi_14'] < dynamic_oversold)
        rsi_momentum = (dataframe['rsi_14'] > dataframe['rsi_14'].shift(2))
        price_confirmation = (dataframe['close'] > dataframe['close'].shift(1))
        
        # Trend confirmation: either aligned bullishly or not in a strong trend
        trend_confirmation = (
            (dataframe['ema_8'] >= dataframe['ema_21']) |
            (dataframe['adx'] < 25)
        )
        
        # Volume confirmation
        volume_confirmation = (
            dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.1
        )
        
        # Strength confirmation
        strength_confirmation = (
            (dataframe['adx'] > 20) &
            (dataframe['adx'] > dataframe['adx'].shift(2))
        )
        
        # Avoid entries during bearish divergence
        no_bearish_divergence = ~dataframe.get('bearish_divergence', False).astype(bool)
        
        rsi_oversold_bounce = (
            rsi_condition &
            rsi_momentum &
            price_confirmation &
            trend_confirmation &
            volume_confirmation &
            strength_confirmation &
            no_bearish_divergence &
            not_at_top &
            basic_env
        )
        dataframe.loc[rsi_oversold_bounce, 'enter_long'] = 1
        dataframe.loc[rsi_oversold_bounce, 'enter_tag'] = 'RSI_Oversold_Bounce'
        
        # Signal 2: EMA golden cross
        ema_golden_cross = (
            (dataframe['ema_8'] > dataframe['ema_21']) &
            (dataframe['ema_8'].shift(3) <= dataframe['ema_21'].shift(3)) &
            (dataframe['close'] <= dataframe['ema_8'] * 1.01) &
            (dataframe['close'] > dataframe['ema_21']) &
            (dataframe['volume_ratio'] > 1.0) &
            (dataframe['momentum_exhaustion_score'] < 0.5) &
            (dataframe['trend_phase'] <= 2) &
            (~dataframe['bearish_divergence'].astype(bool)) &
            basic_env
        )
        dataframe.loc[ema_golden_cross, 'enter_long'] = 1
        dataframe.loc[ema_golden_cross, 'enter_tag'] = 'EMA_Golden_Cross'
        
        # Signal 3: MACD bullish turn
        macd_bullish = (
            (
                ((dataframe['macd'] > dataframe['macd_signal']) & 
                 (dataframe['macd'].shift(1) <= dataframe['macd_signal'].shift(1))) |
                ((dataframe['macd_hist'] > 0) & 
                 (dataframe['macd_hist'].shift(1) <= 0))
            ) &
            basic_env
        )
        dataframe.loc[macd_bullish, 'enter_long'] = 1
        dataframe.loc[macd_bullish, 'enter_tag'] = 'MACD_Bullish'
        
        # Signal 4: Bollinger lower-band bounce
        bb_lower_bounce = (
            (dataframe['close'] <= dataframe['bb_lower'] * 1.005) &
            (dataframe['close'] > dataframe['close'].shift(1)) &
            (dataframe['close'].shift(1) > dataframe['close'].shift(2)) &
            (dataframe['rsi_14'] < 50) &
            (dataframe['rsi_14'] > dataframe['rsi_14'].shift(1)) &
            (dataframe['volume_ratio'] > 1.1) &
            not_at_top &
            no_fake_breakout &
            basic_env
        )
        dataframe.loc[bb_lower_bounce, 'enter_long'] = 1
        dataframe.loc[bb_lower_bounce, 'enter_tag'] = 'BB_Lower_Bounce'
        
        # Additional breakout-style signals are handled by the expanded legacy block below
        
        # === Short signal set ===
        
        # Signal 1: RSI overbought fade
        base_overbought = 70
        volatility_percentile = dataframe['atr_p'].rolling(50).rank(pct=True)
        dynamic_overbought = base_overbought + (volatility_percentile * 8)  # 70-78 range
        
        # Confirmation layers
        rsi_condition = (dataframe['rsi_14'] > dynamic_overbought)
        rsi_momentum = (dataframe['rsi_14'] < dataframe['rsi_14'].shift(2))
        price_confirmation = (dataframe['close'] < dataframe['close'].shift(1))
        
        # Trend confirmation: either aligned bearishly or not in a strong trend
        trend_confirmation = (
            (dataframe['ema_8'] <= dataframe['ema_21']) |
            (dataframe['adx'] < 25)
        )
        
        # Volume confirmation
        volume_confirmation = (
            dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.1
        )
        
        # Strength confirmation
        strength_confirmation = (
            (dataframe['adx'] > 20) &
            (dataframe['adx'] > dataframe['adx'].shift(2))
        )
        
        # Avoid entries during bullish divergence
        no_bullish_divergence = ~dataframe.get('bullish_divergence', False).astype(bool)
        
        rsi_overbought_fall = (
            rsi_condition &
            rsi_momentum &
            price_confirmation &
            trend_confirmation &
            volume_confirmation &
            strength_confirmation &
            no_bullish_divergence &
            not_at_bottom &
            basic_env
        )
        # Signal-quality scores
        rsi_long_score = self._calculate_signal_quality_score(
            dataframe, rsi_oversold_bounce, 'RSI_Oversold_Bounce'
        )
        rsi_short_score = self._calculate_signal_quality_score(
            dataframe, rsi_overbought_fall, 'RSI_Overbought_Fall'
        )
        
        # Apply market-regime preferences to the RSI signals
        
        # Long RSI signal
        rsi_long_regime_ok = 'RSI_Oversold_Bounce' not in signals_advice.get('avoid_signals', [])
        high_quality_long = rsi_oversold_bounce & (rsi_long_score >= 6) & rsi_long_regime_ok
        
        # Short RSI signal
        rsi_short_regime_ok = 'RSI_Overbought_Fall' not in signals_advice.get('avoid_signals', [])
        high_quality_short = rsi_overbought_fall & (rsi_short_score >= 6) & rsi_short_regime_ok
        
        # Give a small score concession to regime-favored signals
        if 'RSI_Oversold_Bounce' in signals_advice.get('recommended_signals', []):
            regime_bonus_long = rsi_oversold_bounce & (rsi_long_score >= 5)
            high_quality_long = high_quality_long | regime_bonus_long
            
        if 'RSI_Overbought_Fall' in signals_advice.get('recommended_signals', []):
            regime_bonus_short = rsi_overbought_fall & (rsi_short_score >= 5)
            high_quality_short = high_quality_short | regime_bonus_short
        
        dataframe.loc[high_quality_long, 'enter_long'] = 1
        dataframe.loc[high_quality_long, 'enter_tag'] = 'RSI_Oversold_Bounce'
        dataframe.loc[high_quality_long, 'signal_quality'] = rsi_long_score
        dataframe.loc[high_quality_long, 'market_regime_bonus'] = 'RSI_Oversold_Bounce' in signals_advice.get('recommended_signals', [])
        
        dataframe.loc[high_quality_short, 'enter_short'] = 1
        dataframe.loc[high_quality_short, 'enter_tag'] = 'RSI_Overbought_Fall'
        dataframe.loc[high_quality_short, 'signal_quality'] = rsi_short_score
        dataframe.loc[high_quality_short, 'market_regime_bonus'] = 'RSI_Overbought_Fall' in signals_advice.get('recommended_signals', [])
        
        # Signal 2: EMA death cross
        ema_death_cross = (
            (dataframe['ema_8'] < dataframe['ema_21']) &
            (dataframe['ema_8'].shift(3) >= dataframe['ema_21'].shift(3)) &
            (dataframe['close'] >= dataframe['ema_8'] * 0.99) &
            (dataframe['close'] < dataframe['ema_21']) &
            (dataframe['volume_ratio'] > 1.0) &
            (dataframe['momentum_exhaustion_score'] < 0.5) &
            (dataframe['trend_phase'] <= 2) &
            (~dataframe['bullish_divergence'].astype(bool)) &
            basic_env
        )
        dataframe.loc[ema_death_cross, 'enter_short'] = 1
        dataframe.loc[ema_death_cross, 'enter_tag'] = 'EMA_Death_Cross'
        
        # Signal 3: MACD bearish turn
        macd_death_cross = (
            (dataframe['macd'] < dataframe['macd_signal']) & 
            (dataframe['macd'].shift(1) >= dataframe['macd_signal'].shift(1))
        )
        macd_hist_negative = (
            (dataframe['macd_hist'] < 0) & 
            (dataframe['macd_hist'].shift(1) >= 0)
        )
        macd_basic_signal = macd_death_cross | macd_hist_negative
        
        # Guardrails for short-side MACD entries
        
        # 1. Trend confirmation: avoid shorting into bullish structure
        trend_bearish = (
            (dataframe['ema_8'] < dataframe['ema_21']) &
            (dataframe['ema_21'] < dataframe['ema_50']) &
            (dataframe['close'] < dataframe['ema_21'])
        )
        
        # 2. Momentum confirmation
        momentum_confirmation = (
            (dataframe['rsi_14'] < 55) &
            (dataframe['rsi_14'] < dataframe['rsi_14'].shift(2)) &
            (dataframe['close'] < dataframe['close'].shift(2))
        )
        
        # 3. Volume confirmation
        volume_confirmation = (
            (dataframe['volume'] > dataframe['volume'].rolling(20).mean() * 1.1) &
            (dataframe['volume'] > dataframe['volume'].shift(1))
        )
        
        # 4. Strength confirmation
        strength_confirmation = (
            (dataframe['adx'] > 25) &
            (dataframe['adx'] > dataframe['adx'].shift(3))
        )
        
        # 5. Avoid sideways markets
        not_sideways = (dataframe['adx'] > 20)
        
        # 6. Confirm price is still extended enough to short
        position_confirmation = (
            dataframe['close'] > dataframe['close'].rolling(20).mean() * 1.02
        )
        
        # 7. Avoid shorting into bullish divergence
        no_bullish_divergence = ~dataframe.get('bullish_divergence', False).astype(bool)
        
        # Final MACD bearish entry
        macd_bearish = (
            macd_basic_signal &
            trend_bearish &
            momentum_confirmation &
            volume_confirmation &
            strength_confirmation &
            not_sideways &
            position_confirmation &
            no_bullish_divergence &
            not_at_bottom &
            basic_env
        )
        
        # MACD quality score
        macd_score = self._calculate_macd_signal_quality(dataframe, macd_bearish, 'MACD_Bearish')
        
        # Market-regime filter for MACD bearish entries
        macd_regime_ok = 'MACD_Bearish' not in signals_advice.get('avoid_signals', [])
        high_quality_macd = macd_bearish & (macd_score >= 7) & macd_regime_ok
        
        # Small score concession if the regime explicitly favors this signal
        if 'MACD_Bearish' in signals_advice.get('recommended_signals', []):
            regime_bonus_macd = macd_bearish & (macd_score >= 6) & macd_regime_ok
            high_quality_macd = high_quality_macd | regime_bonus_macd
        
        dataframe.loc[high_quality_macd, 'enter_short'] = 1
        dataframe.loc[high_quality_macd, 'enter_tag'] = 'MACD_Bearish'
        dataframe.loc[high_quality_macd, 'signal_quality'] = macd_score
        dataframe.loc[high_quality_macd, 'market_regime_bonus'] = 'MACD_Bearish' in signals_advice.get('recommended_signals', [])
        
        # Signal 4: Bollinger upper-band rejection
        bb_upper_rejection = (
            (dataframe['close'] >= dataframe['bb_upper'] * 0.995) &
            (dataframe['close'] < dataframe['close'].shift(1)) &
            (dataframe['rsi_14'] > 50) &
            (dataframe['volume_ratio'] > 1.1) &
            basic_env
        )
        dataframe.loc[bb_upper_rejection, 'enter_short'] = 1
        dataframe.loc[bb_upper_rejection, 'enter_tag'] = 'BB_Upper_Rejection'
        
        # Additional short breakdown-style signals are handled by the expanded legacy block below
        
        # ==============================
        # Signal-derived sizing overlays
        # ==============================
        
        # Aggregate signal quality metrics
        dataframe['signal_quality_score'] = self._calculate_signal_quality(dataframe)
        dataframe['position_weight'] = self._calculate_position_weight(dataframe)
        dataframe['leverage_multiplier'] = self._calculate_leverage_multiplier(dataframe)
        
        # Signal counts
        total_long_signals = dataframe['enter_long'].sum()
        total_short_signals = dataframe['enter_short'].sum()
        
        # Environment pass rates
        env_basic_rate = basic_env.sum() / len(dataframe) * 100
        env_long_rate = long_favourable_environment.sum() / len(dataframe) * 100  
        env_short_rate = short_favourable_environment.sum() / len(dataframe) * 100
        
        # Entry summary
        if total_long_signals > 0 or total_short_signals > 0:
            logger.info(f"""
🔥 Entry summary - {metadata['pair']}:
📊 Signals:
   └─ Long signals: {total_long_signals}
   └─ Short signals: {total_short_signals}
   └─ Total signals: {total_long_signals + total_short_signals}

🌍 Environment pass rates:
   └─ Basic environment: {env_basic_rate:.1f}%
   └─ Long environment: {env_long_rate:.1f}%
   └─ Short environment: {env_short_rate:.1f}%

✅ Entry filters active and producing signals
""")
        
        # No-signal diagnostic
        if total_long_signals == 0 and total_short_signals == 0:
            logger.warning(f"""
⚠️ No entry signals - {metadata['pair']}:
🔍 Rejection breakdown:
   └─ Basic environment rejected: {100-env_basic_rate:.1f}%
   └─ Long environment rejected: {100-env_long_rate:.1f}%
   └─ Short environment rejected: {100-env_short_rate:.1f}%
   
💡 Check RSI ({dataframe['rsi_14'].iloc[-1]:.1f}) and trend strength ({dataframe.get('trend_strength', [0]).iloc[-1]:.1f})
""")
        
        return dataframe
    
    def _legacy_populate_entry_trend_backup(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        """Legacy backup entry-signal implementation."""
        
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
🎯 Signal overview - {pair}:
{'='*60}
📊 Signal counts:
├─ Bullish signals: {total_long_signals}
├─ Bearish signals: {total_short_signals}
├─ Balance ratio: {signal_balance_ratio:.2f} {'✅ balanced' if 0.5 <= signal_balance_ratio <= 2.0 else '⚠️ skewed'}
└─ Signal breakdown: {signal_counts if signal_counts else 'no active signals'}

📈 Market context:
├─ 20-period price percentile: {price_percentile_20.iloc[-1] if len(price_percentile_20) > 0 else 0:.1%} (50-period: {price_percentile_50.iloc[-1] if len(price_percentile_50) > 0 else 0:.1%})
├─ Bullish location bias: {'✅ favorable' if (price_percentile_20.iloc[-1] if len(price_percentile_20) > 0 else 0) < 0.55 else '❌ stretched'}
├─ Bearish location bias: {'✅ favorable' if (price_percentile_20.iloc[-1] if len(price_percentile_20) > 0 else 0) > 0.45 else '❌ compressed'}
├─ RSI: {dataframe['rsi_14'].iloc[-1] if 'rsi_14' in dataframe.columns and len(dataframe) > 0 else 50:.1f}
├─ ADX trend strength: {dataframe['adx'].iloc[-1] if 'adx' in dataframe.columns and len(dataframe) > 0 else 25:.1f}
├─ Volume ratio: {dataframe['volume_ratio'].iloc[-1] if 'volume_ratio' in dataframe.columns and len(dataframe) > 0 else 1:.2f}x
├─ Trend score: {dataframe['trend_strength'].iloc[-1] if 'trend_strength' in dataframe.columns and len(dataframe) > 0 else 50:.0f}/100
├─ Momentum score: {dataframe['momentum_score'].iloc[-1] if 'momentum_score' in dataframe.columns and len(dataframe) > 0 else 0:.3f}
├─ Market sentiment: {dataframe['market_sentiment'].iloc[-1] if 'market_sentiment' in dataframe.columns and len(dataframe) > 0 else 0:.3f}
└─ Divergence strength: {dataframe['rsi_divergence_strength'].iloc[-1] if 'rsi_divergence_strength' in dataframe.columns and len(dataframe) > 0 else 0:.3f}

🎯 Predictive-signal notes:
├─ Bullish side includes high-conviction divergence, momentum, reversal, and MTF setups
├─ Bearish side includes the mirrored high-conviction divergence, momentum, reversal, and MTF setups
└─ This summary compares the broader bullish and bearish signal families
{'='*60}
""")
        
        return dataframe
    
    def _log_enhanced_entry_decision(self, pair: str, dataframe: DataFrame, current_data, direction: str):
        """Log detailed reasoning for an entry decision."""
        
        # Current entry tag
        entry_tag = current_data.get('enter_tag', 'UNKNOWN_SIGNAL')
        
        # signal
        signal_explanations = {
            'GOLDEN_CROSS_BREAKOUT': 'breakout - EMA8upEMA21，confirm rise trend',
            'MACD_MOMENTUM_CONFIRMED': 'MACDmomentum confirm - MACDand long，momentum',
            'OVERSOLD_SUPPORT_BOUNCE': 'oversold support - RSIoversold after，support confirm has',
            'BREAKOUT_RETEST_HOLD': 'breakout confirm - breakout after not，trend',
            'INSTITUTIONAL_ACCUMULATION': 'institutional-style accumulation with volume and orderbook support',
            'DEATH_CROSS_BREAKDOWN': 'EMA8 crossed below EMA21 with bearish trend confirmation',
            'MACD_MOMENTUM_BEARISH': 'MACD bearish momentum confirmation',
            'OVERBOUGHT_RESISTANCE_REJECT': 'overbought rejection from resistance',
            'BREAKDOWN_RETEST_FAIL': 'breakdown retest failed at resistance',
            'INSTITUTIONAL_DISTRIBUTION': 'institutional-style distribution with selling pressure'
        }
        
        signal_type = signal_explanations.get(entry_tag, f'signal confirm - {entry_tag}')
        
        # Normalized technical snapshot used for decision explanations
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
        
        # Human-readable reasoning string
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
        
        # Reserved for integration with an external decision logger
        pass
    
    def _build_entry_reasoning(self, entry_tag: str, tech: dict, direction: str) -> str:
        """Build a concise human-readable explanation for the entry signal."""
        
        reasoning_templates = {
            'GOLDEN_CROSS_BREAKOUT': f"EMA8 ({tech['ema_8']:.2f}) crossed above EMA21 ({tech['ema_21']:.2f}), price is above EMA50 ({tech['ema_50']:.2f}), ADX ({tech['adx']:.1f}) confirms trend strength, and volume ratio ({tech['volume_ratio']:.1f}) supports the breakout",
            
            'MACD_MOMENTUM_CONFIRMED': f"MACD ({tech['macd']:.4f}) is above the signal line ({tech['macd_signal']:.4f}), histogram ({tech['macd_hist']:.4f}) is positive, momentum score ({tech['momentum_score']:.3f}) is improving, and price is holding above VWAP",
            
            'OVERSOLD_SUPPORT_BOUNCE': f"RSI ({tech['rsi_14']:.1f}) is rebounding from oversold, Bollinger position ({tech['bb_position']:.2f}) indicates lower-range support, volume ratio ({tech['volume_ratio']:.1f}) confirms interest, and orderbook imbalance ({tech['ob_depth_imbalance']:.2f}) is supportive",
            
            'BREAKOUT_RETEST_HOLD': f"Price held a breakout above the key trend baseline, EMA21 is acting as support, ADX ({tech['adx']:.1f}) confirms the trend, volatility remains controlled, and volume ratio ({tech['volume_ratio']:.1f}) supports continuation",
            
            'INSTITUTIONAL_ACCUMULATION': f"Orderbook imbalance ({tech['ob_depth_imbalance']:.2f}) suggests accumulation, volume ratio ({tech['volume_ratio']:.1f}) shows elevated activity, price is above VWAP, and trend strength ({tech['trend_strength']:.0f}) supports continuation",
            
            'DEATH_CROSS_BREAKDOWN': f"EMA8 ({tech['ema_8']:.2f}) crossed below EMA21 ({tech['ema_21']:.2f}), price is below EMA50 ({tech['ema_50']:.2f}), ADX ({tech['adx']:.1f}) confirms bearish trend strength, and volume ratio ({tech['volume_ratio']:.1f}) supports the move",
            
            'MACD_MOMENTUM_BEARISH': f"MACD ({tech['macd']:.4f}) is below the signal line ({tech['macd_signal']:.4f}), histogram ({tech['macd_hist']:.4f}) is weakening, momentum score ({tech['momentum_score']:.3f}) is falling, and price remains below VWAP",
            
            'OVERBOUGHT_RESISTANCE_REJECT': f"RSI ({tech['rsi_14']:.1f}) is fading from overbought, Bollinger position ({tech['bb_position']:.2f}) is near the upper band, volume ratio ({tech['volume_ratio']:.1f}) confirms participation, and resistance is holding",
            
            'BREAKDOWN_RETEST_FAIL': f"Price failed a retest after breaking down, EMA21 is acting as resistance, ADX ({tech['adx']:.1f}) confirms bearish trend strength, and volume ratio ({tech['volume_ratio']:.1f}) supports continuation",
            
            'INSTITUTIONAL_DISTRIBUTION': f"Orderbook imbalance ({tech['ob_depth_imbalance']:.2f}) suggests distribution, volume ratio ({tech['volume_ratio']:.1f}) shows elevated activity, price is below VWAP, and trend strength ({tech['trend_strength']:.0f}) supports downside follow-through"
        }
        
        return reasoning_templates.get(entry_tag, f"{entry_tag} signal confirmed with {direction.lower()}-side indicator alignment")
    
    def _assess_entry_risk_level(self, tech: dict) -> str:
        """Assess the qualitative risk level of a proposed entry."""
        risk_score = 0
        
        # ADX contribution
        if tech['adx'] > 30:
            risk_score += 1
        elif tech['adx'] < 20:
            risk_score -= 1
            
        # Volume contribution
        if tech['volume_ratio'] > 1.5:
            risk_score += 1
        elif tech['volume_ratio'] < 0.8:
            risk_score -= 1
            
        # Market-quality contribution
        if tech['ob_market_quality'] > 0.6:
            risk_score += 1
        elif tech['ob_market_quality'] < 0.3:
            risk_score -= 1
            
        # Avoid extreme RSI conditions
        if 25 < tech['rsi_14'] < 75:
            risk_score += 1
        else:
            risk_score -= 1
        
        if risk_score >= 2:
            return "low risk"
        elif risk_score >= 0:
            return "medium risk"
        else:
            return "high risk"
    
    def _log_short_entry_decision(self, pair: str, dataframe: DataFrame, current_data):
        """Record short-entry decision context."""
        
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
        """Initialize the portfolio risk-control system."""
        # Core risk-control state
        self.risk_control_enabled = True
        self.emergency_mode = False
        self.circuit_breaker_active = False
        
        # Risk-budget configuration
        self.risk_budgets = {
            'daily_var_budget': 0.02,      # 2% daily VaR budget
            'weekly_var_budget': 0.05,     # 5% weekly VaR budget
            'monthly_var_budget': 0.12,    # 12% monthly VaR budget
            'position_var_limit': 0.01,    # 1% per-position VaR limit
            'correlation_limit': 0.7,      # 70% correlation limit
            'sector_exposure_limit': 0.3   # 30% sector exposure limit
        }
        
        # Current risk-budget utilization
        self.risk_utilization = {
            'current_daily_var': 0.0,
            'current_weekly_var': 0.0,
            'current_monthly_var': 0.0,
            'used_correlation_capacity': 0.0,
            'sector_exposures': {}
        }
        
        # Circuit-breaker thresholds
        self.circuit_breakers = {
            'daily_loss_limit': -0.08,      # day loss8%
            'hourly_loss_limit': -0.03,     # hours loss3%
            'consecutive_loss_limit': 6,     # loss
            'drawdown_limit': -0.20,        # most large drawdown20%
            'volatility_spike_limit': 5.0,  # volatility
            'correlation_spike_limit': 0.9  # extreme correlation spike
        }
        
        # Risk-event history
        self.risk_events = []
        self.emergency_actions = []
        
        # Risk-check cadence
        self.last_risk_check_time = datetime.now(timezone.utc)
        self.risk_check_interval = 60  # check every 60 seconds
        
    def comprehensive_risk_check(self, pair: str, current_price: float, 
                               proposed_position_size: float, 
                               proposed_leverage: int) -> Dict[str, Any]:
        """Run the full pre-trade risk-control pipeline."""
        
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
            
            # 1. Circuit-breaker check
            circuit_breaker_result = self.check_circuit_breakers()
            if circuit_breaker_result['triggered']:
                risk_status['approved'] = False
                risk_status['emergency_action'] = 'circuit_breaker_halt'
                risk_status['risk_violations'].append(circuit_breaker_result)
                return risk_status
            
            # 2. VaR budget check
            var_check_result = self.check_var_budget_limits(pair, proposed_position_size)
            if not var_check_result['within_limits']:
                risk_status['adjusted_position_size'] *= var_check_result['adjustment_factor']
                risk_status['risk_warnings'].append(var_check_result)
            
            # 3. Correlation check
            correlation_check_result = self.check_correlation_limits(pair, proposed_position_size)
            if not correlation_check_result['within_limits']:
                risk_status['adjusted_position_size'] *= correlation_check_result['adjustment_factor']
                risk_status['risk_warnings'].append(correlation_check_result)
            
            # 4. Concentration-risk check
            concentration_check_result = self.check_concentration_risk(pair, proposed_position_size)
            if not concentration_check_result['within_limits']:
                risk_status['adjusted_position_size'] *= concentration_check_result['adjustment_factor']
                risk_status['risk_warnings'].append(concentration_check_result)
            
            # 5. Liquidity-risk check
            liquidity_check_result = self.check_liquidity_risk(pair, proposed_position_size)
            if not liquidity_check_result['sufficient_liquidity']:
                risk_status['adjusted_position_size'] *= liquidity_check_result['adjustment_factor']
                risk_status['risk_warnings'].append(liquidity_check_result)
            
            # 6. leverage risk check
            leverage_check_result = self.check_leverage_risk(pair, proposed_leverage)
            if not leverage_check_result['within_limits']:
                risk_status['adjusted_leverage'] = leverage_check_result['max_allowed_leverage']
                risk_status['risk_warnings'].append(leverage_check_result)
            
            # 7. Time-based risk check
            time_risk_result = self.check_time_based_risk(current_time)
            if time_risk_result['high_risk_period']:
                risk_status['adjusted_position_size'] *= time_risk_result['adjustment_factor']
                risk_status['risk_warnings'].append(time_risk_result)
            
            # Clamp the final adjusted position size
            risk_status['adjusted_position_size'] = max(
                0.005, 
                min(risk_status['adjusted_position_size'], self.max_position_size * 0.8)
            )
            
            # Record the risk-check result
            self.record_risk_event('risk_check', risk_status)
            
        except Exception as e:
            risk_status['approved'] = False
            risk_status['emergency_action'] = 'system_error'
            risk_status['risk_violations'].append({
                'type': 'system_error',
                    'message': f'Risk-check system error: {str(e)}'
                })
        
        return risk_status
    
    def check_circuit_breakers(self) -> Dict[str, Any]:
        """Check whether any circuit breaker should halt trading."""
        try:
            current_time = datetime.now(timezone.utc)
            
            # Current equity and PnL snapshots
            current_equity = getattr(self, 'current_equity', 100000)  # default value
            daily_pnl = getattr(self, 'daily_pnl', 0)
            hourly_pnl = getattr(self, 'hourly_pnl', 0)
            
            # 1. Daily loss
            daily_loss_pct = daily_pnl / current_equity if current_equity > 0 else 0
            if daily_loss_pct < self.circuit_breakers['daily_loss_limit']:
                return {
                    'triggered': True,
                    'type': 'daily_loss_circuit_breaker',
                    'current_value': daily_loss_pct,
                    'limit': self.circuit_breakers['daily_loss_limit'],
                        'message': f'daily loss limit hit: {daily_loss_pct:.2%}'
                    }
            
            # 2. Hourly loss
            hourly_loss_pct = hourly_pnl / current_equity if current_equity > 0 else 0
            if hourly_loss_pct < self.circuit_breakers['hourly_loss_limit']:
                return {
                    'triggered': True,
                    'type': 'hourly_loss_circuit_breaker',
                    'current_value': hourly_loss_pct,
                    'limit': self.circuit_breakers['hourly_loss_limit'],
                        'message': f'hourly loss limit hit: {hourly_loss_pct:.2%}'
                    }
            
            # 3. Consecutive losses
            if self.consecutive_losses >= self.circuit_breakers['consecutive_loss_limit']:
                return {
                    'triggered': True,
                    'type': 'consecutive_loss_circuit_breaker',
                    'current_value': self.consecutive_losses,
                    'limit': self.circuit_breakers['consecutive_loss_limit'],
                        'message': f'consecutive loss limit hit: {self.consecutive_losses} losses'
                    }
            
            # 4. Maximum drawdown
            max_drawdown = getattr(self, 'current_max_drawdown', 0)
            if max_drawdown < self.circuit_breakers['drawdown_limit']:
                return {
                    'triggered': True,
                    'type': 'drawdown_circuit_breaker',
                    'current_value': max_drawdown,
                    'limit': self.circuit_breakers['drawdown_limit'],
                        'message': f'drawdown limit hit: {max_drawdown:.2%}'
                    }
            
            return {'triggered': False, 'type': None, 'message': 'no circuit breakers triggered'}
            
        except Exception:
            return {
                'triggered': True,
                'type': 'circuit_breaker_error',
                'message': 'circuit-breaker check failed'
            }
    
    def check_var_budget_limits(self, pair: str, position_size: float) -> Dict[str, Any]:
        """Check the proposed position against VaR budgets."""
        try:
            # Estimate VaR contribution of the proposed position
            position_var = self.calculate_position_var(pair, position_size)
            
            # Project updated daily VaR utilization
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
                    'message': f'VaR budget exceeded; reduce position to {adjustment_factor:.1%} of proposed size'
                }
            
            return {
                'within_limits': True,
                'type': 'var_budget_check',
                'utilization': new_daily_var / self.risk_budgets['daily_var_budget'],
                'message': 'VaR budget check passed'
            }
            
        except Exception:
            return {
                'within_limits': False,
                'adjustment_factor': 0.5,
                'message': 'VaR budget check failed; applying defensive sizing'
            }
    
    def calculate_position_var(self, pair: str, position_size: float) -> float:
        """Estimate VaR for the proposed position size."""
        try:
            if pair in self.pair_returns_history and len(self.pair_returns_history[pair]) >= 20:
                returns = self.pair_returns_history[pair]
                position_var = self.calculate_var(returns) * position_size
                return min(position_var, self.risk_budgets['position_var_limit'])
            else:
                # Default conservative VaR estimate
                return position_size * 0.02
        except Exception:
            return position_size * 0.03
    
    def check_correlation_limits(self, pair: str, position_size: float) -> Dict[str, Any]:
        """Check whether the pair breaches portfolio-correlation limits."""
        try:
            current_correlation = self.calculate_portfolio_correlation(pair)
            
            if current_correlation > self.risk_budgets['correlation_limit']:
                # Scale position size down as correlation increases beyond the limit
                excess_correlation = current_correlation - self.risk_budgets['correlation_limit']
                adjustment_factor = max(0.2, 1 - (excess_correlation * 2))
                
                return {
                    'within_limits': False,
                    'type': 'correlation_limit_exceeded',
                    'adjustment_factor': adjustment_factor,
                    'current_correlation': current_correlation,
                    'limit': self.risk_budgets['correlation_limit'],
                    'message': f'Correlation too high ({current_correlation:.1%}); reduce position to {adjustment_factor:.1%} of proposed size'
                }
            
            return {
                'within_limits': True,
                'type': 'correlation_check',
                'current_correlation': current_correlation,
                'message': 'correlation check passed'
            }
            
        except Exception:
            return {
                'within_limits': False,
                'adjustment_factor': 0.7,
                'message': 'correlation check failed; applying defensive sizing'
            }
    
    def check_concentration_risk(self, pair: str, position_size: float) -> Dict[str, Any]:
        """Check concentration risk for the proposed position."""
        try:
            # Current portfolio exposure
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
            
            max_single_position_ratio = 0.4  # max 40% concentration
            
            if concentration_ratio > max_single_position_ratio:
                adjustment_factor = max_single_position_ratio / concentration_ratio
                
                return {
                    'within_limits': False,
                    'type': 'concentration_risk_exceeded',
                    'adjustment_factor': adjustment_factor,
                    'concentration_ratio': concentration_ratio,
                    'limit': max_single_position_ratio,
                    'message': f'concentration too high ({concentration_ratio:.1%}); reduce position size'
                }
            
            return {
                'within_limits': True,
                'type': 'concentration_check',
                'concentration_ratio': concentration_ratio,
                'message': 'concentration check passed'
            }
            
        except Exception:
            return {
                'within_limits': False,
                'adjustment_factor': 0.6,
                'message': 'concentration check failed; applying defensive sizing'
            }
    
    def check_liquidity_risk(self, pair: str, position_size: float) -> Dict[str, Any]:
        """Check liquidity and spread risk for the proposed position."""
        try:
            # Current market microstructure snapshot
            market_data = getattr(self, 'current_market_data', {})
            
            if pair in market_data:
                volume_ratio = market_data[pair].get('volume_ratio', 1.0)
                spread = market_data[pair].get('spread', 0.001)
            else:
                volume_ratio = 1.0  # default value
                spread = 0.002
            
            # Liquidity-risk score
            liquidity_risk_score = 0.0
            
            # Volume-based liquidity risk
            if volume_ratio < 0.5:  # volume low
                liquidity_risk_score += 0.3
            elif volume_ratio < 0.8:
                liquidity_risk_score += 0.1
            
            # Spread-based liquidity risk
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
                    'message': f'liquidity risk high ({liquidity_risk_score:.1f}); reduce position size'
                }
            
            return {
                'sufficient_liquidity': True,
                'type': 'liquidity_check',
                'risk_score': liquidity_risk_score,
                'message': 'liquidity check passed'
            }
            
        except Exception:
            return {
                'sufficient_liquidity': False,
                'adjustment_factor': 0.5,
                'message': 'liquidity check failed; applying defensive sizing'
            }
    
    def check_leverage_risk(self, pair: str, proposed_leverage: int) -> Dict[str, Any]:
        """leverage risk check"""
        try:
            # Market-volatility-based leverage cap
            market_volatility = getattr(self, 'current_market_volatility', {}).get(pair, 0.02)
            
            # Determine max allowed leverage from volatility
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
                    'message': f'leverage too high; cap reduced to {max_allowed_leverage}x'
                }
            
            return {
                'within_limits': True,
                'type': 'leverage_check',
                'approved_leverage': proposed_leverage,
                'message': 'leverage check passed'
            }
            
        except Exception:
            return {
                'within_limits': False,
                'max_allowed_leverage': min(3, proposed_leverage),
                'message': 'leverage check failed; applying defensive cap'
            }
    
    def check_time_based_risk(self, current_time: datetime) -> Dict[str, Any]:
        """Check for elevated risk based on time-of-day and day-of-week."""
        try:
            hour = current_time.hour
            weekday = current_time.weekday()
            
            high_risk_periods = [
                (weekday >= 5),  # weekends
                (hour <= 6 or hour >= 22),  # thin-liquidity hours
                (11 <= hour <= 13),  # midday lull / transition period
            ]
            
            if any(high_risk_periods):
                adjustment_factor = 0.7  # high risk small position size
                
                return {
                    'high_risk_period': True,
                    'type': 'time_based_risk',
                    'adjustment_factor': adjustment_factor,
                    'hour': hour,
                    'weekday': weekday,
                    'message': 'high-risk time window; reduce position size'
                }
            
            return {
                'high_risk_period': False,
                'type': 'time_check',
                'adjustment_factor': 1.0,
                'message': 'time-based risk check passed'
            }
            
        except Exception:
            return {
                'high_risk_period': True,
                'adjustment_factor': 0.8,
                'message': 'time-based risk check failed; applying defensive sizing'
            }
    
    def record_risk_event(self, event_type: str, event_data: Dict[str, Any]):
        """Record a risk event for later review."""
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
        """Classify the severity of a recorded risk event."""
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
        """Activate emergency risk shutdown mode."""
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
            
            # Reserved for external alerting/logging hooks
            
        except Exception:
            pass
    
    def get_risk_control_status(self) -> Dict[str, Any]:
        """Return the current risk-control status snapshot."""
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
            return {'error': 'unable to fetch risk-control status'}
    
    # ===== Execution system =====
    
    def initialize_execution_system(self):
        """Initialize the smart execution subsystem."""
        # Execution algorithm weights
        self.execution_algorithms = {
            'twap': {'enabled': True, 'weight': 0.3},      # time price
            'vwap': {'enabled': True, 'weight': 0.4},      # volume price
            'implementation_shortfall': {'enabled': True, 'weight': 0.3}  # most small
        }
        
        # Slippage-control settings
        self.slippage_control = {
            'max_allowed_slippage': 0.002,     # max allowed slippage: 0.2%
            'slippage_prediction_window': 50,  # lookback window
            'adaptive_threshold': 0.001,       # adaptive threshold: 0.1%
            'emergency_threshold': 0.005       # emergency threshold: 0.5%
        }
        
        # Order-splitting settings
        self.order_splitting = {
            'min_split_size': 0.01,            # minimum split threshold: 1%
            'max_split_count': 10,
            'split_interval_seconds': 30,      # 30 seconds
            'adaptive_splitting': True
        }
        
        # Execution-performance metrics
        self.execution_metrics = {
            'realized_slippage': [],
            'market_impact': [],
            'execution_time': [],
            'fill_ratio': [],
            'cost_basis_deviation': []
        }
        
        # Simple market-impact model coefficients
        self.market_impact_model = {
            'temporary_impact_factor': 0.5,
            'permanent_impact_factor': 0.3,
            'nonlinear_factor': 1.5,
            'decay_factor': 0.1
        }
        
        # Execution tracking
        self.active_executions = {}
        self.execution_history = []
        
    def smart_order_execution(self, pair: str, order_size: float, order_side: str, 
                            current_price: float, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Build a smart execution plan for a proposed order."""
        
        execution_plan = {
            'original_size': order_size,
            'execution_strategy': None,
            'split_orders': [],
            'expected_slippage': 0.0,
            'estimated_execution_time': 0,
            'risk_level': 'normal'
        }
        
        try:
            # 1. Execution-risk assessment
            execution_risk = self.assess_execution_risk(pair, order_size, market_conditions)
            execution_plan['risk_level'] = execution_risk['level']
            
            # 2. Slippage prediction
            predicted_slippage = self.predict_slippage(pair, order_size, order_side, market_conditions)
            execution_plan['expected_slippage'] = predicted_slippage
            
            # 3. Execution-algorithm selection
            optimal_algorithm = self.select_execution_algorithm(pair, order_size, market_conditions, execution_risk)
            execution_plan['execution_strategy'] = optimal_algorithm
            
            # 4. Optional order splitting
            if order_size > self.order_splitting['min_split_size'] and execution_risk['level'] != 'low':
                split_plan = self.optimize_order_splitting(pair, order_size, market_conditions, optimal_algorithm)
                execution_plan['split_orders'] = split_plan['orders']
                execution_plan['estimated_execution_time'] = split_plan['total_time']
            else:
                execution_plan['split_orders'] = [{'size': order_size, 'delay': 0, 'priority': 'high'}]
                execution_plan['estimated_execution_time'] = 30
            
            # 5. Timing optimization
            execution_timing = self.optimize_execution_timing(pair, market_conditions)
            execution_plan['optimal_timing'] = execution_timing
            
            # 6. Execution instructions
            execution_instructions = self.generate_execution_instructions(execution_plan, pair, order_side, current_price)
            execution_plan['instructions'] = execution_instructions
            
            return execution_plan
            
        except Exception as e:
            # Fallback immediate-execution plan
            return {
                'original_size': order_size,
                'execution_strategy': 'immediate',
                'split_orders': [{'size': order_size, 'delay': 0, 'priority': 'high'}],
                'expected_slippage': 0.002,
                'estimated_execution_time': 30,
                'risk_level': 'unknown',
                'error': str(e)
            }
    
    def assess_execution_risk(self, pair: str, order_size: float, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Assess execution risk from size, volatility, and spread."""
        try:
            risk_score = 0.0
            risk_factors = []
            
            # 1. Order-size risk relative to average volume
            avg_volume = market_conditions.get('avg_volume', 1.0)
            order_volume_ratio = order_size / avg_volume if avg_volume > 0 else 1.0
            
            if order_volume_ratio > 0.1:  # >10% of average volume
                risk_score += 0.4
                risk_factors.append('large_order_size')
            elif order_volume_ratio > 0.05:
                risk_score += 0.2
                risk_factors.append('medium_order_size')
            
            # 2. Volatility risk
            volatility = market_conditions.get('volatility', 0.02)
            if volatility > 0.05:
                risk_score += 0.3
                risk_factors.append('high_volatility')
            elif volatility > 0.03:
                risk_score += 0.15
                risk_factors.append('medium_volatility')
            
            # 3. Spread risk
            bid_ask_spread = market_conditions.get('spread', 0.001)
            if bid_ask_spread > 0.003:
                risk_score += 0.2
                risk_factors.append('wide_spread')
            
            # 4. Session risk
            if self.is_high_volatility_session(datetime.now(timezone.utc)):
                risk_score += 0.1
                risk_factors.append('high_volatility_session')
            
            # Risk-level classification
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
        """Predict expected slippage for an order."""
        try:
            # Base slippage from half-spread
            base_slippage = market_conditions.get('spread', 0.001) / 2
            
            # Order size relative to average volume
            avg_volume = market_conditions.get('avg_volume', 1.0)
            volume_ratio = order_size / avg_volume if avg_volume > 0 else 0.1
            
            # Temporary impact
            temporary_impact = (
                self.market_impact_model['temporary_impact_factor'] * 
                (volume_ratio ** self.market_impact_model['nonlinear_factor'])
            )
            
            # Permanent impact
            permanent_impact = (
                self.market_impact_model['permanent_impact_factor'] * 
                (volume_ratio ** 0.5)
            )
            
            # volatility
            volatility = market_conditions.get('volatility', 0.02)
            volatility_adjustment = min(1.0, volatility * 10)  # volatility high large
            
            # Session adjustment
            time_adjustment = 1.0
            if self.is_high_volatility_session(datetime.now(timezone.utc)):
                time_adjustment = 1.2
            elif self.is_low_liquidity_session(datetime.now(timezone.utc)):
                time_adjustment = 1.3
            
            # Historical realized-slippage adjustment
            historical_slippage = self.get_historical_slippage(pair)
            historical_adjustment = max(0.5, min(2.0, historical_slippage / 0.001))
            
            # Final predicted slippage
            predicted_slippage = (
                base_slippage + temporary_impact + permanent_impact
            ) * volatility_adjustment * time_adjustment * historical_adjustment
            
            # Cap at the emergency threshold
            predicted_slippage = min(predicted_slippage, self.slippage_control['emergency_threshold'])
            
            return max(0.0001, predicted_slippage)  # floor at 0.01%
            
        except Exception:
            return 0.002
    
    def get_historical_slippage(self, pair: str) -> float:
        """Return recent realized slippage for the pair."""
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
        """Select the preferred execution algorithm."""
        try:
            algorithm_scores = {}
            
            # TWAP score
            if self.execution_algorithms['twap']['enabled']:
                twap_score = 0.5
                
                # time low
                if execution_risk['level'] == 'low':
                    twap_score += 0.2
                
                # TWAP works better in lower-volatility conditions
                if market_conditions.get('volatility', 0.02) < 0.025:
                    twap_score += 0.1
                
                algorithm_scores['twap'] = twap_score * self.execution_algorithms['twap']['weight']
            
            # VWAP score
            if self.execution_algorithms['vwap']['enabled']:
                vwap_score = 0.6
                
                # volume
                if market_conditions.get('volume_ratio', 1.0) > 1.0:
                    vwap_score += 0.2
                
                # medium risk most
                if execution_risk['level'] == 'medium':
                    vwap_score += 0.15
                
                algorithm_scores['vwap'] = vwap_score * self.execution_algorithms['vwap']['weight']
            
            # Implementation Shortfall score
            if self.execution_algorithms['implementation_shortfall']['enabled']:
                is_score = 0.4
                
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
            
            # Choose the best-scoring algorithm
            if algorithm_scores:
                optimal_algorithm = max(algorithm_scores.items(), key=lambda x: x[1])[0]
                return optimal_algorithm
            else:
                return 'twap'  # default
                
        except Exception:
                return 'twap'
    
    def optimize_order_splitting(self, pair: str, order_size: float, 
                               market_conditions: Dict[str, Any], 
                               algorithm: str) -> Dict[str, Any]:
        """Build an order-splitting plan for the selected execution algorithm."""
        try:
            split_plan = {
                'orders': [],
                'total_time': 0,
                'expected_total_slippage': 0.0
            }
            
            # Order size relative to average volume
            avg_volume = market_conditions.get('avg_volume', 1.0)
            volume_ratio = order_size / avg_volume if avg_volume > 0 else 0.1
            
            if volume_ratio > 0.2:  # large
                split_count = min(self.order_splitting['max_split_count'], 8)
            elif volume_ratio > 0.1:  # large
                split_count = min(self.order_splitting['max_split_count'], 5)
            elif volume_ratio > 0.05:  # mid
                split_count = min(self.order_splitting['max_split_count'], 3)
            else:
                split_count = 1
            
            if split_count == 1:
                split_plan['orders'] = [{'size': order_size, 'delay': 0, 'priority': 'high'}]
                split_plan['total_time'] = 30
                return split_plan
            
            # Algorithm-specific split logic
            if algorithm == 'twap':
                # Time-weighted schedule
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
                # Volume-weighted schedule
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
                
                # Make sure the full order size is allocated
                if cumulative_size < order_size:
                    remaining = order_size - cumulative_size
                    split_plan['orders'][-1]['size'] += remaining
                
                split_plan['total_time'] = len(split_plan['orders']) * 60
                
            else:  # implementation_shortfall
                # Front-load execution based on urgency
                remaining_size = order_size
                time_offset = 0
                urgency_factor = min(1.5, market_conditions.get('volatility', 0.02) * 20)
                
                for i in range(split_count):
                    if i == split_count - 1:
                        # Final child order takes the remainder
                        sub_order_size = remaining_size
                    else:
                        # Allocate dynamically by urgency
                        base_portion = 1.0 / (split_count - i)
                        urgency_adjustment = base_portion * urgency_factor
                        sub_order_size = min(remaining_size, order_size * urgency_adjustment)
                    
                    split_plan['orders'].append({
                        'size': sub_order_size,
                        'delay': time_offset,
                        'priority': 'high' if i < 2 else 'medium'
                    })
                    
                    remaining_size -= sub_order_size
                    time_offset += max(15, int(45 / urgency_factor))
                    
                    if remaining_size <= 0:
                        break
                
                split_plan['total_time'] = time_offset + 30
            
            # Expected blended slippage across child orders
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
        """Return a placeholder forecast for volume distribution."""
        try:
            # Simplified intraday volume profile
            typical_distribution = [
                0.05, 0.08, 0.12, 0.15, 0.18, 0.15, 0.12, 0.08, 0.05, 0.02
            ]
            return typical_distribution
        except Exception:
            return [0.1] * 10
    
    def optimize_execution_timing(self, pair: str, market_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Estimate whether immediate or delayed execution is preferable."""
        try:
            current_time = datetime.now(timezone.utc)
            hour = current_time.hour
            
            timing_score = 0.5
            timing_factors = []
            
            # Session-liquidity score
            if 13 <= hour <= 16:
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
            
            # Timing recommendation
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
        """Generate executable child-order instructions from the execution plan."""
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
            # Fallback to a simple immediate market order instruction
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
        """Choose the order type for a child order."""
        try:
            if order['priority'] == 'high' or execution_plan.get('risk_level') == 'high':
                return 'market'
            elif execution_plan['expected_slippage'] < self.slippage_control['adaptive_threshold']:
                return 'limit'
            else:
                return 'market_with_protection'
        except Exception:
            return 'market'
    
    def calculate_price_limit(self, current_price: float, side: str, 
                            order_size: float, execution_plan: Dict[str, Any]) -> float:
        """Calculate a protected limit price from expected slippage."""
        try:
            expected_slippage = execution_plan['expected_slippage']
            
            # Add a 20% buffer over expected slippage
            slippage_buffer = expected_slippage * 1.2
            
            if side.lower() == 'buy':
                return current_price * (1 + slippage_buffer)
            else:
                return current_price * (1 - slippage_buffer)
                
        except Exception:
            # Simple fallback limit
            if side.lower() == 'buy':
                return current_price * 1.005
            else:
                return current_price * 0.995
    
    def track_execution_performance(self, execution_id: str, execution_result: Dict[str, Any]):
        """Track realized execution quality metrics."""
        try:
            # Realized slippage
            expected_price = execution_result.get('expected_price', 0)
            actual_price = execution_result.get('actual_price', 0)
            
            if expected_price > 0 and actual_price > 0:
                realized_slippage = abs(actual_price - expected_price) / expected_price
                self.execution_metrics['realized_slippage'].append(realized_slippage)
            
            # Market impact
            pre_trade_price = execution_result.get('pre_trade_price', 0)
            post_trade_price = execution_result.get('post_trade_price', 0)
            
            if pre_trade_price > 0 and post_trade_price > 0:
                market_impact = abs(post_trade_price - pre_trade_price) / pre_trade_price
                self.execution_metrics['market_impact'].append(market_impact)
            
            # Execution time
            execution_time = execution_result.get('execution_time_seconds', 0)
            if execution_time > 0:
                self.execution_metrics['execution_time'].append(execution_time)
            
            fill_ratio = execution_result.get('fill_ratio', 1.0)
            self.execution_metrics['fill_ratio'].append(fill_ratio)
            
            # Trim long histories
            for metric in self.execution_metrics.values():
                if len(metric) > 500:
                    metric[:] = metric[-250:]  # most250count record
                    
        except Exception:
            pass
    
    def get_execution_quality_report(self) -> Dict[str, Any]:
        """Return summary statistics for execution quality."""
        try:
            if not any(self.execution_metrics.values()):
                return {'error': 'no execution metrics available'}
            
            report = {}
            
            # Slippage stats
            if self.execution_metrics['realized_slippage']:
                slippage_data = self.execution_metrics['realized_slippage']
                report['slippage'] = {
                    'avg': np.mean(slippage_data),
                    'median': np.median(slippage_data),
                    'std': np.std(slippage_data),
                    'p95': np.percentile(slippage_data, 95),
                    'samples': len(slippage_data)
                }
            
            # Market-impact stats
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
            
            # Fill-ratio stats
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
            return {'error': 'execution quality report unavailable'}
    
    # ===== Sentiment system =====
    
    def initialize_sentiment_system(self):
        """Initialize the placeholder sentiment-analysis subsystem."""
        # market sentiment indicator configuration
        self.sentiment_indicators = {
            'fear_greed_index': {'enabled': True, 'weight': 0.25},
            'vix_equivalent': {'enabled': True, 'weight': 0.20},
            'news_sentiment': {'enabled': True, 'weight': 0.15},
            'social_sentiment': {'enabled': True, 'weight': 0.10},
            'positioning_data': {'enabled': True, 'weight': 0.15},
            'intermarket_sentiment': {'enabled': True, 'weight': 0.15}
        }
        
        # Sentiment thresholds
        self.sentiment_thresholds = {
            'extreme_fear': 20,
            'fear': 35,
            'neutral': 50,           # mid
            'greed': 65,
            'extreme_greed': 80
        }
        
        # configuration
        self.external_data_sources = {
            'economic_calendar': {'enabled': True, 'impact_threshold': 'medium'},
            'central_bank_policy': {'enabled': True, 'lookback_days': 30},
            'geopolitical_events': {'enabled': True, 'risk_threshold': 'medium'},
            'seasonal_patterns': {'enabled': True, 'historical_years': 5},
            'intermarket_correlations': {'enabled': True, 'correlation_threshold': 0.6}
        }
        
        # Sentiment history
        self.sentiment_history = {
            'composite_sentiment': [],
            'sentiment_state': [],
            'market_regime': [],
            'sentiment_extremes': [],
            'contrarian_signals': []
        }
        
        # External-event placeholders
        self.external_events = []
        self.event_impact_history = []
        
        # Placeholder seasonal/intermarket data
        self.seasonal_patterns = {}
        self.intermarket_data = {}
        
    # Simplified aggregate sentiment analysis
    def analyze_market_sentiment(self) -> Dict[str, Any]:
        """Build a composite sentiment snapshot from the enabled components."""
        try:
            sentiment_components = {}
            
            # 1. Fear/greed proxy
            if self.sentiment_indicators['fear_greed_index']['enabled']:
                fear_greed = self.calculate_fear_greed_index()
                sentiment_components['fear_greed'] = fear_greed
            
            # 2. volatility
            if self.sentiment_indicators['vix_equivalent']['enabled']:
                vix_sentiment = self.analyze_volatility_sentiment()
                sentiment_components['volatility_sentiment'] = vix_sentiment
            
            # 3. News sentiment
            if self.sentiment_indicators['news_sentiment']['enabled']:
                news_sentiment = self.analyze_news_sentiment()
                sentiment_components['news_sentiment'] = news_sentiment
            
            # 4. Social sentiment
            if self.sentiment_indicators['social_sentiment']['enabled']:
                social_sentiment = self.analyze_social_sentiment()
                sentiment_components['social_sentiment'] = social_sentiment
            
            # 5. Positioning sentiment
            if self.sentiment_indicators['positioning_data']['enabled']:
                positioning_sentiment = self.analyze_positioning_data()
                sentiment_components['positioning_sentiment'] = positioning_sentiment
            
            # 6. Intermarket sentiment
            if self.sentiment_indicators['intermarket_sentiment']['enabled']:
                intermarket_sentiment = self.analyze_intermarket_sentiment()
                sentiment_components['intermarket_sentiment'] = intermarket_sentiment
            
            # Composite sentiment score
            composite_sentiment = self.calculate_composite_sentiment(sentiment_components)
            
            # Qualitative sentiment state
            sentiment_state = self.determine_sentiment_state(composite_sentiment)
            
            # Trading adjustments
            sentiment_adjustment = self.generate_sentiment_adjustment(sentiment_state, sentiment_components)
            
            sentiment_analysis = {
                'composite_sentiment': composite_sentiment,
                'sentiment_state': sentiment_state,
                'components': sentiment_components,
                'trading_adjustment': sentiment_adjustment,
                'contrarian_opportunity': self.detect_contrarian_opportunity(composite_sentiment),
                'timestamp': datetime.now(timezone.utc)
            }
            
            # Update sentiment history
            self.update_sentiment_history(sentiment_analysis)
            
            return sentiment_analysis
            
        except Exception as e:
            return {
                'composite_sentiment': 50,
                'sentiment_state': 'neutral',
                'error': f'Sentiment analysis failed: {str(e)}',
                'timestamp': datetime.now(timezone.utc)
            }
    
    def calculate_fear_greed_index(self) -> Dict[str, Any]:
        """Calculate a placeholder fear/greed index from internal proxies."""
        try:
            components = {}
            
            # price momentum (25%)
            price_momentum = self.calculate_price_momentum_sentiment()
            components['price_momentum'] = price_momentum
            
            # volatility (25%) - andVIXtranslated
            volatility_fear = self.calculate_volatility_fear()
            components['volatility_fear'] = volatility_fear
            
            # Market breadth (15%)
            market_breadth = self.calculate_market_breadth_sentiment()
            components['market_breadth'] = market_breadth
            
            # Safe-haven demand (15%)
            safe_haven_demand = self.calculate_safe_haven_sentiment()
            components['safe_haven_demand'] = safe_haven_demand
            
            # Junk-bond demand (10%)
            junk_bond_demand = self.calculate_junk_bond_sentiment()
            components['junk_bond_demand'] = junk_bond_demand
            
            # Put/call ratio (10%)
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
        """Calculate a placeholder price-momentum sentiment score."""
        try:
            # Simplified placeholder breadth proxy
            
            stocks_above_ma125 = 0.6  # 60% above the 125-day average
            
            # Convert to a 0-100 score
            momentum_sentiment = stocks_above_ma125 * 100
            
            return min(100, max(0, momentum_sentiment))
            
        except Exception:
            return 50
    
    def calculate_volatility_fear(self) -> float:
        """Calculate a placeholder volatility-based fear score."""
        try:
            # Average current market volatility
            current_volatility = getattr(self, 'current_market_volatility', {})
            avg_vol = sum(current_volatility.values()) / len(current_volatility) if current_volatility else 0.02
            
            # Historical anchor value
            historical_avg_vol = 0.025
            
            # Relative volatility
            vol_ratio = avg_vol / historical_avg_vol if historical_avg_vol > 0 else 1.0
            
            # Higher volatility maps to lower sentiment
            volatility_fear = max(0, min(100, 100 - (vol_ratio - 1) * 50))
            
            return volatility_fear
            
        except Exception:
            return 50
    
    def calculate_market_breadth_sentiment(self) -> float:
        """Calculate a placeholder market-breadth sentiment score."""
        try:
            # Simplified advance/decline ratio placeholder
            
            advancing_stocks_ratio = 0.55  # 55% advancing
            
            # Convert to a score
            breadth_sentiment = advancing_stocks_ratio * 100
            
            return min(100, max(0, breadth_sentiment))
            
        except Exception:
            return 50
    
    def calculate_safe_haven_sentiment(self) -> float:
        """Calculate a placeholder safe-haven-demand sentiment score."""
        try:
            # Placeholder safe-haven proxy performance
            
            safe_haven_performance = -0.02  # -2%
            
            # Better risky-asset sentiment when safe havens underperform
            safe_haven_sentiment = max(0, min(100, 50 - safe_haven_performance * 1000))
            
            return safe_haven_sentiment
            
        except Exception:
            return 50
    
    def calculate_junk_bond_sentiment(self) -> float:
        """Calculate a placeholder junk-bond sentiment score."""
        try:
            # Placeholder credit-spread inputs
            
            credit_spread_bp = 350
            historical_avg_spread = 400  # basis points
            
            # Tighter spreads imply more risk appetite
            spread_ratio = credit_spread_bp / historical_avg_spread
            junk_bond_sentiment = max(0, min(100, 100 - (spread_ratio - 1) * 100))
            
            return junk_bond_sentiment
            
        except Exception:
            return 50
    
    def calculate_put_call_sentiment(self) -> float:
        """Calculate a placeholder put/call-ratio sentiment score."""
        try:
            # Placeholder options sentiment input
            
            put_call_ratio = 0.8
            historical_avg_ratio = 1.0
            
            # Lower put/call ratios imply less fear
            put_call_sentiment = max(0, min(100, 100 - (put_call_ratio / historical_avg_ratio - 1) * 100))
            
            return put_call_sentiment
            
        except Exception:
            return 50
    
    def interpret_fear_greed_index(self, index_value: float) -> str:
        """Map the fear/greed index to a qualitative state."""
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
    
    # Simplified volatility sentiment analysis
    def analyze_volatility_sentiment(self) -> Dict[str, Any]:
        """Analyze a simple volatility-based sentiment regime."""
        try:
            current_volatility = getattr(self, 'current_market_volatility', {})
            
            if not current_volatility:
                return {
                    'volatility_level': 'normal',
                    'sentiment_signal': 'neutral',
                    'volatility_percentile': 50
                }
            
            avg_vol = sum(current_volatility.values()) / len(current_volatility)
            
            # Simplified percentile estimate
            vol_percentile = min(95, max(5, avg_vol * 2000))  # simplified
            
            # Volatility regime classification
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
    
    # Simplified news sentiment placeholder
    def analyze_news_sentiment(self) -> Dict[str, Any]:
        """Return a placeholder news-sentiment snapshot."""
        try:
            # Placeholder output until a real external news feed is wired in
            
            news_sentiment_score = 0.1
            
            news_volume = 1.2  # 120% of baseline volume
            
            # Placeholder keyword buckets
            sentiment_keywords = {
                'positive': ['growth', 'opportunity', 'bullish'],
                'negative': ['uncertainty', 'risk', 'volatile'],
                'neutral': ['stable', 'unchanged', 'maintain']
            }
            
            # Convert score into a directional signal
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
    
    # Simplified social sentiment placeholder
    def analyze_social_sentiment(self) -> Dict[str, Any]:
        """Return a placeholder social-sentiment snapshot."""
        try:
            # Placeholder social-data aggregation
            
            mention_volume = 1.3  # 130% of baseline
            
            # Placeholder sentiment distribution
            sentiment_distribution = {
                'bullish': 0.4,
                'bearish': 0.3,
                'neutral': 0.3
            }
            
            influencer_sentiment = 0.2
            
            # trend strength
            trend_strength = abs(sentiment_distribution['bullish'] - sentiment_distribution['bearish'])
            
            # Composite social score
            social_score = (
                sentiment_distribution['bullish'] * 1 + 
                sentiment_distribution['bearish'] * (-1) + 
                sentiment_distribution['neutral'] * 0
            )
            
            # Weight retail/social chatter and higher-signal influencer sentiment
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
    
    # Simplified positioning-data placeholder
    def analyze_positioning_data(self) -> Dict[str, Any]:
        """Return a placeholder positioning-data snapshot."""
        try:
            # Placeholder positioning proxies
            
            large_trader_net_long = 0.15  # 15% net long
            
            retail_sentiment = -0.1
            
            institutional_flow = 0.05  # 5%
            
            # Largest absolute positioning signal
            positioning_extreme = max(
                abs(large_trader_net_long),
                abs(retail_sentiment),
                abs(institutional_flow)
            )
            
            # Contrarian interpretation of retail sentiment
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
    
    # Simplified intermarket sentiment placeholder
    def analyze_intermarket_sentiment(self) -> Dict[str, Any]:
        """Return a placeholder intermarket-sentiment snapshot."""
        try:
            # Placeholder intermarket proxies
            
            stock_bond_correlation = -0.3
            
            dollar_strength = 0.02  # 2%
            
            commodity_performance = -0.01
            
            safe_haven_flows = 0.5
            
            # Simple stress/risk-appetite proxies
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
        """Combine enabled sentiment components into a 0-100 score."""
        try:
            sentiment_values = []
            weights = []
            
            # Fear/greed component
            if 'fear_greed' in components:
                sentiment_values.append(components['fear_greed']['index_value'])
                weights.append(self.sentiment_indicators['fear_greed_index']['weight'])
            
            # volatility
            if 'volatility_sentiment' in components:
                vol_sentiment = 100 - components['volatility_sentiment']['volatility_percentile']
                sentiment_values.append(vol_sentiment)
                weights.append(self.sentiment_indicators['vix_equivalent']['weight'])
            
            # News sentiment component
            if 'news_sentiment' in components:
                news_score = (components['news_sentiment']['sentiment_score'] + 1) * 50
                sentiment_values.append(news_score)
                weights.append(self.sentiment_indicators['news_sentiment']['weight'])
            
            # Social sentiment component
            if 'social_sentiment' in components:
                social_score = (components['social_sentiment']['sentiment_score'] + 1) * 50
                sentiment_values.append(social_score)
                weights.append(self.sentiment_indicators['social_sentiment']['weight'])
            
            # Positioning sentiment component
            if 'positioning_sentiment' in components:
                pos_score = 50  # neutral placeholder
                sentiment_values.append(pos_score)
                weights.append(self.sentiment_indicators['positioning_data']['weight'])
            
            # Intermarket sentiment component
            if 'intermarket_sentiment' in components:
                inter_score = (components['intermarket_sentiment']['risk_appetite'] + 1) * 50
                sentiment_values.append(inter_score)
                weights.append(self.sentiment_indicators['intermarket_sentiment']['weight'])
            
            # Weighted average
            if sentiment_values and weights:
                total_weight = sum(weights)
                composite_sentiment = sum(s * w for s, w in zip(sentiment_values, weights)) / total_weight
            else:
                composite_sentiment = 50  # neutral default
            
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
        """Detect simple contrarian opportunities from composite sentiment."""
        try:
            # Default no-opportunity state
            contrarian_opportunity = {
                'opportunity_detected': False,
                'opportunity_type': None,
                'strength': 0.0,
                'recommended_action': 'hold'
            }
            
            # Extreme-fear and extreme-greed thresholds
            if composite_sentiment <= 25:
                contrarian_opportunity.update({
                    'opportunity_detected': True,
                    'opportunity_type': 'extreme_fear_buying',
                    'strength': (25 - composite_sentiment) / 25,
                    'recommended_action': 'aggressive_buy'
                })
            elif composite_sentiment >= 75:
                contrarian_opportunity.update({
                    'opportunity_detected': True,
                    'opportunity_type': 'extreme_greed_selling',
                    'strength': (composite_sentiment - 75) / 25,
                    'recommended_action': 'reduce_exposure'
                })
            
            # Detect fast sentiment reversals
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
        """Update the rolling sentiment history."""
        try:
            # Composite sentiment history
            self.sentiment_history['composite_sentiment'].append(sentiment_analysis['composite_sentiment'])
            
            # Sentiment state history
            self.sentiment_history['sentiment_state'].append(sentiment_analysis['sentiment_state'])
            
            # Track extreme sentiment readings
            if sentiment_analysis['composite_sentiment'] <= 25 or sentiment_analysis['composite_sentiment'] >= 75:
                extreme_record = {
                    'timestamp': sentiment_analysis['timestamp'],
                    'sentiment_value': sentiment_analysis['composite_sentiment'],
                    'sentiment_state': sentiment_analysis['sentiment_state']
                }
                self.sentiment_history['sentiment_extremes'].append(extreme_record)
            
            # Track contrarian signals
            if sentiment_analysis.get('contrarian_opportunity', {}).get('opportunity_detected'):
                contrarian_record = {
                    'timestamp': sentiment_analysis['timestamp'],
                    'opportunity_type': sentiment_analysis['contrarian_opportunity']['opportunity_type'],
                    'strength': sentiment_analysis['contrarian_opportunity']['strength']
                }
                self.sentiment_history['contrarian_signals'].append(contrarian_record)
            
            # Trim long histories
            for key, history in self.sentiment_history.items():
                if len(history) > 500:
                    self.sentiment_history[key] = history[-250:]
                    
        except Exception:
            pass
    
    def get_sentiment_analysis_report(self) -> Dict[str, Any]:
        """Return a compact summary of recent sentiment history."""
        try:
            if not self.sentiment_history['composite_sentiment']:
                return {'error': 'no sentiment history available'}
            
            recent_sentiment = self.sentiment_history['composite_sentiment'][-1]
            recent_state = self.sentiment_history['sentiment_state'][-1]
            
            # Recent sentiment statistics
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
            return {'error': 'sentiment report unavailable'}
    
    # === ATR-based stoploss helpers ===
    
    def _get_trade_entry_atr(self, trade: Trade, dataframe: DataFrame) -> float:
        """
        Get the ATR value near trade entry for adaptive stop calculations.
        """
        try:
            # Align the trade open time to the prior candle boundary
            from freqtrade.misc import timeframe_to_prev_date
            
            entry_date = timeframe_to_prev_date(self.timeframe, trade.open_date_utc)
            entry_candles = dataframe[dataframe.index <= entry_date]
            
            if not entry_candles.empty and 'atr_p' in entry_candles.columns:
                entry_atr = entry_candles['atr_p'].iloc[-1]
                # Sanity-check the ATR range
                if 0.005 <= entry_atr <= 0.20:
                    return entry_atr
                    
        except Exception as e:
            logger.warning(f"Failed to get entry ATR: {e}")
            
        # Fallback to the recent 20-candle median ATR
        if 'atr_p' in dataframe.columns and len(dataframe) >= 20:
            return dataframe['atr_p'].tail(20).median()
        
        # Final asset-class fallback
        if 'BTC' in trade.pair or 'ETH' in trade.pair:
            return 0.02
        else:
            return 0.035
    
    def _calculate_atr_multiplier(self, entry_atr_p: float, current_candle: dict, enter_tag: str) -> float:
        """
        Calculate the ATR stop multiplier based on signal type and current conditions.
        """
        # Baseline ATR multiplier
        base_multiplier = 2.8
        
        # === 1. Signal-specific adjustments ===
        signal_adjustments = {
            'RSI_Oversold_Bounce': 2.5,
            'RSI_Overbought_Fall': 2.5,    
            'MACD_Bearish': 3.2,
            'MACD_Bullish': 3.2,
            'EMA_Golden_Cross': 2.6,
            'EMA_Death_Cross': 2.6,
        }
        
        multiplier = signal_adjustments.get(enter_tag, base_multiplier)
        
        # === 2. volatility ===
        current_atr_p = current_candle.get('atr_p', entry_atr_p)
        volatility_ratio = current_atr_p / entry_atr_p
        
        if volatility_ratio > 1.5:
            multiplier *= 1.2
        elif volatility_ratio < 0.7:
            multiplier *= 0.9
        
        # === 3. trend strength ===
        adx = current_candle.get('adx', 25)
        if adx > 35:                    # strong trend
            multiplier *= 1.15          # trend
        elif adx < 20:
            multiplier *= 0.85
        
        # Final bounded multiplier
        return max(1.5, min(4.0, multiplier))
    
    def _calculate_time_decay(self, hours_held: float, current_profit: float) -> float:
        """
        Reduce stop distance as trades age, especially when they are not working.
        """
        # More profitable trades get longer before time decay tightens stops
        if current_profit > 0.02:
            decay_start_hours = 72
        elif current_profit > -0.02:
            decay_start_hours = 48
        else:
            decay_start_hours = 24
        
        if hours_held <= decay_start_hours:
            return 1.0
            
        # Reduce by 10% for every extra 24 hours
        excess_hours = hours_held - decay_start_hours
        decay_periods = excess_hours / 24
        
        # Do not reduce below 50% of the original distance
        min_factor = 0.5
        decay_factor = max(min_factor, 1.0 - (decay_periods * 0.1))
        
        return decay_factor
    
    def _calculate_profit_protection(self, current_profit: float) -> Optional[float]:
        """
        Tighten stoploss once a trade reaches meaningful profit.
        """
        if current_profit > 0.15:
            return -0.0375
        elif current_profit > 0.10:
            return -0.04
        elif current_profit > 0.08:
            return -0.04
        elif current_profit > 0.05:
            return -0.01
        elif current_profit > 0.03:
            return 0.001
        
        return None
    
    def _calculate_trend_adjustment(self, current_candle: dict, is_short: bool, entry_atr_p: float) -> float:
        """
        Adjust stop distance based on trend alignment.
        """
        # Trend indicators
        ema_8 = current_candle.get('ema_8', 0)
        ema_21 = current_candle.get('ema_21', 0)
        adx = current_candle.get('adx', 25)
        current_price = current_candle.get('close', 0)
        
        # Basic trend state
        is_uptrend = ema_8 > ema_21 and adx > 25
        is_downtrend = ema_8 < ema_21 and adx > 25
        
        # Favor looser stops when the trade aligns with the trend
        if is_short and is_downtrend:
            return 1.2
        elif not is_short and is_uptrend:
            return 1.2
        elif is_short and is_uptrend:
            return 0.8
        elif not is_short and is_downtrend:
            return 0.8
        else:
            return 1.0
    
    def _log_stoploss_calculation(self, pair: str, trade: Trade, current_profit: float,
                                 entry_atr_p: float, base_atr_multiplier: float,
                                 time_decay_factor: float, trend_adjustment: float,
                                 final_stoploss: float):
        """
        Log the components of the ATR-based stoploss calculation.
        """
        hours_held = (datetime.now(timezone.utc) - trade.open_date_utc).total_seconds() / 3600
        
        logger.info(
            f"🛡️ ATR stoploss {pair} [{trade.enter_tag}]: "
            f"profit={current_profit:.1%} | "
            f"held={hours_held:.1f}h | "
            f"entry_atr={entry_atr_p:.3f} | "
            f"atr_multiplier={base_atr_multiplier:.1f} | "
            f"time_decay={time_decay_factor:.2f} | "
            f"trend_adj={trend_adjustment:.2f} | "
            f"stop={final_stoploss:.3f}"
        )
    
    def custom_stoploss(self, pair: str, trade: Trade, current_time: datetime,
                       current_rate: float, current_profit: float, 
                       after_fill: bool = False, **kwargs) -> Optional[float]:
        """
        ATR-based adaptive stoploss:
        - entry ATR baseline
        - time decay
        - profit protection
        - trend alignment
        """
        try:
            # get most new
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            if dataframe.empty or len(dataframe) < 50:
                return None
                
            current_candle = dataframe.iloc[-1]
            
            # === 1. Entry ATR baseline ===
            entry_atr_p = self._get_trade_entry_atr(trade, dataframe)
            current_atr_p = current_candle.get('atr_p', 0.02)
            
            # === 2. ATR-based stop distance ===
            base_atr_multiplier = self._calculate_atr_multiplier(
                entry_atr_p, current_candle, trade.enter_tag
            )
            base_stop_distance = entry_atr_p * base_atr_multiplier
            
            # === 3. Time decay ===
            hours_held = (current_time - trade.open_date_utc).total_seconds() / 3600
            time_decay_factor = self._calculate_time_decay(hours_held, current_profit)
            
            # === 4. Profit protection ===
            profit_protection = self._calculate_profit_protection(current_profit)
            if profit_protection is not None:
                return profit_protection
                
            # === 5. Trend adjustment ===
            trend_adjustment = self._calculate_trend_adjustment(
                current_candle, trade.is_short, entry_atr_p
            )
            
            # === 6. Final stop distance ===
            final_stop_distance = (base_stop_distance * time_decay_factor * trend_adjustment)
            
            # Keep stop distance within 1% to 8%
            final_stop_distance = max(0.01, min(0.08, final_stop_distance))
            
            # short
            final_stoploss = -final_stop_distance if not trade.is_short else final_stop_distance
            
            # === 7. Optional verbose logging ===
            if self.config.get('verbosity', 0) > 1:
                self._log_stoploss_calculation(
                    pair, trade, current_profit, entry_atr_p, base_atr_multiplier,
                    time_decay_factor, trend_adjustment, final_stoploss
                )
                
            return final_stoploss
            
        except Exception as e:
            logger.error(f"ATR stoploss calculation failed for {pair}: {e}")
            # Fallback fixed stop
            return -0.03 if not trade.is_short else 0.03
    
    def _calculate_signal_quality_score(self, dataframe: DataFrame, signal_mask: pd.Series, signal_type: str) -> pd.Series:
        """
        Score signal quality on a 1-10 scale for sizing and leverage decisions.
        """
        # initialize score
        scores = pd.Series(0.0, index=dataframe.index)
        
        # has signal calculate score
        signal_indices = signal_mask[signal_mask].index
        
        for idx in signal_indices:
            try:
                score = 3.0  # baseline score
                current_data = dataframe.loc[idx]
                
                # === 1. Indicator quality (0-2) ===
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
                
                # === 2. Trend alignment and strength (0-2) ===
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
                
                # === 3. Volume confirmation (0-1.5) ===
                volume_ratio = current_data.get('volume_ratio', 1.0)
                if volume_ratio > 1.5:
                    score += 1.5  # volume
                elif volume_ratio > 1.2:
                    score += 1.0  # volume large
                elif volume_ratio > 1.0:
                    score += 0.5  # volume
                
                # === 4. Volatility regime (0-1) ===
                atr_percentile = dataframe['atr_p'].rolling(50).rank(pct=True).loc[idx]
                if 0.2 <= atr_percentile <= 0.8:
                    score += 1
                elif atr_percentile > 0.9:  # high volatility，risk large
                    score -= 0.5
                
                # === 5. Divergence filter (0-1) ===
                no_bearish_div = not current_data.get('bearish_divergence', False)
                no_bullish_div = not current_data.get('bullish_divergence', False)
                
                if signal_type in ['RSI_Oversold_Bounce'] and no_bearish_div:
                    score += 1
                elif signal_type in ['RSI_Overbought_Fall'] and no_bullish_div:
                    score += 1
                
                # === 6. Price-position quality (0-0.5) ===
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
        Score MACD-based signals on a 1-10 scale.
        """
        # initialize score
        scores = pd.Series(0.0, index=dataframe.index)
        
        # has signal calculate score
        signal_indices = signal_mask[signal_mask].index
        
        for idx in signal_indices:
            try:
                score = 2.0  # baseline MACD score
                current_data = dataframe.loc[idx]
                
                # === 1. MACD cross strength (0-2.5) ===
                macd = current_data.get('macd', 0)
                macd_signal = current_data.get('macd_signal', 0)
                macd_hist = current_data.get('macd_hist', 0)
                
                # Magnitude of the MACD separation
                cross_magnitude = abs(macd - macd_signal)
                if cross_magnitude > 0.002:
                    score += 2.5
                elif cross_magnitude > 0.001:
                    score += 1.5
                elif cross_magnitude > 0.0005:
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
                technical_exit_short |         # technical exit signal
                microstructure_exit_short |    # orderbook/microstructure exit signal
                (volatility_protection & is_bull_market)  # mid
            ),
            'exit_short'
        ] = 1

        # Assign exit tags
        dataframe.loc[strong_reversal_long_exit, 'exit_tag'] = 'strong_reversal'
        dataframe.loc[trend_exhaustion_long, 'exit_tag'] = 'trend_exhaustion'
        dataframe.loc[technical_exit_long, 'exit_tag'] = 'technical_exit'
        dataframe.loc[microstructure_exit_long, 'exit_tag'] = 'microstructure'
        dataframe.loc[volatility_protection, 'exit_tag'] = 'volatility_protection'

        # ==============================
        # Resolve conflicting long/short entry signals
        # ==============================
        
        # Detect bars where both long and short entries were set
        signal_conflict = (dataframe['enter_long'] == 1) & (dataframe['enter_short'] == 1)
        
        # Favor long when trend/momentum conditions lean bullish
        conflict_resolution_favor_long = (
            signal_conflict &
            (
                (dataframe['trend_strength'] > 0) |
                (dataframe['rsi_14'] < 50) |
                (dataframe['macd_hist'] > dataframe['macd_hist'].shift(1))
            )
        )
        
        # Clear the weaker side of the conflict
        dataframe.loc[conflict_resolution_favor_long, 'enter_short'] = 0
        dataframe.loc[signal_conflict & ~conflict_resolution_favor_long, 'enter_long'] = 0
        
        # recalculate after signal
        clean_enter_long = dataframe['enter_long'] == 1
        clean_enter_short = dataframe['enter_short'] == 1
        
        # Cross-close opposing positions only when the new signal is reasonably strong
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
        
        # Use strong entry signals to close the opposite side
        dataframe.loc[strong_bullish_signal, 'exit_short'] = 1
        dataframe.loc[strong_bearish_signal, 'exit_long'] = 1
        
        # Update exit tags for these smart cross exits
        dataframe.loc[
            strong_bullish_signal & (dataframe['exit_short'] == 1),
            'exit_tag'
        ] = 'smart_cross_exit_bullish'
        
        dataframe.loc[
            strong_bearish_signal & (dataframe['exit_long'] == 1),
            'exit_tag' 
        ] = 'smart_cross_exit_bearish'

        # Exit-signal summary
        exit_long_count = dataframe['exit_long'].sum()
        exit_short_count = dataframe['exit_short'].sum()

        if exit_long_count > 0 or exit_short_count > 0:
            logger.info(f"""
📤 Exit signal summary - {metadata['pair']}:
├─ Long exits: {exit_long_count}
├─ Short exits: {exit_short_count}
└─ Time range: {dataframe.index[0]} - {dataframe.index[-1]}
""")

        return dataframe
    
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                          proposed_stake: float, min_stake: Optional[float], max_stake: float,
                          leverage: float, entry_tag: Optional[str], side: str,
                          **kwargs) -> float:
        """Return a dynamic stake amount for the proposed entry."""
        
        try:
            # get most new
            dataframe = self.get_dataframe_with_indicators(pair, self.timeframe)
            if dataframe.empty:
                return proposed_stake
            
            # Current market regime inputs
            market_state = dataframe['market_state'].iloc[-1] if 'market_state' in dataframe.columns else 'sideways'
            volatility = dataframe['atr_p'].iloc[-1] if 'atr_p' in dataframe.columns else 0.02
            
            # === Coin risk tier ===
            coin_risk_tier = self.identify_coin_risk_tier(pair, dataframe)
            
            # Risk-tier stake multipliers
            coin_risk_multipliers = {
                'low_risk': 1.0,
                'medium_risk': 0.7,
                'high_risk': 0.25
            }
            
            # Selected risk-tier multiplier
            coin_risk_multiplier = coin_risk_multipliers.get(coin_risk_tier, 0.7)
            
            # Base portfolio position-size ratio
            position_size_ratio = self.calculate_position_size(current_rate, market_state, pair)
            
            # Available balance
            available_balance = self.wallets.get_free(self.config['stake_currency'])
            
            # === Raw stake calculation ===
            base_calculated_stake = available_balance * position_size_ratio
            
            # Apply risk-tier reduction
            calculated_stake = base_calculated_stake * coin_risk_multiplier
            
            # Dynamic leverage for this setup
            dynamic_leverage = self.calculate_leverage(market_state, volatility, pair, current_time)
            
            # Freqtrade applies leverage separately, so keep stake in stake-currency terms
            leveraged_stake = calculated_stake  # position size
            
            # Preserve the unadjusted position value for logging
            base_position_value = calculated_stake
            
            # Clamp to configured min/max stake bounds
            final_stake = max(min_stake or 0, min(leveraged_stake, max_stake))

            news_signal = self._get_news_signal(pair, current_time)
            if news_signal and news_signal.backend_available:
                news_multiplier = float(news_signal.recommended_stake_multiplier)
                if news_multiplier < 1.0 or (
                    news_multiplier > 1.0 and news_signal.confidence >= 0.50
                ):
                    final_stake = max(
                        min_stake or 0,
                        min(final_stake * news_multiplier, max_stake),
                    )
                    logger.info(
                        "News sentiment adjusted stake for %s: multiplier=%.2f confidence=%.2f fallback=%s reasons=%s",
                        pair,
                        news_multiplier,
                        news_signal.confidence,
                        news_signal.fallback_used,
                        "; ".join(news_signal.reasons),
                    )
            
            # Human-readable risk tier names
            risk_tier_names = {
                'low_risk': '✅ low risk',
                'medium_risk': '⚡ medium risk', 
                'high_risk': '⚠️ high risk'
            }
            
            logger.info(f"""
🎯 Stake sizing - {pair}:
├─ Market state: {market_state}
├─ Risk tier: {risk_tier_names.get(coin_risk_tier, coin_risk_tier)}
├─ Base stake: ${base_calculated_stake:.2f} ({position_size_ratio:.2%} of free balance)
├─ Risk adjustment: {coin_risk_multiplier:.2f}x ({coin_risk_tier})
├─ Risk-adjusted stake: ${calculated_stake:.2f}
├─ Dynamic leverage: {dynamic_leverage}x
├─ Final stake: ${final_stake:.2f}
├─ Estimated base-asset size: {final_stake / current_rate:.6f}
└─ Timestamp: {current_time}
""")
            
            # Persist the chosen leverage for downstream logging/risk logic
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
                'rating_reason': f'{market_state} market with {volatility*100:.1f}% volatility'
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
        """Evaluate DCA opportunities with confirmation and risk controls."""
        
        # Respect the configured DCA limit
        if trade.nr_of_successful_entries >= self.max_dca_orders:
            logger.info(f"DCA skipped for {trade.pair}: reached max entries ({self.max_dca_orders})")
            return None
            
        # Load the latest indicators
        dataframe = self.get_dataframe_with_indicators(trade.pair, self.timeframe)
        if dataframe.empty:
            logger.warning(f"DCA check skipped for {trade.pair}: no dataframe available")
            return None
            
        # Ensure the required indicators are present
        required_indicators = ['rsi_14', 'adx', 'atr_p', 'macd', 'macd_signal', 'volume_ratio', 'trend_strength', 'momentum_score']
        missing_indicators = [indicator for indicator in required_indicators if indicator not in dataframe.columns]
        
        if missing_indicators:
            logger.warning(f"DCA check skipped for {trade.pair}: missing indicators {missing_indicators}")
            return None
            
        # Current and previous indicator snapshots
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
        
        # Trade state metrics
        entry_price = trade.open_rate
        price_deviation = abs(current_rate - entry_price) / entry_price
        hold_time = current_time - trade.open_date_utc
        hold_hours = hold_time.total_seconds() / 3600
        
        # === DCA opportunity analysis ===
        
        dca_decision = self._analyze_dca_opportunity(
            trade, current_rate, current_profit, price_deviation,
            current_data, prev_data, hold_hours, market_state
        )
        
        if dca_decision['should_dca']:
            # Calculate the proposed DCA amount
            dca_amount = self._calculate_smart_dca_amount(
                trade, dca_decision, current_data, market_state
            )
            
            # Final risk validation
            risk_check = self._dca_risk_validation(trade, dca_amount, current_data)
            
            if risk_check['approved']:
                final_dca_amount = risk_check['adjusted_amount']
                
                # Log the DCA decision
                self._log_dca_decision(
                    trade, current_rate, current_profit, price_deviation,
                    dca_decision, final_dca_amount, current_data
                )
                
                # Track DCA performance for later evaluation
                self.track_dca_performance(trade, dca_decision['dca_type'], final_dca_amount)
                
                return final_dca_amount
            else:
                logger.warning(f"DCA risk validation rejected {trade.pair}: {risk_check['reason']}")
                return None
        
        return None
    
    # Simplified DCA opportunity analysis
    def _analyze_dca_opportunity(self, trade: Trade, current_rate: float, 
                               current_profit: float, price_deviation: float,
                               current_data: dict, prev_data: dict, 
                               hold_hours: float, market_state: str) -> dict:
        """Analyze whether the current trade qualifies for a DCA add."""
        
        decision = {
            'should_dca': False,
            'dca_type': None,
            'confidence': 0.0,
            'risk_level': 'high',
            'technical_reasons': [],
            'market_conditions': {}
        }
        
        try:
            # === Basic DCA trigger ===
            basic_trigger_met = (
                price_deviation > self.dca_price_deviation and
                current_profit < -0.03 and
                hold_hours > 0.5
            )
            
            if not basic_trigger_met:
                return decision
            
            # === Direction-specific DCA setups ===
            
            if not trade.is_short:
                # === Long-side DCA setups ===
                
                # 1. Oversold reversal DCA
                oversold_dca = (
                    current_rate < trade.open_rate and
                    current_data.get('rsi_14', 50) < 35 and
                    current_data.get('bb_position', 0.5) < 0.2 and
                    current_data.get('momentum_score', 0) > prev_data.get('momentum_score', 0)
                )
                
                if oversold_dca:
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'OVERSOLD_REVERSAL_DCA',
                        'confidence': 0.8,
                        'risk_level': 'low'
                    })
                    decision['technical_reasons'].append(f"RSI oversold ({current_data.get('rsi_14', 50):.1f})")
                
                # 2. Support-level DCA
                elif (current_data.get('close', 0) > current_data.get('ema_50', 0) and
                      abs(current_rate - current_data.get('ema_21', 0)) / current_rate < 0.02 and
                      current_data.get('adx', 25) > 20):
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'SUPPORT_LEVEL_DCA',
                        'confidence': 0.7,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append("EMA21 acting as support")
                
                # 3. Trend-continuation DCA
                elif (current_data.get('trend_strength', 50) > 30 and
                      current_data.get('adx', 25) > 25 and
                      current_data.get('signal_strength', 0) > 0):
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'TREND_CONTINUATION_DCA',
                        'confidence': 0.6,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append(f"trend continuation (strength {current_data.get('trend_strength', 50):.0f})")
                
                # 4. Volume-confirmed DCA
                elif (current_data.get('volume_ratio', 1) > 1.2 and
                      current_data.get('ob_depth_imbalance', 0) > 0.1):
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'VOLUME_CONFIRMED_DCA',
                        'confidence': 0.5,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append(f"volume confirmation ({current_data.get('volume_ratio', 1):.1f}x)")
                
            else:
                # === Short-side DCA setups ===
                
                # 1. Overbought rejection DCA
                overbought_dca = (
                    current_rate > trade.open_rate and
                    current_data.get('rsi_14', 50) > 65 and
                    current_data.get('bb_position', 0.5) > 0.8 and
                    current_data.get('momentum_score', 0) < prev_data.get('momentum_score', 0)
                )
                
                if overbought_dca:
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'OVERBOUGHT_REJECTION_DCA',
                        'confidence': 0.8,
                        'risk_level': 'low'
                    })
                    decision['technical_reasons'].append(f"RSI overbought ({current_data.get('rsi_14', 50):.1f})")
                
                # 2. Resistance-level DCA
                elif (current_data.get('close', 0) < current_data.get('ema_50', 0) and
                      abs(current_rate - current_data.get('ema_21', 0)) / current_rate < 0.02 and
                      current_data.get('adx', 25) > 20):
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'RESISTANCE_LEVEL_DCA',
                        'confidence': 0.7,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append("EMA21 acting as resistance")
                
                # 3. Trend-continuation DCA
                elif (current_data.get('trend_strength', 50) < -30 and
                      current_data.get('adx', 25) > 25 and
                      current_data.get('signal_strength', 0) < 0):
                    
                    decision.update({
                        'should_dca': True,
                        'dca_type': 'TREND_CONTINUATION_DCA_SHORT',
                        'confidence': 0.6,
                        'risk_level': 'medium'
                    })
                    decision['technical_reasons'].append(f"downtrend continuation (strength {current_data.get('trend_strength', 50):.0f})")
            
            # Market conditions used to veto or discount DCA
            decision['market_conditions'] = {
                'market_state': market_state,
                'volatility_acceptable': current_data.get('atr_p', 0.02) < 0.06,
                'liquidity_sufficient': current_data.get('ob_market_quality', 0.5) > 0.3,
                'spread_reasonable': current_data.get('ob_spread_pct', 0.1) < 0.4,
                'trend_not_reversing': abs(current_data.get('trend_strength', 50)) > 20
            }
            
            # Reject or downgrade DCA when multiple market conditions are unfavorable
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
                decision['confidence'] *= 0.7
                decision['risk_level'] = 'high'
                
        except Exception as e:
            logger.error(f"DCA analysis failed for {trade.pair}: {e}")
            decision['should_dca'] = False
            
        return decision
    
    def _calculate_smart_dca_amount(self, trade: Trade, dca_decision: dict, 
                                  current_data: dict, market_state: str) -> float:
        """Calculate a risk-aware DCA amount."""
        
        try:
            # Base DCA sizing inputs
            base_amount = trade.stake_amount
            entry_count = trade.nr_of_successful_entries + 1
            
            # DCA type multipliers
            dca_type_multipliers = {
                'OVERSOLD_REVERSAL_DCA': 1.5,
                'OVERBOUGHT_REJECTION_DCA': 1.5,
                'SUPPORT_LEVEL_DCA': 1.3,
                'RESISTANCE_LEVEL_DCA': 1.3,
                'TREND_CONTINUATION_DCA': 1.2,
                'TREND_CONTINUATION_DCA_SHORT': 1.2,  # bearish trend
                'VOLUME_CONFIRMED_DCA': 1.1
            }
            
            type_multiplier = dca_type_multipliers.get(dca_decision['dca_type'], 1.0)
            
            # Confidence multiplier
            confidence_multiplier = 0.5 + (dca_decision['confidence'] * 0.8)  # 0.5-1.3 range
            
            # Market-state multiplier
            market_multipliers = {
                'strong_uptrend': 1.4,
                'strong_downtrend': 1.4,
                'mild_uptrend': 1.2,
                'mild_downtrend': 1.2,
                'sideways': 1.0,
                'volatile': 0.7,
                'consolidation': 1.1
            }
            market_multiplier = market_multipliers.get(market_state, 1.0)
            
            # Reduce size on later DCA entries
            entry_decay = max(0.6, 1.0 - (entry_count - 1) * 0.15)
            
            # Final raw DCA amount
            total_multiplier = (type_multiplier * confidence_multiplier * 
                              market_multiplier * entry_decay)
            
            calculated_dca = base_amount * total_multiplier
            
            # Available balance constraint
            available_balance = self.wallets.get_free(self.config['stake_currency'])
            
            # Cap DCA size by risk level
            max_dca_ratio = {
                'low': 0.15,
                'medium': 0.10,
                'high': 0.05
            }
            
            max_ratio = max_dca_ratio.get(dca_decision['risk_level'], 0.05)
            max_dca_amount = available_balance * max_ratio
            
            final_dca = min(calculated_dca, max_dca_amount, max_stake or float('inf'))
            
            return max(min_stake or 10, final_dca)
            
        except Exception as e:
            logger.error(f"DCA sizing failed for {trade.pair}: {e}")
            return trade.stake_amount * 0.5  # default value
    
    def _dca_risk_validation(self, trade: Trade, dca_amount: float, current_data: dict) -> dict:
        """Run final risk checks before approving a DCA amount."""
        
        risk_check = {
            'approved': True,
            'adjusted_amount': dca_amount,
            'reason': 'DCA risk check passed',
            'risk_factors': []
        }
        
        try:
            # 1. Position-size risk check
            available_balance = self.wallets.get_free(self.config['stake_currency'])
            total_exposure = trade.stake_amount + dca_amount
            exposure_ratio = total_exposure / available_balance
            
            if exposure_ratio > 0.4:
                adjustment = 0.4 / exposure_ratio
                risk_check['adjusted_amount'] = dca_amount * adjustment
                risk_check['risk_factors'].append(f'exposure too large, reduced to {adjustment:.1%}')
            
            # 2. Additional scaling after several DCA entries
            if trade.nr_of_successful_entries >= 3:
                risk_check['adjusted_amount'] *= 0.7
                risk_check['risk_factors'].append('multiple prior DCA entries')
            
            # 3. risk check
            if current_data.get('atr_p', 0.02) > 0.05:  # high volatility
                risk_check['adjusted_amount'] *= 0.8
                risk_check['risk_factors'].append('high volatility risk adjustment')
            
            # 4. Portfolio drawdown
            if hasattr(self, 'current_drawdown') and self.current_drawdown > 0.08:
                risk_check['adjusted_amount'] *= 0.6
                risk_check['risk_factors'].append('portfolio drawdown adjustment')
            
            # 5. Enforce a minimum meaningful DCA size
            min_meaningful_dca = trade.stake_amount * 0.2
            if risk_check['adjusted_amount'] < min_meaningful_dca:
                risk_check['approved'] = False
                risk_check['reason'] = f'DCA amount too small; minimum meaningful add is ${min_meaningful_dca:.2f}'
            
        except Exception as e:
            risk_check['approved'] = False
            risk_check['reason'] = f'DCA risk validation failed: {e}'
            
        return risk_check
    
    def _log_dca_decision(self, trade: Trade, current_rate: float, current_profit: float,
                         price_deviation: float, dca_decision: dict, dca_amount: float,
                         current_data: dict):
        """Log the details behind an approved DCA decision."""
        
        try:
            hold_time = datetime.now(timezone.utc) - trade.open_date_utc
            hold_hours = hold_time.total_seconds() / 3600
            
            dca_log = f"""
==================== DCA Decision ====================
Time: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} | Pair: {trade.pair}
Entry count: {trade.nr_of_successful_entries + 1} / {self.max_dca_orders}

📊 Trade state:
├─ Entry price: ${trade.open_rate:.6f}
├─ Current price: ${current_rate:.6f}
├─ Price deviation: {price_deviation:.2%}
├─ Current profit: {current_profit:.2%}
├─ Hold time: {hold_hours:.1f} hours
├─ Direction: {'short' if trade.is_short else 'long'}
├─ Original stake: ${trade.stake_amount:.2f}

🎯 DCA decision:
├─ DCA type: {dca_decision['dca_type']}
├─ Confidence: {dca_decision['confidence']:.1%}
├─ Risk level: {dca_decision['risk_level']}
├─ Technical reasons: {' | '.join(dca_decision['technical_reasons'])}

📋 Indicators:
├─ RSI(14): {current_data.get('rsi_14', 50):.1f}
├─ trend strength: {current_data.get('trend_strength', 50):.0f}/100
├─ momentum score: {current_data.get('momentum_score', 0):.3f}
├─ ADX: {current_data.get('adx', 25):.1f}
├─ volume: {current_data.get('volume_ratio', 1):.1f}x
├─ Bollinger Bands: {current_data.get('bb_position', 0.5):.2f}
├─ signal strength: {current_data.get('signal_strength', 0):.1f}

💰 DCA sizing:
├─ Original stake: ${trade.stake_amount:.2f}
├─ DCA amount: ${dca_amount:.2f}
├─ Add size vs original: {(dca_amount/trade.stake_amount)*100:.0f}%
├─ New total stake: ${trade.stake_amount + dca_amount:.2f}

🌊 Market conditions:
├─ Market state: {dca_decision['market_conditions'].get('market_state', 'unknown')}
├─ Volatility: {'acceptable' if dca_decision['market_conditions'].get('volatility_acceptable', False) else 'high'}
├─ Liquidity: {'sufficient' if dca_decision['market_conditions'].get('liquidity_sufficient', False) else 'weak'}
├─ Spread: {'reasonable' if dca_decision['market_conditions'].get('spread_reasonable', False) else 'wide'}

=================================================="""
            
            logger.info(dca_log)
            
        except Exception as e:
            logger.error(f"DCA logging failed for {trade.pair}: {e}")
    
    def track_dca_performance(self, trade: Trade, dca_type: str, dca_amount: float):
        """Track DCA usage for later performance review."""
        try:
            # Update aggregate DCA counters
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
                
                # Treat a 0.5% recovery back above the breakout line as meaningful
                if (recent_low < supertrend and 
                    current_rate > supertrend and 
                    current_recovery > 0.005):  # 0.5%
                    return True
                    
                # Bollinger Bands breakout
                if (recent_data['low'].min() < bb_lower and 
                    current_rate > bb_lower and
                    current_rate > recent_data['close'].iloc[-3]):  # stronger than the close 3 candles ago
                    return True
            
            # === bearish breakout ===
            else:
                # check breakout resistance after fast
                recent_high = recent_data['high'].max()
                current_pullback = (recent_high - current_rate) / recent_high
                
                # Treat a 0.5% pullback back below the breakdown line as meaningful
                if (recent_high > supertrend and 
                    current_rate < supertrend and 
                    current_pullback > 0.005):  # 0.5%
                    return True
                
                # Bollinger Bands breakout
                if (recent_data['high'].max() > bb_upper and 
                    current_rate < bb_upper and
                    current_rate < recent_data['close'].iloc[-3]):  # weaker than the close 3 candles ago
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"breakout: {e}")
            return False
    
    # Stoploss confirmation helpers
    
    def _log_trend_protection(self, pair: str, trade: Trade, current_rate: float, 
                            current_profit: float, dataframe: DataFrame):
        """Log details for a trend-protection decision."""
        
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
            
            # Placeholder for a future adjusted stoploss value
            suggested_new_stoploss = self.stoploss
            
            # Reserved for integration with an external decision logger
            pass
            
        except Exception as e:
            logger.warning(f"Trend-protection logging failed: {e}")
    
    def _log_false_breakout_protection(self, pair: str, trade: Trade, 
                                     current_rate: float, dataframe: DataFrame):
        """Log false-breakout protection events."""
        
        try:
            logger.info(f"🚫 False-breakout protection triggered for {pair}; position size reduced to 50%")
            
        except Exception as e:
            logger.warning(f"False-breakout logging failed: {e}")
    
    # ===== new =====
    
    # `_calculate_structure_based_stop` removed in this simplified version
    # `calculate_atr_stop_multiplier` removed in this simplified version
    
    # `calculate_trend_stop_adjustment` removed in this simplified version
    
    # `calculate_volatility_cluster_stop` removed in this simplified version
    
    # `calculate_time_decay_stop` removed in this simplified version
    
    # `calculate_profit_protection_stop` removed in this simplified version
    
    # `calculate_volume_stop_adjustment` removed in this simplified version
    
    # `calculate_microstructure_stop` removed in this simplified version
    
    # `apply_stoploss_limits` removed in this simplified version
    
    # `get_enhanced_technical_stoploss` removed in this simplified version
    
    # `custom_exit` removed; ROI exits are used instead
    
    # `_get_detailed_exit_reason` removed in this simplified version
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float,
                          rate: float, time_in_force: str, current_time: datetime,
                          entry_tag: Optional[str], side: str, **kwargs) -> bool:
        """Confirm whether a trade entry should be allowed."""
        
        try:
            # Final entry checks
            
            # 1. Reserved for time-based filters
            
            # 2. Orderbook quality check
            orderbook_data = self.get_market_orderbook(pair)
            if orderbook_data['spread_pct'] > 0.3:
                logger.warning(f"Rejecting trade entry for {pair}: spread too wide")
                return False

            news_signal = self._get_news_signal(pair, current_time)
            if news_signal and news_signal.backend_available and news_signal.block_entries:
                logger.warning(
                    "Blocking trade entry for %s due to news sentiment: sentiment=%.3f impact=%.3f confidence=%.2f reasons=%s",
                    pair,
                    news_signal.sentiment_score,
                    news_signal.impact_score,
                    news_signal.confidence,
                    "; ".join(news_signal.reasons),
                )
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
