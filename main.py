#!/usr/bin/env python3
"""
INTELLIGENT MULTI-STRATEGY BINANCE BOT
Uses 12+ strategies simultaneously with AI-like decision making
"""

import os
import sys
import json
import hmac
import hashlib
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from enum import Enum
import logging
from urllib.parse import urlencode
from collections import defaultdict

# ============================================================================
# FIREBASE SETUP
# ============================================================================

import firebase_admin
from firebase_admin import credentials, db

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # API Keys
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY', '')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY', '')
    BASE_URL = 'https://testnet.binance.vision'
    DATABASE_URL = 'https://surebet-ke-default-rtdb.firebaseio.com/'
    FIREBASE_CREDENTIALS = os.getenv('FIREBASE_CREDENTIALS', '{}')
    
    # Trading Parameters
    TIMEFRAMES = ['5m', '15m', '1h', '4h']  # Multi-timeframe analysis
    PRIMARY_TIMEFRAME = '15m'
    BASE_TRADE_AMOUNT_USDT = 100
    MAX_POSITIONS = 5
    MAX_POSITION_SIZE_USDT = 500
    MIN_POSITION_SIZE_USDT = 50
    
    # Risk Management
    STOP_LOSS_ATR_MULTIPLIER = 2.0
    TAKE_PROFIT_ATR_MULTIPLIER = 3.0
    TRAILING_STOP_ACTIVATION = 0.03  # 3% profit activates trailing stop
    TRAILING_STOP_DISTANCE = 0.015   # 1.5% trailing distance
    MAX_DRAWDOWN_PERCENT = 15
    RISK_PER_TRADE_PERCENT = 2
    
    # Strategy Weights (Dynamic based on market conditions)
    STRATEGY_WEIGHTS = {
        'trend_following': {
            'ema_crossover': 0.15,
            'ichimoku_cloud': 0.12,
            'adx_trend': 0.08,
            'supertrend': 0.10,
            'parabolic_sar': 0.05
        },
        'momentum': {
            'rsi': 0.10,
            'macd': 0.12,
            'stochastic_rsi': 0.08,
            'awesome_oscillator': 0.05,
            'cci': 0.05
        },
        'volatility': {
            'bollinger_bands': 0.10,
            'atr_breakout': 0.08,
            'keltner_channels': 0.07,
            'donchian_channels': 0.05
        },
        'volume': {
            'volume_surge': 0.05,
            'obv': 0.03,
            'mfi': 0.05,
            'vwap': 0.07
        },
        'pattern_recognition': {
            'support_resistance': 0.10,
            'fibonacci_levels': 0.05,
            'pivot_points': 0.05,
            'candlestick_patterns': 0.08
        }
    }
    
    # Signal Thresholds
    STRONG_BUY_THRESHOLD = 0.75
    BUY_THRESHOLD = 0.65
    NEUTRAL_THRESHOLD = 0.35
    SELL_THRESHOLD = -0.55
    STRONG_SELL_THRESHOLD = -0.70
    
    # Market Regime Detection
    TRENDING_ADX_THRESHOLD = 25
    VOLATILE_ATR_PERCENT = 3
    OVERBOUGHT_RSI = 70
    OVERSOLD_RSI = 30
    
    # Telegram
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID', '')

# ============================================================================
# ENUMS AND DATA CLASSES
# ============================================================================

class SignalType(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    WEAK_BUY = "WEAK_BUY"
    NEUTRAL = "NEUTRAL"
    WEAK_SELL = "WEAK_SELL"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"

class MarketRegime(Enum):
    STRONG_TREND_UP = "STRONG_TREND_UP"
    WEAK_TREND_UP = "WEAK_TREND_UP"
    RANGING = "RANGING"
    WEAK_TREND_DOWN = "WEAK_TREND_DOWN"
    STRONG_TREND_DOWN = "STRONG_TREND_DOWN"
    HIGH_VOLATILITY = "HIGH_VOLATILITY"
    LOW_VOLATILITY = "LOW_VOLATILITY"

@dataclass
class StrategySignal:
    name: str
    signal: float  # -1 to 1
    confidence: float  # 0 to 1
    reasoning: str
    metadata: Dict = field(default_factory=dict)

@dataclass
class CombinedAnalysis:
    symbol: str
    timestamp: datetime
    price: float
    market_regime: MarketRegime
    signals: List[StrategySignal]
    composite_score: float
    confidence: float
    recommendation: SignalType
    position_size_suggestion: float
    stop_loss: float
    take_profit: float
    reasoning_summary: List[str]

# ============================================================================
# TECHNICAL INDICATORS - ALL STRATEGIES
# ============================================================================

class TechnicalAnalysis:
    """Complete technical analysis with all strategies"""
    
    @staticmethod
    def calculate_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
        """Calculate ALL indicators for comprehensive analysis"""
        df = df.copy()
        
        # === TREND FOLLOWING STRATEGIES ===
        
        # 1. EMA Crossover (9/21/50/200)
        for period in [9, 21, 50, 200]:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        
        df['ema_trend'] = ((df['ema_9'] > df['ema_21']) & 
                          (df['ema_21'] > df['ema_50']) & 
                          (df['ema_50'] > df['ema_200'])).astype(int)
        
        # 2. Ichimoku Cloud (Full Calculation)
        # Tenkan-sen (Conversion Line)
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        df['ichimoku_tenkan'] = (high_9 + low_9) / 2
        
        # Kijun-sen (Base Line)
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        df['ichimoku_kijun'] = (high_26 + low_26) / 2
        
        # Senkou Span A (Leading Span A)
        df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
        
        # Senkou Span B (Leading Span B)
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        df['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(26)
        
        # Chikou Span (Lagging Span)
        df['ichimoku_chikou'] = df['close'].shift(-26)
        
        # Cloud Analysis
        df['price_above_cloud'] = (df['close'] > df['ichimoku_senkou_a']) & (df['close'] > df['ichimoku_senkou_b'])
        df['price_below_cloud'] = (df['close'] < df['ichimoku_senkou_a']) & (df['close'] < df['ichimoku_senkou_b'])
        df['cloud_bullish'] = df['ichimoku_senkou_a'] > df['ichimoku_senkou_b']
        df['tk_cross_bullish'] = (df['ichimoku_tenkan'] > df['ichimoku_kijun']) & (df['ichimoku_tenkan'].shift(1) <= df['ichimoku_kijun'].shift(1))
        
        # 3. ADX (Average Directional Index) with DI+/DI-
        df['tr'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift()),
                abs(df['low'] - df['close'].shift())
            )
        )
        df['atr'] = df['tr'].rolling(14).mean()
        
        df['up_move'] = df['high'] - df['high'].shift()
        df['down_move'] = df['low'].shift() - df['low']
        
        df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
        df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
        
        df['plus_di'] = 100 * (df['plus_dm'].ewm(span=14).mean() / df['atr'])
        df['minus_di'] = 100 * (df['minus_dm'].ewm(span=14).mean() / df['atr'])
        df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
        df['adx'] = df['dx'].rolling(14).mean()
        
        df['adx_trending'] = df['adx'] > Config.TRENDING_ADX_THRESHOLD
        df['adx_trend_bullish'] = (df['adx'] > 20) & (df['plus_di'] > df['minus_di'])
        
        # 4. SuperTrend
        atr_mult = 3
        df['st_upper'] = df['high'] - (atr_mult * df['atr'])
        df['st_lower'] = df['low'] + (atr_mult * df['atr'])
        df['supertrend'] = 0
        df['supertrend_signal'] = 0
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] > df['st_upper'].iloc[i-1]:
                df.loc[df.index[i], 'supertrend'] = df['st_lower'].iloc[i]
                df.loc[df.index[i], 'supertrend_signal'] = 1
            elif df['close'].iloc[i] < df['st_lower'].iloc[i-1]:
                df.loc[df.index[i], 'supertrend'] = df['st_upper'].iloc[i]
                df.loc[df.index[i], 'supertrend_signal'] = -1
            else:
                df.loc[df.index[i], 'supertrend'] = df['supertrend'].iloc[i-1]
                df.loc[df.index[i], 'supertrend_signal'] = df['supertrend_signal'].iloc[i-1]
        
        # 5. Parabolic SAR
        df['psar'] = df['close'].copy()
        df['psar_signal'] = 0
        af = 0.02
        max_af = 0.20
        psar = df['low'].iloc[0]
        ep = df['high'].iloc[0]
        trend = 1
        
        for i in range(1, len(df)):
            prev_psar = psar
            if trend == 1:
                psar = prev_psar + af * (ep - prev_psar)
                if df['low'].iloc[i] < psar:
                    trend = -1
                    psar = ep
                    ep = df['low'].iloc[i]
                    af = 0.02
                else:
                    if df['high'].iloc[i] > ep:
                        ep = df['high'].iloc[i]
                        af = min(af + 0.02, max_af)
            else:
                psar = prev_psar - af * (prev_psar - ep)
                if df['high'].iloc[i] > psar:
                    trend = 1
                    psar = ep
                    ep = df['high'].iloc[i]
                    af = 0.02
                else:
                    if df['low'].iloc[i] < ep:
                        ep = df['low'].iloc[i]
                        af = min(af + 0.02, max_af)
            
            df.loc[df.index[i], 'psar'] = psar
            df.loc[df.index[i], 'psar_signal'] = trend
        
        # === MOMENTUM STRATEGIES ===
        
        # 6. RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # 7. MACD (Multiple timeframes)
        for fast, slow, signal in [(12, 26, 9), (5, 35, 5)]:
            exp1 = df['close'].ewm(span=fast, adjust=False).mean()
            exp2 = df['close'].ewm(span=slow, adjust=False).mean()
            df[f'macd_{fast}_{slow}'] = exp1 - exp2
            df[f'macd_signal_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'].ewm(span=signal, adjust=False).mean()
            df[f'macd_hist_{fast}_{slow}'] = df[f'macd_{fast}_{slow}'] - df[f'macd_signal_{fast}_{slow}']
        
        # 8. Stochastic RSI
        period = 14
        smooth_k = 3
        smooth_d = 3
        
        rsi_min = df['rsi'].rolling(window=period).min()
        rsi_max = df['rsi'].rolling(window=period).max()
        df['stoch_rsi'] = (df['rsi'] - rsi_min) / (rsi_max - rsi_min)
        df['stoch_rsi_k'] = df['stoch_rsi'].rolling(window=smooth_k).mean() * 100
        df['stoch_rsi_d'] = df['stoch_rsi_k'].rolling(window=smooth_d).mean()
        
        # 9. Awesome Oscillator
        df['ao'] = df['high'].rolling(5).mean() - df['low'].rolling(34).mean()
        df['ao_signal'] = ((df['ao'] > 0) & (df['ao'].shift(1) <= 0)).astype(int)
        
        # 10. CCI (Commodity Channel Index)
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(20).mean()
        mad = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean())
        df['cci'] = (tp - sma_tp) / (0.015 * mad)
        
        # === VOLATILITY STRATEGIES ===
        
        # 11. Bollinger Bands (Multiple deviations)
        for period, std_dev in [(20, 2), (20, 3)]:
            df[f'bb_mid_{period}'] = df['close'].rolling(window=period).mean()
            bb_std = df['close'].rolling(window=period).std()
            df[f'bb_upper_{period}_{std_dev}'] = df[f'bb_mid_{period}'] + (bb_std * std_dev)
            df[f'bb_lower_{period}_{std_dev}'] = df[f'bb_mid_{period}'] - (bb_std * std_dev)
        
        df['bb_position'] = (df['close'] - df['bb_lower_20_2']) / (df['bb_upper_20_2'] - df['bb_lower_20_2'])
        df['bb_squeeze'] = (df['bb_upper_20_2'] - df['bb_lower_20_2']) / df['bb_mid_20'] < 0.05
        
        # 12. Keltner Channels
        df['kc_mid'] = df['close'].ewm(span=20).mean()
        df['kc_atr'] = df['atr'] * 1.5
        df['kc_upper'] = df['kc_mid'] + df['kc_atr']
        df['kc_lower'] = df['kc_mid'] - df['kc_atr']
        
        # 13. Donchian Channels
        df['dc_upper'] = df['high'].rolling(20).max()
        df['dc_lower'] = df['low'].rolling(20).min()
        df['dc_mid'] = (df['dc_upper'] + df['dc_lower']) / 2
        
        # === VOLUME STRATEGIES ===
        
        # 14. Volume Analysis
        df['volume_sma_20'] = df['volume'].rolling(20).mean()
        df['volume_sma_50'] = df['volume'].rolling(50).mean()
        df['volume_ratio'] = df['volume'] / df['volume_sma_20']
        df['volume_trend'] = df['volume_sma_20'] > df['volume_sma_50']
        
        # 15. OBV (On-Balance Volume)
        df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv_sma'] = df['obv'].rolling(20).mean()
        
        # 16. MFI (Money Flow Index)
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(14).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(14).sum()
        
        mfi_ratio = positive_flow / negative_flow
        df['mfi'] = 100 - (100 / (1 + mfi_ratio))
        
        # 17. VWAP
        df['vwap'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum() / df['volume'].cumsum()
        df['vwap_deviation'] = (df['close'] - df['vwap']) / df['vwap'] * 100
        
        # === PATTERN RECOGNITION ===
        
        # 18. Support/Resistance Levels
        df['resistance'] = df['high'].rolling(20).max()
        df['support'] = df['low'].rolling(20).min()
        df['sr_position'] = (df['close'] - df['support']) / (df['resistance'] - df['support'])
        
        # 19. Fibonacci Levels
        recent_high = df['high'].rolling(50).max()
        recent_low = df['low'].rolling(50).min()
        diff = recent_high - recent_low
        
        for level in [0.236, 0.382, 0.5, 0.618, 0.786]:
            df[f'fib_{level}'] = recent_high - diff * level
        
        # 20. Candlestick Patterns
        df['body'] = df['close'] - df['open']
        df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
        df['body_pct'] = abs(df['body']) / (df['high'] - df['low'])
        
        # Doji
        df['doji'] = df['body_pct'] < 0.1
        
        # Hammer / Shooting Star
        df['hammer'] = (df['lower_wick'] > df['body'].abs() * 2) & (df['upper_wick'] < df['body'].abs() * 0.5)
        df['shooting_star'] = (df['upper_wick'] > df['body'].abs() * 2) & (df['lower_wick'] < df['body'].abs() * 0.5)
        
        # Engulfing
        df['bullish_engulfing'] = (df['close'] > df['open']) & (df['close'].shift(1) < df['open'].shift(1)) & \
                                  (df['close'] > df['open'].shift(1)) & (df['open'] < df['close'].shift(1))
        df['bearish_engulfing'] = (df['close'] < df['open']) & (df['close'].shift(1) > df['open'].shift(1)) & \
                                  (df['close'] < df['open'].shift(1)) & (df['open'] > df['close'].shift(1))
        
        return df

# ============================================================================
# INTELLIGENT SIGNAL GENERATOR
# ============================================================================

class SignalGenerator:
    """Generate signals from all strategies with intelligent weighting"""
    
    def __init__(self):
        self.ta = TechnicalAnalysis()
    
    def analyze_all_strategies(self, df: pd.DataFrame, symbol: str) -> CombinedAnalysis:
        """Run all strategies and combine signals intelligently"""
        
        # Calculate all indicators
        df = self.ta.calculate_all_indicators(df)
        latest = df.iloc[-1]
        prev = df.iloc[-2]
        
        signals = []
        reasoning = []
        
        # Detect market regime first
        market_regime = self._detect_market_regime(df, latest)
        
        # === TREND FOLLOWING SIGNALS ===
        
        # 1. EMA Crossover Signal
        ema_score = self._evaluate_ema_crossover(latest, prev)
        signals.append(StrategySignal(
            name="EMA Crossover",
            signal=ema_score,
            confidence=0.8 if latest['ema_trend'] == 1 else 0.5,
            reasoning=self._get_ema_reasoning(latest),
            metadata={'ema_9': latest['ema_9'], 'ema_21': latest['ema_21'], 'ema_50': latest['ema_50']}
        ))
        
        # 2. Ichimoku Cloud Signal
        ichimoku_score = self._evaluate_ichimoku(latest, prev)
        signals.append(StrategySignal(
            name="Ichimoku Cloud",
            signal=ichimoku_score,
            confidence=self._calculate_ichimoku_confidence(latest),
            reasoning=self._get_ichimoku_reasoning(latest),
            metadata={'cloud_bullish': latest['cloud_bullish'], 'tk_cross': latest['tk_cross_bullish']}
        ))
        
        # 3. ADX Trend Signal
        adx_score = self._evaluate_adx(latest, prev)
        signals.append(StrategySignal(
            name="ADX Trend",
            signal=adx_score,
            confidence=min(latest['adx'] / 50, 1.0),
            reasoning=f"ADX: {latest['adx']:.1f}, {'Strong' if latest['adx'] > 25 else 'Weak'} trend",
            metadata={'adx': latest['adx'], 'plus_di': latest['plus_di'], 'minus_di': latest['minus_di']}
        ))
        
        # 4. SuperTrend Signal
        supertrend_score = latest['supertrend_signal'] if not pd.isna(latest['supertrend_signal']) else 0
        signals.append(StrategySignal(
            name="SuperTrend",
            signal=supertrend_score,
            confidence=0.7,
            reasoning=f"SuperTrend: {'Bullish' if supertrend_score > 0 else 'Bearish'}",
            metadata={'supertrend': latest['supertrend']}
        ))
        
        # 5. Parabolic SAR
        psar_score = 1 if latest['close'] > latest['psar'] else -1
        signals.append(StrategySignal(
            name="Parabolic SAR",
            signal=psar_score,
            confidence=0.6,
            reasoning=f"Price {'above' if psar_score > 0 else 'below'} SAR",
            metadata={'psar': latest['psar']}
        ))
        
        # === MOMENTUM SIGNALS ===
        
        # 6. RSI Signal
        rsi_score = self._evaluate_rsi(latest)
        signals.append(StrategySignal(
            name="RSI",
            signal=rsi_score,
            confidence=0.7 if latest['rsi'] < 30 or latest['rsi'] > 70 else 0.4,
            reasoning=self._get_rsi_reasoning(latest),
            metadata={'rsi': latest['rsi']}
        ))
        
        # 7. MACD Signal
        macd_score = self._evaluate_macd(latest, prev)
        signals.append(StrategySignal(
            name="MACD",
            signal=macd_score,
            confidence=0.75,
            reasoning=self._get_macd_reasoning(latest),
            metadata={'macd': latest['macd_12_26'], 'signal': latest['macd_signal_12_26']}
        ))
        
        # 8. Stochastic RSI Signal
        stoch_score = self._evaluate_stochastic_rsi(latest)
        signals.append(StrategySignal(
            name="Stochastic RSI",
            signal=stoch_score,
            confidence=0.65,
            reasoning=f"Stoch RSI: {latest['stoch_rsi_k']:.1f}",
            metadata={'stoch_k': latest['stoch_rsi_k'], 'stoch_d': latest['stoch_rsi_d']}
        ))
        
        # 9. Awesome Oscillator
        ao_score = 1 if latest['ao_signal'] == 1 else (0.5 if latest['ao'] > 0 else -0.5)
        signals.append(StrategySignal(
            name="Awesome Oscillator",
            signal=ao_score,
            confidence=0.6,
            reasoning=f"AO: {latest['ao']:.2f}",
            metadata={'ao': latest['ao']}
        ))
        
        # 10. CCI Signal
        cci_score = self._evaluate_cci(latest)
        signals.append(StrategySignal(
            name="CCI",
            signal=cci_score,
            confidence=0.55,
            reasoning=f"CCI: {latest['cci']:.1f}",
            metadata={'cci': latest['cci']}
        ))
        
        # === VOLATILITY SIGNALS ===
        
        # 11. Bollinger Bands
        bb_score = self._evaluate_bollinger_bands(latest, prev)
        signals.append(StrategySignal(
            name="Bollinger Bands",
            signal=bb_score,
            confidence=0.7 if latest['bb_squeeze'] else 0.5,
            reasoning=self._get_bb_reasoning(latest),
            metadata={'bb_position': latest['bb_position'], 'squeeze': latest['bb_squeeze']}
        ))
        
        # 12. Keltner Channels
        kc_score = self._evaluate_keltner(latest)
        signals.append(StrategySignal(
            name="Keltner Channels",
            signal=kc_score,
            confidence=0.6,
            reasoning=f"Price vs KC: {((latest['close'] - latest['kc_lower']) / (latest['kc_upper'] - latest['kc_lower']) * 100):.1f}%",
            metadata={'kc_upper': latest['kc_upper'], 'kc_lower': latest['kc_lower']}
        ))
        
        # 13. Donchian Channels
        dc_score = self._evaluate_donchian(latest)
        signals.append(StrategySignal(
            name="Donchian Channels",
            signal=dc_score,
            confidence=0.6,
            reasoning=f"DC Breakout: {'Upper' if latest['close'] > latest['dc_upper'] else 'Lower' if latest['close'] < latest['dc_lower'] else 'None'}",
            metadata={'dc_upper': latest['dc_upper'], 'dc_lower': latest['dc_lower']}
        ))
        
        # === VOLUME SIGNALS ===
        
        # 14. Volume Surge
        volume_score = self._evaluate_volume(latest)
        signals.append(StrategySignal(
            name="Volume Analysis",
            signal=volume_score * 0.3,  # Volume is a confirmation, not primary signal
            confidence=min(latest['volume_ratio'] / 3, 1.0),
            reasoning=f"Volume: {latest['volume_ratio']:.2f}x average",
            metadata={'volume_ratio': latest['volume_ratio']}
        ))
        
        # 15. OBV Signal
        obv_score = 0.5 if latest['obv'] > latest['obv_sma'] else -0.5
        signals.append(StrategySignal(
            name="OBV",
            signal=obv_score,
            confidence=0.5,
            reasoning=f"OBV: {'Above' if obv_score > 0 else 'Below'} SMA",
            metadata={'obv': latest['obv'], 'obv_sma': latest['obv_sma']}
        ))
        
        # 16. MFI Signal
        mfi_score = self._evaluate_mfi(latest)
        signals.append(StrategySignal(
            name="Money Flow Index",
            signal=mfi_score,
            confidence=0.6,
            reasoning=f"MFI: {latest['mfi']:.1f}",
            metadata={'mfi': latest['mfi']}
        ))
        
        # 17. VWAP Signal
        vwap_score = 0.7 if latest['close'] > latest['vwap'] else -0.7
        signals.append(StrategySignal(
            name="VWAP",
            signal=vwap_score,
            confidence=0.65,
            reasoning=f"Price {((latest['close'] - latest['vwap']) / latest['vwap'] * 100):+.2f}% from VWAP",
            metadata={'vwap': latest['vwap'], 'deviation': latest['vwap_deviation']}
        ))
        
        # === PATTERN RECOGNITION ===
        
        # 18. Support/Resistance
        sr_score = self._evaluate_support_resistance(latest)
        signals.append(StrategySignal(
            name="Support/Resistance",
            signal=sr_score,
            confidence=0.7 if latest['sr_position'] < 0.1 or latest['sr_position'] > 0.9 else 0.4,
            reasoning=f"Price at {latest['sr_position']*100:.1f}% of S/R range",
            metadata={'support': latest['support'], 'resistance': latest['resistance']}
        ))
        
        # 19. Fibonacci Levels
        fib_score = self._evaluate_fibonacci(latest)
        signals.append(StrategySignal(
            name="Fibonacci",
            signal=fib_score,
            confidence=0.55,
            reasoning=self._get_fib_reasoning(latest),
            metadata={}
        ))
        
        # 20. Candlestick Patterns
        candle_score = self._evaluate_candlesticks(latest)
        signals.append(StrategySignal(
            name="Candlestick Patterns",
            signal=candle_score,
            confidence=0.5,
            reasoning=self._get_candle_reasoning(latest),
            metadata={'doji': latest['doji'], 'hammer': latest['hammer'], 'shooting_star': latest['shooting_star']}
        ))
        
        # Calculate composite score with dynamic weighting based on market regime
        composite_score = self._calculate_composite_score(signals, market_regime, latest)
        
        # Determine final recommendation
        recommendation = self._determine_recommendation(composite_score, signals, market_regime)
        
        # Calculate position size suggestion
        position_size = self._calculate_position_size(composite_score, market_regime, latest)
        
        # Calculate stop loss and take profit
        stop_loss = self._calculate_stop_loss(latest, recommendation)
        take_profit = self._calculate_take_profit(latest, recommendation)
        
        # Collect reasoning
        reasoning = [s.reasoning for s in signals if abs(s.signal) > 0.5]
        
        return CombinedAnalysis(
            symbol=symbol,
            timestamp=datetime.now(),
            price=latest['close'],
            market_regime=market_regime,
            signals=signals,
            composite_score=composite_score,
            confidence=abs(composite_score),
            recommendation=recommendation,
            position_size_suggestion=position_size,
            stop_loss=stop_loss,
            take_profit=take_profit,
            reasoning_summary=reasoning
        )
    
    def _detect_market_regime(self, df: pd.DataFrame, latest: pd.Series) -> MarketRegime:
        """Detect current market regime"""
        adx = latest['adx']
        rsi = latest['rsi']
        atr_pct = (latest['atr'] / latest['close']) * 100
        ema_trend = latest['ema_trend']
        
        if atr_pct > Config.VOLATILE_ATR_PERCENT:
            return MarketRegime.HIGH_VOLATILITY
        
        if adx > 30:
            if ema_trend == 1:
                return MarketRegime.STRONG_TREND_UP
            else:
                return MarketRegime.STRONG_TREND_DOWN
        elif adx > 20:
            if ema_trend == 1:
                return MarketRegime.WEAK_TREND_UP
            else:
                return MarketRegime.WEAK_TREND_DOWN
        elif adx < 15:
            return MarketRegime.LOW_VOLATILITY
        else:
            return MarketRegime.RANGING
    
    def _calculate_composite_score(self, signals: List[StrategySignal], regime: MarketRegime, latest: pd.Series) -> float:
        """Calculate weighted composite score based on market regime"""
        
        # Adjust weights based on market regime
        weights_multiplier = {
            'trend_following': 1.0,
            'momentum': 1.0,
            'volatility': 1.0,
            'volume': 1.0,
            'pattern_recognition': 1.0
        }
        
        if regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN]:
            weights_multiplier['trend_following'] = 1.5
            weights_multiplier['momentum'] = 1.2
        elif regime == MarketRegime.HIGH_VOLATILITY:
            weights_multiplier['volatility'] = 1.5
            weights_multiplier['volume'] = 1.3
        elif regime == MarketRegime.RANGING:
            weights_multiplier['momentum'] = 1.3
            weights_multiplier['pattern_recognition'] = 1.4
        
        # Apply volume multiplier to all signals
        volume_multiplier = min(latest['volume_ratio'] / 1.5, 1.5) if latest['volume_ratio'] > 1 else 1.0
        
        total_score = 0
        total_weight = 0
        
        # Define strategy categories
        categories = {
            'trend_following': ['EMA Crossover', 'Ichimoku Cloud', 'ADX Trend', 'SuperTrend', 'Parabolic SAR'],
            'momentum': ['RSI', 'MACD', 'Stochastic RSI', 'Awesome Oscillator', 'CCI'],
            'volatility': ['Bollinger Bands', 'Keltner Channels', 'Donchian Channels'],
            'volume': ['Volume Analysis', 'OBV', 'Money Flow Index', 'VWAP'],
            'pattern_recognition': ['Support/Resistance', 'Fibonacci', 'Candlestick Patterns']
        }
        
        for signal in signals:
            # Find category
            category = None
            for cat, names in categories.items():
                if signal.name in names:
                    category = cat
                    break
            
            weight = 1.0
            if category:
                weight = weights_multiplier[category]
            
            total_score += signal.signal * signal.confidence * weight
            total_weight += weight
        
        composite = total_score / total_weight if total_weight > 0 else 0
        
        # Apply volume confirmation
        composite *= volume_multiplier
        
        return max(min(composite, 1.0), -1.0)
    
    def _determine_recommendation(self, score: float, signals: List[StrategySignal], regime: MarketRegime) -> SignalType:
        """Determine final trading recommendation"""
        if score >= Config.STRONG_BUY_THRESHOLD:
            return SignalType.STRONG_BUY
        elif score >= Config.BUY_THRESHOLD:
            return SignalType.BUY
        elif score >= Config.NEUTRAL_THRESHOLD:
            return SignalType.WEAK_BUY
        elif score >= -Config.NEUTRAL_THRESHOLD:
            return SignalType.NEUTRAL
        elif score >= Config.SELL_THRESHOLD:
            return SignalType.WEAK_SELL
        elif score >= Config.STRONG_SELL_THRESHOLD:
            return SignalType.SELL
        else:
            return SignalType.STRONG_SELL
    
    def _calculate_position_size(self, score: float, regime: MarketRegime, latest: pd.Series) -> float:
        """Calculate suggested position size based on confidence and volatility"""
        base_size = Config.BASE_TRADE_AMOUNT_USDT
        
        # Adjust for confidence
        confidence_multiplier = abs(score)
        
        # Adjust for volatility (smaller positions in high volatility)
        atr_pct = (latest['atr'] / latest['close']) * 100
        volatility_multiplier = 1.0
        if atr_pct > 5:
            volatility_multiplier = 0.5
        elif atr_pct > 3:
            volatility_multiplier = 0.7
        
        # Adjust for trend strength
        trend_multiplier = 1.0
        if regime in [MarketRegime.STRONG_TREND_UP, MarketRegime.STRONG_TREND_DOWN]:
            trend_multiplier = 1.2
        
        position_size = base_size * confidence_multiplier * volatility_multiplier * trend_multiplier
        
        return min(max(position_size, Config.MIN_POSITION_SIZE_USDT), Config.MAX_POSITION_SIZE_USDT)
    
    def _calculate_stop_loss(self, latest: pd.Series, recommendation: SignalType) -> float:
        """Calculate stop loss based on ATR"""
        atr_stop = Config.STOP_LOSS_ATR_MULTIPLIER * latest['atr']
        
        if recommendation in [SignalType.BUY, SignalType.STRONG_BUY]:
            return latest['close'] - atr_stop
        else:
            return latest['close'] + atr_stop
    
    def _calculate_take_profit(self, latest: pd.Series, recommendation: SignalType) -> float:
        """Calculate take profit based on ATR"""
        atr_tp = Config.TAKE_PROFIT_ATR_MULTIPLIER * latest['atr']
        
        if recommendation in [SignalType.BUY, SignalType.STRONG_BUY]:
            return latest['close'] + atr_tp
        else:
            return latest['close'] - atr_tp
    
    # Individual strategy evaluation methods
    def _evaluate_ema_crossover(self, latest: pd.Series, prev: pd.Series) -> float:
        if latest['ema_trend'] == 1:
            # Check if it's a fresh crossover
            if prev['ema_9'] <= prev['ema_21'] and latest['ema_9'] > latest['ema_21']:
                return 1.0
            return 0.7
        elif latest['ema_9'] < latest['ema_21'] and latest['ema_21'] < latest['ema_50']:
            return -0.7
        return 0
    
    def _evaluate_ichimoku(self, latest: pd.Series, prev: pd.Series) -> float:
        score = 0
        if latest['price_above_cloud']:
            score += 0.5
        if latest['cloud_bullish']:
            score += 0.3
        if latest['tk_cross_bullish']:
            score += 0.2
        return min(score, 1.0)
    
    def _evaluate_adx(self, latest: pd.Series, prev: pd.Series) -> float:
        if latest['adx'] > 25:
            if latest['plus_di'] > latest['minus_di']:
                return 0.8
            else:
                return -0.8
        return 0
    
    def _evaluate_rsi(self, latest: pd.Series) -> float:
        rsi = latest['rsi']
        if rsi < 30:
            return 0.8
        elif rsi > 70:
            return -0.8
        elif rsi < 40:
            return 0.3
        elif rsi > 60:
            return -0.3
        return 0
    
    def _evaluate_macd(self, latest: pd.Series, prev: pd.Series) -> float:
        if latest['macd_hist_12_26'] > 0 and prev['macd_hist_12_26'] <= 0:
            return 1.0
        elif latest['macd_hist_12_26'] < 0 and prev['macd_hist_12_26'] >= 0:
            return -1.0
        elif latest['macd_hist_12_26'] > 0:
            return 0.5
        else:
            return -0.5
    
    def _evaluate_stochastic_rsi(self, latest: pd.Series) -> float:
        k = latest['stoch_rsi_k']
        if k < 20:
            return 0.8
        elif k > 80:
            return -0.8
        return 0
    
    def _evaluate_cci(self, latest: pd.Series) -> float:
        cci = latest['cci']
        if cci < -100:
            return 0.7
        elif cci > 100:
            return -0.7
        return 0
    
    def _evaluate_bollinger_bands(self, latest: pd.Series, prev: pd.Series) -> float:
        pos = latest['bb_position']
        if pos < 0.1:
            return 0.8
        elif pos > 0.9:
            return -0.8
        elif latest['bb_squeeze']:
            return 0.4 if latest['close'] > latest['bb_mid_20'] else -0.4
        return 0
    
    def _evaluate_keltner(self, latest: pd.Series) -> float:
        if latest['close'] > latest['kc_upper']:
            return 0.7
        elif latest['close'] < latest['kc_lower']:
            return -0.7
        return 0
    
    def _evaluate_donchian(self, latest: pd.Series) -> float:
        if latest['close'] > latest['dc_upper']:
            return 0.8
        elif latest['close'] < latest['dc_lower']:
            return -0.8
        return 0
    
    def _evaluate_volume(self, latest: pd.Series) -> float:
        if latest['volume_ratio'] > 2:
            return 1.0
        elif latest['volume_ratio'] > 1.5:
            return 0.5
        return 0
    
    def _evaluate_mfi(self, latest: pd.Series) -> float:
        mfi = latest['mfi']
        if mfi < 20:
            return 0.7
        elif mfi > 80:
            return -0.7
        return 0
    
    def _evaluate_support_resistance(self, latest: pd.Series) -> float:
        pos = latest['sr_position']
        if pos < 0.1:
            return 0.6
        elif pos > 0.9:
            return -0.6
        return 0
    
    def _evaluate_fibonacci(self, latest: pd.Series) -> float:
        price = latest['close']
        for level in [0.382, 0.5, 0.618]:
            if abs(price - latest[f'fib_{level}']) / price < 0.01:
                return 0.5 if price > latest[f'fib_{level}'] else -0.5
        return 0
    
    def _evaluate_candlesticks(self, latest: pd.Series) -> float:
        score = 0
        if latest['hammer']:
            score += 0.5
        if latest['shooting_star']:
            score -= 0.5
        if latest['bullish_engulfing']:
            score += 0.7
        if latest['bearish_engulfing']:
            score -= 0.7
        return max(min(score, 1.0), -1.0)
    
    # Reasoning methods
    def _get_ema_reasoning(self, latest: pd.Series) -> str:
        if latest['ema_trend'] == 1:
            return f"Bullish EMA alignment: 9({latest['ema_9']:.2f}) > 21({latest['ema_21']:.2f}) > 50({latest['ema_50']:.2f})"
        return "No clear EMA trend"
    
    def _get_ichimoku_reasoning(self, latest: pd.Series) -> str:
        reasons = []
        if latest['price_above_cloud']:
            reasons.append("Price above cloud")
        if latest['cloud_bullish']:
            reasons.append("Bullish cloud")
        return ", ".join(reasons) if reasons else "Neutral Ichimoku"
    
    def _get_rsi_reasoning(self, latest: pd.Series) -> str:
        rsi = latest['rsi']
        if rsi < 30:
            return f"Oversold (RSI: {rsi:.1f})"
        elif rsi > 70:
            return f"Overbought (RSI: {rsi:.1f})"
        return f"Neutral RSI: {rsi:.1f}"
    
    def _get_macd_reasoning(self, latest: pd.Series) -> str:
        if latest['macd_hist_12_26'] > 0:
            return f"Bullish MACD (Histogram: {latest['macd_hist_12_26']:.2f})"
        return f"Bearish MACD (Histogram: {latest['macd_hist_12_26']:.2f})"
    
    def _get_bb_reasoning(self, latest: pd.Series) -> str:
        pos = latest['bb_position']
        if pos < 0.1:
            return "Price at lower band - potential bounce"
        elif pos > 0.9:
            return "Price at upper band - potential pullback"
        elif latest['bb_squeeze']:
            return "Bollinger squeeze - breakout imminent"
        return "Price within bands"
    
    def _get_fib_reasoning(self, latest: pd.Series) -> str:
        price = latest['close']
        for level in [0.382, 0.5, 0.618]:
            if abs(price - latest[f'fib_{level}']) / price < 0.01:
                return f"Price at Fibonacci {level}"
        return "No significant Fibonacci level"
    
    def _get_candle_reasoning(self, latest: pd.Series) -> str:
        if latest['hammer']:
            return "Bullish hammer pattern"
        elif latest['shooting_star']:
            return "Bearish shooting star"
        elif latest['bullish_engulfing']:
            return "Bullish engulfing pattern"
        elif latest['bearish_engulfing']:
            return "Bearish engulfing pattern"
        return "No significant pattern"

# ============================================================================
# MAIN TRADING BOT
# ============================================================================

class IntelligentTradingBot:
    """Main bot that scans all pairs and makes intelligent decisions"""
    
    def __init__(self):
        self.client = BinanceClient()
        self.signal_generator = SignalGenerator()
        self.firebase_ref = None
        
        if init_firebase():
            self.firebase_ref = db.reference('/')
    
    def scan_all_pairs(self) -> List[CombinedAnalysis]:
        """Scan all USDT pairs and return ranked opportunities"""
        logger.info("🔍 Scanning all USDT pairs...")
        
        pairs = self.client.get_all_usdt_pairs()
        analyses = []
        
        for i, symbol in enumerate(pairs[:30]):  # Scan top 30 by volume
            try:
                logger.info(f"Analyzing {symbol} ({i+1}/{min(30, len(pairs))})")
                
                # Get multi-timeframe data
                df_15m = self.client.get_klines(symbol, '15m', limit=200)
                if df_15m.empty:
                    continue
                
                # Analyze with all strategies
                analysis = self.signal_generator.analyze_all_strategies(df_15m, symbol)
                analyses.append(analysis)
                
                # Save to Firebase
                self._save_analysis_to_firebase(analysis)
                
                # Rate limiting
                time.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Error analyzing {symbol}: {e}")
                continue
        
        # Sort by absolute score (best opportunities first)
        analyses.sort(key=lambda x: abs(x.composite_score), reverse=True)
        
        return analyses
    
    def _save_analysis_to_firebase(self, analysis: CombinedAnalysis):
        """Save analysis results to Firebase"""
        if not self.firebase_ref:
            return
        
        try:
            data = {
                'symbol': analysis.symbol,
                'timestamp': analysis.timestamp.isoformat(),
                'price': analysis.price,
                'market_regime': analysis.market_regime.value,
                'composite_score': analysis.composite_score,
                'confidence': analysis.confidence,
                'recommendation': analysis.recommendation.value,
                'position_size': analysis.position_size_suggestion,
                'stop_loss': analysis.stop_loss,
                'take_profit': analysis.take_profit,
                'signals': [
                    {
                        'name': s.name,
                        'signal': s.signal,
                        'confidence': s.confidence,
                        'reasoning': s.reasoning
                    }
                    for s in analysis.signals
                ]
            }
            
            self.firebase_ref.child('analyses').child(analysis.symbol).set(data)
            
        except Exception as e:
            logger.error(f"Firebase save error: {e}")
    
    def select_best_trades(self, analyses: List[CombinedAnalysis]) -> List[CombinedAnalysis]:
        """Intelligently select the best trades from all analyses"""
        
        # Filter by minimum score
        candidates = [a for a in analyses if abs(a.composite_score) >= Config.BUY_THRESHOLD]
        
        # Check current positions
        current_positions = self._get_current_positions()
        
        # Don't exceed max positions
        available_slots = Config.MAX_POSITIONS - len(current_positions)
        
        if available_slots <= 0:
            logger.info(f"Max positions ({Config.MAX_POSITIONS}) reached")
            return []
        
        # Select best trades
        selected = []
        
        for analysis in candidates[:available_slots]:
            # Check if already in position
            if analysis.symbol in current_positions:
                continue
            
            # Check if recommendation is actionable
            if analysis.recommendation in [SignalType.BUY, SignalType.STRONG_BUY]:
                selected.append(analysis)
        
        return selected
    
    def _get_current_positions(self) -> Dict:
        """Get current open positions from Binance"""
        try:
            account = self.client.get_account_info()
            positions = {}
            
            for balance in account.get('balances', []):
                asset = balance['asset']
                free = float(balance['free'])
                
                if free > 0 and asset != 'USDT':
                    # This is simplified - in production you'd track entry prices
                    positions[f"{asset}USDT"] = {
                        'quantity': free,
                        'asset': asset
                    }
            
            return positions
            
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return {}
    
    def execute_trades(self, selected_trades: List[CombinedAnalysis]):
        """Execute the selected trades"""
        for analysis in selected_trades:
            try:
                # Calculate quantity
                price = analysis.price
                quantity = analysis.position_size_suggestion / price
                
                logger.info(f"🚀 Executing {analysis.recommendation.value} for {analysis.symbol}")
                logger.info(f"   Price: ${price:.2f}")
                logger.info(f"   Size: ${analysis.position_size_suggestion:.2f} ({quantity:.6f})")
                logger.info(f"   Confidence: {analysis.confidence:.2%}")
                logger.info(f"   Stop Loss: ${analysis.stop_loss:.2f}")
                logger.info(f"   Take Profit: ${analysis.take_profit:.2f}")
                
                # Place order
                order = self.client.create_order(
                    symbol=analysis.symbol,
                    side='BUY',
                    quantity=quantity
                )
                
                if order and 'orderId' in order:
                    logger.info(f"✅ Order placed: {order['orderId']}")
                    
                    # Save trade to Firebase
                    if self.firebase_ref:
                        trade_data = {
                            'symbol': analysis.symbol,
                            'timestamp': datetime.now().isoformat(),
                            'side': 'BUY',
                            'price': price,
                            'quantity': quantity,
                            'value_usdt': analysis.position_size_suggestion,
                            'order_id': order['orderId'],
                            'analysis': {
                                'score': analysis.composite_score,
                                'confidence': analysis.confidence,
                                'regime': analysis.market_regime.value,
                                'stop_loss': analysis.stop_loss,
                                'take_profit': analysis.take_profit
                            }
                        }
                        self.firebase_ref.child('trades').push(trade_data)
                
                time.sleep(1)  # Rate limiting
                
            except Exception as e:
                logger.error(f"❌ Error executing trade for {analysis.symbol}: {e}")
    
    def run(self):
        """Main execution loop"""
        logger.info("=" * 80)
        logger.info("🤖 INTELLIGENT MULTI-STRATEGY BOT STARTING")
        logger.info(f"   Time: {datetime.now().isoformat()}")
        logger.info(f"   Strategies: 20+ indicators across 4 categories")
        logger.info(f"   Scanning: Top {Config.TOP_PAIRS_TO_ANALYZE} pairs by volume")
        logger.info("=" * 80)
        
        try:
            # Step 1: Scan all pairs
            analyses = self.scan_all_pairs()
            
            # Step 2: Log top opportunities
            logger.info("\n📊 TOP OPPORTUNITIES:")
            for i, analysis in enumerate(analyses[:10]):
                logger.info(f"   {i+1}. {analysis.symbol}: Score={analysis.composite_score:.3f} "
                          f"({analysis.recommendation.value}) - {analysis.market_regime.value}")
            
            # Step 3: Select best trades
            selected = self.select_best_trades(analyses)
            
            if selected:
                logger.info(f"\n🎯 SELECTED {len(selected)} TRADES:")
                for trade in selected:
                    logger.info(f"   • {trade.symbol}: {trade.recommendation.value} "
                              f"(Score: {trade.composite_score:.3f}, Size: ${trade.position_size_suggestion:.0f})")
                
                # Step 4: Execute trades
                self.execute_trades(selected)
            else:
                logger.info("\n⏸️ No trades meet criteria at this time")
            
            # Step 5: Update bot state in Firebase
            if self.firebase_ref:
                self.firebase_ref.child('bot_state').set({
                    'last_run': datetime.now().isoformat(),
                    'pairs_scanned': len(analyses),
                    'opportunities_found': len([a for a in analyses if abs(a.composite_score) > 0.5]),
                    'trades_executed': len(selected)
                })
            
        except Exception as e:
            logger.error(f"❌ Bot execution error: {e}", exc_info=True)
        
        logger.info("=" * 80)
        logger.info("✅ BOT RUN COMPLETE")
        logger.info("=" * 80)

# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    bot = IntelligentTradingBot()
    bot.run()
