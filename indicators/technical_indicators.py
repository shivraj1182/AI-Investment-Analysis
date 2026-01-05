import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import pandas as pd

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Technical indicators for price analysis.
    
    Implements common technical analysis indicators:
    - Simple Moving Average (SMA)
    - Relative Strength Index (RSI)
    - MACD (Moving Average Convergence Divergence)
    - Bollinger Bands
    """
    
    @staticmethod
    def simple_moving_average(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Calculate Simple Moving Average.
        
        Args:
            prices: Array of prices
            period: Period for moving average
            
        Returns:
            Array of SMA values
        """
        if len(prices) < period:
            logger.warning(f"Data length {len(prices)} < period {period}")
            return np.array([])
        
        sma = pd.Series(prices).rolling(window=period).mean().values
        return sma
    
    @staticmethod
    def exponential_moving_average(prices: np.ndarray, period: int = 20) -> np.ndarray:
        """
        Calculate Exponential Moving Average.
        
        Args:
            prices: Array of prices
            period: Period for EMA
            
        Returns:
            Array of EMA values
        """
        if len(prices) < period:
            return np.array([])
        
        ema = pd.Series(prices).ewm(span=period, adjust=False).mean().values
        return ema
    
    @staticmethod
    def relative_strength_index(prices: np.ndarray, period: int = 14) -> np.ndarray:
        """
        Calculate Relative Strength Index (RSI).
        
        Args:
            prices: Array of prices
            period: Period for RSI calculation
            
        Returns:
            Array of RSI values (0-100)
        """
        if len(prices) < period + 1:
            return np.array([])
        
        delta = np.diff(prices)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        avg_gain = pd.Series(gain).rolling(window=period).mean().values
        avg_loss = pd.Series(loss).rolling(window=period).mean().values
        
        rs = np.divide(avg_gain, avg_loss, where=avg_loss != 0, out=np.zeros_like(avg_loss))
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    @staticmethod
    def macd(prices: np.ndarray, fast: int = 12, slow: int = 26,
             signal: int = 9) -> Dict[str, np.ndarray]:
        """
        Calculate MACD (Moving Average Convergence Divergence).
        
        Args:
            prices: Array of prices
            fast: Fast EMA period
            slow: Slow EMA period
            signal: Signal line period
            
        Returns:
            Dictionary with MACD, signal, and histogram
        """
        if len(prices) < slow:
            return {'macd': np.array([]), 'signal': np.array([]), 'histogram': np.array([])}
        
        ema_fast = pd.Series(prices).ewm(span=fast, adjust=False).mean().values
        ema_slow = pd.Series(prices).ewm(span=slow, adjust=False).mean().values
        
        macd_line = ema_fast - ema_slow
        signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().values
        histogram = macd_line - signal_line
        
        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }
    
    @staticmethod
    def bollinger_bands(prices: np.ndarray, period: int = 20,
                       std_dev: float = 2.0) -> Dict[str, np.ndarray]:
        """
        Calculate Bollinger Bands.
        
        Args:
            prices: Array of prices
            period: Period for moving average
            std_dev: Number of standard deviations
            
        Returns:
            Dictionary with upper, middle, and lower bands
        """
        if len(prices) < period:
            return {'upper': np.array([]), 'middle': np.array([]), 'lower': np.array([])}
        
        sma = TechnicalIndicators.simple_moving_average(prices, period)
        std = pd.Series(prices).rolling(window=period).std().values
        
        upper_band = sma + (std_dev * std)
        lower_band = sma - (std_dev * std)
        
        return {
            'upper': upper_band,
            'middle': sma,
            'lower': lower_band
        }
    
    @staticmethod
    def calculate_all_indicators(prices: np.ndarray) -> Dict:
        """
        Calculate all technical indicators at once.
        
        Args:
            prices: Array of prices
            
        Returns:
            Dictionary containing all indicators
        """
        try:
            indicators = {
                'sma_20': TechnicalIndicators.simple_moving_average(prices, 20),
                'sma_50': TechnicalIndicators.simple_moving_average(prices, 50),
                'ema_12': TechnicalIndicators.exponential_moving_average(prices, 12),
                'ema_26': TechnicalIndicators.exponential_moving_average(prices, 26),
                'rsi_14': TechnicalIndicators.relative_strength_index(prices, 14),
                'macd': TechnicalIndicators.macd(prices),
                'bollinger_bands': TechnicalIndicators.bollinger_bands(prices)
            }
            return indicators
        except Exception as e:
            logger.error(f"Failed to calculate indicators: {str(e)}")
            return {}
