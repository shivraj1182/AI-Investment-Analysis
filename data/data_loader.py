"""
DataLoader module for fetching and caching market data.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, List, Dict
from loguru import logger
import pickle
import hashlib


class DataLoader:
    """Handles OHLCV data fetching, caching, and preprocessing."""
    
    def __init__(self, cache_dir: str = './data/cache', cache_hours: int = 24):
        """
        Initialize DataLoader with cache configuration.
        
        Args:
            cache_dir: Directory for caching data
            cache_hours: Cache validity in hours
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_hours = cache_hours
        logger.info(f'DataLoader initialized with cache_dir={cache_dir}')
    
    def _get_cache_key(self, symbol: str, period: str, interval: str) -> str:
        """Generate cache key for data."""
        key = f'{symbol}_{period}_{interval}'
        return hashlib.md5(key.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_file: Path) -> bool:
        """Check if cached data is still valid."""
        if not cache_file.exists():
            return False
        
        file_age = datetime.now() - datetime.fromtimestamp(cache_file.stat().st_mtime)
        return file_age < timedelta(hours=self.cache_hours)
    
    def fetch_ohlcv(self, symbol: str, period: str = '5y', interval: str = '1d',
                    use_cache: bool = True) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'INFY.NS')
            period: Data period ('5y', '1y', '3mo', etc.)
            interval: Candle interval ('1d', '1h', '5m', etc.)
            use_cache: Use cached data if available
            
        Returns:
            DataFrame with OHLCV data indexed by datetime
        """
        cache_key = self._get_cache_key(symbol, period, interval)
        cache_file = self.cache_dir / f'{cache_key}.pkl'
        
        # Try to load from cache
        if use_cache and self._is_cache_valid(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                logger.info(f'Loaded {symbol} from cache')
                return data
            except Exception as e:
                logger.warning(f'Cache load failed: {e}')
        
        # Fetch from yfinance
        try:
            logger.info(f'Fetching {symbol} (period={period}, interval={interval})')
            data = yf.download(symbol, period=period, interval=interval, progress=False)
            
            # Add technical columns
            data['Symbol'] = symbol
            data['Return'] = data['Close'].pct_change()
            data['LogReturn'] = __import__('numpy').log(data['Close'] / data['Close'].shift(1))
            
            # Cache the data
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f'Cached {symbol} with {len(data)} candles')
            return data
        
        except Exception as e:
            logger.error(f'Failed to fetch {symbol}: {e}')
            raise
    
    def fetch_multiple(self, symbols: List[str], **kwargs) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            **kwargs: Arguments to pass to fetch_ohlcv
            
        Returns:
            Dictionary mapping symbols to DataFrames
        """
        data = {}
        for symbol in symbols:
            try:
                data[symbol] = self.fetch_ohlcv(symbol, **kwargs)
            except Exception as e:
                logger.error(f'Failed to fetch {symbol}: {e}')
        
        return data
    
    def clear_cache(self):
        """Clear all cached data."""
        import shutil
        shutil.rmtree(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info('Cache cleared')
