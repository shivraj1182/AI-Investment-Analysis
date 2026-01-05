"""
News Fetcher module for fetching financial news and sentiment analysis.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from loguru import logger
import json


class NewsFetcher:
    """Fetches financial news and performs sentiment analysis."""
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize NewsFetcher.
        
        Args:
            api_key: API key for news service (NewsAPI, Finnhub, etc.)
        """
        self.api_key = api_key
        self.news_sources = ['bloomberg', 'reuters', 'cnbc', 'financial-times']
        logger.info('NewsFetcher initialized')
    
    def fetch_headlines(self, symbol: str, days: int = 7) -> List[Dict]:
        """
        Fetch news headlines for a symbol.
        
        Args:
            symbol: Stock symbol (e.g., 'INFY', 'TCS')
            days: Number of days to look back
            
        Returns:
            List of article dictionaries with title, source, date, url
        """
        try:
            # This is a placeholder for actual API integration
            # In production, use NewsAPI or Finnhub
            logger.info(f'Fetching headlines for {symbol} (last {days} days)')
            
            # Sample headlines structure (replace with actual API call)
            headlines = [
                {
                    'symbol': symbol,
                    'title': f'{symbol} stock surges on strong earnings',
                    'source': 'bloomberg',
                    'date': datetime.now() - timedelta(hours=2),
                    'url': f'https://news.example.com/{symbol}',
                    'text': 'Company reports strong Q4 results...'
                },
                {
                    'symbol': symbol,
                    'title': f'{symbol} launches new product line',
                    'source': 'reuters',
                    'date': datetime.now() - timedelta(hours=6),
                    'url': f'https://news.example.com/{symbol}2',
                    'text': 'In a major strategic move, the company...'
                }
            ]
            
            logger.info(f'Fetched {len(headlines)} headlines for {symbol}')
            return headlines
            
        except Exception as e:
            logger.error(f'Failed to fetch headlines for {symbol}: {e}')
            return []
    
    def get_sentiment(self, symbol: str, lookback_days: int = 7) -> Dict[str, float]:
        """
        Get aggregated sentiment for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_days: Days to aggregate sentiment
            
        Returns:
            Dictionary with sentiment scores {'positive': 0.7, 'negative': 0.2, 'neutral': 0.1}
        """
        try:
            headlines = self.fetch_headlines(symbol, days=lookback_days)
            
            if not headlines:
                return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
            
            # Simple rule-based sentiment (replace with FinBERT in production)
            positive_count = 0
            negative_count = 0
            neutral_count = 0
            
            positive_words = ['surge', 'jump', 'growth', 'strong', 'beat', 'profit', 'wins']
            negative_words = ['drop', 'fall', 'decline', 'weak', 'miss', 'loss', 'fails']
            
            for article in headlines:
                title_lower = article['title'].lower()
                
                if any(word in title_lower for word in positive_words):
                    positive_count += 1
                elif any(word in title_lower for word in negative_words):
                    negative_count += 1
                else:
                    neutral_count += 1
            
            total = len(headlines)
            sentiment = {
                'positive': positive_count / total,
                'negative': negative_count / total,
                'neutral': neutral_count / total,
                'score': (positive_count - negative_count) / total if total > 0 else 0
            }
            
            logger.info(f'{symbol} sentiment: {sentiment}')
            return sentiment
            
        except Exception as e:
            logger.error(f'Failed to get sentiment for {symbol}: {e}')
            return {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0, 'score': 0.0}
    
    def aggregate_sentiment(self, symbols: List[str], lookback_days: int = 7) -> Dict[str, Dict]:
        """
        Get sentiment for multiple symbols.
        
        Args:
            symbols: List of stock symbols
            lookback_days: Days to aggregate sentiment
            
        Returns:
            Dictionary mapping symbols to sentiment scores
        """
        sentiments = {}
        
        for symbol in symbols:
            sentiments[symbol] = self.get_sentiment(symbol, lookback_days)
        
        logger.info(f'Aggregated sentiment for {len(symbols)} symbols')
        return sentiments
    
    def get_sentiment_score(self, symbol: str, lookback_days: int = 7) -> float:
        """
        Get single sentiment score (-1 to 1) for a symbol.
        
        Args:
            symbol: Stock symbol
            lookback_days: Days to aggregate
            
        Returns:
            Float score from -1 (very negative) to 1 (very positive)
        """
        sentiment = self.get_sentiment(symbol, lookback_days)
        return sentiment.get('score', 0.0)
    
    def get_sentiment_distribution(self, symbol: str, lookback_days: int = 7) -> Dict[str, float]:
        """
        Get sentiment distribution as percentages.
        
        Args:
            symbol: Stock symbol
            lookback_days: Days to aggregate
            
        Returns:
            Dict with positive, negative, neutral percentages
        """
        sentiment = self.get_sentiment(symbol, lookback_days)
        return {
            'positive_pct': sentiment['positive'] * 100,
            'negative_pct': sentiment['negative'] * 100,
            'neutral_pct': sentiment['neutral'] * 100
        }
