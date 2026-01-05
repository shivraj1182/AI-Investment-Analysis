"""
Data module for AI-Investment-Analysis

Handles data loading, caching, and news fetching for investment analysis.
"""

from .data_loader import DataLoader
from .news_fetcher import NewsFetcher

__all__ = ['DataLoader', 'NewsFetcher']
