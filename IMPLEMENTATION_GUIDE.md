# Implementation Guide - Phase 2 & Beyond

## Current Completion Status (11 Commits)

### ✅ Completed
- data/data_loader.py - OHLCV data fetching with caching
- data/news_fetcher.py - News fetching with rule-based sentiment
- data/__init__.py - Module initialization
- main.py - CLI with full command support
- config.yaml - Comprehensive configuration
- requirements.txt - All dependencies
- Documentation (README, CONTRIBUTING, QUICKSTART, IMPLEMENTATION_ROADMAP)

### 🚀 Next Priority Implementations

## Phase 1B: Models Module (Priority: CRITICAL)

Create the following files in order:

### 1. `models/__init__.py`
```python
"""
ML Models module for AI-Investment-Analysis
"""
from .sentiment_analyzer import SentimentAnalyzer
from .price_forecasting import LSTMPredictor
from .pattern_detector import AnomalyDetector
from .ensemble_ranker import EnsembleRanker

__all__ = ['SentimentAnalyzer', 'LSTMPredictor', 'AnomalyDetector', 'EnsembleRanker']
```

### 2. `models/sentiment_analyzer.py`
**Key Features:**
- Load FinBERT model: `ProsusAI/finbert-base`
- Batch sentence processing
- Returns: `{'positive': float, 'negative': float, 'neutral': float}`
- Methods:
  - `analyze(text: str) -> Dict[str, float]`
  - `batch_analyze(texts: List[str]) -> List[Dict]`
  - `aggregate(scores: List[Dict], method='mean') -> Dict`

**Implementation Tips:**
```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert-base')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert-base')
```

### 3. `models/price_forecasting.py`
**Key Features:**
- LSTM architecture: 2 layers, 128 hidden units
- Input: 60 days of OHLCV data
- Output: 5-day ahead price predictions
- Validation: train/test split with 20% test
- Methods:
  - `train(data: pd.DataFrame, epochs: int, batch_size: int) -> Dict[str, float]`
  - `predict(data: pd.DataFrame, horizon: int) -> np.ndarray`
  - `evaluate(test_data: pd.DataFrame) -> Dict[str, float]` (MAE, RMSE, etc.)

**Note:** Use TensorFlow/Keras for simplicity or PyTorch for flexibility

### 4. `models/pattern_detector.py`
**Key Features:**
- Detect chart patterns (head & shoulders, triangles)
- Identify support/resistance levels
- Find breakout points
- Uses: Isolation Forest for anomalies
- Methods:
  - `detect_reversals(data: pd.DataFrame) -> List[Dict]`
  - `detect_breakouts(data: pd.DataFrame) -> List[Dict]`
  - `find_support_resistance(data: pd.DataFrame) -> Tuple[List, List]`

### 5. `models/ensemble_ranker.py`
**Key Features:**
- Combine: price_signal (0.4) + sentiment_signal (0.3) + technical_signal (0.3)
- Input: Multiple symbols with their technical & sentiment data
- Output: Ranked DataFrame sorted by composite score
- Methods:
  - `rank_stocks(symbols: List[str], data_dict: Dict) -> pd.DataFrame`
  - `get_confidence(symbol: str) -> float`
  - `rescore(weights: Dict) -> None` (dynamic weight adjustment)

## Phase 2: Indicators Module (Priority: HIGH)

Create `indicators/` folder with:

### 1. `indicators/__init__.py`
```python
from .technical import (sma, ema, rsi, macd, bollinger_bands, atr, obv)

__all__ = ['sma', 'ema', 'rsi', 'macd', 'bollinger_bands', 'atr', 'obv']
```

### 2. `indicators/technical.py`
**Functions needed:**
- `sma(data, period)` - Simple Moving Average
- `ema(data, period)` - Exponential Moving Average  
- `rsi(data, period)` - Relative Strength Index
- `macd(data)` - MACD (12, 26, 9)
- `bollinger_bands(data, period, std)` - Bollinger Bands
- `atr(data, period)` - Average True Range
- `obv(data)` - On-Balance Volume

**Use TA-Lib or pandas_ta for implementation**

## Phase 3: Strategies Module (Priority: HIGH)

Create `strategies/` folder with three key files:

### 1. `strategies/long_term_strategy.py`
- Multi-factor ranking system
- Factors: Quality (0.3), Value (0.3), Momentum (0.2), Sentiment (0.2)
- Outputs: Stock rankings with allocation weights
- Risk levels: conservative, moderate, aggressive

### 2. `strategies/swing_strategy.py`
- Technical + Sentiment fusion
- Entry signals: MACD bullish + RSI < 70
- Exit signals: Take-profit @ 2% or Stop-loss @ 1.5%
- Holding period: 5-15 days

### 3. `strategies/intraday_strategy.py`
- Paper trading only mode
- 5-minute candles
- Support/Resistance breakouts
- High-frequency signals (for research)

## Phase 4: Backtesting Engine (Priority: HIGH)

Create `backtester/` folder:

### 1. `backtester/backtest_engine.py`
- Walk-forward validation
- Slippage modeling (5 bps default)
- Transaction costs
- Portfolio rebalancing logic

### 2. `backtester/performance_metrics.py`
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio
- Max Drawdown
- Win Rate
- Profit Factor

## Implementation Checklist

For each new module, ensure:

- [ ] Full type hints on all functions/methods
- [ ] Comprehensive docstrings (Google format)
- [ ] Unit tests in `tests/` folder
- [ ] Error handling with try-except
- [ ] Logging with loguru
- [ ] Configuration integration via config.yaml
- [ ] README examples for usage

## Testing Each Module

```bash
# After implementing sentiment_analyzer
python -c "
from models import SentimentAnalyzer
sa = SentimentAnalyzer()
result = sa.analyze('Strong earnings beat expectations!')
print(result)
"

# After implementing price_forecasting
python -c "
from models import LSTMPredictor
from data import DataLoader
loader = DataLoader()
data = loader.fetch_ohlcv('INFY.NS', period='2y')
predictor = LSTMPredictor()
pred = predictor.predict(data, horizon=5)
print(pred)
"
```

## Git Workflow for Contributions

```bash
# Create feature branch
git checkout -b feature/implement-models

# Implement with tests
git add models/
git commit -m "feat: Implement sentiment_analyzer and price_forecasting"

# Push and create PR
git push origin feature/implement-models
```

## Performance Benchmarks to Target

Based on backtest samples:
- Long-term: 12-15% annual return
- Swing: 8-10% annual return with 50%+ win rate
- Sharpe ratio: > 1.0 preferred
- Max drawdown: < 25%

## Resources for Implementation

- **FinBERT**: https://huggingface.co/ProsusAI/finbert-base
- **TA-Lib**: https://github.com/mrjbq7/ta-lib
- **TensorFlow LSTM**: https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM
- **Backtrader**: https://www.backtrader.com/ (reference for inspiration)
- **Sample Notebooks**: Create in `examples/` folder

## Next Session Priority

1. Create `models/__init__.py` and `models/sentiment_analyzer.py`
2. Test sentiment analysis with news headlines
3. Implement `models/price_forecasting.py` with LSTM
4. Create indicator functions in `indicators/technical.py`
5. Build first complete strategy in `strategies/long_term_strategy.py`

## Success Metrics

By end of implementation:
- [ ] 50+ total commits
- [ ] 6+ core modules with full test coverage
- [ ] Backtesting engine with sample results
- [ ] Working CLI with all commands
- [ ] > 500 lines of core logic
- [ ] Public GitHub repo with documentation

---

**Last Updated**: January 2026
**Status**: Phase 2 In Progress
**Contributors**: shivraj1182
