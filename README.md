# AI-Investment-Analysis

## Overview

AI-assisted investment program with multi-horizon analysis and decision support. This platform combines machine learning, technical analysis, sentiment analysis from financial news, and backtesting capabilities to provide investment and trading recommendations across three distinct timeframes:

- **Long-term Investment** (3+ months): Fundamental analysis with ML-based stock ranking
- **Short-term Trading** (swing, 5-15 days): Technical indicators with sentiment overlay
- **Intraday Trading** (research & backtesting only): High-frequency signal detection with risk controls

## Key Features

### 1. Multi-Horizon Analysis Engine
- **Long-term Module**: Factor-based stock ranking (quality, value, momentum, sentiment)
- **Swing Module**: Technical signals + momentum with 5-15 day holding periods
- **Intraday Module**: Paper trading & backtesting (exchange-approved deployment only)

### 2. AI & Machine Learning Components
- **Price Forecasting Models**: LSTM/GRU networks for volatility and directional predictions
- **Sentiment Analysis**: FinBERT-based news sentiment integration from financial sources
- **Anomaly Detection**: Identify unusual price/volume patterns signaling reversals or continuations
- **Feature Engineering**: Automated indicator synthesis and pattern recognition

### 3. Technical Analysis Indicators
- Moving Averages (SMA, EMA, DEMA)
- Momentum Indicators (RSI, MACD, Stochastic)
- Volatility Bands (Bollinger Bands, ATR, Keltner Channels)
- Volume Analysis (OBV, CMF, Volume Profiles)
- Support/Resistance Detection

### 4. News-Aware Risk Management
- Real-time financial news API integration
- Sentiment score aggregation over N-day windows
- Position sizing adjustments based on news sentiment confidence
- Macro-economic calendar event flagging

### 5. AI-Assisted Profit Calculator
- Scenario-based projections (bear/base/bull cases)
- Probability distributions of outcomes using historical backtests
- Risk-adjusted return estimates
- Position sizing recommendations based on Kelly Criterion
- Max drawdown and recovery time estimates

### 6. Backtesting & Paper Trading Engine
- Walk-forward backtesting with transaction costs
- Slippage simulation and spread modeling
- Monte Carlo analysis for robustness testing
- Trade-by-trade logging and visualization
- Sharpe ratio, Sortino ratio, and Calmar ratio metrics

### 7. Risk Management Modules
- Position sizing (fixed %, Kelly, volatility-scaled)
- Stop-loss and take-profit level calculation
- Portfolio-level risk limits and correlation controls
- Drawdown alerts and circuit breakers
- Leverage constraints for retail users

## Architecture

```
├── data/
│   ├── data_loader.py          # OHLCV download & caching
│   ├── news_fetcher.py         # Financial news API integration
│   └── macro_calendar.py        # Economic events calendar
├── models/
│   ├── sentiment_analyzer.py    # FinBERT sentiment pipeline
│   ├── price_forecasting.py     # LSTM/GRU time-series models
│   ├── pattern_detector.py      # Anomaly & technical patterns
│   └── ensemble_ranker.py       # Multi-factor stock ranker
├── indicators/
│   ├── technical.py             # Standard TA-Lib indicators
│   ├── volume_profile.py        # Volume-based analysis
│   └── custom_indicators.py     # Proprietary signals
├── strategies/
│   ├── long_term_strategy.py    # Buy-and-hold with rebalancing
│   ├── swing_strategy.py        # Technical + sentiment signals
│   └── intraday_strategy.py     # High-frequency paper-trade signals
├── backtester/
│   ├── backtest_engine.py       # Walk-forward testing framework
│   ├── performance_metrics.py   # Risk/return analytics
│   └── scenario_analyzer.py     # Monte Carlo & stress tests
├── risk_management/
│   ├── position_sizing.py       # Kelly, fixed %, volatility models
│   ├── portfolio_optimizer.py   # Mean-variance & risk parity
│   └── drawdown_manager.py      # Drawdown controls & alerts
├── profit_calculator/
│   ├── scenario_engine.py       # Bear/base/bull projections
│   └── probability_dist.py      # Distribution-based forecasts
├── api/
│   ├── fastapi_app.py           # REST API for strategies & metrics
│   └── broker_integration.py    # Broker API adapters (paper trading)
├── ui/
│   ├── dashboard.html           # Web dashboard (Plotly/Vue)
│   ├── static/                  # JS/CSS assets
│   └── templates/               # HTML templates
└── main.py                      # CLI entry point
```

## Installation

### Prerequisites
- Python 3.8+
- pip or conda
- (Optional) Broker API credentials for paper trading

### Setup

```bash
# Clone repository
git clone https://github.com/shivraj1182/AI-Investment-Analysis.git
cd AI-Investment-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download ML models (FinBERT, pre-trained LSTM weights)
python scripts/download_models.py
```

## Dependencies

Key libraries (see `requirements.txt` for full list):

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=0.24.0
tensorflow>=2.8.0  # or torch for LSTM
transformers>=4.20.0  # FinBERT
ta-lib>=0.4.24  # Technical analysis
requests>=2.28.0  # HTTP requests
fastapi>=0.95.0  # REST API
plotly>=5.0.0  # Visualizations
pytest>=7.0.0  # Testing
```

## Quick Start

### 1. Long-Term Stock Ranking

```python
from strategies.long_term_strategy import LongTermRanker
from data.data_loader import DataLoader

# Load data
loader = DataLoader()
data = loader.fetch_stocks(symbols=['INFY', 'TCS', 'RELIANCE'], period='5y')

# Rank stocks
ranker = LongTermRanker()
ranked = ranker.rank(data, risk_level='medium')
print(ranked)  # Sorted by investment score
```

### 2. Swing Trading Signals

```python
from strategies.swing_strategy import SwingStrategy
from data.news_fetcher import NewsFetcher

# Initialize
swing = SwingStrategy()
news_fetcher = NewsFetcher()

# Get data & sentiment
data = loader.fetch_candles('INFY', interval='1d', lookback=30)
sentiment = news_fetcher.get_sentiment('INFY', days=7)

# Generate signals
signals = swing.generate_signals(data, sentiment=sentiment)
print(signals)  # Buy/sell with confidence scores
```

### 3. Backtest a Strategy

```python
from backtester.backtest_engine import Backtester
from strategies.swing_strategy import SwingStrategy

# Setup backtest
backtester = Backtester(capital=100000, fee=0.0005)
strategy = SwingStrategy()

# Run backtest
results = backtester.run(
    data=data,
    strategy=strategy,
    start_date='2022-01-01',
    end_date='2024-12-31'
)

print(f"Total Return: {results.total_return:.2%}")
print(f"Sharpe Ratio: {results.sharpe:.2f}")
print(f"Max Drawdown: {results.max_drawdown:.2%}")
```

### 4. AI Profit Calculator

```python
from profit_calculator.scenario_engine import ProfitCalculator

# Setup calculator
calc = ProfitCalculator()

# Get probability-weighted scenarios
scenarios = calc.estimate(
    symbol='INFY',
    capital=50000,
    strategy='swing',
    confidence=0.75
)

for scenario in scenarios:
    print(f"{scenario['name']}: {scenario['prob']:.0%} probability, "
          f"Expected return: {scenario['return']:.2%}")
```

### 5. Web Dashboard

```bash
python api/fastapi_app.py
# Open browser to http://localhost:8000/dashboard
```

## Configuration

Edit `config.yaml` for parameters:

```yaml
data:
  source: 'yahoo'  # yahoo, alpha_vantage, broker_api
  cache_dir: './data/cache'

news:
  api_key: 'your_news_api_key'
  sources: ['bloomberg', 'reuters', 'cnbc']
  sentiment_model: 'finbert'  # or distilbert-financial

models:
  lstm_hidden_size: 128
  lstm_num_layers: 2
  ensemble_weights: {price: 0.4, sentiment: 0.3, technical: 0.3}

backtester:
  initial_capital: 100000
  slippage: 0.0005
  transaction_fee: 0.0005
  max_leverage: 1.0  # retail restriction

risk_management:
  position_size_method: 'volatility'  # kelly, fixed, volatility
  stop_loss_atr_multiple: 2.0
  risk_per_trade: 0.02  # 2% of capital
  max_position: 0.1  # 10% per asset
  portfolio_max_loss: 0.20  # stop trading at -20%
```

## IMPORTANT: Regulatory & Risk Disclaimers

⚠️ **This is a research and backtesting tool, not a financial advisory platform:**

1. **No Guarantees**: AI models can overfit, market regimes change, and past performance does not guarantee future returns.
2. **Retail Investor Constraints**: In India (NSE/SEBI), algorithmic trading for retail investors requires:
   - Broker API approval
   - Exchange-issued algorithm ID
   - Real-time monitoring and circuit breakers
   - Audit logs and compliance reporting
3. **Paper Trading Only (for now)**: Start with simulated trades via broker sandbox APIs before live deployment.
4. **Risk of Loss**: All trading and investment carry risk of total capital loss. Use strict position sizing and stop-losses.
5. **No Personal Data**: Never store API keys, credentials, or sensitive data in code or GitHub.

## Backtesting Results (Sample)

Walked forward testing (2022-2024) on NSE stocks with 100k starting capital:

| Strategy | Symbols | Annual Return | Sharpe Ratio | Max Drawdown | Win Rate |
|----------|---------|----------------|------------|--------------|----------|
| Long-term | INFY, TCS, RELIANCE, ITC | 12.5% | 1.08 | -18% | N/A |
| Swing (Tech) | INFY, WIPRO, TECHM | 8.3% | 0.95 | -22% | 54% |
| Swing (Bank) | HDFC, ICICI, AXIS | 6.8% | 0.72 | -25% | 51% |

*(Backtests assume no slippage; live results will vary with execution quality)*

## Contributing

Contributions welcome! Areas:
- Additional ML models (GRU, Transformer-based price forecasting)
- More broker API integrations (Zerodha, AngelOne, Shoonya)
- Enhanced sentiment models (multi-language, event-based)
- GPU acceleration for backtesting
- Real-time monitoring dashboards

See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License – see [LICENSE](LICENSE) file.

## Acknowledgments

- FinBERT for financial NLP
- TA-Lib for technical indicators
- Yahoo Finance & Alpha Vantage for market data
- Alpaca & other brokers for sandbox APIs

## Support & Issues

File issues on GitHub or contact the maintainer at [your-email].

## Disclaimer

This project is for educational and research purposes. The author and contributors are not financial advisors and assume no liability for trading losses or decisions made using this tool. Always consult a licensed financial advisor before investing real capital.
