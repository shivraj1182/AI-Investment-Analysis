# Quick Start Guide

## Getting Started with AI-Investment-Analysis

This guide will help you set up and start using the AI-Investment-Analysis platform in minutes.

### 1. Clone & Setup (5 minutes)

```bash
# Clone the repository
git clone https://github.com/shivraj1182/AI-Investment-Analysis.git
cd AI-Investment-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Create data directories
mkdir -p data/cache logs
```

### 2. First Run (2 minutes)

```bash
# Display help
python main.py --help

# Show version info
python main.py longterm --help
```

### 3. Test Data Loading

```python
from data import DataLoader

# Initialize loader
loader = DataLoader(cache_dir='./data/cache')

# Fetch data for a stock
data = loader.fetch_ohlcv('INFY.NS', period='1y')
print(f"Loaded {len(data)} candles")
print(data.head())
```

### 4. Next Steps

Refer to the [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) for detailed phase-by-phase guidance on:
- Implementing ML models
- Building trading strategies
- Setting up backtesting
- Creating APIs
- And much more!

## Architecture Overview

```
data/              # Data loading & caching
├── data_loader.py       # OHLCV fetching
├── news_fetcher.py      # News & sentiment (TODO)
└── macro_calendar.py    # Economic events (TODO)

models/            # ML & AI Models (TODO)
├── sentiment_analyzer.py
├── price_forecasting.py
├── pattern_detector.py
└── ensemble_ranker.py

indicators/        # Technical Indicators (TODO)
├── technical.py
├── volume_profile.py
└── custom_indicators.py

strategies/        # Trading Strategies (TODO)
├── long_term_strategy.py
├── swing_strategy.py
└── intraday_strategy.py

backtester/        # Backtesting Engine (TODO)
├── backtest_engine.py
├── performance_metrics.py
└── scenario_analyzer.py

risk_management/   # Risk Controls (TODO)
├── position_sizing.py
├── portfolio_optimizer.py
└── drawdown_manager.py

profit_calculator/ # Profit Projection (TODO)
├── scenario_engine.py
└── probability_dist.py

api/               # Web API (TODO)
├── fastapi_app.py
└── broker_integration.py
```

## Key Files

- **config.yaml** - All configuration parameters
- **main.py** - CLI entry point with commands
- **requirements.txt** - All dependencies
- **README.md** - Complete documentation
- **CONTRIBUTING.md** - Contribution guidelines
- **IMPLEMENTATION_ROADMAP.md** - Detailed development plan

## Common Commands

```bash
# Long-term analysis
python main.py longterm -s INFY.NS TCS.NS RELIANCE.NS

# Swing trading signals
python main.py swing -s INFY.NS --days 30 --use-sentiment

# Backtest a strategy
python main.py backtest -st swing -s INFY.NS -sd 2023-01-01 -ed 2024-12-31

# Get profit projections
python main.py profit -s INFY.NS -c 50000 -st swing

# Start API server
python main.py server --port 8000
```

## Troubleshooting

### Issue: ModuleNotFoundError

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: Data fetch fails

```bash
# Clear cache and retry
python -c "from data import DataLoader; DataLoader().clear_cache()"
```

### Issue: Configuration errors

```bash
# Validate configuration
python -c "import yaml; yaml.safe_load(open('config.yaml'))"
```

## Development Workflow

1. **Create feature branch**
   ```bash
   git checkout -b feature/your-feature
   ```

2. **Implement & test**
   ```bash
   # Follow code style
   black .
   isort .
   flake8 .
   
   # Run tests
   pytest
   ```

3. **Commit & push**
   ```bash
   git add .
   git commit -m "feat: Add your feature"
   git push origin feature/your-feature
   ```

4. **Create pull request**

## Learning Resources

- **FinBERT Documentation**: https://github.com/ProsusAI/finbert
- **TA-Lib Guide**: https://github.com/mrjbq7/ta-lib
- **FastAPI Tutorial**: https://fastapi.tiangolo.com/
- **Pandas Documentation**: https://pandas.pydata.org/
- **Scikit-Learn Guide**: https://scikit-learn.org/

## Support

- Check [README.md](README.md) for detailed documentation
- Review [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines
- See [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) for development plan
- File issues on GitHub for bugs or questions

## Next Phase: Building Your First Strategy

After completing setup, refer to the IMPLEMENTATION_ROADMAP for:
1. Implementing sentiment analysis (data/news_fetcher.py)
2. Building price forecasting models (models/price_forecasting.py)
3. Creating technical indicators (indicators/technical.py)
4. Developing trading strategies (strategies/)
5. Building backtesting engine (backtester/)

Good luck! 🚀
