# Contributing to AI-Investment-Analysis

Thank you for your interest in contributing to AI-Investment-Analysis! This document provides guidelines for contributing to the project.

## Code of Conduct

This project is committed to providing a welcoming and inclusive environment. Please be respectful of all contributors.

## Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- Familiarity with financial markets and machine learning (preferred)

### Setup Development Environment

```bash
# Clone the repository
git clone https://github.com/shivraj1182/AI-Investment-Analysis.git
cd AI-Investment-Analysis

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies with dev tools
pip install -r requirements.txt
pip install pytest pytest-cov black flake8 isort

# Create directories
mkdir -p data/cache logs
```

## Contributing Guidelines

### Issues

Before opening an issue, please check if it already exists. When creating an issue:

1. Use a clear, descriptive title
2. Describe the exact steps to reproduce the problem
3. Provide specific examples
4. Describe the observed vs. expected behavior
5. Include your environment (OS, Python version, etc.)

### Pull Requests

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature-name`
3. **Make your changes** following code style guidelines
4. **Add tests** for new functionality
5. **Run tests locally**: `pytest`
6. **Commit with clear messages**: `git commit -m 'Add feature: description'`
7. **Push to your fork**: `git push origin feature/your-feature-name`
8. **Create a Pull Request** with detailed description

## Code Style

This project uses PEP 8 style guidelines:

```bash
# Format code
black --line-length 100 .

# Sort imports
isort .

# Lint check
flake8 --max-line-length 100 --exclude venv
```

## Testing

All contributions must include tests:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=./ --cov-report=html

# Run specific test file
pytest tests/test_strategies.py -v
```

## Documentation

- Add docstrings to all functions and classes (Google style)
- Update README.md if adding major features
- Add comments for complex logic
- Update API documentation if changing interfaces

## Areas for Contribution

We welcome contributions in these areas:

### 1. ML Models
- Implement GRU/Transformer-based price forecasting
- Add ensemble learning techniques
- Improve sentiment analysis with multilingual support
- Develop anomaly detection systems

### 2. Broker Integration
- Add Zerodha, AngelOne, Shoonya APIs
- Implement paper trading via broker sandboxes
- Add order management and execution

### 3. Technical Features
- GPU acceleration for backtesting
- Real-time data streaming
- Web dashboard (React/Vue)
- Mobile app support

### 4. Strategy Development
- Implement additional technical indicators
- Create new trading strategies
- Add portfolio optimization algorithms
- Develop macro-economic factor models

### 5. Documentation & Examples
- Add tutorial notebooks
- Create example strategies
- Improve API documentation
- Add video guides

## Commit Message Convention

Use clear, descriptive commit messages:

```
[TYPE] Brief description (50 chars)

More detailed explanation if needed (72 chars per line).
Reference issues: Closes #123
```

Types: feat, fix, docs, style, refactor, test, chore

Example:
```
feat: Add sentiment-based position sizing

Implement position sizing adjustments based on financial news
sentiment scores. Uses FinBERT for multi-day sentiment aggregation.

Closes #42
```

## Review Process

1. At least one maintainer review required
2. Tests must pass (GitHub Actions CI)
3. Code coverage should not decrease
4. No merge conflicts
5. All conversations resolved

## Legal

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

Feel free to open an issue with the 'question' label or contact the maintainers.

## License

MIT License - see LICENSE file for details.

---

**Thank you for contributing to AI-Investment-Analysis!** 🚀
