import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import pandas as pd

logger = logging.getLogger(__name__)


class LSTMPredictor:
    """LSTM-based price forecasting model for time series prediction.
    
    Uses Long Short-Term Memory neural networks for multi-horizon price
    forecasting (1-day, 7-day, 30-day ahead predictions).
    """
    
    def __init__(self, input_size: int = 5, hidden_size: int = 64,
                 num_layers: int = 2, output_size: int = 3,
                 device: str = "cpu"):
        """
        Initialize the LSTM price predictor.
        
        Args:
            input_size: Number of input features (OHLCV)
            hidden_size: Hidden layer dimension
            num_layers: Number of LSTM layers
            output_size: Number of outputs (predictions)
            device: 'cpu' or 'cuda'
        """
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.scaler = MinMaxScaler()
        
        try:
            self.model = LSTMModel(input_size, hidden_size, num_layers, output_size)
            self.model.to(device)
            self.model.eval()
            logger.info("LSTM model initialized")
        except Exception as e:
            logger.error(f"Failed to initialize LSTM model: {str(e)}")
            raise
    
    def prepare_sequences(self, data: np.ndarray, seq_length: int = 60) -> Tuple:
        """
        Prepare sequences for LSTM input.
        
        Args:
            data: Time series data (normalized)
            seq_length: Length of sequence windows
            
        Returns:
            X and y arrays for training
        """
        X, y = [], []
        
        for i in range(len(data) - seq_length):
            X.append(data[i:i + seq_length])
            y.append(data[i + seq_length])
        
        return np.array(X), np.array(y)
    
    def normalize_data(self, data: np.ndarray) -> np.ndarray:
        """
        Normalize data using MinMaxScaler.
        
        Args:
            data: Raw price data
            
        Returns:
            Normalized data
        """
        data_2d = data.reshape(-1, 1) if len(data.shape) == 1 else data
        return self.scaler.fit_transform(data_2d)
    
    def predict(self, price_sequence: np.ndarray) -> Dict[str, float]:
        """
        Predict price for multiple horizons.
        
        Args:
            price_sequence: Previous price sequence
            
        Returns:
            Dictionary with 1-day, 7-day, and 30-day predictions
        """
        try:
            # Normalize input
            normalized = self.normalize_data(price_sequence)
            
            # Convert to tensor
            input_tensor = torch.FloatTensor(normalized[-60:]).unsqueeze(0).to(self.device)
            
            # Make prediction
            with torch.no_grad():
                output = self.model(input_tensor)
            
            # Denormalize predictions
            predictions = output.cpu().numpy()[0]
            
            # Inverse transform predictions
            denormalized = self.scaler.inverse_transform(
                predictions.reshape(-1, 1)
            ).flatten()
            
            return {
                'next_day': float(denormalized[0]),
                'week_ahead': float(denormalized[1]) if len(denormalized) > 1 else None,
                'month_ahead': float(denormalized[2]) if len(denormalized) > 2 else None,
                'confidence': 0.85  # Placeholder confidence score
            }
        except Exception as e:
            logger.error(f"Prediction failed: {str(e)}")
            return {'error': str(e)}
    
    def batch_predict(self, price_data: List[np.ndarray]) -> List[Dict]:
        """
        Make predictions for multiple price sequences.
        
        Args:
            price_data: List of price sequences
            
        Returns:
            List of prediction dictionaries
        """
        results = []
        
        for prices in price_data:
            result = self.predict(prices)
            results.append(result)
        
        return results
    
    def get_confidence_interval(self, predictions: Dict,
                               confidence_level: float = 0.95) -> Dict:
        """
        Calculate confidence intervals for predictions.
        
        Args:
            predictions: Model predictions
            confidence_level: Confidence level (default 95%)
            
        Returns:
            Dictionary with confidence intervals
        """
        # Simple confidence band calculation
        margin = 0.05  # 5% margin from prediction
        
        return {
            'next_day_lower': predictions['next_day'] * (1 - margin),
            'next_day_upper': predictions['next_day'] * (1 + margin),
            'confidence_level': confidence_level
        }


class LSTMModel(nn.Module):
    """PyTorch LSTM model for sequence prediction."""
    
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 output_size: int):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                           batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through LSTM network.
        
        Args:
            x: Input tensor of shape (batch_size, seq_length, input_size)
            
        Returns:
            Output predictions
        """
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        output = self.fc(last_hidden)
        return output
