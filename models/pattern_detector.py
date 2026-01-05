import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import pandas as pd

logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Anomaly detection using Isolation Forest for price anomalies.
    
    Detects unusual price movements and trading patterns that may indicate
    trading halts, manipulation, or significant market events.
    """
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of outliers
            random_state: Random seed for reproducibility
        """
        self.contamination = contamination
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
        logger.info("AnomalyDetector initialized")
    
    def detect_anomalies(self, price_data: np.ndarray) -> Dict:
        """
        Detect anomalies in price data.
        
        Args:
            price_data: Array of prices
            
        Returns:
            Dictionary with anomaly indicators
        """
        try:
            # Calculate technical indicators
            returns = np.diff(price_data) / price_data[:-1]
            volatility = np.std(returns)
            
            # Prepare features
            features = np.array([
                np.abs(returns),  # Absolute returns
                np.square(returns),  # Squared returns
                volatility  # Volatility
            ]).T
            
            # Scale features
            features_scaled = self.scaler.fit_transform(features)
            
            # Detect anomalies
            predictions = self.model.predict(features_scaled)
            anomaly_scores = self.model.score_samples(features_scaled)
            
            anomaly_indices = np.where(predictions == -1)[0]
            
            return {
                'has_anomalies': len(anomaly_indices) > 0,
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_scores': anomaly_scores.tolist(),
                'volatility': float(volatility)
            }
        except Exception as e:
            logger.error(f"Anomaly detection failed: {str(e)}")
            return {'error': str(e)}
    
    def detect_pattern_anomalies(self, price_data: np.ndarray,
                                  window_size: int = 5) -> Dict:
        """
        Detect anomalous patterns in price movement.
        
        Args:
            price_data: Array of prices
            window_size: Size of rolling window
            
        Returns:
            Dictionary with pattern anomalies
        """
        try:
            anomalies = []
            
            for i in range(len(price_data) - window_size):
                window = price_data[i:i + window_size]
                
                # Calculate statistics
                mean_price = np.mean(window)
                std_price = np.std(window)
                max_return = np.max(np.diff(window) / window[:-1])
                min_return = np.min(np.diff(window) / window[:-1])
                
                # Detect anomalies
                if std_price > mean_price * 0.1 or np.abs(max_return) > 0.05:
                    anomalies.append({
                        'index': i,
                        'max_return': float(max_return),
                        'min_return': float(min_return),
                        'volatility': float(std_price)
                    })
            
            return {
                'pattern_anomalies': anomalies,
                'anomaly_count': len(anomalies),
                'anomaly_positions': [a['index'] for a in anomalies]
            }
        except Exception as e:
            logger.error(f"Pattern anomaly detection failed: {str(e)}")
            return {'error': str(e)}
    
    def detect_volume_anomalies(self, volumes: np.ndarray) -> Dict:
        """
        Detect unusual trading volumes.
        
        Args:
            volumes: Array of trading volumes
            
        Returns:
            Dictionary with volume anomalies
        """
        try:
            mean_volume = np.mean(volumes)
            std_volume = np.std(volumes)
            threshold = mean_volume + 2 * std_volume
            
            anomaly_indices = np.where(volumes > threshold)[0]
            
            return {
                'has_volume_anomalies': len(anomaly_indices) > 0,
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'mean_volume': float(mean_volume),
                'threshold': float(threshold),
                'anomalous_volumes': volumes[anomaly_indices].tolist()
            }
        except Exception as e:
            logger.error(f"Volume anomaly detection failed: {str(e)}")
            return {'error': str(e)}
    
    def detect_spike_anomalies(self, price_data: np.ndarray,
                               threshold: float = 0.05) -> Dict:
        """
        Detect sudden price spikes or crashes.
        
        Args:
            price_data: Array of prices
            threshold: Return threshold for spike detection
            
        Returns:
            Dictionary with spike anomalies
        """
        try:
            returns = np.diff(price_data) / price_data[:-1]
            spike_indices = np.where(np.abs(returns) > threshold)[0]
            
            spikes = []
            for idx in spike_indices:
                spikes.append({
                    'index': int(idx),
                    'return': float(returns[idx]),
                    'price_before': float(price_data[idx]),
                    'price_after': float(price_data[idx + 1]),
                    'type': 'spike' if returns[idx] > 0 else 'crash'
                })
            
            return {
                'has_spikes': len(spikes) > 0,
                'spike_count': len(spikes),
                'spikes': spikes,
                'threshold': threshold
            }
        except Exception as e:
            logger.error(f"Spike detection failed: {str(e)}")
            return {'error': str(e)}
