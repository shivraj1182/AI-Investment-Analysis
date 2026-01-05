import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelPrediction:
    """Container for model predictions."""
    model_name: str
    prediction: float
    confidence: float
    metadata: Dict


class EnsembleRanker:
    """Ensemble model that combines predictions from multiple models.
    
    Uses weighted voting and averaging to produce robust investment signals
    by combining sentiment analysis, price forecasting, and anomaly detection.
    """
    
    def __init__(self):
        """Initialize the ensemble ranker with default weights."""
        self.weights = {
            'sentiment': 0.25,
            'price_forecast': 0.35,
            'anomaly_score': 0.20,
            'volume_signal': 0.10,
            'pattern_signal': 0.10
        }
        logger.info("EnsembleRanker initialized")
    
    def set_weights(self, weights: Dict[str, float]):
        """
        Set custom weights for model predictions.
        
        Args:
            weights: Dictionary mapping model names to weights (must sum to 1)
        """
        if not np.isclose(sum(weights.values()), 1.0):
            raise ValueError("Weights must sum to 1.0")
        self.weights = weights
        logger.info(f"Weights updated: {weights}")
    
    def rank_assets(self, predictions: List[Dict]) -> List[Dict]:
        """
        Rank assets based on ensemble predictions.
        
        Args:
            predictions: List of prediction dictionaries from different models
            
        Returns:
            Ranked list of assets with ensemble scores
        """
        try:
            ranked_assets = []
            
            for pred in predictions:
                # Extract scores from different models
                sentiment_score = pred.get('sentiment_score', 0) * self.weights['sentiment']
                price_score = pred.get('price_momentum', 0) * self.weights['price_forecast']
                anomaly_penalty = (1 - pred.get('anomaly_risk', 0)) * self.weights['anomaly_score']
                volume_score = pred.get('volume_signal', 0) * self.weights['volume_signal']
                pattern_score = pred.get('pattern_score', 0) * self.weights['pattern_signal']
                
                # Combine scores
                ensemble_score = (
                    sentiment_score + price_score + anomaly_penalty +
                    volume_score + pattern_score
                )
                
                # Normalize to -1 to 1 range
                ensemble_score = np.clip(ensemble_score, -1, 1)
                
                ranked_assets.append({
                    'asset': pred.get('symbol', pred.get('asset')),
                    'ensemble_score': float(ensemble_score),
                    'signal': self._score_to_signal(ensemble_score),
                    'confidence': float(np.abs(ensemble_score)),
                    'component_scores': {
                        'sentiment': float(sentiment_score),
                        'price_forecast': float(price_score),
                        'anomaly': float(anomaly_penalty),
                        'volume': float(volume_score),
                        'pattern': float(pattern_score)
                    }
                })
            
            # Sort by ensemble score (descending)
            ranked_assets.sort(key=lambda x: x['ensemble_score'], reverse=True)
            
            # Add rank
            for i, asset in enumerate(ranked_assets):
                asset['rank'] = i + 1
            
            return ranked_assets
        except Exception as e:
            logger.error(f"Asset ranking failed: {str(e)}")
            return []
    
    def _score_to_signal(self, score: float) -> str:
        """
        Convert numerical score to trading signal.
        
        Args:
            score: Score from -1 to 1
            
        Returns:
            Signal: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
        """
        if score >= 0.6:
            return 'STRONG_BUY'
        elif score >= 0.2:
            return 'BUY'
        elif score >= -0.2:
            return 'HOLD'
        elif score >= -0.6:
            return 'SELL'
        else:
            return 'STRONG_SELL'
    
    def get_top_picks(self, ranked_assets: List[Dict],
                     top_n: int = 5) -> List[Dict]:
        """
        Get top N assets from ranked list.
        
        Args:
            ranked_assets: Pre-ranked assets
            top_n: Number of top assets to return
            
        Returns:
            Top N assets
        """
        return ranked_assets[:top_n]
    
    def calculate_portfolio_score(self, ranked_assets: List[Dict]) -> float:
        """
        Calculate overall portfolio score.
        
        Args:
            ranked_assets: Ranked assets list
            
        Returns:
            Portfolio score (weighted average)
        """
        if not ranked_assets:
            return 0.0
        
        scores = [asset['ensemble_score'] for asset in ranked_assets]
        portfolio_score = np.mean(scores)
        
        return float(portfolio_score)
    
    def explain_prediction(self, asset: Dict) -> Dict:
        """
        Explain ensemble prediction for an asset.
        
        Args:
            asset: Ranked asset dictionary
            
        Returns:
            Explanation dictionary
        """
        components = asset.get('component_scores', {})
        
        explanation = {
            'asset': asset.get('asset'),
            'final_score': asset.get('ensemble_score'),
            'signal': asset.get('signal'),
            'explanation': self._generate_explanation(components),
            'component_breakdown': components
        }
        
        return explanation
    
    def _generate_explanation(self, components: Dict) -> str:
        """
        Generate textual explanation of prediction.
        
        Args:
            components: Component scores
            
        Returns:
            Explanation string
        """
        max_component = max(components.items(), key=lambda x: abs(x[1]))
        
        explanation = f"Ensemble decision primarily driven by {max_component[0]} signal. "
        
        if max_component[1] > 0:
            explanation += "Positive indicators suggest upside potential."
        else:
            explanation += "Negative indicators suggest downside risk."
        
        return explanation
