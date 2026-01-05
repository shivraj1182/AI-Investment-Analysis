import logging
import numpy as np
from typing import Dict, List, Tuple, Optional
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

logger = logging.getLogger(__name__)


class SentimentAnalyzer:
    """Sentiment analysis using FinBERT model for financial text analysis.
    
    FinBERT is a BERT-based model pre-trained on financial data, providing
    superior sentiment analysis for financial news and documents.
    """
    
    def __init__(self, model_name: str = "ProsusAI/finbert", device: str = "cpu"):
        """
        Initialize the sentiment analyzer.
        
        Args:
            model_name: HuggingFace model identifier
            device: 'cpu' or 'cuda' for processing device
        """
        self.device = device
        self.model_name = model_name
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.to(device)
            self.model.eval()
            logger.info(f"FinBERT model loaded: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def analyze_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Input text for analysis
            
        Returns:
            Dictionary with sentiment scores (negative, neutral, positive)
        """
        try:
            # Tokenize input
            inputs = self.tokenizer(text, return_tensors="pt", 
                                   max_length=512, truncation=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=1)
            
            # Map to sentiment labels
            labels = ['negative', 'neutral', 'positive']
            scores = probabilities[0].cpu().numpy()
            
            return {
                'negative': float(scores[0]),
                'neutral': float(scores[1]),
                'positive': float(scores[2]),
                'dominant_sentiment': labels[np.argmax(scores)]
            }
        except Exception as e:
            logger.error(f"Sentiment analysis failed: {str(e)}")
            return {'error': str(e)}
    
    def batch_analyze(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        """
        Analyze sentiment for multiple texts efficiently.
        
        Args:
            texts: List of input texts
            batch_size: Number of texts to process at once
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            
            try:
                inputs = self.tokenizer(batch, return_tensors="pt",
                                       max_length=512, truncation=True,
                                       padding=True)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits
                    probabilities = torch.softmax(logits, dim=1)
                
                labels = ['negative', 'neutral', 'positive']
                scores = probabilities.cpu().numpy()
                
                for j, text in enumerate(batch):
                    results.append({
                        'text': text[:100],  # Store first 100 chars
                        'negative': float(scores[j, 0]),
                        'neutral': float(scores[j, 1]),
                        'positive': float(scores[j, 2]),
                        'dominant_sentiment': labels[np.argmax(scores[j])]
                    })
            except Exception as e:
                logger.error(f"Batch analysis failed: {str(e)}")
                for text in batch:
                    results.append({'text': text[:100], 'error': str(e)})
        
        return results
    
    def get_sentiment_score(self, text: str) -> float:
        """
        Get aggregated sentiment score (-1 to 1).
        
        Args:
            text: Input text
            
        Returns:
            Score from -1 (most negative) to 1 (most positive)
        """
        sentiment = self.analyze_sentiment(text)
        
        if 'error' in sentiment:
            return 0.0
        
        # Compute weighted score
        score = (sentiment['positive'] * 1.0 + 
                sentiment['neutral'] * 0.0 - 
                sentiment['negative'] * 1.0)
        
        return float(np.clip(score, -1, 1))
    
    def analyze_news_list(self, news_items: List[Dict]) -> List[Dict]:
        """
        Analyze sentiment for a list of news items.
        
        Args:
            news_items: List of dicts with 'headline' and 'content' keys
            
        Returns:
            News items with added sentiment analysis
        """
        results = []
        
        for item in news_items:
            try:
                # Combine headline and content for analysis
                text = f"{item.get('headline', '')} {item.get('content', '')}"
                sentiment = self.analyze_sentiment(text)
                
                item['sentiment'] = sentiment
                item['sentiment_score'] = self.get_sentiment_score(text)
                results.append(item)
            except Exception as e:
                logger.error(f"Failed to analyze news item: {str(e)}")
                item['sentiment'] = {'error': str(e)}
                results.append(item)
        
        return results
