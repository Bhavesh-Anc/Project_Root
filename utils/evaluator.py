import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
from typing import Dict, Any
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from tqdm import tqdm
from functools import lru_cache
from concurrent.futures import ProcessPoolExecutor, as_completed
from utils.model import ModelMonitor

# Define HORIZONS locally to avoid circular import
HORIZONS = {
    'next_day': 1,
    'next_week': 5,
    'next_month': 22,
    'next_quarter': 66,
    'next_year': 252,
    'next_3_years': 756,
    'next_5_years': 1260
}

# Configuration
COLORS = {
    'good': '#4CAF50',
    'bad': '#F44336',
    'neutral': '#2196F3'
}

class StockEvaluator:
    """Comprehensive model evaluation and reporting system with monitoring integration"""
    
    def __init__(self, model_registry: Dict[str, Any], data_registry: Dict[str, pd.DataFrame]):
        self.models = model_registry
        self.data = data_registry
        self.metrics = pd.DataFrame()
        self.risk_scores = pd.DataFrame()
        self.feature_importances = pd.DataFrame()
        self.monte_carlo_results = {}
        self.monitor = ModelMonitor()

    def full_evaluation(self, output_dir: str = "reports"):
        """Run complete evaluation pipeline with monitoring"""
        try:
            self._calculate_metrics()
            self._analyze_feature_importance()
            self._evaluate_risk()
            self._monte_carlo_analysis()
            self._monitor_performance()
            self.generate_reports(output_dir)
        except Exception as e:
            warnings.warn(f"Full evaluation failed: {str(e)}")
    
    def _monitor_performance(self):
        """Log performance metrics to monitoring system"""
        if self.metrics.empty:
            return
            
        for _, row in self.metrics.iterrows():
            model_id = f"{row['ticker']}_{row['horizon']}"
            metrics = {
                'accuracy': row.get('accuracy', 0.5),
                'precision': row.get('precision', 0.5),
                'recall': row.get('recall', 0.5),
                'f1': row.get('f1', 0.5),
                'roc_auc': row.get('roc_auc', 0.5)
            }
            self.monitor.log_performance(model_id, metrics)

def load_evaluation(output_dir: str = "reports") -> Dict[str, pd.DataFrame]:
    """Load saved evaluation results"""
    try:
        return {
            'metrics': pd.read_csv(os.path.join(output_dir, 'performance_metrics.csv')),
            'risk_scores': pd.read_csv(os.path.join(output_dir, 'risk_scores.csv')),
            'feature_importances': pd.read_csv(os.path.join(output_dir, 'feature_importances.csv'))
        }
    except FileNotFoundError as e:
        warnings.warn(f"Missing evaluation files: {str(e)}")
        return {}
    except Exception as e:
        warnings.warn(f"Error loading evaluation: {str(e)}")
        return {}

if __name__ == "__main__":
    # Example usage
    try:
        from utils.model import load_models
        from utils.feature_engineer import load_processed_data
        
        data = load_processed_data()
        models = load_models()
        
        if models and data:
            evaluator = StockEvaluator(models, data)
            evaluator.full_evaluation()
            print("Top Performers:", evaluator.get_top_performers())
        else:
            print("Evaluation failed - missing data or models")
    except ImportError as e:
        print(f"Import error: {e}")
        print("Please ensure all required modules are available")