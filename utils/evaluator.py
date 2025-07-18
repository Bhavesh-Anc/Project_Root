# utils/evaluator.py
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

def _single_simulation(args):
    """Helper function for parallel Monte Carlo simulation"""
    mean, std, horizon_days, n_simulations = args
    # Vectorized simulation for one ticker
    simulations = np.random.normal(mean, std, (n_simulations, horizon_days)).cumsum(axis=1)
    return simulations[:, -1]

class StockEvaluator:
    """Comprehensive model evaluation and reporting system"""
    
    def __init__(self, model_registry: Dict[str, Any], data_registry: Dict[str, pd.DataFrame]):
        self.models = model_registry
        self.data = data_registry
        self.metrics = pd.DataFrame()
        self.risk_scores = pd.DataFrame()
        self.feature_importances = pd.DataFrame()
        self.monte_carlo_results = {}

    def full_evaluation(self, output_dir: str = "reports"):
        """Run complete evaluation pipeline"""
        try:
            self._calculate_metrics()
            self._analyze_feature_importance()
            self._evaluate_risk()
            self._monte_carlo_analysis()
            self.generate_reports(output_dir)
        except Exception as e:
            warnings.warn(f"Full evaluation failed: {str(e)}")

    def _calculate_metrics(self):
        """Calculate performance metrics for all models"""
        metrics = []
        
        if not self.models:
            warnings.warn("No models available for evaluation")
            return

        for ticker, model_dict in tqdm(self.models.items(), desc="Evaluating models"):
            if ticker not in self.data:
                continue
                
            df = self.data[ticker]
            
            for horizon, model in model_dict.items():
                try:
                    # Recreate test split
                    split_idx = int(len(df) * 0.8)
                    test = df.iloc[split_idx:]
                    
                    if test.empty:
                        continue
                        
                    target_col = f"Target_{horizon}"
                    if target_col not in test.columns:
                        continue
                        
                    X_test = test.drop(columns=[target_col], errors='ignore')
                    y_test = test[target_col]
                    
                    if X_test.empty or y_test.empty:
                        continue

                    # Check if model has required attributes
                    if not hasattr(model, 'model') or model.model is None:
                        continue

                    preds = model.model.predict(X_test)
                    probas = model.model.predict_proba(X_test)[:, 1] if hasattr(model.model, "predict_proba") else [0.5]*len(X_test)

                    metrics.append({
                        'ticker': ticker,
                        'horizon': horizon,
                        'precision': precision_score(y_test, preds, zero_division=0),
                        'recall': recall_score(y_test, preds, zero_division=0),
                        'f1': f1_score(y_test, preds, zero_division=0),
                        'roc_auc': roc_auc_score(y_test, probas) if len(np.unique(y_test)) > 1 else 0.5,
                        'confusion_matrix': confusion_matrix(y_test, preds).tolist(),
                        'classification_report': classification_report(
                            y_test, preds, output_dict=True, zero_division=0
                        )
                    })
                except Exception as e:
                    warnings.warn(f"Evaluation failed for {ticker} {horizon}: {str(e)}")
                    
        self.metrics = pd.DataFrame(metrics)

    def _analyze_feature_importance(self):
        """Aggregate feature importance across models"""
        fi_data = []
        
        if not self.models:
            return

        for ticker, model_dict in self.models.items():
            for horizon, model in model_dict.items():
                try:
                    if not hasattr(model, 'feature_importances') or not model.feature_importances:
                        continue
                        
                    for feature, importance in model.feature_importances.items():
                        fi_data.append({
                            'ticker': ticker,
                            'horizon': horizon,
                            'feature': feature,
                            'importance': importance
                        })
                except Exception as e:
                    warnings.warn(f"Feature importance failed for {ticker} {horizon}: {str(e)}")
                    
        self.feature_importances = pd.DataFrame(fi_data)

    def _evaluate_risk(self):
        """Calculate composite risk scores"""
        risk_data = []
        
        for ticker, df in self.data.items():
            try:
                if 'Close' not in df.columns or 'Volume' not in df.columns:
                    continue
                    
                returns = df['Close'].pct_change().dropna()
                if returns.empty:
                    continue
                    
                volatility = returns.std() * np.sqrt(252)
                max_drawdown = (df['Close'].cummax() - df['Close']).max() / df['Close'].cummax().max()
                volume_volatility = df['Volume'].pct_change().std()
                
                # Handle NaN values
                if np.isnan(volume_volatility):
                    volume_volatility = 0
                
                risk_score = 0.4*volatility + 0.3*max_drawdown + 0.2*volume_volatility
                
                risk_data.append({
                    'ticker': ticker,
                    'volatility': volatility,
                    'max_drawdown': max_drawdown,
                    'volume_risk': volume_volatility,
                    'composite_risk': risk_score
                })
            except Exception as e:
                warnings.warn(f"Risk evaluation failed for {ticker}: {str(e)}")
                
        self.risk_scores = pd.DataFrame(risk_data)

    @lru_cache(maxsize=32)
    def _returns_cache(self, ticker, horizon_days):
        """Cache returns calculation for performance"""
        df = self.data[ticker]
        returns = np.log(1 + df['Close'].pct_change().dropna())
        return returns.mean(), returns.std()

    def _monte_carlo_analysis(self, n_simulations: int = 1000, parallel: bool = True):
        """Run Monte Carlo simulations for long-term predictions, parallelized and cached"""
        if not self.models:
            return
            
        try:
            # Get sample model to find long-term horizons
            sample_model = next(iter(self.models.values()))
            long_term_horizons = [h for h in sample_model.keys() if 'year' in h]
        except StopIteration:
            return

        for horizon in long_term_horizons:
            try:
                # Extract horizon name without model type prefix
                horizon_name = horizon.split('_', 1)[-1] if '_' in horizon else horizon
                horizon_days = HORIZONS.get(horizon_name)
                
                if not horizon_days:
                    continue
                    
                all_simulations = []
                tasks = []
                tickers = []
                
                for ticker, df in self.data.items():
                    if 'Close' not in df.columns:
                        continue

                    # Skip if not enough data
                    if len(df) < horizon_days:
                        continue
                        
                    returns = np.log(1 + df['Close'].pct_change().dropna())
                    if returns.empty:
                        continue

                    mean, std = returns.mean(), returns.std()
                    tasks.append((mean, std, horizon_days, n_simulations))
                    tickers.append(ticker)

                # Skip if no tasks
                if not tasks:
                    continue
                    
                # Parallel execution
                if parallel and len(tasks) > 1:
                    with ProcessPoolExecutor() as executor:
                        results = list(executor.map(_single_simulation, tasks))
                else:
                    results = [_single_simulation(args) for args in tasks]

                for ticker, final_returns in zip(tickers, results):
                    if len(final_returns) > 0:
                        all_simulations.append({
                            'ticker': ticker,
                            'mean_return': np.mean(final_returns),
                            'p5': np.percentile(final_returns, 5),
                            'p95': np.percentile(final_returns, 95)
                        })

                self.monte_carlo_results[horizon] = pd.DataFrame(all_simulations)
            except Exception as e:
                warnings.warn(f"Monte Carlo failed for {horizon}: {str(e)}")

    def generate_reports(self, output_dir: str):
        """Generate all evaluation reports and visualizations"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # Save numerical reports
            self._save_metric_reports(output_dir)
            
            # Generate visualizations
            self._plot_performance_metrics(output_dir)
            self._plot_feature_importances(output_dir)
            self._plot_risk_analysis(output_dir)
            self._plot_monte_carlo_results(output_dir)
            
        except Exception as e:
            warnings.warn(f"Report generation failed: {str(e)}")

    def _save_metric_reports(self, output_dir: str):
        """Save numerical reports to CSV"""
        try:
            if not self.metrics.empty:
                self.metrics.to_csv(os.path.join(output_dir, 'performance_metrics.csv'), index=False)
            if not self.risk_scores.empty:
                self.risk_scores.to_csv(os.path.join(output_dir, 'risk_scores.csv'), index=False)
            if not self.feature_importances.empty:
                self.feature_importances.to_csv(os.path.join(output_dir, 'feature_importances.csv'), index=False)
        except Exception as e:
            warnings.warn(f"Failed to save reports: {str(e)}")

    def _plot_performance_metrics(self, output_dir: str):
        """Visualize model performance metrics"""
        if not self.metrics.empty:
            try:
                plt.figure(figsize=(12, 8))
                sns.boxplot(data=self.metrics, x='horizon', y='roc_auc', palette='viridis')
                plt.title('Model Performance by Horizon')
                plt.xticks(rotation=45)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'performance_by_horizon.png'), bbox_inches='tight')
                plt.close()
            except Exception as e:
                warnings.warn(f"Performance metrics plot failed: {str(e)}")

    def _plot_feature_importances(self, output_dir: str):
        """Plot aggregated feature importances"""
        if not self.feature_importances.empty:
            try:
                top_features = (self.feature_importances
                                .groupby('feature')['importance']
                                .mean()
                                .nlargest(10))
                
                plt.figure(figsize=(10, 6))
                top_features.sort_values().plot(kind='barh', color=COLORS['neutral'])
                plt.title('Top 10 Most Important Features')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'feature_importances.png'), bbox_inches='tight')
                plt.close()
            except Exception as e:
                warnings.warn(f"Feature importance plot failed: {str(e)}")

    def _plot_risk_analysis(self, output_dir: str):
        """Visualize risk-reward relationship"""
        if not self.metrics.empty and not self.risk_scores.empty:
            try:
                merged = self.metrics.merge(self.risk_scores, on='ticker')
                
                plt.figure(figsize=(10, 6))
                sns.scatterplot(data=merged, x='composite_risk', y='roc_auc', 
                                hue='horizon', palette='viridis', s=100)
                plt.title('Risk-Reward Relationship')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'risk_reward.png'), bbox_inches='tight')
                plt.close()
            except Exception as e:
                warnings.warn(f"Risk analysis plot failed: {str(e)}")

    def _plot_monte_carlo_results(self, output_dir: str):
        """Visualize Monte Carlo simulation results"""
        for horizon, results in self.monte_carlo_results.items():
            try:
                if not results.empty:
                    plt.figure(figsize=(12, 6))
                    top_results = results.nlargest(10, 'mean_return')
                    sns.barplot(data=top_results, 
                                x='ticker', y='mean_return',
                                palette='viridis')
                    plt.title(f'{horizon.replace("_", " ").title()} Projections')
                    plt.xticks(rotation=45)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'monte_carlo_{horizon}.png'), 
                                bbox_inches='tight')
                    plt.close()
            except Exception as e:
                warnings.warn(f"Monte Carlo plot failed for {horizon}: {str(e)}")

    def get_top_performers(self, n: int = 10, metric: str = 'roc_auc'):
        """Get top performing tickers by metric"""
        if self.metrics.empty:
            return pd.DataFrame()
            
        return (self.metrics
                .groupby('ticker')
                [metric]
                .mean()
                .nlargest(n)
                .reset_index()
                .rename(columns={metric: 'score'}))

    def get_risk_adjusted_returns(self, n: int = 10):
        """Get best risk-adjusted returns"""
        if self.metrics.empty or self.risk_scores.empty:
            return pd.DataFrame()
            
        merged = self.metrics.merge(self.risk_scores, on='ticker')
        merged['risk_adj_return'] = merged['roc_auc'] / (merged['composite_risk'] + 1e-6)  # Add small epsilon
        return merged.nlargest(n, 'risk_adj_return')[['ticker', 'risk_adj_return']]

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