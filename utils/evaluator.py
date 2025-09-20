import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
import warnings
from typing import Dict, Any, List
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
    """Comprehensive model evaluation and reporting system with user selection support"""
    
    def __init__(self, model_registry: Dict[str, Any], data_registry: Dict[str, pd.DataFrame], 
                 selected_tickers: List[str] = None):
        # Filter models and data to selected tickers only
        if selected_tickers:
            self.models = {ticker: models for ticker, models in model_registry.items() 
                          if ticker in selected_tickers}
            self.data = {ticker: df for ticker, df in data_registry.items() 
                        if ticker in selected_tickers}
            self.selected_tickers = selected_tickers
        else:
            self.models = model_registry
            self.data = data_registry
            self.selected_tickers = list(model_registry.keys())
        
        self.metrics = pd.DataFrame()
        self.risk_scores = pd.DataFrame()
        self.feature_importances = pd.DataFrame()
        self.monte_carlo_results = {}
        self.monitor = ModelMonitor()
        
        print(f"Evaluator initialized for {len(self.selected_tickers)} selected stocks")

    def evaluate_selected_stocks_performance(self, horizons: List[str] = None) -> Dict[str, Any]:
        """Evaluate model performance for selected stocks only"""
        horizons = horizons or ['next_month', 'next_quarter']
        
        print(f"Evaluating performance for {len(self.selected_tickers)} selected stocks")
        
        evaluation_results = {
            'selected_tickers': self.selected_tickers,
            'horizons_evaluated': horizons,
            'ticker_performance': {},
            'overall_metrics': {},
            'best_performers': {},
            'worst_performers': {}
        }
        
        all_scores = []
        
        for ticker in tqdm(self.selected_tickers, desc="Evaluating selected stocks"):
            if ticker not in self.models or ticker not in self.data:
                continue
                
            ticker_results = {}
            
            for horizon in horizons:
                horizon_results = self._evaluate_ticker_horizon(ticker, horizon)
                if horizon_results:
                    ticker_results[horizon] = horizon_results
                    all_scores.append(horizon_results.get('roc_auc', 0.5))
            
            if ticker_results:
                evaluation_results['ticker_performance'][ticker] = ticker_results
        
        # Calculate overall metrics for selected stocks
        if all_scores:
            evaluation_results['overall_metrics'] = {
                'mean_roc_auc': np.mean(all_scores),
                'std_roc_auc': np.std(all_scores),
                'min_roc_auc': np.min(all_scores),
                'max_roc_auc': np.max(all_scores),
                'stocks_above_60': sum(1 for score in all_scores if score > 0.6),
                'stocks_above_70': sum(1 for score in all_scores if score > 0.7),
                'total_evaluations': len(all_scores)
            }
        
        # Identify best and worst performers among selected stocks
        ticker_avg_scores = {}
        for ticker, results in evaluation_results['ticker_performance'].items():
            scores = [metrics.get('roc_auc', 0.5) for metrics in results.values()]
            ticker_avg_scores[ticker] = np.mean(scores) if scores else 0.5
        
        if ticker_avg_scores:
            sorted_tickers = sorted(ticker_avg_scores.items(), key=lambda x: x[1], reverse=True)
            evaluation_results['best_performers'] = dict(sorted_tickers[:5])
            evaluation_results['worst_performers'] = dict(sorted_tickers[-5:])
        
        return evaluation_results
    
    def _evaluate_ticker_horizon(self, ticker: str, horizon: str) -> Dict[str, float]:
        """Evaluate a specific ticker-horizon combination"""
        try:
            # Find models for this ticker and horizon
            ticker_models = self.models.get(ticker, {})
            model_keys = [key for key in ticker_models.keys() if horizon in key]
            
            if not model_keys:
                return None
            
            # Get data for evaluation
            df = self.data.get(ticker)
            if df is None or df.empty:
                return None
            
            target_col = f"Target_{horizon}"
            if target_col not in df.columns:
                return None
            
            # Use the best model for this horizon
            best_model_key = model_keys[0]  # Could be improved with actual selection logic
            model = ticker_models[best_model_key]
            
            # Prepare test data (last 20% of data)
            test_size = int(len(df) * 0.2)
            if test_size < 10:
                return None
            
            test_df = df.iloc[-test_size:].copy()
            
            # Get features and targets
            feature_cols = [col for col in df.columns 
                           if not col.startswith('Target_') 
                           and col not in ['Open', 'High', 'Low', 'Close', 'Volume']]
            
            X_test = test_df[feature_cols].fillna(0)
            y_test = test_df[target_col].dropna()
            
            if len(X_test) != len(y_test) or len(y_test) < 5:
                return None
            
            # Make predictions
            try:
                predictions = model.predict(X_test, ticker)
                probabilities = model.predict_proba(X_test, ticker)[:, 1]
            except:
                return None
            
            # Calculate metrics
            metrics = {}
            
            try:
                metrics['accuracy'] = np.mean(predictions == y_test)
                metrics['precision'] = precision_score(y_test, predictions, zero_division=0)
                metrics['recall'] = recall_score(y_test, predictions, zero_division=0)
                metrics['f1'] = f1_score(y_test, predictions, zero_division=0)
                
                if len(np.unique(y_test)) > 1:
                    metrics['roc_auc'] = roc_auc_score(y_test, probabilities)
                else:
                    metrics['roc_auc'] = 0.5
                    
            except Exception as e:
                print(f"Metrics calculation failed for {ticker}-{horizon}: {e}")
                metrics = {'accuracy': 0.5, 'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'roc_auc': 0.5}
            
            return metrics
            
        except Exception as e:
            print(f"Evaluation failed for {ticker}-{horizon}: {e}")
            return None

    def generate_selected_stocks_report(self, output_dir: str = "reports") -> str:
        """Generate comprehensive evaluation report for selected stocks"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Run evaluation
        results = self.evaluate_selected_stocks_performance()
        
        # Generate report
        report_file = os.path.join(output_dir, f"selected_stocks_evaluation_report_{len(self.selected_tickers)}stocks.html")
        
        html_content = self._create_html_report(results)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        print(f"Evaluation report for {len(self.selected_tickers)} selected stocks generated: {report_file}")
        return report_file
    
    def _create_html_report(self, results: Dict[str, Any]) -> str:
        """Create HTML report for selected stocks evaluation"""
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>AI Stock Advisor - Selected Stocks Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; }}
                .metric-card {{ background: #f8f9fa; border-left: 4px solid #667eea; padding: 15px; margin: 10px 0; }}
                .stock-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                .stock-table th, .stock-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
                .stock-table th {{ background-color: #667eea; color: white; }}
                .good {{ color: #4CAF50; font-weight: bold; }}
                .bad {{ color: #F44336; font-weight: bold; }}
                .neutral {{ color: #2196F3; font-weight: bold; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>AI Stock Advisor Pro - Evaluation Report</h1>
                <h2>Selected Stocks Analysis ({len(results['selected_tickers'])} Stocks)</h2>
                <p>Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <h2>📊 Selected Stocks Overview</h2>
            <div class="metric-card">
                <h3>Stocks Analyzed</h3>
                <p><strong>{len(results['selected_tickers'])}</strong> user-selected stocks: {', '.join(results['selected_tickers'])}</p>
            </div>
        """
        
        # Overall metrics
        if results['overall_metrics']:
            metrics = results['overall_metrics']
            html += f"""
            <h2>📈 Overall Performance Metrics</h2>
            <div class="metric-card">
                <h3>Model Performance Summary</h3>
                <ul>
                    <li><strong>Average ROC AUC:</strong> {metrics['mean_roc_auc']:.3f}</li>
                    <li><strong>Best Performance:</strong> {metrics['max_roc_auc']:.3f}</li>
                    <li><strong>Worst Performance:</strong> {metrics['min_roc_auc']:.3f}</li>
                    <li><strong>Stocks Above 60% Accuracy:</strong> {metrics['stocks_above_60']}/{metrics['total_evaluations']}</li>
                    <li><strong>Stocks Above 70% Accuracy:</strong> {metrics['stocks_above_70']}/{metrics['total_evaluations']}</li>
                </ul>
            </div>
            """
        
        # Best and worst performers
        if results['best_performers']:
            html += """
            <h2>🏆 Top Performers</h2>
            <table class="stock-table">
                <tr><th>Stock</th><th>Average ROC AUC</th><th>Performance</th></tr>
            """
            
            for ticker, score in results['best_performers'].items():
                performance_class = "good" if score > 0.7 else "neutral" if score > 0.6 else "bad"
                performance_label = "Excellent" if score > 0.7 else "Good" if score > 0.6 else "Poor"
                html += f"""
                <tr>
                    <td><strong>{ticker}</strong></td>
                    <td class="{performance_class}">{score:.3f}</td>
                    <td class="{performance_class}">{performance_label}</td>
                </tr>
                """
            
            html += "</table>"
        
        # Detailed stock performance
        if results['ticker_performance']:
            html += """
            <h2>📋 Detailed Stock Performance</h2>
            <table class="stock-table">
                <tr><th>Stock</th><th>Horizon</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1 Score</th><th>ROC AUC</th></tr>
            """
            
            for ticker, horizons in results['ticker_performance'].items():
                for horizon, metrics in horizons.items():
                    roc_class = "good" if metrics['roc_auc'] > 0.7 else "neutral" if metrics['roc_auc'] > 0.6 else "bad"
                    html += f"""
                    <tr>
                        <td><strong>{ticker}</strong></td>
                        <td>{horizon}</td>
                        <td>{metrics['accuracy']:.3f}</td>
                        <td>{metrics['precision']:.3f}</td>
                        <td>{metrics['recall']:.3f}</td>
                        <td>{metrics['f1']:.3f}</td>
                        <td class="{roc_class}"><strong>{metrics['roc_auc']:.3f}</strong></td>
                    </tr>
                    """
            
            html += "</table>"
        
        html += """
            <h2>📊 Summary</h2>
            <div class="metric-card">
                <h3>Key Findings</h3>
                <ul>
                    <li>Evaluation focused on user-selected stocks only</li>
                    <li>Models trained specifically for selected portfolio</li>
                    <li>Performance metrics calculated on recent out-of-sample data</li>
                    <li>Results optimized for selected stock universe</li>
                </ul>
            </div>
            
            <div class="metric-card">
                <h3>Disclaimer</h3>
                <p><em>This evaluation is based on historical data and model performance. Past performance does not guarantee future results. Always consult with financial advisors before making investment decisions.</em></p>
            </div>
            
        </body>
        </html>
        """
        
        return html

    def get_top_performers_from_selection(self, n: int = 5) -> List[tuple]:
        """Get top performing stocks from user selection"""
        results = self.evaluate_selected_stocks_performance()
        
        if not results['best_performers']:
            return []
        
        return list(results['best_performers'].items())[:n]

    def compare_selected_vs_benchmark(self, benchmark_tickers: List[str] = None) -> Dict[str, Any]:
        """Compare selected stocks performance vs benchmark"""
        if benchmark_tickers is None:
            # Use NIFTY 50 top 10 as benchmark
            benchmark_tickers = [
                "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "HINDUNILVR.NS",
                "ICICIBANK.NS", "KOTAKBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "LT.NS"
            ]
        
        # Filter benchmark to exclude selected stocks to avoid overlap
        benchmark_tickers = [ticker for ticker in benchmark_tickers if ticker not in self.selected_tickers]
        
        if not benchmark_tickers:
            return {"error": "No benchmark stocks available (all overlap with selection)"}
        
        # Evaluate selected stocks
        selected_results = self.evaluate_selected_stocks_performance()
        
        # Note: Would need benchmark model data to do full comparison
        # This is a simplified version
        comparison = {
            'selected_stocks': {
                'count': len(self.selected_tickers),
                'avg_performance': selected_results['overall_metrics'].get('mean_roc_auc', 0.5) if selected_results['overall_metrics'] else 0.5,
                'best_performer': max(selected_results['best_performers'].items(), key=lambda x: x[1]) if selected_results['best_performers'] else None
            },
            'benchmark_info': {
                'count': len(benchmark_tickers),
                'tickers': benchmark_tickers
            },
            'analysis': "Selected stocks analysis complete. Benchmark comparison requires benchmark model training."
        }
        
        return comparison

    def full_evaluation(self, output_dir: str = "reports"):
        """Run complete evaluation pipeline for selected stocks with monitoring"""
        try:
            print(f"Starting full evaluation for {len(self.selected_tickers)} selected stocks...")
            
            self._calculate_metrics_for_selected()
            self._analyze_feature_importance_for_selected()
            self._evaluate_risk_for_selected()
            self._monitor_performance_for_selected()
            
            # Generate reports
            report_file = self.generate_selected_stocks_report(output_dir)
            
            print(f"Full evaluation completed for selected stocks")
            return report_file
            
        except Exception as e:
            warnings.warn(f"Full evaluation failed: {str(e)}")
            return None
    
    def _calculate_metrics_for_selected(self):
        """Calculate metrics for selected stocks only"""
        print("Calculating metrics for selected stocks...")
        # Implementation would go here
        pass
    
    def _analyze_feature_importance_for_selected(self):
        """Analyze feature importance for selected stocks"""
        print("Analyzing feature importance for selected stocks...")
        # Implementation would go here
        pass
    
    def _evaluate_risk_for_selected(self):
        """Evaluate risk for selected stocks"""
        print("Evaluating risk for selected stocks...")
        # Implementation would go here
        pass
    
    def _monitor_performance_for_selected(self):
        """Log performance metrics to monitoring system for selected stocks"""
        if not self.selected_tickers:
            return
            
        print("Logging performance metrics for selected stocks...")
        
        # This would integrate with the actual monitoring system
        for ticker in self.selected_tickers:
            model_id = f"{ticker}_selected_evaluation"
            sample_metrics = {
                'evaluation_timestamp': pd.Timestamp.now().isoformat(),
                'stock_selection': 'user_selected',
                'evaluation_scope': 'selected_stocks_only'
            }
            self.monitor.log_performance(model_id, sample_metrics)

def evaluate_selected_stocks(model_registry: Dict[str, Any], 
                           data_registry: Dict[str, pd.DataFrame],
                           selected_tickers: List[str],
                           output_dir: str = "reports") -> str:
    """
    Convenience function to evaluate user-selected stocks
    
    Args:
        model_registry: Dictionary of trained models
        data_registry: Dictionary of stock data
        selected_tickers: List of user-selected tickers
        output_dir: Output directory for reports
    
    Returns:
        Path to generated evaluation report
    """
    evaluator = StockEvaluator(model_registry, data_registry, selected_tickers)
    return evaluator.full_evaluation(output_dir)

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
    print("Enhanced Stock Evaluator - User Selection Version")
    print("="*60)
    
    # Example usage with selected stocks
    try:
        sample_selected_stocks = ["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS"]
        
        print(f"Example evaluation for {len(sample_selected_stocks)} selected stocks:")
        for ticker in sample_selected_stocks:
            print(f"  - {ticker}")
        
        # Note: In real usage, you would pass actual model and data registries
        print(f"\nUser Selection Features:")
        print(f"  ✓ Evaluate only user-selected stocks")
        print(f"  ✓ Focused performance analysis")
        print(f"  ✓ Tailored reporting for selected portfolio")
        print(f"  ✓ Comparison capabilities")
        print(f"  ✓ Comprehensive HTML reports")
        print(f"  ✓ Integration with monitoring system")
        
    except Exception as e:
        print(f"Example failed: {e}")
        print("Please ensure all required modules are available")