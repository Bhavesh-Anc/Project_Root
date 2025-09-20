# backtest_integration.py
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from utils.backtesting import (
    BacktestEngine, BacktestConfig, MLStrategy, 
    BacktestAnalyzer, BacktestDB, PerformanceMetrics
)

def create_backtesting_tab(models, featured_data, raw_data):
    """Create backtesting interface in Streamlit"""
    
    st.header("🔬 Advanced Backtesting & Strategy Validation")
    
    # Configuration section
    st.subheader("⚙️ Backtest Configuration")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_capital = st.number_input(
            "Initial Capital (₹)", 
            min_value=100000, 
            max_value=50000000, 
            value=1000000,
            step=100000
        )
        
        transaction_cost = st.slider(
            "Transaction Cost (%)", 
            min_value=0.01, 
            max_value=0.5, 
            value=0.1,
            step=0.01
        ) / 100
        
        slippage = st.slider(
            "Slippage (%)", 
            min_value=0.01, 
            max_value=0.2, 
            value=0.05,
            step=0.01
        ) / 100
    
    with col2:
        max_positions = st.slider("Max Positions", 5, 25, 10)
        
        rebalance_freq = st.selectbox(
            "Rebalancing Frequency",
            ["weekly", "monthly", "quarterly"],
            index=1
        )
        
        position_sizing = st.selectbox(
            "Position Sizing Method",
            ["equal_weight", "risk_parity", "kelly"],
            index=0
        )
    
    with col3:
        max_drawdown_limit = st.slider(
            "Max Drawdown Limit (%)", 
            min_value=5, 
            max_value=30, 
            value=15
        ) / 100
        
        investment_horizon = st.selectbox(
            "ML Model Horizon",
            ["next_week", "next_month", "next_quarter"],
            index=1
        )
        
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)", 
            min_value=3, 
            max_value=10, 
            value=6
        ) / 100
    
    # Date range selection
    st.subheader("📅 Backtest Period")
    
    # Get available date range from data
    if raw_data:
        all_dates = []
        for ticker_data in raw_data.values():
            all_dates.extend(ticker_data.index)
        
        min_date = min(all_dates)
        max_date = max(all_dates)
        
        col1, col2 = st.columns(2)
        
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=max(min_date, max_date - timedelta(days=730)),  # Default: 2 years
                min_value=min_date,
                max_value=max_date
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date - timedelta(days=30),  # Leave some buffer
                min_value=min_date,
                max_value=max_date
            )
        
        if start_date >= end_date:
            st.error("Start date must be before end date")
            return
    else:
        st.error("No data available for backtesting")
        return
    
    # Advanced settings
    with st.expander("🔧 Advanced Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_signal_strength = st.slider(
                "Minimum Signal Strength", 
                0.5, 0.9, 0.6, 0.05
            )
            profit_target = st.slider(
                "Profit Target (%)", 
                5, 30, 10
            ) / 100
        
        with col2:
            stop_loss = st.slider(
                "Stop Loss (%)", 
                2, 15, 5
            ) / 100
            max_holding_days = st.slider(
                "Max Holding Period (days)", 
                10, 90, 30
            )
    
    # Run backtest button
    if st.button("🚀 Run Backtest", type="primary"):
        
        if not models or not featured_data:
            st.error("No trained models available. Please train models first.")
            return
        
        # Create configuration
        config = BacktestConfig(
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost,
            slippage_pct=slippage,
            max_positions=max_positions,
            rebalance_frequency=rebalance_freq,
            position_sizing_method=position_sizing,
            max_drawdown_limit=max_drawdown_limit,
            risk_free_rate=risk_free_rate
        )
        
        # Create ML strategy with enhanced exit rules
        class EnhancedMLStrategy(MLStrategy):
            def __init__(self, models, featured_data, horizon, min_signal, profit_target, stop_loss, max_days):
                super().__init__(models, featured_data, horizon)
                self.min_signal_strength = min_signal
                self.profit_target = profit_target
                self.stop_loss = stop_loss
                self.max_holding_days = max_days
            
            def generate_signals(self, data, date):
                signals = super().generate_signals(data, date)
                # Filter by minimum signal strength
                return {k: v for k, v in signals.items() if v >= self.min_signal_strength}
            
            def should_exit(self, ticker, entry_date, current_date, current_price, entry_price):
                # Enhanced exit logic
                holding_period = (current_date - entry_date).days
                return_pct = (current_price - entry_price) / entry_price
                
                # Time-based exit
                if holding_period > self.max_holding_days:
                    return True
                
                # Profit target
                if return_pct > self.profit_target:
                    return True
                
                # Stop loss
                if return_pct < -self.stop_loss:
                    return True
                
                return False
        
        strategy = EnhancedMLStrategy(
            models, featured_data, investment_horizon,
            min_signal_strength, profit_target, stop_loss, max_holding_days
        )
        
        # Initialize and run backtest
        with st.spinner("Running backtest... This may take a few minutes."):
            try:
                engine = BacktestEngine(config)
                
                results = engine.run_backtest(
                    strategy=strategy,
                    data=raw_data,
                    start_date=datetime.combine(start_date, datetime.min.time()),
                    end_date=datetime.combine(end_date, datetime.min.time())
                )
                
                if 'error' in results:
                    st.error(f"Backtest failed: {results['error']}")
                    return
                
                # Display results
                display_backtest_results(results, config)
                
                # Save results
                db = BacktestDB()
                backtest_name = f"ML_Strategy_{datetime.now().strftime('%Y%m%d_%H%M')}"
                backtest_id = db.save_backtest(backtest_name, results)
                
                st.success(f"Backtest completed and saved with ID: {backtest_id}")
                
            except Exception as e:
                st.error(f"Backtest execution failed: {str(e)}")
                st.exception(e)
    
    # Historical backtests section
    st.subheader("📊 Historical Backtests")
    
    try:
        db = BacktestDB()
        historical_backtests = db.list_backtests()
        
        if not historical_backtests.empty:
            # Display table of historical backtests
            display_cols = ['name', 'initial_capital', 'final_value', 'total_return', 
                          'sharpe_ratio', 'max_drawdown', 'total_trades', 'win_rate']
            
            formatted_df = historical_backtests[display_cols].copy()
            
            # Format columns for display
            formatted_df['total_return'] = formatted_df['total_return'].apply(lambda x: f"{x:.2%}")
            formatted_df['sharpe_ratio'] = formatted_df['sharpe_ratio'].apply(lambda x: f"{x:.3f}")
            formatted_df['max_drawdown'] = formatted_df['max_drawdown'].apply(lambda x: f"{x:.2%}")
            formatted_df['win_rate'] = formatted_df['win_rate'].apply(lambda x: f"{x:.2%}")
            
            st.dataframe(formatted_df, use_container_width=True)
            
            # Load and compare specific backtests
            selected_backtests = st.multiselect(
                "Select backtests to compare:",
                options=historical_backtests['id'].tolist(),
                format_func=lambda x: f"ID {x}: {historical_backtests[historical_backtests['id']==x]['name'].iloc[0]}"
            )
            
            if selected_backtests and st.button("Compare Selected Backtests"):
                compare_backtests(selected_backtests, db)
                
        else:
            st.info("No historical backtests found. Run your first backtest above!")
            
    except Exception as e:
        st.warning(f"Could not load historical backtests: {e}")

def display_backtest_results(results, config):
    """Display comprehensive backtest results"""
    
    metrics = results['metrics']
    portfolio_df = results['portfolio_history']
    trades = results['trades']
    
    # Key metrics overview
    st.subheader("📈 Performance Summary")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Total Return", 
            f"{metrics['total_return']:.2%}",
            delta=f"vs {config.risk_free_rate:.1%} risk-free"
        )
    
    with col2:
        st.metric("Sharpe Ratio", f"{metrics['sharpe_ratio']:.3f}")
    
    with col3:
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2%}")
    
    with col4:
        st.metric("Total Trades", f"{metrics['total_trades']}")
    
    with col5:
        st.metric("Win Rate", f"{metrics['win_rate']:.1%}")
    
    # Performance charts
    st.subheader("📊 Performance Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Portfolio Value", "Drawdown", "Returns Distribution", "Trade Analysis"
    ])
    
    with tab1:
        # Portfolio value over time
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ))
        
        # Add benchmark if available
        initial_value = config.initial_capital
        benchmark_value = initial_value * (1 + config.risk_free_rate) ** (
            (portfolio_df.index - portfolio_df.index[0]).days / 365.25
        )
        
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=benchmark_value,
            mode='lines',
            name=f'Risk-Free Rate ({config.risk_free_rate:.1%})',
            line=dict(color='gray', dash='dash')
        ))
        
        fig.update_layout(
            title="Portfolio Value Over Time",
            xaxis_title="Date",
            yaxis_title="Portfolio Value (₹)",
            height=500
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        # Drawdown chart
        peak = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - peak) / peak * 100
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=portfolio_df.index,
            y=drawdown,
            fill='tonexty',
            mode='lines',
            name='Drawdown',
            line=dict(color='red'),
            fillcolor='rgba(255,0,0,0.3)'
        ))
        
        fig.add_hline(
            y=-config.max_drawdown_limit * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Max Drawdown Limit ({config.max_drawdown_limit:.1%})"
        )
        
        fig.update_layout(
            title="Portfolio Drawdown",
            xaxis_title="Date",
            yaxis_title="Drawdown (%)",
            height=400
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        # Returns distribution
        if 'returns' in results:
            returns = results['returns'] * 100  # Convert to percentage
            
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                name='Daily Returns',
                opacity=0.7
            ))
            
            fig.add_vline(
                x=returns.mean(),
                line_dash="dash",
                line_color="red",
                annotation_text=f"Mean: {returns.mean():.3f}%"
            )
            
            fig.update_layout(
                title="Daily Returns Distribution",
                xaxis_title="Daily Return (%)",
                yaxis_title="Frequency",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Risk metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("VaR (95%)", f"{metrics['var_95']:.2%}")
            with col2:
                st.metric("CVaR (95%)", f"{metrics['cvar_95']:.2%}")
            with col3:
                st.metric("Volatility", f"{metrics['volatility']:.2%}")
    
    with tab4:
        # Trade analysis
        if trades:
            trade_df = pd.DataFrame([{
                'Ticker': trade.ticker,
                'Entry Date': trade.entry_date,
                'Exit Date': trade.exit_date,
                'Holding Days': trade.holding_period,
                'Entry Price': trade.entry_price,
                'Exit Price': trade.exit_price,
                'Return %': trade.return_pct * 100,
                'P&L (₹)': trade.net_pnl,
                'Entry Signal': trade.entry_signal
            } for trade in trades])
            
            # Trade statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Avg Trade Return", f"{metrics.get('avg_trade_return', 0):.2%}")
            with col2:
                st.metric("Best Trade", f"₹{metrics.get('best_trade', 0):,.0f}")
            with col3:
                st.metric("Worst Trade", f"₹{metrics.get('worst_trade', 0):,.0f}")
            with col4:
                st.metric("Avg Duration", f"{metrics['avg_trade_duration']:.1f} days")
            
            # Trade table
            st.subheader("Individual Trades")
            
            # Format the dataframe for display
            display_df = trade_df.copy()
            display_df['Return %'] = display_df['Return %'].apply(lambda x: f"{x:.2f}%")
            display_df['P&L (₹)'] = display_df['P&L (₹)'].apply(lambda x: f"₹{x:,.0f}")
            display_df['Entry Price'] = display_df['Entry Price'].apply(lambda x: f"₹{x:.2f}")
            display_df['Exit Price'] = display_df['Exit Price'].apply(lambda x: f"₹{x:.2f}")
            
            st.dataframe(display_df, use_container_width=True)
            
            # Trade returns distribution
            fig = go.Figure()
            
            fig.add_trace(go.Histogram(
                x=trade_df['Return %'],
                nbinsx=30,
                name='Trade Returns',
                opacity=0.7
            ))
            
            fig.update_layout(
                title="Trade Returns Distribution",
                xaxis_title="Return (%)",
                yaxis_title="Number of Trades",
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.info("No trades executed during backtest period.")
    
    # Detailed performance report
    with st.expander("📋 Detailed Performance Report"):
        analyzer = BacktestAnalyzer()
        report = analyzer.create_performance_report(results)
        st.text(report)

def compare_backtests(backtest_ids, db):
    """Compare multiple backtests"""
    
    st.subheader("📊 Backtest Comparison")
    
    comparison_data = []
    portfolio_data = {}
    
    for backtest_id in backtest_ids:
        try:
            results = db.load_backtest(backtest_id)
            if results:
                metrics = results['metrics']
                portfolio_df = results['portfolio_history']
                
                comparison_data.append({
                    'Backtest ID': backtest_id,
                    'Total Return': f"{metrics['total_return']:.2%}",
                    'Annual Return': f"{metrics['annual_return']:.2%}",
                    'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                    'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                    'Win Rate': f"{metrics['win_rate']:.1%}",
                    'Total Trades': metrics['total_trades']
                })
                
                # Normalize portfolio values for comparison
                normalized_portfolio = portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0]
                portfolio_data[f"Backtest {backtest_id}"] = normalized_portfolio
                
        except Exception as e:
            st.error(f"Could not load backtest {backtest_id}: {e}")
    
    if comparison_data:
        # Comparison table
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Portfolio comparison chart
        if portfolio_data:
            fig = go.Figure()
            
            for name, portfolio_values in portfolio_data.items():
                fig.add_trace(go.Scatter(
                    x=portfolio_values.index,
                    y=portfolio_values,
                    mode='lines',
                    name=name,
                    line=dict(width=2)
                ))
            
            fig.update_layout(
                title="Normalized Portfolio Performance Comparison",
                xaxis_title="Date",
                yaxis_title="Normalized Portfolio Value",
                height=500
            )
            
            st.plotly_chart(fig, use_container_width=True)

# Add this to your main app.py tabs
def add_backtesting_to_main_app():
    """Instructions for adding backtesting to main app"""
    
    instructions = """
    # Adding Backtesting to Your Main App
    
    1. Add the backtesting import to your app.py:
    ```python
    from backtest_integration import create_backtesting_tab
    ```
    
    2. Add a new tab to your main tabs:
    ```python
    tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
        "📈 Enhanced Predictions", 
        "💰 Price Targets",
        "💼 Advanced Portfolio", 
        "📊 Model Analytics", 
        "🔍 Market Intelligence",
        "⚙️ System Dashboard",
        "🔬 Backtesting"  # New tab
    ])
    
    # In the backtesting tab:
    with tab7:
        create_backtesting_tab(models, featured_data, raw_data)
    ```
    
    3. The backtesting system will integrate with your existing ML models
       and provide comprehensive strategy validation.
    """
    
    return instructions

if __name__ == "__main__":
    print("Backtesting integration module loaded successfully!")
    print(add_backtesting_to_main_app())