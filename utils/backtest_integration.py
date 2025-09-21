# utils/backtest_integration.py - Enhanced with Comprehensive Risk Management
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import enhanced backtesting and risk management
from utils.backtesting import (
    EnhancedBacktestEngine, EnhancedBacktestConfig, MLStrategy, 
    BacktestAnalyzer, BacktestDB, PerformanceMetrics
)

from utils.risk_management import (
    ComprehensiveRiskManager, RiskConfig, CorrelationAnalyzer, 
    DrawdownTracker, PositionSizer, StressTester
)

def create_enhanced_backtesting_tab(models, featured_data, raw_data, selected_tickers):
    """Create enhanced backtesting interface with comprehensive risk management"""
    
    st.header("🔬 Enhanced Backtesting & Risk-Managed Strategy Validation")
    
    # Enhanced introduction
    st.markdown("""
    This enhanced backtesting framework integrates comprehensive risk management including:
    - **Real-time correlation monitoring** with automatic position adjustments
    - **Dynamic drawdown tracking** with configurable stop-loss mechanisms  
    - **Advanced position sizing** using Kelly Criterion, Risk Parity, and ERC methods
    - **Continuous stress testing** with historical and Monte Carlo scenarios
    - **Risk-based exit signals** beyond traditional strategy signals
    """)
    
    # Enhanced Configuration section
    st.subheader("⚙️ Enhanced Backtest Configuration")
    
    # Core backtesting parameters
    st.markdown("**📊 Core Backtesting Parameters**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        initial_capital = st.number_input(
            "Initial Capital (₹)", 
            min_value=100000, 
            max_value=50000000, 
            value=1000000,
            step=100000,
            help="Starting portfolio value"
        )
        
        transaction_cost = st.slider(
            "Transaction Cost (%)", 
            min_value=0.01, 
            max_value=0.5, 
            value=0.1,
            step=0.01,
            help="Cost per trade as percentage of trade value"
        ) / 100
        
        slippage = st.slider(
            "Slippage (%)", 
            min_value=0.01, 
            max_value=0.2, 
            value=0.05,
            step=0.01,
            help="Market impact cost"
        ) / 100
    
    with col2:
        max_positions = st.slider(
            "Max Positions", 
            5, 25, 10,
            help="Maximum number of concurrent positions"
        )
        
        rebalance_freq = st.selectbox(
            "Rebalancing Frequency",
            ["weekly", "monthly", "quarterly"],
            index=1,
            help="How often to rebalance the portfolio"
        )
        
        investment_horizon = st.selectbox(
            "ML Model Horizon",
            ["next_week", "next_month", "next_quarter"],
            index=1,
            help="Investment horizon for ML predictions"
        )
    
    with col3:
        risk_free_rate = st.slider(
            "Risk-Free Rate (%)", 
            min_value=3, 
            max_value=10, 
            value=6,
            help="Annual risk-free rate for Sharpe ratio calculation"
        ) / 100
        
        benchmark = st.selectbox(
            "Benchmark",
            ["NIFTY50", "SENSEX", "NIFTY100"],
            index=0,
            help="Benchmark for performance comparison"
        )

    # Risk Management Parameters
    st.markdown("**🛡️ Risk Management Parameters**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        max_drawdown_limit = st.slider(
            "Max Portfolio Drawdown (%)", 
            min_value=5, 
            max_value=30, 
            value=15,
            help="Maximum allowed portfolio drawdown before risk controls activate"
        ) / 100
        
        max_correlation = st.slider(
            "Max Position Correlation", 
            min_value=0.3, 
            max_value=0.9, 
            value=0.7,
            step=0.05,
            help="Maximum allowed correlation between any two positions"
        )
        
        var_confidence = st.slider(
            "VaR Confidence Level (%)", 
            min_value=90, 
            max_value=99, 
            value=95,
            help="Confidence level for Value at Risk calculations"
        ) / 100
    
    with col2:
        position_sizing = st.selectbox(
            "Position Sizing Method",
            ["risk_parity", "kelly_criterion", "equal_weight", "erc"],
            index=0,
            help="Method for determining position sizes"
        )
        
        kelly_cap = st.slider(
            "Kelly Fraction Cap (%)", 
            min_value=10, 
            max_value=50, 
            value=25,
            help="Maximum Kelly fraction to prevent over-leveraging"
        ) / 100
        
        risk_budget_limit = st.slider(
            "Risk Budget Limit (%)", 
            min_value=10, 
            max_value=50, 
            value=20,
            help="Maximum percentage of risk budget to utilize"
        ) / 100
    
    with col3:
        stress_test_frequency = st.number_input(
            "Stress Test Frequency (days)", 
            min_value=1, 
            max_value=30, 
            value=5,
            help="How often to run stress tests during backtesting"
        )
        
        rebalance_threshold = st.slider(
            "Rebalance Threshold (%)", 
            min_value=1, 
            max_value=10, 
            value=5,
            help="Portfolio drift threshold that triggers rebalancing"
        ) / 100
    
    # Advanced Risk Settings
    with st.expander("🔧 Advanced Risk Management Settings"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Monitoring Options**")
            risk_monitoring = st.checkbox("Enable Real-time Risk Monitoring", True)
            correlation_monitoring = st.checkbox("Correlation Monitoring", True)
            drawdown_monitoring = st.checkbox("Drawdown Monitoring", True)
        
        with col2:
            st.markdown("**Stress Testing Options**")
            stress_testing = st.checkbox("Periodic Stress Testing", True)
            monte_carlo_scenarios = st.number_input("Monte Carlo Scenarios", 100, 5000, 1000)
            historical_scenarios = st.checkbox("Historical Scenario Testing", True)
        
        with col3:
            st.markdown("**Exit Signal Options**")
            risk_based_exits = st.checkbox("Risk-based Exit Signals", True)
            volatility_exits = st.checkbox("Volatility-based Exits", True)
            correlation_exits = st.checkbox("Correlation-based Exits", True)
    
    # Enhanced Strategy Settings
    with st.expander("🎯 Enhanced Strategy Settings"):
        col1, col2 = st.columns(2)
        
        with col1:
            min_signal_strength = st.slider(
                "Minimum Signal Strength", 
                0.5, 0.9, 0.6, 0.05,
                help="Minimum ML signal strength to enter positions"
            )
            
            profit_target = st.slider(
                "Profit Target (%)", 
                5, 30, 10,
                help="Automatic profit-taking threshold"
            ) / 100
            
            stop_loss = st.slider(
                "Stop Loss (%)", 
                2, 15, 5,
                help="Automatic stop-loss threshold"
            ) / 100
        
        with col2:
            max_holding_days = st.slider(
                "Max Holding Period (days)", 
                10, 90, 30,
                help="Maximum time to hold any position"
            )
            
            position_concentration_limit = st.slider(
                "Max Single Position (%)",
                5, 25, 15,
                help="Maximum percentage in any single position"
            ) / 100
            
            sector_concentration_limit = st.slider(
                "Max Sector Concentration (%)",
                20, 60, 40,
                help="Maximum percentage in any single sector"
            ) / 100
    
    # Date range selection
    st.subheader("📅 Backtest Period")
    
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
                max_value=max_date,
                help="Backtest start date"
            )
        
        with col2:
            end_date = st.date_input(
                "End Date",
                value=max_date - timedelta(days=30),  # Leave some buffer
                min_value=min_date,
                max_value=max_date,
                help="Backtest end date"
            )
        
        # Date validation
        if start_date >= end_date:
            st.error("❌ Start date must be before end date")
            return
        
        # Data sufficiency check
        total_days = (end_date - start_date).days
        if total_days < 90:
            st.warning("⚠️ Short backtest period may not provide reliable results. Consider extending to at least 3 months.")
        
        st.info(f"ℹ️ Backtest period: {total_days} days ({total_days/365:.1f} years)")
        
    else:
        st.error("❌ No data available for backtesting")
        return
    
    # Pre-flight checks
    st.subheader("✅ Pre-flight Checks")
    
    checks_passed = True
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Data availability checks
        if models and featured_data:
            st.success("✅ Models and features available")
        else:
            st.error("❌ Models or features missing")
            checks_passed = False
        
        if selected_tickers:
            st.success(f"✅ {len(selected_tickers)} stocks selected")
        else:
            st.error("❌ No stocks selected")
            checks_passed = False
        
        if raw_data:
            total_data_points = sum(len(df) for df in raw_data.values())
            st.success(f"✅ {total_data_points:,} data points available")
        else:
            st.error("❌ No historical data available")
            checks_passed = False
    
    with col2:
        # Configuration checks
        if max_drawdown_limit > 0 and max_drawdown_limit < 0.5:
            st.success("✅ Reasonable drawdown limit")
        else:
            st.warning("⚠️ Check drawdown limit setting")
        
        if max_correlation > 0 and max_correlation < 1:
            st.success("✅ Valid correlation threshold")
        else:
            st.warning("⚠️ Check correlation threshold")
        
        if initial_capital >= 100000:
            st.success("✅ Sufficient initial capital")
        else:
            st.warning("⚠️ Low initial capital may affect results")
    
    # Run enhanced backtest button
    if st.button("🚀 Run Enhanced Backtest with Risk Management", 
                type="primary", 
                disabled=not checks_passed,
                help="Execute comprehensive backtest with risk management"):
        
        if not checks_passed:
            st.error("❌ Please address the issues above before running backtest")
            return
        
        # Create enhanced configuration
        config = EnhancedBacktestConfig(
            initial_capital=initial_capital,
            transaction_cost_pct=transaction_cost,
            slippage_pct=slippage,
            max_positions=max_positions,
            rebalance_frequency=rebalance_freq,
            position_sizing_method=position_sizing,
            max_portfolio_drawdown=max_drawdown_limit,
            max_position_correlation=max_correlation,
            risk_free_rate=risk_free_rate,
            var_confidence_level=var_confidence,
            kelly_fraction_cap=kelly_cap,
            stress_test_frequency=stress_test_frequency,
            risk_budget_limit=risk_budget_limit,
            rebalance_threshold=rebalance_threshold,
            risk_monitoring_enabled=risk_monitoring,
            correlation_monitoring=correlation_monitoring,
            stress_testing_enabled=stress_testing
        )
        
        # Enhanced ML strategy with risk management
        class EnhancedMLStrategyWithRisk(MLStrategy):
            def __init__(self, models, featured_data, horizon, enhanced_params):
                super().__init__(models, featured_data, horizon)
                self.min_signal_strength = enhanced_params['min_signal_strength']
                self.profit_target = enhanced_params['profit_target']
                self.stop_loss = enhanced_params['stop_loss']
                self.max_holding_days = enhanced_params['max_holding_days']
                self.risk_based_exits = enhanced_params['risk_based_exits']
                self.volatility_exits = enhanced_params['volatility_exits']
                self.correlation_exits = enhanced_params['correlation_exits']
            
            def generate_signals(self, data, date):
                signals = super().generate_signals(data, date)
                # Enhanced filtering by minimum signal strength
                return {k: v for k, v in signals.items() if v >= self.min_signal_strength}
            
            def should_exit(self, ticker, entry_date, current_date, current_price, entry_price):
                # Enhanced exit logic with multiple criteria
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
                
                # Additional risk-based exits can be implemented here
                # based on volatility, correlation, etc.
                
                return False
        
        enhanced_params = {
            'min_signal_strength': min_signal_strength,
            'profit_target': profit_target,
            'stop_loss': stop_loss,
            'max_holding_days': max_holding_days,
            'risk_based_exits': risk_based_exits,
            'volatility_exits': volatility_exits,
            'correlation_exits': correlation_exits
        }
        
        strategy = EnhancedMLStrategyWithRisk(
            models, featured_data, investment_horizon, enhanced_params
        )
        
        # Initialize and run enhanced backtest
        progress_container = st.container()
        status_container = st.container()
        
        with progress_container:
            progress_bar = st.progress(0)
            status_text = st.empty()
        
        with status_container:
            with st.spinner("Initializing enhanced backtesting engine with risk management..."):
                try:
                    engine = EnhancedBacktestEngine(config)
                    
                    status_text.text("🔄 Running enhanced backtest with comprehensive risk management...")
                    progress_bar.progress(0.1)
                    
                    # Run the enhanced backtest
                    results = engine.run_enhanced_backtest(
                        strategy=strategy,
                        data=raw_data,
                        start_date=datetime.combine(start_date, datetime.min.time()),
                        end_date=datetime.combine(end_date, datetime.min.time())
                    )
                    
                    progress_bar.progress(0.8)
                    status_text.text("📊 Processing results and generating reports...")
                    
                    if 'error' in results:
                        st.error(f"❌ Enhanced backtest failed: {results['error']}")
                        return
                    
                    progress_bar.progress(0.9)
                    status_text.text("💾 Saving results to database...")
                    
                    # Save results to database
                    db = BacktestDB()
                    backtest_name = f"Enhanced_Risk_Managed_{len(selected_tickers)}stocks_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    backtest_id = db.save_backtest(backtest_name, results)
                    
                    progress_bar.progress(1.0)
                    status_text.text("✅ Enhanced backtest completed successfully!")
                    
                    # Display enhanced results
                    display_enhanced_backtest_results(results, config, enhanced_params)
                    
                    st.success(f"🎉 Enhanced backtest completed successfully and saved with ID: {backtest_id}")
                    
                    # Provide download option for detailed results
                    if st.button("📥 Download Detailed Results"):
                        generate_downloadable_report(results, config, backtest_name)
                    
                except Exception as e:
                    progress_bar.progress(0)
                    status_text.text("❌ Enhanced backtest failed")
                    st.error(f"Enhanced backtest execution failed: {str(e)}")
                    
                    with st.expander("🔍 Error Details"):
                        st.exception(e)
    
    # Historical backtests section with enhanced features
    create_enhanced_historical_backtests_section()

def display_enhanced_backtest_results(results, config, enhanced_params):
    """Display comprehensive enhanced backtest results"""
    
    enhanced_metrics = results['enhanced_metrics']
    portfolio_df = results['portfolio_history']
    enhanced_trades = results['enhanced_trades']
    risk_events = results['risk_events']
    risk_analysis = results['risk_analysis']
    
    # Executive Summary
    st.subheader("📋 Executive Summary")
    
    summary_col1, summary_col2 = st.columns(2)
    
    with summary_col1:
        st.markdown("**🎯 Performance Highlights**")
        total_return = enhanced_metrics['total_return']
        annual_return = enhanced_metrics['annual_return'] 
        sharpe_ratio = enhanced_metrics['sharpe_ratio']
        max_drawdown = enhanced_metrics['max_drawdown']
        
        performance_grade = "A+" if sharpe_ratio > 2 and max_drawdown > -0.1 else \
                           "A" if sharpe_ratio > 1.5 and max_drawdown > -0.15 else \
                           "B+" if sharpe_ratio > 1 and max_drawdown > -0.2 else \
                           "B" if sharpe_ratio > 0.5 else "C"
        
        st.markdown(f"""
        - **Performance Grade**: {performance_grade}
        - **Total Return**: {total_return:.2%}
        - **Annualized Return**: {annual_return:.2%}
        - **Risk-Adjusted Return (Sharpe)**: {sharpe_ratio:.2f}
        - **Maximum Drawdown**: {max_drawdown:.2%}
        """)
    
    with summary_col2:
        st.markdown("**🛡️ Risk Management Summary**")
        
        risk_score = "Excellent" if risk_analysis.get('drawdown_violations', 0) == 0 and risk_analysis.get('stress_test_failures', 0) == 0 else \
                    "Good" if risk_analysis.get('drawdown_violations', 0) <= 1 else \
                    "Fair" if risk_analysis.get('drawdown_violations', 0) <= 3 else "Poor"
        
        st.markdown(f"""
        - **Risk Management Score**: {risk_score}
        - **Risk Events**: {enhanced_metrics.get('risk_events_count', 0)}
        - **Drawdown Violations**: {risk_analysis.get('drawdown_violations', 0)}
        - **Risk-Adjusted Trades**: {risk_analysis.get('risk_adjusted_trades', 0)}
        - **Max Correlation Reached**: {risk_analysis.get('avg_position_correlation', 0):.2f}
        """)
    
    # Enhanced Performance Metrics Dashboard
    st.subheader("📈 Enhanced Performance Dashboard")
    
    # Key metrics in prominent display
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    
    with col1:
        st.metric(
            "Total Return", 
            f"{enhanced_metrics['total_return']:.2%}",
            delta=f"vs {config.risk_free_rate:.1%} risk-free"
        )
    
    with col2:
        st.metric(
            "Annual Return", 
            f"{enhanced_metrics['annual_return']:.2%}",
            delta=f"{(enhanced_metrics['annual_return'] - config.risk_free_rate):.1%} excess"
        )
    
    with col3:
        st.metric("Sharpe Ratio", f"{enhanced_metrics['sharpe_ratio']:.3f}")
    
    with col4:
        st.metric("Max Drawdown", f"{enhanced_metrics['max_drawdown']:.2%}")
    
    with col5:
        st.metric("Total Trades", f"{len(enhanced_trades)}")
    
    with col6:
        if 'var_adjusted_return' in enhanced_metrics:
            st.metric("VaR-Adjusted Return", f"{enhanced_metrics['var_adjusted_return']:.3f}")
        else:
            st.metric("Sortino Ratio", f"{enhanced_metrics['sortino_ratio']:.3f}")
    
    # Risk Management Performance Dashboard
    st.subheader("🛡️ Risk Management Performance")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Risk Events", f"{enhanced_metrics.get('risk_events_count', 0)}")
    
    with col2:
        st.metric("Drawdown Violations", f"{risk_analysis.get('drawdown_violations', 0)}")
    
    with col3:
        st.metric("Correlation Violations", f"{risk_analysis.get('correlation_violations', 0)}")
    
    with col4:
        st.metric("Stress Test Failures", f"{risk_analysis.get('stress_test_failures', 0)}")
    
    with col5:
        risk_adjusted_pct = (risk_analysis.get('risk_adjusted_trades', 0) / len(enhanced_trades) * 100) if enhanced_trades else 0
        st.metric("Risk-Based Exits", f"{risk_adjusted_pct:.1f}%")
    
    # Comprehensive Analysis Tabs
    st.subheader("📊 Comprehensive Analysis")
    
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Portfolio Performance", "Risk Analytics", "Trade Analysis", 
        "Risk Events", "Comparison Analysis", "Detailed Report"
    ])
    
    with tab1:
        create_portfolio_performance_tab(portfolio_df, enhanced_metrics, config)
    
    with tab2:
        create_risk_analytics_tab(portfolio_df, risk_analysis, config)
    
    with tab3:
        create_enhanced_trade_analysis_tab(enhanced_trades, enhanced_params)
    
    with tab4:
        create_risk_events_tab(risk_events)
    
    with tab5:
        create_comparison_analysis_tab(enhanced_metrics, config)
    
    with tab6:
        create_detailed_report_tab(results, config, enhanced_params)

def create_portfolio_performance_tab(portfolio_df, enhanced_metrics, config):
    """Create comprehensive portfolio performance analysis"""
    
    # Multi-subplot performance chart
    fig = make_subplots(
        rows=3, cols=2,
        subplot_titles=[
            "Portfolio Value Over Time", "Cumulative Returns vs Benchmark",
            "Rolling Sharpe Ratio (60-day)", "Rolling Volatility (30-day)",
            "Underwater Curve (Drawdown Periods)", "Position Count Over Time"
        ],
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.1
    )
    
    # Portfolio value
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=portfolio_df['portfolio_value'],
            mode='lines',
            name='Portfolio Value',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Cumulative returns
    returns = portfolio_df['portfolio_value'].pct_change().fillna(0)
    cumulative_returns = (1 + returns).cumprod()
    
    # Benchmark comparison (simplified)
    benchmark_return = (1 + config.risk_free_rate / 252) ** np.arange(len(portfolio_df))
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=cumulative_returns,
            mode='lines',
            name='Portfolio',
            line=dict(color='blue', width=2)
        ),
        row=1, col=2
    )
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=benchmark_return,
            mode='lines',
            name='Risk-Free Rate',
            line=dict(color='gray', dash='dash')
        ),
        row=1, col=2
    )
    
    # Rolling Sharpe Ratio
    if len(returns) > 60:
        rolling_sharpe = []
        for i in range(60, len(returns)):
            window_returns = returns.iloc[i-60:i]
            excess_returns = window_returns - config.risk_free_rate / 252
            if window_returns.std() > 0:
                sharpe = excess_returns.mean() / window_returns.std() * np.sqrt(252)
            else:
                sharpe = 0
            rolling_sharpe.append(sharpe)
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index[60:],
                y=rolling_sharpe,
                mode='lines',
                name='Rolling Sharpe',
                line=dict(color='green')
            ),
            row=2, col=1
        )
    
    # Rolling Volatility
    if len(returns) > 30:
        rolling_vol = returns.rolling(30).std() * np.sqrt(252)
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=rolling_vol,
                mode='lines',
                name='Rolling Volatility',
                line=dict(color='orange')
            ),
            row=2, col=2
        )
    
    # Underwater curve (drawdown)
    peak = portfolio_df['portfolio_value'].cummax()
    drawdown = (portfolio_df['portfolio_value'] - peak) / peak
    
    fig.add_trace(
        go.Scatter(
            x=portfolio_df.index,
            y=drawdown,
            mode='lines',
            fill='tonexty',
            name='Drawdown',
            line=dict(color='red'),
            fillcolor='rgba(255,0,0,0.3)'
        ),
        row=3, col=1
    )
    
    # Position count over time
    if 'positions' in portfolio_df.columns:
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['positions'],
                mode='lines+markers',
                name='Position Count',
                line=dict(color='purple')
            ),
            row=3, col=2
        )
    
    fig.update_layout(height=900, title="Comprehensive Portfolio Performance Analysis")
    st.plotly_chart(fig, use_container_width=True)
    
    # Performance statistics table
    st.subheader("📊 Performance Statistics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        stats_df = pd.DataFrame({
            'Metric': ['Total Return', 'Annualized Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio'],
            'Value': [
                f"{enhanced_metrics['total_return']:.2%}",
                f"{enhanced_metrics['annual_return']:.2%}",
                f"{enhanced_metrics['volatility']:.2%}",
                f"{enhanced_metrics['sharpe_ratio']:.3f}",
                f"{enhanced_metrics['sortino_ratio']:.3f}"
            ]
        })
        st.dataframe(stats_df, use_container_width=True)
    
    with col2:
        risk_stats_df = pd.DataFrame({
            'Risk Metric': ['Maximum Drawdown', 'VaR (95%)', 'CVaR (95%)', 'Calmar Ratio', 'Max Daily Loss'],
            'Value': [
                f"{enhanced_metrics['max_drawdown']:.2%}",
                f"{enhanced_metrics['var_95']:.2%}",
                f"{enhanced_metrics['cvar_95']:.2%}",
                f"{enhanced_metrics['calmar_ratio']:.3f}",
                f"{returns.min():.2%}" if len(returns) > 0 else "N/A"
            ]
        })
        st.dataframe(risk_stats_df, use_container_width=True)

def create_risk_analytics_tab(portfolio_df, risk_analysis, config):
    """Create comprehensive risk analytics dashboard"""
    
    st.markdown("### 🛡️ Risk Management Effectiveness")
    
    # Risk metrics over time
    if 'current_var' in portfolio_df.columns and 'portfolio_correlation' in portfolio_df.columns:
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                "Value at Risk Over Time",
                "Portfolio Correlation",
                "Risk Budget Utilization", 
                "Drawdown vs Limits"
            ]
        )
        
        # VaR over time
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['current_var'] * 100,
                mode='lines',
                name='VaR (95%)',
                line=dict(color='red')
            ),
            row=1, col=1
        )
        
        # Portfolio correlation
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=portfolio_df['portfolio_correlation'],
                mode='lines',
                name='Max Correlation',
                line=dict(color='orange')
            ),
            row=1, col=2
        )
        
        fig.add_hline(
            y=config.max_position_correlation,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Limit ({config.max_position_correlation:.2f})",
            row=1, col=2
        )
        
        # Risk budget utilization
        if 'risk_budget_used' in portfolio_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=portfolio_df.index,
                    y=portfolio_df['risk_budget_used'] * 100,
                    mode='lines',
                    name='Risk Budget Used',
                    line=dict(color='blue')
                ),
                row=2, col=1
            )
        
        # Drawdown vs limits
        peak = portfolio_df['portfolio_value'].cummax()
        drawdown = (portfolio_df['portfolio_value'] - peak) / peak * 100
        
        fig.add_trace(
            go.Scatter(
                x=portfolio_df.index,
                y=drawdown,
                mode='lines',
                name='Drawdown',
                line=dict(color='red')
            ),
            row=2, col=2
        )
        
        fig.add_hline(
            y=-config.max_portfolio_drawdown * 100,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Limit ({config.max_portfolio_drawdown:.1%})",
            row=2, col=2
        )
        
        fig.update_layout(height=600, title="Risk Analytics Dashboard")
        st.plotly_chart(fig, use_container_width=True)
    
    # Risk management scorecard
    st.subheader("📊 Risk Management Scorecard")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        drawdown_score = 100 if risk_analysis.get('drawdown_violations', 0) == 0 else max(0, 100 - risk_analysis.get('drawdown_violations', 0) * 20)
        st.metric("Drawdown Management", f"{drawdown_score}/100")
        st.progress(drawdown_score / 100)
    
    with col2:
        correlation_score = 100 if risk_analysis.get('correlation_violations', 0) == 0 else max(0, 100 - risk_analysis.get('correlation_violations', 0) * 15)
        st.metric("Correlation Management", f"{correlation_score}/100")
        st.progress(correlation_score / 100)
    
    with col3:
        stress_score = 100 if risk_analysis.get('stress_test_failures', 0) == 0 else max(0, 100 - risk_analysis.get('stress_test_failures', 0) * 25)
        st.metric("Stress Test Performance", f"{stress_score}/100")
        st.progress(stress_score / 100)

def create_enhanced_trade_analysis_tab(enhanced_trades, enhanced_params):
    """Create enhanced trade analysis with risk insights"""
    
    if not enhanced_trades:
        st.warning("No trades executed during backtest period")
        return
    
    # Trade statistics
    st.subheader("📈 Trade Performance Analysis")
    
    # Convert trades to DataFrame for analysis
    trade_data = []
    for trade in enhanced_trades:
        trade_data.append({
            'Ticker': trade.ticker,
            'Entry Date': trade.entry_date,
            'Exit Date': trade.exit_date,
            'Holding Days': trade.holding_period,
            'Return %': trade.return_pct * 100,
            'P&L (₹)': trade.net_pnl,
            'Exit Reason': trade.exit_signal,
            'Entry Signal Strength': 'N/A'  # Could be enhanced
        })
    
    trade_df = pd.DataFrame(trade_data)
    
    # Enhanced trade metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        avg_return = trade_df['Return %'].mean()
        st.metric("Avg Trade Return", f"{avg_return:.2%}")
    
    with col2:
        win_rate = (trade_df['Return %'] > 0).mean()
        st.metric("Win Rate", f"{win_rate:.1%}")
    
    with col3:
        avg_holding = trade_df['Holding Days'].mean()
        st.metric("Avg Holding Period", f"{avg_holding:.1f} days")
    
    with col4:
        best_trade = trade_df['P&L (₹)'].max()
        st.metric("Best Trade", f"₹{best_trade:,.0f}")
    
    with col5:
        worst_trade = trade_df['P&L (₹)'].min()
        st.metric("Worst Trade", f"₹{worst_trade:,.0f}")
    
    # Risk-based exit analysis
    st.subheader("🛡️ Risk-Based Exit Analysis")
    
    exit_reasons = trade_df['Exit Reason'].value_counts()
    risk_exits = sum(1 for reason in exit_reasons.index if 'risk' in reason.lower())
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Exit reasons pie chart
        fig_exits = px.pie(
            values=exit_reasons.values,
            names=exit_reasons.index,
            title="Exit Reasons Distribution"
        )
        st.plotly_chart(fig_exits, use_container_width=True)
    
    with col2:
        # Trade returns distribution
        fig_returns = px.histogram(
            trade_df,
            x='Return %',
            nbins=20,
            title="Trade Returns Distribution",
            labels={'Return %': 'Return (%)', 'count': 'Frequency'}
        )
        fig_returns.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_returns, use_container_width=True)
    
    # Detailed trade table
    st.subheader("📋 Detailed Trade Log")
    
    # Format the trade table for display
    display_df = trade_df.copy()
    display_df['Return %'] = display_df['Return %'].apply(lambda x: f"{x:.2f}%")
    display_df['P&L (₹)'] = display_df['P&L (₹)'].apply(lambda x: f"₹{x:,.0f}")
    display_df['Entry Date'] = display_df['Entry Date'].dt.strftime('%Y-%m-%d')
    display_df['Exit Date'] = display_df['Exit Date'].dt.strftime('%Y-%m-%d')
    
    # Color coding for returns
    def color_returns(val):
        if 'Return %' in val.name:
            return ['background-color: lightgreen' if '+' in v else 'background-color: lightcoral' if '-' in v else '' for v in val]
        return [''] * len(val)
    
    # Apply conditional formatting
    styled_df = display_df.style.apply(color_returns, axis=0)
    st.dataframe(styled_df, use_container_width=True)

def create_risk_events_tab(risk_events):
    """Create risk events analysis tab"""
    
    st.subheader("🚨 Risk Events Timeline")
    
    if not risk_events:
        st.success("✅ No risk events occurred during the backtest period")
        st.markdown("""
        This indicates excellent risk management performance:
        - No drawdown limit breaches
        - No correlation threshold violations  
        - No stress test failures
        - Effective risk controls throughout the period
        """)
        return
    
    # Risk events summary
    col1, col2, col3 = st.columns(3)
    
    violation_events = [e for e in risk_events if 'violations' in e]
    stress_events = [e for e in risk_events if e.get('type') == 'stress_test_failure']
    
    with col1:
        st.metric("Total Risk Events", len(risk_events))
    with col2:
        st.metric("Violation Events", len(violation_events))
    with col3:
        st.metric("Stress Test Failures", len(stress_events))
    
    # Risk events timeline
    for i, event in enumerate(risk_events):
        with st.container():
            event_date = event['date'].strftime('%Y-%m-%d %H:%M')
            
            if 'violations' in event:
                st.error(f"**🚨 Risk Violation #{i+1}** - {event_date}")
                st.markdown(f"**Violations**: {', '.join(event['violations'])}")
                
            elif event.get('type') == 'stress_test_failure':
                st.warning(f"**⚠️ Stress Test Failure #{i+1}** - {event_date}")
                var_95 = event.get('var_95', 0)
                st.markdown(f"**VaR (95%)**: {var_95:.2%}")
                
            else:
                st.info(f"**ℹ️ Risk Event #{i+1}** - {event_date}")
            
            # Actions taken
            if event.get('actions_taken'):
                st.markdown("**Actions Taken**:")
                for action in event['actions_taken']:
                    st.markdown(f"- {action}")
            
            st.markdown("---")

def create_comparison_analysis_tab(enhanced_metrics, config):
    """Create benchmark and strategy comparison analysis"""
    
    st.subheader("📊 Performance Comparison")
    
    # Benchmark comparison
    risk_free_annual = config.risk_free_rate
    strategy_annual = enhanced_metrics['annual_return']
    
    # Create comparison chart
    comparison_data = pd.DataFrame({
        'Strategy': ['Risk-Free Rate', 'Our Strategy'],
        'Annual Return': [risk_free_annual, strategy_annual],
        'Volatility': [0.01, enhanced_metrics['volatility']],  # Assume minimal vol for risk-free
        'Sharpe Ratio': [0, enhanced_metrics['sharpe_ratio']]
    })
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Return comparison
        fig_returns = px.bar(
            comparison_data,
            x='Strategy',
            y='Annual Return',
            title="Annual Return Comparison",
            color='Annual Return',
            color_continuous_scale='RdYlGn'
        )
        fig_returns.update_layout(showlegend=False)
        st.plotly_chart(fig_returns, use_container_width=True)
    
    with col2:
        # Risk-return scatter
        fig_scatter = px.scatter(
            comparison_data,
            x='Volatility',
            y='Annual Return',
            color='Strategy',
            size='Sharpe Ratio',
            title="Risk-Return Profile",
            hover_data=['Sharpe Ratio']
        )
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Performance attribution
    st.subheader("🎯 Performance Attribution")
    
    excess_return = strategy_annual - risk_free_annual
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Excess Return", f"{excess_return:.2%}")
        st.caption("Return above risk-free rate")
    
    with col2:
        information_ratio = excess_return / enhanced_metrics['volatility'] if enhanced_metrics['volatility'] > 0 else 0
        st.metric("Information Ratio", f"{information_ratio:.3f}")
        st.caption("Excess return per unit of risk")
    
    with col3:
        risk_adjusted_excess = enhanced_metrics['sharpe_ratio'] * enhanced_metrics['volatility']
        st.metric("Risk-Adjusted Excess", f"{risk_adjusted_excess:.2%}")
        st.caption("Sharpe-adjusted excess return")

def create_detailed_report_tab(results, config, enhanced_params):
    """Create comprehensive detailed report"""
    
    st.subheader("📋 Comprehensive Backtest Report")
    
    enhanced_metrics = results['enhanced_metrics']
    risk_analysis = results['risk_analysis']
    enhanced_trades = results['enhanced_trades']
    risk_events = results['risk_events']
    
    # Executive summary
    st.markdown("### 📈 Executive Summary")
    
    st.markdown(f"""
    **Backtest Period**: {config.initial_capital:,.0f} initial capital
    
    **Performance Summary**:
    - Total Return: {enhanced_metrics['total_return']:.2%}
    - Annualized Return: {enhanced_metrics['annual_return']:.2%}
    - Volatility: {enhanced_metrics['volatility']:.2%}
    - Sharpe Ratio: {enhanced_metrics['sharpe_ratio']:.3f}
    - Maximum Drawdown: {enhanced_metrics['max_drawdown']:.2%}
    
    **Risk Management Summary**:
    - Risk Events: {enhanced_metrics.get('risk_events_count', 0)}
    - Drawdown Violations: {risk_analysis.get('drawdown_violations', 0)}
    - Correlation Violations: {risk_analysis.get('correlation_violations', 0)}
    - Risk-Adjusted Trades: {risk_analysis.get('risk_adjusted_trades', 0)}
    
    **Trading Summary**:
    - Total Trades: {len(enhanced_trades)}
    - Average Holding Period: {np.mean([t.holding_period for t in enhanced_trades]):.1f} days
    - Win Rate: {sum(1 for t in enhanced_trades if t.net_pnl > 0) / len(enhanced_trades):.1%}
    """)
    
    # Configuration summary
    st.markdown("### ⚙️ Configuration Summary")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Backtest Parameters**:")
        st.markdown(f"- Initial Capital: ₹{config.initial_capital:,}")
        st.markdown(f"- Transaction Cost: {config.transaction_cost_pct:.3%}")
        st.markdown(f"- Max Positions: {config.max_positions}")
        st.markdown(f"- Rebalancing: {config.rebalance_frequency}")
        st.markdown(f"- Position Sizing: {config.position_sizing_method}")
    
    with col2:
        st.markdown("**Risk Management Parameters**:")
        st.markdown(f"- Max Drawdown: {config.max_portfolio_drawdown:.1%}")
        st.markdown(f"- Max Correlation: {config.max_position_correlation:.2f}")
        st.markdown(f"- VaR Confidence: {config.var_confidence_level:.1%}")
        st.markdown(f"- Kelly Cap: {config.kelly_fraction_cap:.1%}")
        st.markdown(f"- Stress Test Frequency: {config.stress_test_frequency} days")
    
    # Downloadable detailed report
    if st.button("📥 Generate Downloadable Report"):
        detailed_report = generate_detailed_text_report(results, config, enhanced_params)
        st.download_button(
            label="Download Detailed Report",
            data=detailed_report,
            file_name=f"enhanced_backtest_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain"
        )

def create_enhanced_historical_backtests_section():
    """Create enhanced historical backtests section"""
    
    st.subheader("📊 Historical Enhanced Backtests")
    
    try:
        db = BacktestDB()
        historical_backtests = db.list_backtests()
        
        if not historical_backtests.empty:
            # Enhanced display with filtering
            col1, col2 = st.columns(2)
            
            with col1:
                min_return = st.slider("Min Total Return (%)", -50, 100, -50)
                max_drawdown_filter = st.slider("Max Drawdown Filter (%)", 0, 50, 50)
            
            with col2:
                min_sharpe = st.slider("Min Sharpe Ratio", -2.0, 5.0, -2.0, 0.1)
                min_trades = st.number_input("Min Trades", 0, 1000, 0)
            
            # Apply filters
            filtered_backtests = historical_backtests[
                (historical_backtests['total_return'] >= min_return/100) &
                (historical_backtests['max_drawdown'] >= -max_drawdown_filter/100) &
                (historical_backtests['sharpe_ratio'] >= min_sharpe) &
                (historical_backtests['total_trades'] >= min_trades)
            ]
            
            if not filtered_backtests.empty:
                # Display enhanced table
                display_cols = ['name', 'initial_capital', 'final_value', 'total_return', 
                              'sharpe_ratio', 'max_drawdown', 'total_trades', 'win_rate', 'created_at']
                
                formatted_df = filtered_backtests[display_cols].copy()
                formatted_df['total_return'] = formatted_df['total_return'].apply(lambda x: f"{x:.2%}")
                formatted_df['sharpe_ratio'] = formatted_df['sharpe_ratio'].apply(lambda x: f"{x:.3f}")
                formatted_df['max_drawdown'] = formatted_df['max_drawdown'].apply(lambda x: f"{x:.2%}")
                formatted_df['win_rate'] = formatted_df['win_rate'].apply(lambda x: f"{x:.2%}")
                formatted_df['initial_capital'] = formatted_df['initial_capital'].apply(lambda x: f"₹{x:,.0f}")
                formatted_df['final_value'] = formatted_df['final_value'].apply(lambda x: f"₹{x:,.0f}")
                
                st.dataframe(formatted_df, use_container_width=True)
                
                # Enhanced comparison functionality
                selected_backtests = st.multiselect(
                    "Select backtests to compare:",
                    options=filtered_backtests['id'].tolist(),
                    format_func=lambda x: f"ID {x}: {filtered_backtests[filtered_backtests['id']==x]['name'].iloc[0]}"
                )
                
                if selected_backtests and st.button("🔍 Compare Selected Enhanced Backtests"):
                    compare_enhanced_backtests(selected_backtests, db)
            
            else:
                st.info("No backtests match the current filters.")
        
        else:
            st.info("No historical enhanced backtests found. Run your first enhanced backtest above!")
            
    except Exception as e:
        st.warning(f"Could not load historical backtests: {e}")

def compare_enhanced_backtests(backtest_ids, db):
    """Compare multiple enhanced backtests"""
    
    st.subheader("🔍 Enhanced Backtest Comparison")
    
    comparison_data = []
    portfolio_data = {}
    
    for backtest_id in backtest_ids:
        try:
            results = db.load_backtest(backtest_id)
            if results and 'enhanced_metrics' in results:
                metrics = results['enhanced_metrics']
                portfolio_df = results['portfolio_history']
                
                comparison_data.append({
                    'Backtest ID': backtest_id,
                    'Total Return': f"{metrics['total_return']:.2%}",
                    'Annual Return': f"{metrics['annual_return']:.2%}",
                    'Sharpe Ratio': f"{metrics['sharpe_ratio']:.3f}",
                    'Max Drawdown': f"{metrics['max_drawdown']:.2%}",
                    'VaR (95%)': f"{metrics.get('var_95', 0):.2%}",
                    'Risk Events': metrics.get('risk_events_count', 0),
                    'Total Trades': len(results.get('enhanced_trades', []))
                })
                
                # Normalize portfolio values for comparison
                normalized_portfolio = portfolio_df['portfolio_value'] / portfolio_df['portfolio_value'].iloc[0]
                portfolio_data[f"Backtest {backtest_id}"] = normalized_portfolio
                
        except Exception as e:
            st.error(f"Could not load enhanced backtest {backtest_id}: {e}")
    
    if comparison_data:
        # Enhanced comparison table
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Enhanced portfolio comparison charts
        if portfolio_data:
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=["Normalized Portfolio Performance", "Performance Comparison"],
                vertical_spacing=0.1
            )
            
            # Portfolio comparison
            for name, portfolio_values in portfolio_data.items():
                fig.add_trace(
                    go.Scatter(
                        x=portfolio_values.index,
                        y=portfolio_values,
                        mode='lines',
                        name=name,
                        line=dict(width=2)
                    ),
                    row=1, col=1
                )
            
            # Performance metrics comparison
            backtest_names = [f"Backtest {bid}" for bid in backtest_ids]
            total_returns = [float(row['Total Return'].strip('%'))/100 for row in comparison_data]
            sharpe_ratios = [float(row['Sharpe Ratio']) for row in comparison_data]
            
            fig.add_trace(
                go.Bar(
                    x=backtest_names,
                    y=total_returns,
                    name='Total Return',
                    yaxis='y2'
                ),
                row=2, col=1
            )
            
            fig.update_layout(
                height=800,
                title="Enhanced Backtest Comparison Analysis"
            )
            
            fig.update_yaxes(title_text="Normalized Value", row=1, col=1)
            fig.update_yaxes(title_text="Total Return", row=2, col=1)
            
            st.plotly_chart(fig, use_container_width=True)

def generate_detailed_text_report(results, config, enhanced_params):
    """Generate detailed text report for download"""
    
    enhanced_metrics = results['enhanced_metrics']
    risk_analysis = results['risk_analysis']
    enhanced_trades = results['enhanced_trades']
    risk_events = results['risk_events']
    
    report = f"""
ENHANCED BACKTEST REPORT
{'='*50}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

EXECUTIVE SUMMARY
{'='*50}
Total Return: {enhanced_metrics['total_return']:.2%}
Annual Return: {enhanced_metrics['annual_return']:.2%}
Volatility: {enhanced_metrics['volatility']:.2%}
Sharpe Ratio: {enhanced_metrics['sharpe_ratio']:.3f}
Sortino Ratio: {enhanced_metrics['sortino_ratio']:.3f}
Calmar Ratio: {enhanced_metrics['calmar_ratio']:.3f}
Maximum Drawdown: {enhanced_metrics['max_drawdown']:.2%}

RISK MANAGEMENT PERFORMANCE
{'='*50}
Risk Events: {enhanced_metrics.get('risk_events_count', 0)}
Drawdown Violations: {risk_analysis.get('drawdown_violations', 0)}
Correlation Violations: {risk_analysis.get('correlation_violations', 0)}
Stress Test Failures: {risk_analysis.get('stress_test_failures', 0)}
Risk-Adjusted Trades: {risk_analysis.get('risk_adjusted_trades', 0)}

TRADING PERFORMANCE
{'='*50}
Total Trades: {len(enhanced_trades)}
Win Rate: {sum(1 for t in enhanced_trades if t.net_pnl > 0) / len(enhanced_trades):.1%}
Average Holding Period: {np.mean([t.holding_period for t in enhanced_trades]):.1f} days
Best Trade: ₹{max(t.net_pnl for t in enhanced_trades):,.0f}
Worst Trade: ₹{min(t.net_pnl for t in enhanced_trades):,.0f}

CONFIGURATION DETAILS
{'='*50}
Initial Capital: ₹{config.initial_capital:,}
Transaction Cost: {config.transaction_cost_pct:.3%}
Max Positions: {config.max_positions}
Rebalancing Frequency: {config.rebalance_frequency}
Position Sizing Method: {config.position_sizing_method}
Max Drawdown Limit: {config.max_portfolio_drawdown:.1%}
Max Correlation: {config.max_position_correlation:.2f}
VaR Confidence Level: {config.var_confidence_level:.1%}

DETAILED TRADE LOG
{'='*50}
"""
    
    for i, trade in enumerate(enhanced_trades):
        report += f"""
Trade {i+1}:
  Ticker: {trade.ticker}
  Entry: {trade.entry_date.strftime('%Y-%m-%d')} @ ₹{trade.entry_price:.2f}
  Exit: {trade.exit_date.strftime('%Y-%m-%d')} @ ₹{trade.exit_price:.2f}
  Holding Period: {trade.holding_period} days
  Return: {trade.return_pct:.2%}
  P&L: ₹{trade.net_pnl:,.0f}
  Exit Reason: {trade.exit_signal}
"""
    
    if risk_events:
        report += f"\n\nRISK EVENTS LOG\n{'='*50}\n"
        for i, event in enumerate(risk_events):
            report += f"""
Risk Event {i+1}:
  Date: {event['date'].strftime('%Y-%m-%d %H:%M')}
  Type: {event.get('type', 'Violation')}
  Details: {event.get('violations', [])}
  Actions: {event.get('actions_taken', [])}
"""
    
    report += f"\n\nREPORT END\n{'='*50}\n"
    
    return report

def generate_downloadable_report(results, config, backtest_name):
    """Generate and offer downloadable report"""
    
    detailed_report = generate_detailed_text_report(results, config, {})
    
    st.download_button(
        label="📥 Download Enhanced Backtest Report",
        data=detailed_report,
        file_name=f"{backtest_name}_detailed_report.txt",
        mime="text/plain",
        help="Download comprehensive backtest report as text file"
    )

# Additional utility functions for integration
def get_available_models_for_backtesting(models):
    """Get available models that can be used for backtesting"""
    
    available_models = {}
    model_count = 0
    
    for ticker, model_dict in models.items():
        if model_dict:
            available_models[ticker] = model_dict
            model_count += len(model_dict)
    
    return available_models, model_count

def validate_backtest_requirements(models, featured_data, raw_data, selected_tickers):
    """Validate that all requirements for backtesting are met"""
    
    requirements = {
        'models_available': bool(models),
        'featured_data_available': bool(featured_data),
        'raw_data_available': bool(raw_data),
        'tickers_selected': bool(selected_tickers),
        'sufficient_data': False,
        'models_trained_for_tickers': False
    }
    
    if raw_data:
        # Check if we have sufficient data points
        total_data_points = sum(len(df) for df in raw_data.values())
        requirements['sufficient_data'] = total_data_points > 1000
    
    if models and selected_tickers:
        # Check if models are trained for selected tickers
        model_tickers = set(models.keys())
        selected_set = set(selected_tickers)
        overlap = len(model_tickers.intersection(selected_set))
        requirements['models_trained_for_tickers'] = overlap >= len(selected_tickers) * 0.8
    
    return requirements

# Export functions for use in main app
__all__ = [
    'create_enhanced_backtesting_tab',
    'display_enhanced_backtest_results', 
    'get_available_models_for_backtesting',
    'validate_backtest_requirements'
]

if __name__ == "__main__":
    print("Enhanced Backtesting Integration with Comprehensive Risk Management")
    print("="*70)
    print("Features:")
    print("✓ Real-time risk monitoring during backtesting")
    print("✓ Advanced position sizing with Kelly Criterion & Risk Parity")
    print("✓ Continuous correlation and drawdown tracking")
    print("✓ Periodic stress testing with Monte Carlo scenarios")
    print("✓ Risk-based exit signals beyond strategy signals")
    print("✓ Comprehensive risk event logging and analysis")
    print("✓ Enhanced performance visualization and reporting")
    print("✓ Historical backtest comparison with risk metrics")
    print("✓ Downloadable detailed reports")
    print("✓ Pre-flight validation and requirement checking")