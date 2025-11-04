"""
Improved Risk Management Display
Makes risk metrics easy to understand for end users
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List
import numpy as np


def display_improved_risk_metrics(risk_analysis: Dict, predictions_df: pd.DataFrame):
    """
    Display risk metrics in a user-friendly way

    Args:
        risk_analysis: Risk analysis results
        predictions_df: Predictions dataframe
    """

    st.header("🛡️ Portfolio Risk Analysis - Easy to Understand")

    # Check if risk analysis has errors or is empty
    if not risk_analysis or 'error' in risk_analysis:
        st.error("⚠️ Risk analysis data unavailable")
        st.info("**Why this might happen:**")
        st.caption("• Not enough historical data for reliable risk calculations")
        st.caption("• Predictions data is insufficient")
        st.caption("• Less than 30 days of price history")

        st.info("**💡 Solution:** Select stocks with at least 6 months of trading history")
        return

    # Extract metrics with safe defaults
    portfolio_summary = risk_analysis.get('portfolio_summary', {})
    drawdown_analysis = risk_analysis.get('drawdown_analysis', {})
    stress_testing = risk_analysis.get('stress_testing', {})
    correlation_analysis = risk_analysis.get('correlation_analysis', {})

    # ==================== SECTION 1: RISK LEVEL INDICATOR ====================
    st.subheader("📊 Overall Risk Level")

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        # Calculate overall risk level
        risk_score = _calculate_risk_score(risk_analysis)

        if risk_score < 30:
            risk_level = "LOW"
            risk_color = "#28a745"
            risk_emoji = "🟢"
            risk_message = "Conservative portfolio with minimal risk"
        elif risk_score < 60:
            risk_level = "MEDIUM"
            risk_color = "#ffc107"
            risk_emoji = "🟡"
            risk_message = "Balanced risk-reward profile"
        else:
            risk_level = "HIGH"
            risk_color = "#dc3545"
            risk_emoji = "🔴"
            risk_message = "Aggressive portfolio with higher volatility"

        st.markdown(f"""
        <div style='background: {risk_color}; padding: 20px; border-radius: 10px; text-align: center;'>
            <h1 style='color: white; margin: 0;'>{risk_emoji} {risk_level} RISK</h1>
            <p style='color: white; margin: 10px 0 0 0;'>{risk_message}</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        risk_score_display = min(risk_score, 100)
        st.metric("Risk Score", f"{risk_score_display}/100",
                 help="0-30: Low Risk | 30-60: Medium Risk | 60-100: High Risk")

    with col3:
        n_positions = portfolio_summary.get('n_positions', len(predictions_df))
        st.metric("Holdings", n_positions,
                 help="Number of stocks in your portfolio")

    # ==================== SECTION 2: KEY METRICS (SIMPLIFIED) ====================
    st.markdown("---")
    st.subheader("📈 Key Risk Metrics Explained")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

    # Get Monte Carlo stress test results
    monte_carlo = stress_testing.get('monte_carlo', {})
    var_95 = monte_carlo.get('var_95', 0)
    expected_shortfall = monte_carlo.get('expected_shortfall', 0)

    with metric_col1:
        var_display = abs(var_95 * 100) if var_95 != 0 else 0
        st.metric(
            "Maximum Loss Risk",
            f"{var_display:.1f}%",
            help="In 95% of cases, you won't lose more than this in a day"
        )
        if var_display == 0:
            st.caption("⚠️ Insufficient data")
        elif var_display < 2:
            st.caption("✅ Very Low")
        elif var_display < 5:
            st.caption("⚠️ Moderate")
        else:
            st.caption("🔴 High")

    with metric_col2:
        max_dd = drawdown_analysis.get('max_drawdown', 0)
        max_dd_display = abs(max_dd * 100) if max_dd != 0 else 0
        st.metric(
            "Worst Historical Drop",
            f"{max_dd_display:.1f}%",
            help="Biggest drop from peak to bottom in the past"
        )
        if max_dd_display == 0:
            st.caption("⚠️ Insufficient data")
        elif max_dd_display < 10:
            st.caption("✅ Stable")
        elif max_dd_display < 20:
            st.caption("⚠️ Moderate Volatility")
        else:
            st.caption("🔴 High Volatility")

    with metric_col3:
        current_dd = drawdown_analysis.get('current_drawdown', 0)
        current_dd_display = abs(current_dd * 100) if current_dd != 0 else 0
        st.metric(
            "Current Position",
            f"{current_dd_display:.1f}%",
            help="How far are you from your all-time high?"
        )
        if current_dd_display == 0:
            st.caption("✅ At Peak")
        elif current_dd_display < 5:
            st.caption("✅ Near Peak")
        elif current_dd_display < 10:
            st.caption("⚠️ Slight Dip")
        else:
            st.caption("📉 Recovering")

    with metric_col4:
        # Calculate portfolio correlation
        max_corr = correlation_analysis.get('max_correlation', 0)
        max_corr_display = abs(max_corr * 100) if max_corr != 0 else 0
        st.metric(
            "Diversification",
            f"{100 - max_corr_display:.0f}%",
            help="How well diversified is your portfolio? (Higher is better)"
        )
        if max_corr_display == 0:
            st.caption("⚠️ Needs calculation")
        elif max_corr_display < 50:
            st.caption("✅ Well Diversified")
        elif max_corr_display < 70:
            st.caption("⚠️ Moderate")
        else:
            st.caption("🔴 Too Concentrated")

    # ==================== SECTION 3: WHAT DOES THIS MEAN? ====================
    st.markdown("---")
    st.subheader("💡 What Does This Mean For You?")

    interpretation_col1, interpretation_col2 = st.columns(2)

    with interpretation_col1:
        st.markdown("### 🎯 Your Risk Profile")

        if risk_level == "LOW":
            st.success("""
            **Conservative Investor**
            - Your portfolio is relatively stable
            - Lower potential returns, but also lower losses
            - Good for capital preservation
            - Suitable for risk-averse investors
            """)
        elif risk_level == "MEDIUM":
            st.info("""
            **Balanced Investor**
            - Moderate risk-reward balance
            - Some volatility expected
            - Good mix of growth and stability
            - Suitable for most investors
            """)
        else:
            st.warning("""
            **Aggressive Investor**
            - Higher volatility expected
            - Potential for larger gains AND losses
            - Requires careful monitoring
            - Suitable for risk-tolerant investors
            """)

    with interpretation_col2:
        st.markdown("### 📋 Recommendations")

        recommendations = []

        # Generate contextual recommendations
        if var_display > 5:
            recommendations.append("🔴 Consider reducing position sizes to lower daily risk")

        if max_dd_display > 20:
            recommendations.append("⚠️ Portfolio has experienced large swings - review stop-losses")

        if max_corr_display > 70:
            recommendations.append("🔴 Add stocks from different sectors for better diversification")

        if current_dd_display > 10:
            recommendations.append("📊 Portfolio is in drawdown - wait for recovery or rebalance")

        if n_positions < 5:
            recommendations.append("💡 Consider adding more stocks to spread risk")

        if n_positions > 20:
            recommendations.append("⚠️ Too many positions may be hard to monitor - consider consolidation")

        if not recommendations:
            recommendations.append("✅ Portfolio looks well-balanced!")

        for rec in recommendations:
            st.markdown(f"- {rec}")

    # ==================== SECTION 4: STRESS TEST SCENARIOS ====================
    st.markdown("---")
    st.subheader("🔥 How Would Your Portfolio Handle Crises?")

    if 'historical_scenarios' in stress_testing:
        scenarios = stress_testing['historical_scenarios']

        st.caption("These scenarios show how your portfolio might perform during major market events:")

        scenario_data = []
        for scenario_name, scenario_result in scenarios.items():
            impact = scenario_result.get('portfolio_impact', 0) * 100

            scenario_data.append({
                'Scenario': scenario_name.replace('_', ' ').title(),
                'Impact': impact,
                'Impact_Abs': abs(impact)
            })

        if scenario_data:
            df_scenarios = pd.DataFrame(scenario_data)
            df_scenarios = df_scenarios.sort_values('Impact')

            fig = go.Figure()

            colors = ['#28a745' if x >= 0 else '#dc3545' for x in df_scenarios['Impact']]

            fig.add_trace(go.Bar(
                x=df_scenarios['Impact'],
                y=df_scenarios['Scenario'],
                orientation='h',
                marker_color=colors,
                text=[f"{x:+.1f}%" for x in df_scenarios['Impact']],
                textposition='outside',
                hovertemplate='<b>%{y}</b><br>Impact: %{x:.2f}%<extra></extra>'
            ))

            fig.update_layout(
                title="Portfolio Performance in Historical Market Crises",
                xaxis_title="Portfolio Impact (%)",
                yaxis_title="",
                height=300,
                showlegend=False,
                xaxis=dict(zeroline=True, zerolinewidth=2, zerolinecolor='gray')
            )

            st.plotly_chart(fig, use_container_width=True, key="stress_scenarios_chart")

            # Find worst scenario
            worst_scenario = df_scenarios.loc[df_scenarios['Impact'].idxmin()]
            st.info(f"📉 **Worst Case:** In a {worst_scenario['Scenario']}-like event, your portfolio might lose **{abs(worst_scenario['Impact']):.1f}%**")

    else:
        st.info("Stress testing results not available - need more historical data")

    # ==================== SECTION 5: CORRELATION HEATMAP (IMPROVED) ====================
    st.markdown("---")
    st.subheader("🔗 How Are Your Stocks Connected?")

    st.caption("**Correlation shows if stocks move together:**")
    st.caption("🟢 Green (low correlation) = Good! Stocks move independently")
    st.caption("🔴 Red (high correlation) = Warning! Stocks move together (less diversification)")

    if correlation_analysis and 'correlation_matrix' in correlation_analysis:
        corr_matrix = correlation_analysis['correlation_matrix']

        if corr_matrix:
            corr_df = pd.DataFrame(corr_matrix)

            # Create annotated heatmap
            fig = go.Figure(data=go.Heatmap(
                z=corr_df.values,
                x=corr_df.columns,
                y=corr_df.index,
                colorscale=[
                    [0, '#28a745'],     # Green (good - low correlation)
                    [0.5, '#ffc107'],   # Yellow (moderate)
                    [1, '#dc3545']      # Red (bad - high correlation)
                ],
                zmid=0,
                text=[[f"{val:.2f}" for val in row] for row in corr_df.values],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="Correlation")
            ))

            fig.update_layout(
                title="Stock Correlation Matrix (Lower is Better for Diversification)",
                height=500,
                xaxis_title="",
                yaxis_title=""
            )

            st.plotly_chart(fig, use_container_width=True, key="improved_correlation_heatmap")

            # Explain high correlations
            high_correlations = correlation_analysis.get('high_correlations', [])
            if high_correlations:
                st.warning(f"⚠️ **Found {len(high_correlations)} pairs of highly correlated stocks:**")

                for pair in high_correlations[:5]:  # Show top 5
                    stock1 = pair[0]
                    stock2 = pair[1]
                    corr_val = pair[2]
                    st.caption(f"   • **{stock1}** and **{stock2}** move together {abs(corr_val)*100:.0f}% of the time")

                st.info("💡 **Tip:** Consider replacing one stock in each pair with a stock from a different sector")
            else:
                st.success("✅ Great! Your stocks are well-diversified with low correlations")
        else:
            st.info("Correlation data not available")


def _calculate_risk_score(risk_analysis: Dict) -> float:
    """Calculate overall risk score (0-100)"""
    score = 50  # Start with medium risk

    # Adjust based on various factors
    monte_carlo = risk_analysis.get('stress_testing', {}).get('monte_carlo', {})
    var_95 = abs(monte_carlo.get('var_95', 0))

    drawdown = risk_analysis.get('drawdown_analysis', {})
    max_dd = abs(drawdown.get('max_drawdown', 0))

    correlation = risk_analysis.get('correlation_analysis', {})
    max_corr = abs(correlation.get('max_correlation', 0))

    # VaR contribution (0-30 points)
    if var_95 == 0:
        score += 0  # No data
    elif var_95 > 0.05:  # > 5% daily VaR
        score += 30
    elif var_95 > 0.03:  # > 3% daily VaR
        score += 20
    elif var_95 > 0.02:  # > 2% daily VaR
        score += 10
    else:
        score -= 10  # Very low risk

    # Drawdown contribution (0-30 points)
    if max_dd == 0:
        score += 0
    elif max_dd > 0.30:  # > 30% max drawdown
        score += 30
    elif max_dd > 0.20:  # > 20% max drawdown
        score += 20
    elif max_dd > 0.10:  # > 10% max drawdown
        score += 10
    else:
        score -= 10

    # Correlation contribution (0-20 points)
    if max_corr > 0.7:
        score += 20  # High concentration risk
    elif max_corr > 0.5:
        score += 10
    else:
        score -= 10  # Well diversified

    # Clamp to 0-100
    return max(0, min(100, score))


if __name__ == "__main__":
    print("Improved Risk Display Module - Ready for integration")
