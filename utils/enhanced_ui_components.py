# utils/enhanced_ui_components.py
"""
Enhanced UI Components Module
Reusable Streamlit components with professional styling
Author: AI Stock Advisor Pro Team
Version: 2.0 - Enhanced Edition
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
import logging

# ==================== COLOR SCHEMES ====================

class ColorSchemes:
    """Professional color schemes for different themes"""
    
    PRIMARY = {
        'main': '#667eea',
        'secondary': '#764ba2',
        'light': '#a8b7f5',
        'dark': '#4a5bcc'
    }
    
    SUCCESS = {
        'main': '#28a745',
        'light': '#d4edda',
        'dark': '#155724'
    }
    
    DANGER = {
        'main': '#dc3545',
        'light': '#f8d7da',
        'dark': '#721c24'
    }
    
    WARNING = {
        'main': '#ffc107',
        'light': '#fff3cd',
        'dark': '#856404'
    }
    
    INFO = {
        'main': '#17a2b8',
        'light': '#d1ecf1',
        'dark': '#0c5460'
    }
    
    GRADIENTS = {
        'primary': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
        'success': 'linear-gradient(135deg, #28a745 0%, #20c997 100%)',
        'danger': 'linear-gradient(135deg, #dc3545 0%, #e83e8c 100%)',
        'warning': 'linear-gradient(135deg, #ffc107 0%, #fd7e14 100%)',
        'info': 'linear-gradient(135deg, #17a2b8 0%, #6f42c1 100%)',
        'cool': 'linear-gradient(135deg, #4facfe 0%, #00f2fe 100%)',
        'warm': 'linear-gradient(135deg, #43e97b 0%, #38f9d7 100%)',
        'sunset': 'linear-gradient(135deg, #f093fb 0%, #f5576c 100%)'
    }

# ==================== BASIC COMPONENTS ====================

class EnhancedComponents:
    """Collection of enhanced UI components"""
    
    @staticmethod
    def create_metric_card(
        title: str, 
        value: str, 
        delta: Optional[str] = None, 
        color: str = "primary",
        icon: Optional[str] = None,
        help_text: Optional[str] = None
    ):
        """Create an enhanced metric card with professional styling"""
        
        gradient = ColorSchemes.GRADIENTS.get(color, ColorSchemes.GRADIENTS['primary'])
        
        icon_html = f"<div style='font-size: 2rem; margin-bottom: 0.5rem;'>{icon}</div>" if icon else ""
        delta_html = f"<p style='margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.8); font-size: 0.9rem;'>{delta}</p>" if delta else ""
        help_html = f"<p style='margin: 0.5rem 0 0 0; color: rgba(255,255,255,0.7); font-size: 0.8rem; font-style: italic;'>{help_text}</p>" if help_text else ""
        
        st.markdown(f"""
        <div style='background: {gradient}; 
                    color: white; 
                    padding: 2rem; 
                    border-radius: 20px; 
                    text-align: center; 
                    margin: 1rem 0;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2);
                    transition: transform 0.3s ease, box-shadow 0.3s ease;'>
            {icon_html}
            <h4 style='margin: 0 0 0.5rem 0; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 1px; opacity: 0.9;'>{title}</h4>
            <h2 style='margin: 0; font-size: 2.2rem; font-weight: 700;'>{value}</h2>
            {delta_html}
            {help_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_status_indicator(status: str, message: str, details: Optional[str] = None):
        """Create status indicators with appropriate styling"""
        
        status_config = {
            'success': {'color': ColorSchemes.SUCCESS['main'], 'icon': '✅', 'bg': ColorSchemes.GRADIENTS['success']},
            'warning': {'color': ColorSchemes.WARNING['main'], 'icon': '⚠️', 'bg': ColorSchemes.GRADIENTS['warning']},
            'error': {'color': ColorSchemes.DANGER['main'], 'icon': '❌', 'bg': ColorSchemes.GRADIENTS['danger']},
            'info': {'color': ColorSchemes.INFO['main'], 'icon': 'ℹ️', 'bg': ColorSchemes.GRADIENTS['info']},
            'loading': {'color': ColorSchemes.PRIMARY['main'], 'icon': '🔄', 'bg': ColorSchemes.GRADIENTS['primary']}
        }
        
        config = status_config.get(status, status_config['info'])
        details_html = f"<p style='margin: 0.5rem 0 0 0; opacity: 0.9; font-size: 0.9rem;'>{details}</p>" if details else ""
        
        st.markdown(f"""
        <div style='background: {config["bg"]}; 
                    color: white; 
                    padding: 1.5rem; 
                    border-radius: 15px; 
                    margin: 1rem 0; 
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    display: flex;
                    align-items: center;'>
            <div style='font-size: 1.5rem; margin-right: 1rem;'>{config["icon"]}</div>
            <div style='flex: 1;'>
                <p style='margin: 0; font-weight: 600; font-size: 1.1rem;'>{message}</p>
                {details_html}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_progress_indicator(steps: List[str], current_step: int, title: str = "Progress"):
        """Create a visual progress indicator"""
        
        st.markdown(f"""
        <div style='background: {ColorSchemes.GRADIENTS["primary"]}; 
                    color: white; 
                    padding: 2rem; 
                    border-radius: 20px; 
                    margin: 2rem 0;'>
            <h3 style='margin: 0 0 2rem 0; text-align: center; font-size: 1.5rem;'>{title}</h3>
        </div>
        """, unsafe_allow_html=True)
        
        progress_html = "<div style='display: flex; justify-content: space-between; margin: 2rem 0; position: relative;'>"
        
        # Add connecting line
        progress_html += """
        <div style='position: absolute; top: 20px; left: 5%; right: 5%; height: 2px; 
                    background: linear-gradient(90deg, #28a745 0%, #ffc107 50%, #dee2e6 100%);'></div>
        """
        
        for i, step in enumerate(steps):
            if i < current_step:
                color = ColorSchemes.SUCCESS['main']
                icon = "✅"
                text_color = ColorSchemes.SUCCESS['main']
            elif i == current_step:
                color = ColorSchemes.WARNING['main'] 
                icon = "🔄"
                text_color = ColorSchemes.WARNING['main']
            else:
                color = "#dee2e6"
                icon = "⏳"
                text_color = "#6c757d"
                
            progress_html += f"""
            <div style='text-align: center; flex: 1; position: relative; z-index: 2;'>
                <div style='width: 40px; height: 40px; border-radius: 50%; 
                           background: {color}; color: white; 
                           display: flex; align-items: center; justify-content: center; 
                           margin: 0 auto; font-size: 1.2rem; font-weight: bold;
                           box-shadow: 0 3px 10px rgba(0,0,0,0.2);'>{icon}</div>
                <p style='margin: 1rem 0 0 0; font-size: 0.9rem; color: {text_color}; 
                          font-weight: 600; max-width: 120px; margin-left: auto; margin-right: auto;'>{step}</p>
            </div>
            """
        
        progress_html += "</div>"
        st.markdown(progress_html, unsafe_allow_html=True)
    
    @staticmethod
    def create_feature_card(
        title: str, 
        description: str, 
        features: List[str], 
        color: str = "primary",
        icon: Optional[str] = None
    ):
        """Create feature showcase cards"""
        
        gradient = ColorSchemes.GRADIENTS.get(color, ColorSchemes.GRADIENTS['primary'])
        icon_html = f"<div style='font-size: 3rem; margin-bottom: 1rem;'>{icon}</div>" if icon else ""
        
        features_html = ""
        for feature in features:
            features_html += f"<li style='margin: 0.5rem 0; font-size: 0.95rem; line-height: 1.4;'>{feature}</li>"
        
        st.markdown(f"""
        <div style='background: {gradient}; 
                    color: white; 
                    padding: 2.5rem; 
                    border-radius: 20px; 
                    text-align: center; 
                    height: 320px;
                    box-shadow: 0 10px 30px rgba(0,0,0,0.2); 
                    transition: transform 0.3s ease;
                    display: flex;
                    flex-direction: column;
                    justify-content: space-between;'>
            <div>
                {icon_html}
                <h3 style='margin-bottom: 1rem; font-size: 1.3rem; font-weight: 600;'>{title}</h3>
                <p style='font-size: 0.95rem; line-height: 1.4; opacity: 0.9; margin-bottom: 1.5rem;'>{description}</p>
            </div>
            <ul style='text-align: left; padding-left: 1.5rem; margin: 0; flex-grow: 1;'>
                {features_html}
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_info_panel(
        title: str,
        content: str,
        panel_type: str = "info",
        collapsible: bool = False,
        expanded: bool = True
    ):
        """Create informational panels"""
        
        type_config = {
            'info': {'gradient': ColorSchemes.GRADIENTS['info'], 'icon': 'ℹ️'},
            'tip': {'gradient': ColorSchemes.GRADIENTS['success'], 'icon': '💡'},
            'warning': {'gradient': ColorSchemes.GRADIENTS['warning'], 'icon': '⚠️'},
            'error': {'gradient': ColorSchemes.GRADIENTS['danger'], 'icon': '🚨'}
        }
        
        config = type_config.get(panel_type, type_config['info'])
        
        if collapsible:
            with st.expander(f"{config['icon']} {title}", expanded=expanded):
                st.markdown(f"""
                <div style='background: {config["gradient"]}; 
                            color: white; 
                            padding: 1.5rem; 
                            border-radius: 15px; 
                            margin: 1rem 0;'>
                    <p style='margin: 0; line-height: 1.6;'>{content}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div style='background: {config["gradient"]}; 
                        color: white; 
                        padding: 2rem; 
                        border-radius: 20px; 
                        margin: 2rem 0;
                        box-shadow: 0 8px 25px rgba(0,0,0,0.1);'>
                <div style='display: flex; align-items: center; margin-bottom: 1rem;'>
                    <span style='font-size: 1.8rem; margin-right: 1rem;'>{config["icon"]}</span>
                    <h3 style='margin: 0; font-size: 1.4rem; font-weight: 600;'>{title}</h3>
                </div>
                <p style='margin: 0; line-height: 1.6; font-size: 1.05rem;'>{content}</p>
            </div>
            """, unsafe_allow_html=True)

# ==================== CHART COMPONENTS ====================

class ChartComponents:
    """Enhanced chart components with professional styling"""
    
    @staticmethod
    def create_enhanced_gauge(
        value: float,
        title: str,
        min_val: float = 0,
        max_val: float = 100,
        thresholds: Optional[Dict[str, float]] = None,
        color_scheme: str = "primary"
    ) -> go.Figure:
        """Create an enhanced gauge chart"""
        
        if thresholds is None:
            thresholds = {'good': max_val * 0.7, 'excellent': max_val * 0.9}
        
        # Determine color based on value
        if value >= thresholds.get('excellent', max_val * 0.9):
            bar_color = ColorSchemes.SUCCESS['main']
        elif value >= thresholds.get('good', max_val * 0.7):
            bar_color = ColorSchemes.WARNING['main']
        else:
            bar_color = ColorSchemes.DANGER['main']
        
        fig = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=value,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': title, 'font': {'size': 16, 'color': '#2E4057'}},
            gauge={
                'axis': {'range': [None, max_val]},
                'bar': {'color': bar_color},
                'steps': [
                    {'range': [min_val, thresholds.get('good', max_val * 0.7)], 'color': "lightgray"},
                    {'range': [thresholds.get('good', max_val * 0.7), thresholds.get('excellent', max_val * 0.9)], 'color': "yellow"},
                    {'range': [thresholds.get('excellent', max_val * 0.9), max_val], 'color': "lightgreen"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': thresholds.get('excellent', max_val * 0.9)
                }
            }
        ))
        
        fig.update_layout(
            height=300,
            font={'family': 'Inter, sans-serif'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    @staticmethod
    def create_performance_comparison_chart(
        data: Dict[str, float],
        title: str = "Performance Comparison",
        chart_type: str = "bar"
    ) -> go.Figure:
        """Create performance comparison charts"""
        
        categories = list(data.keys())
        values = list(data.values())
        
        # Color mapping based on values
        colors = []
        for value in values:
            if value > 0.1:  # >10%
                colors.append(ColorSchemes.SUCCESS['main'])
            elif value > 0:
                colors.append(ColorSchemes.WARNING['main'])
            else:
                colors.append(ColorSchemes.DANGER['main'])
        
        if chart_type == "bar":
            fig = go.Figure(data=[
                go.Bar(
                    x=categories,
                    y=values,
                    marker=dict(
                        color=colors,
                        opacity=0.8,
                        line=dict(color='white', width=2)
                    ),
                    text=[f"{v:.1%}" for v in values],
                    textposition='outside'
                )
            ])
            
            fig.update_layout(
                title=title,
                xaxis_title="Metrics",
                yaxis_title="Value",
                yaxis=dict(tickformat='.1%')
            )
        
        elif chart_type == "radar":
            fig = go.Figure()
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=categories,
                fill='toself',
                name='Performance',
                line=dict(color=ColorSchemes.PRIMARY['main'], width=2),
                fillcolor=f'rgba(102, 126, 234, 0.2)'
            ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[min(values + [0]), max(values + [0])]
                    )),
                showlegend=True,
                title=title
            )
        
        fig.update_layout(
            height=400,
            font={'family': 'Inter, sans-serif'},
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)'
        )
        
        return fig
    
    @staticmethod
    def create_timeline_chart(
        data: pd.DataFrame,
        x_col: str,
        y_col: str,
        title: str = "Timeline Chart",
        color_col: Optional[str] = None
    ) -> go.Figure:
        """Create enhanced timeline charts"""
        
        fig = go.Figure()
        
        if color_col and color_col in data.columns:
            # Multi-colored timeline
            unique_categories = data[color_col].unique()
            color_map = {
                cat: ColorSchemes.PRIMARY['main'] if i == 0 else 
                     ColorSchemes.SUCCESS['main'] if i == 1 else
                     ColorSchemes.WARNING['main']
                for i, cat in enumerate(unique_categories)
            }
            
            for category in unique_categories:
                cat_data = data[data[color_col] == category]
                fig.add_trace(go.Scatter(
                    x=cat_data[x_col],
                    y=cat_data[y_col],
                    mode='lines+markers',
                    name=category,
                    line=dict(color=color_map[category], width=3),
                    marker=dict(size=8, color=color_map[category])
                ))
        else:
            # Single color timeline
            fig.add_trace(go.Scatter(
                x=data[x_col],
                y=data[y_col],
                mode='lines+markers',
                name='Value',
                line=dict(color=ColorSchemes.PRIMARY['main'], width=3),
                marker=dict(size=8, color=ColorSchemes.PRIMARY['main'])
            ))
        
        fig.update_layout(
            title=title,
            xaxis_title=x_col.title(),
            yaxis_title=y_col.title(),
            height=400,
            font={'family': 'Inter, sans-serif'},
            plot_bgcolor='rgba(248,249,250,0.8)',
            paper_bgcolor='rgba(0,0,0,0)',
            hovermode='x unified'
        )
        
        return fig

# ==================== DATA DISPLAY COMPONENTS ====================

class DataDisplayComponents:
    """Components for displaying data with enhanced styling"""
    
    @staticmethod
    def create_enhanced_dataframe(
        df: pd.DataFrame,
        title: Optional[str] = None,
        highlight_cols: Optional[List[str]] = None,
        format_cols: Optional[Dict[str, str]] = None,
        height: int = 400
    ):
        """Create enhanced dataframe display"""
        
        if title:
            st.markdown(f"""
            <div style='background: {ColorSchemes.GRADIENTS["primary"]}; 
                        color: white; 
                        padding: 1rem 2rem; 
                        border-radius: 15px 15px 0 0; 
                        margin-bottom: 0;'>
                <h4 style='margin: 0; font-size: 1.2rem; font-weight: 600;'>{title}</h4>
            </div>
            """, unsafe_allow_html=True)
        
        # Apply formatting if specified
        display_df = df.copy()
        if format_cols:
            for col, fmt in format_cols.items():
                if col in display_df.columns:
                    if fmt == 'percentage':
                        display_df[col] = display_df[col].apply(lambda x: f"{x:.2%}" if pd.notnull(x) else "N/A")
                    elif fmt == 'currency':
                        display_df[col] = display_df[col].apply(lambda x: f"₹{x:,.2f}" if pd.notnull(x) else "N/A")
                    elif fmt == 'number':
                        display_df[col] = display_df[col].apply(lambda x: f"{x:,.2f}" if pd.notnull(x) else "N/A")
        
        # Custom CSS for enhanced styling
        st.markdown("""
        <style>
        .enhanced-dataframe {
            border-radius: 0 0 15px 15px;
            overflow: hidden;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        .enhanced-dataframe .dataframe th {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            font-weight: 600;
            text-align: center;
            padding: 1rem 0.8rem;
        }
        .enhanced-dataframe .dataframe td {
            text-align: center;
            padding: 0.8rem;
            border-bottom: 1px solid #f0f2f6;
        }
        .enhanced-dataframe .dataframe tr:nth-child(even) {
            background-color: #f8f9fa;
        }
        .enhanced-dataframe .dataframe tr:hover {
            background-color: rgba(102, 126, 234, 0.1);
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Display dataframe with custom styling
        st.dataframe(
            display_df,
            use_container_width=True,
            height=height
        )
    
    @staticmethod
    def create_summary_stats_card(
        data: pd.Series,
        title: str,
        color: str = "primary"
    ):
        """Create summary statistics card"""
        
        gradient = ColorSchemes.GRADIENTS.get(color, ColorSchemes.GRADIENTS['primary'])
        
        stats = {
            'Count': len(data.dropna()),
            'Mean': data.mean(),
            'Median': data.median(),
            'Std Dev': data.std(),
            'Min': data.min(),
            'Max': data.max()
        }
        
        stats_html = ""
        for key, value in stats.items():
            if isinstance(value, (int, float)):
                if key == 'Count':
                    formatted_value = f"{value:,}"
                else:
                    formatted_value = f"{value:.3f}"
            else:
                formatted_value = "N/A"
            
            stats_html += f"""
            <div style='display: flex; justify-content: space-between; margin: 0.8rem 0; 
                        padding: 0.5rem; background: rgba(255,255,255,0.1); border-radius: 8px;'>
                <span style='font-weight: 500;'>{key}:</span>
                <span style='font-weight: 700;'>{formatted_value}</span>
            </div>
            """
        
        st.markdown(f"""
        <div style='background: {gradient}; 
                    color: white; 
                    padding: 2rem; 
                    border-radius: 20px; 
                    margin: 1rem 0;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.2);'>
            <h3 style='margin: 0 0 1.5rem 0; text-align: center; font-size: 1.3rem; font-weight: 600;'>{title}</h3>
            {stats_html}
        </div>
        """, unsafe_allow_html=True)

# ==================== LAYOUT COMPONENTS ====================

class LayoutComponents:
    """Layout and structural components"""
    
    @staticmethod
    def create_hero_section(
        title: str,
        subtitle: str,
        description: Optional[str] = None,
        background_gradient: str = "primary"
    ):
        """Create hero section with gradient background"""
        
        gradient = ColorSchemes.GRADIENTS.get(background_gradient, ColorSchemes.GRADIENTS['primary'])
        description_html = f"<p style='font-size: 1.1rem; opacity: 0.9; line-height: 1.6; margin: 1.5rem 0;'>{description}</p>" if description else ""
        
        st.markdown(f"""
        <div style='background: {gradient}; 
                    color: white; 
                    padding: 4rem 2rem; 
                    border-radius: 25px; 
                    text-align: center; 
                    margin: 2rem 0;
                    box-shadow: 0 15px 40px rgba(0,0,0,0.2);'>
            <h1 style='margin: 0; font-size: 3.2rem; font-weight: 700; text-shadow: 0 2px 10px rgba(0,0,0,0.3);'>{title}</h1>
            <h2 style='margin: 1rem 0; font-size: 1.4rem; font-weight: 400; opacity: 0.95;'>{subtitle}</h2>
            {description_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_section_header(
        title: str,
        description: Optional[str] = None,
        icon: Optional[str] = None,
        color: str = "primary"
    ):
        """Create section headers with consistent styling"""
        
        gradient = ColorSchemes.GRADIENTS.get(color, ColorSchemes.GRADIENTS['primary'])
        icon_html = f"<span style='margin-right: 1rem; font-size: 1.8rem;'>{icon}</span>" if icon else ""
        description_html = f"<p style='margin: 1rem 0 0 0; opacity: 0.9; font-size: 1.05rem;'>{description}</p>" if description else ""
        
        st.markdown(f"""
        <div style='background: {gradient}; 
                    color: white; 
                    padding: 2rem; 
                    border-radius: 20px; 
                    margin: 2rem 0; 
                    text-align: center;
                    box-shadow: 0 8px 25px rgba(0,0,0,0.1);'>
            <h2 style='margin: 0; font-size: 2rem; font-weight: 600; display: flex; align-items: center; justify-content: center;'>
                {icon_html}{title}
            </h2>
            {description_html}
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def create_divider(style: str = "gradient", height: str = "3px", margin: str = "2rem 0"):
        """Create styled dividers"""
        
        if style == "gradient":
            background = ColorSchemes.GRADIENTS['primary']
        elif style == "dashed":
            background = f"repeating-linear-gradient(90deg, {ColorSchemes.PRIMARY['main']}, {ColorSchemes.PRIMARY['main']} 10px, transparent 10px, transparent 20px)"
        else:
            background = ColorSchemes.PRIMARY['main']
        
        st.markdown(f"""
        <div style='height: {height}; 
                    background: {background}; 
                    margin: {margin}; 
                    border-radius: 2px;
                    box-shadow: 0 2px 5px rgba(0,0,0,0.1);'></div>
        """, unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================

def apply_custom_css():
    """Apply custom CSS for enhanced components"""
    
    st.markdown("""
    <style>
    /* Enhanced component animations */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.7; }
        100% { opacity: 1; }
    }
    
    /* Hover effects for cards */
    .metric-card:hover {
        transform: translateY(-8px) scale(1.02);
        box-shadow: 0 20px 40px rgba(0,0,0,0.3);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    /* Enhanced button styles */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        border-radius: 25px;
        color: white;
        font-weight: 600;
        padding: 0.8rem 2.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 8px 20px rgba(102, 126, 234, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.5);
    }
    
    /* Loading animation */
    .loading {
        animation: pulse 2s infinite;
    }
    
    /* Enhanced tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 25px 25px 0 0;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        border-radius: 20px;
        margin: 0.2rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.25);
        color: white;
        font-weight: 600;
    }
    
    /* Enhanced selectbox */
    .stSelectbox > div > div > select {
        border-radius: 15px;
        border: 2px solid #e9ecef;
        background: linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div > select:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1);
    }
    
    /* Enhanced metrics */
    .stMetric {
        background: linear-gradient(135deg, #f8f9fa 0%, #ffffff 100%);
        border-radius: 15px;
        padding: 1.5rem;
        box-shadow: 0 5px 15px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
    }
    
    .stMetric:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(0,0,0,0.15);
    }
    </style>
    """, unsafe_allow_html=True)

def get_component_theme(theme_name: str = "default") -> Dict[str, Any]:
    """Get component theme configuration"""
    
    themes = {
        "default": {
            "primary_gradient": ColorSchemes.GRADIENTS['primary'],
            "success_color": ColorSchemes.SUCCESS['main'],
            "warning_color": ColorSchemes.WARNING['main'],
            "danger_color": ColorSchemes.DANGER['main'],
            "font_family": "Inter, sans-serif",
            "border_radius": "15px"
        },
        "dark": {
            "primary_gradient": "linear-gradient(135deg, #2c3e50 0%, #34495e 100%)",
            "success_color": "#27ae60",
            "warning_color": "#f39c12",
            "danger_color": "#e74c3c",
            "font_family": "Inter, sans-serif",
            "border_radius": "10px"
        },
        "modern": {
            "primary_gradient": "linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
            "success_color": "#00d4aa",
            "warning_color": "#ff9500",
            "danger_color": "#ff3b30",
            "font_family": "SF Pro Display, Inter, sans-serif",
            "border_radius": "20px"
        }
    }
    
    return themes.get(theme_name, themes["default"])

# ==================== EXPORT ====================

__all__ = [
    'ColorSchemes',
    'EnhancedComponents',
    'ChartComponents', 
    'DataDisplayComponents',
    'LayoutComponents',
    'apply_custom_css',
    'get_component_theme'
]

# ==================== DEMO USAGE ====================

if __name__ == "__main__":
    st.set_page_config(page_title="Enhanced UI Components Demo", layout="wide")
    
    # Apply custom CSS
    apply_custom_css()
    
    # Demo the components
    LayoutComponents.create_hero_section(
        "Enhanced UI Components",
        "Professional Streamlit components for AI Stock Advisor Pro",
        "Showcase of enhanced UI components with professional styling"
    )
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        EnhancedComponents.create_metric_card(
            "Total Return", 
            "23.4%", 
            delta="vs 15.2% benchmark",
            color="success",
            icon="📈"
        )
    
    with col2:
        EnhancedComponents.create_metric_card(
            "Sharpe Ratio", 
            "1.87", 
            delta="Excellent performance",
            color="primary",
            icon="⚡"
        )
    
    with col3:
        EnhancedComponents.create_metric_card(
            "Max Drawdown", 
            "-8.3%", 
            delta="Within acceptable limits",
            color="warning",
            icon="🛡️"
        )
    
    # Demo progress indicator
    EnhancedComponents.create_progress_indicator(
        ["Data Loading", "Feature Engineering", "Model Training", "Analysis", "Results"],
        2,
        "Analysis Progress"
    )
    
    # Demo status indicators
    col1, col2 = st.columns(2)
    
    with col1:
        EnhancedComponents.create_status_indicator(
            "success",
            "Analysis completed successfully",
            "All models trained and predictions generated"
        )
    
    with col2:
        EnhancedComponents.create_status_indicator(
            "warning",
            "Some models showed low confidence",
            "Consider reviewing prediction thresholds"
        )
    
    st.write("Enhanced UI Components Demo Complete!")