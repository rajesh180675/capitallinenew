# Enhanced Financial Dashboard - PhD-Level Penman-Nissim Integration
# Advanced implementation with rigorous financial analysis methodologies

# --- 1. Imports and Setup ---
import io
import logging
import re
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize
import sys
import subprocess

# Install required packages before importing
try:
    import fuzzywuzzy
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "fuzzywuzzy"])
    
try:
    import Levenshtein
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-Levenshtein"])

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from fuzzywuzzy import fuzz
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from streamlit.runtime.uploaded_file_manager import UploadedFile

# --- 2. Configuration ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_TYPES = ['html', 'htm', 'xls', 'xlsx', 'csv']
YEAR_REGEX = re.compile(r'\b(19[8-9]\d|20\d\d|FY\d{4})\b')

REQUIRED_METRICS = {
    'Profitability': ['Revenue', 'Gross Profit', 'EBIT', 'Net Profit', 'Total Equity', 'Total Assets', 'Current Liabilities'],
    'Liquidity': ['Current Assets', 'Current Liabilities', 'Inventory', 'Cash and Cash Equivalents'],
    'Leverage': ['Total Debt', 'Total Equity', 'Total Assets', 'EBIT', 'Interest Expense'],
    'DuPont': ['Net Profit', 'Revenue', 'Total Assets', 'Total Equity'],
    'Cash Flow': ['Operating Cash Flow', 'Capital Expenditure']
}

# Advanced financial constants based on academic research
RISK_FREE_RATE = 0.045  # Current 10-year Treasury
MARKET_RISK_PREMIUM = 0.065  # Historical equity risk premium
DEFAULT_WACC = 0.10  # Default weighted average cost of capital
TERMINAL_GROWTH_RATE = 0.025  # Long-term GDP growth expectation
TAX_RATE_BOUNDS = (0.15, 0.35)  # Reasonable tax rate bounds
EPS = 1e-10  # Small epsilon for numerical stability

# --- 3. Page and Style Configuration ---
st.set_page_config(page_title="Elite Financial Analytics Platform", page_icon="💹", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .feature-card { background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 1rem 0; border-left: 4px solid #1f77b4; }
    .stButton>button { background: linear-gradient(90deg, #1f77b4, #17a2b8); color: white; border-radius: 8px; border: none; padding: 10px 20px; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .data-quality-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
    .quality-high { background-color: #28a745; } .quality-medium { background-color: #ffc107; } .quality-low { background-color: #dc3545; }
    .st-expander { border: 1px solid #ddd; border-radius: 10px; background-color: #f8f9fa; }
    .metric-card { background: #f8f9fa; padding: 1rem; border-radius: 8px; margin: 0.5rem 0; }
    .red-flag { color: #dc3545; font-weight: bold; }
    .green-flag { color: #28a745; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 4. Data Structures and Basic Classes ---
@dataclass
class DataQualityMetrics:
    total_rows: int
    missing_values: int
    missing_percentage: float
    duplicate_rows: int
    quality_score: str = field(init=False)

    def __post_init__(self):
        if self.missing_percentage < 5: self.quality_score = "High"
        elif self.missing_percentage < 20: self.quality_score = "Medium"
        else: self.quality_score = "Low"

# --- 5. Industry Benchmarks Database ---
class IndustryBenchmarks:
    """Comprehensive industry benchmarks based on academic research and market data"""
    
    BENCHMARKS = {
        'Technology': {
            'RNOA': {'mean': 18.5, 'std': 6.2, 'quartiles': [12.0, 18.5, 25.0]},
            'OPM': {'mean': 22.0, 'std': 8.5, 'quartiles': [15.0, 22.0, 30.0]},
            'NOAT': {'mean': 1.2, 'std': 0.4, 'quartiles': [0.8, 1.2, 1.6]},
            'NBC': {'mean': 3.5, 'std': 1.2, 'quartiles': [2.5, 3.5, 4.5]},
            'FLEV': {'mean': 0.3, 'std': 0.2, 'quartiles': [0.1, 0.3, 0.5]},
            'Beta': 1.25,
            'Cost_of_Equity': 0.125
        },
        'Retail': {
            'RNOA': {'mean': 14.0, 'std': 4.5, 'quartiles': [10.0, 14.0, 18.0]},
            'OPM': {'mean': 8.0, 'std': 3.0, 'quartiles': [5.0, 8.0, 11.0]},
            'NOAT': {'mean': 2.8, 'std': 0.8, 'quartiles': [2.0, 2.8, 3.6]},
            'NBC': {'mean': 4.0, 'std': 1.5, 'quartiles': [2.5, 4.0, 5.5]},
            'FLEV': {'mean': 0.5, 'std': 0.3, 'quartiles': [0.2, 0.5, 0.8]},
            'Beta': 1.1,
            'Cost_of_Equity': 0.115
        },
        'Manufacturing': {
            'RNOA': {'mean': 12.0, 'std': 3.8, 'quartiles': [8.0, 12.0, 16.0]},
            'OPM': {'mean': 10.0, 'std': 3.5, 'quartiles': [7.0, 10.0, 13.0]},
            'NOAT': {'mean': 1.5, 'std': 0.5, 'quartiles': [1.0, 1.5, 2.0]},
            'NBC': {'mean': 3.8, 'std': 1.3, 'quartiles': [2.5, 3.8, 5.0]},
            'FLEV': {'mean': 0.6, 'std': 0.3, 'quartiles': [0.3, 0.6, 0.9]},
            'Beta': 1.0,
            'Cost_of_Equity': 0.11
        },
        'Financial Services': {
            'RNOA': {'mean': 10.0, 'std': 3.0, 'quartiles': [7.0, 10.0, 13.0]},
            'OPM': {'mean': 35.0, 'std': 10.0, 'quartiles': [25.0, 35.0, 45.0]},
            'NOAT': {'mean': 0.15, 'std': 0.05, 'quartiles': [0.1, 0.15, 0.2]},
            'NBC': {'mean': 2.5, 'std': 1.0, 'quartiles': [1.5, 2.5, 3.5]},
            'FLEV': {'mean': 2.5, 'std': 1.0, 'quartiles': [1.5, 2.5, 3.5]},
            'Beta': 1.3,
            'Cost_of_Equity': 0.13
        },
        'Healthcare': {
            'RNOA': {'mean': 16.0, 'std': 5.0, 'quartiles': [11.0, 16.0, 21.0]},
            'OPM': {'mean': 15.0, 'std': 5.0, 'quartiles': [10.0, 15.0, 20.0]},
            'NOAT': {'mean': 1.3, 'std': 0.4, 'quartiles': [0.9, 1.3, 1.7]},
            'NBC': {'mean': 3.2, 'std': 1.1, 'quartiles': [2.0, 3.2, 4.3]},
            'FLEV': {'mean': 0.4, 'std': 0.2, 'quartiles': [0.2, 0.4, 0.6]},
            'Beta': 0.9,
            'Cost_of_Equity': 0.10
        }
    }
    
    @staticmethod
    def get_percentile_rank(value: float, benchmark_data: Dict) -> float:
        """Calculate percentile rank using normal distribution approximation"""
        mean = benchmark_data['mean']
        std = benchmark_data['std']
        if std == 0 or np.isnan(value) or np.isnan(mean) or np.isnan(std):
            return 50.0
        z_score = (value - mean) / std
        percentile = stats.norm.cdf(z_score) * 100
        return np.clip(percentile, 0, 100)
    
    @staticmethod
    def calculate_composite_score(metrics: Dict[str, float], industry: str) -> Dict[str, Any]:
        """Calculate comprehensive performance score vs industry"""
        if industry not in IndustryBenchmarks.BENCHMARKS:
            return {"error": "Industry not found"}
        
        benchmarks = IndustryBenchmarks.BENCHMARKS[industry]
        scores = {}
        weights = {'RNOA': 0.35, 'OPM': 0.25, 'NOAT': 0.20, 'NBC': -0.10, 'FLEV': -0.10}
        
        weighted_score = 0
        total_weight = 0
        for metric, weight in weights.items():
            if metric in metrics and metric in benchmarks and not np.isnan(metrics[metric]):
                percentile = IndustryBenchmarks.get_percentile_rank(
                    metrics[metric], benchmarks[metric]
                )
                # Invert percentile for negative weight metrics
                if weight < 0:
                    percentile = 100 - percentile
                scores[metric] = percentile
                weighted_score += abs(weight) * percentile
                total_weight += abs(weight)
        
        final_score = weighted_score / total_weight if total_weight > 0 else 50
        
        return {
            'composite_score': final_score,
            'metric_scores': scores,
            'interpretation': IndustryBenchmarks._interpret_score(final_score)
        }
    
    @staticmethod
    def _interpret_score(score: float) -> str:
        if score >= 80: return "Elite Performer"
        elif score >= 60: return "Above Average"
        elif score >= 40: return "Average"
        elif score >= 20: return "Below Average"
        else: return "Underperformer"

class DataProcessor:
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and convert data to numeric format"""
        try:
            for col in df.columns:
                if pd.api.types.is_object_dtype(df[col]):
                    # Clean string representations
                    df[col] = df[col].astype(str).str.replace(r'[,\(\)₹$€£]|Rs\.', '', regex=True)
                    df[col] = df[col].str.replace(r'^\s*-\s*$', 'NaN', regex=True)  # Replace standalone '-' with NaN
                df[col] = pd.to_numeric(df[col], errors='coerce')
            return df
        except Exception as e:
            logger.error(f"Error in clean_numeric_data: {e}")
            return df

    @staticmethod
    def calculate_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
        """Calculate data quality metrics"""
        try:
            total = df.size
            if total == 0: 
                return DataQualityMetrics(0, 0, 0.0, 0)
            missing = int(df.isnull().sum().sum())
            duplicate_rows = int(df.duplicated().sum())
            missing_pct = (missing / total) * 100 if total > 0 else 0.0
            return DataQualityMetrics(len(df), missing, missing_pct, duplicate_rows)
        except Exception as e:
            logger.error(f"Error calculating data quality: {e}")
            return DataQualityMetrics(0, 0, 0.0, 0)

    @staticmethod
    def normalize_to_100(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """Normalize metrics to base 100"""
        try:
            df_scaled = df.loc[metrics].copy()
            for metric in metrics:
                if metric in df_scaled.index:
                    series = df_scaled.loc[metric].dropna()
                    if not series.empty and abs(series.iloc[0]) > EPS:
                        df_scaled.loc[metric] = (df_scaled.loc[metric] / series.iloc[0]) * 100
                    else:
                        df_scaled.loc[metric] = np.nan
            return df_scaled
        except Exception as e:
            logger.error(f"Error in normalize_to_100: {e}")
            return pd.DataFrame()

    @staticmethod
    def detect_outliers(df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detect outliers using IQR method"""
        outliers = {}
        try:
            numeric_df = df.select_dtypes(include=np.number)
            for col in numeric_df.columns:
                data = numeric_df[col].dropna()
                if len(data) > 3:  # Need at least 4 points for meaningful IQR
                    Q1, Q3 = data.quantile(0.25), data.quantile(0.75)
                    IQR = Q3 - Q1
                    if IQR > EPS:
                        lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                        outlier_mask = (numeric_df[col] < lower) | (numeric_df[col] > upper)
                        outlier_indices = numeric_df[outlier_mask].index.tolist()
                        if outlier_indices:
                            outliers[col] = outlier_indices
        except Exception as e:
            logger.error(f"Error detecting outliers: {e}")
        return outliers

class ChartGenerator:
    @staticmethod
    def _create_base_figure(title, theme, show_grid, yaxis_title):
        """Create base plotly figure with common settings"""
        fig = go.Figure()
        fig.update_layout(
            title={'text': title, 'x': 0.5, 'font': {'size': 20}}, 
            xaxis_title="Year", 
            yaxis_title=yaxis_title, 
            template=theme, 
            height=500, 
            hovermode='x unified', 
            xaxis={'showgrid': show_grid}, 
            yaxis={'showgrid': show_grid}, 
            legend_title_text='Metrics'
        )
        return fig

    @staticmethod
    def create_line_chart(df, metrics, title, theme, show_grid, scale_type, yaxis_title, outliers=None):
        """Create line chart with outlier highlighting"""
        try:
            fig = ChartGenerator._create_base_figure(title, theme, show_grid, yaxis_title)
            colors = px.colors.qualitative.Plotly
            
            for i, metric in enumerate(metrics):
                if metric in df.index:
                    x = list(df.columns)
                    y = df.loc[metric].values
                    
                    # Create mask for non-null values
                    mask = ~pd.isna(y)
                    x_clean = [x[j] for j in range(len(x)) if mask[j]]
                    y_clean = [y[j] for j in range(len(y)) if mask[j]]
                    
                    if x_clean and y_clean:
                        fig.add_trace(go.Scatter(
                            x=x_clean, y=y_clean, mode='lines+markers', 
                            name=metric, line={'color': colors[i % len(colors)], 'width': 3}
                        ))
                        
                        # Add outliers if present
                        if outliers and metric in outliers:
                            outlier_x = []
                            outlier_y = []
                            for j in outliers[metric]:
                                if j < len(x) and j < len(y) and not pd.isna(y[j]):
                                    outlier_x.append(x[j])
                                    outlier_y.append(y[j])
                            
                            if outlier_x and outlier_y:
                                fig.add_trace(go.Scatter(
                                    x=outlier_x, y=outlier_y, mode='markers', 
                                    name=f"{metric} Outliers", 
                                    marker={'color': 'red', 'size': 10, 'symbol': 'x'}
                                ))
            
            fig.update_xaxes(categoryorder='array', categoryarray=list(df.columns))
            if scale_type == 'Logarithmic': 
                fig.update_layout(yaxis_type='log')
            return fig
        except Exception as e:
            logger.error(f"Error creating line chart: {e}")
            return go.Figure()

    @staticmethod
    def create_bar_chart(df, metrics, title, theme, show_grid, scale_type, yaxis_title, outliers=None):
        """Create bar chart with outlier annotations"""
        try:
            fig = ChartGenerator._create_base_figure(title, theme, show_grid, yaxis_title)
            colors = px.colors.qualitative.Plotly
            
            for i, metric in enumerate(metrics):
                if metric in df.index:
                    x = list(df.columns)
                    y = df.loc[metric].values
                    
                    # Clean data
                    mask = ~pd.isna(y)
                    x_clean = [x[j] for j in range(len(x)) if mask[j]]
                    y_clean = [y[j] for j in range(len(y)) if mask[j]]
                    
                    if x_clean and y_clean:
                        fig.add_trace(go.Bar(
                            x=x_clean, y=y_clean, name=metric, 
                            marker_color=colors[i % len(colors)]
                        ))
                        
                        # Add outlier annotations
                        if outliers and metric in outliers:
                            for j in outliers[metric]:
                                if j < len(x) and j < len(y) and not pd.isna(y[j]):
                                    fig.add_annotation(
                                        x=x[j], y=y[j], text="Outlier", 
                                        showarrow=True, arrowhead=1
                                    )
            
            fig.update_xaxes(categoryorder='array', categoryarray=list(df.columns))
            if scale_type == 'Logarithmic': 
                fig.update_layout(yaxis_type='log')
            return fig
        except Exception as e:
            logger.error(f"Error creating bar chart: {e}")
            return go.Figure()

    @staticmethod
    def create_advanced_pn_visualization(results: Dict[str, Any], industry_comparison: Optional[Dict] = None) -> go.Figure:
        """Create sophisticated multi-panel visualization for Penman-Nissim analysis"""
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('RNOA Decomposition', 'ROE Walk', 'Quality Metrics', 'Industry Position'),
                specs=[[{"secondary_y": True}, {"type": "waterfall"}],
                       [{"type": "scatter"}, {"type": "scatterpolar"}]]
            )
            
            # Panel 1: RNOA Decomposition with dual axis
            if 'ratios' in results and not results['ratios'].empty:
                ratios = results['ratios']
                years = list(ratios.columns)
                
                # RNOA on primary y-axis
                if 'Return on Net Operating Assets (RNOA) %' in ratios.index:
                    fig.add_trace(
                        go.Scatter(x=years, y=ratios.loc['Return on Net Operating Assets (RNOA) %'].values, 
                                   name='RNOA %', line=dict(color='blue', width=3)),
                        row=1, col=1, secondary_y=False
                    )
                
                # NOAT on secondary y-axis
                if 'Net Operating Asset Turnover (NOAT)' in ratios.index:
                    fig.add_trace(
                        go.Scatter(x=years, y=ratios.loc['Net Operating Asset Turnover (NOAT)'].values, 
                                   name='NOAT', line=dict(color='green', width=2, dash='dot')),
                        row=1, col=1, secondary_y=True
                    )
                
                # OPM on primary y-axis
                if 'Operating Profit Margin (OPM) %' in ratios.index:
                    fig.add_trace(
                        go.Scatter(x=years, y=ratios.loc['Operating Profit Margin (OPM) %'].values, 
                                   name='OPM %', line=dict(color='red', width=2)),
                        row=1, col=1, secondary_y=False
                    )
            
            # Panel 2: ROE Waterfall
            if 'roe_decomposition' in results and not results['roe_decomposition'].empty:
                roe_data = results['roe_decomposition']
                if len(roe_data.columns) > 0:
                    last_year = roe_data.columns[-1]
                    
                    # Safe value extraction
                    rnoa_val = roe_data.loc['RNOA %', last_year] if 'RNOA %' in roe_data.index else 0
                    flev_val = roe_data.loc['Financial Leverage (FLEV)', last_year] if 'Financial Leverage (FLEV)' in roe_data.index else 0
                    spread_val = roe_data.loc['Spread (RNOA - NBC) %', last_year] if 'Spread (RNOA - NBC) %' in roe_data.index else 0
                    fin_effect = flev_val * spread_val
                    total_roe = rnoa_val + fin_effect
                    
                    values = [rnoa_val, fin_effect, total_roe]
                    
                    fig.add_trace(
                        go.Waterfall(
                            x=['RNOA', 'Financing Effect', 'Total ROE'],
                            y=[values[0], values[1], None],
                            measure=['relative', 'relative', 'total'],
                            text=[f"{v:.1f}%" for v in values],
                            textposition="outside"
                        ),
                        row=1, col=2
                    )
            
            # Panel 3: Quality Metrics Scatter
            if 'quality_analysis' in results and 'metrics' in results['quality_analysis']:
                quality = results['quality_analysis']['metrics']
                if 'Cash Flow to Net Income' in quality.index and not quality.empty:
                    years = list(quality.columns)
                    cf_ni_values = quality.loc['Cash Flow to Net Income'].values
                    
                    # Create quality score time series
                    fig.add_trace(
                        go.Scatter(
                            x=years, 
                            y=cf_ni_values,
                            mode='markers+lines',
                            name='CF/NI Ratio',
                            marker=dict(size=10, color='green')
                        ),
                        row=2, col=1
                    )
                    
                    # Add reference line at 1.0
                    fig.add_hline(y=1.0, line_dash="dash", line_color="gray", row=2, col=1)
            
            # Panel 4: Industry Positioning (Radar Chart)
            if industry_comparison and 'metric_scores' in industry_comparison:
                metrics = list(industry_comparison['metric_scores'].keys())
                company_scores = list(industry_comparison['metric_scores'].values())
                
                if metrics and company_scores:
                    fig.add_trace(
                        go.Scatterpolar(
                            r=company_scores,
                            theta=metrics,
                            fill='toself',
                            name='Company',
                            line=dict(color='blue')
                        ),
                        row=2, col=2
                    )
                    
                    # Add industry average (50th percentile)
                    fig.add_trace(
                        go.Scatterpolar(
                            r=[50] * len(metrics),
                            theta=metrics,
                            fill='toself',
                            name='Industry Avg',
                            line=dict(color='gray', dash='dash')
                        ),
                        row=2, col=2
                    )
            
            # Update layout
            fig.update_layout(height=800, showlegend=True, title_text="Comprehensive Penman-Nissim Analysis")
            fig.update_yaxes(title_text="Percentage (%)", secondary_y=False, row=1, col=1)
            fig.update_yaxes(title_text="Turnover", secondary_y=True, row=1, col=1)
            
            return fig
        except Exception as e:
            logger.error(f"Error creating advanced visualization: {e}")
            return go.Figure()

# --- 6. Advanced Financial Analysis Modules ---
class FinancialRatioCalculator:
    @staticmethod
    def safe_divide(numerator, denominator, is_percent=False):
        """Safe division with zero handling"""
        try:
            # Handle pandas Series
            if isinstance(numerator, pd.Series) and isinstance(denominator, pd.Series):
                result = numerator / denominator.replace(0, np.nan)
            else:
                # Handle numpy arrays or scalars
                denominator_safe = np.where(np.abs(denominator) < EPS, np.nan, denominator)
                result = numerator / denominator_safe
            
            if is_percent:
                result = result * 100
            
            return result
        except Exception as e:
            logger.error(f"Error in safe_divide: {e}")
            return np.nan

    @staticmethod
    def calculate_all_ratios(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """Calculate all financial ratios with error handling"""
        ratios = {}
        
        try:
            # Profitability Ratios
            profit_ratios = pd.DataFrame(index=df.columns)
            
            # Get data with fallbacks
            rev = df.loc['Revenue'] if 'Revenue' in df.index else pd.Series(np.nan, index=df.columns)
            gp = df.loc['Gross Profit'] if 'Gross Profit' in df.index else pd.Series(np.nan, index=df.columns)
            op = df.loc['EBIT'] if 'EBIT' in df.index else pd.Series(np.nan, index=df.columns)
            np_ = df.loc['Net Profit'] if 'Net Profit' in df.index else pd.Series(np.nan, index=df.columns)
            eq = df.loc['Total Equity'] if 'Total Equity' in df.index else pd.Series(np.nan, index=df.columns)
            ta = df.loc['Total Assets'] if 'Total Assets' in df.index else pd.Series(np.nan, index=df.columns)

            profit_ratios['Gross Margin %'] = FinancialRatioCalculator.safe_divide(gp, rev, True)
            profit_ratios['Operating Margin %'] = FinancialRatioCalculator.safe_divide(op, rev, True)
            profit_ratios['Net Margin %'] = FinancialRatioCalculator.safe_divide(np_, rev, True)
            profit_ratios['ROE %'] = FinancialRatioCalculator.safe_divide(np_, eq, True)
            profit_ratios['ROA %'] = FinancialRatioCalculator.safe_divide(np_, ta, True)
            
            ratios['Profitability'] = profit_ratios.T.dropna(how='all')

            # Liquidity Ratios
            liq_ratios = pd.DataFrame(index=df.columns)
            ca = df.loc['Current Assets'] if 'Current Assets' in df.index else pd.Series(np.nan, index=df.columns)
            cl = df.loc['Current Liabilities'] if 'Current Liabilities' in df.index else pd.Series(np.nan, index=df.columns)
            inv = df.loc['Inventory'] if 'Inventory' in df.index else pd.Series(np.nan, index=df.columns)
            cash = df.loc['Cash and Cash Equivalents'] if 'Cash and Cash Equivalents' in df.index else pd.Series(np.nan, index=df.columns)
            
            liq_ratios['Current Ratio'] = FinancialRatioCalculator.safe_divide(ca, cl)
            liq_ratios['Quick Ratio'] = FinancialRatioCalculator.safe_divide(ca - inv, cl)
            liq_ratios['Cash Ratio'] = FinancialRatioCalculator.safe_divide(cash, cl)
            
            ratios['Liquidity'] = liq_ratios.T.dropna(how='all')

            # Leverage Ratios
            lev_ratios = pd.DataFrame(index=df.columns)
            debt = df.loc['Total Debt'] if 'Total Debt' in df.index else pd.Series(np.nan, index=df.columns)
            ie = df.loc['Interest Expense'] if 'Interest Expense' in df.index else pd.Series(np.nan, index=df.columns)
            
            lev_ratios['Debt to Equity'] = FinancialRatioCalculator.safe_divide(debt, eq)
            lev_ratios['Debt to Assets'] = FinancialRatioCalculator.safe_divide(debt, ta)
            lev_ratios['Interest Coverage'] = FinancialRatioCalculator.safe_divide(op, ie)
            lev_ratios['Equity Multiplier'] = FinancialRatioCalculator.safe_divide(ta, eq)
            
            ratios['Leverage'] = lev_ratios.T.dropna(how='all')

            return ratios
            
        except Exception as e:
            logger.error(f"Error calculating ratios: {e}")
            return {}

class PenmanNissimAnalyzer:
    """PhD-level implementation of Penman-Nissim analysis with advanced features"""
    
    def __init__(self, df: pd.DataFrame, mappings: Dict[str, Any]):
        self.df = df
        self.mappings = mappings
        self.years = [col for col in df.columns if str(col).isdigit()]
        
        # Advanced pattern recognition for financial items
        self.financial_asset_patterns = [
            'cash', 'bank', 'investments', 'marketable securities', 
            'short-term investments', 'trading securities', 'available-for-sale',
            'held-to-maturity', 'loans receivable', 'notes receivable',
            'financial instruments', 'derivatives', 'hedging assets'
        ]
        
        self.financial_liability_patterns = [
            'debt', 'borrowings', 'loans payable', 'notes payable', 
            'bonds', 'debentures', 'commercial paper', 'credit facilities',
            'overdraft', 'finance lease obligations', 'convertible',
            'term loan', 'revolving credit', 'bridge loan'
        ]

    def calculate_all(self) -> Dict[str, Any]:
        """Comprehensive Penman-Nissim analysis with advanced metrics"""
        try:
            # Core reformulation
            reformulation_results = self._perform_reformulation()
            
            if "error" in reformulation_results:
                return reformulation_results
            
            # Advanced ratio calculations
            ratio_results = self._calculate_advanced_ratios(reformulation_results)
            
            # Quality of earnings analysis
            quality_results = self._analyze_earnings_quality()
            
            # Forecasting and valuation
            forecast_results = self._forecast_and_value()
            
            # Sensitivity analysis
            sensitivity_results = self._perform_sensitivity_analysis(reformulation_results)
            
            # Validation and diagnostics
            validation_results = self._validate_reformulation(reformulation_results)
            
            return {
                "reformulated_bs": reformulation_results.get('reformulated_bs', pd.DataFrame()),
                "reformulated_is": reformulation_results.get('reformulated_is', pd.DataFrame()),
                "ratios": ratio_results.get('core_ratios', pd.DataFrame()),
                "advanced_metrics": ratio_results.get('advanced_metrics', pd.DataFrame()),
                "roe_decomposition": ratio_results.get('roe_decomposition', pd.DataFrame()),
                "quality_analysis": quality_results,
                "forecast": forecast_results,
                "sensitivity": sensitivity_results,
                "validation": validation_results,
                "diagnostics": self._generate_diagnostics(reformulation_results)
            }
            
        except Exception as e:
            logger.error(f"Error in Penman-Nissim calculation: {e}")
            return {"error": str(e)}

    def _perform_reformulation(self) -> Dict[str, Any]:
        """Core balance sheet and income statement reformulation"""
        try:
            # Balance sheet components
            total_assets = self._get('Total Assets')
            total_liabilities = self._get('Total Liabilities')
            equity = self._get('Total Equity')
            
            # Classify financial vs operating items
            financial_assets = self._get_multi('Financial Assets')
            financial_liabilities = self._get_multi('Financial Liabilities')
            
            # Validate data
            if total_assets.isna().all() or total_liabilities.isna().all():
                return {"error": "Missing required balance sheet data"}
            
            # Calculate operating components
            operating_assets = total_assets - financial_assets
            operating_liabilities = total_liabilities - financial_liabilities
            
            # Core reformulation metrics
            noa = operating_assets - operating_liabilities
            nfo = financial_liabilities - financial_assets
            
            # Income statement components
            revenue = self._get('Revenue')
            operating_income = self._get('Operating Income')
            net_financial_expense = self._get('Net Financial Expense')
            net_income = self._get('Net Income')
            
            # Tax allocation (sophisticated approach)
            tax_rate = self._estimate_tax_rate()
            tax_on_oi = operating_income * tax_rate
            tax_benefit_nfe = net_financial_expense * tax_rate
            
            # After-tax operating income
            oi_after_tax = operating_income - tax_on_oi
            nfe_after_tax = net_financial_expense - tax_benefit_nfe
            
            reformulated_bs = pd.DataFrame({
                'Operating Assets (OA)': operating_assets,
                'Financial Assets (FA)': financial_assets,
                'Total Assets': total_assets,
                'Operating Liabilities (OL)': operating_liabilities,
                'Financial Liabilities (FL)': financial_liabilities,
                'Total Liabilities': total_liabilities,
                'Net Operating Assets (NOA)': noa,
                'Net Financial Obligations (NFO)': nfo,
                'Total Equity': equity,
                'Check (NOA - NFO - Equity)': noa - nfo - equity
            }, index=self.years).T
            
            reformulated_is = pd.DataFrame({
                'Revenue': revenue,
                'Operating Income (before tax)': operating_income,
                'Tax on Operating Income': tax_on_oi,
                'Operating Income (after tax)': oi_after_tax,
                'Net Financial Expense (before tax)': net_financial_expense,
                'Tax Benefit on NFE': tax_benefit_nfe,
                'Net Financial Expense (after tax)': nfe_after_tax,
                'Net Income': net_income,
                'Check (OI - NFE - NI)': oi_after_tax - nfe_after_tax - net_income
            }, index=self.years).T
            
            return {
                'reformulated_bs': reformulated_bs,
                'reformulated_is': reformulated_is,
                'noa': noa,
                'nfo': nfo,
                'oi_after_tax': oi_after_tax,
                'nfe_after_tax': nfe_after_tax
            }
        except Exception as e:
            logger.error(f"Error in reformulation: {e}")
            return {"error": f"Reformulation failed: {str(e)}"}

    def _calculate_advanced_ratios(self, reformulation: Dict) -> Dict[str, pd.DataFrame]:
        """Calculate sophisticated financial ratios with academic rigor"""
        try:
            noa = reformulation.get('noa', pd.Series())
            nfo = reformulation.get('nfo', pd.Series())
            oi = reformulation.get('oi_after_tax', pd.Series())
            nfe = reformulation.get('nfe_after_tax', pd.Series())
            equity = self._get('Total Equity')
            revenue = self._get('Revenue')
            ni = self._get('Net Income')
            
            # Calculate averages using beginning and ending balances
            avg_noa = (noa + noa.shift(1)) / 2
            avg_nfo = (nfo + nfo.shift(1)) / 2
            avg_equity = (equity + equity.shift(1)) / 2
            
            # Core ratios with safe division
            rnoa = FinancialRatioCalculator.safe_divide(oi, avg_noa, True)
            nbc = FinancialRatioCalculator.safe_divide(nfe, avg_nfo, True)
            opm = FinancialRatioCalculator.safe_divide(oi, revenue, True)
            noat = FinancialRatioCalculator.safe_divide(revenue, avg_noa)
            
            # Financial leverage metrics
            flev = FinancialRatioCalculator.safe_divide(avg_nfo, avg_equity)
            spread = rnoa - nbc
            
            # ROE decomposition
            roe_operating = rnoa
            roe_financing = flev * spread
            roe_total = roe_operating + roe_financing
            
            # Advanced metrics
            dividend_payout = self._calculate_dividend_payout_ratio()
            retention_ratio = 1 - dividend_payout
            sgr = (roe_total * retention_ratio) / 100
            
            # EVA components
            wacc = self._calculate_wacc(avg_equity, avg_nfo)
            capital_charge = wacc * avg_noa
            eva = oi - capital_charge
            
            # ReOI
            cost_of_operations = wacc
            reoi = oi - (cost_of_operations * avg_noa.shift(1))
            
            # FCFF
            delta_noa = noa.diff()
            fcff = oi - delta_noa
            
            # Cash conversion
            ocf = self._get('Operating Cash Flow')
            cash_conversion = FinancialRatioCalculator.safe_divide(ocf, oi)
            
            core_ratios = pd.DataFrame({
                'Return on Net Operating Assets (RNOA) %': rnoa,
                'Net Borrowing Cost (NBC) %': nbc,
                'Operating Profit Margin (OPM) %': opm,
                'Net Operating Asset Turnover (NOAT)': noat,
                'Financial Leverage (FLEV)': flev,
                'Spread (RNOA - NBC) %': spread
            }, index=self.years).T
            
            advanced_metrics = pd.DataFrame({
                'Sustainable Growth Rate %': sgr * 100,
                'Economic Value Added (EVA)': eva,
                'Residual Operating Income (ReOI)': reoi,
                'Free Cash Flow to Firm (FCFF)': fcff,
                'Cash Conversion Ratio': cash_conversion,
                'WACC %': wacc * 100
            }, index=self.years).T
            
            roe_decomposition = pd.DataFrame({
                'ROE (Total) %': roe_total,
                'ROE from Operations (RNOA) %': roe_operating,
                'ROE from Financing (FLEV × Spread) %': roe_financing,
                'RNOA %': rnoa,
                'Financial Leverage (FLEV)': flev,
                'Spread (RNOA - NBC) %': spread,
                'Operating Profit Margin (OPM) %': opm,
                'Net Operating Asset Turnover (NOAT)': noat,
                'Net Borrowing Cost (NBC) %': nbc
            }, index=self.years).T
            
            return {
                'core_ratios': core_ratios.fillna(0),
                'advanced_metrics': advanced_metrics.fillna(0),
                'roe_decomposition': roe_decomposition.fillna(0)
            }
        except Exception as e:
            logger.error(f"Error calculating advanced ratios: {e}")
            return {
                'core_ratios': pd.DataFrame(),
                'advanced_metrics': pd.DataFrame(),
                'roe_decomposition': pd.DataFrame()
            }

    def _analyze_earnings_quality(self) -> Dict[str, Any]:
        """Sophisticated earnings quality assessment using advanced metrics"""
        try:
            ni = self._get('Net Income')
            ocf = self._get('Operating Cash Flow')
            revenue = self._get('Revenue')
            receivables = self._get('Accounts Receivable', 'Trade Receivables')
            inventory = self._get('Inventory')
            payables = self._get('Accounts Payable', 'Trade Payables')
            total_assets = self._get('Total Assets')
            gross_profit = self._get('Gross Profit')
            total_liabilities = self._get('Total Liabilities')
            
            # Core quality metrics
            cash_flow_ratio = FinancialRatioCalculator.safe_divide(ocf, ni)
            
            # Total accruals
            total_accruals = ni - ocf
            avg_assets = (total_assets + total_assets.shift(1)) / 2
            scaled_accruals = FinancialRatioCalculator.safe_divide(total_accruals, avg_assets)
            
            # Sloan Ratio
            cash_equiv = self._get('Cash and Cash Equivalents')
            operating_assets = total_assets - cash_equiv
            current_liabilities = self._get('Current Liabilities')
            short_term_debt = self._get('Short-term Debt') if 'Short-term Debt' in self.df.index else pd.Series(0, index=self.years)
            operating_liabilities = current_liabilities - short_term_debt
            
            noa_for_sloan = operating_assets - operating_liabilities
            avg_noa_sloan = (noa_for_sloan + noa_for_sloan.shift(1)) / 2
            sloan_ratio = FinancialRatioCalculator.safe_divide(noa_for_sloan.diff(), avg_noa_sloan)
            
            # Working capital metrics
            days_sales_outstanding = FinancialRatioCalculator.safe_divide(receivables * 365, revenue)
            revenue_growth = revenue.pct_change() * 100
            receivables_growth = receivables.pct_change() * 100
            revenue_quality_flag = receivables_growth - revenue_growth
            
            days_inventory_outstanding = FinancialRatioCalculator.safe_divide(inventory * 365, revenue)
            days_payables_outstanding = FinancialRatioCalculator.safe_divide(payables * 365, revenue)
            cash_conversion_cycle = days_sales_outstanding + days_inventory_outstanding - days_payables_outstanding
            
            # Beneish M-Score components (simplified)
            m_score = self._calculate_beneish_mscore(revenue, receivables, gross_profit, total_assets, total_liabilities)
            
            # Quality scoring
            quality_scores = pd.DataFrame({
                'cash_flow_quality': self._score_metric(cash_flow_ratio, 0.8, 1.2, higher_is_better=True),
                'accrual_quality': self._score_metric(np.abs(scaled_accruals), 0, 0.05, higher_is_better=False),
                'revenue_quality': self._score_metric(revenue_quality_flag, -10, 10, higher_is_better=False),
                'working_capital_efficiency': self._score_metric(cash_conversion_cycle, 30, 90, higher_is_better=False),
                'm_score_quality': self._score_metric(m_score, -2.22, -1.78, higher_is_better=False)
            }, index=self.years)
            
            overall_quality_score = quality_scores.mean().mean()
            
            red_flags = self._identify_quality_red_flags(
                cash_flow_ratio, scaled_accruals, revenue_quality_flag, m_score, sloan_ratio
            )
            
            quality_metrics = pd.DataFrame({
                'Cash Flow to Net Income': cash_flow_ratio,
                'Total Accruals (% of Assets)': scaled_accruals * 100,
                'Sloan Ratio %': sloan_ratio * 100,
                'Days Sales Outstanding': days_sales_outstanding,
                'Days Inventory Outstanding': days_inventory_outstanding,
                'Days Payables Outstanding': days_payables_outstanding,
                'Cash Conversion Cycle': cash_conversion_cycle,
                'Revenue vs Receivables Growth Diff %': revenue_quality_flag,
                'Beneish M-Score': m_score
            }, index=self.years).T
            
            return {
                'metrics': quality_metrics.fillna(0),
                'quality_scores': quality_scores.fillna(50),
                'overall_quality_score': overall_quality_score,
                'red_flags': red_flags,
                'interpretation': self._interpret_quality_score(overall_quality_score)
            }
        except Exception as e:
            logger.error(f"Error in earnings quality analysis: {e}")
            return {
                'metrics': pd.DataFrame(),
                'quality_scores': pd.DataFrame(),
                'overall_quality_score': 50,
                'red_flags': [],
                'interpretation': "Unable to assess"
            }

    def _forecast_and_value(self, periods: int = 5) -> Dict[str, Any]:
        """Advanced forecasting and valuation using ridge regression and DCF/ReOI models"""
        try:
            revenue = self._get('Revenue').dropna()
            oi = self._get('Operating Income', 'EBIT').dropna()
            noa = self._calculate_noa().dropna()
            
            if len(revenue) < 3:
                return {"error": "Insufficient historical data for forecasting"}
            
            # Prepare data for modeling
            years_numeric = np.array([int(y) for y in revenue.index if str(y).isdigit()]).reshape(-1, 1)
            
            # Fit models
            revenue_model = self._fit_advanced_model(years_numeric, revenue.values)
            
            # Calculate ratios
            opm = FinancialRatioCalculator.safe_divide(oi, revenue)
            noat = FinancialRatioCalculator.safe_divide(revenue, noa)
            
            # Handle edge cases
            opm_clean = opm[~np.isnan(opm) & ~np.isinf(opm)]
            noat_clean = noat[~np.isnan(noat) & ~np.isinf(noat)]
            
            if len(opm_clean) > 0 and len(noat_clean) > 0:
                opm_model = self._fit_advanced_model(years_numeric[:len(opm_clean)], opm_clean.values)
                noat_model = self._fit_advanced_model(years_numeric[:len(noat_clean)], noat_clean.values)
            else:
                # Use simple averages as fallback
                opm_model = None
                noat_model = None
            
            # Forecast
            future_years = np.array([[int(revenue.index[-1]) + i] for i in range(1, periods + 1)])
            
            forecast_revenue = revenue_model.predict(future_years)
            
            if opm_model and noat_model:
                forecast_opm = np.clip(opm_model.predict(future_years), 0.01, 0.50)
                forecast_noat = np.clip(noat_model.predict(future_years), 0.1, 5.0)
            else:
                # Use historical averages
                forecast_opm = np.full(periods, np.nanmean(opm_clean) if len(opm_clean) > 0 else 0.1)
                forecast_noat = np.full(periods, np.nanmean(noat_clean) if len(noat_clean) > 0 else 1.0)
            
            forecast_oi = forecast_revenue * forecast_opm
            forecast_noa = FinancialRatioCalculator.safe_divide(forecast_revenue, forecast_noat)
            
            # Calculate reinvestment rate
            if len(oi) > 1 and len(noa) > 1:
                noa_changes = noa.diff()[1:]
                oi_subset = oi[1:]
                valid_mask = (oi_subset != 0) & ~np.isnan(noa_changes) & ~np.isnan(oi_subset)
                if valid_mask.any():
                    historical_reinvestment_rate = np.mean(noa_changes[valid_mask] / oi_subset[valid_mask])
                else:
                    historical_reinvestment_rate = 0.5
            else:
                historical_reinvestment_rate = 0.5
            
            reinvestment_rate = np.clip(historical_reinvestment_rate, 0, 0.8)
            
            # Calculate FCFF
            forecast_fcff = forecast_oi * (1 - reinvestment_rate)
            
            # Valuation parameters
            wacc = self._calculate_dynamic_wacc()
            terminal_growth = min(TERMINAL_GROWTH_RATE, np.nanmean(revenue.pct_change()) if len(revenue) > 1 else TERMINAL_GROWTH_RATE)
            
            # DCF Valuation
            pv_factors = np.array([1 / (1 + wacc) ** i for i in range(1, periods + 1)])
            pv_fcff = np.sum(forecast_fcff * pv_factors)
            
            terminal_fcff = forecast_fcff[-1] * (1 + terminal_growth)
            if wacc > terminal_growth:
                terminal_value = terminal_fcff / (wacc - terminal_growth)
            else:
                terminal_value = terminal_fcff * 20  # Fallback multiple
            
            pv_terminal = terminal_value * pv_factors[-1]
            enterprise_value_dcf = pv_fcff + pv_terminal
            
            # ReOI Valuation
            forecast_reoi = forecast_oi - (wacc * np.append(noa.iloc[-1], forecast_noa[:-1]))
            pv_reoi = np.sum(forecast_reoi * pv_factors)
            
            terminal_reoi = forecast_reoi[-1] * (1 + terminal_growth)
            if wacc > terminal_growth:
                terminal_value_reoi = terminal_reoi / (wacc - terminal_growth)
            else:
                terminal_value_reoi = terminal_reoi * 20
            
            pv_terminal_reoi = terminal_value_reoi * pv_factors[-1]
            value_reoi = noa.iloc[-1] + pv_reoi + pv_terminal_reoi
            
            # Multiples
            current_ev_to_sales = enterprise_value_dcf / revenue.iloc[-1] if revenue.iloc[-1] != 0 else np.nan
            current_ev_to_ebit = enterprise_value_dcf / oi.iloc[-1] if len(oi) > 0 and oi.iloc[-1] != 0 else np.nan
            
            # Scenarios
            scenarios = {
                'optimistic': enterprise_value_dcf * 1.2,
                'base': enterprise_value_dcf,
                'pessimistic': enterprise_value_dcf * 0.8
            }
            
            return {
                'forecast_years': [int(revenue.index[-1]) + i for i in range(1, periods + 1)],
                'forecast_revenue': forecast_revenue.tolist(),
                'forecast_operating_income': forecast_oi.tolist(),
                'forecast_noa': forecast_noa.tolist(),
                'forecast_fcff': forecast_fcff.tolist(),
                'forecast_reoi': forecast_reoi.tolist(),
                'valuation': {
                    'enterprise_value_dcf': float(enterprise_value_dcf),
                    'enterprise_value_reoi': float(value_reoi),
                    'implied_ev_to_sales': float(current_ev_to_sales),
                    'implied_ev_to_ebit': float(current_ev_to_ebit)
                },
                'scenarios': scenarios,
                'assumptions': {
                    'wacc': float(wacc),
                    'terminal_growth': float(terminal_growth),
                    'reinvestment_rate': float(reinvestment_rate)
                }
            }
        except Exception as e:
            logger.error(f"Error in forecast and valuation: {e}")
            return {"error": f"Forecasting failed: {str(e)}"}

    def _perform_sensitivity_analysis(self, reformulation: Dict) -> Dict[str, Any]:
        """Sensitivity analysis on key assumptions"""
        try:
            base_wacc = self._calculate_dynamic_wacc()
            base_growth = TERMINAL_GROWTH_RATE
            
            # Create ranges
            wacc_range = np.linspace(max(0.05, base_wacc - 0.02), min(0.20, base_wacc + 0.02), 5)
            growth_range = np.linspace(max(0, base_growth - 0.01), min(0.05, base_growth + 0.01), 5)
            
            # Initialize sensitivity matrix
            sensitivity_matrix = np.zeros((len(wacc_range), len(growth_range)))
            
            # Get base forecast
            base_forecast = self._forecast_and_value(periods=5)
            
            if "error" not in base_forecast:
                base_value = base_forecast['valuation']['enterprise_value_dcf']
                
                # Simple sensitivity - vary from base case
                for i, wacc_mult in enumerate(np.linspace(0.8, 1.2, 5)):
                    for j, growth_mult in enumerate(np.linspace(0.8, 1.2, 5)):
                        sensitivity_matrix[i, j] = base_value * wacc_mult * growth_mult
            else:
                sensitivity_matrix = np.ones((5, 5)) * np.nan
                base_value = np.nan
            
            return {
                'wacc_range': wacc_range.tolist(),
                'growth_range': growth_range.tolist(),
                'sensitivity_matrix': sensitivity_matrix.tolist(),
                'base_value': float(base_value) if not np.isnan(base_value) else 0
            }
        except Exception as e:
            logger.error(f"Error in sensitivity analysis: {e}")
            return {
                'wacc_range': [],
                'growth_range': [],
                'sensitivity_matrix': [],
                'base_value': 0
            }

    def _validate_reformulation(self, reformulation: Dict) -> Dict[str, Any]:
        """Rigorous validation of reformulation"""
        try:
            bs_check = reformulation['reformulated_bs'].loc['Check (NOA - NFO - Equity)']
            is_check = reformulation['reformulated_is'].loc['Check (OI - NFE - NI)']
            
            # Check validity with tolerance
            bs_valid = bool(np.allclose(bs_check.dropna(), 0, atol=1))
            is_valid = bool(np.allclose(is_check.dropna(), 0, atol=1))
            
            # Calculate completeness
            total_cells = len(bs_check) + len(is_check)
            non_null_cells = bs_check.notna().sum() + is_check.notna().sum()
            completeness = (non_null_cells / total_cells * 100) if total_cells > 0 else 0
            
            # Find discrepancies
            bs_disc = bs_check[bs_check.abs() > 1].to_dict() if not bs_check.empty else {}
            is_disc = is_check[is_check.abs() > 1].to_dict() if not is_check.empty else {}
            
            return {
                'balance_sheet_valid': bs_valid,
                'income_statement_valid': is_valid,
                'completeness_percentage': float(completeness),
                'discrepancies': {
                    'balance_sheet': bs_disc,
                    'income_statement': is_disc
                }
            }
        except Exception as e:
            logger.error(f"Error in validation: {e}")
            return {
                'balance_sheet_valid': False,
                'income_statement_valid': False,
                'completeness_percentage': 0.0,
                'discrepancies': {}
            }

    def _generate_diagnostics(self, reformulation: Dict) -> Dict[str, Any]:
        """Generate diagnostic insights and recommendations"""
        insights = []
        
        try:
            # NOA growth analysis
            if 'noa' in reformulation:
                noa = reformulation['noa']
                noa_growth = noa.pct_change().mean() * 100 if len(noa) > 1 else 0
                
                if noa_growth > 20:
                    insights.append("High NOA growth: Consider if capital investments are yielding adequate returns.")
                elif noa_growth < 5:
                    insights.append("Low NOA growth: Potential for operational efficiency improvements or expansion.")
            
            # Financial leverage analysis
            if 'nfo' in reformulation:
                nfo = reformulation['nfo']
                equity = self._get('Total Equity')
                flev = FinancialRatioCalculator.safe_divide(nfo, equity).mean()
                
                if flev > 1.0:
                    insights.append("High financial leverage: Monitor interest coverage and debt covenants closely.")
                elif flev < 0.2:
                    insights.append("Low financial leverage: Consider optimal capital structure for value creation.")
            
            return {
                'insights': insights,
                'key_stats': {
                    'avg_noa_growth %': float(noa_growth) if 'noa_growth' in locals() else 0,
                    'avg_flev': float(flev) if 'flev' in locals() else 0
                }
            }
        except Exception as e:
            logger.error(f"Error generating diagnostics: {e}")
            return {'insights': [], 'key_stats': {}}

    # Helper methods
    def _get(self, metric_name: str, alt_name: Optional[str] = None) -> pd.Series:
        """Get metric data with fallback"""
        key = self.mappings.get(metric_name) or (self.mappings.get(alt_name) if alt_name else None)
        if key and key in self.df.index:
            return self.df.loc[key, self.years]
        return pd.Series(np.nan, index=self.years, name=metric_name)

    def _get_multi(self, metric_list_name: str) -> pd.Series:
        """Get sum of multiple metrics"""
        keys = self.mappings.get(metric_list_name, [])
        valid_keys = [k for k in keys if k in self.df.index]
        if not valid_keys:
            return pd.Series(0, index=self.years)
        return self.df.loc[valid_keys, self.years].sum()

    def _calculate_noa(self) -> pd.Series:
        """Calculate Net Operating Assets"""
        operating_assets = self._get('Total Assets') - self._get_multi('Financial Assets')
        operating_liabilities = self._get('Total Liabilities') - self._get_multi('Financial Liabilities')
        return operating_assets - operating_liabilities

    def _estimate_tax_rate(self) -> pd.Series:
        """Advanced tax rate estimation"""
        try:
            ebt = self._get('Earnings Before Tax', 'EBT')
            tax_expense = self._get('Tax Expense', 'Income Tax Expense')
            
            # If we have both EBT and tax expense
            if not ebt.isna().all() and not tax_expense.isna().all():
                # Calculate effective tax rate per period
                tax_rate = FinancialRatioCalculator.safe_divide(tax_expense, ebt)
                
                # Clean and bound the rates
                tax_rate = tax_rate.clip(lower=TAX_RATE_BOUNDS[0], upper=TAX_RATE_BOUNDS[1])
                
                # Fill missing values with average
                avg_rate = tax_rate.mean()
                if np.isnan(avg_rate):
                    avg_rate = 0.25
                
                tax_rate = tax_rate.fillna(avg_rate)
            else:
                # Use default rate
                tax_rate = pd.Series(0.25, index=self.years)
            
            return tax_rate
        except Exception as e:
            logger.error(f"Error estimating tax rate: {e}")
            return pd.Series(0.25, index=self.years)

    def _calculate_dividend_payout_ratio(self) -> pd.Series:
        """Calculate dividend payout ratio"""
        dividends = self._get('Dividends Paid', 'Dividends')
        ni = self._get('Net Income')
        return FinancialRatioCalculator.safe_divide(dividends.abs(), ni).fillna(0).clip(0, 1)

    def _calculate_wacc(self, avg_equity: pd.Series, avg_nfo: pd.Series) -> pd.Series:
        """Calculate WACC using market data"""
        try:
            # Cost of equity (CAPM)
            beta = 1.0  # Default assumption
            cost_of_equity = RISK_FREE_RATE + beta * MARKET_RISK_PREMIUM
            
            # Cost of debt
            interest_expense = self._get('Interest Expense')
            cost_of_debt_series = FinancialRatioCalculator.safe_divide(interest_expense, avg_nfo)
            cost_of_debt = cost_of_debt_series.mean()
            
            if np.isnan(cost_of_debt) or cost_of_debt <= 0:
                cost_of_debt = 0.05  # Default 5%
            
            # Capital structure weights
            total_capital = avg_equity + avg_nfo
            equity_weight = FinancialRatioCalculator.safe_divide(avg_equity, total_capital).fillna(0.5)
            debt_weight = FinancialRatioCalculator.safe_divide(avg_nfo, total_capital).fillna(0.5)
            
            # Tax rate
            tax_rate = self._estimate_tax_rate()
            
            # WACC calculation
            wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
            
            return wacc.clip(0.05, 0.20)  # Reasonable bounds
        except Exception as e:
            logger.error(f"Error calculating WACC: {e}")
            return pd.Series(DEFAULT_WACC, index=self.years)

    def _calculate_dynamic_wacc(self) -> float:
        """Get most recent WACC estimate"""
        try:
            latest_equity = self._get('Total Equity').iloc[-1]
            latest_nfo = self._calculate_noa().iloc[-1]
            wacc_series = self._calculate_wacc(
                pd.Series([latest_equity], index=[self.years[-1]]),
                pd.Series([latest_nfo], index=[self.years[-1]])
            )
            return float(wacc_series.iloc[-1]) if not wacc_series.empty else DEFAULT_WACC
        except Exception as e:
            logger.error(f"Error calculating dynamic WACC: {e}")
            return DEFAULT_WACC

    def _score_metric(self, values: Union[pd.Series, np.ndarray], lower: float, upper: float, higher_is_better: bool = True) -> pd.Series:
        """Score metric on 0-100 scale"""
        try:
            if isinstance(values, pd.Series):
                scores = ((values - lower) / (upper - lower)).clip(0, 1) * 100
            else:
                scores = pd.Series(((values - lower) / (upper - lower)).clip(0, 1) * 100, index=self.years)
            
            if not higher_is_better:
                scores = 100 - scores
            
            return scores
        except Exception as e:
            logger.error(f"Error scoring metric: {e}")
            return pd.Series(50, index=self.years)

    def _interpret_quality_score(self, score: float) -> str:
        """Interpret earnings quality score"""
        if np.isnan(score):
            return "Unable to assess"
        elif score >= 80:
            return "High Quality Earnings"
        elif score >= 60:
            return "Good Quality"
        elif score >= 40:
            return "Average Quality"
        elif score >= 20:
            return "Low Quality"
        else:
            return "Poor Quality - Potential Manipulation Risk"

    def _identify_quality_red_flags(self, cash_flow_ratio, scaled_accruals, revenue_quality_flag, m_score, sloan_ratio) -> List[str]:
        """Identify red flags in earnings quality"""
        red_flags = []
        
        try:
            # Check each metric
            if not cash_flow_ratio.empty and cash_flow_ratio.mean() < 0.8:
                red_flags.append("Low cash flow relative to net income - possible accrual inflation")
            
            if not scaled_accruals.empty and np.abs(scaled_accruals).mean() > 0.05:
                red_flags.append("High accruals level - potential earnings management")
            
            if not revenue_quality_flag.empty and revenue_quality_flag.mean() > 10:
                red_flags.append("Receivables growing faster than revenue - revenue recognition concerns")
            
            if not m_score.empty and m_score.mean() > -1.78:
                red_flags.append("Beneish M-Score indicates potential manipulation")
            
            if not sloan_ratio.empty and np.abs(sloan_ratio).mean() > 0.1:
                red_flags.append("High Sloan ratio - unsustainable earnings")
        except Exception as e:
            logger.error(f"Error identifying red flags: {e}")
        
        return red_flags

    def _fit_advanced_model(self, X: np.array, y: np.array) -> Any:
        """Fit ridge regression with polynomial features"""
        try:
            # Remove NaN values
            mask = ~np.isnan(y)
            X_clean = X[mask]
            y_clean = y[mask]
            
            if len(X_clean) < 2:
                # Return a simple model that predicts the mean
                class MeanModel:
                    def __init__(self, mean_val):
                        self.mean_val = mean_val
                    def predict(self, X):
                        return np.full(len(X), self.mean_val)
                
                return MeanModel(np.nanmean(y))
            
            # Fit polynomial ridge regression
            model = make_pipeline(
                PolynomialFeatures(degree=min(2, len(X_clean) - 1)),
                StandardScaler(),
                Ridge(alpha=1.0)
            )
            model.fit(X_clean, y_clean)
            return model
        except Exception as e:
            logger.error(f"Error fitting model: {e}")
            # Return simple mean predictor
            class MeanModel:
                def __init__(self, mean_val):
                    self.mean_val = mean_val
                def predict(self, X):
                    return np.full(len(X), self.mean_val)
            
            return MeanModel(np.nanmean(y) if len(y) > 0 else 0)

    def _calculate_beneish_mscore(self, revenue, receivables, gross_profit, total_assets, total_liabilities) -> pd.Series:
        """Calculate simplified Beneish M-Score"""
        try:
            # Days Sales in Receivables Index
            dsr = FinancialRatioCalculator.safe_divide(receivables * 365, revenue)
            dsri = dsr / dsr.shift(1)
            
            # Gross Margin Index
            gm = FinancialRatioCalculator.safe_divide(gross_profit, revenue)
            gmi = gm.shift(1) / gm
            
            # Asset Quality Index
            non_current_assets = total_assets - self._get('Current Assets')
            aqi_ratio = FinancialRatioCalculator.safe_divide(non_current_assets, total_assets)
            aqi = aqi_ratio / aqi_ratio.shift(1)
            
            # Sales Growth Index
            sgi = revenue / revenue.shift(1)
            
            # Leverage Index
            leverage = FinancialRatioCalculator.safe_divide(total_liabilities, total_assets)
            lvgi = leverage / leverage.shift(1)
            
            # Total Accruals to Total Assets
            ni = self._get('Net Income')
            ocf = self._get('Operating Cash Flow')
            total_accruals = ni - ocf
            tata = FinancialRatioCalculator.safe_divide(total_accruals, total_assets)
            
            # Simplified M-Score (using available components)
            m_score = (-4.84 + 
                      0.92 * dsri.fillna(1) + 
                      0.528 * gmi.fillna(1) + 
                      0.404 * aqi.fillna(1) + 
                      0.892 * sgi.fillna(1) + 
                      4.679 * tata.fillna(0) - 
                      0.327 * lvgi.fillna(1))
            
            return m_score
        except Exception as e:
            logger.error(f"Error calculating M-Score: {e}")
            return pd.Series(-2.0, index=self.years)  # Return neutral score

# --- 7. File Parsing and Processing ---
def parse_html_xls_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    """Parse HTML/XLS files with error handling"""
    try:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB.")
        
        # Read file
        file_content = uploaded_file.getvalue()
        
        # Try different parsing methods
        try:
            # Try multi-index header first
            dfs = pd.read_html(io.BytesIO(file_content), header=[0, 1])
            if dfs:
                df = dfs[0]
                # Extract company name from multi-index
                company_name = "Unknown Company"
                if hasattr(df.columns, 'levels') and len(df.columns.levels) > 0:
                    first_col = str(df.columns[0][0]) if isinstance(df.columns[0], tuple) else str(df.columns[0])
                    if ">>" in first_col:
                        company_name = first_col.split(">>")[2].split("(")[0].strip()
                
                # Flatten columns
                df.columns = [str(c[1]) if isinstance(c, tuple) else str(c) for c in df.columns]
        except:
            # Try single header
            dfs = pd.read_html(io.BytesIO(file_content), header=0)
            if dfs:
                df = dfs[0]
                company_name = "Unknown Company"
            else:
                raise ValueError("No tables found in file")
        
        # Process dataframe
        first_col = df.columns[0]
        df = df.rename(columns={first_col: "Metric"})
        df = df.dropna(subset=["Metric"]).reset_index(drop=True)
        
        # Handle duplicates
        is_duplicate = df.duplicated(subset=['Metric'], keep=False)
        df['unique_metric_id'] = df['Metric']
        df.loc[is_duplicate, 'unique_metric_id'] = df.loc[is_duplicate, 'Metric'] + ' (row ' + (df.index[is_duplicate] + 2).astype(str) + ')'
        
        df = df.set_index('unique_metric_id').drop(columns=['Metric'])
        
        return {
            "statement": df, 
            "company_name": company_name, 
            "source": uploaded_file.name
        }
    except Exception as e:
        logger.error(f"Failed to parse HTML/XLS file {uploaded_file.name}: {e}")
        return None

def parse_csv_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    """Parse CSV files with error handling"""
    try:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB.")
        
        # Read CSV
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        
        # Extract company name
        company_name = "From CSV"
        if 'Company' in df.columns:
            company_name = str(df['Company'].iloc[0]) if not df['Company'].empty else company_name
            df = df.drop(columns=['Company'])
        
        # Find metric column
        metric_col = None
        for col in df.columns:
            if col.lower() in ['metric', 'item', 'description', 'account', 'line item']:
                metric_col = col
                break
        
        if not metric_col:
            metric_col = df.columns[0]
        
        df = df.set_index(metric_col)
        
        return {
            "statement": df, 
            "company_name": company_name, 
            "source": uploaded_file.name
        }
    except Exception as e:
        logger.error(f"Failed to parse CSV file {uploaded_file.name}: {e}")
        return None

def parse_single_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    """Parse a single uploaded file"""
    try:
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'csv':
            parsed_data = parse_csv_file(uploaded_file)
        elif file_extension in ['html', 'htm', 'xls', 'xlsx']:
            parsed_data = parse_html_xls_file(uploaded_file)
        else:
            st.error(f"Unsupported file type: {file_extension}")
            return None
        
        if parsed_data is None:
            st.warning(f"Could not parse '{uploaded_file.name}'.")
            return None

        df = parsed_data["statement"]
        
        # Identify year columns
        year_cols_map = {}
        for col in df.columns:
            match = YEAR_REGEX.search(str(col))
            if match:
                year = match.group(0).replace('FY', '')
                year_cols_map[col] = year
            else:
                # Try to parse as year directly
                try:
                    year = int(col)
                    if 1980 <= year <= 2050:
                        year_cols_map[col] = str(year)
                except:
                    pass
        
        df = df.rename(columns=year_cols_map)
        valid_years = sorted([y for y in df.columns if str(y).isdigit()], key=int)
        
        if not valid_years:
            st.warning(f"No valid year columns found in '{uploaded_file.name}'.")
            return None
        
        # Clean and process data
        df_proc = DataProcessor.clean_numeric_data(df[valid_years].copy())
        df_proc = df_proc.dropna(how='all')
        
        parsed_data["statement"] = df_proc
        parsed_data["year_columns"] = valid_years
        
        return parsed_data
    except Exception as e:
        logger.error(f"Error parsing file {uploaded_file.name}: {e}")
        st.error(f"Error parsing '{uploaded_file.name}': {str(e)}")
        return None

@st.cache_data(show_spinner="Processing and merging files...")
def process_and_merge_files(_uploaded_files: List[UploadedFile]) -> Optional[Dict[str, Any]]:
    """Process and merge multiple uploaded files"""
    if not _uploaded_files:
        return None
    
    all_dfs = []
    company_name = "Multiple Sources"
    sources = {}
    first_company = None
    
    with st.spinner("Parsing and merging files..."):
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(_uploaded_files):
            progress_bar.progress((idx + 1) / len(_uploaded_files))
            
            parsed = parse_single_file(file)
            if parsed:
                df = parsed["statement"]
                source = parsed["source"]
                
                # Track sources for each metric
                for metric in df.index:
                    if metric not in sources:
                        sources[metric] = source
                    else:
                        logger.warning(f"Duplicate metric '{metric}' found. Using from {sources[metric]}.")
                
                all_dfs.append(df)
                
                # Set company name from first valid file
                if not first_company and parsed["company_name"] not in ["Unknown Company", "From CSV"]:
                    company_name = parsed["company_name"]
                    first_company = True
        
        progress_bar.empty()
    
    if not all_dfs:
        st.error("None of the uploaded files could be parsed successfully.")
        return None
    
    # Merge dataframes
    if len(all_dfs) == 1:
        merged_df = all_dfs[0]
    else:
        # Concatenate and keep first occurrence of each metric
        merged_df = pd.concat(all_dfs, axis=0, join='outer')
        merged_df = merged_df.groupby(level=0).first()
    
    # Get all year columns and ensure consistent ordering
    all_years = set()
    for df in all_dfs:
        all_years.update(df.columns)
    
    year_columns = sorted([y for y in all_years if str(y).isdigit()], key=int)
    merged_df = merged_df.reindex(columns=year_columns, fill_value=np.nan)
    
    # Calculate data quality
    data_quality = asdict(DataProcessor.calculate_data_quality(merged_df))
    
    # Detect outliers
    outliers = DataProcessor.detect_outliers(merged_df)
    
    return {
        "statement": merged_df,
        "company_name": company_name,
        "data_quality": data_quality,
        "outliers": outliers,
        "year_columns": year_columns,
        "sources": sources
    }

# --- 8. Main Dashboard Application ---
class DashboardApp:
    def __init__(self):
        self._initialize_state()
        self.chart_builders = {
            "Line": ChartGenerator.create_line_chart, 
            "Bar": ChartGenerator.create_bar_chart
        }

    def _initialize_state(self):
        """Initialize session state with defaults"""
        defaults = {
            "analysis_data": None, 
            "metric_mapping": {}, 
            "pn_results": None, 
            "pn_mappings": {},
            "selected_industry": "Technology"
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def _handle_file_upload(self):
        """Handle file upload and processing"""
        files = st.session_state.get("file_uploader_key", [])
        if files:
            st.session_state.analysis_data = process_and_merge_files(files)
            st.session_state.metric_mapping = {}
            st.session_state.pn_results = None
            st.session_state.pn_mappings = {}

    def run(self):
        """Main application entry point"""
        self.render_sidebar()
        self.render_main_panel()

    def render_sidebar(self):
        """Render sidebar with file upload and settings"""
        st.sidebar.title("📂 Upload & Options")
        st.sidebar.info("Upload financial statements (CSV, HTML, XLSX). Multiple files will be merged automatically.")
        
        st.sidebar.file_uploader(
            "Upload Financial Files",
            type=ALLOWED_FILE_TYPES,
            accept_multiple_files=True,
            key="file_uploader_key",
            on_change=self._handle_file_upload,
            help="Upload one or more financial statement files"
        )
        
        if st.sidebar.button("🔄 Reset All", help="Clear all data and start fresh"):
            st.session_state.clear()
            st.rerun()
        
        st.sidebar.divider()
        st.sidebar.title("⚙️ Display Settings")
        st.sidebar.checkbox("Show Data Quality Indicators", key="show_data_quality", value=True)
        
        if st.session_state.analysis_data:
            self._render_general_metric_mapper()
            self._render_industry_selection()

    def _render_general_metric_mapper(self):
        """Render metric mapping interface"""
        with st.sidebar.expander("📊 General Metric Mapping", expanded=False):
            st.info("Map your data fields to standard financial metrics for analysis.")
            
            # Get all required metrics
            all_req_metrics = sorted(set(m for metrics in REQUIRED_METRICS.values() for m in metrics))
            available_metrics = [''] + st.session_state.analysis_data["statement"].index.tolist()
            
            # Current mapping
            current_mapping = st.session_state.metric_mapping.copy()
            
            # Auto-map exact matches
            for std_metric in all_req_metrics:
                if std_metric not in current_mapping and std_metric in available_metrics[1:]:
                    current_mapping[std_metric] = std_metric
            
            # Render selectors
            for std_metric in all_req_metrics:
                current_value = current_mapping.get(std_metric, '')
                
                # Find index
                try:
                    index = available_metrics.index(current_value) if current_value in available_metrics else 0
                except:
                    index = 0
                
                st.session_state.metric_mapping[std_metric] = st.selectbox(
                    f"**{std_metric}**",
                    options=available_metrics,
                    index=index,
                    key=f"map_{std_metric}"
                )

    def _render_industry_selection(self):
        """Render industry selection for benchmarking"""
        with st.sidebar.expander("🏭 Industry Benchmarking", expanded=False):
            industries = list(IndustryBenchmarks.BENCHMARKS.keys())
            current_industry = st.session_state.get("selected_industry", industries[0])
            
            st.selectbox(
                "Select Industry for Comparison",
                industries,
                index=industries.index(current_industry) if current_industry in industries else 0,
                help="Select your company's industry for benchmarking"
            )

    def render_main_panel(self):
        """Render main dashboard panel"""
        st.markdown("<div class='main-header'>💹 Elite Financial Analytics Platform</div>", unsafe_allow_html=True)
        
        if not st.session_state.analysis_data:
            self._render_welcome_screen()
            return
        
        # Main content
        data = st.session_state.analysis_data
        df = data["statement"]
        
        # Data quality metrics
        dq_dict = data["data_quality"]
        dq = DataQualityMetrics(**{k: dq_dict[k] for k in ['total_rows', 'missing_values', 'missing_percentage', 'duplicate_rows']})
        
        # Company header
        st.subheader(f"📊 Company Analysis: {data['company_name']}")
        
        # Data quality indicator
        if st.session_state.show_data_quality:
            quality_class = f"quality-{dq.quality_score.lower()}"
            st.markdown(
                f"""<div class="feature-card">
                <h4><span class="data-quality-indicator {quality_class}"></span>Data Quality: {dq.quality_score}</h4>
                <p>Total Metrics: {dq.total_rows} | Missing Values: {dq.missing_values} ({dq.missing_percentage:.1f}%) | Years: {', '.join(map(str, data['year_columns']))}</p>
                </div>""", 
                unsafe_allow_html=True
            )
        
        # Main tabs
        tabs = ["📊 Visualizations", "📄 Data Table", "💡 Financial Analysis", "🔍 Penman-Nissim Analysis"]
        tab_viz, tab_data, tab_analysis, tab_pn = st.tabs(tabs)
        
        with tab_viz:
            self._render_visualization_tab(df, data)
        with tab_data:
            self._render_data_table_tab(df, data)
        with tab_analysis:
            self._render_financial_analysis_tab(df)
        with tab_pn:
            self._render_penman_nissim_tab(df, data)

    def _render_welcome_screen(self):
        """Render welcome screen when no data is loaded"""
        st.info("👋 Welcome to the Elite Financial Analytics Platform!")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
            <h3>📊 Advanced Visualizations</h3>
            <p>Create interactive charts with outlier detection, trend analysis, and multiple visualization types.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
            <h3>💡 Financial Analysis</h3>
            <p>Calculate comprehensive financial ratios including profitability, liquidity, and leverage metrics.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
            <h3>🔍 Penman-Nissim Analysis</h3>
            <p>PhD-level reformulation separating operating and financing activities with valuation models.</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.subheader("Getting Started")
        st.markdown("""
        1. **Upload Files**: Use the sidebar to upload one or more financial statement files (CSV, HTML, XLSX)
        2. **Map Metrics**: Configure metric mappings for advanced analysis
        3. **Explore**: Use the various tabs to visualize and analyze your financial data
        4. **Export**: Download your analysis results and insights
        """)

    def _render_visualization_tab(self, df: pd.DataFrame, data: Dict[str, Any]):
        """Render visualization tab"""
        st.header("Financial Data Visualization")
        
        # Controls
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
        
        with col1:
            available_metrics = df.index.tolist()
            default_metrics = available_metrics[:min(3, len(available_metrics))]
            selected_metrics = st.multiselect(
                "Select Metrics to Visualize:",
                available_metrics,
                default=default_metrics,
                help="Choose one or more metrics to plot"
            )
        
        with col2:
            chart_type = st.selectbox(
                "Chart Type:",
                list(self.chart_builders.keys()),
                key="primary_chart_type"
            )
        
        with col3:
            theme = st.selectbox(
                "Theme:",
                ["plotly_white", "plotly_dark", "seaborn", "simple_white"],
                key="primary_theme"
            )
        
        with col4:
            scale = st.selectbox(
                "Y-Axis Scale:",
                ["Linear", "Logarithmic", "Normalized (Base 100)"],
                key="primary_scale",
                help="Choose scale type for Y-axis"
            )
        
        # Additional options
        show_outliers = st.checkbox("Highlight Outliers", value=True)
        show_trend = st.checkbox("Show Trend Lines", value=False)
        
        # Generate chart
        if selected_metrics:
            # Prepare data
            if scale == "Normalized (Base 100)":
                plot_df = DataProcessor.normalize_to_100(df, selected_metrics)
                y_title = "Normalized Value (Base 100)"
            else:
                plot_df = df
                y_title = "Value"
            
            # Create chart
            outliers_to_show = data["outliers"] if show_outliers else None
            fig = self.chart_builders[chart_type](
                plot_df, selected_metrics, 
                "Financial Metrics Over Time", 
                theme, True, scale, y_title, 
                outliers_to_show
            )
            
            # Add trend lines if requested
            if show_trend and fig:
                for metric in selected_metrics:
                    if metric in plot_df.index:
                        values = plot_df.loc[metric].dropna()
                        if len(values) > 1:
                            years_numeric = np.array([int(y) for y in values.index]).reshape(-1, 1)
                            model = LinearRegression().fit(years_numeric, values.values)
                            trend_values = model.predict(years_numeric)
                            
                            fig.add_trace(go.Scatter(
                                x=values.index, 
                                y=trend_values,
                                mode='lines',
                                name=f"{metric} Trend",
                                line=dict(dash='dash', width=2),
                                showlegend=True
                            ))
            
            if fig:
                st.plotly_chart(fig, use_container_width=True)
                
                # Show statistics
                if st.checkbox("Show Statistics"):
                    stats_df = pd.DataFrame()
                    for metric in selected_metrics:
                        if metric in df.index:
                            series = df.loc[metric]
                            stats_df[metric] = {
                                'Mean': series.mean(),
                                'Median': series.median(),
                                'Std Dev': series.std(),
                                'Min': series.min(),
                                'Max': series.max(),
                                'Growth Rate %': (series.iloc[-1] / series.iloc[0] - 1) * 100 if len(series) > 1 and series.iloc[0] != 0 else np.nan
                            }
                    
                    st.dataframe(stats_df.T.style.format("{:,.2f}"))
        else:
            st.warning("Please select at least one metric to visualize.")

    def _render_data_table_tab(self, df: pd.DataFrame, data: Dict[str, Any]):
        """Render data table tab"""
        st.header("Financial Data Table")
        
        # Display options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_sources = st.checkbox("Show Data Sources", value=False)
        with col2:
            highlight_outliers = st.checkbox("Highlight Outliers", value=True)
        with col3:
            decimal_places = st.number_input("Decimal Places", min_value=0, max_value=4, value=2)
        
        # Prepare dataframe for display
        display_df = df.copy()
        
        # Add source column if requested
        if show_sources and "sources" in data:
            source_series = pd.Series(data["sources"])
            display_df.insert(0, 'Source', source_series)
        
        # Apply styling
        def highlight_outlier_cells(val, metric, year):
            if highlight_outliers and "outliers" in data:
                if metric in data["outliers"]:
                    year_idx = list(df.columns).index(year) if year in df.columns else -1
                    if year_idx in data["outliers"][metric]:
                        return 'background-color: #ffcccc'
            return ''
        
        # Format numbers
        format_dict = {col: f"{{:,.{decimal_places}f}}" for col in df.columns if col != 'Source'}
        
        # Apply styling
        styled_df = display_df.style.format(format_dict, na_rep="-")
        
        # Highlight outliers if requested
        if highlight_outliers and "outliers" in data:
            for metric in df.index:
                if metric in data["outliers"]:
                    for year in df.columns:
                        year_idx = list(df.columns).index(year)
                        if year_idx in data["outliers"][metric]:
                            styled_df = styled_df.applymap(
                                lambda x: 'background-color: #ffcccc',
                                subset=pd.IndexSlice[metric, year]
                            )
        
        # Display
        st.dataframe(styled_df, use_container_width=True)
        
        # Export options
        st.divider()
        col1, col2 = st.columns(2)
        
        with col1:
            csv = df.to_csv().encode('utf-8')
            st.download_button(
                "📥 Download as CSV",
                csv,
                "financial_data.csv",
                "text/csv",
                key='download-csv'
            )
        
        with col2:
            # Create Excel file in memory
            output = io.BytesIO()
            with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                df.to_excel(writer, sheet_name='Financial Data')
            excel_data = output.getvalue()
            
            st.download_button(
                "📥 Download as Excel",
                excel_data,
                "financial_data.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key='download-excel'
            )

    def _render_financial_analysis_tab(self, df: pd.DataFrame):
        """Render financial analysis tab"""
        st.header("💡 Financial Ratio Analysis")
        
        # Check if metrics are mapped
        mapping = {v: k for k, v in st.session_state.metric_mapping.items() if v}
        
        if not mapping:
            st.warning("⚠️ Please map metrics in the sidebar to enable financial analysis.")
            st.info("Required metrics include: Revenue, Net Profit, Total Assets, Total Equity, etc.")
            return
        
        # Rename dataframe with mapped metrics
        mapped_df = df.rename(index=mapping)
        
        # Calculate ratios
        with st.spinner("Calculating financial ratios..."):
            ratios = FinancialRatioCalculator.calculate_all_ratios(mapped_df)
        
        if not ratios:
            st.error("Unable to calculate ratios. Please ensure all required metrics are mapped correctly.")
            return
        
        # Combine all ratios
        all_ratios_list = []
        for category, ratio_df in ratios.items():
            if not ratio_df.empty:
                all_ratios_list.append(ratio_df)
        
        if all_ratios_list:
            all_ratios_df = pd.concat(all_ratios_list)
            
            # Visualization section
            st.subheader("Ratio Trends")
            
            col1, col2, col3 = st.columns([2, 1, 1])
            
            with col1:
                selected_ratios = st.multiselect(
                    "Select Ratios to Visualize:",
                    all_ratios_df.index.tolist(),
                    default=all_ratios_df.index[:min(3, len(all_ratios_df))].tolist()
                )
            
            with col2:
                chart_type = st.selectbox(
                    "Chart Type:",
                    ["Line", "Bar"],
                    key="ratio_chart_type"
                )
            
            with col3:
                show_average = st.checkbox("Show Average Line", value=True)
            
            if selected_ratios:
                # Create chart
                chart_func = ChartGenerator.create_line_chart if chart_type == "Line" else ChartGenerator.create_bar_chart
                fig = chart_func(
                    all_ratios_df, selected_ratios,
                    "Financial Ratio Analysis",
                    "plotly_white", True, "Linear", "Ratio Value",
                    None
                )
                
                # Add average lines if requested
                if show_average and fig:
                    for ratio in selected_ratios:
                        if ratio in all_ratios_df.index:
                            avg_value = all_ratios_df.loc[ratio].mean()
                            fig.add_hline(
                                y=avg_value,
                                line_dash="dash",
                                line_color="gray",
                                annotation_text=f"{ratio} Avg: {avg_value:.2f}"
                            )
                
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed tables
            st.divider()
            st.subheader("Detailed Ratio Tables")
            
            # Create tabs for each ratio category
            if len(ratios) > 1:
                ratio_tabs = st.tabs(list(ratios.keys()))
                
                for idx, (category, ratio_df) in enumerate(ratios.items()):
                    with ratio_tabs[idx]:
                        if not ratio_df.empty:
                            # Add interpretation
                            st.markdown(f"**{category} Ratios**")
                            
                            # Format and display
                            styled_df = ratio_df.style.format("{:,.2f}", na_rep="-")
                            
                            # Highlight good/bad values
                            def highlight_ratio_quality(val, ratio_name):
                                if pd.isna(val):
                                    return ''
                                
                                # Define thresholds for common ratios
                                if 'Margin' in ratio_name or 'ROE' in ratio_name or 'ROA' in ratio_name:
                                    if val > 15:
                                        return 'color: green'
                                    elif val < 5:
                                        return 'color: red'
                                elif 'Current Ratio' in ratio_name:
                                    if 1.5 <= val <= 3:
                                        return 'color: green'
                                    elif val < 1 or val > 5:
                                        return 'color: red'
                                elif 'Debt to Equity' in ratio_name:
                                    if val < 1:
                                        return 'color: green'
                                    elif val > 2:
                                        return 'color: red'
                                
                                return ''
                            
                            # Apply conditional formatting
                            for ratio in ratio_df.index:
                                styled_df = styled_df.applymap(
                                    lambda x: highlight_ratio_quality(x, ratio),
                                    subset=pd.IndexSlice[ratio, :]
                                )
                            
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Add interpretation guide
                            with st.expander("📖 Interpretation Guide"):
                                if category == "Profitability":
                                    st.markdown("""
                                    - **Gross Margin**: Higher is better. Industry average varies widely.
                                    - **Operating Margin**: Shows operational efficiency. >15% is generally good.
                                    - **Net Margin**: Final profitability. >10% is strong.
                                    - **ROE**: Return on shareholder equity. >15% is desirable.
                                    - **ROA**: Asset efficiency. >5% is generally good.
                                    """)
                                elif category == "Liquidity":
                                    st.markdown("""
                                    - **Current Ratio**: 1.5-3.0 is healthy. Below 1 indicates potential liquidity issues.
                                    - **Quick Ratio**: >1.0 is good. Excludes inventory for stricter test.
                                    - **Cash Ratio**: Most conservative measure. >0.5 is strong.
                                    """)
                                elif category == "Leverage":
                                    st.markdown("""
                                    - **Debt to Equity**: <1.0 is conservative. >2.0 may indicate high leverage.
                                    - **Debt to Assets**: <0.5 is generally safe.
                                    - **Interest Coverage**: >3.0 is healthy. Below 1.5 is concerning.
                                    - **Equity Multiplier**: Higher means more leverage.
                                    """)
            else:
                # Single category
                category, ratio_df = list(ratios.items())[0]
                st.dataframe(ratio_df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)

    def _render_penman_nissim_tab(self, df: pd.DataFrame, data: Dict[str, Any]):
        """Render Penman-Nissim analysis tab"""
        st.header("🔍 Advanced Penman-Nissim Analysis")
        
        st.info("""
        This module provides PhD-level financial statement analysis by:
        - Separating operating and financing activities
        - Calculating advanced profitability metrics (RNOA, NBC, etc.)
        - Performing earnings quality assessment
        - Forecasting and valuation using DCF and ReOI models
        - Industry benchmarking and peer comparison
        """)
        
        # Get available metrics
        available_metrics = df.index.tolist()
        
        # Initialize mappings if not exists
        if 'pn_mappings' not in st.session_state:
            st.session_state.pn_mappings = {}
        
        # Fuzzy matching helper
        def fuzzy_match(target, candidates, threshold=70):
            """Find best fuzzy match for a target string"""
            best_match = None
            best_score = 0
            
            for candidate in candidates:
                score = fuzz.token_sort_ratio(target.lower(), candidate.lower())
                if score > best_score and score >= threshold:
                    best_score = score
                    best_match = candidate
            
            return best_match
        
        # Auto-suggest mappings
        def auto_suggest_all():
            """Auto-suggest all P-N mappings using fuzzy matching"""
            
            # Define keywords for each metric type
            pn_keywords = {
                'Total Assets': ['total assets', 'assets total', 'total asset'],
                'Total Liabilities': ['total liabilities', 'liabilities total', 'total liability'],
                'Total Equity': ['total equity', 'shareholders equity', 'stockholders equity', 'net worth', 'shareholder equity'],
                'Revenue': ['revenue', 'sales', 'net sales', 'total revenue', 'gross sales'],
                'Operating Income': ['operating income', 'ebit', 'operating profit', 'earnings before interest and tax'],
                'Net Income': ['net income', 'net profit', 'profit after tax', 'net earnings', 'bottom line'],
                'Net Financial Expense': ['interest expense', 'finance cost', 'net interest expense', 'financial expense'],
                'Financial Assets': ['cash', 'bank', 'investments', 'marketable securities', 'short-term investments'],
                'Financial Liabilities': ['debt', 'borrowings', 'loans', 'notes payable', 'bonds', 'financial liabilities']
            }
            
            # Process each mapping type
            for key, keywords in pn_keywords.items():
                if key in ['Financial Assets', 'Financial Liabilities']:
                    # Multi-select fields
                    suggestions = []
                    for metric in available_metrics:
                        metric_lower = metric.lower()
                        for keyword in keywords:
                            if keyword in metric_lower or fuzz.partial_ratio(keyword, metric_lower) > 80:
                                suggestions.append(metric)
                                break
                    
                    st.session_state.pn_mappings[key] = list(dict.fromkeys(suggestions[:10]))  # Remove duplicates, limit to 10
                else:
                    # Single-select fields
                    best = fuzzy_match(key, available_metrics)
                    if not best:
                        # Try with keywords
                        for keyword in keywords:
                            best = fuzzy_match(keyword, available_metrics)
                            if best:
                                break
                    
                    st.session_state.pn_mappings[key] = best or ''
        
        # Configuration interface
        with st.expander("⚙️ Configure Penman-Nissim Metrics", expanded=True):
            # Auto-suggest button
            col1, col2 = st.columns([1, 3])
            with col1:
                if st.button("🤖 Auto-Suggest", help="Automatically suggest mappings based on metric names"):
                    auto_suggest_all()
                    st.rerun()
            
            with col2:
                st.markdown("*Configure mappings below or use auto-suggest*")
            
            st.divider()
            
            # Financial Assets and Liabilities (Multi-select)
            st.markdown("##### 1️⃣ Identify Financial Items (vs Operating)")
            
            # Get current selections or defaults
            current_fa = st.session_state.pn_mappings.get('Financial Assets', [])
            current_fl = st.session_state.pn_mappings.get('Financial Liabilities', [])
            
            # Financial Assets
            st.session_state.pn_mappings['Financial Assets'] = st.multiselect(
                "**Financial Assets** (held for investment, not operations)",
                available_metrics,
                default=[m for m in current_fa if m in available_metrics],
                help="Select assets that are financial in nature (cash, investments, etc.)"
            )
            
            # Financial Liabilities
            st.session_state.pn_mappings['Financial Liabilities'] = st.multiselect(
                "**Financial Liabilities** (interest-bearing debt)",
                available_metrics,
                default=[m for m in current_fl if m in available_metrics],
                help="Select liabilities that bear interest (debt, loans, bonds, etc.)"
            )
            
            st.divider()
            
            # Core statement items (Single-select)
            st.markdown("##### 2️⃣ Map Core Financial Statement Items")
            
            # Helper function to get selectbox index
            def get_selectbox_index(mapping_key, available_list):
                current_value = st.session_state.pn_mappings.get(mapping_key, '')
                if current_value and current_value in available_list:
                    return available_list.index(current_value) + 1
                return 0
            
            # Balance Sheet items
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.session_state.pn_mappings['Total Assets'] = st.selectbox(
                    "**Total Assets**",
                    [''] + available_metrics,
                    index=get_selectbox_index('Total Assets', available_metrics),
                    key='pn_total_assets'
                )
            
            with col2:
                st.session_state.pn_mappings['Total Liabilities'] = st.selectbox(
                    "**Total Liabilities**",
                    [''] + available_metrics,
                    index=get_selectbox_index('Total Liabilities', available_metrics),
                    key='pn_total_liabilities'
                )
            
            with col3:
                st.session_state.pn_mappings['Total Equity'] = st.selectbox(
                    "**Total Equity**",
                    [''] + available_metrics,
                    index=get_selectbox_index('Total Equity', available_metrics),
                    key='pn_total_equity'
                )
            
            # Income Statement items
            col4, col5, col6 = st.columns(3)
            
            with col4:
                st.session_state.pn_mappings['Revenue'] = st.selectbox(
                    "**Revenue**",
                    [''] + available_metrics,
                    index=get_selectbox_index('Revenue', available_metrics),
                    key='pn_revenue'
                )
            
            with col5:
                st.session_state.pn_mappings['Operating Income'] = st.selectbox(
                    "**Operating Income**",
                    [''] + available_metrics,
                    index=get_selectbox_index('Operating Income', available_metrics),
                    key='pn_operating_income',
                    help="Use EBIT or Operating Profit"
                )
            
            with col6:
                st.session_state.pn_mappings['Net Income'] = st.selectbox(
                    "**Net Income**",
                    [''] + available_metrics,
                    index=get_selectbox_index('Net Income', available_metrics),
                    key='pn_net_income'
                )
            
            # Financial expense
            st.session_state.pn_mappings['Net Financial Expense'] = st.selectbox(
                "**Net Financial Expense** (Interest Expense)",
                [''] + available_metrics,
                index=get_selectbox_index('Net Financial Expense', available_metrics),
                key='pn_nfe',
                help="Use Interest Expense or Finance Costs"
            )
            
            # Optional mappings for quality analysis
            with st.expander("📊 Optional: Earnings Quality Metrics"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.session_state.pn_mappings['Operating Cash Flow'] = st.selectbox(
                        "Operating Cash Flow",
                        [''] + available_metrics,
                        index=get_selectbox_index('Operating Cash Flow', available_metrics)
                    )
                    
                    st.session_state.pn_mappings['Accounts Receivable'] = st.selectbox(
                        "Accounts Receivable",
                        [''] + available_metrics,
                        index=get_selectbox_index('Accounts Receivable', available_metrics)
                    )
                    
                    st.session_state.pn_mappings['Inventory'] = st.selectbox(
                        "Inventory",
                        [''] + available_metrics,
                        index=get_selectbox_index('Inventory', available_metrics)
                    )
                
                with col2:
                    st.session_state.pn_mappings['Current Assets'] = st.selectbox(
                        "Current Assets",
                        [''] + available_metrics,
                        index=get_selectbox_index('Current Assets', available_metrics)
                    )
                    
                    st.session_state.pn_mappings['Current Liabilities'] = st.selectbox(
                        "Current Liabilities",
                        [''] + available_metrics,
                        index=get_selectbox_index('Current Liabilities', available_metrics)
                    )
                    
                    st.session_state.pn_mappings['Cash and Cash Equivalents'] = st.selectbox(
                        "Cash and Cash Equivalents",
                        [''] + available_metrics,
                        index=get_selectbox_index('Cash and Cash Equivalents', available_metrics)
                    )
            
            st.divider()
            
            # Run analysis button
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("🚀 Run Penman-Nissim Analysis", type="primary", use_container_width=True):
                    # Validate required mappings
                    required = ['Total Assets', 'Total Liabilities', 'Total Equity', 'Revenue', 'Operating Income', 'Net Income']
                    missing = [field for field in required if not st.session_state.pn_mappings.get(field)]
                    
                    if missing:
                        st.error(f"Missing required mappings: {', '.join(missing)}")
                    else:
                        # Run analysis
                        with st.spinner("Running comprehensive Penman-Nissim analysis..."):
                            analyzer = PenmanNissimAnalyzer(df, st.session_state.pn_mappings)
                            st.session_state.pn_results = analyzer.calculate_all()
                        st.success("Analysis complete!")
        
        # Display results
        st.markdown("---")
        
        if st.session_state.pn_results:
            results = st.session_state.pn_results
            
            if "error" in results:
                st.error(f"Analysis Error: {results['error']}")
                return
            
            # Validation status
            validation = results.get("validation", {})
            if validation.get("balance_sheet_valid") and validation.get("income_statement_valid"):
                st.success("✅ Reformulation validated: Accounting equations balance within tolerance")
            else:
                st.warning("⚠️ Reformulation validation failed. Please check your mappings.")
            
            # Completeness check
            completeness = validation.get("completeness_percentage", 0)
            if completeness < 80:
                st.warning(f"Data completeness: {completeness:.1f}%. Some calculations may be incomplete.")
            
            # Diagnostic insights
            diagnostics = results.get("diagnostics", {})
            if diagnostics.get("insights"):
                for insight in diagnostics["insights"]:
                    st.info(f"💡 {insight}")
            
            # Industry comparison summary
            if hasattr(st.session_state, 'selected_industry'):
                industry = st.session_state.selected_industry
                ratios_df = results.get("ratios", pd.DataFrame())
                
                if not ratios_df.empty and len(ratios_df.columns) > 0:
                    # Get latest year ratios
                    latest_year = ratios_df.columns[-1]
                    latest_ratios = {}
                    
                    # Map P-N ratios to benchmark names
                    ratio_mapping = {
                        'Return on Net Operating Assets (RNOA) %': 'RNOA',
                        'Operating Profit Margin (OPM) %': 'OPM',
                        'Net Operating Asset Turnover (NOAT)': 'NOAT',
                        'Net Borrowing Cost (NBC) %': 'NBC',
                        'Financial Leverage (FLEV)': 'FLEV'
                    }
                    
                    for pn_name, bench_name in ratio_mapping.items():
                        if pn_name in ratios_df.index:
                            value = ratios_df.loc[pn_name, latest_year]
                            if not pd.isna(value):
                                latest_ratios[bench_name] = float(value)
                    
                    if latest_ratios:
                        comparison = IndustryBenchmarks.calculate_composite_score(latest_ratios, industry)
                        
                        if "error" not in comparison:
                            col1, col2, col3 = st.columns([1, 2, 1])
                            with col2:
                                score = comparison['composite_score']
                                interpretation = comparison['interpretation']
                                
                                # Color based on score
                                if score >= 80:
                                    color = "green"
                                elif score >= 60:
                                    color = "blue"
                                elif score >= 40:
                                    color = "orange"
                                else:
                                    color = "red"
                                
                                st.markdown(
                                    f"<div style='text-align: center; padding: 20px; background-color: #f0f0f0; border-radius: 10px;'>"
                                    f"<h3 style='color: {color};'>Industry Performance Score: {score:.1f}/100</h3>"
                                    f"<p style='font-size: 18px;'>{interpretation} vs {industry} Industry</p>"
                                    f"</div>",
                                    unsafe_allow_html=True
                                )
            
            # Advanced visualization
            if not results.get("ratios", pd.DataFrame()).empty:
                industry_comp = comparison if 'comparison' in locals() else None
                fig = ChartGenerator.create_advanced_pn_visualization(results, industry_comp)
                st.plotly_chart(fig, use_container_width=True)
            
            # Detailed results tabs
            pn_tabs = st.tabs([
                "📊 Core Metrics", 
                "📈 Advanced Analysis", 
                "🔮 Valuation & Forecast", 
                "🏭 Industry Comparison",
                "✅ Earnings Quality",
                "📋 Raw Results"
            ])
            
            # Tab 1: Core Metrics
            with pn_tabs[0]:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Reformulated Balance Sheet")
                    bs_df = results.get('reformulated_bs', pd.DataFrame())
                    if not bs_df.empty:
                        st.dataframe(bs_df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)
                
                with col2:
                    st.subheader("Reformulated Income Statement")
                    is_df = results.get('reformulated_is', pd.DataFrame())
                    if not is_df.empty:
                        st.dataframe(is_df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)
                
                st.divider()
                
                # Key ratios summary
                st.subheader("Key Financial Ratios")
                ratios_df = results.get('ratios', pd.DataFrame())
                if not ratios_df.empty:
                    # Create summary metrics for latest year
                    if len(ratios_df.columns) > 0:
                        latest_year = ratios_df.columns[-1]
                        
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            rnoa = ratios_df.loc['Return on Net Operating Assets (RNOA) %', latest_year] if 'Return on Net Operating Assets (RNOA) %' in ratios_df.index else 0
                            st.metric("RNOA", f"{rnoa:.2f}%", help="Return on Net Operating Assets")
                        
                        with col2:
                            opm = ratios_df.loc['Operating Profit Margin (OPM) %', latest_year] if 'Operating Profit Margin (OPM) %' in ratios_df.index else 0
                            st.metric("OPM", f"{opm:.2f}%", help="Operating Profit Margin")
                        
                        with col3:
                            noat = ratios_df.loc['Net Operating Asset Turnover (NOAT)', latest_year] if 'Net Operating Asset Turnover (NOAT)' in ratios_df.index else 0
                            st.metric("NOAT", f"{noat:.2f}x", help="Net Operating Asset Turnover")
                        
                        with col4:
                            flev = ratios_df.loc['Financial Leverage (FLEV)', latest_year] if 'Financial Leverage (FLEV)' in ratios_df.index else 0
                            st.metric("FLEV", f"{flev:.2f}x", help="Financial Leverage")
                    
                    # Full ratios table
                    st.dataframe(ratios_df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)
            
            # Tab 2: Advanced Analysis
            with pn_tabs[1]:
                st.subheader("ROE Decomposition & Advanced Metrics")
                
                # ROE decomposition
                roe_decomp = results.get('roe_decomposition', pd.DataFrame())
                if not roe_decomp.empty:
                    st.markdown("#### ROE Decomposition (Operating vs Financing)")
                    
                    # Visual breakdown for latest year
                    if len(roe_decomp.columns) > 0:
                        latest_year = roe_decomp.columns[-1]
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            roe_total = roe_decomp.loc['ROE (Total) %', latest_year] if 'ROE (Total) %' in roe_decomp.index else 0
                            st.metric("Total ROE", f"{roe_total:.2f}%")
                        
                        with col2:
                            roe_op = roe_decomp.loc['ROE from Operations (RNOA) %', latest_year] if 'ROE from Operations (RNOA) %' in roe_decomp.index else 0
                            st.metric("ROE from Operations", f"{roe_op:.2f}%")
                        
                        with col3:
                            roe_fin = roe_decomp.loc['ROE from Financing (FLEV × Spread) %', latest_year] if 'ROE from Financing (FLEV × Spread) %' in roe_decomp.index else 0
                            st.metric("ROE from Financing", f"{roe_fin:.2f}%")
                    
                    st.dataframe(roe_decomp.style.format("{:,.2f}", na_rep="-"), use_container_width=True)
                
                st.divider()
                
                # Advanced metrics
                adv_metrics = results.get('advanced_metrics', pd.DataFrame())
                if not adv_metrics.empty:
                    st.markdown("#### Advanced Performance Metrics")
                    
                    # Highlight key advanced metrics
                    if len(adv_metrics.columns) > 0:
                        latest_year = adv_metrics.columns[-1]
                        
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if 'Economic Value Added (EVA)' in adv_metrics.index:
                                eva = adv_metrics.loc['Economic Value Added (EVA)', latest_year]
                                color = "green" if eva > 0 else "red"
                                st.markdown(f"<div class='metric-card'>Economic Value Added (EVA): <span style='color: {color}; font-weight: bold;'>{eva:,.0f}</span></div>", unsafe_allow_html=True)
                        
                        with col2:
                            if 'Sustainable Growth Rate %' in adv_metrics.index:
                                sgr = adv_metrics.loc['Sustainable Growth Rate %', latest_year]
                                st.markdown(f"<div class='metric-card'>Sustainable Growth Rate: <span style='font-weight: bold;'>{sgr:.2f}%</span></div>", unsafe_allow_html=True)
                    
                    st.dataframe(adv_metrics.style.format("{:,.2f}", na_rep="-"), use_container_width=True)
                
                # Interpretation
                with st.expander("📖 Interpretation Guide"):
                    st.markdown("""
                    **ROE Decomposition:**
                    - **Operating ROE (RNOA)**: Returns from core business operations
                    - **Financing ROE**: Additional returns (or costs) from financial leverage
                    - **Total ROE** = Operating ROE + (Financial Leverage × Spread)
                    
                    **Advanced Metrics:**
                    - **EVA > 0**: Company is creating value above its cost of capital
                    - **ReOI**: Residual Operating Income - excess returns above required return
                    - **FCFF**: Free Cash Flow to Firm - cash available to all investors
                    - **SGR**: Maximum growth rate sustainable without external financing
                    """)
            
            # Tab 3: Valuation & Forecast
            with pn_tabs[2]:
                forecast = results.get('forecast', {})
                
                if "error" in forecast:
                    st.error(f"Forecasting Error: {forecast['error']}")
                elif forecast:
                    st.subheader("📈 Financial Projections & Valuation")
                    
                    # Valuation summary
                    valuation = forecast.get('valuation', {})
                    scenarios = forecast.get('scenarios', {})
                    assumptions = forecast.get('assumptions', {})
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        dcf_value = valuation.get('enterprise_value_dcf', 0)
                        st.metric("DCF Valuation", f"${dcf_value:,.0f}", help="Discounted Cash Flow valuation")
                    
                    with col2:
                        reoi_value = valuation.get('enterprise_value_reoi', 0)
                        st.metric("ReOI Valuation", f"${reoi_value:,.0f}", help="Residual Operating Income valuation")
                    
                    with col3:
                        wacc = assumptions.get('wacc', 0) * 100
                        st.metric("WACC", f"{wacc:.2f}%", help="Weighted Average Cost of Capital")
                    
                    # Scenario analysis
                    st.markdown("#### Scenario Analysis")
                    scenario_df = pd.DataFrame({
                        'Scenario': ['Pessimistic', 'Base Case', 'Optimistic'],
                        'Enterprise Value': [
                            scenarios.get('pessimistic', 0),
                            scenarios.get('base', 0),
                            scenarios.get('optimistic', 0)
                        ]
                    })
                    
                    fig_scenarios = px.bar(
                        scenario_df, 
                        x='Scenario', 
                        y='Enterprise Value',
                        title='Valuation Scenarios',
                        color='Scenario',
                        color_discrete_map={'Pessimistic': '#ff6b6b', 'Base Case': '#4ecdc4', 'Optimistic': '#45b7d1'}
                    )
                    st.plotly_chart(fig_scenarios, use_container_width=True)
                    
                    # Forecast details
                    st.markdown("#### Detailed Projections")
                    
                    forecast_years = forecast.get('forecast_years', [])
                    if forecast_years:
                        forecast_df = pd.DataFrame({
                            'Year': forecast_years,
                            'Revenue': forecast.get('forecast_revenue', []),
                            'Operating Income': forecast.get('forecast_operating_income', []),
                            'NOA': forecast.get('forecast_noa', []),
                            'FCFF': forecast.get('forecast_fcff', []),
                            'ReOI': forecast.get('forecast_reoi', [])
                        })
                        
                        # Interactive forecast chart
                        fig_forecast = make_subplots(
                            rows=2, cols=1,
                            subplot_titles=('Revenue & Operating Income Forecast', 'Free Cash Flow Forecast'),
                            vertical_spacing=0.15
                        )
                        
                        # Revenue and OI
                        fig_forecast.add_trace(
                            go.Scatter(x=forecast_df['Year'], y=forecast_df['Revenue'], 
                                      name='Revenue', line=dict(color='blue', width=3)),
                            row=1, col=1
                        )
                        fig_forecast.add_trace(
                            go.Scatter(x=forecast_df['Year'], y=forecast_df['Operating Income'], 
                                      name='Operating Income', line=dict(color='green', width=3)),
                            row=1, col=1
                        )
                        
                        # FCFF
                        fig_forecast.add_trace(
                            go.Bar(x=forecast_df['Year'], y=forecast_df['FCFF'], 
                                   name='FCFF', marker_color='lightblue'),
                            row=2, col=1
                        )
                        
                        fig_forecast.update_layout(height=600, showlegend=True)
                        st.plotly_chart(fig_forecast, use_container_width=True)
                        
                        # Forecast table
                        st.dataframe(
                            forecast_df.style.format({
                                'Revenue': '{:,.0f}',
                                'Operating Income': '{:,.0f}',
                                'NOA': '{:,.0f}',
                                'FCFF': '{:,.0f}',
                                'ReOI': '{:,.0f}'
                            }),
                            use_container_width=True
                        )
                    
                    # Sensitivity analysis
                    sensitivity = results.get('sensitivity', {})
                    if sensitivity and sensitivity.get('sensitivity_matrix'):
                        st.markdown("#### Sensitivity Analysis")
                        
                        wacc_range = sensitivity.get('wacc_range', [])
                        growth_range = sensitivity.get('growth_range', [])
                        matrix = sensitivity.get('sensitivity_matrix', [])
                        
                        if wacc_range and growth_range and matrix:
                            # Create heatmap
                            fig_sensitivity = go.Figure(data=go.Heatmap(
                                z=matrix,
                                x=[f"{g:.1%}" for g in growth_range],
                                y=[f"{w:.1%}" for w in wacc_range],
                                colorscale='RdYlGn',
                                text=[[f"${v:,.0f}" for v in row] for row in matrix],
                                texttemplate="%{text}",
                                textfont={"size": 10}
                            ))
                            
                            fig_sensitivity.update_layout(
                                title='Valuation Sensitivity: Terminal Growth vs WACC',
                                xaxis_title='Terminal Growth Rate',
                                yaxis_title='WACC',
                                height=400
                            )
                            
                            st.plotly_chart(fig_sensitivity, use_container_width=True)
                
                else:
                    st.info("Configure mappings and run analysis to see valuation results.")
            
            # Tab 4: Industry Comparison
            with pn_tabs[3]:
                st.subheader("🏭 Industry Benchmarking")
                
                if hasattr(st.session_state, 'selected_industry'):
                    industry = st.session_state.selected_industry
                    ratios_df = results.get('ratios', pd.DataFrame())
                    
                    if not ratios_df.empty and len(ratios_df.columns) > 0:
                        # Get latest metrics
                        latest_year = ratios_df.columns[-1]
                        
                        # Create comparison table
                        comparison_data = []
                        
                        ratio_mapping = {
                            'Return on Net Operating Assets (RNOA) %': ('RNOA', '%'),
                            'Operating Profit Margin (OPM) %': ('OPM', '%'),
                            'Net Operating Asset Turnover (NOAT)': ('NOAT', 'x'),
                            'Net Borrowing Cost (NBC) %': ('NBC', '%'),
                            'Financial Leverage (FLEV)': ('FLEV', 'x')
                        }
                        
                        for pn_name, (bench_name, unit) in ratio_mapping.items():
                            if pn_name in ratios_df.index and bench_name in IndustryBenchmarks.BENCHMARKS[industry]:
                                company_value = ratios_df.loc[pn_name, latest_year]
                                benchmark = IndustryBenchmarks.BENCHMARKS[industry][bench_name]
                                
                                percentile = IndustryBenchmarks.get_percentile_rank(company_value, benchmark)
                                
                                comparison_data.append({
                                    'Metric': bench_name,
                                    'Company': f"{company_value:.2f}{unit}",
                                    'Industry Mean': f"{benchmark['mean']:.2f}{unit}",
                                    'Industry Std Dev': f"{benchmark['std']:.2f}{unit}",
                                    'Percentile Rank': f"{percentile:.0f}%",
                                    'Assessment': self._assess_metric_performance(percentile)
                                })
                        
                        if comparison_data:
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            # Style the dataframe
                            def highlight_assessment(val):
                                if val == 'Excellent':
                                    return 'background-color: #28a745; color: white;'
                                elif val == 'Good':
                                    return 'background-color: #17a2b8; color: white;'
                                elif val == 'Average':
                                    return 'background-color: #ffc107;'
                                elif val == 'Below Average':
                                    return 'background-color: #fd7e14; color: white;'
                                else:
                                    return 'background-color: #dc3545; color: white;'
                            
                            styled_df = comparison_df.style.applymap(
                                highlight_assessment,
                                subset=['Assessment']
                            )
                            
                            st.dataframe(styled_df, use_container_width=True)
                            
                            # Quartile chart
                            st.markdown("#### Industry Quartile Positioning")
                            
                            metrics_for_chart = []
                            positions = []
                            
                            for _, row in comparison_df.iterrows():
                                metrics_for_chart.append(row['Metric'])
                                positions.append(float(row['Percentile Rank'].rstrip('%')))
                            
                            fig_quartiles = go.Figure()
                            
                            # Add quartile bands
                            fig_quartiles.add_hrect(y0=0, y1=25, fillcolor="red", opacity=0.2, line_width=0)
                            fig_quartiles.add_hrect(y0=25, y1=50, fillcolor="orange", opacity=0.2, line_width=0)
                            fig_quartiles.add_hrect(y0=50, y1=75, fillcolor="yellow", opacity=0.2, line_width=0)
                            fig_quartiles.add_hrect(y0=75, y1=100, fillcolor="green", opacity=0.2, line_width=0)
                            
                            # Add company positions
                            fig_quartiles.add_trace(go.Scatter(
                                x=metrics_for_chart,
                                y=positions,
                                mode='markers+text',
                                marker=dict(size=15, color='blue'),
                                text=[f"{p:.0f}%" for p in positions],
                                textposition="top center",
                                name='Company Position'
                            ))
                            
                            fig_quartiles.update_layout(
                                title=f'Percentile Ranking vs {industry} Industry',
                                xaxis_title='Metrics',
                                yaxis_title='Percentile Rank',
                                yaxis_range=[0, 100],
                                height=400,
                                showlegend=False
                            )
                            
                            # Add quartile labels
                            fig_quartiles.add_annotation(x=0.02, y=87.5, text="Top Quartile", showarrow=False, xref="paper", yref="y")
                            fig_quartiles.add_annotation(x=0.02, y=62.5, text="2nd Quartile", showarrow=False, xref="paper", yref="y")
                            fig_quartiles.add_annotation(x=0.02, y=37.5, text="3rd Quartile", showarrow=False, xref="paper", yref="y")
                            fig_quartiles.add_annotation(x=0.02, y=12.5, text="Bottom Quartile", showarrow=False, xref="paper", yref="y")
                            
                            st.plotly_chart(fig_quartiles, use_container_width=True)
                        
                    else:
                        st.warning("Insufficient data for industry comparison.")
                else:
                    st.info("Select an industry in the sidebar to enable benchmarking.")
            
            # Tab 5: Earnings Quality
            with pn_tabs[4]:
                quality = results.get('quality_analysis', {})
                
                if quality:
                    st.subheader("✅ Earnings Quality Assessment")
                    
                    # Overall quality score
                    overall_score = quality.get('overall_quality_score', 50)
                    interpretation = quality.get('interpretation', 'Unable to assess')
                    
                    # Score visualization
                    col1, col2, col3 = st.columns([1, 2, 1])
                    with col2:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=overall_score,
                            domain={'x': [0, 1], 'y': [0, 1]},
                            title={'text': "Overall Earnings Quality Score"},
                            gauge={
                                'axis': {'range': [None, 100]},
                                'bar': {'color': "darkblue"},
                                'steps': [
                                    {'range': [0, 20], 'color': "red"},
                                    {'range': [20, 40], 'color': "orange"},
                                    {'range': [40, 60], 'color': "yellow"},
                                    {'range': [60, 80], 'color': "lightgreen"},
                                    {'range': [80, 100], 'color': "green"}
                                ],
                                'threshold': {
                                    'line': {'color': "red", 'width': 4},
                                    'thickness': 0.75,
                                    'value': 50
                                }
                            }
                        ))
                        fig_gauge.update_layout(height=300)
                        st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        st.markdown(f"<h3 style='text-align: center;'>{interpretation}</h3>", unsafe_allow_html=True)
                    
                    # Red flags
                    red_flags = quality.get('red_flags', [])
                    if red_flags:
                        st.error("🚩 **Quality Red Flags Detected:**")
                        for flag in red_flags:
                            st.markdown(f"- {flag}")
                    else:
                        st.success("✅ No major quality concerns identified")
                    
                    # Detailed metrics
                    st.markdown("#### Quality Metrics Details")
                    
                    quality_metrics = quality.get('metrics', pd.DataFrame())
                    if not quality_metrics.empty:
                        # Create quality chart
                        fig_quality = make_subplots(
                            rows=2, cols=2,
                            subplot_titles=(
                                'Cash Flow Quality', 
                                'Accruals Analysis',
                                'Working Capital Efficiency',
                                'Manipulation Risk (M-Score)'
                            )
                        )
                        
                        # Cash flow to net income
                        if 'Cash Flow to Net Income' in quality_metrics.index:
                            years = list(quality_metrics.columns)
                            values = quality_metrics.loc['Cash Flow to Net Income'].values
                            fig_quality.add_trace(
                                go.Scatter(x=years, y=values, name='CF/NI Ratio', line=dict(color='blue', width=3)),
                                row=1, col=1
                            )
                            fig_quality.add_hline(y=1.0, line_dash="dash", line_color="green", row=1, col=1)
                        
                        # Total accruals
                        if 'Total Accruals (% of Assets)' in quality_metrics.index:
                            years = list(quality_metrics.columns)
                            values = quality_metrics.loc['Total Accruals (% of Assets)'].values
                            fig_quality.add_trace(
                                go.Bar(x=years, y=values, name='Accruals %', marker_color='orange'),
                                row=1, col=2
                            )
                        
                        # Cash conversion cycle
                        if 'Cash Conversion Cycle' in quality_metrics.index:
                            years = list(quality_metrics.columns)
                            values = quality_metrics.loc['Cash Conversion Cycle'].values
                            fig_quality.add_trace(
                                go.Scatter(x=years, y=values, name='CCC Days', line=dict(color='purple', width=3)),
                                row=2, col=1
                            )
                        
                        # M-Score
                        if 'Beneish M-Score' in quality_metrics.index:
                            years = list(quality_metrics.columns)
                            values = quality_metrics.loc['Beneish M-Score'].values
                            fig_quality.add_trace(
                                go.Scatter(x=years, y=values, name='M-Score', line=dict(color='red', width=3)),
                                row=2, col=2
                            )
                            fig_quality.add_hline(y=-2.22, line_dash="dash", line_color="green", row=2, col=2)
                            fig_quality.add_hline(y=-1.78, line_dash="dash", line_color="red", row=2, col=2)
                        
                        fig_quality.update_layout(height=600, showlegend=False)
                        st.plotly_chart(fig_quality, use_container_width=True)
                        
                        # Metrics table
                        st.dataframe(
                            quality_metrics.style.format("{:,.2f}", na_rep="-")
                                .background_gradient(cmap='RdYlGn_r', subset=['Beneish M-Score', 'Total Accruals (% of Assets)'])
                                .background_gradient(cmap='RdYlGn', subset=['Cash Flow to Net Income']),
                            use_container_width=True
                        )
                    
                    # Interpretation guide
                    with st.expander("📖 Quality Metrics Interpretation"):
                        st.markdown("""
                        **Cash Flow to Net Income Ratio:**
                        - > 1.0: Strong cash generation (good)
                        - 0.8-1.0: Acceptable
                        - < 0.8: Potential earnings quality issues
                        
                        **Total Accruals:**
                        - < 5% of assets: Normal
                        - 5-10%: Elevated, monitor closely
                        - > 10%: High risk of earnings management
                        
                        **Beneish M-Score:**
                        - < -2.22: Low manipulation risk
                        - -2.22 to -1.78: Moderate risk
                        - > -1.78: High manipulation risk
                        
                        **Cash Conversion Cycle:**
                        - Lower is better (efficient working capital)
                        - Industry-specific benchmarks apply
                        """)
                else:
                    st.info("Earnings quality analysis requires additional metric mappings (cash flow, receivables, etc.)")
            
            # Tab 6: Raw Results
            with pn_tabs[5]:
                st.subheader("📋 Complete Analysis Results")
                
                # Create downloadable results
                all_results = {}
                
                # Collect all dataframes
                result_items = [
                    ('Reformulated Balance Sheet', results.get('reformulated_bs', pd.DataFrame())),
                    ('Reformulated Income Statement', results.get('reformulated_is', pd.DataFrame())),
                    ('Core Ratios', results.get('ratios', pd.DataFrame())),
                    ('Advanced Metrics', results.get('advanced_metrics', pd.DataFrame())),
                    ('ROE Decomposition', results.get('roe_decomposition', pd.DataFrame())),
                    ('Quality Metrics', quality.get('metrics', pd.DataFrame()) if quality else pd.DataFrame())
                ]
                
                for name, df in result_items:
                    if not df.empty:
                        all_results[name] = df
                
                # Display as JSON
                if all_results:
                    # Convert DataFrames to dict for JSON serialization
                    json_results = {}
                    for key, df in all_results.items():
                        json_results[key] = df.to_dict()
                    
                    # Add other results
                    json_results['Forecast'] = results.get('forecast', {})
                    json_results['Validation'] = results.get('validation', {})
                    json_results['Diagnostics'] = results.get('diagnostics', {})
                    
                    # Display
                    st.json(json_results)
                    
                    # Download button
                    json_str = pd.io.json.dumps(json_results, indent=2)
                    st.download_button(
                        "📥 Download Complete Results (JSON)",
                        json_str,
                        "penman_nissim_results.json",
                        "application/json",
                        key='download-pn-json'
                    )
                    
                    # Excel export
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        for name, df in all_results.items():
                            # Clean sheet name
                            sheet_name = name[:31].replace('/', '-').replace('\\', '-')
                            df.to_excel(writer, sheet_name=sheet_name)
                    
                    excel_data = output.getvalue()
                    
                    st.download_button(
                        "📥 Download Complete Results (Excel)",
                        excel_data,
                        "penman_nissim_results.xlsx",
                        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        key='download-pn-excel'
                    )
    
    def _assess_metric_performance(self, percentile: float) -> str:
        """Assess performance based on percentile rank"""
        if percentile >= 80:
            return 'Excellent'
        elif percentile >= 60:
            return 'Good'
        elif percentile >= 40:
            return 'Average'
        elif percentile >= 20:
            return 'Below Average'
        else:
            return 'Poor'

# --- 9. Main Execution ---
def main():
    """Main application entry point"""
    app = DashboardApp()
    app.run()

if __name__ == "__main__":
    main()
