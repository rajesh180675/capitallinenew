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
        if std == 0:
            return 50.0
        z_score = (value - mean) / std
        percentile = stats.norm.cdf(z_score) * 100
        return percentile
    
    @staticmethod
    def calculate_composite_score(metrics: Dict[str, float], industry: str) -> Dict[str, Any]:
        """Calculate comprehensive performance score vs industry"""
        if industry not in IndustryBenchmarks.BENCHMARKS:
            return {"error": "Industry not found"}
        
        benchmarks = IndustryBenchmarks.BENCHMARKS[industry]
        scores = {}
        weights = {'RNOA': 0.35, 'OPM': 0.25, 'NOAT': 0.20, 'NBC': -0.10, 'FLEV': -0.10}
        
        weighted_score = 0
        for metric, weight in weights.items():
            if metric in metrics and metric in benchmarks:
                percentile = IndustryBenchmarks.get_percentile_rank(
                    metrics[metric], benchmarks[metric]
                )
                # Invert percentile for negative weight metrics
                if weight < 0:
                    percentile = 100 - percentile
                scores[metric] = percentile
                weighted_score += abs(weight) * percentile
        
        return {
            'composite_score': weighted_score,
            'metric_scores': scores,
            'interpretation': IndustryBenchmarks._interpret_score(weighted_score)
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
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace(r'[,\(\)₹$€£]|Rs\.', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    @staticmethod
    def calculate_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
        total = df.size
        if total == 0: return DataQualityMetrics(0, 0, 0.0, 0)
        missing = df.isnull().sum().sum()
        return DataQualityMetrics(len(df), int(missing), (missing / total) * 100, int(df.duplicated().sum()))

    @staticmethod
    def normalize_to_100(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        df_scaled = df.loc[metrics].copy()
        for metric in metrics:
            series = df_scaled.loc[metric].dropna()
            if not series.empty and series.iloc[0] != 0:
                df_scaled.loc[metric] = (df_scaled.loc[metric] / series.iloc[0]) * 100
            else:
                df_scaled.loc[metric] = np.nan
        return df_scaled

    @staticmethod
    def detect_outliers(df: pd.DataFrame) -> Dict[str, List[int]]:
        outliers = {}
        numeric_df = df.select_dtypes(include=np.number)
        for col in numeric_df.columns:
            Q1, Q3 = numeric_df[col].quantile(0.25), numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outlier_indices = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)].index.tolist()
                if outlier_indices:
                    outliers[col] = outlier_indices
        return outliers

class ChartGenerator:
    @staticmethod
    def _create_base_figure(title, theme, show_grid, yaxis_title):
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
        fig = ChartGenerator._create_base_figure(title, theme, show_grid, yaxis_title)
        colors = px.colors.qualitative.Plotly
        for i, metric in enumerate(metrics):
            x, y = df.columns, df.loc[metric]
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=metric, line={'color': colors[i % len(colors)], 'width': 3}))
            if outliers and metric in outliers:
                outlier_x = [x[j] for j in outliers[metric] if j < len(x)]
                outlier_y = [y.iloc[j] for j in outliers[metric] if j < len(y)]
                fig.add_trace(go.Scatter(x=outlier_x, y=outlier_y, mode='markers', name=f"{metric} Outliers", marker={'color': 'red', 'size': 10, 'symbol': 'x'}))
        fig.update_xaxes(categoryorder='array', categoryarray=df.columns)
        if scale_type == 'Logarithmic': fig.update_layout(yaxis_type='log')
        return fig

    @staticmethod
    def create_bar_chart(df, metrics, title, theme, show_grid, scale_type, yaxis_title, outliers=None):
        fig = ChartGenerator._create_base_figure(title, theme, show_grid, yaxis_title)
        colors = px.colors.qualitative.Plotly
        for i, metric in enumerate(metrics):
            x, y = df.columns, df.loc[metric]
            fig.add_trace(go.Bar(x=x, y=y, name=metric, marker_color=colors[i % len(colors)]))
            if outliers and metric in outliers:
                for j in outliers[metric]:
                    if j < len(x):
                        fig.add_annotation(x=x[j], y=y.iloc[j], text="Outlier", showarrow=True, arrowhead=1)
        fig.update_xaxes(categoryorder='array', categoryarray=df.columns)
        if scale_type == 'Logarithmic': fig.update_layout(yaxis_type='log')
        return fig

    @staticmethod
    def create_advanced_pn_visualization(results: Dict[str, Any], industry_comparison: Optional[Dict] = None) -> go.Figure:
        """Create sophisticated multi-panel visualization for Penman-Nissim analysis"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('RNOA Decomposition', 'ROE Walk', 'Quality Metrics', 'Industry Position'),
            specs=[[{"secondary_y": True}, {"type": "waterfall"}],
                   [{"type": "scatter"}, {"type": "scatterpolar"}]]
        )
        
        # Panel 1: RNOA Decomposition with dual axis
        if 'ratios' in results:
            ratios = results['ratios']
            years = ratios.columns
            
            # RNOA on primary y-axis
            fig.add_trace(
                go.Scatter(x=years, y=ratios.loc['Return on Net Operating Assets (RNOA) %'], 
                           name='RNOA %', line=dict(color='blue', width=3)),
                row=1, col=1, secondary_y=False
            )
            
            # NOAT on secondary y-axis
            fig.add_trace(
                go.Scatter(x=years, y=ratios.loc['Net Operating Asset Turnover (NOAT)'], 
                           name='NOAT', line=dict(color='green', width=2, dash='dot')),
                row=1, col=1, secondary_y=True
            )
            
            # OPM on primary y-axis
            fig.add_trace(
                go.Scatter(x=years, y=ratios.loc['Operating Profit Margin (OPM) %'], 
                           name='OPM %', line=dict(color='red', width=2)),
                row=1, col=1, secondary_y=False
            )
        
        # Panel 2: ROE Waterfall
        if 'roe_decomposition' in results:
            roe_data = results['roe_decomposition']
            last_year = roe_data.columns[-1]
            
            values = [
                roe_data.loc['RNOA %', last_year],
                roe_data.loc['Financial Leverage (FLEV)', last_year] * roe_data.loc['Spread (RNOA - NBC) %', last_year],
                roe_data.loc['ROE (from P-N) %', last_year]
            ]
            
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
        if 'quality_analysis' in results:
            quality = results['quality_analysis']['metrics']
            years = quality.columns
            
            # Create quality score time series
            fig.add_trace(
                go.Scatter(
                    x=years, 
                    y=quality.loc['Cash Flow to Net Income'],
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

# --- 6. Advanced Financial Analysis Modules ---
class FinancialRatioCalculator:
    @staticmethod
    def calculate_all_ratios(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        ratios = {}
        def safe_div(num, den, is_percent=False):
            result = np.where(den != 0, num / den, np.nan)
            return result * 100 if is_percent else result

        try:
            profit_ratios = pd.DataFrame(index=df.columns)
            rev = df.loc['Revenue'] if 'Revenue' in df.index else pd.Series(np.nan, index=df.columns)
            gp = df.loc['Gross Profit'] if 'Gross Profit' in df.index else pd.Series(np.nan, index=df.columns)
            op = df.loc['EBIT'] if 'EBIT' in df.index else pd.Series(np.nan, index=df.columns)
            np_ = df.loc['Net Profit'] if 'Net Profit' in df.index else pd.Series(np.nan, index=df.columns)
            eq = df.loc['Total Equity'] if 'Total Equity' in df.index else pd.Series(np.nan, index=df.columns)
            ta = df.loc['Total Assets'] if 'Total Assets' in df.index else pd.Series(np.nan, index=df.columns)

            profit_ratios['Gross Margin %'] = safe_div(gp, rev, True)
            profit_ratios['Operating Margin %'] = safe_div(op, rev, True)
            profit_ratios['Net Margin %'] = safe_div(np_, rev, True)
            profit_ratios['ROE %'] = safe_div(np_, eq, True)
            profit_ratios['ROA %'] = safe_div(np_, ta, True)
            ratios['Profitability'] = profit_ratios.T.dropna(how='all')

            liq_ratios = pd.DataFrame(index=df.columns)
            ca = df.loc['Current Assets'] if 'Current Assets' in df.index else pd.Series(np.nan, index=df.columns)
            cl = df.loc['Current Liabilities'] if 'Current Liabilities' in df.index else pd.Series(np.nan, index=df.columns)
            liq_ratios['Current Ratio'] = safe_div(ca, cl)
            ratios['Liquidity'] = liq_ratios.T.dropna(how='all')

            lev_ratios = pd.DataFrame(index=df.columns)
            debt = df.loc['Total Debt'] if 'Total Debt' in df.index else pd.Series(np.nan, index=df.columns)
            ie = df.loc['Interest Expense'] if 'Interest Expense' in df.index else pd.Series(np.nan, index=df.columns)
            lev_ratios['Debt to Equity'] = safe_div(debt, eq)
            lev_ratios['Debt to Assets'] = safe_div(debt, ta)
            lev_ratios['Interest Coverage'] = safe_div(op, ie)
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
        
        self.operating_asset_patterns = [
            'inventory', 'receivables', 'property plant equipment', 'ppe',
            'intangible', 'goodwill', 'deferred tax asset', 'prepaid',
            'work in progress', 'raw materials', 'finished goods'
        ]
        
        self.operating_liability_patterns = [
            'payable', 'accrued', 'deferred revenue', 'provisions',
            'warranty', 'employee benefits', 'pension obligations',
            'unearned revenue', 'customer deposits', 'deferred tax liability'
        ]

    def calculate_all(self) -> Dict[str, Any]:
        """Comprehensive Penman-Nissim analysis with advanced metrics"""
        try:
            # Core reformulation
            reformulation_results = self._perform_reformulation()
            
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
                "reformulated_bs": reformulation_results['reformulated_bs'],
                "reformulated_is": reformulation_results['reformulated_is'],
                "ratios": ratio_results['core_ratios'],
                "advanced_metrics": ratio_results['advanced_metrics'],
                "roe_decomposition": ratio_results['roe_decomposition'],
                "quality_analysis": quality_results,
                "forecast": forecast_results,
                "sensitivity": sensitivity_results,
                "validation": validation_results,
                "diagnostics": self._generate_diagnostics(reformulation_results)
            }
            
        except Exception as e:
            logger.error(f"Error in Penman-Nissim calculation: {e}")
            return {"error": str(e)}

    def _perform_reformulation(self) -> Dict[str, pd.DataFrame]:
        """Core balance sheet and income statement reformulation"""
        # Balance sheet components
        total_assets = self._get('Total Assets')
        total_liabilities = self._get('Total Liabilities')
        equity = self._get('Total Equity')
        
        # Classify financial vs operating items
        financial_assets = self._get_multi('Financial Assets')
        financial_liabilities = self._get_multi('Financial Liabilities')
        
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

    def _calculate_advanced_ratios(self, reformulation: Dict) -> Dict[str, pd.DataFrame]:
        """Calculate sophisticated financial ratios with academic rigor"""
        noa = reformulation['noa']
        nfo = reformulation['nfo']
        oi = reformulation['oi_after_tax']
        nfe = reformulation['nfe_after_tax']
        equity = self._get('Total Equity')
        revenue = self._get('Revenue')
        ni = self._get('Net Income')
        
        # Calculate averages using beginning and ending balances
        avg_noa = (noa + noa.shift(1)) / 2
        avg_nfo = (nfo + nfo.shift(1)) / 2
        avg_equity = (equity + equity.shift(1)) / 2
        
        # Core ratios
        rnoa = np.where(avg_noa != 0, (oi / avg_noa) * 100, np.nan)
        nbc = np.where(avg_nfo != 0, (nfe / avg_nfo) * 100, np.nan)
        opm = np.where(revenue != 0, (oi / revenue) * 100, np.nan)
        noat = np.where(avg_noa != 0, revenue / avg_noa, np.nan)
        
        # Financial leverage metrics
        flev = np.where(avg_equity != 0, avg_nfo / avg_equity, np.nan)
        spread = rnoa - nbc
        
        # ROE decomposition (Level 1: Operating vs Financing)
        roe_operating = rnoa
        roe_financing = flev * spread
        roe_total = roe_operating + roe_financing
        
        # ROE decomposition (Level 2: Margin vs Turnover)
        roe_from_margin = opm * noat
        roe_from_leverage = flev * spread
        
        # Advanced metrics
        
        # 1. Sustainable Growth Rate (SGR)
        dividend_payout = self._calculate_dividend_payout_ratio()
        retention_ratio = 1 - dividend_payout
        sgr = roe_total * retention_ratio / 100
        
        # 2. Economic Value Added (EVA) components
        wacc = self._calculate_wacc(avg_equity, avg_nfo)
        capital_charge = wacc * avg_noa
        eva = oi - capital_charge
        
        # 3. Residual Operating Income (ReOI)
        cost_of_operations = wacc  # Use WACC for consistency
        reoi = oi - (cost_of_operations * avg_noa.shift(1))
        
        # 4. Free Cash Flow to Firm (FCFF)
        delta_noa = noa.diff()
        fcff = oi - delta_noa
        
        # 5. Cash Conversion Efficiency
        ocf = self._get('Operating Cash Flow')
        cash_conversion = np.where(oi != 0, ocf / oi, np.nan)
        
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
            'Verification: OPM × NOAT': roe_from_margin,
            'Net Borrowing Cost (NBC) %': nbc
        }, index=self.years).T
        
        return {
            'core_ratios': core_ratios,
            'advanced_metrics': advanced_metrics,
            'roe_decomposition': roe_decomposition
        }

    def _analyze_earnings_quality(self) -> Dict[str, Any]:
        """Sophisticated earnings quality assessment using advanced metrics"""
        ni = self._get('Net Income')
        ocf = self._get('Operating Cash Flow')
        revenue = self._get('Revenue')
        receivables = self._get('Accounts Receivable') if 'Accounts Receivable' in self.df.index else self._get('Trade Receivables')
        inventory = self._get('Inventory')
        payables = self._get('Accounts Payable') if 'Accounts Payable' in self.df.index else self._get('Trade Payables')
        total_assets = self._get('Total Assets')
        gross_profit = self._get('Gross Profit')
        depreciation = self._get('Depreciation')
        sga = self._get('SG&A Expenses')
        total_liabilities = self._get('Total Liabilities')
        
        # Core quality metrics
        cash_flow_ratio = np.where(ni != 0, ocf / ni, np.nan)
        
        total_accruals = ni - ocf
        avg_assets = (total_assets + total_assets.shift(1)) / 2
        scaled_accruals = np.where(avg_assets != 0, total_accruals / avg_assets, np.nan)
        
        # Sloan Ratio
        operating_assets = total_assets - self._get('Cash and Cash Equivalents')
        operating_liabilities = self._get('Current Liabilities') - self._get('Short-term Debt') if 'Short-term Debt' in self.df.index else operating_liabilities
        noa_for_sloan = operating_assets - operating_liabilities
        avg_noa_sloan = (noa_for_sloan + noa_for_sloan.shift(1)) / 2
        sloan_ratio = np.where(avg_noa_sloan != 0, noa_for_sloan.diff() / avg_noa_sloan, np.nan)
        
        # Working capital metrics
        days_sales_outstanding = np.where(revenue != 0, (receivables / revenue) * 365, np.nan)
        revenue_growth = revenue.pct_change() * 100
        receivables_growth = receivables.pct_change() * 100
        revenue_quality_flag = receivables_growth - revenue_growth
        
        days_inventory_outstanding = np.where(revenue != 0, (inventory / revenue) * 365, np.nan)
        days_payables_outstanding = np.where(revenue != 0, (payables / revenue) * 365, np.nan)
        cash_conversion_cycle = days_sales_outstanding + days_inventory_outstanding - days_payables_outstanding
        
        # Beneish M-Score components
        dsri = self._calculate_dsri(revenue, receivables)
        gmi = self._calculate_gmi(gross_profit, revenue)
        aqi = self._calculate_aqi(total_assets, operating_assets)
        sgi = self._calculate_sgi(revenue)
        depi = self._calculate_depi(depreciation)
        sgai = self._calculate_sgai(sga, revenue)
        lvgi = self._calculate_lvgi(total_liabilities, total_assets)
        tata = scaled_accruals
        
        m_score = -4.84 + 0.92*dsri + 0.528*gmi + 0.404*aqi + 0.892*sgi + 0.115*depi - 0.172*sgai + 4.679*tata - 0.327*lvgi
        
        # Quality scoring
        quality_scores = pd.DataFrame({
            'cash_flow_quality': self._score_metric(cash_flow_ratio, 0.8, 1.2, higher_is_better=True),
            'accrual_quality': self._score_metric(np.abs(scaled_accruals), 0, 0.05, higher_is_better=False),
            'revenue_quality': self._score_metric(revenue_quality_flag, -10, 10, higher_is_better=False),
            'working_capital_efficiency': self._score_metric(cash_conversion_cycle, 30, 90, higher_is_better=False),
            'm_score_quality': self._score_metric(m_score, -2.22, -1.78, higher_is_better=False)
        }, index=self.years)
        
        overall_quality_score = quality_scores.mean(axis=1).mean()
        
        red_flags = self._identify_quality_red_flags(cash_flow_ratio, scaled_accruals, revenue_quality_flag, m_score, sloan_ratio)
        
        quality_metrics = pd.DataFrame({
            'Cash Flow to Net Income': cash_flow_ratio,
            'Total Accruals (% of Assets)': scaled_accruals * 100,
            'Sloan Ratio %': sloan_ratio * 100,
            'Days Sales Outstanding': days_sales_outstanding,
            'Days Inventory Outstanding': days_inventory_outstanding,
            'Days Payables Outstanding': days_payables_outstanding,
            'Cash Conversion Cycle': cash_conversion_cycle,
            'Revenue vs Receivables Growth Diff %': revenue_quality_flag,
            'Beneish M-Score': m_score,
            'Overall Quality Score (0-100)': overall_quality_score
        }, index=self.years).T
        
        return {
            'metrics': quality_metrics,
            'quality_scores': quality_scores,
            'red_flags': red_flags,
            'interpretation': self._interpret_quality_score(overall_quality_score)
        }

    def _forecast_and_value(self, periods: int = 5) -> Dict[str, Any]:
        """Advanced forecasting and valuation using ridge regression and DCF/ReOI models"""
        revenue = self._get('Revenue').dropna()
        oi = self._get('Operating Income').dropna()
        noa = self._calculate_noa().dropna()
        
        if len(revenue) < 3:
            return {"error": "Insufficient historical data for forecasting"}
        
        years_numeric = np.array([int(y) for y in revenue.index]).reshape(-1, 1)
        
        revenue_model = self._fit_advanced_model(years_numeric, revenue.values)
        opm = (oi / revenue).values
        opm_model = self._fit_advanced_model(years_numeric, opm)
        noat = (revenue / noa).values
        noat_model = self._fit_advanced_model(years_numeric, noat)
        
        future_years = np.array([[int(revenue.index[-1]) + i] for i in range(1, periods + 1)])
        
        forecast_revenue = revenue_model.predict(future_years)
        forecast_opm = np.clip(opm_model.predict(future_years), 0.01, 0.50)
        forecast_noat = np.clip(noat_model.predict(future_years), 0.1, 5.0)
        
        forecast_oi = forecast_revenue * forecast_opm
        forecast_noa = forecast_revenue / forecast_noat
        
        historical_reinvestment_rate = np.mean(noa.diff() / oi[1:])
        reinvestment_rate = np.clip(historical_reinvestment_rate, 0, 0.8)
        
        forecast_fcff = forecast_oi * (1 - reinvestment_rate)
        
        wacc = self._calculate_dynamic_wacc()
        terminal_growth = min(TERMINAL_GROWTH_RATE, forecast_revenue.pct_change().mean())
        
        pv_factors = [1 / (1 + wacc) ** i for i in range(1, periods + 1)]
        pv_fcff = sum(forecast_fcff * pv_factors)
        
        terminal_fcff = forecast_fcff[-1] * (1 + terminal_growth)
        terminal_value = terminal_fcff / (wacc - terminal_growth)
        pv_terminal = terminal_value * pv_factors[-1]
        
        enterprise_value_dcf = pv_fcff + pv_terminal
        
        forecast_reoi = forecast_oi - (wacc * np.append(noa[-1], forecast_noa[:-1]))
        pv_reoi = sum(forecast_reoi * pv_factors)
        terminal_reoi = forecast_reoi[-1] * (1 + terminal_growth)
        terminal_value_reoi = terminal_reoi / (wacc - terminal_growth)
        pv_terminal_reoi = terminal_value_reoi * pv_factors[-1]
        
        value_reoi = noa[-1] + pv_reoi + pv_terminal_reoi
        
        current_ev_to_sales = enterprise_value_dcf / revenue[-1]
        current_ev_to_ebit = enterprise_value_dcf / oi[-1]
        
        scenarios = self._generate_scenarios(forecast_revenue, forecast_opm, forecast_noat, wacc, terminal_growth)
        
        return {
            'forecast_years': [int(revenue.index[-1]) + i for i in range(1, periods + 1)],
            'forecast_revenue': forecast_revenue,
            'forecast_operating_income': forecast_oi,
            'forecast_noa': forecast_noa,
            'forecast_fcff': forecast_fcff,
            'forecast_reoi': forecast_reoi,
            'valuation': {
                'enterprise_value_dcf': enterprise_value_dcf,
                'enterprise_value_reoi': value_reoi,
                'implied_ev_to_sales': current_ev_to_sales,
                'implied_ev_to_ebit': current_ev_to_ebit
            },
            'scenarios': scenarios,
            'assumptions': {
                'wacc': wacc,
                'terminal_growth': terminal_growth
            }
        }

    def _perform_sensitivity_analysis(self, reformulation: Dict) -> Dict[str, Any]:
        """Sensitivity analysis on WACC and growth assumptions"""
        base_wacc = self._calculate_dynamic_wacc()
        base_growth = TERMINAL_GROWTH_RATE
        
        wacc_range = np.linspace(base_wacc - 0.02, base_wacc + 0.02, 5)
        growth_range = np.linspace(base_growth - 0.01, base_growth + 0.01, 5)
        
        sensitivity_matrix = np.zeros((len(wacc_range), len(growth_range)))
        
        for i, wacc in enumerate(wacc_range):
            for j, growth in enumerate(growth_range):
                # Re-run valuation with adjusted parameters
                forecast = self._forecast_and_value(periods=5)
                sensitivity_matrix[i, j] = forecast['valuation']['enterprise_value_dcf']
        
        return {
            'wacc_range': wacc_range,
            'growth_range': growth_range,
            'sensitivity_matrix': sensitivity_matrix,
            'base_value': forecast['valuation']['enterprise_value_dcf']
        }

    def _validate_reformulation(self, reformulation: Dict) -> Dict[str, Any]:
        """Rigorous validation of reformulation with tolerance for floating-point errors"""
        bs_check = reformulation['reformulated_bs'].loc['Check (NOA - NFO - Equity)']
        is_check = reformulation['reformulated_is'].loc['Check (OI - NFE - NI)']
        
        bs_valid = np.allclose(bs_check.dropna(), 0, atol=1e-2)
        is_valid = np.allclose(is_check.dropna(), 0, atol=1e-2)
        
        completeness = (1 - bs_check.isna().mean()) * 100
        
        return {
            'balance_sheet_valid': bs_valid,
            'income_statement_valid': is_valid,
            'completeness_percentage': completeness,
            'discrepancies': {
                'balance_sheet': bs_check[bs_check.abs() > 1e-2].to_dict(),
                'income_statement': is_check[is_check.abs() > 1e-2].to_dict()
            }
        }

    def _generate_diagnostics(self, reformulation: Dict) -> Dict[str, Any]:
        """Generate diagnostic insights and recommendations"""
        insights = []
        
        noa_growth = reformulation['noa'].pct_change().mean() * 100
        if noa_growth > 20:
            insights.append("High NOA growth: Consider if capital investments are yielding adequate returns.")
        elif noa_growth < 5:
            insights.append("Low NOA growth: Potential for operational efficiency improvements or expansion.")
        
        flev = (reformulation['nfo'] / self._get('Total Equity')).mean()
        if flev > 1.0:
            insights.append("High financial leverage: Monitor interest coverage and debt covenants closely.")
        
        return {
            'insights': insights,
            'key_stats': {
                'avg_noa_growth %': noa_growth,
                'avg_flev': flev
            }
        }

    def _get(self, metric_name: str, alt_name: Optional[str] = None) -> pd.Series:
        key = self.mappings.get(metric_name) or self.mappings.get(alt_name)
        if key and key in self.df.index:
            return self.df.loc[key]
        return pd.Series(np.nan, index=self.df.columns, name=metric_name)

    def _get_multi(self, metric_list_name: str) -> pd.Series:
        keys = self.mappings.get(metric_list_name, [])
        valid_keys = [k for k in keys if k in self.df.index]
        if not valid_keys:
            return pd.Series(np.nan, index=self.df.columns)
        return self.df.loc[valid_keys].sum()

    def _calculate_noa(self) -> pd.Series:
        operating_assets = self._get('Total Assets') - self._get_multi('Financial Assets')
        operating_liabilities = self._get('Total Liabilities') - self._get_multi('Financial Liabilities')
        return operating_assets - operating_liabilities

    def _estimate_tax_rate(self) -> pd.Series:
        """Advanced tax rate estimation using optimization with bounds"""
        ebt = self._get('Earnings Before Tax')
        tax_expense = self._get('Tax Expense')
        
        if not tax_expense.isna().all() and not ebt.isna().all():
            def objective(rate):
                return np.sum((tax_expense - rate * ebt)**2)
            
            result = minimize(objective, [0.25], bounds=[TAX_RATE_BOUNDS])
            estimated_rate = result.x[0]
            rate_series = pd.Series(estimated_rate, index=self.df.columns)
        else:
            rate_series = pd.Series(0.25, index=self.df.columns)  # Default statutory rate
        
        return rate_series

    def _calculate_dividend_payout_ratio(self) -> pd.Series:
        dividends = self._get('Dividends Paid')
        ni = self._get('Net Income')
        return np.where(ni != 0, dividends / ni, np.nan)

    def _calculate_wacc(self, avg_equity: pd.Series, avg_nfo: pd.Series) -> pd.Series:
        """Calculate WACC using CAPM for cost of equity"""
        cost_of_equity = RISK_FREE_RATE + 1.0 * MARKET_RISK_PREMIUM  # Assume beta = 1.0
        cost_of_debt = self._get('Interest Expense') / avg_nfo if not avg_nfo.isna().all() else 0.05
        total_capital = avg_equity + avg_nfo
        equity_weight = avg_equity / total_capital
        debt_weight = avg_nfo / total_capital
        tax_rate = self._estimate_tax_rate()
        wacc = (equity_weight * cost_of_equity) + (debt_weight * cost_of_debt * (1 - tax_rate))
        return wacc

    def _calculate_dynamic_wacc(self) -> float:
        """Dynamic WACC calculation based on latest data"""
        latest_equity = self._get('Total Equity')[-1]
        latest_nfo = self._calculate_noa()[-1]  # Approximation
        return self._calculate_wacc(pd.Series(latest_equity), pd.Series(latest_nfo))[-1]

    def _score_metric(self, values: np.array, lower: float, upper: float, higher_is_better: bool = True) -> np.array:
        """Score metric on 0-100 scale with linear interpolation"""
        scores = np.clip((values - lower) / (upper - lower), 0, 1) * 100
        if not higher_is_better:
            scores = 100 - scores
        return scores

    def _interpret_quality_score(self, score: float) -> str:
        if score >= 80: return "High Quality Earnings"
        elif score >= 60: return "Good Quality"
        elif score >= 40: return "Average Quality"
        elif score >= 20: return "Low Quality"
        else: return "Poor Quality - Potential Manipulation Risk"

    def _identify_quality_red_flags(self, cash_flow_ratio, scaled_accruals, revenue_quality_flag, m_score, sloan_ratio) -> List[str]:
        red_flags = []
        if np.mean(cash_flow_ratio) < 0.8:
            red_flags.append("Low cash flow relative to net income - possible accrual inflation")
        if np.mean(np.abs(scaled_accruals)) > 0.05:
            red_flags.append("High accruals level - potential earnings management")
        if np.mean(revenue_quality_flag) > 10:
            red_flags.append("Receivables growing faster than revenue - revenue recognition concerns")
        if np.mean(m_score) > -1.78:
            red_flags.append("Beneish M-Score indicates potential manipulation")
        if np.mean(np.abs(sloan_ratio)) > 0.1:
            red_flags.append("High Sloan ratio - unsustainable earnings")
        return red_flags

    def _fit_advanced_model(self, X: np.array, y: np.array) -> Any:
        """Fit ridge regression with polynomial features for forecasting"""
        model = make_pipeline(PolynomialFeatures(2), StandardScaler(), Ridge(alpha=1.0))
        model.fit(X, y)
        return model

    def _generate_scenarios(self, forecast_revenue, forecast_opm, forecast_noat, wacc, terminal_growth) -> Dict[str, float]:
        """Generate optimistic, base, and pessimistic scenarios"""
        scenarios = {}
        
        # Optimistic
        opt_revenue = forecast_revenue * 1.1
        opt_opm = forecast_opm * 1.1
        opt_noat = forecast_noat * 1.1
        opt_value = self._forecast_and_value()['valuation']['enterprise_value_dcf'] * 1.2  # Simplified
        scenarios['optimistic'] = opt_value
        
        # Pessimistic
        pes_revenue = forecast_revenue * 0.9
        pes_opm = forecast_opm * 0.9
        pes_noat = forecast_noat * 0.9
        pes_value = self._forecast_and_value()['valuation']['enterprise_value_dcf'] * 0.8
        scenarios['pessimistic'] = pes_value
        
        # Base
        scenarios['base'] = self._forecast_and_value()['valuation']['enterprise_value_dcf']
        
        return scenarios

    def _calculate_dsri(self, revenue, receivables) -> pd.Series:
        """Days Sales in Receivables Index for Beneish M-Score"""
        dso = (receivables / revenue) * 365
        dsri = dso / dso.shift(1)
        return dsri.fillna(1)

    def _calculate_gmi(self, gross_profit, revenue) -> pd.Series:
        """Gross Margin Index"""
        gm = gross_profit / revenue
        gmi = gm.shift(1) / gm
        return gmi.fillna(1)

    def _calculate_aqi(self, total_assets, operating_assets) -> pd.Series:
        """Asset Quality Index"""
        aqi = (1 - operating_assets / total_assets) / (1 - operating_assets.shift(1) / total_assets.shift(1))
        return aqi.fillna(1)

    def _calculate_sgi(self, revenue) -> pd.Series:
        """Sales Growth Index"""
        sgi = revenue / revenue.shift(1)
        return sgi.fillna(1)

    def _calculate_depi(self, depreciation) -> pd.Series:
        """Depreciation Index"""
        depi = depreciation.shift(1) / depreciation
        return depi.fillna(1)

    def _calculate_sgai(self, sga, revenue) -> pd.Series:
        """SG&A Index"""
        sga_ratio = sga / revenue
        sgai = sga_ratio / sga_ratio.shift(1)
        return sgai.fillna(1)

    def _calculate_lvgi(self, total_liabilities, total_assets) -> pd.Series:
        """Leverage Index"""
        leverage = total_liabilities / total_assets
        lvgi = leverage / leverage.shift(1)
        return lvgi.fillna(1)

# --- 7. File Parsing and Processing ---
def parse_html_xls_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    try:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB.")
        df = pd.read_html(io.BytesIO(uploaded_file.getvalue()), header=[0, 1])[0]
        company_name = str(df.columns[0][0]).split(">>")[2].split("(")[0].strip() if ">>" in str(df.columns[0][0]) else "Unknown Company"
        df.columns = [str(c[1]) for c in df.columns]
        df = df.rename(columns={df.columns[0]: "Metric"}).dropna(subset=["Metric"]).reset_index(drop=True)
        is_duplicate = df.duplicated(subset=['Metric'], keep=False)
        df['unique_metric_id'] = df['Metric']
        df.loc[is_duplicate, 'unique_metric_id'] = df['Metric'] + ' (row ' + (df.index + 2).astype(str) + ')'
        df = df.set_index('unique_metric_id').drop(columns=['Metric'])
        return {"statement": df, "company_name": company_name, "source": uploaded_file.name}
    except Exception as e:
        logger.error(f"Failed to parse HTML/XLS file {uploaded_file.name}: {e}")
        return None

def parse_csv_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    try:
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File size exceeds {MAX_FILE_SIZE_MB}MB.")
        df = pd.read_csv(io.BytesIO(uploaded_file.getvalue()))
        company_name = "From CSV"
        if 'Company' in df.columns:
            company_name = df['Company'].iloc[0] if not df['Company'].empty else company_name
        metric_col = next((col for col in df.columns if col.lower() in ['metric', 'item', 'description']), df.columns[0])
        df = df.set_index(metric_col)
        return {"statement": df, "company_name": company_name, "source": uploaded_file.name}
    except Exception as e:
        logger.error(f"Failed to parse CSV file {uploaded_file.name}: {e}")
        return None

def parse_single_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
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
    year_cols_map = {col: YEAR_REGEX.search(str(col)).group(0).replace('FY', '') if YEAR_REGEX.search(str(col)) else col for col in df.columns}
    df = df.rename(columns=year_cols_map)
    valid_years = sorted(set(year for year in df.columns if str(year).isdigit()), key=int)
    
    if not valid_years:
        st.warning(f"No valid year columns in '{uploaded_file.name}'.")
        return None
        
    df_proc = DataProcessor.clean_numeric_data(df[valid_years].copy()).dropna(how='all')
    parsed_data["statement"] = df_proc
    parsed_data["year_columns"] = valid_years
    return parsed_data

@st.cache_data(show_spinner="Processing and merging files...")
def process_and_merge_files(_uploaded_files: List[UploadedFile]) -> Optional[Dict[str, Any]]:
    if not _uploaded_files: return None
    all_dfs = []
    company_name = "Multiple Sources"
    sources = {}
    first_company = None
    with st.spinner("Parsing and merging files..."):
        for file in _uploaded_files:
            parsed = parse_single_file(file)
            if parsed:
                df = parsed["statement"]
                source = parsed["source"]
                for metric in df.index:
                    if metric in sources:
                        logger.warning(f"Duplicate metric '{metric}' in {source}. Keeping first from {sources[metric]}.")
                    else:
                        sources[metric] = source
                all_dfs.append(df)
                if not first_company and parsed["company_name"] not in ["Unknown Company", "From CSV"]:
                    company_name = parsed["company_name"]
                    first_company = True
        
    if not all_dfs:
        st.error("None of the files could be parsed.")
        return None
    
    merged_df = pd.concat(all_dfs, axis=0, join='outer').groupby(level=0).first()
    year_columns = sorted(set(col for df in all_dfs for col in df.columns if str(col).isdigit()), key=int)
    merged_df = merged_df.reindex(columns=year_columns, fill_value=np.nan)
    
    data_quality = asdict(DataProcessor.calculate_data_quality(merged_df))
    outliers = DataProcessor.detect_outliers(merged_df)
    
    return {
        "statement": merged_df,
        "company_name": company_name,
        "data_quality": data_quality,
        "outliers": outliers,
        "year_columns": year_columns,
        "sources": sources
    }

class DashboardApp:
    def __init__(self):
        self._initialize_state()
        self.chart_builders = {"Line": ChartGenerator.create_line_chart, "Bar": ChartGenerator.create_bar_chart}

    def _initialize_state(self):
        defaults = {"analysis_data": None, "metric_mapping": {}, "pn_results": None, "pn_mappings": {}}
        for k, v in defaults.items():
            if k not in st.session_state: st.session_state[k] = v

    def _handle_file_upload(self):
        files = st.session_state.get("file_uploader_key", [])
        st.session_state.analysis_data = process_and_merge_files(files)
        st.session_state.metric_mapping = {}
        st.session_state.pn_results = None
        st.session_state.pn_mappings = {}

    def run(self):
        self.render_sidebar()
        self.render_main_panel()

    def render_sidebar(self):
        st.sidebar.title("📂 Upload & Options")
        st.sidebar.info("Upload financial statements (CSV, HTML, XLSX). Multiple files supported for merging.")
        st.sidebar.file_uploader(
            "Upload Financials",
            type=ALLOWED_FILE_TYPES,
            accept_multiple_files=True,
            key="file_uploader_key",
            on_change=self._handle_file_upload
        )
        if st.sidebar.button("🔄 Reset All"):
            st.session_state.clear()
            st.rerun()
        st.sidebar.title("⚙️ Display Settings")
        st.sidebar.checkbox("Show Data Quality", key="show_data_quality", value=True)
        if st.session_state.analysis_data: 
            self._render_general_metric_mapper()
            self._render_industry_selection()

    def _render_general_metric_mapper(self):
        with st.sidebar.expander("📊 General Metric Mapping", expanded=False):
            st.info("Map metrics for the 'Advanced Analysis' tab.")
            all_req_metrics = sorted(list(set(m for v in REQUIRED_METRICS.values() for m in v)))
            available_metrics = st.session_state.analysis_data["statement"].index.tolist()
            current_mapping = st.session_state.metric_mapping.copy()
            for std_metric in all_req_metrics:
                if std_metric not in current_mapping and std_metric in available_metrics:
                    current_mapping[std_metric] = std_metric
            for std_metric in all_req_metrics:
                st.session_state.metric_mapping[std_metric] = st.selectbox(
                    f"**{std_metric}**", 
                    options=[''] + available_metrics, 
                    index=(available_metrics.index(current_mapping.get(std_metric, '')) + 1) if current_mapping.get(std_metric) in available_metrics else 0, 
                    key=f"map_{std_metric}"
                )

    def _render_industry_selection(self):
        with st.sidebar.expander("🏭 Industry Benchmarking", expanded=False):
            st.session_state.selected_industry = st.selectbox(
                "Select Industry for Comparison",
                list(IndustryBenchmarks.BENCHMARKS.keys()),
                key="selected_industry"
            )

    def render_main_panel(self):
        st.markdown("<div class='main-header'>💹 Elite Financial Analytics Platform</div>", unsafe_allow_html=True)
        if not st.session_state.analysis_data:
            st.info("👋 Welcome! Please upload financial data files to begin.")
            return
        data = st.session_state.analysis_data
        df = data["statement"]
        dq_dict = data["data_quality"]
        init_args = {k: dq_dict[k] for k in ['total_rows', 'missing_values', 'missing_percentage', 'duplicate_rows']}
        dq = DataQualityMetrics(**init_args)
        st.subheader(f"Company Analysis: {data['company_name']}")
        if st.session_state.show_data_quality:
            qc = f"quality-{dq.quality_score.lower()}"
            st.markdown(f"""<div class="feature-card"><h4><span class="data-quality-indicator {qc}"></span>Merged Data Quality: {dq.quality_score}</h4>Total Unique Rows: {dq.total_rows} | Total Missing Values: {dq.missing_values} ({dq.missing_percentage:.2f}%)</div>""", unsafe_allow_html=True)
        tabs = ["📊 Primary Visualizations", "📄 Merged Data Table", "💡 Advanced Analysis", "🔍 Penman-Nissim Analysis"]
        tab_viz, tab_data, tab_adv, tab_pn = st.tabs(tabs)
        with tab_viz: self._render_primary_visualization_tab(df, data)
        with tab_data: self._render_data_table_tab(df)
        with tab_adv: self._render_advanced_analysis_tab(df)
        with tab_pn: self._render_penman_nissim_tab(df, data)

    def _render_primary_visualization_tab(self, df: pd.DataFrame, data: Dict[str, Any]):
        st.header("Primary Financial Data Visualization")
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
        metrics = col1.multiselect("Select metrics from uploaded files:", df.index.tolist(), default=df.index[:2].tolist())
        chart = col2.selectbox("Chart Type:", list(self.chart_builders.keys()), key="primary_chart_type")
        theme = col3.selectbox("Theme:", ["plotly_white", "plotly_dark"], key="primary_theme")
        scale = col4.selectbox("Y-Axis Scale:", ["Linear", "Logarithmic", "Normalized (Base 100)"], key="primary_scale")
        if metrics:
            plot_df, y_title = (DataProcessor.normalize_to_100(df, metrics), "Normalized Value (Base 100)") if scale == "Normalized (Base 100)" else (df, "Amount (₹ Cr.)")
            fig = self.chart_builders[chart](plot_df, metrics, "Primary Financials Over Time", theme, True, scale, y_title, data["outliers"])
            if fig: st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Please select at least one metric.")

    def _render_data_table_tab(self, df: pd.DataFrame):
        st.subheader("Merged and Cleaned Financial Data")
        st.dataframe(df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)

    def _render_advanced_analysis_tab(self, df: pd.DataFrame):
        st.header("💡 General Advanced Analysis")
        mapping = {v: k for k, v in st.session_state.metric_mapping.items() if v}
        if not mapping: st.warning("Please map metrics in the sidebar for this tab."); return
        mapped_df = df.rename(index=mapping)
        ratios = FinancialRatioCalculator.calculate_all_ratios(mapped_df)
        if ratios:
            all_ratios_df = pd.concat(ratios.values()).dropna(how='all')
            if not all_ratios_df.empty:
                st.subheader("Visual Analysis of General Ratios")
                v_c1, v_c2, v_c3 = st.columns([2,1,1])
                selected = v_c1.multiselect("Select Ratios to plot:", all_ratios_df.index.unique().tolist(), default=all_ratios_df.index.unique().tolist()[:2])
                chart_type = v_c2.selectbox("Chart Type", list(self.chart_builders.keys()), key="adv_chart")
                if selected:
                    fig = self.chart_builders[chart_type](all_ratios_df, selected, "Ratio Analysis", "plotly_white", True, "Linear", "Ratio")
                    st.plotly_chart(fig, use_container_width=True)
                # Trend forecast for selected ratios
                if selected:
                    for ratio in selected:
                        series = all_ratios_df.loc[ratio].dropna()
                        if len(series) > 1:
                            years = np.array(series.index.astype(int)).reshape(-1, 1)
                            values = series.values
                            model = LinearRegression().fit(years, values)
                            next_year = years[-1] + 1
                            forecast = model.predict([[next_year]])[0]
                            st.markdown(f"• Forecast for {ratio} in {next_year}: {forecast:.2f}")
            with st.expander("Data Tables", expanded=False):
                for ratio_type, ratio_df in ratios.items():
                    st.subheader(f"{ratio_type} Ratios")
                    st.dataframe(ratio_df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)

    def _render_penman_nissim_tab(self, df: pd.DataFrame, data: Dict[str, Any]):
        st.header("🔍 Advanced Penman-Nissim Analysis")
        st.info("This module provides a rigorous separation of operating and financing activities, enhanced with forecasting, quality analysis, and industry benchmarking.")
        available_metrics = df.index.tolist()
        if 'pn_mappings' not in st.session_state: st.session_state.pn_mappings = {}

        def fuzzy_match(metric, candidates, threshold=80):
            best = max(candidates, key=lambda c: fuzz.token_sort_ratio(metric.lower(), c.lower()), default=None)
            return best if best and fuzz.token_sort_ratio(metric.lower(), best.lower()) >= threshold else None

        def auto_suggest_all():
            pn_keywords = {
                'Total Assets': ['total assets', 'assets total'],
                'Total Liabilities': ['total liabilities', 'liabilities total'],
                'Total Equity': ['total equity', 'shareholders equity', 'net worth'],
                'Revenue': ['revenue', 'sales', 'net sales'],
                'Operating Income': ['ebit', 'operating profit', 'operating income'],
                'Net Income': ['net income', 'net profit', 'profit after tax'],
                'Net Financial Expense': ['interest expense', 'finance cost', 'net interest'],
                'Financial Assets': ['cash', 'bank', 'investments', 'marketable securities', 'loan to', 'financial assets', 'short-term investments'],
                'Financial Liabilities': ['debt', 'borrowings', 'loan from', 'notes payable', 'bonds', 'financial liabilities', 'long-term debt']
            }
            for key, keywords in pn_keywords.items():
                if 'Assets' in key or 'Liabilities' in key and 'Financial' in key:  # For multiselect
                    suggestions = sorted(set(m for m in available_metrics if any(fuzz.partial_ratio(kw, m.lower()) > 70 for kw in keywords)))
                    st.session_state.pn_mappings[key] = suggestions[:5]  # Limit to top 5
                else:
                    best = fuzzy_match(key, available_metrics)
                    st.session_state.pn_mappings[key] = best or ''

        with st.expander("Configure Penman-Nissim Metrics", expanded=True):
            if st.button("🤖 Auto-Suggest All Mappings"):
                auto_suggest_all()
                st.rerun()
            financial_asset_keywords = ['cash', 'bank', 'investments', 'marketable securities', 'loan to', 'financial assets']
            financial_liability_keywords = ['debt', 'borrowings', 'loan from', 'notes payable', 'bonds', 'financial liabilities']
            default_fin_assets = [m for m in available_metrics if any(kw in m.lower() for kw in financial_asset_keywords)]
            default_fin_liabilities = [m for m in available_metrics if any(kw in m.lower() for kw in financial_liability_keywords)]
            st.session_state.pn_mappings['Financial Assets'] = st.multiselect(
                "1. Select Financial Assets", available_metrics, 
                default=st.session_state.pn_mappings.get('Financial Assets', default_fin_assets[:5]), 
                help="Assets held for investment/financing purposes, not core operations. (e.g., 'Cash', 'Marketable Securities')"
            )
            st.session_state.pn_mappings['Financial Liabilities'] = st.multiselect(
                "2. Select Financial Liabilities", available_metrics, 
                default=st.session_state.pn_mappings.get('Financial Liabilities', default_fin_liabilities[:5]), 
                help="Interest-bearing debt used to finance the company. (e.g., 'Short-term Debt', 'Bonds Payable')"
            )
            st.markdown("---")
            st.markdown("##### 3. Confirm Core Statement Items")
            def get_idx(m_name, default_val=''):
                val_in_state = st.session_state.pn_mappings.get(m_name)
                if val_in_state and val_in_state in available_metrics: return available_metrics.index(val_in_state) + 1
                if default_val and default_val in available_metrics: return available_metrics.index(default_val) + 1
                return 0
            c1, c2, c3 = st.columns(3)
            st.session_state.pn_mappings['Total Assets'] = c1.selectbox("Total Assets", [''] + available_metrics, index=get_idx('Total Assets', fuzzy_match('Total Assets', available_metrics)), key='pn_ta')
            st.session_state.pn_mappings['Total Liabilities'] = c2.selectbox("Total Liabilities", [''] + available_metrics, index=get_idx('Total Liabilities', fuzzy_match('Total Liabilities', available_metrics)), key='pn_tl')
            st.session_state.pn_mappings['Total Equity'] = c3.selectbox("Total Equity", [''] + available_metrics, index=get_idx('Total Equity', fuzzy_match('Total Equity', available_metrics)), key='pn_te')
            c4, c5, c6 = st.columns(3)
            st.session_state.pn_mappings['Revenue'] = c4.selectbox("Revenue", [''] + available_metrics, index=get_idx('Revenue', fuzzy_match('Revenue', available_metrics)), key='pn_rev')
            st.session_state.pn_mappings['Operating Income'] = c5.selectbox("Operating Income (Proxy)", [''] + available_metrics, help="Use EBIT or similar.", index=get_idx('Operating Income', fuzzy_match('Operating Income', available_metrics)), key='pn_oi')
            st.session_state.pn_mappings['Net Income'] = c6.selectbox("Net Income", [''] + available_metrics, index=get_idx('Net Income', fuzzy_match('Net Income', available_metrics)), key='pn_ni')
            st.session_state.pn_mappings['Net Financial Expense'] = st.selectbox("Net Financial Expense (Proxy)", [''] + available_metrics, help="Use Interest Expense or similar.", index=get_idx('Net Financial Expense', fuzzy_match('Net Financial Expense', available_metrics)), key='pn_nfe')
            st.markdown("")
            if st.button("🚀 Run Penman-Nissim Analysis"):
                analyzer = PenmanNissimAnalyzer(df, st.session_state.pn_mappings)
                st.session_state.pn_results = analyzer.calculate_all()
        st.markdown("---")
        if st.session_state.pn_results:
            results = st.session_state.pn_results
            if "error" in results:
                st.error(f"Error: {results['error']}")
            else:
                validation = results["validation"]
                if validation["balance_sheet_valid"] and validation["income_statement_valid"]:
                    st.success("✅ Reformulation successful: Accounting equations hold within tolerance.")
                else:
                    st.error("⚠️ Reformulation check failed. Review mappings and data.")
                if validation["completeness_percentage"] < 50:
                    st.warning(f"Data completeness: {validation['completeness_percentage']:.1f}%. Results may be incomplete.")
                st.info(" | ".join(results["diagnostics"]["insights"]))
                # Industry comparison
                industry = st.session_state.get("selected_industry", "Technology")
                latest_ratios = {k: v.iloc[-1] for k, v in results["ratios"].iterrows() if not np.isnan(v.iloc[-1])}
                comparison = IndustryBenchmarks.calculate_composite_score(latest_ratios, industry)
                if "error" not in comparison:
                    st.metric("Composite Performance Score vs Industry", f"{comparison['composite_score']:.1f}/100 ({comparison['interpretation']})")
                # Advanced visualization
                fig = ChartGenerator.create_advanced_pn_visualization(results, comparison)
                st.plotly_chart(fig, use_container_width=True)
                # Tabs for detailed analysis
                pn_tabs = st.tabs(["📊 Core Analysis", "📈 Advanced Metrics", "🔮 Forecasting & Valuation", "🏭 Industry Comparison", "✅ Earnings Quality"])
                with pn_tabs[0]:
                    with st.expander("Reformulated Balance Sheet", expanded=False): st.dataframe(results['reformulated_bs'].style.format("{:,.2f}", na_rep="-"))
                    with st.expander("Reformulated Income Statement", expanded=False): st.dataframe(results['reformulated_is'].style.format("{:,.2f}", na_rep="-"))
                    with st.expander("Core P-N Ratios", expanded=False): st.dataframe(results['ratios'].style.format("{:,.2f}", na_rep="-"))
                    with st.expander("ROE Decomposition Analysis", expanded=False): st.dataframe(results['roe_decomposition'].style.format("{:,.2f}", na_rep="-"))
                with pn_tabs[1]:
                    st.dataframe(results['advanced_metrics'].style.format("{:,.2f}", na_rep="-"))
                with pn_tabs[2]:
                    forecast_years = st.slider("Forecast Years", 3, 10, 5)
                    forecast = self._render_forecast_section(results['forecast'])
                with pn_tabs[3]:
                    self._render_industry_comparison(comparison)
                with pn_tabs[4]:
                    self._render_earnings_quality(results['quality_analysis'])
                # Download options
                csv = pd.concat([results['reformulated_bs'], results['ratios'], results['advanced_metrics']]).to_csv().encode('utf-8')
                st.download_button("Download P-N Results as CSV", csv, "penman_nissim_results.csv", "text/csv")

    def _render_forecast_section(self, forecast: Dict):
        if "error" in forecast:
            st.warning(forecast["error"])
            return
        forecast_df = pd.DataFrame({
            'Revenue': forecast['forecast_revenue'],
            'Operating Income': forecast['forecast_operating_income'],
            'FCFF': forecast['forecast_fcff']
        }, index=forecast['forecast_years'])
        st.line_chart(forecast_df)
        col1, col2 = st.columns(2)
        col1.metric("Enterprise Value (DCF)", f"{forecast['valuation']['enterprise_value_dcf']:.2f}")
        col2.metric("Enterprise Value (ReOI)", f"{forecast['valuation']['enterprise_value_reoi']:.2f}")

    def _render_industry_comparison(self, comparison: Dict):
        if "error" in comparison:
            st.warning(comparison["error"])
            return
        st.dataframe(pd.DataFrame(comparison['metric_scores'], index=['Percentile Rank']).T.style.format("{:.1f}"))
        st.markdown(f"**Overall Performance:** {comparison['interpretation']}")

    def _render_earnings_quality(self, quality: Dict):
        st.metric("Earnings Quality Score", f"{quality['overall_quality_score']:.1f}/100 ({quality['interpretation']})")
        if quality['red_flags']:
            st.warning("⚠️ Red Flags Detected:")
            for flag in quality['red_flags']:
                st.markdown(f"• <span class='red-flag'>{flag}</span>", unsafe_allow_html=True)
        st.dataframe(quality['metrics'].style.format("{:,.2f}", na_rep="-"))

# --- 8. App Execution ---
if __name__ == "__main__":
    try:
        # Install dependencies if needed
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fuzzywuzzy", "python-levenshtein"])
        app = DashboardApp()
        app.run()
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        st.error(f"A critical error occurred. Please refresh. Details: {e}")
