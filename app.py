"""
Enhanced Financial Dashboard - Complete Integrated Version
A robust Streamlit application for financial data analysis with enhanced error handling,
performance optimization, and additional features.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import io
import logging
from datetime import datetime
import re
from dataclasses import dataclass, field
from functools import lru_cache
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="Advanced Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 0.75rem 1.25rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .error-message {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 0.75rem 1.25rem;
        border-radius: 0.25rem;
        margin: 1rem 0;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1f77b4, #17a2b8);
        color: white;
        border-radius: 8px;
        border: none;
        padding: 12px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .sidebar .stSelectbox label {
        font-weight: 600;
        color: #2c3e50;
    }
    .data-quality-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .quality-high { background-color: #28a745; }
    .quality-medium { background-color: #ffc107; }
    .quality-low { background-color: #dc3545; }
    .welcome-container {
        text-align: center;
        padding: 3rem 1rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 15px;
        margin: 2rem 0;
    }
    .feature-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

@dataclass
class DataQualityMetrics:
    """Data class to store data quality metrics."""
    total_rows: int
    missing_values: int
    missing_percentage: float
    duplicate_rows: int
    quality_score: str = field(init=False)
    
    def __post_init__(self):
        if self.missing_percentage < 5:
            self.quality_score = "High"
        elif self.missing_percentage < 20:
            self.quality_score = "Medium"
        else:
            self.quality_score = "Low"

class FileValidator:
    """Validates uploaded files and their content."""
    
    @staticmethod
    def validate_file(uploaded_file) -> Tuple[bool, str]:
        if uploaded_file is None:
            return False, "No file uploaded"
        if uploaded_file.size > 10 * 1024 * 1024:
            return False, "File size exceeds 10MB limit"
        allowed_types = ['xls', 'xlsx', 'html', 'htm']
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in allowed_types:
            return False, f"Unsupported file type: {file_extension}"
        return True, file_extension

class DataProcessor:
    """Handles data processing and cleaning operations."""
    
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and converts numeric data."""
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(r'[,\(\)₹]|Rs\.', '', regex=True)
                df[col] = df[col].str.replace('(', '-').str.replace(')', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detects outliers in the data using IQR method."""
        outliers = {}
        numeric_df = df.select_dtypes(include=np.number)
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                outlier_indices = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)].index.tolist()
                if outlier_indices:
                    outliers[col] = outlier_indices
        return outliers
    
    @staticmethod
    def calculate_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
        """Calculates data quality metrics."""
        total_cells = df.size
        if total_cells == 0:
            return DataQualityMetrics(0, 0, 0, 0)
        missing_values = df.isnull().sum().sum()
        return DataQualityMetrics(
            total_rows=len(df),
            missing_values=missing_values,
            missing_percentage=(missing_values / total_cells) * 100,
            duplicate_rows=df.duplicated().sum()
        )

@st.cache_data(show_spinner=False)
def parse_capitaline_file(uploaded_file) -> Optional[Dict[str, Any]]:
    """Enhanced parser for Capitaline files with better error handling and validation."""
    if uploaded_file is None: 
        return None
    
    is_valid, file_extension = FileValidator.validate_file(uploaded_file)
    if not is_valid:
        st.error(f"File validation failed: {file_extension}")
        return None

    try:
        file_content = uploaded_file.getvalue()
        
        # Parse the file using the successful method
        df = pd.read_html(io.BytesIO(file_content), header=[0, 1])[0]

        # Extract company name
        company_name = "Unknown Company"
        try:
            company_info_tuple = str(df.columns[0][0])
            if ">>" in company_info_tuple:
                company_name = company_info_tuple.split(">>")[2].split("(")[0].strip()
        except (IndexError, AttributeError): 
            pass

        # Flatten multi-level columns
        df.columns = [str(col[1]) for col in df.columns]
        df = df.rename(columns={df.columns[0]: "Metric"}).dropna(subset=['Metric']).set_index('Metric')
        
        # Extract year columns
        renamed_cols = {}
        for col in df.columns:
            year = ''.join(filter(str.isdigit, str(col)))
            if len(year) >= 4:
                renamed_cols[col] = year[:4]
        df = df.rename(columns=renamed_cols)

        # Filter valid year columns
        year_columns = sorted([col for col in df.columns if col.isdigit() and len(col) == 4 and '1990' < col < '2050'], reverse=True)
        if not year_columns:
            st.error("Could not find valid year columns in the data.")
            return None
        
        # Clean and prepare final dataframe
        df_final = df[year_columns].copy()
        df_final = DataProcessor.clean_numeric_data(df_final).dropna(how='all')
        
        return {
            "statement": df_final, 
            "company_name": company_name,
            "data_quality": DataProcessor.calculate_data_quality(df_final),
            "outliers": DataProcessor.detect_outliers(df_final),
            "year_columns": year_columns,
            "file_info": {
                "name": uploaded_file.name, 
                "size": uploaded_file.size,
                "type": file_extension
            }
        }
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        st.error(f"An error occurred while parsing: {str(e)}")
        return None

class ChartGenerator:
    """Enhanced chart generation with multiple chart types and customization options."""
    
    @staticmethod
    def _create_base_figure(title: str, chart_theme: str, show_grid: bool) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#2c3e50'), x=0.5),
            xaxis_title="Year", 
            yaxis_title="Amount (₹ Cr.)",
            template=chart_theme, 
            height=500, 
            legend_title_text='Metrics',
            xaxis=dict(showgrid=show_grid), 
            yaxis=dict(showgrid=show_grid),
            hovermode='x unified'
        )
        return fig

    @staticmethod
    def create_line_chart(df: pd.DataFrame, selected_metrics: List[str], title: str, theme: str, grid: bool) -> go.Figure:
        fig = ChartGenerator._create_base_figure(title, theme, grid)
        colors = px.colors.qualitative.Set1
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(go.Scatter(
                x=df.columns, 
                y=df.loc[metric], 
                mode='lines+markers', 
                name=metric,
                line=dict(color=colors[i % len(colors)], width=3),
                marker=dict(size=8),
                hovertemplate=f'<b>{metric}</b><br>Year: %{{x}}<br>Value: ₹%{{y:,.2f}} Cr.<extra></extra>'
            ))
        return fig

    @staticmethod
    def create_bar_chart(df: pd.DataFrame, selected_metrics: List[str], title: str, theme: str, grid: bool) -> go.Figure:
        fig = ChartGenerator._create_base_figure(title, theme, grid)
        fig.update_layout(barmode='group')
        colors = px.colors.qualitative.Set1
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(go.Bar(
                x=df.columns, 
                y=df.loc[metric], 
                name=metric,
                marker_color=colors[i % len(colors)],
                hovertemplate=f'<b>{metric}</b><br>Year: %{{x}}<br>Value: ₹%{{y:,.2f}} Cr.<extra></extra>'
            ))
        return fig
    
    @staticmethod
    def create_area_chart(df: pd.DataFrame, selected_metrics: List[str], title: str, theme: str, grid: bool) -> go.Figure:
        fig = ChartGenerator._create_base_figure(title, theme, grid)
        colors = px.colors.qualitative.Set1
        for i, metric in enumerate(selected_metrics):
            fig.add_trace(go.Scatter(
                x=df.columns, 
                y=df.loc[metric], 
                mode='lines', 
                name=metric, 
                fill='tonexty' if i > 0 else 'tozeroy',
                line=dict(color=colors[i % len(colors)]),
                hovertemplate=f'<b>{metric}</b><br>Year: %{{x}}<br>Value: ₹%{{y:,.2f}} Cr.<extra></extra>'
            ))
        return fig

    @staticmethod
    def create_heatmap(df: pd.DataFrame, selected_metrics: List[str], title: str, theme: str, grid: bool) -> go.Figure:
        if len(selected_metrics) < 2:
            st.warning("Heatmap requires at least two metrics.")
            return None
        
        corr_matrix = df.loc[selected_metrics].T.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=corr_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        fig.update_layout(
            title=dict(text=f"Correlation Matrix - {title}", font=dict(size=20, color='#2c3e50')),
            template=theme,
            height=500
        )
        return fig

class DashboardUI:
    """Enhanced UI class with improved functionality and user experience."""
    
    def __init__(self):
        self.init_session_state()
    
    def init_session_state(self):
        """Initialize session state variables."""
        session_vars = {
            "analysis_data": None,
            "_uploaded_file_memo": None,
            "chart_figure": None,
            "selected_metrics": [],
            "chart_type": "Line Chart",
            "show_data_quality": False,
            "show_outliers": False
        }
        
        for var, default_value in session_vars.items():
            if var not in st.session_state:
                st.session_state[var] = default
