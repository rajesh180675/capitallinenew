# Enhanced Financial Dashboard - Complete Integrated Version
# A robust Streamlit application for financial data analysis with enhanced error handling,
# performance optimization, and additional features.

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
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].astype(str).str.replace(r'[,\(\)₹]|Rs\.', '', regex=True)
                df[col] = df[col].str.replace('(', '-').str.replace(')', '')
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    @staticmethod
    def detect_outliers(df: pd.DataFrame) -> Dict[str, List[str]]:
        outliers = {}
        numeric_df = df.select_dtypes(include=np.number)
        for col in numeric_df.columns:
            Q1 = numeric_df[col].quantile(0.25)
            Q3 = numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outlier_idx = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)].index.tolist()
                if outlier_idx:
                    outliers[col] = outlier_idx
        return outliers

    @staticmethod
    def calculate_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
        total_cells = df.size
        if total_cells == 0:
            return DataQualityMetrics(0, 0, 0, 0)
        missing = df.isnull().sum().sum()
        return DataQualityMetrics(
            total_rows=len(df),
            missing_values=missing,
            missing_percentage=(missing / total_cells) * 100,
            duplicate_rows=df.duplicated().sum()
        )

@st.cache_data(show_spinner=False)
def parse_capitaline_file(uploaded_file) -> Optional[Dict[str, Any]]:
    if uploaded_file is None: return None
    is_valid, file_ext = FileValidator.validate_file(uploaded_file)
    if not is_valid:
        st.error(f"File validation failed: {file_ext}")
        return None
    try:
        content = uploaded_file.getvalue()
        df = pd.read_html(io.BytesIO(content), header=[0, 1])[0]
        company_name = "Unknown Company"
        try:
            header_str = str(df.columns[0][0])
            if ">>" in header_str:
                company_name = header_str.split(">>")[2].split("(")[0].strip()
        except: pass
        df.columns = [str(col[1]) for col in df.columns]
        df = df.rename(columns={df.columns[0]: "Metric"}).dropna(subset=["Metric"]).set_index("Metric")
        year_cols = {}
        for col in df.columns:
            year = ''.join(filter(str.isdigit, col))
            if len(year) >= 4:
                year_cols[col] = year[:4]
        df = df.rename(columns=year_cols)
        year_columns = sorted([col for col in df.columns if col.isdigit() and '1990' < col < '2050'], reverse=True)
        if not year_columns:
            st.error("No valid year columns found.")
            return None
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
                "type": file_ext
            }
        }
    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        st.error(f"An error occurred while parsing: {str(e)}")
        return None

class ChartGenerator:
    @staticmethod
    def _create_base_figure(title: str, theme: str, grid: bool) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title=dict(text=title, font=dict(size=20), x=0.5),
            xaxis_title="Year", yaxis_title="Amount (₹ Cr.)",
            template=theme, height=500, hovermode='x unified',
            xaxis=dict(showgrid=grid), yaxis=dict(showgrid=grid),
            legend_title_text='Metrics'
        )
        return fig

    @staticmethod
    def create_line_chart(df, metrics, title, theme, grid):
        fig = ChartGenerator._create_base_figure(title, theme, grid)
        colors = px.colors.qualitative.Set1
        for i, m in enumerate(metrics):
            fig.add_trace(go.Scatter(x=df.columns, y=df.loc[m], mode='lines+markers',
                                     name=m, line=dict(color=colors[i % len(colors)], width=3)))
        return fig

    @staticmethod
    def create_bar_chart(df, metrics, title, theme, grid):
        fig = ChartGenerator._create_base_figure(title, theme, grid)
        fig.update_layout(barmode='group')
        colors = px.colors.qualitative.Set1
        for i, m in enumerate(metrics):
            fig.add_trace(go.Bar(x=df.columns, y=df.loc[m], name=m,
                                 marker_color=colors[i % len(colors)]))
        return fig

    @staticmethod
    def create_area_chart(df, metrics, title, theme, grid):
        fig = ChartGenerator._create_base_figure(title, theme, grid)
        colors = px.colors.qualitative.Set1
        for i, m in enumerate(metrics):
            fig.add_trace(go.Scatter(x=df.columns, y=df.loc[m], mode='lines',
                                     name=m, fill='tonexty' if i > 0 else 'tozeroy',
                                     line=dict(color=colors[i % len(colors)])))
        return fig

    @staticmethod
    def create_heatmap(df, metrics, title, theme, grid):
        if len(metrics) < 2:
            st.warning("Heatmap requires at least two metrics.")
            return None
        corr = df.loc[metrics].T.corr()
        fig = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns,
                                        colorscale='RdBu', zmid=0))
        fig.update_layout(title=dict(text=title, font=dict(size=20)), template=theme, height=500)
        return fig

class DashboardUI:
    def __init__(self):
        self.init_session_state()

    def init_session_state(self):
        defaults = {
            "analysis_data": None,
            "_uploaded_file_memo": None,
            "chart_figure": None,
            "selected_metrics": [],
            "chart_type": "Line Chart",
            "show_data_quality": False,
            "show_outliers": False
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def render_sidebar(self):
        st.sidebar.title("📂 Upload Financial File")
        uploaded_file = st.sidebar.file_uploader("Upload Capitaline File", type=["html", "htm", "xls", "xlsx"])
        if uploaded_file and uploaded_file != st.session_state["_uploaded_file_memo"]:
            st.session_state["_uploaded_file_memo"] = uploaded_file
            st.session_state["analysis_data"] = parse_capitaline_file(uploaded_file)
        st.sidebar.checkbox("Show Data Quality Info", key="show_data_quality")
        st.sidebar.checkbox("Show Outliers", key="show_outliers")

    def render_main_panel(self):
        st.markdown("<div class='main-header'>📊 Financial Dashboard</div>", unsafe_allow_html=True)
        if st.session_state["analysis_data"] is None:
            st.markdown(\"\"\"
                <div class='welcome-container'>
                    <h2>Welcome to the Financial Dashboard</h2>
                    <p>Upload a Capitaline file to begin analyzing financial data visually.</p>
                </div>
            \"\"\", unsafe_allow_html=True)
            return

        data = st.session_state["analysis_data"]
        df = data["statement"]
        st.subheader(f"Company: {data['company_name']}")

        if st.session_state["show_data_quality"]:
            q = data["data_quality"]
            cls = "quality-high" if q.quality_score == "High" else "quality-medium" if q.quality_score == "Medium" else "quality-low"
            st.markdown(f\"\"\"
                <div class="feature-card">
                    <span class="data-quality-indicator {cls}"></span>
                    <b>Data Quality:</b> {q.quality_score}<br>
                    Missing: {q.missing_values} values ({q.missing_percentage:.2f}%)<br>
                    Duplicates: {q.duplicate_rows} rows
                </div>
            \"\"\", unsafe_allow_html=True)

        if st.session_state["show_outliers"] and data["outliers"]:
            st.markdown("<div class='feature-card'><b>Outliers Detected:</b>", unsafe_allow_html=True)
            for metric, indices in data["outliers"].items():
                st.markdown(f"- {metric}: {len(indices)} outlier(s)")
            st.markdown("</div>", unsafe_allow_html=True)

        metrics = df.index.tolist()
        selected = st.multiselect("Select metrics to visualize:", metrics, default=metrics[:2])
        st.session_state["selected_metrics"] = selected

        chart_type = st.selectbox("Select Chart Type:", ["Line Chart", "Bar Chart", "Area Chart", "Heatmap"])
        theme = st.selectbox("Theme:", ["plotly_white", "plotly_dark", "ggplot2"])
        grid = st.checkbox("Show Grid", value=True)

        if selected:
            if chart_type == "Line Chart":
                fig = ChartGenerator.create_line_chart(df, selected, "Financial Trend", theme, grid)
            elif chart_type == "Bar Chart":
                fig = ChartGenerator.create_bar_chart(df, selected, "Financial Bar View", theme, grid)
            elif chart_type == "Area Chart":
                fig = ChartGenerator.create_area_chart(df, selected, "Area Representation", theme, grid)
            elif chart_type == "Heatmap":
                fig = ChartGenerator.create_heatmap(df, selected, "Correlation Map", theme, grid)
            else:
                fig = None
            if fig:
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one metric to generate chart.")

# Run the app
if __name__ == "__main__":
    try:
        dashboard = DashboardUI()
        dashboard.render_sidebar()
        dashboard.render_main_panel()
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error(f"Something went wrong: {e}")
