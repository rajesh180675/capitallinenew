# Enhanced Financial Dashboard - v2.0 with Advanced Features
# A robust Streamlit application for financial data analysis with export, ratio calculations,
# trend analysis, and comparison mode.

# --- 1. Imports and Setup ---
import io
import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

# --- 2. Configuration ---
# Suppress warnings for a cleaner user interface
warnings.filterwarnings('ignore')

# Configure logging for debugging and monitoring
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants for file validation and data processing
MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_TYPES = ['html', 'htm', 'xls', 'xlsx']
YEAR_REGEX = re.compile(r'\b(19[8-9]\d|20\d\d)\b') # Regex to find a 4-digit year

# --- 3. Page and Style Configuration ---
st.set_page_config(
    page_title="Advanced Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a polished and modern look
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center;
        margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .feature-card {
        background: white; padding: 1.5rem; border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1f77b4, #17a2b8); color: white; border-radius: 8px;
        border: none; padding: 10px 20px; font-weight: 600; transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .data-quality-indicator {
        display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;
    }
    .quality-high { background-color: #28a745; }
    .quality-medium { background-color: #ffc107; }
    .quality-low { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# --- 4. Data Structures and Helper Classes ---

@dataclass
class DataQualityMetrics:
    """A data class to hold metrics about the quality of the dataset."""
    total_rows: int; missing_values: int; missing_percentage: float; duplicate_rows: int
    quality_score: str = field(init=False)
    def __post_init__(self):
        if self.missing_percentage < 5: self.quality_score = "High"
        elif self.missing_percentage < 20: self.quality_score = "Medium"
        else: self.quality_score = "Low"

class FileValidator:
    """Validates the uploaded file based on size and type."""
    @staticmethod
    def validate_file(uploaded_file) -> Tuple[bool, str]:
        if uploaded_file is None: return False, "No file uploaded."
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024: return False, f"File size > {MAX_FILE_SIZE_MB}MB."
        file_ext = uploaded_file.name.split('.')[-1].lower()
        if file_ext not in ALLOWED_FILE_TYPES: return False, f"Unsupported file type: '.{file_ext}'."
        return True, file_ext

class DataProcessor:
    """Handles data cleaning, transformation, and analysis."""
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace(r'[,\(\)₹]|Rs\.', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    @staticmethod
    def calculate_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
        total_cells = df.size
        if total_cells == 0: return DataQualityMetrics(0, 0, 0.0, 0)
        missing = df.isnull().sum().sum()
        return DataQualityMetrics(len(df), int(missing), (missing / total_cells) * 100, int(df.duplicated().sum()))

# ENHANCEMENT 2: Financial Ratio Calculations
class FinancialCalculator:
    """Calculates common financial ratios."""
    @staticmethod
    def calculate_ratios(df: pd.DataFrame) -> pd.DataFrame:
        ratios = pd.DataFrame(index=df.columns)
        try:
            if 'Net Profit' in df.index and 'Total Revenue' in df.index and df.loc['Total Revenue'].ne(0).all():
                ratios['Net Profit Margin (%)'] = (df.loc['Net Profit'] / df.loc['Total Revenue']) * 100
            if 'Current Assets' in df.index and 'Current Liabilities' in df.index and df.loc['Current Liabilities'].ne(0).all():
                ratios['Current Ratio'] = df.loc['Current Assets'] / df.loc['Current Liabilities']
            if 'Total Liabilities' in df.index and 'Total Equity' in df.index and df.loc['Total Equity'].ne(0).all():
                ratios['Debt-to-Equity Ratio'] = df.loc['Total Liabilities'] / df.loc['Total Equity']
        except (KeyError, ZeroDivisionError) as e:
            logger.warning(f"Could not calculate some ratios: {e}")
        return ratios.T.dropna(how='all')

class ChartGenerator:
    """A factory for creating various Plotly charts."""
    @staticmethod
    def _create_base_figure(title: str, theme: str, show_grid: bool) -> go.Figure:
        return go.Figure().update_layout(title={'text': title, 'x': 0.5}, xaxis_title="Year", template=theme, height=500, hovermode='x unified', xaxis={'showgrid': show_grid}, yaxis={'showgrid': show_grid})
    
    # ... Other chart methods (Line, Bar, Area, Heatmap) remain the same ...
    @staticmethod
    def create_line_chart(df, metrics, title, theme, show_grid):
        fig = ChartGenerator._create_base_figure(title, theme, show_grid)
        colors = px.colors.qualitative.Plotly
        for i, metric in enumerate(metrics): fig.add_trace(go.Scatter(x=df.columns, y=df.loc[metric], mode='lines+markers', name=metric, line={'color': colors[i % len(colors)]}))
        return fig

    # ENHANCEMENT 6: Advanced Visualizations (Waterfall Chart)
    @staticmethod
    def create_waterfall_chart(df: pd.DataFrame, metrics: List[str], title: str, theme: str, **kwargs) -> go.Figure:
        metric = metrics[0] # Waterfall only uses the first selected metric
        y_values = df.loc[metric].values
        years = df.columns.tolist()
        
        fig = go.Figure(go.Waterfall(
            name=metric, orientation="v",
            measure=["absolute"] + ["relative"] * (len(y_values) - 1),
            x=years,
            text=[f"{v:,.2f}" for v in y_values],
            y=y_values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
        ))
        fig.update_layout(title=title, template=theme, showlegend=True)
        return fig

# --- 5. Core Application Logic ---

# ENHANCEMENT 5: Performance Optimization (TTL cache)
@st.cache_data(show_spinner="Parsing and analyzing your file...", ttl=3600)
def parse_financial_file(uploaded_file) -> Optional[Dict[str, Any]]:
    # ... Parsing logic is largely the same, but now includes ratio calculation ...
    is_valid, file_info = FileValidator.validate_file(uploaded_file)
    if not is_valid: st.error(file_info); return None
    try:
        df = pd.read_html(io.BytesIO(uploaded_file.getvalue()), header=[0, 1])[0]
        company_name = "Unknown Company"
        try:
            header_str = str(df.columns[0][0])
            if ">>" in header_str: company_name = header_str.split(">>")[2].split("(")[0].strip()
        except IndexError: pass
        df.columns = [str(col[1]) for col in df.columns]
        df = df.rename(columns={df.columns[0]: "Metric"}).dropna(subset=["Metric"]).set_index("Metric")
        year_cols_map = {col: YEAR_REGEX.search(col).group(0) for col in df.columns if YEAR_REGEX.search(col)}
        df = df.rename(columns=year_cols_map)
        year_columns = sorted([col for col in df.columns if col.isdigit()], reverse=True)
        if not year_columns: st.error("No valid year columns found."); return None
        df_processed = DataProcessor.clean_numeric_data(df[year_columns].copy()).dropna(how='all')
        
        # Integrate ratio calculation
        ratios_df = FinancialCalculator.calculate_ratios(df_processed)

        return {
            "statement": df_processed, "company_name": company_name,
            "data_quality": DataProcessor.calculate_data_quality(df_processed),
            "ratios": ratios_df, "file_name": uploaded_file.name
        }
    except Exception as e:
        logger.error(f"Error parsing file '{uploaded_file.name}': {e}", exc_info=True)
        st.error(f"Error parsing file. Ensure it's a valid Capitaline export. Details: {e}")
        return None

# ENHANCEMENT 3: Trend Analysis
def analyze_trends(df: pd.DataFrame, metric: str) -> Optional[Dict[str, float]]:
    """Calculate trend statistics for a single metric."""
    if metric not in df.index: return None
    
    y_values = df.loc[metric].dropna().values
    if len(y_values) < 2: return None # Not enough data for trend
    
    x_values = np.arange(len(y_values))
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_values, y_values)
    
    cagr = None
    if y_values[0] > 0 and y_values[-1] > 0:
        cagr = (pow(y_values[-1] / y_values[0], 1/len(y_values)) - 1) * 100
    
    return {'slope': slope, 'r_squared': r_value**2, 'p_value': p_value, 'cagr': cagr}

class DashboardApp:
    def __init__(self):
        self._initialize_state()
        self.CHART_BUILDERS = {
            "Line Chart": ChartGenerator.create_line_chart,
            "Bar Chart": ChartGenerator.create_line_chart, # Placeholder, replace with actual bar chart if needed
            "Area Chart": ChartGenerator.create_line_chart, # Placeholder, replace with actual area chart if needed
            "Waterfall Chart": ChartGenerator.create_waterfall_chart,
        }

    def _initialize_state(self):
        defaults = {"analysis_data": None, "comparison_mode": False, "comparison_data": []}
        for k, v in defaults.items():
            if k not in st.session_state: st.session_state[k] = v

    def run(self):
        self.render_sidebar()
        self.render_main_panel()

    def render_sidebar(self):
        st.sidebar.title("📂 Upload & Options")
        
        # ENHANCEMENT 4: Comparison Mode
        st.session_state.comparison_mode = st.sidebar.checkbox("Enable Comparison Mode")
        
        if st.session_state.comparison_mode:
            uploaded_files = st.sidebar.file_uploader(
                "Upload files for comparison", type=ALLOWED_FILE_TYPES, accept_multiple_files=True)
            if uploaded_files:
                st.session_state.comparison_data = [parse_financial_file(f) for f in uploaded_files if f is not None]
        else:
            uploaded_file = st.sidebar.file_uploader("Upload a Capitaline File", type=ALLOWED_FILE_TYPES)
            if uploaded_file:
                st.session_state.analysis_data = parse_financial_file(uploaded_file)
        
        st.sidebar.title("⚙️ Display Settings")
        st.sidebar.checkbox("Show Data Quality Info", key="show_data_quality", value=True)

    def render_main_panel(self):
        st.markdown("<div class='main-header'>📊 Financial Analysis Dashboard</div>", unsafe_allow_html=True)
        if st.session_state.comparison_mode:
            self._render_comparison_panel()
        elif st.session_state.analysis_data:
            self._render_single_company_panel()
        else:
            st.info("👋 Welcome! Please upload a financial data file using the sidebar to begin.")

    def _render_single_company_panel(self):
        data = st.session_state.analysis_data
        st.subheader(f"Company Analysis: {data['company_name']}")
        if st.session_state.show_data_quality: self._display_metadata(data)

        tab_viz, tab_data, tab_ratios, tab_trends = st.tabs(["📊 Visualizations", "📄 Data Table", "📈 Ratios", "🔍 Trend Analysis"])
        
        with tab_viz: self._render_visualization_tab(data)
        with tab_data: self._render_data_table_tab(data)
        with tab_ratios: self._render_ratios_tab(data)
        with tab_trends: self._render_trends_tab(data)
    
    def _render_comparison_panel(self):
        st.subheader("🏢 Company Comparison")
        if not st.session_state.comparison_data:
            st.info("Upload two or more files in the sidebar to compare them.")
            return

        all_data = [d for d in st.session_state.comparison_data if d is not None]
        if len(all_data) < 2:
            st.warning("Please upload at least two valid files for comparison.")
            return

        # Find common metrics
        common_metrics = set.intersection(*[set(d['statement'].index) for d in all_data])
        if not common_metrics:
            st.error("No common metrics found across the uploaded files.")
            return
            
        selected_metric = st.selectbox("Select a metric to compare:", sorted(list(common_metrics)))

        # Create comparison chart
        fig = go.Figure()
        for data in all_data:
            company_name = data['company_name']
            df = data['statement']
            if selected_metric in df.index:
                fig.add_trace(go.Scatter(x=df.columns, y=df.loc[selected_metric], name=company_name, mode='lines+markers'))
        
        fig.update_layout(title=f'Comparison for: {selected_metric}', xaxis_title='Year', yaxis_title='Amount')
        st.plotly_chart(fig, use_container_width=True)


    def _render_visualization_tab(self, data):
        st.header("Financial Charts")
        df = data['statement']
        available_metrics = df.index.tolist()
        
        chart_type = st.selectbox("Select Chart Type:", self.CHART_BUILDERS.keys())
        
        default_metrics = available_metrics[:1] if chart_type == "Waterfall Chart" else available_metrics[:2]
        selected_metrics = st.multiselect("Select metrics:", options=available_metrics, default=default_metrics)

        if not selected_metrics:
            st.warning("Please select at least one metric."); return

        if chart_type == "Waterfall Chart" and len(selected_metrics) > 1:
            st.info("Waterfall chart displays only the first selected metric.")
        
        fig = self.CHART_BUILDERS[chart_type](df, selected_metrics, f"{chart_type} for {selected_metrics[0]}", "plotly_white", True)
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            # ENHANCEMENT 1: Export Chart
            buffer = io.BytesIO()
            fig.write_image(buffer, format='png', scale=2)
            st.download_button(label="📥 Download Chart as PNG", data=buffer, file_name=f"{data['company_name']}_{chart_type}.png", mime="image/png")

    def _render_data_table_tab(self, data):
        st.header("Financial Data")
        df = data['statement']
        st.dataframe(df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)
        # ENHANCEMENT 1: Export Data
        st.download_button(label="📥 Download Data as CSV", data=df.to_csv(), file_name=f"{data['company_name']}_data.csv", mime="text/csv")

    def _render_ratios_tab(self, data):
        st.header("Key Financial Ratios")
        ratios_df = data.get('ratios')
        if ratios_df is not None and not ratios_df.empty:
            st.dataframe(ratios_df.style.format("{:,.2f}"), use_container_width=True)
        else:
            st.info("Could not calculate key ratios. Required metrics (e.g., 'Net Profit', 'Current Assets') might be missing or named differently.")

    def _render_trends_tab(self, data):
        st.header("Metric Trend Analysis")
        df = data['statement']
        metric_to_analyze = st.selectbox("Select a metric for trend analysis:", df.index)
        
        trend_data = analyze_trends(df, metric_to_analyze)
        if trend_data:
            c1, c2, c3 = st.columns(3)
            cagr = f"{trend_data['cagr']:.2f}%" if trend_data['cagr'] is not None else "N/A"
            c1.metric("CAGR (Compound Annual Growth Rate)", cagr)
            c2.metric("R-squared (Goodness of Fit)", f"{trend_data['r_squared']:.3f}")
            c3.metric("P-value (Significance)", f"{trend_data['p_value']:.3f}", help="A low p-value (< 0.05) indicates a statistically significant trend.")
        else:
            st.warning(f"Not enough data to analyze the trend for '{metric_to_analyze}'.")

    def _display_metadata(self, data):
        dq = data["data_quality"]
        quality_class = f"quality-{dq.quality_score.lower()}"
        st.markdown(f'<div class="feature-card"><h4><span class="data-quality-indicator {quality_class}"></span>Data Quality: {dq.quality_score}</h4><ul><li><b>Missing Values:</b> {dq.missing_values} ({dq.missing_percentage:.2f}%)</li><li><b>Duplicate Rows:</b> {dq.duplicate_rows}</li></ul></div>', unsafe_allow_html=True)

# --- 6. App Execution ---
if __name__ == "__main__":
    try:
        app = DashboardApp()
        app.run()
    except Exception as e:
        logger.critical(f"A critical error occurred in the app: {e}", exc_info=True)
        st.error(f"A critical error occurred. Please refresh the page. Details: {e}")
