# Enhanced Financial Dashboard - Complete Integrated Version
# A robust Streamlit application for financial data analysis with enhanced error handling,
# performance optimization, and additional features.

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
from streamlit.runtime.uploaded_file_manager import UploadedFile

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

# Custom CSS for a polished and modern look (minor update for download button)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center;
        margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 1rem;
        border-radius: 10px; color: white; margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .feature-card {
        background: white; padding: 1.5rem; border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 1rem 0;
        border-left: 4px solid #1f77b4;
    }
    .stButton>button {
        background: linear-gradient(90deg, #1f77b4, #17a2b8); color: white; border-radius: 8px;
        border: none; padding: 12px 24px; font-weight: 600; transition: all 0.3s ease;
    }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .stDownloadButton>button {
        background: linear-gradient(90deg, #28a745, #17a2b8); color: white; border-radius: 8px;
        border: none; padding: 10px 20px; font-weight: 500; margin-top: 1rem;
    }
    .data-quality-indicator {
        display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px;
    }
    .quality-high { background-color: #28a745; }
    .quality-medium { background-color: #ffc107; }
    .quality-low { background-color: #dc3545; }
</style>
""", unsafe_allow_html=True)

# --- 4. Data Structures and Classes ---

@dataclass
class DataQualityMetrics:
    """A data class to hold metrics about the quality of the dataset."""
    total_rows: int
    missing_values: int
    missing_percentage: float
    duplicate_rows: int
    quality_score: str = field(init=False)

    def __post_init__(self):
        """Calculates a qualitative score based on the percentage of missing data."""
        if self.missing_percentage < 5:
            self.quality_score = "High"
        elif self.missing_percentage < 20:
            self.quality_score = "Medium"
        else:
            self.quality_score = "Low"

class FileValidator:
    """Validates the uploaded file based on size and type."""
    @staticmethod
    def validate_file(uploaded_file: UploadedFile) -> Tuple[bool, str]:
        if uploaded_file is None:
            return False, "No file uploaded. Please upload a file in the sidebar."
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            return False, f"File size exceeds the {MAX_FILE_SIZE_MB}MB limit."
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in ALLOWED_FILE_TYPES:
            return False, f"Unsupported file type: '.{file_extension}'. Please upload one of {ALLOWED_FILE_TYPES}."
        return True, file_extension

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
    def detect_outliers(df: pd.DataFrame) -> Dict[str, List[int]]:
        outliers = {}
        numeric_df = df.select_dtypes(include=np.number)
        for col in numeric_df.columns:
            Q1, Q3 = numeric_df[col].quantile(0.25), numeric_df[col].quantile(0.75)
            IQR = Q3 - Q1
            if IQR > 0:
                lower_bound, upper_bound = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
                outlier_indices = numeric_df[(numeric_df[col] < lower_bound) | (numeric_df[col] > upper_bound)].index.tolist()
                if outlier_indices:
                    outliers[col] = outlier_indices
        return outliers

    @staticmethod
    def calculate_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
        total_cells = df.size
        if total_cells == 0:
            return DataQualityMetrics(total_rows=0, missing_values=0, missing_percentage=0.0, duplicate_rows=0)
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / total_cells) * 100 if total_cells > 0 else 0
        return DataQualityMetrics(
            total_rows=len(df), missing_values=int(missing_values),
            missing_percentage=missing_percentage, duplicate_rows=int(df.duplicated().sum())
        )

    # --- NEW: Function to normalize data to a base of 100 ---
    @staticmethod
    def normalize_to_100(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        """Normalizes selected rows of a DataFrame to a base of 100."""
        df_scaled = df.loc[metrics].copy()
        for metric in metrics:
            series = df_scaled.loc[metric].dropna()
            if not series.empty:
                base_value = series.iloc[0]
                if base_value != 0:
                    df_scaled.loc[metric] = (df_scaled.loc[metric] / base_value) * 100
                else:
                    df_scaled.loc[metric] = np.nan
        return df_scaled

    # --- NEW: Function to calculate key financial ratios ---
    @staticmethod
    @st.cache_data
    def calculate_ratios(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Calculates key financial ratios if relevant metrics are available."""
        try:
            required_metrics = {
                'Profit Margin (%)': ('Net Profit', 'Net Sales'),
                'ROE (%)': ('Net Profit', 'Shareholders Funds'),
                'Current Ratio': ('Current Assets', 'Current Liabilities'),
                'Debt to Equity': ('Total Debt', 'Shareholders Funds')
            }
            ratios = {}
            for ratio_name, (num_metric, den_metric) in required_metrics.items():
                if num_metric in df.index and den_metric in df.index:
                    num = df.loc[num_metric]
                    den = df.loc[den_metric]
                    ratio = (num / den) * 100 if '%' in ratio_name else (num / den)
                    ratios[ratio_name] = ratio.dropna()
            if not ratios:
                return None
            return pd.DataFrame(ratios).T
        except Exception as e:
            logger.warning(f"Error calculating ratios: {e}")
            return None

class ChartGenerator:
    """A factory for creating various Plotly charts."""
    
    # --- UPDATED: To accept custom y-axis title ---
    @staticmethod
    def _create_base_figure(title: str, theme: str, show_grid: bool, yaxis_title: str) -> go.Figure:
        fig = go.Figure()
        fig.update_layout(
            title={'text': title, 'font': {'size': 20}, 'x': 0.5},
            xaxis_title="Year",
            yaxis_title=yaxis_title, # Use the provided title
            template=theme, height=500, hovermode='x unified',
            xaxis={'showgrid': show_grid}, yaxis={'showgrid': show_grid},
            legend_title_text='Metrics'
        )
        return fig

    # --- UPDATED: All chart functions now accept scale_type, yaxis_title, and outliers for highlighting ---
    @staticmethod
    def create_line_chart(df: pd.DataFrame, metrics: List[str], title: str, theme: str, show_grid: bool, scale_type: str, yaxis_title: str, outliers: Dict[str, List[int]]) -> go.Figure:
        fig = ChartGenerator._create_base_figure(title, theme, show_grid, yaxis_title)
        colors = px.colors.qualitative.Plotly
        for i, metric in enumerate(metrics):
            x = df.columns
            y = df.loc[metric]
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines+markers', name=metric,
                                     line={'color': colors[i % len(colors)], 'width': 3}))
            # Highlight outliers
            if metric in outliers:
                outlier_indices = outliers[metric]
                outlier_x = [x[j] for j in outlier_indices if j < len(x)]
                outlier_y = [y.iloc[j] for j in outlier_indices if j < len(y)]
                fig.add_trace(go.Scatter(x=outlier_x, y=outlier_y, mode='markers', name=f"{metric} Outliers",
                                         marker={'color': 'red', 'size': 10, 'symbol': 'x'}))
        fig.update_xaxes(categoryorder='array', categoryarray=df.columns)
        if scale_type == 'Logarithmic':
            fig.update_layout(yaxis_type='log')
        return fig

    @staticmethod
    def create_bar_chart(df: pd.DataFrame, metrics: List[str], title: str, theme: str, show_grid: bool, scale_type: str, yaxis_title: str, outliers: Dict[str, List[int]]) -> go.Figure:
        fig = ChartGenerator._create_base_figure(title, theme, show_grid, yaxis_title)
        fig.update_layout(barmode='group')
        colors = px.colors.qualitative.Plotly
        for i, metric in enumerate(metrics):
            x = df.columns
            y = df.loc[metric]
            fig.add_trace(go.Bar(x=x, y=y, name=metric,
                                 marker_color=colors[i % len(colors)]))
            # Highlight outliers (as text labels for bars)
            if metric in outliers:
                for j in outliers[metric]:
                    if j < len(x):
                        fig.add_annotation(x=x[j], y=y.iloc[j], text="Outlier", showarrow=True, arrowhead=1)
        fig.update_xaxes(categoryorder='array', categoryarray=df.columns)
        if scale_type == 'Logarithmic':
            fig.update_layout(yaxis_type='log')
        return fig

    @staticmethod
    def create_area_chart(df: pd.DataFrame, metrics: List[str], title: str, theme: str, show_grid: bool, scale_type: str, yaxis_title: str, outliers: Dict[str, List[int]]) -> go.Figure:
        fig = ChartGenerator._create_base_figure(title, theme, show_grid, yaxis_title)
        colors = px.colors.qualitative.Plotly
        for i, metric in enumerate(metrics):
            x = df.columns
            y = df.loc[metric]
            fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=metric,
                                     fill='tonexty' if i > 0 else 'tozeroy',
                                     line={'color': colors[i % len(colors)]}))
            # Highlight outliers
            if metric in outliers:
                outlier_indices = outliers[metric]
                outlier_x = [x[j] for j in outlier_indices if j < len(x)]
                outlier_y = [y.iloc[j] for j in outlier_indices if j < len(y)]
                fig.add_trace(go.Scatter(x=outlier_x, y=outlier_y, mode='markers', name=f"{metric} Outliers",
                                         marker={'color': 'red', 'size': 10, 'symbol': 'x'}))
        fig.update_xaxes(categoryorder='array', categoryarray=df.columns)
        if scale_type == 'Logarithmic':
            fig.update_layout(yaxis_type='log')
        return fig

    @staticmethod
    def create_heatmap(df: pd.DataFrame, metrics: List[str], title: str, theme: str, **kwargs) -> Optional[go.Figure]:
        if len(metrics) < 2:
            st.warning("Heatmap requires at least two metrics for correlation.")
            return None
        corr_matrix = df.loc[metrics].T.corr()
        fig = go.Figure(data=go.Heatmap(z=corr_matrix.values, x=corr_matrix.columns, y=corr_matrix.columns,
                                        colorscale='RdBu_r', zmid=0))
        fig.update_layout(title={'text': title, 'font': {'size': 20}, 'x': 0.5}, template=theme, height=500)
        return fig

# --- 5. Core Application Logic ---

@st.cache_data(show_spinner="Parsing and analyzing your file...")
def parse_financial_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    if uploaded_file is None: return None
    is_valid, file_info = FileValidator.validate_file(uploaded_file)
    if not is_valid:
        st.error(file_info)
        return None
    try:
        content = uploaded_file.getvalue()
        df = pd.read_html(io.BytesIO(content), header=[0, 1])[0]
        company_name = "Unknown Company"
        try:
            header_str = str(df.columns[0][0])
            if ">>" in header_str: company_name = header_str.split(">>")[2].split("(")[0].strip()
        except IndexError: logger.warning("Could not parse company name from header.")
        df.columns = [str(col[1]) for col in df.columns]
        df = df.rename(columns={df.columns[0]: "Metric"}).dropna(subset=["Metric"])
        df = df.reset_index(drop=True)
        is_duplicate = df.duplicated(subset=['Metric'], keep=False)
        df['unique_metric_id'] = df['Metric']
        df.loc[is_duplicate, 'unique_metric_id'] = df['Metric'] + ' (row ' + (df.index + 1).astype(str) + ')'
        df = df.set_index('unique_metric_id').drop(columns=['Metric'])
        year_cols_map = {col: YEAR_REGEX.search(col).group(0) for col in df.columns if YEAR_REGEX.search(col)}
        df = df.rename(columns=year_cols_map)
        year_columns = sorted([col for col in df.columns if col.isdigit()], key=int)
        if not year_columns:
            st.error("No valid year columns found.")
            return None
        df_processed = df[year_columns].copy()
        df_processed = DataProcessor.clean_numeric_data(df_processed).dropna(how='all')
        return {"statement": df_processed, "company_name": company_name, "data_quality": DataProcessor.calculate_data_quality(df_processed),
                "outliers": DataProcessor.detect_outliers(df_processed), "year_columns": year_columns,
                "file_info": {"name": uploaded_file.name, "size": uploaded_file.size, "type": file_info}}
    except Exception as e:
        logger.error(f"Error parsing file '{uploaded_file.name}': {e}", exc_info=True)
        st.error(f"An unexpected error occurred while parsing the file. Please check if the file format is correct. Error: {e}")
        return None

class DashboardApp:
    def __init__(self):
        self._initialize_state()
        self.CHART_BUILDERS = {"Line Chart": ChartGenerator.create_line_chart, "Bar Chart": ChartGenerator.create_bar_chart,
                               "Area Chart": ChartGenerator.create_area_chart, "Heatmap": ChartGenerator.create_heatmap}

    def _initialize_state(self):
        defaults = {"analysis_data": None, "uploaded_file_id": None}
        for key, value in defaults.items():
            if key not in st.session_state: st.session_state[key] = value

    def run(self):
        self.render_sidebar()
        self.render_main_panel()

    def render_sidebar(self):
        st.sidebar.title("📂 Upload & Options")
        uploaded_file = st.sidebar.file_uploader("Upload a Capitaline File", type=ALLOWED_FILE_TYPES)
        if uploaded_file and uploaded_file.file_id != st.session_state.get("uploaded_file_id"):
            st.session_state.uploaded_file_id = uploaded_file.file_id
            st.session_state.analysis_data = parse_financial_file(uploaded_file)
        st.sidebar.title("⚙️ Display Settings")
        st.sidebar.checkbox("Show Data Quality Info", key="show_data_quality")
        st.sidebar.checkbox("Show Outlier Summary", key="show_outliers")

    def render_main_panel(self):
        st.markdown("<div class='main-header'>📊 Financial Analysis Dashboard</div>", unsafe_allow_html=True)
        if not st.session_state.analysis_data:
            st.info("👋 Welcome! Please upload a financial data file using the sidebar to begin analysis.")
            return
        data = st.session_state.analysis_data
        df = data["statement"]
        st.subheader(f"Company Analysis: {data['company_name']}")
        self._display_metadata(data)
        # --- UPDATED: Added new tab for ratios ---
        tab_viz, tab_data, tab_ratios = st.tabs(["📊 Visualizations", "📄 Data Table", "📈 Ratios & Insights"])
        with tab_viz: self._render_visualization_tab(df, data)
        with tab_data: self._render_data_table_tab(df)
        with tab_ratios: self._render_ratios_tab(df)

    def _render_visualization_tab(self, df: pd.DataFrame, data: Dict[str, Any]):
        st.header("Financial Charts")
        available_metrics = df.index.tolist()
        year_columns = data["year_columns"]
        
        col1, col2, col3, col4, col5 = st.columns([3, 2, 2, 2, 2])  # Added extra column for year selection
        with col1:
            selected_metrics = st.multiselect("Select metrics to visualize:", options=available_metrics,
                                              default=available_metrics[:2] if len(available_metrics) > 1 else available_metrics)
        with col2:
            chart_type = st.selectbox("Select Chart Type:", self.CHART_BUILDERS.keys())
        with col3:
            theme = st.selectbox("Chart Theme:", ["plotly_white", "plotly_dark", "ggplot2"])
        with col4:
            # --- NEW: UI for Y-Axis Scaling ---
            # Disable the scaling option if Heatmap is selected
            is_heatmap = chart_type == 'Heatmap'
            scale_type = st.selectbox(
                "Y-Axis Scale:",
                ["Linear (Default)", "Logarithmic", "Normalized (Base 100)"],
                disabled=is_heatmap,
                help="Logarithmic scale is useful for data with exponential growth. Normalized scale is useful for comparing the performance of different metrics from a common base."
            )
        with col5:
            # --- NEW: Year selection ---
            selected_years = st.multiselect("Select years:", options=year_columns, default=year_columns,
                                            help="Filter data to specific years for focused analysis.")
        
        show_grid = st.checkbox("Show Chart Gridlines", value=True)

        if not selected_metrics:
            st.warning("Please select at least one metric to generate a chart.")
            return
        if not selected_years:
            st.warning("Please select at least one year to generate a chart.")
            return

        # --- NEW: Filter dataframe by selected years ---
        filtered_df = df[selected_years]

        # --- NEW: Logic to prepare data and titles based on scale type ---
        plot_df = filtered_df
        yaxis_title = "Amount (₹ Cr.)"
        if scale_type == "Normalized (Base 100)" and not is_heatmap:
            plot_df = DataProcessor.normalize_to_100(filtered_df, selected_metrics)
            yaxis_title = "Normalized Value (Base 100)"

        chart_builder = self.CHART_BUILDERS[chart_type]
        fig = chart_builder(
            df=plot_df,  # Use the potentially scaled and filtered dataframe
            metrics=selected_metrics,
            title=f"{chart_type} of Selected Financials",
            theme=theme,
            show_grid=show_grid,
            scale_type=scale_type, # Pass the scale type to the builder
            yaxis_title=yaxis_title, # Pass the correct axis title
            outliers=data["outliers"]  # Pass outliers for highlighting
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)

    def _render_data_table_tab(self, df: pd.DataFrame):
        st.header("Financial Data")
        st.info("This table shows the cleaned financial data used for the visualizations.")
        formatted_df = df.style.format("{:,.2f}", na_rep="-")
        st.dataframe(formatted_df, use_container_width=True)
        # --- NEW: Download button for exporting data ---
        csv = df.to_csv().encode('utf-8')
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name=f"{st.session_state.analysis_data['company_name']}_financial_data.csv",
            mime="text/csv",
            help="Export the cleaned data for further analysis in tools like Excel."
        )

    # --- NEW: Render tab for ratios and insights ---
    def _render_ratios_tab(self, df: pd.DataFrame):
        st.header("Key Financial Ratios & Insights")
        ratios_df = DataProcessor.calculate_ratios(df)
        if ratios_df is not None:
            st.info("Automatically calculated ratios based on available metrics. Values are per year.")
            formatted_ratios = ratios_df.style.format("{:,.2f}", na_rep="-")
            st.dataframe(formatted_ratios, use_container_width=True)
            # Quick insight
            for ratio in ratios_df.index:
                avg = ratios_df.loc[ratio].mean()
                st.markdown(f"• **{ratio}** Average: {avg:.2f}")
        else:
            st.warning("Insufficient metrics available to calculate ratios (e.g., need 'Net Profit' and 'Net Sales'). Upload a file with more data.")

    def _display_metadata(self, data: Dict[str, Any]):
        if st.session_state.show_data_quality:
            dq = data["data_quality"]
            quality_class = f"quality-{dq.quality_score.lower()}"
            st.markdown(f"""<div class="feature-card"><h4><span class="data-quality-indicator {quality_class}"></span>Data Quality: {dq.quality_score}</h4><ul>
                        <li><b>Total Rows:</b> {dq.total_rows}</li>
                        <li><b>Missing Values:</b> {dq.missing_values} ({dq.missing_percentage:.2f}%)</li>
                        <li><b>Duplicate Rows:</b> {dq.duplicate_rows}</li></ul></div>""", unsafe_allow_html=True)
        if st.session_state.show_outliers and data["outliers"]:
            st.markdown("<div class='feature-card'><h4>Outlier Summary (by metric)</h4>", unsafe_allow_html=True)
            for metric, indices in data["outliers"].items():
                st.markdown(f"• <b>{metric}:</b> {len(indices)} potential outlier(s) detected.")
            st.markdown("</div>", unsafe_allow_html=True)

# --- 6. App Execution ---
if __name__ == "__main__":
    try:
        app = DashboardApp()
        app.run()
    except Exception as e:
        logger.critical(f"A critical error occurred in the app: {e}", exc_info=True)
        st.error(f"A critical error occurred. Please refresh the page. Details: {e}")
