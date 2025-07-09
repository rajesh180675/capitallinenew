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
    def validate_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Tuple[bool, str]:
        """
        Validates the uploaded file.
        Returns: A tuple (is_valid: bool, message: str).
        """
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
        """Converts object columns to numeric, removing currency symbols and parentheses."""
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                # Regex to remove currency symbols, commas, and handle negative numbers in parentheses
                df[col] = df[col].astype(str).str.replace(r'[,\(\)₹]|Rs\.', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    @staticmethod
    def detect_outliers(df: pd.DataFrame) -> Dict[str, List[int]]:
        """Detects outliers in numeric columns using the IQR method."""
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
        """Calculates and returns a DataQualityMetrics object for a DataFrame."""
        total_cells = df.size
        if total_cells == 0:
            return DataQualityMetrics(total_rows=0, missing_values=0, missing_percentage=0.0, duplicate_rows=0)
        
        missing_values = df.isnull().sum().sum()
        missing_percentage = (missing_values / total_cells) * 100 if total_cells > 0 else 0
        
        return DataQualityMetrics(
            total_rows=len(df),
            missing_values=int(missing_values),
            missing_percentage=missing_percentage,
            duplicate_rows=int(df.duplicated().sum())
        )

class ChartGenerator:
    """A factory for creating various Plotly charts."""
    
    @staticmethod
    def _create_base_figure(title: str, theme: str, show_grid: bool) -> go.Figure:
        """Creates a base Plotly figure with common layout settings."""
        fig = go.Figure()
        fig.update_layout(
            title={'text': title, 'font': {'size': 20}, 'x': 0.5},
            xaxis_title="Year",
            yaxis_title="Amount (₹ Cr.)",
            template=theme,
            height=500,
            hovermode='x unified',
            xaxis={'showgrid': show_grid},
            yaxis={'showgrid': show_grid},
            legend_title_text='Metrics'
        )
        return fig

    @staticmethod
    def create_line_chart(df: pd.DataFrame, metrics: List[str], title: str, theme: str, show_grid: bool) -> go.Figure:
        fig = ChartGenerator._create_base_figure(title, theme, show_grid)
        colors = px.colors.qualitative.Plotly
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Scatter(x=df.columns, y=df.loc[metric], mode='lines+markers', name=metric,
                                     line={'color': colors[i % len(colors)], 'width': 3}))
        fig.update_xaxes(categoryorder='array', categoryarray=df.columns)
        return fig

    @staticmethod
    def create_bar_chart(df: pd.DataFrame, metrics: List[str], title: str, theme: str, show_grid: bool) -> go.Figure:
        fig = ChartGenerator._create_base_figure(title, theme, show_grid)
        fig.update_layout(barmode='group')
        colors = px.colors.qualitative.Plotly
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Bar(x=df.columns, y=df.loc[metric], name=metric,
                                 marker_color=colors[i % len(colors)]))
        fig.update_xaxes(categoryorder='array', categoryarray=df.columns)
        return fig

    @staticmethod
    def create_area_chart(df: pd.DataFrame, metrics: List[str], title: str, theme: str, show_grid: bool) -> go.Figure:
        fig = ChartGenerator._create_base_figure(title, theme, show_grid)
        colors = px.colors.qualitative.Plotly
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Scatter(x=df.columns, y=df.loc[metric], mode='lines', name=metric,
                                     fill='tonexty' if i > 0 else 'tozeroy',
                                     line={'color': colors[i % len(colors)]}))
        fig.update_xaxes(categoryorder='array', categoryarray=df.columns)
        return fig

    @staticmethod
    def create_heatmap(df: pd.DataFrame, metrics: List[str], title: str, theme: str, **kwargs) -> Optional[go.Figure]:
        if len(metrics) < 2:
            st.warning("Heatmap requires at least two metrics for correlation.")
            return None
        
        corr_matrix = df.loc[metrics].T.corr()
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu_r',  # Reversed Red-Blue for intuitive correlation
            zmid=0
        ))
        fig.update_layout(title={'text': title, 'font': {'size': 20}, 'x': 0.5}, template=theme, height=500)
        return fig

# --- 5. Core Application Logic ---

@st.cache_data(show_spinner="Parsing and analyzing your file...")
def parse_financial_file(uploaded_file: st.runtime.uploaded_file_manager.UploadedFile) -> Optional[Dict[str, Any]]:
    """
    Parses an uploaded Capitaline file, extracts data, and performs initial analysis.
    This version creates unique identifiers for metrics with duplicate names.
    """
    if uploaded_file is None:
        return None
    
    is_valid, file_info = FileValidator.validate_file(uploaded_file)
    if not is_valid:
        st.error(file_info)
        return None

    try:
        content = uploaded_file.getvalue()
        df = pd.read_html(io.BytesIO(content), header=[0, 1])[0]

        # --- Data Extraction and Cleaning ---
        company_name = "Unknown Company"
        try:
            header_str = str(df.columns[0][0])
            if ">>" in header_str:
                company_name = header_str.split(">>")[2].split("(")[0].strip()
        except IndexError:
            logger.warning("Could not parse company name from header.")

        # Flatten multi-index columns and rename the first column to "Metric"
        df.columns = [str(col[1]) for col in df.columns]
        df = df.rename(columns={df.columns[0]: "Metric"}).dropna(subset=["Metric"])
        
        # --- FIX: Create a unique index to handle duplicate metric names ---
        # Reset index to ensure it's a clean 0, 1, 2... sequence for referencing
        df = df.reset_index(drop=True)
        
        # Find which 'Metric' names appear more than once
        is_duplicate = df.duplicated(subset=['Metric'], keep=False)
        
        # Create a new column for the unique index. Default to the original name.
        df['unique_metric_id'] = df['Metric']
        
        # For the duplicated rows, append the row number to make the name unique
        # e.g., "Other Income" at row 23 becomes "Other Income (row 24)"
        df.loc[is_duplicate, 'unique_metric_id'] = df['Metric'] + ' (row ' + (df.index + 1).astype(str) + ')'
        
        # Set this new unique column as the DataFrame's index
        df = df.set_index('unique_metric_id')
        # We can now drop the original 'Metric' column as it's redundant
        df = df.drop(columns=['Metric'])
        # --- End of fix ---

        # Identify and rename year columns using regex for robustness
        year_cols_map = {col: YEAR_REGEX.search(col).group(0) for col in df.columns if YEAR_REGEX.search(col)}
        df = df.rename(columns=year_cols_map)

        # Sort years as integers to ensure correct chronological order
        year_columns = sorted([col for col in df.columns if col.isdigit()], key=int)
        if not year_columns:
            st.error("No valid year columns (e.g., '2023', '2022') were found in the file.")
            return None

        # Reorder the DataFrame based on the correctly sorted years
        df_processed = df[year_columns].copy()
        df_processed = DataProcessor.clean_numeric_data(df_processed).dropna(how='all')

        return {
            "statement": df_processed,
            "company_name": company_name,
            "data_quality": DataProcessor.calculate_data_quality(df_processed),
            "outliers": DataProcessor.detect_outliers(df_processed),
            "year_columns": year_columns,
            "file_info": {"name": uploaded_file.name, "size": uploaded_file.size, "type": file_info}
        }
    except Exception as e:
        logger.error(f"Error parsing file '{uploaded_file.name}': {e}", exc_info=True)
        st.error(f"An unexpected error occurred while parsing the file. Please check if the file format is correct. Error: {e}")
        return None

class DashboardApp:
    """Encapsulates the entire Streamlit dashboard UI and state management."""

    def __init__(self):
        """Initializes the app and its session state."""
        self._initialize_state()
        # Map chart names to their generator functions for clean, dynamic dispatch
        self.CHART_BUILDERS = {
            "Line Chart": ChartGenerator.create_line_chart,
            "Bar Chart": ChartGenerator.create_bar_chart,
            "Area Chart": ChartGenerator.create_area_chart,
            "Heatmap": ChartGenerator.create_heatmap,
        }

    def _initialize_state(self):
        """Sets up the default session state values."""
        defaults = {
            "analysis_data": None,
            "uploaded_file_id": None,
        }
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def run(self):
        """Main method to render the entire dashboard."""
        self.render_sidebar()
        self.render_main_panel()

    def render_sidebar(self):
        """Renders the sidebar for file uploads and display options."""
        st.sidebar.title("📂 Upload & Options")
        uploaded_file = st.sidebar.file_uploader(
            "Upload a Capitaline File",
            type=ALLOWED_FILE_TYPES
        )

        # Process file only if it's new
        if uploaded_file and uploaded_file.file_id != st.session_state.get("uploaded_file_id"):
            st.session_state.uploaded_file_id = uploaded_file.file_id
            st.session_state.analysis_data = parse_financial_file(uploaded_file)
        
        st.sidebar.title("⚙️ Display Settings")
        st.sidebar.checkbox("Show Data Quality Info", key="show_data_quality")
        st.sidebar.checkbox("Show Outlier Summary", key="show_outliers")

    def render_main_panel(self):
        """Renders the main content area of the dashboard."""
        st.markdown("<div class='main-header'>📊 Financial Analysis Dashboard</div>", unsafe_allow_html=True)

        if not st.session_state.analysis_data:
            st.info("👋 Welcome! Please upload a financial data file using the sidebar to begin analysis.")
            return

        data = st.session_state.analysis_data
        df = data["statement"]
        
        st.subheader(f"Company Analysis: {data['company_name']}")

        # --- Display Data Quality and Outlier Information ---
        self._display_metadata(data)

        # --- Create tabs for visualizations and data table ---
        tab_viz, tab_data = st.tabs(["📊 Visualizations", "📄 Data Table"])

        with tab_viz:
            self._render_visualization_tab(df)
            
        with tab_data:
            self._render_data_table_tab(df)

    def _render_visualization_tab(self, df: pd.DataFrame):
        """Renders the content for the visualization tab."""
        st.header("Financial Charts")
        
        available_metrics = df.index.tolist()
        
        col1, col2, col3 = st.columns([3, 2, 2])
        with col1:
            selected_metrics = st.multiselect(
                "Select metrics to visualize:",
                options=available_metrics,
                default=available_metrics[:2] if available_metrics else []
            )
        with col2:
            chart_type = st.selectbox("Select Chart Type:", self.CHART_BUILDERS.keys())
        with col3:
            theme = st.selectbox("Chart Theme:", ["plotly_white", "plotly_dark", "ggplot2"])

        show_grid = st.checkbox("Show Chart Gridlines", value=True)

        if not selected_metrics:
            st.warning("Please select at least one metric to generate a chart.")
            return

        chart_builder = self.CHART_BUILDERS[chart_type]
        fig = chart_builder(
            df=df,
            metrics=selected_metrics,
            title=f"{chart_type} of Selected Financials",
            theme=theme,
            show_grid=show_grid
        )
        
        if fig:
            st.plotly_chart(fig, use_container_width=True)
            
    def _render_data_table_tab(self, df: pd.DataFrame):
        """Renders the content for the data table tab."""
        st.header("Financial Data")
        st.info("This table shows the cleaned financial data used for the visualizations.")
        
        # Format the dataframe for better readability
        formatted_df = df.style.format("{:,.2f}", na_rep="-")
        
        st.dataframe(formatted_df, use_container_width=True)

    def _display_metadata(self, data: Dict[str, Any]):
        """Displays the data quality and outlier cards if toggled."""
        if st.session_state.show_data_quality:
            dq = data["data_quality"]
            quality_class = f"quality-{dq.quality_score.lower()}"
            st.markdown(f"""
                <div class="feature-card">
                    <h4><span class="data-quality-indicator {quality_class}"></span>Data Quality: {dq.quality_score}</h4>
                    <ul>
                        <li><b>Total Rows:</b> {dq.total_rows}</li>
                        <li><b>Missing Values:</b> {dq.missing_values} ({dq.missing_percentage:.2f}%)</li>
                        <li><b>Duplicate Rows:</b> {dq.duplicate_rows}</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

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
