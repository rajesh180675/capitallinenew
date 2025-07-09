"""
Enhanced Financial Dashboard - Improved Version
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
from dataclasses import dataclass
from functools import lru_cache
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(level=logging.INFO)
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
    quality_score: str
    
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
        """Validates the uploaded file."""
        if uploaded_file is None:
            return False, "No file uploaded"
        
        # Check file size (max 10MB)
        if uploaded_file.size > 10 * 1024 * 1024:
            return False, "File size exceeds 10MB limit"
        
        # Check file type
        allowed_types = ['xls', 'xlsx', 'html', 'htm']
        file_extension = uploaded_file.name.split('.')[-1].lower()
        if file_extension not in allowed_types:
            return False, f"Unsupported file type: {file_extension}"
        
        return True, "File validation passed"

class DataProcessor:
    """Handles data processing and cleaning operations."""
    
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        """Cleans and converts numeric data."""
        numeric_df = df.copy()
        
        # Convert to numeric, handling various formats
        for col in numeric_df.columns:
            if col != 'Metric':
                # Remove commas, parentheses, and other formatting
                numeric_df[col] = numeric_df[col].astype(str).str.replace(',', '')
                numeric_df[col] = numeric_df[col].str.replace('(', '-').str.replace(')', '')
                numeric_df[col] = numeric_df[col].str.replace('₹', '').str.replace('Rs.', '')
                numeric_df[col] = pd.to_numeric(numeric_df[col], errors='coerce')
        
        return numeric_df
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame) -> Dict[str, List[str]]:
        """Detects outliers in the data using IQR method."""
        outliers = {}
        
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_indices = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
                if outlier_indices:
                    outliers[col] = outlier_indices
        
        return outliers
    
    @staticmethod
    def calculate_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
        """Calculates data quality metrics."""
        total_rows = len(df)
        missing_values = df.isnull().sum().sum()
        total_cells = df.size
        missing_percentage = (missing_values / total_cells) * 100 if total_cells > 0 else 0
        duplicate_rows = df.duplicated().sum()
        
        return DataQualityMetrics(
            total_rows=total_rows,
            missing_values=missing_values,
            missing_percentage=missing_percentage,
            duplicate_rows=duplicate_rows,
            quality_score=""
        )

def parse_capitaline_file(uploaded_file) -> Optional[Dict[str, Any]]:
    """
    Enhanced parser for Capitaline files with better error handling and validation.
    """
    if uploaded_file is None:
        return None

    try:
        # Validate file first
        is_valid, validation_message = FileValidator.validate_file(uploaded_file)
        if not is_valid:
            st.error(f"File validation failed: {validation_message}")
            return None

        # Read file content
        file_content = uploaded_file.getvalue()
        
        # Determine file type and read accordingly
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension in ['html', 'htm']:
            df = pd.read_html(io.BytesIO(file_content), header=[0, 1])[0]
        else:
            df = pd.read_excel(io.BytesIO(file_content), header=[0, 1])
        
        # Extract company name with better error handling
        company_name = "Unknown Company"
        try:
            company_info_tuple = str(df.columns[0][0])
            if ">>" in company_info_tuple:
                company_name = company_info_tuple.split(">>")[2].split("(")[0].strip()
            else:
                # Alternative parsing methods
                company_match = re.search(r'([A-Z][A-Za-z\s&]+)', company_info_tuple)
                if company_match:
                    company_name = company_match.group(1).strip()
        except (IndexError, AttributeError) as e:
            logger.warning(f"Could not extract company name: {e}")
            company_name = f"Company_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Flatten column names
        new_cols = []
        for col in df.columns:
            if isinstance(col, tuple):
                # Join non-null parts of the tuple
                col_name = '_'.join([str(part) for part in col if pd.notna(part) and str(part) != 'nan'])
            else:
                col_name = str(col)
            new_cols.append(col_name)
        
        df.columns = new_cols
        
        # Find and rename metric column
        metric_col_name = df.columns[0]
        df = df.rename(columns={metric_col_name: "Metric"})
        
        # Clean and filter data
        df = df.dropna(subset=['Metric'])
        df = df[df['Metric'].str.strip() != '']
        df = df.set_index('Metric')

        # Enhanced year column detection and cleaning
        year_pattern = re.compile(r'20\d{2}')
        renamed_cols = {}
        
        for col in df.columns:
            col_str = str(col).strip()
            
            # Extract year from various formats
            year_match = year_pattern.search(col_str)
            if year_match:
                year = year_match.group()
                renamed_cols[col] = year
            elif col_str.isdigit() and len(col_str) == 4 and col_str.startswith('20'):
                renamed_cols[col] = col_str
            elif col_str.isdigit() and len(col_str) == 6:
                # Handle YYYYMM format
                year = col_str[:4]
                if year.startswith('20'):
                    renamed_cols[col] = year
        
        df = df.rename(columns=renamed_cols)
        
        # Filter and sort year columns
        year_columns = [col for col in df.columns 
                       if str(col).isdigit() and len(str(col)) == 4 and str(col).startswith('20')]
        year_columns = sorted(year_columns, reverse=True)
        
        if not year_columns:
            st.error("Could not find valid year columns in the data. Please check the file format.")
            return None
        
        # Select only year columns and clean data
        df_final = df[year_columns].copy()
        df_final = DataProcessor.clean_numeric_data(df_final)
        
        # Remove rows with all NaN values
        df_final = df_final.dropna(how='all')
        
        # Calculate data quality metrics
        data_quality = DataProcessor.calculate_data_quality(df_final)
        
        # Detect outliers
        outliers = DataProcessor.detect_outliers(df_final)
        
        return {
            "statement": df_final,
            "company_name": company_name,
            "data_quality": data_quality,
            "outliers": outliers,
            "year_columns": year_columns,
            "file_info": {
                "name": uploaded_file.name,
                "size": uploaded_file.size,
                "type": file_extension
            }
        }

    except Exception as e:
        logger.error(f"Error parsing file: {e}")
        st.error(f"An error occurred while parsing the file: {str(e)}")
        return None

class ChartGenerator:
    """Enhanced chart generation with multiple chart types and customization options."""
    
    @staticmethod
    def create_line_chart(df: pd.DataFrame, selected_metrics: List[str], title: str) -> go.Figure:
        """Creates an enhanced line chart."""
        plot_df = df.loc[selected_metrics].dropna(axis=1, how='all').T
        plot_df.index = plot_df.index.astype(str)
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        for i, metric in enumerate(selected_metrics):
            if metric in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df.index,
                    y=plot_df[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(color=colors[i % len(colors)], width=3),
                    marker=dict(size=8),
                    hovertemplate=f'<b>{metric}</b><br>Year: %{{x}}<br>Value: ₹%{{y:,.2f}} Cr.<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#2c3e50')),
            xaxis_title="Year",
            yaxis_title="Amount (₹ Cr.)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_bar_chart(df: pd.DataFrame, selected_metrics: List[str], title: str) -> go.Figure:
        """Creates an enhanced bar chart."""
        plot_df = df.loc[selected_metrics].dropna(axis=1, how='all').T
        plot_df.index = plot_df.index.astype(str)
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        for i, metric in enumerate(selected_metrics):
            if metric in plot_df.columns:
                fig.add_trace(go.Bar(
                    x=plot_df.index,
                    y=plot_df[metric],
                    name=metric,
                    marker_color=colors[i % len(colors)],
                    hovertemplate=f'<b>{metric}</b><br>Year: %{{x}}<br>Value: ₹%{{y:,.2f}} Cr.<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#2c3e50')),
            xaxis_title="Year",
            yaxis_title="Amount (₹ Cr.)",
            barmode='group',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_area_chart(df: pd.DataFrame, selected_metrics: List[str], title: str) -> go.Figure:
        """Creates a stacked area chart."""
        plot_df = df.loc[selected_metrics].dropna(axis=1, how='all').T
        plot_df.index = plot_df.index.astype(str)
        
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set1
        for i, metric in enumerate(selected_metrics):
            if metric in plot_df.columns:
                fig.add_trace(go.Scatter(
                    x=plot_df.index,
                    y=plot_df[metric],
                    mode='lines',
                    name=metric,
                    fill='tonexty' if i > 0 else 'tozeroy',
                    line=dict(color=colors[i % len(colors)]),
                    hovertemplate=f'<b>{metric}</b><br>Year: %{{x}}<br>Value: ₹%{{y:,.2f}} Cr.<extra></extra>'
                ))
        
        fig.update_layout(
            title=dict(text=title, font=dict(size=20, color='#2c3e50')),
            xaxis_title="Year",
            yaxis_title="Amount (₹ Cr.)",
            hovermode='x unified',
            template='plotly_white',
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_heatmap(df: pd.DataFrame, selected_metrics: List[str], title: str) -> go.Figure:
        """Creates a correlation heatmap."""
        plot_df = df.loc[selected_metrics].dropna(axis=1, how='all').T
        correlation_matrix = plot_df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
        ))
        
        fig.update_layout(
            title=dict(text=f"Correlation Matrix - {title}", font=dict(size=20, color='#2c3e50')),
            template='plotly_white',
            height=500
        )
        
        return fig

class DashboardUI:
    """Enhanced UI class with improved functionality and user experience."""

    def __init__(self):
        """Initialize the UI class and set up session state."""
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
                st.session_state[var] = default_value

    def render_header(self):
        """Render enhanced header with company information."""
        st.markdown('<div class="main-header">📊 Advanced Financial Dashboard</div>', unsafe_allow_html=True)
        
        if st.session_state.analysis_data:
            company_name = st.session_state.analysis_data.get("company_name", "Unknown Company")
            file_info = st.session_state.analysis_data.get("file_info", {})
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"**Company:** {company_name}")
            with col2:
                st.markdown(f"**File:** {file_info.get('name', 'Unknown')}")
            with col3:
                st.markdown(f"**Size:** {file_info.get('size', 0) / 1024:.1f} KB")
        
        st.markdown("---")

    def render_sidebar(self):
        """Render enhanced sidebar with additional controls."""
        with st.sidebar:
            st.header("🎛️ Controls")
            
            # File upload section
            st.subheader("📁 File Upload")
            st.info("Upload a Capitaline .xls/.xlsx/.html file to begin analysis.")
            uploaded_file = st.file_uploader(
                "Choose a file",
                type=['xls', 'xlsx', 'html', 'htm'],
                help="Supported formats: XLS, XLSX, HTML"
            )
            
            # Display options
            st.subheader("📊 Display Options")
            show_data_quality = st.checkbox("Show Data Quality Metrics", value=False)
            show_outliers = st.checkbox("Show Outlier Detection", value=False)
            
            # Chart customization
            if st.session_state.analysis_data:
                st.subheader("🎨 Chart Settings")
                chart_theme = st.selectbox(
                    "Chart Theme",
                    ["plotly_white", "plotly_dark", "ggplot2", "seaborn"]
                )
                
                show_grid = st.checkbox("Show Grid Lines", value=True)
                
                return {
                    "file": uploaded_file,
                    "show_data_quality": show_data_quality,
                    "show_outliers": show_outliers,
                    "chart_theme": chart_theme,
                    "show_grid": show_grid
                }
            
            return {
                "file": uploaded_file,
                "show_data_quality": show_data_quality,
                "show_outliers": show_outliers
            }

    def display_data_quality_metrics(self, data_quality: DataQualityMetrics):
        """Display data quality metrics in an attractive format."""
        st.subheader("🔍 Data Quality Assessment")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Rows", data_quality.total_rows)
        
        with col2:
            st.metric("Missing Values", data_quality.missing_values)
        
        with col3:
            st.metric("Missing %", f"{data_quality.missing_percentage:.1f}%")
        
        with col4:
            quality_color = {
                "High": "quality-high",
                "Medium": "quality-medium",
                "Low": "quality-low"
            }
            st.markdown(
                f'<div class="data-quality-indicator {quality_color[data_quality.quality_score]}"></div>'
                f'<strong>{data_quality.quality_score} Quality</strong>',
                unsafe_allow_html=True
            )

    def display_outliers(self, outliers: Dict[str, List[str]]):
        """Display outlier detection results."""
        if outliers:
            st.subheader("⚠️ Outlier Detection")
            
            for year, outlier_metrics in outliers.items():
                with st.expander(f"Outliers in {year}"):
                    for metric in outlier_metrics:
                        st.write(f"• {metric}")
        else:
            st.success("No outliers detected in the data.")

    def generate_chart(self, df: pd.DataFrame, selected_metrics: List[str], 
                      chart_type: str, chart_theme: str = "plotly_white") -> Optional[go.Figure]:
        """Generate enhanced charts with multiple options."""
        if not selected_metrics:
            return None

        title = f"Financial Analysis: {', '.join(selected_metrics)}"
        
        chart_generators = {
            'Line Chart': ChartGenerator.create_line_chart,
            'Bar Chart': ChartGenerator.create_bar_chart,
            'Area Chart': ChartGenerator.create_area_chart,
            'Heatmap': ChartGenerator.create_heatmap
        }
        
        if chart_type in chart_generators:
            fig = chart_generators[chart_type](df, selected_metrics, title)
            fig.update_layout(template=chart_theme)
            return fig
        
        return None

    def display_summary_statistics(self, df: pd.DataFrame, selected_metrics: List[str]):
        """Display summary statistics for selected metrics."""
        if not selected_metrics:
            return
        
        st.subheader("📈 Summary Statistics")
        
        summary_df = df.loc[selected_metrics].describe().round(2)
        st.dataframe(summary_df, use_container_width=True)
        
        # Calculate year-over-year growth
        if len(df.columns) >= 2:
            st.subheader("📊 Year-over-Year Growth")
            growth_data = []
            
            for metric in selected_metrics:
                metric_data = df.loc[metric].dropna()
                if len(metric_data) >= 2:
                    years = sorted(metric_data.index)
                    for i in range(1, len(years)):
                        current_year = years[i]
                        previous_year = years[i-1]
                        
                        current_value = metric_data[current_year]
                        previous_value = metric_data[previous_year]
                        
                        if previous_value != 0:
                            growth_rate = ((current_value - previous_value) / previous_value) * 100
                            growth_data.append({
                                'Metric': metric,
                                'Year': current_year,
                                'Growth Rate (%)': round(growth_rate, 2)
                            })
            
            if growth_data:
                growth_df = pd.DataFrame(growth_data)
                st.dataframe(growth_df, use_container_width=True)

    def display_welcome_message(self):
        """Display welcome message when no data is loaded."""
        st.markdown("""
        <div class="welcome-container">
            <h2>🎯 Welcome to the Advanced Financial Dashboard</h2>
            <p style="font-size: 1.2em; color: #555; margin: 1rem 0;">
                Upload your Capitaline financial data files to get started with comprehensive analysis
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Feature highlights
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="feature-card">
                <h3>📊 Multi-Chart Visualization</h3>
                <p>Create line charts, bar charts, area charts, and correlation heatmaps for comprehensive data analysis.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="feature-card">
                <h3>🔍 Data Quality Assessment</h3>
                <p>Automatic data quality scoring, outlier detection, and comprehensive data validation.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="feature-card">
                <h3>📈 Advanced Analytics</h3>
                <p>Summary statistics, year-over-year growth analysis, and interactive data exploration.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Instructions
        st.markdown("---")
        st.subheader("📋 Getting Started")
        st.markdown("""
        1. **Upload Your File**: Use the file uploader in the sidebar to upload your Capitaline .xls, .xlsx, or .html file
        2. **Select Metrics**: Choose the financial metrics you want to analyze from the dropdown menu
        3. **Choose Chart Type**: Select from Line Chart, Bar Chart, Area Chart, or Heatmap
        4. **Generate Visualization**: Click the "Generate Chart" button to create your visualization
        5. **Explore Data**: Use the search functionality and export options to further analyze your data
        """)
        
        st.info("💡 **Tip**: Enable 'Show Data Quality Metrics' and 'Show Outlier Detection' in the sidebar for advanced data insights!")

    def display_capitaline_data(self, analysis_data: Dict[str, Any], controls: Dict[str, Any]):
        """Enhanced display of Capitaline data with additional features."""
        company_name = analysis_data.get("company_name", "Uploaded Data")
        statement_df = analysis_data.get("statement")
        data_quality = analysis_data.get("data_quality")
        outliers = analysis_data.get("outliers", {})
        
        st.header(f"📊 Analysis for: {company_name}")
        
        # Display data quality metrics if enabled
        if controls.get("show_data_quality") and data_quality:
            self.display_data_quality_metrics(data_quality)
            st.markdown("---")
        
        # Display outliers if enabled
        if controls.get("show_outliers") and outliers:
            self.display_outliers(outliers)
            st.markdown("---")
        
        if statement_df is not None and not statement_df.empty:
            st.info("Select metrics from the dropdown below, then choose a chart type and click 'Generate Chart'.")

            # Enhanced charting controls
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                selected_rows = st.multiselect(
                    "Select metrics to chart:",
                    options=statement_df.index.tolist(),
                    default=st.session_state.selected_metrics,
                    key="metric_selector",
                    help="Choose one or more financial metrics to visualize"
                )
                st.session_state.selected_metrics = selected_rows
            
            with col2:
                chart_type = st.selectbox(
                    "Chart Type",
                    ["Line Chart", "Bar Chart", "Area Chart", "Heatmap"],
                    index=0,
                    key="chart_type_selector"
                )
            
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
                generate = st.button(
                    "📊 Generate Chart",
                    type="primary",
                    use_container_width=True,
                    disabled=(not selected_rows)
                )

            # Generate chart
            if generate and selected_rows:
                with st.spinner("Generating chart..."):
                    fig = self.generate_chart(
                        statement_df,
                        selected_rows,
                        chart_type,
                        controls.get("chart_theme", "plotly_white")
                    )
                    st.session_state.chart_figure = fig
                    st.success("Chart generated successfully!")

            # Display chart
            if st.session_state.chart_figure:
                st.markdown("---")
                st.subheader("📈 Visualization")
                st.plotly_chart(st.session_state.chart_figure, use_container_width=True)
                
                # Display summary statistics
                if selected_rows:
                    self.display_summary_statistics(statement_df, selected_rows)

                        # Enhanced data table display
            st.markdown("---")
            st.subheader("📋 Data Table")
            
            # Add search functionality
            search_term = st.text_input("🔍 Search metrics:", placeholder="Type to filter metrics...")
            
            # Filter dataframe based on search
            if search_term:
                filtered_df = statement_df[statement_df.index.str.contains(search_term, case=False)]
            else:
                filtered_df = statement_df
            
            # Display options for the data table
            col1, col2, col3 = st.columns(3)
            with col1:
                show_all_data = st.checkbox("Show all data", value=False)
            with col2:
                highlight_negatives = st.checkbox("Highlight negative values", value=True)
            with col3:
                decimal_places = st.selectbox("Decimal places", [0, 1, 2], index=2)
            
            # Format and display the dataframe
            if not filtered_df.empty:
                # Apply formatting
                styled_df = filtered_df.round(decimal_places)
                
                if highlight_negatives:
                    styled_df = styled_df.style.applymap(
                        lambda x: 'color: red' if pd.notna(x) and x < 0 else '',
                        subset=pd.IndexSlice[:, :]
                    )
                
                # Display limited rows or all data
                if show_all_data:
                    st.dataframe(styled_df, use_container_width=True, height=600)
                else:
                    st.dataframe(styled_df.head(20), use_container_width=True)
                    if len(filtered_df) > 20:
                        st.info(f"Showing first 20 rows of {len(filtered_df)} total. Check 'Show all data' to see everything.")
                
                # Export functionality
                st.markdown("---")
                st.subheader("📥 Export Options")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Export to CSV
                    csv = filtered_df.to_csv()
                    st.download_button(
                        label="📄 Download as CSV",
                        data=csv,
                        file_name=f"{company_name}_financial_data_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv"
                    )
                
                with col2:
                    # Export to Excel
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
                        filtered_df.to_excel(writer, sheet_name='Financial Data')
                        
                        # Add chart to Excel if available
                        if st.session_state.chart_figure and selected_rows:
                            # Create a summary sheet
                            summary_df = filtered_df.loc[selected_rows]
                            summary_df.to_excel(writer, sheet_name='Selected Metrics')
                    
                    excel_data = output.getvalue()
                    st.download_button(
                        label="📊 Download as Excel",
                        data=excel_data,
                        file_name=f"{company_name}_financial_analysis_{datetime.now().strftime('%Y%m%d')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            else:
                st.warning("No data matches your search criteria.")
        else:
            st.error("No data available to display.")

def main():
    """Main application entry point."""
    # Initialize UI
    ui = DashboardUI()
    
    # Render header
    ui.render_header()
    
    # Render sidebar and get controls
    controls = ui.render_sidebar()
    
    # Process uploaded file if available
    if controls["file"] is not None:
        # Check if file has changed
        file_id = f"{controls['file'].name}_{controls['file'].size}"
        
        if st.session_state._uploaded_file_memo != file_id:
            st.session_state._uploaded_file_memo = file_id
            st.session_state.chart_figure = None
            st.session_state.selected_metrics = []
            
            with st.spinner("Processing file..."):
                analysis_data = parse_capitaline_file(controls["file"])
                st.session_state.analysis_data = analysis_data
            
            if analysis_data:
                st.success("✅ File processed successfully!")
    
    # Display content based on whether data is loaded
    if st.session_state.analysis_data:
        ui.display_capitaline_data(st.session_state.analysis_data, controls)
    else:
        ui.display_welcome_message()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #666; padding: 1rem;'>
            <p>Advanced Financial Dashboard v2.0 | Built with Streamlit & Plotly</p>
            <p style='font-size: 0.9em;'>© 2024 | For educational and analytical purposes</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
