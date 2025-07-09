# Enhanced Financial Dashboard - Complete Integrated Version
# A robust Streamlit application for financial data analysis with enhanced error handling,
# performance optimization, and advanced analytical features including multi-file support.

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
from sklearn.linear_model import LinearRegression
from streamlit.runtime.uploaded_file_manager import UploadedFile

# --- 2. Configuration ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_TYPES = ['html', 'htm', 'xls', 'xlsx']
YEAR_REGEX = re.compile(r'\b(19[8-9]\d|20\d\d)\b')

# Standard metric names required for advanced analysis
REQUIRED_METRICS = {
    'Profitability': ['Revenue', 'Gross Profit', 'Operating Profit', 'EBIT', 'Net Profit', 'Shareholders Equity', 'Total Equity', 'Total Assets', 'Current Liabilities'],
    'Liquidity': ['Current Assets', 'Current Liabilities', 'Inventory', 'Cash and Cash Equivalents'],
    'Leverage': ['Total Debt', 'Shareholders Equity', 'Total Equity', 'Total Assets', 'EBIT', 'Interest Expense', 'Operating Cash Flow', 'Total Debt Service'],
    'Efficiency': ['Revenue', 'Total Assets', 'Cost of Goods Sold', 'Inventory', 'Accounts Receivable', 'Current Assets', 'Current Liabilities'],
    'DuPont': ['Net Profit', 'Revenue', 'Total Assets', 'Shareholders Equity', 'EBIT', 'Interest Expense', 'Tax Expense'],
    'Cash Flow': ['Operating Cash Flow', 'Capital Expenditure', 'Revenue', 'Net Profit'],
    'Growth & Trends': ['Revenue', 'Net Profit', 'EBIT']
}

# --- 3. Page and Style Configuration ---
st.set_page_config(
    page_title="Advanced Financial Dashboard",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .feature-card { background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 1rem 0; border-left: 4px solid #1f77b4; }
    .stButton>button { background: linear-gradient(90deg, #1f77b4, #17a2b8); color: white; border-radius: 8px; border: none; padding: 10px 20px; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .data-quality-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
    .quality-high { background-color: #28a745; }
    .quality-medium { background-color: #ffc107; }
    .quality-low { background-color: #dc3545; }
    .st-expander { border: 1px solid #ddd; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 4. Data Structures and Basic Classes (from original script) ---
@dataclass
class DataQualityMetrics:
    total_rows: int; missing_values: int; missing_percentage: float; duplicate_rows: int
    quality_score: str = field(init=False)
    def __post_init__(self):
        if self.missing_percentage < 5: self.quality_score = "High"
        elif self.missing_percentage < 20: self.quality_score = "Medium"
        else: self.quality_score = "Low"

class FileValidator:
    @staticmethod
    def validate_file(uploaded_file: UploadedFile) -> Tuple[bool, str]:
        if uploaded_file is None: return False, "No file uploaded."
        if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024: return False, f"File size exceeds {MAX_FILE_SIZE_MB}MB."
        ext = uploaded_file.name.split('.')[-1].lower()
        if ext not in ALLOWED_FILE_TYPES: return False, f"Unsupported file type: '.{ext}'."
        return True, ext

class DataProcessor:
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace(r'[,\(\)₹]|Rs\.', '', regex=True)
            df[col] = pd.to_numeric(df[col], errors='coerce')
        return df

    @staticmethod
    def calculate_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
        total = df.size
        if total == 0: return DataQualityMetrics(0, 0, 0.0, 0)
        missing = df.isnull().sum().sum()
        return DataQualityMetrics(len(df), int(missing), (missing/total)*100, int(df.duplicated().sum()))

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

class ChartGenerator: # (No changes needed in this class)
    @staticmethod
    def _create_base_figure(title, theme, show_grid, yaxis_title):
        fig = go.Figure()
        fig.update_layout(title={'text': title, 'x': 0.5}, xaxis_title="Year", yaxis_title=yaxis_title, template=theme, height=500, hovermode='x unified', xaxis={'showgrid': show_grid}, yaxis={'showgrid': show_grid}, legend_title_text='Metrics')
        return fig

    @staticmethod
    def create_line_chart(df, metrics, title, theme, show_grid, scale_type, yaxis_title):
        fig = ChartGenerator._create_base_figure(title, theme, show_grid, yaxis_title)
        for i, metric in enumerate(metrics):
            fig.add_trace(go.Scatter(x=df.columns, y=df.loc[metric], mode='lines+markers', name=metric))
        if scale_type == 'Logarithmic': fig.update_layout(yaxis_type='log')
        return fig

    @staticmethod
    def create_bar_chart(df, metrics, title, theme, show_grid, scale_type, yaxis_title):
        fig = ChartGenerator._create_base_figure(title, theme, show_grid, yaxis_title)
        for metric in metrics: fig.add_trace(go.Bar(x=df.columns, y=df.loc[metric], name=metric))
        if scale_type == 'Logarithmic': fig.update_layout(yaxis_type='log')
        return fig

    @staticmethod
    def create_heatmap(df, metrics, title, theme, **kwargs):
        if len(metrics) < 2:
            st.warning("Heatmap requires at least two metrics.")
            return None
        corr = df.loc[metrics].T.corr()
        return go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='RdBu_r', zmid=0)).update_layout(title={'text': title, 'x': 0.5}, template=theme)

# --- 5. Advanced Financial Analysis Modules ---
# (No changes needed in these classes)
class FinancialRatioCalculator:
    @staticmethod
    def calculate_all_ratios(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        ratios = {}
        def get(name): return df.loc[name] if name in df.index else None
        
        profit_ratios = pd.DataFrame(index=df.columns)
        if (rev := get('Revenue')) is not None:
            if (gp := get('Gross Profit')) is not None: profit_ratios['Gross Margin %'] = (gp / rev) * 100
            if (op := get('EBIT')) is not None: profit_ratios['Operating Margin %'] = (op / rev) * 100
            if (np := get('Net Profit')) is not None: profit_ratios['Net Margin %'] = (np / rev) * 100
        if (np := get('Net Profit')) is not None:
            if (eq := get('Total Equity')) is not None: profit_ratios['ROE %'] = (np / eq) * 100
            if (ta := get('Total Assets')) is not None: profit_ratios['ROA %'] = (np / ta) * 100
        if (ebit := get('EBIT')) is not None and (ta := get('Total Assets')) is not None and (cl := get('Current Liabilities')) is not None:
            profit_ratios['ROCE %'] = (ebit / (ta - cl)) * 100
        if not profit_ratios.empty: ratios['Profitability'] = profit_ratios.T.dropna(how='all')

        liq_ratios = pd.DataFrame(index=df.columns)
        if (ca := get('Current Assets')) is not None and (cl := get('Current Liabilities')) is not None:
            liq_ratios['Current Ratio'] = ca / cl
            if (inv := get('Inventory')) is not None: liq_ratios['Quick Ratio'] = (ca - inv) / cl
        if (cash := get('Cash and Cash Equivalents')) is not None and (cl := get('Current Liabilities')) is not None:
            liq_ratios['Cash Ratio'] = cash / cl
        if not liq_ratios.empty: ratios['Liquidity'] = liq_ratios.T.dropna(how='all')

        lev_ratios = pd.DataFrame(index=df.columns)
        if (debt := get('Total Debt')) is not None:
            if (eq := get('Total Equity')) is not None: lev_ratios['Debt to Equity'] = debt / eq
            if (ta := get('Total Assets')) is not None: lev_ratios['Debt to Assets'] = debt / ta
        if (ebit := get('EBIT')) is not None and (ie := get('Interest Expense')) is not None:
            lev_ratios['Interest Coverage'] = ebit / ie
        if not lev_ratios.empty: ratios['Leverage'] = lev_ratios.T.dropna(how='all')

        return ratios

class GrowthAnalyzer:
    @staticmethod
    def calculate_cagr(df: pd.DataFrame, metrics: List[str]) -> pd.DataFrame:
        results = {}
        for metric in metrics:
            if metric in df.index and not df.loc[metric].dropna().empty:
                series = df.loc[metric].dropna()
                if len(series) >= 2 and series.iloc[0] > 0:
                    years = len(series) - 1
                    cagr = ((series.iloc[-1] / series.iloc[0]) ** (1/years) - 1) * 100
                    results[metric] = {'CAGR %': cagr, 'Start Value': series.iloc[0], 'End Value': series.iloc[-1], 'Years': years}
        return pd.DataFrame(results).T

class TrendAnalyzer:
    @staticmethod
    def forecast_metrics(df: pd.DataFrame, metrics: List[str], periods: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        forecasts, stats = pd.DataFrame(), pd.DataFrame()
        X = df.columns.astype(int).values.reshape(-1, 1)
        for metric in metrics:
            if metric in df.index:
                y = df.loc[metric].values
                mask = ~np.isnan(y)
                if mask.sum() >= 3:
                    model = LinearRegression().fit(X[mask], y[mask])
                    r2 = model.score(X[mask], y[mask])
                    future_X = np.arange(X.max() + 1, X.max() + 1 + periods).reshape(-1, 1)
                    preds = model.predict(future_X)
                    for i, year in enumerate(future_X.flatten()):
                        forecasts.loc[metric, str(year)] = preds[i]
                    stats.loc[metric, ['R-squared', 'Trend']] = [r2, 'Increasing' if model.coef_[0] > 0 else 'Decreasing']
        return forecasts, stats

class DuPontAnalyzer:
    @staticmethod
    def calculate_dupont(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        dupont = pd.DataFrame(index=df.columns)
        req = ['Net Profit', 'Revenue', 'Total Assets', 'Total Equity']
        if not all(m in df.index for m in req): return None
        dupont['Net Margin %'] = (df.loc['Net Profit'] / df.loc['Revenue']) * 100
        dupont['Asset Turnover'] = df.loc['Revenue'] / df.loc['Total Assets']
        dupont['Equity Multiplier'] = df.loc['Total Assets'] / df.loc['Total Equity']
        dupont['ROE (DuPont) %'] = (dupont['Net Margin %'] * dupont['Asset Turnover'] * dupont['Equity Multiplier']) / 100
        return dupont.T

class CashFlowAnalyzer:
    @staticmethod
    def calculate_fcf_metrics(df: pd.DataFrame) -> Optional[pd.DataFrame]:
        metrics = pd.DataFrame(index=df.columns)
        req = ['Operating Cash Flow', 'Capital Expenditure']
        if not all(m in df.index for m in req): return None
        fcf = df.loc['Operating Cash Flow'] - abs(df.loc['Capital Expenditure'])
        metrics['Free Cash Flow'] = fcf
        if 'Revenue' in df.index: metrics['FCF to Revenue %'] = (fcf / df.loc['Revenue']) * 100
        if 'Net Profit' in df.index: metrics['FCF to Net Income'] = fcf / df.loc['Net Profit']
        return metrics.T

# --- 6. Core Application Logic ---

# This helper function now just parses a single file and is NOT cached.
def parse_single_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    """Parses a single financial data file and returns a dictionary with the dataframe and metadata."""
    if not FileValidator.validate_file(uploaded_file)[0]: return None
    try:
        df = pd.read_html(io.BytesIO(uploaded_file.getvalue()), header=[0, 1])[0]
        company_name = "Unknown Company"
        try:
            if ">>" in str(df.columns[0][0]): company_name = str(df.columns[0][0]).split(">>")[2].split("(")[0].strip()
        except IndexError: pass
        
        df.columns = [str(c[1]) for c in df.columns]
        df = df.rename(columns={df.columns[0]: "Metric"}).dropna(subset=["Metric"]).reset_index(drop=True)
        
        # Improved unique metric ID generation
        is_duplicate = df.duplicated(subset=['Metric'], keep=False)
        df['unique_metric_id'] = df['Metric']
        df.loc[is_duplicate, 'unique_metric_id'] = df['Metric'] + ' (row ' + (df.index + 2).astype(str) + ')'
        df = df.set_index('unique_metric_id').drop(columns=['Metric'])

        year_cols = {c: YEAR_REGEX.search(c).group(0) for c in df.columns if YEAR_REGEX.search(c)}
        df = df.rename(columns=year_cols)
        valid_years = sorted([c for c in df.columns if c.isdigit()], key=int)
        
        if not valid_years: return None
        
        df_proc = DataProcessor.clean_numeric_data(df[valid_years].copy()).dropna(how='all')
        return {"statement": df_proc, "company_name": company_name}
    except Exception as e:
        logger.error(f"Error parsing file '{uploaded_file.name}': {e}")
        st.warning(f"Could not parse '{uploaded_file.name}'. It might be in an unsupported format.")
        return None

class DashboardApp:
    def __init__(self):
        self._initialize_state()
        self.chart_builders = {"Line": ChartGenerator.create_line_chart, "Bar": ChartGenerator.create_bar_chart, "Heatmap": ChartGenerator.create_heatmap}

    def _initialize_state(self):
        defaults = {"analysis_data": None, "metric_mapping": {}}
        for k, v in defaults.items():
            if k not in st.session_state: st.session_state[k] = v

    @st.cache_data(show_spinner="Processing and merging files...")
    def process_and_merge_files(_self, uploaded_files: List[UploadedFile]) -> Optional[Dict[str, Any]]:
        """Processes a list of uploaded files, merges them, and returns a single analysis dictionary."""
        if not uploaded_files:
            return None

        all_dfs = []
        company_name = "Multiple Sources"
        first_company_name_found = False

        for file in uploaded_files:
            parsed_data = parse_single_file(file)
            if parsed_data:
                all_dfs.append(parsed_data["statement"])
                if not first_company_name_found and parsed_data["company_name"] != "Unknown Company":
                    company_name = parsed_data["company_name"]
                    first_company_name_found = True

        if not all_dfs:
            st.error("None of the uploaded files could be parsed. Please check the file formats.")
            return None

        # Merge dataframes, combining columns for the same year and stacking rows
        merged_df = pd.concat(all_dfs, axis=0)
        
        # Handle metrics that appear in multiple files by keeping the first one.
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        
        # Ensure year columns are sorted
        year_cols = sorted([col for col in merged_df.columns if col.isdigit()], key=int)
        merged_df = merged_df[year_cols]

        return {
            "statement": merged_df,
            "company_name": company_name,
            "data_quality": DataProcessor.calculate_data_quality(merged_df)
        }

    def _handle_file_upload(self):
        """Callback function to process files when the uploader changes."""
        uploaded_files = st.session_state.get("file_uploader_key", [])
        st.session_state.analysis_data = self.process_and_merge_files(uploaded_files)
        # Reset mapping whenever files are changed
        st.session_state.metric_mapping = {}

    def run(self):
        self.render_sidebar()
        self.render_main_panel()

    def render_sidebar(self):
        st.sidebar.title("📂 Upload & Options")
        st.sidebar.info("Upload one or more financial statements (e.g., Income, Balance Sheet). They will be automatically merged.")
        
        st.sidebar.file_uploader(
            "Upload Financials (HTML/XLSX)",
            type=ALLOWED_FILE_TYPES,
            accept_multiple_files=True,
            key="file_uploader_key",
            on_change=self._handle_file_upload
        )
        
        st.sidebar.title("⚙️ Display Settings")
        st.sidebar.checkbox("Show Data Quality", key="show_data_quality", value=True)
        
        if st.session_state.analysis_data:
            self._render_metric_mapper()

    def _render_metric_mapper(self):
        with st.sidebar.expander("📊 Metric Mapping for Analysis", expanded=False):
            st.info("Map metrics from your files to standard names for advanced analysis.")
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
                    index= (available_metrics.index(current_mapping[std_metric]) + 1) if std_metric in current_mapping and current_mapping[std_metric] in available_metrics else 0,
                    key=f"map_{std_metric}"
                )
    
    # --- Main Panel Rendering (no major changes, they adapt to the new data structure) ---

    def render_main_panel(self):
        st.markdown("<div class='main-header'>💹 Advanced Financial Dashboard</div>", unsafe_allow_html=True)
        if not st.session_state.analysis_data:
            st.info("👋 Welcome! Please upload one or more financial data files to begin.")
            return

        data, df = st.session_state.analysis_data, st.session_state.analysis_data["statement"]
        st.subheader(f"Company Analysis: {data['company_name']}")
        if st.session_state.show_data_quality:
            dq = data["data_quality"]
            qc = f"quality-{dq.quality_score.lower()}"
            st.markdown(f"""<div class="feature-card"><h4><span class="data-quality-indicator {qc}"></span>Merged Data Quality: {dq.quality_score}</h4>
            Total Unique Rows: {dq.total_rows} | Total Missing Values: {dq.missing_values} ({dq.missing_percentage:.2f}%)</div>""", unsafe_allow_html=True)

        tabs = ["📊 Visualizations", "📄 Merged Data Table", "💡 Advanced Analysis"]
        tab_viz, tab_data, tab_adv = st.tabs(tabs)
        with tab_viz: self._render_visualization_tab(df)
        with tab_data: self._render_data_table_tab(df)
        with tab_adv: self._render_advanced_analysis_tab(df)

    def _get_mapped_df(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        mapping = {v: k for k, v in st.session_state.metric_mapping.items() if v}
        if not mapping:
            st.warning("Please map metrics in the sidebar to run advanced analysis.")
            return None
        
        mapped_df = df.rename(index=mapping)
        return mapped_df

    def _render_advanced_analysis_tab(self, df: pd.DataFrame):
        st.header("💡 Advanced Financial Analysis")
        mapped_df = self._get_mapped_df(df)
        if mapped_df is None: return
        
        with st.expander("📈 Financial Ratio Analysis", expanded=True):
            ratios = FinancialRatioCalculator.calculate_all_ratios(mapped_df)
            if not ratios:
                st.info("Could not calculate ratios. Please ensure metrics for Profitability, Liquidity, and Leverage are mapped.")
            else:
                for ratio_type, ratio_df in ratios.items():
                    st.subheader(f"{ratio_type} Ratios")
                    st.dataframe(ratio_df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)
        
        with st.expander("🌱 Growth Analysis", expanded=False):
            st.subheader("Compound Annual Growth Rate (CAGR)")
            cagr_metrics = st.multiselect("Select metrics for CAGR:", mapped_df.index.tolist(), default=['Revenue', 'Net Profit'] if all(m in mapped_df.index for m in ['Revenue', 'Net Profit']) else None, key="cagr_ms")
            if cagr_metrics:
                cagr_results = GrowthAnalyzer.calculate_cagr(mapped_df, cagr_metrics)
                if not cagr_results.empty:
                    st.dataframe(cagr_results.style.format("{:,.2f}", na_rep="-"), use_container_width=True)
                else:
                    st.info("Could not calculate CAGR. Ensure selected metrics have at least two years of positive data.")
        
        with st.expander("🔮 Trend Analysis & Forecasting", expanded=False):
            st.subheader("Metric Forecasting")
            col1, col2 = st.columns([3,1])
            forecast_metrics = col1.multiselect("Select metrics to forecast:", mapped_df.index.tolist(), default=['Revenue'] if 'Revenue' in mapped_df.index else None, key="forecast_ms")
            periods = col2.number_input("Forecast Periods (Years)", min_value=1, max_value=5, value=2)
            if forecast_metrics:
                forecasts, stats = TrendAnalyzer.forecast_metrics(mapped_df, forecast_metrics, periods)
                if not forecasts.empty:
                    st.subheader("Forecasted Values")
                    st.dataframe(forecasts.style.format("{:,.2f}"), use_container_width=True)
                    st.subheader("Trend Model Statistics")
                    st.dataframe(stats.style.format({'R-squared': '{:.2%}'}), use_container_width=True)
        
        with st.expander("分解 DuPont Analysis (ROE Decomposition)", expanded=False):
            dupont_df = DuPontAnalyzer.calculate_dupont(mapped_df)
            if dupont_df is not None:
                st.subheader("DuPont Analysis Components")
                st.dataframe(dupont_df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)
            else:
                st.info("Could not perform DuPont analysis. Please map: Net Profit, Revenue, Total Assets, and Total Equity.")
        
        with st.expander("🌊 Cash Flow Analysis", expanded=False):
            fcf_df = CashFlowAnalyzer.calculate_fcf_metrics(mapped_df)
            if fcf_df is not None:
                st.subheader("Free Cash Flow Metrics")
                st.dataframe(fcf_df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)
            else:
                st.info("Could not perform Cash Flow analysis. Please map: Operating Cash Flow and Capital Expenditure.")

    def _render_visualization_tab(self, df: pd.DataFrame):
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
        metrics = col1.multiselect("Select metrics:", df.index.tolist(), default=df.index[:2].tolist())
        chart = col2.selectbox("Chart Type:", self.chart_builders.keys())
        theme = col3.selectbox("Theme:", ["plotly_white", "plotly_dark"])
        scale = col4.selectbox("Y-Axis Scale:", ["Linear", "Logarithmic", "Normalized (Base 100)"], disabled=(chart == 'Heatmap'))

        plot_df, y_title = df, "Amount (₹ Cr.)"
        if scale == "Normalized (Base 100)" and metrics:
            plot_df = DataProcessor.normalize_to_100(df, metrics)
            y_title = "Normalized Value (Base 100)"

        if metrics:
            fig = self.chart_builders[chart](df=plot_df, metrics=metrics, title=f"{chart} of Financials", theme=theme, show_grid=True, scale_type=scale, yaxis_title=y_title)
            if fig: st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Please select at least one metric.")

    def _render_data_table_tab(self, df: pd.DataFrame):
        st.subheader("Merged and Cleaned Financial Data")
        st.dataframe(df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)

# --- 7. App Execution ---
if __name__ == "__main__":
    try:
        app = DashboardApp()
        app.run()
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        st.error(f"A critical error occurred. Please refresh. Details: {e}")
