# Enhanced Financial Dashboard - Complete Integrated Version
# A robust Streamlit application with CSV/HTML/XLSX support, Penman-Nissim analysis,
# multi-file support, and integrated visualization of calculated metrics.

# --- 1. Imports and Setup ---
import io
import logging
import re
import warnings
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from fuzzywuzzy import fuzz  # For fuzzy matching in auto-population (pip install fuzzywuzzy python-levenshtein)
from sklearn.linear_model import LinearRegression
from streamlit.runtime.uploaded_file_manager import UploadedFile

# --- 2. Configuration ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_TYPES = ['html', 'htm', 'xls', 'xlsx', 'csv']
YEAR_REGEX = re.compile(r'\b(19[8-9]\d|20\d\d|FY\d{4})\b')  # Enhanced to handle 'FY2020' formats

REQUIRED_METRICS = {
    'Profitability': ['Revenue', 'Gross Profit', 'EBIT', 'Net Profit', 'Total Equity', 'Total Assets', 'Current Liabilities'],
    'Liquidity': ['Current Assets', 'Current Liabilities', 'Inventory', 'Cash and Cash Equivalents'],
    'Leverage': ['Total Debt', 'Total Equity', 'Total Assets', 'EBIT', 'Interest Expense'],
    'DuPont': ['Net Profit', 'Revenue', 'Total Assets', 'Total Equity'],
    'Cash Flow': ['Operating Cash Flow', 'Capital Expenditure']
}

# --- 3. Page and Style Configuration ---
st.set_page_config(page_title="Advanced Financial Dashboard", page_icon="💹", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; text-align: center; margin-bottom: 1rem; text-shadow: 2px 2px 4px rgba(0,0,0,0.1); }
    .feature-card { background: white; padding: 1.5rem; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); margin: 1rem 0; border-left: 4px solid #1f77b4; }
    .stButton>button { background: linear-gradient(90deg, #1f77b4, #17a2b8); color: white; border-radius: 8px; border: none; padding: 10px 20px; font-weight: 600; transition: all 0.3s ease; }
    .stButton>button:hover { transform: translateY(-2px); box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
    .data-quality-indicator { display: inline-block; width: 12px; height: 12px; border-radius: 50%; margin-right: 8px; }
    .quality-high { background-color: #28a745; } .quality-medium { background-color: #ffc107; } .quality-low { background-color: #dc3545; }
    .st-expander { border: 1px solid #ddd; border-radius: 10px; background-color: #f8f9fa; }
    .st-multiselect [data-testid="stMarkdownContainer"] { font-size: 0.9rem; }
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

class DataProcessor:
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str).str.replace(r'[,\(\)₹$€£]|Rs\.', '', regex=True)  # Enhanced currency removal
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
        fig.update_layout(title={'text': title, 'x': 0.5}, xaxis_title="Year", yaxis_title=yaxis_title, template=theme, height=500, hovermode='x unified', xaxis={'showgrid': show_grid}, yaxis={'showgrid': show_grid}, legend_title_text='Metrics')
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

# --- 5. Advanced Financial Analysis Modules ---
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
    def __init__(self, df: pd.DataFrame, mappings: Dict[str, Any]):
        self.df = df
        self.mappings = mappings

    def calculate_all(self) -> Dict[str, Any]:
        try:
            total_assets = self._get('Total Assets')
            total_liabilities = self._get('Total Liabilities')
            equity = self._get('Total Equity')
            financial_assets = self._get_multi('Financial Assets')
            financial_liabilities = self._get_multi('Financial Liabilities')
            operating_assets = total_assets - financial_assets
            operating_liabilities = total_liabilities - financial_liabilities
            noa = operating_assets - operating_liabilities
            nfo = financial_liabilities - financial_assets

            check_series = noa - nfo - equity
            valid_mask = np.isfinite(check_series) & np.isfinite(equity)
            validation_check = np.allclose(check_series[valid_mask], 0, atol=1e-2)  # Allow small tolerance for floating-point

            reformulated_bs = pd.DataFrame({
                'Operating Assets (OA)': operating_assets,
                'Financial Assets (FA)': financial_assets,
                'Operating Liabilities (OL)': operating_liabilities,
                'Financial Liabilities (FL)': financial_liabilities,
                'Net Operating Assets (NOA)': noa,
                'Net Financial Obligations (NFO)': nfo,
                'Total Equity': equity,
                'Check (NOA - NFO - Equity)': check_series
            }).T.dropna(how='all', axis=1)

            oi = self._get('Operating Income')
            sales = self._get('Revenue')
            nfe = self._get('Net Financial Expense')
            ni = self._get('Net Income')

            avg_noa = (noa + noa.shift(1)) / 2
            avg_nfo = (nfo + nfo.shift(1)) / 2
            avg_equity = (equity + equity.shift(1)) / 2

            rnoa = np.where(avg_noa != 0, (oi / avg_noa) * 100, np.nan)
            opm = np.where(sales != 0, (oi / sales) * 100, np.nan)
            noat = np.where(avg_noa != 0, sales / avg_noa, np.nan)
            ratios = pd.DataFrame({'Return on Net Operating Assets (RNOA) %': rnoa, 'Operating Profit Margin (OPM) %': opm, 'Net Operating Asset Turnover (NOAT)': noat}, index=self.df.columns).T.dropna(how='all', axis=1)

            nbc = np.where(avg_nfo != 0, (nfe / avg_nfo) * 100, np.nan)
            flev = np.where(avg_equity != 0, avg_nfo / avg_equity, np.nan)
            spread = rnoa - nbc
            roe_from_pn = rnoa + (flev * spread)
            roe_from_stmt = np.where(avg_equity != 0, (ni / avg_equity) * 100, np.nan)
            roe_decomposed = pd.DataFrame({
                'RNOA %': rnoa,
                'Financial Leverage (FLEV)': flev,
                'Spread (RNOA - NBC) %': spread,
                'Financing Contribution (FLEV * Spread)': flev * spread,
                'ROE (from P-N) %': roe_from_pn,
                'ROE (from statements) %': roe_from_stmt
            }, index=self.df.columns).T.dropna(how='all', axis=1)

            # NEW: Trend analysis (linear regression on RNOA for forecast)
            years = np.array([int(y) for y in ratios.columns if y.isdigit()]).reshape(-1, 1)
            rnoa_values = ratios.loc['Return on Net Operating Assets (RNOA) %'].dropna().values
            if len(years) > 1 and len(rnoa_values) > 1:
                model = LinearRegression().fit(years[:len(rnoa_values)], rnoa_values)
                next_year = years[-1] + 1
                forecast = model.predict([[next_year]])[0]
                insights = f"Forecasted RNOA for {next_year}: {forecast:.2f}%"
                if forecast > 10: insights += " (Strong operational efficiency projected)"
                elif forecast < 5: insights += " (Potential efficiency concerns)"
            else:
                insights = "Insufficient data for RNOA trend forecast."

            completeness = (valid_mask.sum() / len(valid_mask)) * 100 if len(valid_mask) > 0 else 0
            validation_data = {"ok": validation_check, "completeness": completeness, "insights": insights}

            return {
                "reformulated_bs": reformulated_bs,
                "ratios": ratios,
                "roe_decomposition": roe_decomposed,
                "validation": validation_data
            }
        except Exception as e:
            logger.error(f"Error in Penman-Nissim calculation: {e}")
            return {"error": str(e)}

    def _get(self, metric_name: str) -> pd.Series:
        key = self.mappings.get(metric_name)
        return self.df.loc[key] if key and key in self.df.index else pd.Series(np.nan, index=self.df.columns)

    def _get_multi(self, metric_list_name: str) -> pd.Series:
        keys = self.mappings.get(metric_list_name, [])
        valid_keys = [k for k in keys if k in self.df.index]
        return self.df.loc[valid_keys].sum() if valid_keys else pd.Series(np.nan, index=self.df.columns)

# --- 6. Core Application Logic ---

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
        metric_col = next((col for col in df.columns if col.lower() in ['metric', 'item', 'description']), df.columns[0])  # Enhanced to find likely metric column
        df = df.set_index(metric_col)
        return {"statement": df, "company_name": company_name, "source": uploaded_file.name}
    except Exception as e:
        logger.error(f"Failed to parse CSV file {uploaded_file.name}: {e}")
        return None

def parse_single_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    """Routes the uploaded file to the correct parser based on its extension."""
    file_extension = uploaded_file.name.split('.')[-1].lower()
    if file_extension == 'csv':
        parsed_data = parse_csv_file(uploaded_file)
    elif file_extension in ['html', 'htm', 'xls', 'xlsx']:
        parsed_data = parse_html_xls_file(uploaded_file)
    else:
        st.error(f"Unsupported file type: {file_extension}")
        return None
    
    if parsed_data is None:
        st.warning(f"Could not parse '{uploaded_file.name}'. It might be in an unsupported format or corrupted.")
        return None

    # Common processing for all file types
    df = parsed_data["statement"]
    year_cols_map = {}
    for col in df.columns:
        match = YEAR_REGEX.search(str(col))
        if match:
            year = match.group(0).replace('FY', '')  # Normalize 'FY2020' to '2020'
            year_cols_map[col] = year
    df = df.rename(columns=year_cols_map)
    valid_years = sorted(set(year for year in df.columns if str(year).isdigit()), key=int)
    
    if not valid_years:
        st.warning(f"No valid year columns found in '{uploaded_file.name}'.")
        return None
        
    df_proc = DataProcessor.clean_numeric_data(df[valid_years].copy()).dropna(how='all')
    parsed_data["statement"] = df_proc
    parsed_data["year_columns"] = valid_years
    return parsed_data

# Move process_and_merge_files outside the class and make it a standalone function
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
                        logger.warning(f"Duplicate metric '{metric}' found in {source}. Keeping first occurrence from {sources[metric]}.")
                    else:
                        sources[metric] = source
                all_dfs.append(df)
                if not first_company and parsed["company_name"] not in ["Unknown Company", "From CSV"]:
                    company_name = parsed["company_name"]
                    first_company = True
        
    if not all_dfs:
        st.error("None of the files could be parsed.")
        return None
    
    # Enhanced merging: Align on years, fill NaNs, add source traceability
    merged_df = pd.concat(all_dfs, axis=0, join='outer').groupby(level=0).first()  # Avoid duplicates by taking first
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
        # Call the standalone function directly, not as a method - THIS IS THE FIX
        st.session_state.analysis_data = process_and_merge_files(files)  # Removed self.
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
            st.experimental_rerun()
        st.sidebar.title("⚙️ Display Settings")
        st.sidebar.checkbox("Show Data Quality", key="show_data_quality", value=True)
        if st.session_state.analysis_data: 
            self._render_general_metric_mapper()

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

    def render_main_panel(self):
        st.markdown("<div class='main-header'>💹 Advanced Financial Dashboard</div>", unsafe_allow_html=True)
        if not st.session_state.analysis_data:
            st.info("👋 Welcome! Please upload one or more financial data files to begin.")
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
                # NEW: Trend forecast for selected ratios
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
        st.header("🔍 Penman-Nissim Reformulation Analysis")
        st.info("This analysis separates operating and financing activities to reveal a company's core operational profitability (RNOA). It provides a clearer view than traditional accounting metrics.")
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
                st.experimental_rerun()  # Refresh to show suggestions
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
                if validation["ok"]:
                    st.success("✅ Reformulation successful: The accounting equation holds.")
                else:
                    st.error("⚠️ Reformulation check failed. Review mappings.")
                if validation["completeness"] < 50:
                    st.warning(f"Data completeness: {validation['completeness']:.1f}%. Results may be incomplete.")
                st.info(validation["insights"])
                all_pn_metrics = pd.concat([results['reformulated_bs'], results['ratios'], results['roe_decomposition']])
                st.subheader("Visual Analysis of P-N Metrics")
                v_c1, v_c2, v_c3 = st.columns([2,1,1])
                selected = v_c1.multiselect("Select P-N metrics to plot:", all_pn_metrics.index.unique().tolist(), default=['Return on Net Operating Assets (RNOA) %', 'Operating Profit Margin (OPM) %'])
                chart_type = v_c2.selectbox("Chart Type", list(self.chart_builders.keys()), key="pn_chart")
                if selected:
                    fig = self.chart_builders[chart_type](all_pn_metrics, selected, "Penman-Nissim Metrics Over Time", "plotly_white", True, "Linear", "Value / Ratio", data["outliers"])
                    st.plotly_chart(fig, use_container_width=True)
                # NEW: Download P-N results
                csv = all_pn_metrics.to_csv().encode('utf-8')
                st.download_button("Download P-N Results as CSV", csv, "penman_nissim_results.csv", "text/csv")
                with st.expander("Reformulated Balance Sheet", expanded=False): st.dataframe(results['reformulated_bs'].style.format("{:,.2f}", na_rep="-"))
                with st.expander("Core P-N Ratios", expanded=False): st.dataframe(results['ratios'].style.format("{:,.2f}", na_rep="-"))
                with st.expander("ROE Decomposition Analysis", expanded=False): st.dataframe(results['roe_decomposition'].style.format("{:,.2f}", na_rep="-"))

# --- 7. App Execution ---
if __name__ == "__main__":
    try:
        # Install fuzzywuzzy if not present (for auto-population)
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "fuzzywuzzy", "python-levenshtein"])
        app = DashboardApp()
        app.run()
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        st.error(f"A critical error occurred. Please refresh. Details: {e}")
