# Enhanced Financial Dashboard - Complete Integrated Version
# A robust Streamlit application with Penman-Nissim analysis, multi-file support,
# and integrated visualization of calculated metrics.

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
from sklearn.linear_model import LinearRegression
from streamlit.runtime.uploaded_file_manager import UploadedFile

# --- 2. Configuration ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_TYPES = ['html', 'htm', 'xls', 'xlsx']
YEAR_REGEX = re.compile(r'\b(19[8-9]\d|20\d\d)\b')

# General required metrics for other tabs
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
    .st-expander { border: 1px solid #ddd; border-radius: 10px; }
</style>
""", unsafe_allow_html=True)

# --- 4. Data Structures and Basic Classes ---
@dataclass
class DataQualityMetrics:
    total_rows: int; missing_values: int; missing_percentage: float; duplicate_rows: int
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

class ChartGenerator:
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

# --- 5. Advanced Financial Analysis Modules ---
class FinancialRatioCalculator:
    @staticmethod
    def calculate_all_ratios(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        ratios = {}
        def get(name): return df.loc[name] if name in df.index else pd.Series(dtype=float, index=df.columns)
        
        profit_ratios, liq_ratios, lev_ratios = pd.DataFrame(index=df.columns), pd.DataFrame(index=df.columns), pd.DataFrame(index=df.columns)
        
        if not (rev := get('Revenue')).empty:
            if not (gp := get('Gross Profit')).empty: profit_ratios['Gross Margin %'] = (gp / rev) * 100
            if not (op := get('EBIT')).empty: profit_ratios['Operating Margin %'] = (op / rev) * 100
            if not (np := get('Net Profit')).empty: profit_ratios['Net Margin %'] = (np / rev) * 100
        
        if not (np := get('Net Profit')).empty:
            if not (eq := get('Total Equity')).empty: profit_ratios['ROE %'] = (np / eq) * 100
            if not (ta := get('Total Assets')).empty: profit_ratios['ROA %'] = (np / ta) * 100
        
        if not profit_ratios.empty: ratios['Profitability'] = profit_ratios.T.dropna(how='all')
        
        if not (ca := get('Current Assets')).empty and not (cl := get('Current Liabilities')).empty:
            liq_ratios['Current Ratio'] = ca / cl
        
        if not liq_ratios.empty: ratios['Liquidity'] = liq_ratios.T.dropna(how='all')
        
        if not (debt := get('Total Debt')).empty:
            if not (eq := get('Total Equity')).empty: lev_ratios['Debt to Equity'] = debt / eq
            if not (ta := get('Total Assets')).empty: lev_ratios['Debt to Assets'] = debt / ta
        
        if not (ebit := get('EBIT')).empty and not (ie := get('Interest Expense')).empty:
            if not ie.eq(0).all():
                lev_ratios['Interest Coverage'] = ebit / ie
        
        if not lev_ratios.empty: ratios['Leverage'] = lev_ratios.T.dropna(how='all')
        return ratios

class PenmanNissimAnalyzer:
    def __init__(self, df: pd.DataFrame, mappings: Dict[str, Any]):
        self.df = df
        self.mappings = mappings

    def _get(self, metric_name: str) -> pd.Series:
        key = self.mappings.get(metric_name)
        return self.df.loc[key] if key and key in self.df.index else pd.Series(dtype=float, index=self.df.columns)

    def _get_multi(self, metric_list_name: str) -> pd.Series:
        keys = self.mappings.get(metric_list_name, [])
        valid_keys = [k for k in keys if k and k in self.df.index]
        return self.df.loc[valid_keys].sum(axis=0) if valid_keys else pd.Series(0, index=self.df.columns)

    def calculate_all(self) -> Dict[str, Any]:
        total_assets = self._get('Total Assets')
        total_liabilities = self._get('Total Liabilities')
        equity = self._get('Total Equity')
        
        financial_assets = self._get_multi('Financial Assets')
        financial_liabilities = self._get_multi('Financial Liabilities')
        
        operating_assets = total_assets - financial_assets
        operating_liabilities = total_liabilities - financial_liabilities
        
        noa = operating_assets - operating_liabilities
        nfo = financial_liabilities - financial_assets
        
        validation_check = np.allclose(noa - nfo, equity, nan_policy='omit')
        
        reformulated_bs = pd.DataFrame({
            'Operating Assets (OA)': operating_assets, 'Financial Assets (FA)': financial_assets,
            'Operating Liabilities (OL)': operating_liabilities, 'Financial Liabilities (FL)': financial_liabilities,
            'Net Operating Assets (NOA)': noa, 'Net Financial Obligations (NFO)': nfo,
            'Total Equity': equity, 'Check (NOA-NFO-Equity)': noa - nfo - equity
        }).T
        
        oi = self._get('Operating Income')
        sales = self._get('Revenue')
        nfe = self._get('Net Financial Expense')
        
        avg_noa = (noa + noa.shift(1)) / 2
        avg_nfo = (nfo + nfo.shift(1)) / 2
        avg_equity = (equity + equity.shift(1)) / 2
        
        rnoa = (oi / avg_noa.replace(0, np.nan)) * 100
        opm = (oi / sales.replace(0, np.nan)) * 100
        noat = sales / avg_noa.replace(0, np.nan)
        
        ratios = pd.DataFrame({
            'Return on Net Operating Assets (RNOA) %': rnoa,
            'Operating Profit Margin (OPM) %': opm,
            'Net Operating Asset Turnover (NOAT)': noat,
        }).T
        
        nbc = (nfe / avg_nfo.replace(0, np.nan)) * 100
        flev = avg_nfo / avg_equity.replace(0, np.nan)
        spread = rnoa - nbc
        
        roe_decomposed = pd.DataFrame({
            'RNOA %': rnoa, 'Financial Leverage (FLEV)': flev,
            'Spread (RNOA - NBC) %': spread, 'Financing Contribution (FLEV * Spread)': flev * (spread / 100),
            'ROE (from P-N) %': rnoa + (flev * spread),
            'ROE (from statements) %': (self._get('Net Income') / avg_equity.replace(0, np.nan)) * 100,
        }).T
        
        return {"reformulated_bs": reformulated_bs.dropna(how='all', axis=1), "ratios": ratios.dropna(how='all', axis=1),
                "roe_decomposition": roe_decomposed.dropna(how='all', axis=1), "validation_ok": validation_check}

# --- 6. Core Application Logic ---
def parse_single_file(uploaded_file: UploadedFile) -> Optional[Dict[str, Any]]:
    # ... (Implementation unchanged)
    if not uploaded_file.name.split('.')[-1].lower() in ALLOWED_FILE_TYPES: return None
    try:
        df = pd.read_html(io.BytesIO(uploaded_file.getvalue()), header=[0, 1])[0]
        company_name = str(df.columns[0][0]).split(">>")[2].split("(")[0].strip() if ">>" in str(df.columns[0][0]) else "Unknown Company"
        df.columns = [str(c[1]) for c in df.columns]
        df = df.rename(columns={df.columns[0]: "Metric"}).dropna(subset=["Metric"]).reset_index(drop=True)
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
    except Exception: return None

class DashboardApp:
    def __init__(self):
        self._initialize_state()
        self.chart_builders = {"Line": ChartGenerator.create_line_chart, "Bar": ChartGenerator.create_bar_chart}

    def _initialize_state(self):
        defaults = {"analysis_data": None, "metric_mapping": {}, "pn_results": None}
        for k, v in defaults.items():
            if k not in st.session_state: st.session_state[k] = v

    @st.cache_data(show_spinner="Processing and merging files...")
    def process_and_merge_files(_self, uploaded_files: List[UploadedFile]) -> Optional[Dict[str, Any]]:
        if not uploaded_files: return None
        all_dfs, company_name, first_found = [], "Multiple Sources", False
        for file in uploaded_files:
            if parsed := parse_single_file(file):
                all_dfs.append(parsed["statement"])
                if not first_found and parsed["company_name"] != "Unknown Company":
                    company_name, first_found = parsed["company_name"], True
        if not all_dfs: st.error("None of the files could be parsed."); return None
        merged_df = pd.concat(all_dfs, axis=0)
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        year_cols = sorted([col for col in merged_df.columns if col.isdigit()], key=int)
        merged_df = merged_df[year_cols]
        return {"statement": merged_df, "company_name": company_name, "data_quality": asdict(DataProcessor.calculate_data_quality(merged_df))}

    def _handle_file_upload(self):
        files = st.session_state.get("file_uploader_key", [])
        st.session_state.analysis_data = self.process_and_merge_files(files)
        st.session_state.metric_mapping = {}
        st.session_state.pn_results = None

    def run(self):
        self.render_sidebar()
        self.render_main_panel()

    def render_sidebar(self):
        st.sidebar.title("📂 Upload & Options")
        st.sidebar.info("Upload one or more financial statements.")
        st.sidebar.file_uploader("Upload Financials (HTML/XLSX)", type=ALLOWED_FILE_TYPES, accept_multiple_files=True, key="file_uploader_key", on_change=self._handle_file_upload)
        st.sidebar.title("⚙️ Display Settings")
        st.sidebar.checkbox("Show Data Quality", key="show_data_quality", value=True)
        if st.session_state.analysis_data: self._render_general_metric_mapper()

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
                st.session_state.metric_mapping[std_metric] = st.selectbox(f"**{std_metric}**", options=[''] + available_metrics, index=(available_metrics.index(current_mapping.get(std_metric, '')) + 1) if current_mapping.get(std_metric) in available_metrics else 0, key=f"map_{std_metric}")

    def render_main_panel(self):
        st.markdown("<div class='main-header'>💹 Advanced Financial Dashboard</div>", unsafe_allow_html=True)
        if not st.session_state.analysis_data:
            st.info("👋 Welcome! Please upload one or more financial data files to begin.")
            return

        data = st.session_state.analysis_data
        df = data["statement"]
        dq = DataQualityMetrics(**data["data_quality"])
        
        st.subheader(f"Company Analysis: {data['company_name']}")
        if st.session_state.show_data_quality:
            qc = f"quality-{dq.quality_score.lower()}"
            st.markdown(f"""<div class="feature-card"><h4><span class="data-quality-indicator {qc}"></span>Merged Data Quality: {dq.quality_score}</h4>
            Total Unique Rows: {dq.total_rows} | Total Missing Values: {dq.missing_values} ({dq.missing_percentage:.2f}%)</div>""", unsafe_allow_html=True)

        tabs = ["📊 Primary Visualizations", "📄 Merged Data Table", "💡 Advanced Analysis", "🔍 Penman-Nissim Analysis"]
        tab_viz, tab_data, tab_adv, tab_pn = st.tabs(tabs)
        
        with tab_viz:
            self._render_primary_visualization_tab(df)
        with tab_data:
            self._render_data_table_tab(df)
        with tab_adv:
            self._render_advanced_analysis_tab(df)
        with tab_pn:
            self._render_penman_nissim_tab(df)

    def _render_data_table_tab(self, df: pd.DataFrame):
        st.subheader("Merged and Cleaned Financial Data")
        st.dataframe(df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)

    def _render_penman_nissim_tab(self, df: pd.DataFrame):
        st.header("🔍 Penman-Nissim Reformulation Analysis")
        st.info("This analysis separates operating and financing activities to reveal a company's core operational profitability (RNOA).")
        
        available_metrics = df.index.tolist()
        pn_mappings = {}

        with st.expander("Configure Penman-Nissim Metrics", expanded=True):
            st.markdown("##### Balance Sheet Items")
            c1, c2 = st.columns(2)
            pn_mappings['Financial Assets'] = c1.multiselect("Select Financial Assets", available_metrics, help="e.g., Cash, Marketable Securities")
            pn_mappings['Financial Liabilities'] = c2.multiselect("Select Financial Liabilities", available_metrics, help="e.g., Short-term Debt, Long-term Debt")
            
            st.markdown("##### Core Statement Items")
            c1, c2, c3, c4, c5 = st.columns(5)
            def get_idx(m): return available_metrics.index(m) + 1 if m in available_metrics else 0
            pn_mappings['Total Assets'] = c1.selectbox("Total Assets", [''] + available_metrics, index=get_idx('Total Assets'))
            pn_mappings['Total Liabilities'] = c2.selectbox("Total Liabilities", [''] + available_metrics, index=get_idx('Total Liabilities'))
            pn_mappings['Total Equity'] = c3.selectbox("Total Equity", [''] + available_metrics, index=get_idx('Total Equity'))
            pn_mappings['Revenue'] = c4.selectbox("Revenue", [''] + available_metrics, index=get_idx('Revenue'))
            pn_mappings['Net Income'] = c5.selectbox("Net Income", [''] + available_metrics, index=get_idx('Net Income'))

            st.markdown("##### Income and Expense Proxies")
            c1, c2 = st.columns(2)
            pn_mappings['Operating Income'] = c1.selectbox("Operating Income (Proxy)", [''] + available_metrics, help="EBIT is a common proxy.", index=get_idx('EBIT'))
            pn_mappings['Net Financial Expense'] = c2.selectbox("Net Financial Expense (Proxy)", [''] + available_metrics, help="Interest Expense is a common proxy.", index=get_idx('Interest Expense'))

            if st.button("🚀 Run Penman-Nissim Analysis"):
                analyzer = PenmanNissimAnalyzer(df, pn_mappings)
                st.session_state.pn_results = analyzer.calculate_all()
        
        st.markdown("---")

        if st.session_state.pn_results:
            results = st.session_state.pn_results
            if results["validation_ok"]:
                st.success("✅ Reformulation successful: The accounting equation (NOA - NFO = Equity) holds true.")
            else:
                st.error("⚠️ Reformulation check failed. The accounting equation does not balance. Please review your metric selections.")

            all_pn_metrics = pd.concat([results['reformulated_bs'], results['ratios'], results['roe_decomposition']])
            st.subheader("Visual Analysis of P-N Metrics")
            v_c1, v_c2, v_c3 = st.columns([2,1,1])
            selected = v_c1.multiselect("Select P-N metrics to plot:", all_pn_metrics.index.unique().tolist(), default=['Return on Net Operating Assets (RNOA) %', 'Operating Profit Margin (OPM) %'])
            chart_type = v_c2.selectbox("Chart Type", self.chart_builders.keys(), key="pn_chart")
            if selected:
                fig = self.chart_builders[chart_type](all_pn_metrics, selected, "Penman-Nissim Metrics Over Time", "plotly_white", True, "Linear", "Value / Ratio")
                st.plotly_chart(fig, use_container_width=True)

            with st.expander("Reformulated Balance Sheet", expanded=False): st.dataframe(results['reformulated_bs'].style.format("{:,.2f}", na_rep="-"))
            with st.expander("Core P-N Ratios", expanded=False): st.dataframe(results['ratios'].style.format("{:,.2f}", na_rep="-"))
            with st.expander("ROE Decomposition Analysis", expanded=False): st.dataframe(results['roe_decomposition'].style.format("{:,.2f}", na_rep="-"))
    
    def _render_advanced_analysis_tab(self, df: pd.DataFrame):
        st.header("💡 General Advanced Analysis")
        mapping = {v: k for k, v in st.session_state.metric_mapping.items() if v}
        if not mapping: st.warning("Please map metrics in the sidebar for this tab."); return
        mapped_df = df.rename(index=mapping)
        ratios = FinancialRatioCalculator.calculate_all_ratios(mapped_df)
        if ratios:
            all_ratios_df = pd.concat([df for df in ratios.values()]).dropna(how='all')
            if not all_ratios_df.empty:
                st.subheader("Visual Analysis of General Ratios")
                v_c1, v_c2, v_c3 = st.columns([2,1,1])
                selected = v_c1.multiselect("Select Ratios to plot:", all_ratios_df.index.unique().tolist(), default=all_ratios_df.index.unique().tolist()[:2])
                chart_type = v_c2.selectbox("Chart Type", self.chart_builders.keys(), key="adv_chart")
                if selected:
                    fig = self.chart_builders[chart_type](all_ratios_df, selected, "Ratio Analysis", "plotly_white", True, "Linear", "Ratio")
                    st.plotly_chart(fig, use_container_width=True)
            with st.expander("Data Tables", expanded=False):
                for ratio_type, ratio_df in ratios.items():
                    st.subheader(f"{ratio_type} Ratios")
                    st.dataframe(ratio_df.style.format("{:,.2f}", na_rep="-"), use_container_width=True)

    def _render_primary_visualization_tab(self, df: pd.DataFrame):
        st.header("Primary Financial Data Visualization")
        col1, col2, col3, col4 = st.columns([3, 1, 1, 2])
        metrics = col1.multiselect("Select metrics from uploaded files:", df.index.tolist(), default=df.index[:2].tolist())
        chart = col2.selectbox("Chart Type:", self.chart_builders.keys(), key="primary_chart_type")
        theme = col3.selectbox("Theme:", ["plotly_white", "plotly_dark"], key="primary_theme")
        scale = col4.selectbox("Y-Axis Scale:", ["Linear", "Logarithmic", "Normalized (Base 100)"], key="primary_scale")
        if metrics:
            plot_df, y_title = (DataProcessor.normalize_to_100(df, metrics), "Normalized Value (Base 100)") if scale == "Normalized (Base 100)" else (df, "Amount (₹ Cr.)")
            fig = self.chart_builders[chart](plot_df, metrics, "Primary Financials Over Time", theme, True, scale, y_title)
            if fig: st.plotly_chart(fig, use_container_width=True)
        else: st.warning("Please select at least one metric.")

# --- 7. App Execution ---
if __name__ == "__main__":
    try:
        app = DashboardApp()
        app.run()
    except Exception as e:
        logger.critical(f"A critical error occurred: {e}", exc_info=True)
        st.error(f"A critical error occurred. Please refresh. Details: {e}")
