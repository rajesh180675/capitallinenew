"""
Enhanced Financial Dashboard - Final Integrated Version
A powerful Streamlit-based dashboard for financial data analysis.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import logging
import re
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import warnings

# Configure page and logging
warnings.filterwarnings("ignore")
st.set_page_config(page_title="Capitaline Financial Dashboard", page_icon="📊", layout="wide")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------- CSS Styling --------------------
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem;
    font-weight: bold;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.stButton>button {
    background: linear-gradient(90deg, #1f77b4, #17a2b8);
    color: white;
    border-radius: 8px;
    padding: 12px 24px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Data Classes --------------------

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
            return False, "File too large (max 10MB)"
        ext = uploaded_file.name.lower().split('.')[-1]
        if ext not in ['xlsx', 'xls', 'html', 'htm']:
            return False, f"Unsupported file type: {ext}"
        return True, ext


class DataProcessor:
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        for col in df.columns:
            df[col] = df[col].astype(str).str.replace(r"[,\(\)₹]|Rs\.", "", regex=True)
            df[col] = df[col].str.replace('(', '-').str.replace(')', '')
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df

    @staticmethod
    def detect_outliers(df: pd.DataFrame) -> Dict[str, List[str]]:
        outliers = {}
        numeric_df = df.select_dtypes(include=np.number)
        for col in numeric_df.columns:
            q1 = numeric_df[col].quantile(0.25)
            q3 = numeric_df[col].quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                lower = q1 - 1.5 * iqr
                upper = q3 + 1.5 * iqr
                indices = numeric_df[(numeric_df[col] < lower) | (numeric_df[col] > upper)].index.tolist()
                if indices:
                    outliers[col] = indices
        return outliers

    @staticmethod
    def calculate_data_quality(df: pd.DataFrame) -> DataQualityMetrics:
        total_cells = df.size
        missing_values = df.isnull().sum().sum()
        return DataQualityMetrics(
            total_rows=df.shape[0],
            missing_values=missing_values,
            missing_percentage=(missing_values / total_cells * 100) if total_cells else 0,
            duplicate_rows=df.duplicated().sum()
        )

# -------------------- File Parser --------------------

@st.cache_data(show_spinner=False)
def parse_capitaline_file(uploaded_file) -> Optional[Dict[str, Any]]:
    is_valid, ext = FileValidator.validate_file(uploaded_file)
    if not is_valid:
        st.error(ext)
        return None

    try:
        content = uploaded_file.getvalue()

        if ext in ['html', 'htm']:
            df = pd.read_html(io.BytesIO(content), header=[0, 1])[0]
        elif ext == 'xls':
            df = pd.read_excel(io.BytesIO(content), header=[0, 1], engine="xlrd")
        else:
            df = pd.read_excel(io.BytesIO(content), header=[0, 1], engine="openpyxl")

        # Company Name
        company_name = "Unknown Company"
        try:
            info = str(df.columns[0][0])
            if ">>" in info:
                company_name = info.split(">>")[2].split("(")[0].strip()
        except:
            pass

        df.columns = [str(col[1]) for col in df.columns]
        df = df.rename(columns={df.columns[0]: "Metric"}).dropna(subset=["Metric"]).set_index("Metric")

        year_cols = {}
        for col in df.columns:
            year = "".join(filter(str.isdigit, str(col)))
            if len(year) == 4:
                year_cols[col] = year
        df = df.rename(columns=year_cols)

        valid_years = sorted([col for col in df.columns if col.isdigit() and 1990 < int(col) < 2050], reverse=True)
        if not valid_years:
            st.error("No valid year columns found.")
            return None

        df_final = df[valid_years].copy()
        df_final = DataProcessor.clean_numeric_data(df_final).dropna(how='all')

        return {
            "statement": df_final,
            "company_name": company_name,
            "data_quality": DataProcessor.calculate_data_quality(df_final),
            "outliers": DataProcessor.detect_outliers(df_final),
            "year_columns": valid_years,
            "file_info": {"name": uploaded_file.name, "size": uploaded_file.size, "type": ext}
        }

    except Exception as e:
        logger.error(f"Parsing error: {e}")
        st.error(f"Parsing failed: {e}")
        return None

# -------------------- Chart Renderer --------------------

def generate_chart(df: pd.DataFrame, metrics: List[str], chart_type: str):
    df_plot = df.loc[metrics].T
    df_plot.index = df_plot.index.astype(str)
    fig = go.Figure()
    for metric in metrics:
        y = df_plot[metric]
        if chart_type == "Line Chart":
            fig.add_trace(go.Scatter(x=df_plot.index, y=y, mode="lines+markers", name=metric))
        elif chart_type == "Bar Chart":
            fig.add_trace(go.Bar(x=df_plot.index, y=y, name=metric))
        elif chart_type == "Area Chart":
            fig.add_trace(go.Scatter(x=df_plot.index, y=y, fill="tozeroy", mode="lines", name=metric))
    fig.update_layout(
        title=chart_type,
        xaxis_title="Year",
        yaxis_title="Amount (₹ Cr.)",
        template="plotly_white"
    )
    return fig

# -------------------- Dashboard UI --------------------

def main():
    st.markdown('<div class="main-header">📊 Capitaline Financial Dashboard</div>', unsafe_allow_html=True)

    st.sidebar.header("Upload Capitaline File")
    uploaded_file = st.sidebar.file_uploader("Choose a file", type=["xls", "xlsx", "html", "htm"])

    if uploaded_file:
        st.sidebar.write("Parsing file...")
        result = parse_capitaline_file(uploaded_file)
        if result:
            df = result["statement"]
            company = result["company_name"]

            st.success(f"Loaded data for: {company}")

            # Display data quality
            with st.expander("📋 Data Quality"):
                dq = result["data_quality"]
                st.metric("Total Rows", dq.total_rows)
                st.metric("Missing Values", dq.missing_values)
                st.metric("Missing %", f"{dq.missing_percentage:.2f}%")
                st.metric("Duplicate Rows", dq.duplicate_rows)
                st.write(f"Quality Score: **{dq.quality_score}**")

            st.markdown("---")
            st.subheader("📈 Chart Visualization")
            metrics = st.multiselect("Select Metrics to Visualize", df.index.tolist()[:30])
            chart_type = st.selectbox("Select Chart Type", ["Line Chart", "Bar Chart", "Area Chart"])

            if st.button("Generate Chart"):
                if metrics:
                    chart = generate_chart(df, metrics, chart_type)
                    st.plotly_chart(chart, use_container_width=True)
                else:
                    st.warning("Please select at least one metric.")

            st.markdown("---")
            st.subheader("📄 Data Table")
            st.dataframe(df.style.format("{:,.2f}"))

            with st.expander("⚠️ Outliers Detected"):
                if result["outliers"]:
                    for year, metrics in result["outliers"].items():
                        st.write(f"**{year}**: {', '.join(metrics)}")
                else:
                    st.success("No significant outliers detected.")
        else:
            st.error("Failed to parse file.")

    else:
        st.markdown("""
            <div class="welcome-container">
            <h2>🎯 Welcome to the Capitaline Financial Dashboard</h2>
            <p>Upload your Capitaline xls/xlsx/html file in the sidebar to get started.</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
