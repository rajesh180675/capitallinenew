# Enhanced Financial Dashboard - PhD-Level Penman-Nissim Integration
# Now with Ind-AS Text Parsing Support and Advanced Features

# --- 1. Imports and Setup ---
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime, timedelta
import warnings
import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, asdict
import json
import re
from io import StringIO, BytesIO
import base64
from pathlib import Path
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns
import matplotlib.pyplot as plt
from bs4 import BeautifulSoup
from fuzzywuzzy import fuzz
from sentence_transformers import SentenceTransformer
from streamlit.runtime.uploaded_file_manager import UploadedFile

# --- 2. Configuration ---
warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
ALLOWED_FILE_TYPES = ['csv', 'html', 'xls', 'xlsx']
PENMAN_NISSIM_METRICS = [
    'Revenue', 'Operating Income', 'Operating Cash Flow', 'Net Income',
    'Net Financial Expense', 'Total Assets', 'Total Liabilities', 
    'Total Equity', 'Current Assets', 'Current Liabilities',
    'Cash and Cash Equivalents', 'Inventories', 'Accounts Receivable',
    'Accounts Payable', 'Financial Assets', 'Financial Liabilities'
]

INDUSTRY_BENCHMARKS = {
    "Technology": {
        "Current Ratio": 2.5,
        "Quick Ratio": 2.0,
        "Debt to Equity": 0.5,
        "ROE": 0.20,
        "ROA": 0.15,
        "Asset Turnover": 0.8,
        "Profit Margin": 0.15
    },
    "Manufacturing": {
        "Current Ratio": 1.8,
        "Quick Ratio": 1.0,
        "Debt to Equity": 1.0,
        "ROE": 0.15,
        "ROA": 0.08,
        "Asset Turnover": 1.2,
        "Profit Margin": 0.08
    },
    "Retail": {
        "Current Ratio": 1.5,
        "Quick Ratio": 0.7,
        "Debt to Equity": 1.2,
        "ROE": 0.18,
        "ROA": 0.06,
        "Asset Turnover": 2.0,
        "Profit Margin": 0.04
    },
    "Financial Services": {
        "Current Ratio": 1.2,
        "Quick Ratio": 1.0,
        "Debt to Equity": 3.0,
        "ROE": 0.12,
        "ROA": 0.02,
        "Asset Turnover": 0.1,
        "Profit Margin": 0.20
    }
}

# --- 3. Page and Style Configuration ---
st.set_page_config(
    page_title="Elite Financial Analytics Platform",
    page_icon="💹",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1f77b4;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        transition: transform 0.3s ease;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-left: 20px;
        padding-right: 20px;
        background-color: #f0f2f6;
        border-radius: 10px 10px 0 0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
    .insight-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #ffc107;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #28a745;
        margin: 1rem 0;
    }
    .mapping-quality-high { color: #28a745; font-weight: bold; }
    .mapping-quality-medium { color: #ffc107; font-weight: bold; }
    .mapping-quality-low { color: #dc3545; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# --- 4. Data Structures and Basic Classes ---
@dataclass
class DataQuality:
    completeness: float
    consistency: float
    outlier_percentage: float
    data_points: int
    time_span_years: int
    missing_critical_metrics: List[str]

@dataclass
class PenmanNissimResults:
    roce: pd.Series
    rnoa: pd.Series
    financial_leverage: pd.Series
    operating_liability_leverage: pd.Series
    spread: pd.Series
    nbc: pd.Series
    pm: pd.Series
    ato: pd.Series
    decomposition_df: pd.DataFrame
    quality_score: float
    warnings: List[str]

class DataProcessor:
    @staticmethod
    def clean_numeric_data(df: pd.DataFrame) -> pd.DataFrame:
        """Clean and standardize numeric data"""
        numeric_df = df.copy()
        for col in numeric_df.columns:
            if numeric_df[col].dtype == 'object':
                numeric_df[col] = pd.to_numeric(
                    numeric_df[col].astype(str).str.replace(',', '').str.replace('$', ''),
                    errors='coerce'
                )
        return numeric_df
    
    @staticmethod
    def calculate_data_quality(df: pd.DataFrame) -> DataQuality:
        """Calculate comprehensive data quality metrics"""
        total_cells = df.size
        non_null_cells = df.count().sum()
        completeness = non_null_cells / total_cells if total_cells > 0 else 0
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        consistency = 1.0
        for col in numeric_cols:
            if len(df[col].dropna()) > 1:
                cv = df[col].std() / df[col].mean() if df[col].mean() != 0 else 0
                if cv > 2:
                    consistency *= 0.9
        
        outlier_count = 0
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[col] < (Q1 - 1.5 * IQR)) | (df[col] > (Q3 + 1.5 * IQR))).sum()
            outlier_count += outliers
        
        outlier_percentage = outlier_count / (len(numeric_cols) * len(df)) if len(numeric_cols) > 0 else 0
        
        year_columns = [col for col in df.columns if str(col).isdigit()]
        time_span_years = len(year_columns)
        
        critical_metrics = ['Revenue', 'Total Assets', 'Total Equity']
        missing_critical = [m for m in critical_metrics if m not in df.index]
        
        return DataQuality(
            completeness=completeness,
            consistency=consistency,
            outlier_percentage=outlier_percentage,
            data_points=len(df),
            time_span_years=time_span_years,
            missing_critical_metrics=missing_critical
        )
    
    @staticmethod
    def detect_outliers(df: pd.DataFrame) -> Dict[str, List[Tuple[str, float]]]:
        """Detect outliers using IQR method"""
        outliers = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            col_outliers = []
            for idx, value in df[col].items():
                if pd.notna(value) and (value < lower_bound or value > upper_bound):
                    col_outliers.append((str(idx), float(value)))
            
            if col_outliers:
                outliers[str(col)] = col_outliers
        
        return outliers

# --- 5. Enhanced Ind-AS Parser Class ---
class IndASParser:
    """Parser for Ind-AS formatted text reports (BS and P&L)"""
    
    @staticmethod
    def parse_number(val_str: str) -> float:
        """Enhanced number parsing with support for various formats"""
        val_str = val_str.strip()
        if val_str in ['-', '--', '', 'NA', 'N/A', 'nil', 'Nil']:
            return 0.0
        
        # Handle lakhs/crores notation
        multiplier = 1
        if 'cr' in val_str.lower():
            multiplier = 10000000
            val_str = re.sub(r'cr\.?', '', val_str, flags=re.IGNORECASE)
        elif 'lakh' in val_str.lower() or 'lac' in val_str.lower():
            multiplier = 100000
            val_str = re.sub(r'la(kh|c)s?\.?', '', val_str, flags=re.IGNORECASE)
        elif 'mn' in val_str.lower() or 'million' in val_str.lower():
            multiplier = 1000000
            val_str = re.sub(r'(mn|million)\.?', '', val_str, flags=re.IGNORECASE)
        
        # Remove common symbols and parse
        val_str = val_str.replace(',', '').replace('₹', '').replace('Rs', '').replace('INR', '')
        val_str = val_str.replace('(', '-').replace(')', '')
        
        try:
            return float(val_str) * multiplier
        except:
            return 0.0
    
    @staticmethod
    def validate_ind_as_text(text: str) -> Tuple[bool, str]:
        """Validate if the pasted text appears to be a financial statement"""
        if len(text.strip()) < 100:
            return False, "Text too short to be a financial statement"
        
        # Check for key indicators
        indicators = ['assets', 'liabilities', 'revenue', 'expenses', 'profit', 'loss', 
                     'income', 'equity', 'cash', 'operations']
        text_lower = text.lower()
        
        if not any(ind in text_lower for ind in indicators):
            return False, "Text doesn't appear to contain financial statement data"
        
        # Check for numeric values
        numbers = re.findall(r'\d+\.?\d*', text)
        if len(numbers) < 10:
            return False, "Insufficient numeric data found"
        
        return True, "Valid"
    
    @staticmethod
    def parse_ind_as_text(text: str, report_type: str = 'auto') -> pd.DataFrame:
        """
        Parse pasted Ind-AS report text into DataFrame
        
        Args:
            text: The pasted text from Ind-AS report
            report_type: 'bs', 'pl', or 'auto' (auto-detect)
        """
        lines = text.strip().split('\n')
        data = []
        years = []
        current_section = None
        
        # Auto-detect report type if needed
        if report_type == 'auto':
            text_lower = text.lower()
            if any(term in text_lower for term in ['revenue from operations', 'total revenue', 'profit before tax', 'profit after tax']):
                report_type = 'pl'
            elif any(term in text_lower for term in ['total assets', 'shareholders fund', 'non-current assets', 'current assets']):
                report_type = 'bs'
        
        # Extract years from header if present
        for line in lines[:10]:  # Check first 10 lines for year pattern
            year_matches = re.findall(r'20\d{2}', line)
            if len(year_matches) >= 2:  # Found year row
                years = year_matches[:10]  # Take up to 10 years
                break
        
        # Also check for fiscal year patterns (e.g., "2023-24")
        if not years:
            for line in lines[:10]:
                fy_matches = re.findall(r'20\d{2}-\d{2}', line)
                if len(fy_matches) >= 2:
                    years = [match.split('-')[0] for match in fy_matches[:10]]
                    break
        
        if not years:
            years = [f'Year_{i+1}' for i in range(10)]
        
        # Parse data rows
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip headers and section markers
            if line.endswith(':') or line.upper() in ['ASSETS', 'LIABILITIES', 'REVENUE', 'EXPENSES', 'EQUITY AND LIABILITIES']:
                current_section = line
                continue
            
            # Skip lines that are clearly headers or notes
            if any(skip in line.lower() for skip in ['particulars', 'note', 'as at', 'for the year']):
                continue
            
            # Parse metric rows
            parts = re.split(r'\s{2,}|\t+', line)
            if len(parts) >= 2:  # At least metric name + 1 value
                metric = parts[0].strip()
                
                # Clean metric name
                metric = re.sub(r'^\d+\.\s*', '', metric)  # Remove numbering
                metric = re.sub(r'^[a-zA-Z]\)\s*', '', metric)  # Remove letter numbering
                metric = metric.replace('...', '').strip()
                
                if not metric or metric.isdigit():
                    continue
                
                # Extract values
                values = []
                for i in range(1, min(len(parts), 11)):  # Up to 10 values
                    val_str = parts[i].strip()
                    values.append(IndASParser.parse_number(val_str))
                
                # Pad with zeros if needed
                while len(values) < len(years):
                    values.append(0.0)
                
                data.append([metric] + values[:len(years)])
        
        # Create DataFrame
        columns = ['Metric'] + years
        df = pd.DataFrame(data, columns=columns)
        df = df.set_index('Metric')
        
        # P&L specific adjustments
        if report_type == 'pl':
            df = IndASParser._adjust_pl_calculations(df)
        
        return DataProcessor.clean_numeric_data(df)
    
    @staticmethod
    def _adjust_pl_calculations(df: pd.DataFrame) -> pd.DataFrame:
        """Apply P&L specific calculations"""
        # Calculate net revenue if needed
        if 'Revenue From Operations' in df.index and 'Less: Excise Duty' in df.index:
            df.loc['Revenue From Operations (Net)'] = (
                df.loc['Revenue From Operations'] - df.loc['Less: Excise Duty']
            )
        
        # Ensure Operating Income exists
        if 'Operating Income' not in df.index:
            if 'Profit Before Exceptional Items and Tax' in df.index:
                df.loc['Operating Income'] = df.loc['Profit Before Exceptional Items and Tax']
            elif 'Profit Before Tax' in df.index:
                df.loc['Operating Income'] = df.loc['Profit Before Tax']
        
        return df

# --- 6. Enhanced Intelligent Financial Mapper ---
class IntelligentFinancialMapper:
    """AI-powered financial metric mapping with Ind-AS support"""
    
    def __init__(self, accounting_standard: str = 'IND-AS'):
        self.accounting_standard = accounting_standard
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        except:
            self.embedder = None
            logger.warning("Sentence transformer not available, using pattern matching only")
        
        # Ind-AS specific mappings
        self.ind_as_mappings = {
            # Balance Sheet mappings
            'Total Assets': ['total assets', 'assets total'],
            'Total Liabilities': ['total liabilities', 'liabilities total'],
            'Total Equity': ['total equity', 'shareholders fund', 'shareholders funds', 'shareholder fund'],
            'Current Assets': ['current assets', 'total current assets'],
            'Current Liabilities': ['current liabilities', 'total current liabilities'],
            'Cash and Cash Equivalents': ['cash and cash equivalents', 'cash & cash equivalents'],
            'Inventories': ['inventories', 'inventory', 'stock'],
            'Accounts Receivable': ['trade receivables', 'debtors', 'accounts receivable', 'sundry debtors'],
            'Accounts Payable': ['trade payables', 'creditors', 'accounts payable', 'sundry creditors'],
            
            # P&L mappings
            'Revenue': ['revenue from operations', 'total revenue', 'sales', 'turnover', 'revenue from operations (net)'],
            'Operating Income': ['profit before exceptional items and tax', 'operating profit', 'ebit', 
                               'earnings before interest and tax', 'profit before tax'],
            'Net Income': ['profit after tax', 'net profit', 'profit for the year', 'pat'],
            'Net Financial Expense': ['finance cost', 'finance costs', 'interest expense', 
                                    'interest and finance charges'],
            'Operating Cash Flow': ['cash flow from operating activities', 'operating cash flow', 
                                  'cash generated from operations'],
            
            # Quality metrics
            'Gross Profit': ['gross profit', 'gross margin'],
            'Depreciation': ['depreciation and amortisation', 'depreciation', 'd&a'],
            'Tax Expense': ['tax expense', 'income tax expense', 'provision for tax'],
            'Interest Expense': ['finance cost', 'interest expense', 'finance charges']
        }
        
        # Financial vs Operating classifiers
        self.financial_patterns = [
            'investment', 'cash', 'bank', 'securities', 'borrowing', 'loan', 'debt', 
            'debenture', 'bond', 'finance lease', 'interest bearing'
        ]
        
        self.operating_patterns = [
            'property', 'plant', 'equipment', 'inventory', 'receivable', 'payable',
            'prepaid', 'accrued', 'deferred revenue', 'advance', 'trade'
        ]
    
    def map_metrics(self, available_metrics: List[str]) -> Dict[str, Any]:
        """Map available metrics to P-N requirements"""
        mappings = {}
        
        # Single-value mappings
        for pn_metric, patterns in self.ind_as_mappings.items():
            if pn_metric not in ['Financial Assets', 'Financial Liabilities']:
                best_match = self._find_best_match(pn_metric, patterns, available_metrics)
                mappings[pn_metric] = best_match
        
        # Multi-value mappings (Financial Assets/Liabilities)
        mappings['Financial Assets'] = self._classify_financial_items(
            available_metrics, 'assets'
        )
        mappings['Financial Liabilities'] = self._classify_financial_items(
            available_metrics, 'liabilities'
        )
        
        # Apply proxies for missing critical items
        mappings = self._apply_ind_as_proxies(mappings, available_metrics)
        
        return mappings
    
    def _find_best_match(self, target: str, patterns: List[str], candidates: List[str]) -> Optional[str]:
        """Find best match using pattern matching and embeddings"""
        # First try exact pattern matching
        for candidate in candidates:
            candidate_lower = candidate.lower()
            for pattern in patterns:
                if pattern in candidate_lower:
                    return candidate
        
        # Fallback to fuzzy matching
        best_match = None
        best_score = 0
        for candidate in candidates:
            for pattern in patterns:
                score = fuzz.token_sort_ratio(pattern, candidate.lower())
                if score > best_score and score > 70:
                    best_score = score
                    best_match = candidate
        
        # Try embedding similarity if available
        if self.embedder and not best_match:
            try:
                target_embedding = self.embedder.encode([target])
                candidate_embeddings = self.embedder.encode(candidates)
                similarities = cosine_similarity(target_embedding, candidate_embeddings)[0]
                
                best_idx = np.argmax(similarities)
                if similarities[best_idx] > 0.7:  # Threshold
                    best_match = candidates[best_idx]
            except:
                pass
        
        return best_match
    
    def _classify_financial_items(self, metrics: List[str], item_type: str) -> List[str]:
        """Classify items as financial vs operating"""
        financial_items = []
        
        for metric in metrics:
            metric_lower = metric.lower()
            
            # Check if it's the right type (asset or liability)
            if item_type == 'assets' and any(x in metric_lower for x in ['asset', 'investment', 'cash']):
                if any(pattern in metric_lower for pattern in self.financial_patterns):
                    financial_items.append(metric)
            elif item_type == 'liabilities' and any(x in metric_lower for x in ['liability', 'borrowing', 'debt']):
                if any(pattern in metric_lower for pattern in self.financial_patterns):
                    financial_items.append(metric)
        
        return financial_items[:10]  # Limit to top 10
    
    def _apply_ind_as_proxies(self, mappings: Dict, available_metrics: List[str]) -> Dict:
        """Apply Ind-AS specific proxies"""
        # Operating Income proxy
        if not mappings.get('Operating Income'):
            # Try EBITDA or EBIT patterns
            for metric in available_metrics:
                if any(x in metric.lower() for x in ['ebitda', 'ebit', 'operating profit']):
                    mappings['Operating Income'] = metric
                    break
        
        # Net Financial Expense proxy
        if not mappings.get('Net Financial Expense'):
            # Check for Finance Cost (common in Ind-AS)
            for metric in available_metrics:
                if 'finance cost' in metric.lower():
                    mappings['Net Financial Expense'] = metric
                    break
            
            # If still not found, set to None (will be handled as 0)
            if not mappings.get('Net Financial Expense'):
                mappings['Net Financial Expense'] = None
        
        return mappings

# --- 7. Utility Functions ---
def export_mappings(mappings: Dict[str, Any], filename: str = "pn_mappings.json") -> str:
    """Export mappings for reuse"""
    export_data = {
        "timestamp": datetime.now().isoformat(),
        "accounting_standard": "IND-AS",
        "mappings": mappings,
        "version": "1.0"
    }
    
    return json.dumps(export_data, indent=2)

def display_mapping_quality(mappings: Dict[str, Any]) -> None:
    """Display mapping quality metrics"""
    total_required = len(PENMAN_NISSIM_METRICS)
    mapped = sum(1 for v in mappings.values() if v and v != [])
    
    # Calculate mapping score
    critical_metrics = ['Revenue', 'Operating Income', 'Net Income', 'Total Assets', 'Total Equity']
    critical_mapped = sum(1 for m in critical_metrics if mappings.get(m))
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Mappings", f"{mapped}/{total_required}", 
                  delta=f"{(mapped/total_required)*100:.0f}%")
    with col2:
        st.metric("Critical Metrics", f"{critical_mapped}/{len(critical_metrics)}")
    with col3:
        quality = "High" if mapped/total_required > 0.8 else "Medium" if mapped/total_required > 0.5 else "Low"
        st.metric("Mapping Quality", quality)

def load_sample_ind_as_data() -> Tuple[str, str]:
    """Load sample Ind-AS formatted data for testing"""
    sample_bs = """BALANCE SHEET
As at March 31 (Rs. in Lakhs)

Particulars                          2024      2023      2022      2021      2020
ASSETS:
Non-Current Assets:
Property, Plant and Equipment      12,345.67  11,234.56  10,123.45  9,012.34  8,901.23
Intangible Assets                   1,234.56   1,123.45   1,012.34    901.23    890.12
Financial Assets                    5,678.90   5,123.45   4,567.89   4,012.34  3,456.78
Deferred Tax Assets                   456.78     412.34     378.90     345.67    312.45

Current Assets:
Inventories                         3,456.78   3,123.45   2,890.12   2,567.89  2,234.56
Trade Receivables                   4,567.89   4,123.45   3,789.01   3,345.67  3,012.34
Cash and Cash Equivalents           2,345.67   2,123.45   1,901.23   1,678.90  1,456.78
Other Current Assets                  890.12     812.34     734.56     656.78    589.90

TOTAL ASSETS                       30,976.37  27,076.39  24,397.50  21,521.82 19,854.16

EQUITY AND LIABILITIES:
Equity:
Share Capital                       1,000.00   1,000.00   1,000.00   1,000.00  1,000.00
Other Equity                       12,456.78  11,234.56  10,123.45   9,012.34  8,123.45
Total Equity                       13,456.78  12,234.56  11,123.45  10,012.34  9,123.45

Non-Current Liabilities:
Long-term Borrowings                5,678.90   4,567.89   3,789.01   3,012.34  2,456.78
Other Non-Current Liabilities         890.12     789.01     678.90     567.89    456.78

Current Liabilities:
Short-term Borrowings               2,345.67   2,123.45   1,901.23   1,678.90  1,456.78
Trade Payables                      3,456.78   3,123.45   2,890.12   2,567.89  2,345.67
Other Current Liabilities           5,148.12   4,238.03   4,014.79   3,682.46  4,014.70

TOTAL EQUITY AND LIABILITIES       30,976.37  27,076.39  24,397.50  21,521.82 19,854.16"""
    
    sample_pl = """PROFIT AND LOSS STATEMENT
For the year ended March 31 (Rs. in Lakhs)

Particulars                          2024      2023      2022      2021      2020
Revenue From Operations            45,678.90  41,234.56  37,890.12  34,567.89 31,234.56
Other Income                          567.89     512.34     456.78     401.23    345.67

Total Revenue                      46,246.79  41,746.90  38,346.90  34,969.12 31,580.23

EXPENSES:
Cost of Materials Consumed         23,456.78  21,123.45  19,456.78  17,890.12 16,234.56
Employee Benefits Expense           5,678.90   5,123.45   4,678.90   4,234.56  3,890.12
Finance Costs                       1,234.56   1,123.45   1,012.34     901.23    789.01
Depreciation and Amortisation       2,345.67   2,123.45   1,901.23   1,678.90  1,456.78
Other Expenses                      3,999.00   3,499.00   2,999.00   2,599.00  2,209.99

Total Expenses                     36,714.91  32,992.80  30,048.25  27,303.81 24,580.46

Profit Before Exceptional Items    
and Tax                             9,531.88   8,754.10   8,298.65   7,665.31  7,000.77

Exceptional Items                          -          -          -          -         -

Profit Before Tax                   9,531.88   8,754.10   8,298.65   7,665.31  7,000.77

Tax Expense:
Current Tax                         2,382.97   2,188.53   2,074.66   1,916.33  1,750.19
Deferred Tax                          100.00      75.00      50.00      25.00     20.00

Total Tax Expense                   2,482.97   2,263.53   2,124.66   1,941.33  1,770.19

Profit After Tax                    7,048.91   6,490.57   6,173.99   5,723.98  5,230.58"""
    
    return sample_bs, sample_pl

# --- 8. File Parsing and Processing ---
def parse_html_xls_file(file) -> Optional[pd.DataFrame]:
    """Parse HTML or XLS file containing financial data"""
    try:
        if file.name.endswith('.html'):
            soup = BeautifulSoup(file.read(), 'html.parser')
            tables = soup.find_all('table')
            
            for table in tables:
                try:
                    df = pd.read_html(str(table))[0]
                    if len(df.columns) > 3 and len(df) > 5:
                        return df
                except:
                    continue
                    
        elif file.name.endswith(('.xls', '.xlsx')):
            xl_file = pd.ExcelFile(file)
            
            for sheet_name in xl_file.sheet_names:
                df = xl_file.parse(sheet_name)
                if len(df.columns) > 3 and len(df) > 5:
                    return df
                    
    except Exception as e:
        logger.error(f"Error parsing {file.name}: {str(e)}")
    
    return None

def parse_csv_file(file) -> Optional[pd.DataFrame]:
    """Parse CSV file with intelligent delimiter detection"""
    try:
        content = file.read().decode('utf-8')
        file.seek(0)
        
        # Try different delimiters
        for delimiter in [',', '\t', ';', '|']:
            try:
                df = pd.read_csv(StringIO(content), delimiter=delimiter)
                if len(df.columns) > 1:
                    return df
            except:
                continue
                
    except Exception as e:
        logger.error(f"Error parsing CSV {file.name}: {str(e)}")
    
    return None

def parse_ind_as_text_input(bs_text: str, pl_text: str, company_name: str = "Company") -> Optional[Dict[str, Any]]:
    """Parse pasted Ind-AS text inputs"""
    try:
        dfs_to_merge = []
        
        # Parse BS if provided
        if bs_text.strip():
            bs_df = IndASParser.parse_ind_as_text(bs_text, 'bs')
            dfs_to_merge.append(bs_df)
            logger.info(f"Parsed BS with {len(bs_df)} metrics")
        
        # Parse P&L if provided
        if pl_text.strip():
            pl_df = IndASParser.parse_ind_as_text(pl_text, 'pl')
            dfs_to_merge.append(pl_df)
            logger.info(f"Parsed P&L with {len(pl_df)} metrics")
        
        if not dfs_to_merge:
            return None
        
        # Merge DataFrames
        merged_df = pd.concat(dfs_to_merge, axis=0)
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
        
        # Get year columns
        year_columns = [col for col in merged_df.columns if str(col).isdigit() or col.startswith('Year_')]
        
        # Calculate data quality
        data_quality = asdict(DataProcessor.calculate_data_quality(merged_df))
        
        # Detect outliers
        outliers = DataProcessor.detect_outliers(merged_df)
        
        return {
            "statement": merged_df,
            "company_name": company_name,
            "data_quality": data_quality,
            "outliers": outliers,
            "year_columns": year_columns,
            "sources": {"All": "Pasted Text"}
        }
        
    except Exception as e:
        logger.error(f"Error parsing Ind-AS text: {e}")
        return None

@st.cache_data(show_spinner="Processing and merging data...")
def process_and_merge_files(_uploaded_files: List[UploadedFile] = None, 
                           text_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """Process and merge multiple uploaded files OR text input"""
    if text_data:
        # Return text-parsed data directly
        return text_data
    
    if not _uploaded_files:
        return None
    
    all_dfs = []
    sources = {}
    
    for file in _uploaded_files:
        df = None
        
        try:
            if file.name.endswith('.csv'):
                df = parse_csv_file(file)
            elif file.name.endswith(('.html', '.xls', '.xlsx')):
                df = parse_html_xls_file(file)
            
            if df is not None:
                # Clean column names
                df.columns = df.columns.astype(str).str.strip()
                
                # Try to identify metric column
                possible_metric_cols = ['Metric', 'Item', 'Account', 'Description', 'Particulars']
                metric_col = None
                
                for col in possible_metric_cols:
                    if col in df.columns or any(col.lower() in c.lower() for c in df.columns):
                        metric_col = next(c for c in df.columns if col.lower() in c.lower())
                        break
                
                if not metric_col and df.columns[0].dtype == 'object':
                    metric_col = df.columns[0]
                
                if metric_col:
                    df = df.set_index(metric_col)
                    df.index.name = 'Metric'
                    
                    # Track source
                    for idx in df.index:
                        sources[idx] = file.name
                    
                    all_dfs.append(df)
                    logger.info(f"Successfully processed {file.name}: {len(df)} metrics")
                    
        except Exception as e:
            logger.error(f"Error processing {file.name}: {str(e)}")
            st.warning(f"Could not process {file.name}: {str(e)}")
    
    if not all_dfs:
        return None
    
    # Merge all DataFrames
    if len(all_dfs) == 1:
        merged_df = all_dfs[0]
    else:
        merged_df = pd.concat(all_dfs, axis=0)
        merged_df = merged_df[~merged_df.index.duplicated(keep='first')]
    
    # Clean numeric data
    merged_df = DataProcessor.clean_numeric_data(merged_df)
    
    # Identify year columns
    year_columns = []
    for col in merged_df.columns:
        if str(col).isdigit() and 1900 <= int(col) <= 2100:
            year_columns.append(col)
        elif re.match(r'^\d{4}$', str(col)):
            year_columns.append(col)
    
    # Sort year columns
    year_columns = sorted(year_columns, key=lambda x: int(x))
    
    # Calculate data quality
    data_quality = asdict(DataProcessor.calculate_data_quality(merged_df))
    
    # Detect outliers
    outliers = DataProcessor.detect_outliers(merged_df)
    
    return {
        "statement": merged_df,
        "company_name": _uploaded_files[0].name.split('.')[0] if _uploaded_files else "Company",
        "data_quality": data_quality,
        "outliers": outliers,
        "year_columns": year_columns,
        "sources": sources
    }

# --- 9. Chart Generation ---
class ChartGenerator:
    @staticmethod
    def create_line_chart(data: pd.DataFrame, metrics: List[str], title: str = "") -> go.Figure:
        """Create interactive line chart"""
        fig = go.Figure()
        
        for metric in metrics:
            if metric in data.index:
                fig.add_trace(go.Scatter(
                    x=data.columns,
                    y=data.loc[metric],
                    mode='lines+markers',
                    name=metric,
                    line=dict(width=3),
                    marker=dict(size=8)
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Value",
            hovermode='x unified',
            template="plotly_white",
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_bar_chart(data: pd.DataFrame, metrics: List[str], title: str = "") -> go.Figure:
        """Create interactive bar chart"""
        fig = go.Figure()
        
        for metric in metrics:
            if metric in data.index:
                fig.add_trace(go.Bar(
                    x=data.columns,
                    y=data.loc[metric],
                    name=metric,
                    text=data.loc[metric].round(2),
                    textposition='auto'
                ))
        
        fig.update_layout(
            title=title,
            xaxis_title="Year",
            yaxis_title="Value",
            barmode='group',
            template="plotly_white",
            height=500
        )
        
        return fig
    
    @staticmethod
    def create_waterfall_chart(data: Dict[str, float], title: str = "") -> go.Figure:
        """Create waterfall chart for decomposition"""
        labels = list(data.keys())
        values = list(data.values())
        
        fig = go.Figure(go.Waterfall(
            name="",
            orientation="v",
            x=labels,
            y=values,
            connector={"line": {"color": "rgb(63, 63, 63)"}},
            decreasing={"marker": {"color": "crimson"}},
            increasing={"marker": {"color": "lightgreen"}},
            totals={"marker": {"color": "deep sky blue"}}
        ))
        
        fig.update_layout(
            title=title,
            showlegend=False,
            template="plotly_white",
            height=500
        )
        
        return fig

# --- 10. Financial Analysis Engine ---
class FinancialAnalysisEngine:
    def __init__(self, data: pd.DataFrame, mappings: Dict[str, Any]):
        self.data = data
        self.mappings = mappings
        self._prepare_metrics()
    
    def _prepare_metrics(self):
        """Prepare and validate metrics for analysis"""
        self.metrics = {}
        for key, value in self.mappings.items():
            if value:
                if isinstance(value, list):
                    # Sum multiple items
                    self.metrics[key] = self.data.loc[value].sum()
                else:
                    # Single item
                    if value in self.data.index:
                        self.metrics[key] = self.data.loc[value]
                    else:
                        self.metrics[key] = pd.Series(0, index=self.data.columns)
            else:
                self.metrics[key] = pd.Series(0, index=self.data.columns)
    
    def calculate_ratios(self) -> pd.DataFrame:
        """Calculate financial ratios"""
        ratios = pd.DataFrame()
        
        # Liquidity Ratios
        if 'Current Assets' in self.metrics and 'Current Liabilities' in self.metrics:
            ratios.loc['Current Ratio'] = self.metrics['Current Assets'] / self.metrics['Current Liabilities'].replace(0, np.nan)
            
            if 'Inventories' in self.metrics:
                quick_assets = self.metrics['Current Assets'] - self.metrics['Inventories']
                ratios.loc['Quick Ratio'] = quick_assets / self.metrics['Current Liabilities'].replace(0, np.nan)
        
        # Leverage Ratios
        if 'Total Liabilities' in self.metrics and 'Total Equity' in self.metrics:
            ratios.loc['Debt to Equity'] = self.metrics['Total Liabilities'] / self.metrics['Total Equity'].replace(0, np.nan)
        
        # Profitability Ratios
        if 'Net Income' in self.metrics:
            if 'Total Equity' in self.metrics:
                ratios.loc['ROE'] = self.metrics['Net Income'] / self.metrics['Total Equity'].replace(0, np.nan)
            
            if 'Total Assets' in self.metrics:
                ratios.loc['ROA'] = self.metrics['Net Income'] / self.metrics['Total Assets'].replace(0, np.nan)
            
            if 'Revenue' in self.metrics:
                ratios.loc['Profit Margin'] = self.metrics['Net Income'] / self.metrics['Revenue'].replace(0, np.nan)
        
        # Efficiency Ratios
        if 'Revenue' in self.metrics and 'Total Assets' in self.metrics:
            ratios.loc['Asset Turnover'] = self.metrics['Revenue'] / self.metrics['Total Assets'].replace(0, np.nan)
        
        return ratios
    
    def perform_trend_analysis(self) -> Dict[str, Dict[str, float]]:
        """Perform trend analysis on key metrics"""
        trends = {}
        
        for metric_name, metric_data in self.metrics.items():
            if metric_name in ['Revenue', 'Net Income', 'Total Assets', 'Total Equity']:
                valid_data = metric_data.dropna()
                if len(valid_data) >= 2:
                    years = np.arange(len(valid_data))
                    slope, _, _, _, _ = stats.linregress(years, valid_data.values)
                    
                    cagr = ((valid_data.iloc[-1] / valid_data.iloc[0]) ** (1 / len(valid_data))) - 1
                    
                    trends[metric_name] = {
                        'CAGR': cagr * 100,
                        'Trend': 'Increasing' if slope > 0 else 'Decreasing',
                        'Average': valid_data.mean(),
                        'Volatility': valid_data.std() / valid_data.mean() if valid_data.mean() != 0 else 0
                    }
        
        return trends
    
    def generate_insights(self) -> List[str]:
        """Generate automated insights"""
        insights = []
        
        # Profitability insights
        if 'ROE' in self.calculate_ratios().index:
            roe = self.calculate_ratios().loc['ROE'].mean()
            if roe > 0.15:
                insights.append(f"💰 Strong ROE of {roe:.1%} indicates effective equity utilization")
            elif roe < 0.08:
                insights.append(f"⚠️ Low ROE of {roe:.1%} suggests potential profitability concerns")
        
        # Growth insights
        trends = self.perform_trend_analysis()
        if 'Revenue' in trends:
            cagr = trends['Revenue']['CAGR']
            if cagr > 10:
                insights.append(f"📈 Revenue growing at {cagr:.1f}% CAGR - strong growth trajectory")
            elif cagr < 0:
                insights.append(f"📉 Revenue declining at {abs(cagr):.1f}% annually - requires attention")
        
        # Liquidity insights
        if 'Current Ratio' in self.calculate_ratios().index:
            current_ratio = self.calculate_ratios().loc['Current Ratio'].mean()
            if current_ratio < 1:
                insights.append(f"⚠️ Current ratio of {current_ratio:.2f} indicates potential liquidity risk")
            elif current_ratio > 3:
                insights.append(f"💎 Strong liquidity with current ratio of {current_ratio:.2f}")
        
        # Leverage insights
        if 'Debt to Equity' in self.calculate_ratios().index:
            de_ratio = self.calculate_ratios().loc['Debt to Equity'].mean()
            if de_ratio > 2:
                insights.append(f"⚡ High leverage with D/E ratio of {de_ratio:.2f} - monitor debt levels")
            elif de_ratio < 0.5:
                insights.append(f"🛡️ Conservative capital structure with D/E ratio of {de_ratio:.2f}")
        
        return insights

# --- 11. Penman-Nissim Implementation ---
class PenmanNissimCalculator:
    def __init__(self, metrics: Dict[str, pd.Series]):
        self.metrics = metrics
        self.results = {}
    
    def calculate(self) -> PenmanNissimResults:
        """Perform complete Penman-Nissim analysis"""
        warnings = []
        
        try:
            # Calculate Operating Assets and Liabilities
            operating_assets = (
                self.metrics.get('Total Assets', pd.Series(0)) - 
                self.metrics.get('Cash and Cash Equivalents', pd.Series(0)) -
                sum(self.metrics.get('Financial Assets', []))
            )
            
            operating_liabilities = (
                self.metrics.get('Total Liabilities', pd.Series(0)) -
                sum(self.metrics.get('Financial Liabilities', []))
            )
            
            # Net Operating Assets (NOA)
            noa = operating_assets - operating_liabilities
            
            # Net Financial Debt (NFD)
            financial_liabilities = sum(self.metrics.get('Financial Liabilities', []))
            financial_assets = (
                self.metrics.get('Cash and Cash Equivalents', pd.Series(0)) +
                sum(self.metrics.get('Financial Assets', []))
            )
            nfd = financial_liabilities - financial_assets
            
            # Check balance sheet equation
            equity = self.metrics.get('Total Equity', pd.Series(0))
            balance_check = abs((noa - nfd) - equity).mean()
            if balance_check > equity.mean() * 0.05:
                warnings.append(f"Balance sheet equation check failed. Difference: {balance_check:.2f}")
            
            # Operating Income calculations
            oi = self.metrics.get('Operating Income', pd.Series(0))
            if oi.sum() == 0:
                # Try to derive from Net Income
                ni = self.metrics.get('Net Income', pd.Series(0))
                nfe = self.metrics.get('Net Financial Expense', pd.Series(0))
                oi = ni + nfe
                warnings.append("Operating Income derived from Net Income + NFE")
            
            # RNOA (Return on Net Operating Assets)
            noa_avg = (noa + noa.shift(1)) / 2
            rnoa = oi / noa_avg.replace(0, np.nan)
            
            # Financial Leverage
            financial_leverage = nfd / equity.replace(0, np.nan)
            
            # Operating Liability Leverage
            operating_liability_leverage = operating_liabilities / noa.replace(0, np.nan)
            
            # NBC (Net Borrowing Cost)
            nfe = self.metrics.get('Net Financial Expense', pd.Series(0))
            nfd_avg = (nfd + nfd.shift(1)) / 2
            nbc = nfe / nfd_avg.replace(0, np.nan)
            
            # Spread
            spread = rnoa - nbc
            
            # ROCE decomposition
            roce = rnoa + (financial_leverage * spread)
            
            # Operating profitability decomposition
            revenue = self.metrics.get('Revenue', pd.Series(0))
            pm = oi / revenue.replace(0, np.nan)  # Profit Margin
            ato = revenue / noa_avg.replace(0, np.nan)  # Asset Turnover
            
            # Create decomposition DataFrame
            decomposition_data = {
                'ROCE': roce,
                'RNOA': rnoa,
                'Financial Leverage': financial_leverage,
                'Spread': spread,
                'NBC': nbc,
                'Profit Margin': pm,
                'Asset Turnover': ato,
                'Operating Liability Leverage': operating_liability_leverage
            }
            
            decomposition_df = pd.DataFrame(decomposition_data)
            
            # Calculate quality score
            quality_score = self._calculate_quality_score(decomposition_df, warnings)
            
            return PenmanNissimResults(
                roce=roce,
                rnoa=rnoa,
                financial_leverage=financial_leverage,
                operating_liability_leverage=operating_liability_leverage,
                spread=spread,
                nbc=nbc,
                pm=pm,
                ato=ato,
                decomposition_df=decomposition_df,
                quality_score=quality_score,
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Error in Penman-Nissim calculation: {str(e)}")
            warnings.append(f"Calculation error: {str(e)}")
            
            # Return empty results
            empty_series = pd.Series()
            return PenmanNissimResults(
                roce=empty_series,
                rnoa=empty_series,
                financial_leverage=empty_series,
                operating_liability_leverage=empty_series,
                spread=empty_series,
                nbc=empty_series,
                pm=empty_series,
                ato=empty_series,
                decomposition_df=pd.DataFrame(),
                quality_score=0.0,
                warnings=warnings
            )
    
    def _calculate_quality_score(self, decomposition_df: pd.DataFrame, warnings: List[str]) -> float:
        """Calculate quality score for P-N analysis"""
        score = 100.0
        
        # Penalize for warnings
        score -= len(warnings) * 10
        
        # Check for data completeness
        missing_pct = decomposition_df.isna().sum().sum() / decomposition_df.size
        score -= missing_pct * 50
        
        # Check for reasonable values
        if not decomposition_df.empty:
            # RNOA should typically be positive and reasonable
            rnoa_mean = decomposition_df['RNOA'].mean()
            if rnoa_mean < -0.5 or rnoa_mean > 1.0:
                score -= 20
            
            # Financial leverage should be reasonable
            fl_mean = decomposition_df['Financial Leverage'].mean()
            if fl_mean < -2 or fl_mean > 10:
                score -= 20
        
        return max(0, min(100, score))

# --- 12. Main Dashboard Application ---
class DashboardApp:
    def __init__(self):
        self._initialize_state()
        self.chart_builders = {
            "Line": ChartGenerator.create_line_chart, 
            "Bar": ChartGenerator.create_bar_chart
        }

    def _initialize_state(self):
        """Initialize session state with defaults"""
        defaults = {
            "analysis_data": None, 
            "metric_mapping": {}, 
            "pn_results": None, 
            "pn_mappings": {},
            "selected_industry": "Technology",
            "input_mode": "file",  # 'file' or 'text'
            "show_text_input": False
        }
        for k, v in defaults.items():
            if k not in st.session_state:
                st.session_state[k] = v

    def _handle_file_upload(self):
        """Handle file upload and processing"""
        files = st.session_state.get("file_uploader_key", [])
        if files:
            st.session_state.analysis_data = process_and_merge_files(_uploaded_files=files)
            st.session_state.metric_mapping = {}
            st.session_state.pn_results = None
            st.session_state.pn_mappings = {}

    def run(self):
        """Main application entry point"""
        self.render_sidebar()
        self.render_main_panel()

    def render_sidebar(self):
        """Render sidebar with file upload and settings"""
        st.sidebar.title("📂 Data Input")
        
        # Input mode selector
        input_mode = st.sidebar.radio(
            "Choose input method:",
            ["Upload Files", "Paste Text"],
            key="input_mode_selector"
        )
        st.session_state.input_mode = input_mode
        
        if input_mode == "Upload Files":
            st.sidebar.info("Upload financial statements (CSV, HTML, XLSX). Multiple files will be merged automatically.")
            
            st.sidebar.file_uploader(
                "Upload Financial Files",
                type=ALLOWED_FILE_TYPES,
                accept_multiple_files=True,
                key="file_uploader_key",
                on_change=self._handle_file_upload,
                help="Upload one or more financial statement files"
            )
        else:
            # Text input interface
            st.sidebar.info("Paste Ind-AS formatted financial statements directly")
            if st.sidebar.button("📋 Open Text Input Panel", type="primary"):
                st.session_state.show_text_input = True
            
            # Sample data button
            if st.sidebar.button("📄 Load Sample Data"):
                st.session_state.show_sample_data = True
        
        if st.sidebar.button("🔄 Reset All", help="Clear all data and start fresh"):
            st.session_state.clear()
            st.rerun()
        
        st.sidebar.divider()
        st.sidebar.title("⚙️ Display Settings")
        st.sidebar.checkbox("Show Data Quality Indicators", key="show_data_quality", value=True)
        
        if st.session_state.analysis_data:
            self._render_general_metric_mapper()
            self._render_industry_selection()

    def _render_general_metric_mapper(self):
        """Render metric mapping interface in sidebar"""
        st.sidebar.divider()
        st.sidebar.subheader("📊 Metric Mapping")
        
        df = st.session_state.analysis_data["statement"]
        available_metrics = df.index.tolist()
        
        common_mappings = {
            "Revenue": ["Revenue", "Sales", "Total Revenue", "Net Sales"],
            "Net Income": ["Net Income", "Net Profit", "Profit After Tax", "PAT"],
            "Total Assets": ["Total Assets", "Assets"],
            "Total Equity": ["Total Equity", "Shareholders Equity", "Net Worth"]
        }
        
        for target, suggestions in common_mappings.items():
            default = next((m for m in available_metrics if any(s.lower() in m.lower() for s in suggestions)), None)
            
            st.session_state.metric_mapping[target] = st.sidebar.selectbox(
                f"Map '{target}':",
                ["None"] + available_metrics,
                index=available_metrics.index(default) + 1 if default else 0,
                key=f"map_{target}"
            )

    def _render_industry_selection(self):
        """Render industry selection in sidebar"""
        st.sidebar.divider()
        st.sidebar.subheader("🏭 Industry Selection")
        st.session_state.selected_industry = st.sidebar.selectbox(
            "Select Industry for Benchmarking:",
            list(INDUSTRY_BENCHMARKS.keys()),
            key="industry_selector"
        )

    def render_main_panel(self):
        """Render main dashboard panel"""
        st.markdown("<div class='main-header'>💹 Elite Financial Analytics Platform</div>", unsafe_allow_html=True)
        
        # Show text input panel if requested
        if st.session_state.get('show_text_input', False):
            self._render_text_input_panel()
            st.session_state.show_text_input = False
            return
        
        # Show sample data if requested
        if st.session_state.get('show_sample_data', False):
            self._show_sample_data()
            st.session_state.show_sample_data = False
            return
        
        if not st.session_state.analysis_data:
            self._render_welcome_screen()
            return
        
        # Main dashboard tabs
        tabs = st.tabs([
            "📊 Visualizations", 
            "📋 Data Table", 
            "💡 Financial Analysis", 
            "🔍 Penman-Nissim Analysis",
            "📈 Comparative Analysis"
        ])
        
        with tabs[0]:
            self._render_visualization_tab(
                st.session_state.analysis_data["statement"], 
                st.session_state.analysis_data
            )
        
        with tabs[1]:
            self._render_data_table_tab(
                st.session_state.analysis_data["statement"], 
                st.session_state.analysis_data
            )
        
        with tabs[2]:
            self._render_financial_analysis_tab(
                st.session_state.analysis_data["statement"], 
                st.session_state.analysis_data
            )
        
        with tabs[3]:
            self._render_penman_nissim_tab(
                st.session_state.analysis_data["statement"], 
                st.session_state.analysis_data
            )
        
        with tabs[4]:
            self._render_comparative_analysis_tab(
                st.session_state.analysis_data["statement"], 
                st.session_state.analysis_data
            )

    def _render_text_input_panel(self):
        """Render text input panel for Ind-AS statements"""
        st.header("📋 Paste Financial Statements")
        st.info("Paste your Ind-AS formatted Balance Sheet and P&L statements below")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Balance Sheet")
            bs_text = st.text_area(
                "Paste Balance Sheet text:",
                height=400,
                placeholder="ASSETS:\nNon-Current Assets:\nProperty, Plant and Equipment    1234.56    2345.67...\n\nCurrent Assets:\n..."
            )
        
        with col2:
            st.subheader("Profit & Loss Statement")
            pl_text = st.text_area(
                "Paste P&L text:",
                height=400,
                placeholder="REVENUE:\nRevenue From Operations    5678.90    6789.01...\n\nEXPENSES:\n..."
            )
        
        company_name = st.text_input("Company Name", value="Company", placeholder="e.g., VST Industries Ltd")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🔄 Process Statements", type="primary", use_container_width=True):
                # Validate inputs
                bs_valid, bs_msg = IndASParser.validate_ind_as_text(bs_text) if bs_text.strip() else (True, "")
                pl_valid, pl_msg = IndASParser.validate_ind_as_text(pl_text) if pl_text.strip() else (True, "")
                
                if not bs_text.strip() and not pl_text.strip():
                    st.error("❌ Please paste at least one financial statement")
                    return
                
                if bs_text.strip() and not bs_valid:
                    st.error(f"❌ Balance Sheet: {bs_msg}")
                    return
                    
                if pl_text.strip() and not pl_valid:
                    st.error(f"❌ P&L Statement: {pl_msg}")
                    return
                
                # Parse the text inputs
                with st.spinner("Processing pasted statements..."):
                    text_data = parse_ind_as_text_input(bs_text, pl_text, company_name)
                    
                    if text_data:
                        st.session_state.analysis_data = text_data
                        st.session_state.metric_mapping = {}
                        st.session_state.pn_results = None
                        st.session_state.pn_mappings = {}
                        
                        # Auto-apply AI mapping
                        if st.checkbox("🤖 Auto-map with AI", value=True):
                            with st.spinner("Applying intelligent mapping..."):
                                mapper = IntelligentFinancialMapper(accounting_standard='IND-AS')
                                available_metrics = text_data["statement"].index.tolist()
                                ai_mappings = mapper.map_metrics(available_metrics)
                                
                                # Update P-N mappings
                                st.session_state.pn_mappings = ai_mappings
                                
                                # Display mapping quality
                                display_mapping_quality(ai_mappings)
                                
                                # Display key mappings
                                with st.expander("View AI Mappings", expanded=True):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**Balance Sheet Mappings:**")
                                        for key in ['Total Assets', 'Total Liabilities', 'Total Equity', 
                                                   'Current Assets', 'Current Liabilities']:
                                            value = ai_mappings.get(key)
                                            if value:
                                                st.write(f"✓ {key}: *{value}*")
                                            else:
                                                st.write(f"❌ {key}: Not found")
                                    
                                    with col2:
                                        st.markdown("**P&L Mappings:**")
                                        for key in ['Revenue', 'Operating Income', 'Net Income', 
                                                   'Net Financial Expense']:
                                            value = ai_mappings.get(key)
                                            if value:
                                                st.write(f"✓ {key}: *{value}*")
                                            else:
                                                st.write(f"❌ {key}: Not found")
                                
                                # Export mappings option
                                if st.button("💾 Export Mappings"):
                                    json_str = export_mappings(ai_mappings)
                                    st.download_button(
                                        label="Download Mappings JSON",
                                        data=json_str,
                                        file_name=f"{company_name}_mappings.json",
                                        mime="application/json"
                                    )
                        
                        st.success("✅ Data processed successfully!")
                        st.info("📊 Click on any tab above to start analysis.")
                        
                        # Quick navigation buttons
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            if st.button("📊 Go to Visualizations"):
                                st.session_state.active_tab = 0
                                st.rerun()
                        with col2:
                            if st.button("💡 Go to Financial Analysis"):
                                st.session_state.active_tab = 2
                                st.rerun()
                        with col3:
                            if st.button("🔍 Go to P-N Analysis"):
                                st.session_state.active_tab = 3
                                st.rerun()
                        with col4:
                            if st.button("📋 View Data Table"):
                                st.session_state.active_tab = 1
                                st.rerun()
                    else:
                        st.error("Failed to process the pasted text. Please check the format.")

    def _show_sample_data(self):
        """Show sample data and load it"""
        st.header("📄 Sample Ind-AS Data")
        st.info("Below is sample data in Ind-AS format. You can copy and use it for testing.")
        
        sample_bs, sample_pl = load_sample_ind_as_data()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Sample Balance Sheet")
            st.text_area("Balance Sheet", value=sample_bs, height=400, key="sample_bs_display")
        
        with col2:
            st.subheader("Sample P&L Statement")
            st.text_area("P&L Statement", value=sample_pl, height=400, key="sample_pl_display")
        
        if st.button("📥 Load Sample Data", type="primary"):
            with st.spinner("Loading sample data..."):
                text_data = parse_ind_as_text_input(sample_bs, sample_pl, "Sample Company")
                
                if text_data:
                    st.session_state.analysis_data = text_data
                    st.session_state.metric_mapping = {}
                    st.session_state.pn_results = None
                    st.session_state.pn_mappings = {}
                    
                    # Auto-apply mappings
                    mapper = IntelligentFinancialMapper(accounting_standard='IND-AS')
                    available_metrics = text_data["statement"].index.tolist()
                    ai_mappings = mapper.map_metrics(available_metrics)
                    st.session_state.pn_mappings = ai_mappings
                    
                    st.success("✅ Sample data loaded successfully!")
                    st.rerun()

    def _render_welcome_screen(self):
        """Render welcome screen when no data is loaded"""
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h2>Welcome to Elite Financial Analytics</h2>
            <p style='font-size: 1.2rem; color: #666;'>
                Advanced financial statement analysis with PhD-level Penman-Nissim decomposition
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class='metric-card'>
                <h3>📊 Smart Visualizations</h3>
                <p>Interactive charts with drill-down capabilities and trend analysis</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class='metric-card'>
                <h3>🧮 Advanced Analytics</h3>
                <p>Penman-Nissim decomposition, ratio analysis, and peer comparison</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class='metric-card'>
                <h3>🤖 AI-Powered Insights</h3>
                <p>Automated metric mapping and intelligent financial insights</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Getting Started Guide
        with st.expander("🚀 Getting Started", expanded=True):
            st.markdown("""
            ### Choose your input method:
            
            **Option 1: Upload Files**
            1. Select "Upload Files" in the sidebar
            2. Upload your financial statements (CSV, XLSX, HTML)
            3. Multiple files will be automatically merged
            
            **Option 2: Paste Text** (NEW!)
            1. Select "Paste Text" in the sidebar
            2. Click "Open Text Input Panel"
            3. Paste your Ind-AS formatted statements
            4. Enable AI auto-mapping for best results
            
            **Try Sample Data:**
            - Click "Load Sample Data" in the sidebar to explore features
            """)

    def _render_visualization_tab(self, df: pd.DataFrame, data: Dict[str, Any]):
        """Render visualization tab"""
        st.header("📊 Interactive Visualizations")
        
        # Data quality indicator
        if st.session_state.show_data_quality:
            quality = data["data_quality"]
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Completeness", f"{quality['completeness']*100:.1f}%")
            with col2:
                st.metric("Data Points", quality['data_points'])
            with col3:
                st.metric("Time Span", f"{quality['time_span_years']} years")
            with col4:
                st.metric("Outliers", f"{quality['outlier_percentage']*100:.1f}%")
        
        # Metric selection
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_metrics = st.multiselect(
                "Select metrics to visualize:",
                df.index.tolist(),
                default=df.index[:3].tolist() if len(df) >= 3 else df.index.tolist()
            )
        
        with col2:
            chart_type = st.selectbox("Chart Type:", ["Line", "Bar"])
        
        if selected_metrics:
            # Create chart
            chart = self.chart_builders[chart_type](
                df, 
                selected_metrics,
                f"{chart_type} Chart - {', '.join(selected_metrics[:3])}"
            )
            st.plotly_chart(chart, use_container_width=True)
            
            # Show statistics
            with st.expander("📊 Statistical Summary"):
                stats_df = df.loc[selected_metrics].T.describe()
                st.dataframe(stats_df.style.format("{:.2f}"))

    def _render_data_table_tab(self, df: pd.DataFrame, data: Dict[str, Any]):
        """Render data table tab"""
        st.header("📋 Financial Data Table")
        
        # Search and filter
        col1, col2 = st.columns([2, 1])
        with col1:
            search_term = st.text_input("🔍 Search metrics:", "")
        with col2:
            show_all = st.checkbox("Show all metrics", value=True)
        
        # Filter data
        if search_term:
            filtered_df = df[df.index.str.contains(search_term, case=False)]
        else:
            filtered_df = df
        
        if not show_all and len(filtered_df) > 20:
            filtered_df = filtered_df.head(20)
            st.info("Showing top 20 metrics. Check 'Show all metrics' to see more.")
        
        # Display table with formatting
        st.dataframe(
            filtered_df.style.format("{:,.2f}").highlight_max(axis=1, color='lightgreen'),
            use_container_width=True
        )
        
        # Export options
        col1, col2 = st.columns(2)
        with col1:
            csv = filtered_df.to_csv()
            st.download_button(
                "📥 Download as CSV",
                csv,
                "financial_data.csv",
                "text/csv"
            )

    def _render_financial_analysis_tab(self, df: pd.DataFrame, data: Dict[str, Any]):
        """Render financial analysis tab"""
        st.header("💡 Comprehensive Financial Analysis")
        
        # Prepare mappings
        mappings = {}
        for key, value in st.session_state.metric_mapping.items():
            if value and value != "None":
                mappings[key] = value
        
        # Add P-N mappings if available
        if st.session_state.pn_mappings:
            mappings.update(st.session_state.pn_mappings)
        
        if not mappings:
            st.warning("Please map metrics in the sidebar to enable analysis")
            return
        
        # Initialize analysis engine
        engine = FinancialAnalysisEngine(df, mappings)
        
        # Ratio Analysis
        st.subheader("📊 Financial Ratios")
        ratios = engine.calculate_ratios()
        
        if not ratios.empty:
            # Display ratios with industry comparison
            industry = st.session_state.selected_industry
            benchmarks = INDUSTRY_BENCHMARKS[industry]
            
            for ratio_name in ratios.index:
                if ratio_name in benchmarks:
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        st.write(f"**{ratio_name}**")
                    with col2:
                        current_value = ratios.loc[ratio_name].iloc[-1]
                        st.metric("Current", f"{current_value:.2f}")
                    with col3:
                        benchmark = benchmarks[ratio_name]
                        delta = current_value - benchmark
                        st.metric("vs Benchmark", f"{benchmark:.2f}", f"{delta:+.2f}")
            
            # Ratio trends chart
            st.plotly_chart(
                ChartGenerator.create_line_chart(ratios, ratios.index.tolist(), "Financial Ratios Trend"),
                use_container_width=True
            )
        
        # Trend Analysis
        st.subheader("📈 Trend Analysis")
        trends = engine.perform_trend_analysis()
        
        if trends:
            trend_cols = st.columns(len(trends))
            for i, (metric, trend_data) in enumerate(trends.items()):
                with trend_cols[i]:
                    st.metric(
                        metric,
                        f"{trend_data['CAGR']:.1f}% CAGR",
                        trend_data['Trend']
                    )
        
        # Automated Insights
        st.subheader("🤖 AI-Generated Insights")
        insights = engine.generate_insights()
        
        for insight in insights:
            if "⚠️" in insight:
                st.markdown(f'<div class="warning-box">{insight}</div>', unsafe_allow_html=True)
            elif "📈" in insight or "💰" in insight or "💎" in insight:
                st.markdown(f'<div class="success-box">{insight}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)

    def _render_penman_nissim_tab(self, df: pd.DataFrame, data: Dict[str, Any]):
        """Render Penman-Nissim analysis tab"""
        st.header("🔍 Advanced Penman-Nissim Analysis")
        
        # Add Ind-AS indicator if applicable
        if st.session_state.input_mode == "text":
            st.info("📊 Using Ind-AS standard mappings for analysis")
        
        # Introduction
        with st.expander("📚 About Penman-Nissim Analysis", expanded=False):
            st.markdown("""
            The Penman-Nissim framework decomposes Return on Common Equity (ROCE) into:
            - **Operating Performance** (RNOA - Return on Net Operating Assets)
            - **Financial Leverage Effects**
            
            Key Formula: **ROCE = RNOA + (Financial Leverage × Spread)**
            
            Where Spread = RNOA - NBC (Net Borrowing Cost)
            """)
        
        # Metric Mapping Section
        st.subheader("🎯 Metric Mapping for P-N Analysis")
        
        available_metrics = df.index.tolist()
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🤖 Auto-Suggest All", help="Use AI to suggest mappings"):
                # Auto-suggest all P-N mappings using intelligent mapper
                mapper = IntelligentFinancialMapper(
                    accounting_standard='IND-AS' if st.session_state.input_mode == "text" else 'GENERIC'
                )
                ai_mappings = mapper.map_metrics(available_metrics)
                
                # Update session state with AI mappings
                for key, value in ai_mappings.items():
                    st.session_state.pn_mappings[key] = value
                
                st.success("✅ AI mappings applied!")
                st.rerun()
        
        # Mapping interface
        with st.form("pn_mapping_form"):
            mapping_cols = st.columns(2)
            
            # Single-value mappings
            single_metrics = [m for m in PENMAN_NISSIM_METRICS if m not in ['Financial Assets', 'Financial Liabilities']]
            
            for i, metric in enumerate(single_metrics):
                col = mapping_cols[i % 2]
                with col:
                    current_value = st.session_state.pn_mappings.get(metric, None)
                    default_idx = available_metrics.index(current_value) + 1 if current_value in available_metrics else 0
                    
                    st.session_state.pn_mappings[metric] = st.selectbox(
                        f"{metric}:",
                        ["None"] + available_metrics,
                        index=default_idx,
                        key=f"pn_map_{metric}"
                    )
            
            # Multi-value mappings
            st.markdown("**Financial Items (Select Multiple):**")
            
            col1, col2 = st.columns(2)
            with col1:
                financial_assets = st.multiselect(
                    "Financial Assets:",
                    available_metrics,
                    default=st.session_state.pn_mappings.get('Financial Assets', []),
                    key="pn_financial_assets"
                )
                st.session_state.pn_mappings['Financial Assets'] = financial_assets
            
            with col2:
                financial_liabilities = st.multiselect(
                    "Financial Liabilities:",
                    available_metrics,
                    default=st.session_state.pn_mappings.get('Financial Liabilities', []),
                    key="pn_financial_liabilities"
                )
                st.session_state.pn_mappings['Financial Liabilities'] = financial_liabilities
            
            submitted = st.form_submit_button("🚀 Run P-N Analysis", type="primary")
        
        # Display mapping quality
        if st.session_state.pn_mappings:
            display_mapping_quality(st.session_state.pn_mappings)
        
        # Run analysis
        if submitted or st.session_state.pn_results:
            # Prepare metrics
            pn_metrics = {}
            for key, value in st.session_state.pn_mappings.items():
                if value and value != "None":
                    if isinstance(value, list):
                        # For Financial Assets/Liabilities
                        series_list = []
                        for item in value:
                            if item in df.index:
                                series_list.append(df.loc[item])
                        pn_metrics[key] = series_list
                    else:
                        # Single metrics
                        if value in df.index:
                            pn_metrics[key] = df.loc[value]
            
            # Calculate P-N
            calculator = PenmanNissimCalculator(pn_metrics)
            results = calculator.calculate()
            st.session_state.pn_results = results
            
            # Display results
            st.markdown("---")
            st.subheader("📊 Penman-Nissim Results")
            
            # Quality score
            quality_color = "green" if results.quality_score > 80 else "orange" if results.quality_score > 50 else "red"
            st.markdown(f"**Analysis Quality Score:** <span style='color: {quality_color}; font-size: 1.5em;'>{results.quality_score:.0f}/100</span>", unsafe_allow_html=True)
            
            # Warnings
            if results.warnings:
                for warning in results.warnings:
                    st.warning(warning)
            
            # Key metrics
            if not results.decomposition_df.empty:
                latest_year = results.decomposition_df.columns[-1]
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    roce = results.roce.iloc[-1] if len(results.roce) > 0 else 0
                    st.metric("ROCE", f"{roce:.2%}")
                with col2:
                    rnoa = results.rnoa.iloc[-1] if len(results.rnoa) > 0 else 0
                    st.metric("RNOA", f"{rnoa:.2%}")
                with col3:
                    fl = results.financial_leverage.iloc[-1] if len(results.financial_leverage) > 0 else 0
                    st.metric("Financial Leverage", f"{fl:.2f}")
                with col4:
                    spread = results.spread.iloc[-1] if len(results.spread) > 0 else 0
                    st.metric("Spread", f"{spread:.2%}")
                
                # Decomposition waterfall
                st.subheader("🌊 ROCE Decomposition Waterfall")
                
                waterfall_data = {
                    "RNOA": results.rnoa.iloc[-1] if len(results.rnoa) > 0 else 0,
                    "Leverage Effect": (results.financial_leverage.iloc[-1] * results.spread.iloc[-1]) if len(results.financial_leverage) > 0 else 0,
                    "ROCE": results.roce.iloc[-1] if len(results.roce) > 0 else 0
                }
                
                waterfall_chart = ChartGenerator.create_waterfall_chart(
                    waterfall_data,
                    "ROCE Decomposition"
                )
                st.plotly_chart(waterfall_chart, use_container_width=True)
                
                # Trend analysis
                st.subheader("📈 P-N Metrics Trends")
                
                trend_metrics = ['ROCE', 'RNOA', 'Financial Leverage', 'NBC', 'Spread']
                selected_trends = st.multiselect(
                    "Select metrics to plot:",
                    trend_metrics,
                    default=['ROCE', 'RNOA']
                )
                
                if selected_trends:
                    trend_chart = ChartGenerator.create_line_chart(
                        results.decomposition_df,
                        selected_trends,
                        "Penman-Nissim Metrics Over Time"
                    )
                    st.plotly_chart(trend_chart, use_container_width=True)
                
                # Detailed decomposition table
                with st.expander("📋 Detailed Decomposition Table"):
                    st.dataframe(
                        results.decomposition_df.style.format("{:.2%}"),
                        use_container_width=True
                    )
                
                # Export results
                col1, col2 = st.columns(2)
                with col1:
                    csv = results.decomposition_df.to_csv()
                    st.download_button(
                        "📥 Download P-N Results",
                        csv,
                        "penman_nissim_results.csv",
                        "text/csv"
                    )
                with col2:
                    if st.button("💾 Export Mappings"):
                        json_str = export_mappings(st.session_state.pn_mappings)
                        st.download_button(
                            label="Download Mappings JSON",
                            data=json_str,
                            file_name="pn_mappings.json",
                            mime="application/json"
                        )

    def _render_comparative_analysis_tab(self, df: pd.DataFrame, data: Dict[str, Any]):
        """Render comparative analysis tab"""
        st.header("📈 Comparative Analysis")
        
        # Peer comparison setup
        st.subheader("🏢 Peer Comparison")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            peer_ticker = st.text_input("Enter peer company ticker (e.g., AAPL):")
        with col2:
            if st.button("Fetch Peer Data"):
                if peer_ticker:
                    with st.spinner(f"Fetching data for {peer_ticker}..."):
                        try:
                            peer = yf.Ticker(peer_ticker)
                            peer_info = peer.info
                            
                            st.success(f"Loaded data for {peer_info.get('longName', peer_ticker)}")
                            
                            # Display peer metrics
                            peer_metrics = {
                                "Market Cap": peer_info.get('marketCap', 0),
                                "Revenue": peer_info.get('totalRevenue', 0),
                                "Net Income": peer_info.get('netIncomeToCommon', 0),
                                "Total Assets": peer_info.get('totalAssets', 0),
                                "Total Debt": peer_info.get('totalDebt', 0)
                            }
                            
                            # Create comparison
                            st.subheader("📊 Key Metrics Comparison")
                            
                            comparison_data = []
                            for metric, peer_value in peer_metrics.items():
                                our_value = 0
                                if metric in st.session_state.metric_mapping:
                                    mapped = st.session_state.metric_mapping[metric]
                                    if mapped and mapped != "None" and mapped in df.index:
                                        our_value = df.loc[mapped].iloc[-1]
                                
                                comparison_data.append({
                                    "Metric": metric,
                                    "Our Company": our_value,
                                    peer_info.get('symbol', 'Peer'): peer_value
                                })
                            
                            comparison_df = pd.DataFrame(comparison_data)
                            
                            # Visualization
                            fig = go.Figure()
                            
                            fig.add_trace(go.Bar(
                                name="Our Company",
                                x=comparison_df["Metric"],
                                y=comparison_df["Our Company"],
                                text=[f"{v:,.0f}" for v in comparison_df["Our Company"]],
                                textposition='auto'
                            ))
                            
                            fig.add_trace(go.Bar(
                                name=peer_info.get('symbol', 'Peer'),
                                x=comparison_df["Metric"],
                                y=comparison_df[peer_info.get('symbol', 'Peer')],
                                text=[f"{v:,.0f}" for v in comparison_df[peer_info.get('symbol', 'Peer')]],
                                textposition='auto'
                            ))
                            
                            fig.update_layout(
                                title="Company vs Peer Comparison",
                                barmode='group',
                                template="plotly_white",
                                height=500
                            )
                            
                            st.plotly_chart(fig, use_container_width=True)
                            
                        except Exception as e:
                            st.error(f"Error fetching peer data: {str(e)}")
        
        # Time period comparison
        st.subheader("📅 Time Period Analysis")
        
        if len(data["year_columns"]) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                period1 = st.selectbox("Select first period:", data["year_columns"])
            with col2:
                period2 = st.selectbox("Select second period:", data["year_columns"], index=len(data["year_columns"])-1)
            
            if period1 != period2:
                # Calculate changes
                metrics_to_compare = st.multiselect(
                    "Select metrics to compare:",
                    df.index.tolist(),
                    default=df.index[:5].tolist() if len(df) >= 5 else df.index.tolist()
                )
                
                if metrics_to_compare:
                    comparison_data = []
                    for metric in metrics_to_compare:
                        val1 = df.loc[metric, period1]
                        val2 = df.loc[metric, period2]
                        change = val2 - val1
                        pct_change = (change / val1 * 100) if val1 != 0 else 0
                        
                        comparison_data.append({
                            "Metric": metric,
                            period1: val1,
                            period2: val2,
                            "Change": change,
                            "% Change": pct_change
                        })
                    
                    comp_df = pd.DataFrame(comparison_data)
                    
                    # Display comparison
                    st.dataframe(
                        comp_df.style.format({
                            period1: "{:,.2f}",
                            period2: "{:,.2f}",
                            "Change": "{:,.2f}",
                            "% Change": "{:.1f}%"
                        }).background_gradient(subset=["% Change"], cmap="RdYlGn"),
                        use_container_width=True
                    )
                    
                    # Visualization
                    fig = go.Figure()
                    
                    fig.add_trace(go.Bar(
                        name="% Change",
                        x=comp_df["Metric"],
                        y=comp_df["% Change"],
                        text=[f"{v:.1f}%" for v in comp_df["% Change"]],
                        textposition='auto',
                        marker_color=['green' if x > 0 else 'red' for x in comp_df["% Change"]]
                    ))
                    
                    fig.update_layout(
                        title=f"Percentage Change: {period1} to {period2}",
                        xaxis_title="Metrics",
                        yaxis_title="% Change",
                        template="plotly_white",
                        height=500
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)

# --- 13. Main Execution ---
def main():
    """Main application entry point"""
    app = DashboardApp()
    app.run()

if __name__ == "__main__":
    main()
