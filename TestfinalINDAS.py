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
                
