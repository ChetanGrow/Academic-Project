import pandas as pd
from sqlalchemy import create_engine, text
from datetime import datetime
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Database connection string
DB_STRING = "postgresql://postgres:root@192.168.120.41:5432/Lab"
ENGINE = create_engine(DB_STRING)

# Define numeric columns for FBC data (lowercase to match query output)
NUMERIC_COLS = [
    "mcv", "wbc", "param_11", "mch", "plt", "basophil", "monocyte",
    "eosinophil", "lymphocyte", "rbc", "neutrophil", "hgb", "mchc"
]

# Units for FBC parameters (lowercase keys) - only keeping units since ranges come from query
PARAM_DETAILS = {
    "hgb": {"unit": "g/dL"},
    "plt": {"unit": "K/µL"},
    "wbc": {"unit": "K/µL"},
    "rbc": {"unit": "M/µL"},
    "mcv": {"unit": "fL"},
    "mch": {"unit": "pg"},
    "mchc": {"unit": "g/dL"},
    "basophil": {"unit": "K/µL"},
    "monocyte": {"unit": "K/µL"},
    "eosinophil": {"unit": "K/µL"},
    "lymphocyte": {"unit": "K/µL"},
    "neutrophil": {"unit": "K/µL"}
}

# SQL query template (unchanged)
BASE_QUERY = """
    SELECT 
        tp.mrn,
        MAX(sampledtls.created_date) AS created_date,
        MAX(labresult.test_name) AS test_name,
        MAX(paramdtls.parameter_name) AS parameter_name,
        MAX(paramdtls.normal_range) AS normal_range,
        MAX(paramdtls.critical_range) AS critical_range,
        MAX(paramdtls.final_result) AS final_result,
        MAX(CASE WHEN paramdtls.parameter_name = 'MCV' THEN paramdtls.final_result END) AS mcv,
        MAX(CASE WHEN paramdtls.parameter_name = 'WBC' THEN paramdtls.final_result END) AS wbc,
        MAX(CASE WHEN paramdtls.parameter_name = 'Param_11' THEN paramdtls.final_result END) AS param_11,
        MAX(CASE WHEN paramdtls.parameter_name = 'MCH' THEN paramdtls.final_result END) AS mch,
        MAX(CASE WHEN paramdtls.parameter_name = 'PLT' THEN paramdtls.final_result END) AS plt,
        MAX(CASE WHEN paramdtls.parameter_name = 'BASOPHIL' THEN paramdtls.final_result END) AS basophil,
        MAX(CASE WHEN paramdtls.parameter_name = 'MONOCYTE' THEN paramdtls.final_result END) AS monocyte,
        MAX(CASE WHEN paramdtls.parameter_name = 'EOSINOPHIL' THEN paramdtls.final_result END) AS eosinophil,
        MAX(CASE WHEN paramdtls.parameter_name = 'LYMPHOCYTE' THEN paramdtls.final_result END) AS lymphocyte,
        MAX(CASE WHEN paramdtls.parameter_name = 'RBC' THEN paramdtls.final_result END) AS rbc,
        MAX(CASE WHEN paramdtls.parameter_name = 'NEUTROPHIL' THEN paramdtls.final_result END) AS neutrophil,
        MAX(CASE WHEN paramdtls.parameter_name = 'HGB' THEN paramdtls.final_result END) AS hgb,
        MAX(CASE WHEN paramdtls.parameter_name = 'MCHC' THEN paramdtls.final_result END) AS mchc
    FROM 
        "transaction"."t_lab_result" labresult
    INNER JOIN "transaction"."t_lab_result_param_dtls" paramdtls 
        ON paramdtls.lab_result_id = labresult.id
    INNER JOIN "transaction"."t_sample_dtls" sampledtls 
        ON sampledtls.id = labresult.sample_dtls_id
    INNER JOIN "master"."m_status" ms 
        ON sampledtls.screen_status_id = ms.id
    INNER JOIN "transaction"."t_sample_mst" tsm 
        ON tsm.id = sampledtls.sample_mst_id
    INNER JOIN "public"."t_patient" tp 
        ON tp.id = tsm.patient_id
    WHERE 
        ms.id IN (53, 13)
        AND labresult.test_name LIKE '%Blood Count%'
        {condition}
    GROUP BY 
        tp.mrn
    ORDER BY 
        MAX(sampledtls.created_date) DESC;
"""

def fetch_mrn_data(mrn=None):
    """Fetch FBC data for a specific MRN or all MRNs if none provided."""
    try:
        if mrn:
            query = text(BASE_QUERY.replace("{condition}", "AND tp.mrn = :mrn"))
            df = pd.read_sql(query, ENGINE, params={"mrn": mrn})
        else:
            query = text(BASE_QUERY.replace("{condition}", ""))
            df = pd.read_sql(query, ENGINE)
        
        logging.info(f"Columns returned by query: {list(df.columns)}")
        logging.info(f"Sample data: {df.head().to_dict(orient='records')}")
        if df.empty:
            return None, f"No data found{' for MRN: ' + mrn if mrn else ''}"
        return df, None
    except Exception as e:
        logging.error(f"Database error: {e}")
        return None, f"Database error: {e}"

def preprocess_data(df):
    """Preprocess FBC data by converting to numeric and handling timestamps."""
    logging.info("Preprocessing data...")
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            logging.info(f"Column '{col}' values: {df[col].tolist()}")
        else:
            logging.warning(f"Column '{col}' not found in DataFrame.")
            df[col] = pd.NA
    df["created_date"] = pd.to_datetime(df["created_date"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    return df

def determine_status(value, normal_range, critical_range):
    """Determine if a value is normal, critical, or invalid."""
    if pd.isna(value) or not normal_range or not critical_range:
        return "unknown"
    
    try:
        # Convert value to float
        value = float(value)
        
        # Parse normal range
        normal_min, normal_max = map(float, normal_range.split("-"))
        
        # Parse critical range
        if critical_range.startswith("<"):
            critical_max = float(critical_range[1:])
            in_critical = value < critical_max
        elif critical_range.startswith(">"):
            critical_min = float(critical_range[1:])
            in_critical = value > critical_min
        else:
            critical_min, critical_max = map(float, critical_range.split("-"))
            in_critical = critical_min <= value <= critical_max
        
        # Determine status
        if normal_min <= value <= normal_max:
            return "normal"
        elif in_critical:
            return "critical"
        else:
            return "invalid value"
    
    except (ValueError, AttributeError):
        return value


def calculate_risk_level(status):
    """Calculate overall risk level based on top-level status."""
    if status == "critical":
        return "high"
    if status == "green":
        return "low"
    if status == "red":
        return "high"
    elif status == "invalid value":
        return "moderate"
    return "low"

def format_result(df):
    """Format the DataFrame into the desired JSON structure with 'mrn' first."""
    df = preprocess_data(df)
    results = []
    
    for _, row in df.iterrows():
        metrics = {}
        for param in NUMERIC_COLS:
            if param in df.columns and pd.notna(row[param]):
                if param in PARAM_DETAILS:
                    param_upper = param.upper()
                    metrics[param_upper] = {
                        "value": float(row[param]),
                        "unit": PARAM_DETAILS[param]["unit"]
                    }
                else:
                    logging.warning(f"Parameter '{param}' not found in PARAM_DETAILS, skipping.")
        
        # Calculate status for top-level use
        status = determine_status(
            row["final_result"],  # Using final_result for status calculation
            row["normal_range"],
            row["critical_range"]
        )
        
        # Explicitly construct the dictionary with 'mrn' first
        result = {}
        result["mrn"] = row["mrn"]  # First key
        result["created_date"] = row["created_date"]
        result["test_name"] = row["test_name"]
        result["metrics"] = metrics
        result["normal_range"] = row["normal_range"]
        result["critical_range"] = row["critical_range"]
        result["final_result"] = row["final_result"]
        result["status"] = status
        result["risk_level"] = calculate_risk_level(status)  # Pass top-level status
        
        results.append(result)
    
    return results

def get_lab_results(mrn=None):
    """Get lab results for a given MRN or all MRNs."""
    df, error = fetch_mrn_data(mrn)
    if df is None:
        return None, error
    return format_result(df), None