import pandas as pd
import pytz
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from db import get_db_engine
import pickle
import os
import json
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODEL_PATH = "ip_model.pkl"
TIMESTAMP_PATH = "last_ip_training_timestamp.pkl"

def load_previous_model():
    #Load the previously trained model if it exists.
    if os.path.exists(MODEL_PATH):
        try:
            with open(MODEL_PATH, "rb") as file:
                model = pickle.load(file)
                logger.info("Successfully loaded previous model.")
                return model
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            logger.error(f"Failed to load model from {MODEL_PATH}: {e}. Deleting corrupted file.")
            os.remove(MODEL_PATH)
            return None
    logger.info(f"No model found at {MODEL_PATH}.")
    return None

def load_last_timestamp():
    #Load the last training timestamp if it exists.
    if os.path.exists(TIMESTAMP_PATH):
        try:
            with open(TIMESTAMP_PATH, "rb") as file:
                timestamp = pickle.load(file)
                logger.info(f"Loaded last timestamp: {timestamp}")
                return timestamp
        except (pickle.UnpicklingError, EOFError, ValueError) as e:
            logger.error(f"Failed to load timestamp from {TIMESTAMP_PATH}: {e}. Deleting corrupted file.")
            os.remove(TIMESTAMP_PATH)
            return None
    logger.info(f"No timestamp found at {TIMESTAMP_PATH}.")
    return None

def save_timestamp(timestamp):
    #Save the latest timestamp to disk.
    try:
        with open(TIMESTAMP_PATH, "wb") as file:
            pickle.dump(timestamp, file)
        logger.info(f"Saved timestamp: {timestamp}")
    except Exception as e:
        logger.error(f"Failed to save timestamp to {TIMESTAMP_PATH}: {e}")

def merge_previous_and_new(previous_result, new_df, top_days, peak_hour_ranges, top_doctors, top_departments):
    #Merge previous model results with new data.
    if not previous_result:
        return top_days, peak_hour_ranges, top_doctors, top_departments

    # Merge busiest days and peak hours
    combined_days = {day["day"]: day["peak_hours"] for day in previous_result["busiest_days"]}
    for day in top_days:
        combined_days[day] = peak_hour_ranges[day]
    top_days = sorted(combined_days.keys(), key=lambda x: new_df[new_df['day_of_week'] == x]['hour'].count() if x in new_df['day_of_week'].values else 0, reverse=True)[:3]
    peak_hour_ranges = {day: combined_days.get(day, "No Peak Hours") for day in top_days}

    # Merge top doctors
    combined_doctors = {doc["doctor_name"]: doc["visit_count"] for doc in previous_result["top_doctors"]}
    for name, count in top_doctors.items():
        combined_doctors[name] = combined_doctors.get(name, 0) + count
    top_doctors = dict(sorted(combined_doctors.items(), key=lambda x: x[1], reverse=True)[:3])

    # Merge top departments
    combined_depts = {dept["department_name"]: dept["visit_count"] for dept in previous_result["top_departments"]}
    for name, count in top_departments.items():
        combined_depts[name] = combined_depts.get(name, 0) + count
    top_departments = dict(sorted(combined_depts.items(), key=lambda x: x[1], reverse=True)[:3])

    return top_days, peak_hour_ranges, top_doctors, top_departments

def train_ip_model():
    """Train a model to analyze inpatient visit patterns."""
    engine = get_db_engine()
    
    last_timestamp = load_last_timestamp()
    previous_result = load_previous_model()
    print(f"Latest updated date before training: {last_timestamp if last_timestamp else 'None (first run)'}")
    
    # Fetch data from database
    if last_timestamp is None:
        query = """
        SELECT 
            tv.id, 
            tv.created_date, 
            tv.updated_date,
            CASE 
                WHEN me.description ~ '^[A-Za-z0-9+/]+={0,2}$' 
                THEN convert_from(decrypt(decode(me.description, 'base64'), 
                    'bAbHf961YGedwsboiSjTHHhfESckV6AH', 'aes-cbc/pad:pkcs'), 'UTF8')
                ELSE me.description
            END AS doctor_name,
            md.description AS department_name
        FROM transaction.t_visit tv
        INNER JOIN master.m_employee_details me ON me.id = tv.doctor_id AND me.active = TRUE
        INNER JOIN master.m_department md ON md.id = tv.department_id
        WHERE tv.visit_type_id = 2;
        """
        df = pd.read_sql(query, engine)
    else:
        query = """
        SELECT 
            tv.id, 
            tv.created_date, 
            tv.updated_date,
            CASE 
                WHEN me.description ~ '^[A-Za-z0-9+/]+={0,2}$' 
                THEN convert_from(decrypt(decode(me.description, 'base64'), 
                    'bAbHf961YGedwsboiSjTHHhfESckV6AH', 'aes-cbc/pad:pkcs'), 'UTF8')
                ELSE me.description
            END AS doctor_name,
            md.description AS department_name
        FROM transaction.t_visit tv
        INNER JOIN master.m_employee_details me ON me.id = tv.doctor_id AND me.active = TRUE
        INNER JOIN master.m_department md ON md.id = tv.department_id
        WHERE tv.updated_date > %s AND tv.visit_type_id = 2;
        """
        df = pd.read_sql(query, engine, params=(last_timestamp,))
    

    # Calculate total visits
    new_visits = len(df)
    total_ip_visits = new_visits
    if previous_result and 'total_ip_visits' in previous_result:
        total_ip_visits = previous_result['total_ip_visits'] + new_visits
    
    if df.empty:
        if previous_result:
            print("No new data found. Returning previous model results.")
            top_days = [day["day"] for day in previous_result["busiest_days"]]
            peak_hour_ranges = {day["day"]: day["peak_hours"] for day in previous_result["busiest_days"]}
            top_doctors = {doc["doctor_name"]: doc["visit_count"] for doc in previous_result["top_doctors"]}
            top_departments = {dept["department_name"]: dept["visit_count"] for dept in previous_result["top_departments"]}
            return top_days, peak_hour_ranges, 0.0, top_doctors, top_departments, previous_result.get('total_ip_visits', 0)
        print("No data found and no previous model exists. Returning default values.")
        return [], {}, 0.0, {}, {}
    
    # Timezone conversion
    malaysia_tz = pytz.timezone('Asia/Kuala_Lumpur')
    india_tz = pytz.timezone('Asia/Kolkata')
    df['created_date'] = pd.to_datetime(df['created_date']).dt.tz_localize(malaysia_tz).dt.tz_convert(india_tz)
    df['day_of_week'] = df['created_date'].dt.day_name()
    df['hour'] = df['created_date'].dt.hour
    
    # Identify top 3 busiest days from new data
    visit_counts = df.groupby(['day_of_week', 'hour']).size().reset_index(name='visit_count')
    day_totals = visit_counts.groupby('day_of_week')['visit_count'].sum().reset_index()
    top_days = day_totals.nlargest(3, 'visit_count')['day_of_week'].tolist()
    
    # Identify peak hours using threshold (85th percentile)
    top_days_data = visit_counts[visit_counts['day_of_week'].isin(top_days)].copy()
    threshold_85 = top_days_data['visit_count'].quantile(0.85)
    top_days_data.loc[:, 'peak_hour'] = (top_days_data['visit_count'] >= threshold_85).astype(int)
    
    # Check if there's enough data for training
    if len(top_days_data) < 2:
        print("Insufficient data for training (less than 2 samples). Proceeding without model training.")
    else:
        # Train the model
        X_train, X_test, y_train, y_test = train_test_split(
            top_days_data[['hour']], top_days_data['peak_hour'], test_size=0.2, random_state=42
        )
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
    
    # Get peak hours per day
    peak_hours = top_days_data[top_days_data['peak_hour'] == 1][['day_of_week', 'hour']]
    
    def get_time_range(hours):
        if not hours:
            return "No Peak Hours"
        hours = sorted(hours)
        start_hour, end_hour = hours[0], hours[-1] + 1
        def format_hour(hour):
            if hour == 12:
                return "12 PM"
            elif hour == 0:
                return "12 AM"
            elif hour < 12:
                return f"{hour} AM"
            else:
                return f"{hour - 12} PM"
        return f"{format_hour(start_hour)} to {format_hour(end_hour)}"
    
    peak_hour_ranges = {day: get_time_range(peak_hours[peak_hours['day_of_week'] == day]['hour'].tolist()) 
                        for day in top_days}
    
    # Calculate top doctors and departments from new data
    top_doctors_counts = df['doctor_name'].value_counts().nlargest(3)
    top_doctors = {name: int(count) for name, count in top_doctors_counts.items()}
    top_depts_counts = df['department_name'].value_counts().nlargest(3)
    top_departments = {name: int(count) for name, count in top_depts_counts.items()}
    
    # Merge with previous results if available
    if previous_result:
        top_days, peak_hour_ranges, top_doctors, top_departments = merge_previous_and_new(
            previous_result, df, top_days, peak_hour_ranges, top_doctors, top_departments
        )
    
    # Save latest timestamp and model
    latest_timestamp = pd.to_datetime(df['updated_date'], utc=True).max()
    save_timestamp(latest_timestamp)
    
    result = {
        "busiest_days": [{"day": day, "peak_hours": peak_hour_ranges[day]} for day in top_days],
        "top_departments": [{"department_name": name, "visit_count": count} for name, count in top_departments.items()],
        "top_doctors": [{"doctor_name": name, "visit_count": count} for name, count in top_doctors.items()],
        "latest_updated_date": str(latest_timestamp),
        "total_ip_visits": total_ip_visits
    }
    try:
        with open(MODEL_PATH, "wb") as file:
            pickle.dump(result, file)
        logger.info(f"Model saved to {MODEL_PATH}")
    except Exception as e:
        logger.error(f"Failed to save model to {MODEL_PATH}: {e}")
    
    print(f"Latest updated date after training: {latest_timestamp}")
    return top_days, peak_hour_ranges, accuracy if 'accuracy' in locals() else 0.0, top_doctors, top_departments, total_ip_visits

if __name__ == "__main__":
    top_days, peak_hour_ranges, accuracy, top_doctors, top_departments,total_ip_visits = train_ip_model()
    response_data = {
        "busiest_days": [{"day": day, "peak_hours": peak_hour_ranges[day]} for day in top_days],
        "top_doctors": [{"doctor_name": name, "visit_count": count} for name, count in top_doctors.items()],
        "top_departments": [{"department_name": name, "visit_count": count} for name, count in top_departments.items()],
        "total_ip_visits": total_ip_visits
    }
    print(json.dumps(response_data, indent=4))