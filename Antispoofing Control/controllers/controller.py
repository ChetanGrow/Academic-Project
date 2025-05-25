from flask import Blueprint, jsonify, request
from flask_cors import CORS  # Import CORS
from models.visit_model import train_visit_model
from models.op_model import train_op_model
from models.ip_model import train_ip_model
from models.lab_test_model import get_lab_results
from models.face_model import register_patient, search_patient
from util import convert_base64_to_embedding  # Import new utility function
from util import convert_base64_to_embedding, normalize_embedding
import logging  # Import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the blueprint
visit_blueprint = Blueprint('hati', __name__)

@visit_blueprint.route('/0', methods=['GET'])
def predict_visit():
    """Return visit analytics: busiest days, peak hours, top doctors, top departments, and total visits."""
    top_days, peak_hour_ranges, accuracy, top_doctors, top_departments, total_visits = train_visit_model()

    response_data = {
        "busiest_days": [{"day": day, "peak_hours": peak_hour_ranges[day]} for day in top_days],
        "top_doctors": [{"doctor_name": doctor_name, "visit_count": count} for doctor_name, count in top_doctors.items()],
        "top_departments": [{"department_name": dept_name, "visit_count": count} for dept_name, count in top_departments.items()],
        "total_visits": total_visits
        #"model_accuracy": f"{accuracy * 100:.2f}%"  # Commented out as in original
    }

    return jsonify(response_data)

@visit_blueprint.route('/1', methods=['GET'])
def predict_op():
    """Return OP analytics: busiest days, peak hours, top doctors, top departments, and total OP visits."""
    top_days, peak_hour_ranges, accuracy, top_doctors, top_departments, total_op_visits = train_op_model()

    response_data = {
        "busiest_days": [{"day": day, "peak_hours": peak_hour_ranges[day]} for day in top_days],
        "top_doctors": [{"doctor_name": doctor_name, "visit_count": count} for doctor_name, count in top_doctors.items()],
        "top_departments": [{"department_name": dept_name, "visit_count": count} for dept_name, count in top_departments.items()],
        "total_op_visits": total_op_visits
        #"model_accuracy": f"{accuracy * 100:.2f}%"
    }

    return jsonify(response_data)


@visit_blueprint.route('/2', methods=['GET'])
def predict_ip():
    """Return IP analytics: busiest days, peak hours, top departments, and total IP visits."""
    top_days, peak_hour_ranges, accuracy, top_doctors, top_departments, total_ip_visits = train_ip_model()

    response_data = {
        "busiest_days": [{"day": day, "peak_hours": peak_hour_ranges[day]} for day in top_days],
        "top_doctors": [{"doctor_name": doctor_name, "visit_count": count} for doctor_name, count in top_doctors.items()],
        "top_departments": [{"department_name": dept_name, "visit_count": count} for dept_name, count in top_departments.items()],
        "total_ip_visits": total_ip_visits
        #"model_accuracy": f"{accuracy * 100:.2f}%"
    }

    return jsonify(response_data)


@visit_blueprint.route("/test_results", methods=['GET', 'PATCH'])
def fetch_lab_results():
    """
    Endpoint to return lab results for a given MRN or all MRNs.
    Expects 'mrn' as an optional query parameter.
    """
    # Extract MRN from query parameters (optional)
    mrn = request.args.get("mrn")

    # Get lab results from the model
    results, error = get_lab_results(mrn)

    if error:
        status_code = 404 if "No data found" in error else 500
        return jsonify({"error": error}), status_code

    return jsonify(results), 200


@visit_blueprint.route('/register_patient', methods=['POST'])
def register_patient_route():
    # Get mrn and base64_image from JSON body, form data, or query parameters
    if request.is_json:
        data = request.get_json()
        mrn = data.get('mrn', '').strip()
        base64_image = data.get('base64_image', '').strip()
    else:
        mrn = request.form.get('mrn', '').strip() or request.args.get('mrn', '').strip()
        base64_image = request.form.get('base64_image', '').strip() or request.args.get('base64_image', '').strip()
    
    # Validate inputs
    if not mrn:
        return jsonify({'status': 'error', 'message': 'MRN is required.'}), 400
    if not base64_image:
        return jsonify({'status': 'error', 'message': 'Base64 image data is required.'}), 400
    
    # Convert base64 image to embedding vector
    embedding_vector, error = convert_base64_to_embedding(base64_image)
    if error:
        return jsonify({'status': 'error', 'message': error}), 400
    
    # Call model function with the converted embedding
    result, status_code = register_patient(mrn, embedding_vector)
    return jsonify(result), status_code

@visit_blueprint.route('/search_patient', methods=['POST'])
def search_patient_route():
    # Get base64_image from JSON body, form data, or query parameters
    if request.is_json:
        data = request.get_json()
        base64_image = data.get('base64_image', '').strip()
    else:
        base64_image = request.form.get('base64_image', '').strip() or request.args.get('base64_image', '').strip()
    
    # Validate input
    if not base64_image:
        return jsonify({'status': 'error', 'message': 'Base64 image data is required.'}), 400
    
    # Convert base64 image to embedding vector
    embedding_vector, error = convert_base64_to_embedding(base64_image)
    if error:
        return jsonify({'status': 'error', 'message': error}), 400
    
    # Normalize the embedding
    normalized_embedding = normalize_embedding(embedding_vector)
    logger.info(f"Normalized embedding for search: {normalized_embedding[:5]}...")  # Log first 5 elements for brevity

    # Extract Bearer token from Authorization header
    auth_header = request.headers.get('Authorization')  # Fixed typo
    
    bearer_token = None
    if auth_header and auth_header.startswith('Bearer '):
        bearer_token = auth_header[len('Bearer '):].strip()
    
    # Call model function with the normalized embedding
    result, status_code = search_patient(normalized_embedding, bearer_token)
    return jsonify(result), status_code