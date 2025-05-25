import os
import yaml
import psycopg2
from psycopg2 import Error
import httpx
import logging
from psycopg2.pool import ThreadedConnectionPool
import base64
import numpy as np
import face_recognition
from PIL import Image
import io

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load database configuration
try:
    with open('config.yml', 'r') as file:
        config = yaml.safe_load(file)
    if config is None:
        raise ValueError("config.yml is empty or invalid")
    DB_CONFIG_AI = config['database']['backend_ai']
except FileNotFoundError:
    logger.error("config.yml not found in the project directory")
    raise
except KeyError as e:
    logger.error(f"Missing key in config.yml: {e}")
    raise
except Exception as e:
    logger.error(f"Error loading config.yml: {e}")
    raise

# Create a connection pool for Backend_AI
try:
    db_pool_ai = ThreadedConnectionPool(1, 20, **DB_CONFIG_AI)
except Exception as e:
    logger.error(f"Error initializing database connection pool: {e}")
    raise

def get_db_connection_ai():
    # Establish a connection to Backend_AI from the pool.
    try:
        return db_pool_ai.getconn()
    except Error as e:
        logger.error(f"Error connecting to Backend_AI: {e}")
        return None

def release_db_connection(conn):
    # Release the connection back to the pool.
    if conn:
        db_pool_ai.putconn(conn)

def get_patient_data(mrn, bearer_token):
    # Retrieve patient data from the external API based on MRN using the provided Bearer token.
    base_url = "https://mykare360-dev.hatiintl.com/opd-service/extendedPatient/filter"
    params = {
        "operation": "search",
        "identificationId": "",
        "name": "",
        "mrn": mrn,
        "gender": "",
        "unitCode": "",
        "city": "",
        "identificationType": "",
        "frDate": "",
        "toDate": "",
        "id": "",
        "mobile": "",
        "birthDate": "",
        "offset": 0,
        "size": 10,
        "pincode": "",
        "country": "",
        "state": ""
    }

    # Prepare headers with the provided Bearer token
    headers = {}
    if bearer_token:
        headers['Authorization'] = f'Bearer {bearer_token}'
    else:
        logger.error("No Bearer token provided")
        return None

    try:
        with httpx.Client() as client:
            response = client.get(base_url, params=params, headers=headers, timeout=10.0)
            response.raise_for_status()  # Raises an exception for 4xx/5xx errors
            data = response.json()
            logger.info(f"API Response for MRN {mrn}: {data} (Type: {type(data)})")

            # Handle the API response
            if isinstance(data, list) and len(data) > 0:
                return data[0]  # Return the first matching record as a dictionary
            elif isinstance(data, dict) and 'result' in data and len(data['result']) > 0:
                return data['result'][0]  # Handle case where data is nested under 'result'
            elif isinstance(data, dict):
                return data  # Return the dictionary if no nested result
            logger.warning(f"No patient data found for MRN {mrn}")
            return None
    except httpx.HTTPStatusError as e:
        logger.error(f"HTTP error retrieving patient data for MRN {mrn}: {e.response.text if e.response else str(e)}")
        return None
    except httpx.RequestError as e:
        logger.error(f"Request error retrieving patient data for MRN {mrn}: {e}")
        return None
    except ValueError as e:
        logger.error(f"Error parsing API response for MRN {mrn}: {e}")
        return None

def normalize_embedding(embedding_vector):
    """
    Normalize the embedding vector to unit length.
    Returns: normalized embedding as a list
    """
    embedding_array = np.array(embedding_vector)
    norm = np.linalg.norm(embedding_array)
    if norm == 0:
        return embedding_vector  # Avoid division by zero
    normalized = (embedding_array / norm).tolist()
    return normalized

def convert_base64_to_embedding(base64_image):
    """
    Convert base64 image data to a 128-dimensional embedding vector.
    Returns: (embedding_vector, error_message)
    """
    if not base64_image:
        return None, "No image data provided."
    
    try:
        # Decode base64 string to bytes
        image_data = base64.b64decode(base64_image)
        
        # Load image from bytes
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_array = np.array(image)
        
        # Generate face embeddings
        embeddings = face_recognition.face_encodings(image_array)
        if not embeddings:
            return None, "No faces detected in the image."
        
        # Return the first embedding as a list, normalized
        embedding_vector = embeddings[0].tolist()
        if len(embedding_vector) != 128:
            return None, "Generated embedding vector is not 128-dimensional."
        
        # Normalize the embedding
        normalized_embedding = normalize_embedding(embedding_vector)
        return normalized_embedding, None
    except base64.binascii.Error:
        return None, "Invalid base64 image data."
    except Exception as e:
        logger.error(f"Error converting base64 to embedding: {e}")
        return None, f"Failed to process image: {str(e)}"