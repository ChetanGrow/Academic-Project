import json
import logging
import numpy as np
from psycopg2 import Error
from util import get_patient_data, db_pool_ai

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_db_connection_ai():
    # Establish a connection to the Backend_AI database from the pool.
    try:
        return db_pool_ai.getconn()
    except Error as e:
        logger.error(f"Error connecting to Backend_AI: {e}")
        return None

def release_db_connection(conn):
    # Release the connection back to the pool.
    if conn:
        db_pool_ai.putconn(conn)

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

def register_patient(mrn, embedding_vector):
    # Register a patient with the provided MRN and face embedding.
    logger.info("Starting register_patient model function")
    
    # Validate inputs
    if not mrn:
        return {'status': 'error', 'message': 'Please provide an MRN.'}, 400
    if not embedding_vector:
        return {'status': 'error', 'message': 'Please provide a valid image.'}, 400
    
    # Validate embedding vector
    if not isinstance(embedding_vector, list) or len(embedding_vector) != 128:
        return {
            'status': 'error',
            'message': 'Invalid embedding vector format. Must be a list of 128 floats.'
        }, 400

    # Normalize the embedding
    normalized_embedding = normalize_embedding(embedding_vector)
    logger.info(f"Normalized embedding: {normalized_embedding[:5]}...")  # Log first 5 elements for brevity

    conn = get_db_connection_ai()
    if conn is None:
        return {'status': 'error', 'message': 'Database connection failed.'}, 500
    
    try:
        cur = conn.cursor()
        # Check if MRN already exists
        cur.execute("SELECT mrn FROM face_embeddings WHERE mrn = %s", (mrn,))
        if cur.fetchone():
            logger.info(f"Register - MRN {mrn} already exists in the database")
            return {
                'status': 'error',
                'message': f'MRN {mrn} is already registered. Please use a unique MRN.'
            }, 400

        # Check for similar face embeddings (top 3 matches for debugging)
        cur.execute(
            """
            SELECT mrn, embedding_vector <-> %s::vector AS distance 
            FROM face_embeddings 
            WHERE embedding_vector IS NOT NULL 
            AND embedding_vector <-> %s::vector < 1.0 
            ORDER BY distance LIMIT 1
            """,
            (str(normalized_embedding), str(normalized_embedding))
        )
        rows = cur.fetchall()
        if rows:
            for i, row in enumerate(rows):
                logger.info(f"Register - Match {i+1}: MRN {row[0]} at distance {row[1]}")
            closest_row = rows[0]  # Closest match
            if closest_row[1] < 0.35:  # Adjusted threshold to 0.35
                return {
                    'status': 'error',
                    'message': f'Face is already registered under MRN: {closest_row[0]}'
                }, 400

        # Insert new embedding
        cur.execute(
            """
            INSERT INTO face_embeddings (mrn, embedding_vector) 
            VALUES (%s, %s::vector)
            """,
            (mrn, str(normalized_embedding))
        )
        conn.commit()
        logger.info(f"Register - New patient registered with MRN: {mrn}")
        return {
            'status': 'success',
            'message': f'Patient with MRN {mrn} registered successfully!'
        }, 201
    except Error as e:
        logger.error(f"Database error during registration: {e}")
        return {
            'status': 'error',
            'message': 'Registration failed due to database error.'
        }, 500
    finally:
        if 'cur' in locals():
            cur.close()
        release_db_connection(conn)

def search_patient(embedding_vector, bearer_token):
    # Search for a patient based on face embedding.
    logger.info("Starting search_patient model function")
    
    # Validate input
    if not embedding_vector:
        return {
            'status': 'error',
            'message': 'Please provide a valid image.'
        }, 400
    
    # Validate embedding vector
    if not isinstance(embedding_vector, list) or len(embedding_vector) != 128:
        return {
            'status': 'error',
            'message': 'Invalid embedding vector format. Must be a list of 128 floats.'
        }, 400

    # Normalize the embedding
    normalized_embedding = normalize_embedding(embedding_vector)

    conn = get_db_connection_ai()
    if conn is None:
        return {
            'status': 'error',
            'message': 'Database connection failed.'
        }, 500
    
    try:
        cur = conn.cursor()
        # Check for similar face embeddings (top 3 matches for debugging)
        cur.execute(
            """
            SELECT mrn, embedding_vector <-> %s::vector AS distance 
            FROM face_embeddings 
            WHERE embedding_vector IS NOT NULL 
            AND embedding_vector <-> %s::vector < 1.0 
            ORDER BY distance LIMIT 1
            """,
            (str(normalized_embedding), str(normalized_embedding))
        )
        rows = cur.fetchall()
        if rows:
            for i, row in enumerate(rows):
                logger.info(f"Search - Match {i+1}: MRN {row[0]} at distance {row[1]}")
            closest_row = rows[0]  # Closest match
            if closest_row[1] < 0.35:  # Adjusted threshold to 0.35 for consistency
                logger.info(f"Match found: MRN {closest_row[0]}")
                patient_data = get_patient_data(closest_row[0], bearer_token)
                if patient_data is not None:
                    logger.info(f"Patient data retrieved for MRN {closest_row[0]}: {patient_data}")
                    return {'status': 'success', 'data': patient_data}, 200
                else:
                    logger.warning(f"No patient data available for MRN {closest_row[0]}")
                    return {
                        'status': 'success',
                        'message': f'MRN: {closest_row[0]}\nID: Not available (API unavailable)'
                    }, 200
        else:
            logger.info("No matching patient found")
            return {
                'status': 'error',
                'message': 'Unknown patient. Please register or try again.'
            }, 404
    except Error as e:
        logger.error(f"Database error during search: {e}")
        return {
            'status': 'error',
            'message': 'Search failed due to database error.'
        }, 500
    finally:
        if 'cur' in locals():
            cur.close()
        release_db_connection(conn)