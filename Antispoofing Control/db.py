import psycopg2
from sqlalchemy import create_engine
from config import db_params

def get_db_engine():
    return create_engine(f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['dbname']}")

