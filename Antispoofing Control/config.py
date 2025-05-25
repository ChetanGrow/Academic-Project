import os

db_params = {
    "dbname": os.getenv("DB_NAME", "PatientDB"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "jDrv73ltVuo42RCoqX4C"),
    "host": os.getenv("DB_HOST", "192.168.120.51"),
    "port": os.getenv("DB_PORT", "5432"),
}
