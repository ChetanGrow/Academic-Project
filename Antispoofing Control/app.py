from flask import Flask
from controllers.controller import visit_blueprint
#from controllers.face_controller import face_blueprint
from health import health_bp        # Import health blueprint
from flask_cors import CORS

app = Flask(__name__)

# Enable CORS globally
CORS(
    app,
    origins=["*"],                  # Allow all origins (Change this to specific domains if needed)
    supports_credentials=False,
    methods=["*"],                  # Allow all HTTP methods
    allow_headers=["*"],            # Allow all headers
)

# Registering Blueprints
app.register_blueprint(visit_blueprint, url_prefix="/hati")
#app.register_blueprint(face_blueprint, url_prefix="/hati")
app.register_blueprint(health_bp, url_prefix="/")  # Healthcheck available at `/health`

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
