#!/bin/sh

# Run Flask app with Gunicorn
exec gunicorn -w 4 -b 0.0.0.0:5000 app:app
