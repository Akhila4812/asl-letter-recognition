import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from app import app

def test_flask_endpoint():
    client = app.test_client()
    response = client.get('/test')
    assert response.status_code == 200
    assert response.json == {'message': 'Flask is working!'}