# tests/test_api.py - Version compatible avec vos dépendances
import sys
import os
import json
from unittest.mock import Mock, patch

# Ajoute le chemin du projet pour pouvoir importer l'application
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fastapi.testclient import TestClient
from app.main import app

# Données de test
TEST_CUSTOMER = {
    "CreditScore": 650,
    "Age": 35,
    "Tenure": 5,
    "Balance": 50000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 75000.0,
    "Geography_Germany": 0,
    "Geography_Spain": 1
}

# === TESTS SANS MOCK (pour vérifier que l'API répond) ===

def test_read_root():
    """Test l'endpoint racine /"""
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["message"] == "Bank Churn Prediction API"

def test_health_check():
    """Test l'endpoint /health"""
    client = TestClient(app)
    response = client.get("/health")
    # Accepte 200 (healthy) ou 503 (modèle non chargé) pour l'instant
    assert response.status_code in [200, 503]

def test_predict_structure():
    """Test que l'endpoint /predict existe"""
    client = TestClient(app)
    response = client.post("/predict", json=TEST_CUSTOMER)
    # L'important est que l'API réponde
    assert response.status_code in [200, 503]

def test_docs_available():
    """Test que la documentation Swagger est accessible"""
    client = TestClient(app)
    response = client.get("/docs")
    assert response.status_code == 200

# === TESTS AVEC MOCK (pour éviter l'erreur 503) ===

def test_predict_with_mock():
    """Test /predict avec un mock du modèle"""
    # Mock le modèle avant d'importer/créer le client
    with patch('app.main.model') as mock_model:
        # Configure le mock
        mock_model.predict_proba.return_value = [[0.2, 0.8]]  # 80% de churn
        mock_model.predict.return_value = [1]
        
        # Crée le client APRÈS avoir mocké
        client = TestClient(app)
        
        response = client.post("/predict", json=TEST_CUSTOMER)
        assert response.status_code == 200
        
        data = response.json()
        assert "churn_probability" in data
        assert data["churn_probability"] == 0.8
        assert data["prediction"] == 1
        assert data["risk_level"] == "High"

def test_health_with_mock():
    """Test /health avec mock"""
    with patch('app.main.model') as mock_model:
        # Simule un modèle chargé
        mock_model.__bool__ = lambda self: True  # Pour que 'if model:' retourne True
        
        client = TestClient(app)
        response = client.get("/health")
        
        # Devrait maintenant être 200 grâce au mock
        assert response.status_code == 200
        assert response.json()["model_loaded"] == True

# === TEST DE VALIDATION DES DONNEES ===

def test_invalid_data_validation():
    """Test que la validation Pydantic fonctionne"""
    client = TestClient(app)
    
    # Données invalides : CreditScore trop bas
    invalid_data = TEST_CUSTOMER.copy()
    invalid_data["CreditScore"] = 250  # Doit être >= 300
    
    response = client.post("/predict", json=invalid_data)
    # Pydantic devrait rejeter avec 422
    assert response.status_code == 422
    
def test_batch_predict():
    """Test l'endpoint /predict/batch"""
    client = TestClient(app)
    
    batch_data = [TEST_CUSTOMER, TEST_CUSTOMER]
    
    with patch('app.main.model') as mock_model:
        mock_model.predict_proba.return_value = [[0.3, 0.7], [0.3, 0.7]]
        
        response = client.post("/predict/batch", json=batch_data)
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert data["count"] == 2