"""
Comprehensive Unit Tests for AI-Powered Smart Farm Assistant.
Validates OOP architecture, custom exceptions, ML predictions, Flask API,
Crop Recommendation, and PDF/Audio generation.
"""

import os
import io
import unittest
import numpy as np
from PIL import Image

from modules.inventory_manager import (
    InventoryManager, InventoryItem, ItemNotFoundError,
    InsufficientStockError, InvalidItemDataError
)
from modules.weather_service import WeatherService
from models.weather_predictor import WeatherPredictor
from modules.iot_soil_monitor import SoilSensorNode
from modules.farm_advisor import FarmAdvisor
from models.crop_fertilizer_recommender import CropRecommender
from utils.pdf_generator import generate_diagnostic_pdf
from api_server import app

class TestInventoryManager(unittest.TestCase):
    """Tests OOP Inventory logic, custom exceptions, and bounds enforcement."""

    def setUp(self):
        self.test_json = "data/test_inventory.json"
        self.test_csv = "data/test_inventory.csv"
        self.mgr = InventoryManager(json_path=self.test_json, csv_path=self.test_csv)

    def tearDown(self):
        for path in [self.test_json, self.test_csv]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception:
                    pass

    def test_add_and_get_item(self):
        item = InventoryItem(
            item_id="TEST-001",
            name="Organic Compost",
            category="Fertilizer",
            quantity=50.0,
            unit="Bags",
            min_threshold=10.0,
            price_per_unit=250.0,
            location="Shed 1",
            expiry_date="2027-12-31"
        )
        self.mgr.add_item(item)
        fetched = self.mgr.get_item("TEST-001")
        self.assertEqual(fetched.name, "Organic Compost")
        self.assertEqual(fetched.total_value, 12500.0)

    def test_item_not_found_exception(self):
        with self.assertRaises(ItemNotFoundError):
            self.mgr.get_item("NON-EXISTENT-ID")

    def test_invalid_category_exception(self):
        with self.assertRaises(InvalidItemDataError):
            InventoryItem(
                item_id="BAD-01",
                name="Bad Item",
                category="InvalidCategoryX",
                quantity=10.0,
                unit="kg",
                min_threshold=2.0,
                price_per_unit=100.0,
                location="A",
                expiry_date="2027-01-01"
            )

    def test_insufficient_stock_exception(self):
        item = InventoryItem(
            item_id="TEST-002",
            name="Seed Pack",
            category="Seeds",
            quantity=5.0,
            unit="Packets",
            min_threshold=2.0,
            price_per_unit=300.0,
            location="Shelf",
            expiry_date="2027-01-01"
        )
        self.mgr.add_item(item)
        with self.assertRaises(InsufficientStockError):
            self.mgr.use_stock("TEST-002", 10.0, "Test field trial")

    def test_restock_and_deduct(self):
        item = InventoryItem(
            item_id="TEST-003",
            name="Neem Extract",
            category="Pesticide",
            quantity=10.0,
            unit="Liters",
            min_threshold=4.0,
            price_per_unit=500.0,
            location="Rack",
            expiry_date="2027-01-01"
        )
        self.mgr.add_item(item)
        self.mgr.use_stock("TEST-003", 3.0)
        self.assertEqual(self.mgr.get_item("TEST-003").quantity, 7.0)
        self.mgr.restock("TEST-003", 5.0)
        self.assertEqual(self.mgr.get_item("TEST-003").quantity, 12.0)

class TestWeatherPredictor(unittest.TestCase):
    """Tests ML weather forecast engine."""

    def setUp(self):
        self.wp = WeatherPredictor()

    def test_ml_models_loaded(self):
        self.assertIsNotNone(self.wp.temp_regressor)
        self.assertIsNotNone(self.wp.rain_classifier)
        self.assertIsNotNone(self.wp.rain_regressor)

    def test_7day_prediction(self):
        forecast = self.wp.predict_multi_day_trend(days=7)
        self.assertEqual(len(forecast), 7)
        for day in forecast:
            self.assertIn("mean_temp_c", day)
            self.assertIn("rain_prob_pct", day)
            self.assertIn("predicted_rain_mm", day)
            self.assertTrue(0 <= day["rain_prob_pct"] <= 100)
            self.assertTrue(day["max_temp_c"] >= day["min_temp_c"])

class TestIoTSensor(unittest.TestCase):
    """Tests IoT soil node simulation and automated irrigation logic."""

    def test_sensor_telemetry(self):
        node = SoilSensorNode(crop_type="Tomato")
        telemetry = node.read_telemetry(ambient_temp=30.0)
        self.assertIn("soil_moisture_pct", telemetry)
        self.assertIn("soil_temp_c", telemetry)
        self.assertIn("soil_ph", telemetry)
        self.assertIn("npk", telemetry)

    def test_irrigation_recommendation(self):
        node = SoilSensorNode(crop_type="Tomato")
        node.soil_moisture_pct = 25.0
        rec = node.calculate_irrigation_recommendation()
        self.assertTrue(rec["water_required_liters_per_ha"] > 0)
        self.assertEqual(rec["urgency"], "HIGH")

class TestFarmAdvisor(unittest.TestCase):
    """Tests agricultural offline advisory engine."""

    def test_advisor_response(self):
        advisor = FarmAdvisor()
        res = advisor.generate_response("What is the fertilizer dosage for tomato?")
        self.assertIn("Fertilizer", res.get("title", ""))
        self.assertIn("DAP", res.get("response", ""))

class TestCropFertilizerRecommender(unittest.TestCase):
    """Tests Crop Recommendation ML model and fertilizer dosage engine."""

    def setUp(self):
        self.rec = CropRecommender()

    def test_crop_recommendation(self):
        recs = self.rec.recommend(n=80, p=40, k=40, temp=24, hum=82, ph=6.5, rain=200)
        self.assertTrue(len(recs) >= 1)
        self.assertIn("crop", recs[0])
        self.assertIn("confidence_pct", recs[0])

    def test_fertilizer_prescription(self):
        presc = CropRecommender.calculate_fertilizer_prescription(soil_n=40, soil_p=20, soil_k=20, target_crop="Tomato")
        self.assertEqual(presc["target_crop"], "Tomato")
        self.assertIn("Urea (46% N)", presc["fertilizer_dosage_kg_ha"])
        self.assertIn("DAP (18-46-0)", presc["fertilizer_dosage_kg_ha"])
        self.assertIn("MOP (60% K)", presc["fertilizer_dosage_kg_ha"])
        self.assertTrue(presc["fertilizer_dosage_kg_ha"]["Urea (46% N)"] >= 0)

class TestPDFGenerator(unittest.TestCase):
    """Tests PDF diagnostic prescription creation."""

    def test_generate_pdf_bytes(self):
        mock_diag = {
            "confidence_pct": 92.5,
            "diagnostic": {
                "crop": "Tomato",
                "disease_name": "Early Blight",
                "severity": "Moderate",
                "pathogen": "Alternaria solani",
                "immediate_action": "Isolate leaves",
                "organic_treatment": ["Spray Neem oil"],
                "chemical_treatment": ["Mancozeb 75 WP"],
                "prevention": "Rotate crops",
                "safe_harvest_wait_days": 7
            }
        }
        mock_weather = {"city": "Bangalore", "temp_c": 28.0, "humidity_pct": 65, "condition": "Clear"}
        pdf_bytes = generate_diagnostic_pdf(mock_diag, mock_weather)
        self.assertTrue(len(pdf_bytes) > 500)
        self.assertTrue(pdf_bytes.startswith(b"%PDF"))

class TestFlaskAPI(unittest.TestCase):
    """Tests Flask REST microservice API endpoints."""

    def setUp(self):
        self.client = app.test_client()

    def test_health_endpoint(self):
        res = self.client.get("/api/health")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertEqual(data["status"], "online")

    def test_inventory_endpoint(self):
        res = self.client.get("/api/inventory")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("items", data)

    def test_iot_telemetry_endpoint(self):
        res = self.client.get("/api/iot/telemetry")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertIn("soil_moisture_pct", data)

    def test_crop_recommend_endpoint(self):
        res = self.client.get("/api/crop/recommend?n=80&p=40&k=40&temp=25&hum=80&ph=6.5&rain=180")
        self.assertEqual(res.status_code, 200)
        data = res.get_json()
        self.assertTrue(data["success"])
        self.assertTrue(len(data["recommendations"]) > 0)

if __name__ == "__main__":
    unittest.main()
