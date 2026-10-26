
import os
import io
import json
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS

from modules.inventory_manager import (
    InventoryManager, InventoryItem, ItemNotFoundError,
    InsufficientStockError, InvalidItemDataError
)
from modules.weather_service import WeatherService
from models.weather_predictor import WeatherPredictor
from modules.disease_classifier import DiseaseClassifier
from modules.iot_soil_monitor import SoilSensorNode
from modules.farm_advisor import FarmAdvisor
from models.crop_fertilizer_recommender import CropRecommender

app = Flask(__name__)
CORS(app)

inv_mgr = InventoryManager()
weather_svc = WeatherService()
weather_pred = WeatherPredictor()
disease_clf = DiseaseClassifier()
soil_node = SoilSensorNode(crop_type="Tomato", soil_type="Loamy")
farm_advisor = FarmAdvisor()
crop_rec = CropRecommender()

DOCS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Smart Farm AI - REST API Documentation</title>
    <style>
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; margin: 40px; background:
        .header { background: linear-gradient(135deg,
        .endpoint { background: white; border: 1px solid
        .method { font-weight: bold; padding: 4px 10px; border-radius: 6px; font-size: 0.85rem; }
        .get { background:
        .post { background:
        code { background:
        pre { background:
    </style>
</head>
<body>
    <div class="header">
        <h1>AI Smart Farm Assistant - REST API</h1>
        <p>Comprehensive backend endpoints for Mobile Apps, IoT Edge Nodes, and Agronomic Services.</p>
    </div>

    <h2>Available Endpoints</h2>

    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/health</code>
        <p>Checks API status and loaded Machine Learning models.</p>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/weather/current?city=Bangalore</code>
        <p>Returns current meteorological observations with offline micro-climate fallback.</p>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/weather/predict?days=7</code>
        <p>Generates ML-predicted 7-day temperature and rainfall trajectories.</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <code>/api/crop/disease/predict</code>
        <p>Multipart form-data with <code>image</code> file. Returns CNN classification, confidence %, and organic & chemical treatments.</p>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/crop/recommend?n=80&p=40&k=40&temp=24&hum=80&ph=6.5&rain=180</code>
        <p>Recommends top suitable crops with ML suitability confidence.</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <code>/api/crop/fertilizer/calculate</code>
        <p>JSON Body: <code>{"soil_n": 40, "soil_p": 20, "soil_k": 30, "target_crop": "Tomato"}</code>. Returns exact Urea, DAP, and MOP dosages.</p>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/inventory</code>
        <p>Returns list of all inventory items with low-stock badges and valuation.</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <code>/api/inventory/use</code>
        <p>JSON Body: <code>{"item_id": "PEST-001", "quantity": 1.5, "purpose": "Blight spray"}</code>. Implements <code>InsufficientStockError</code> handling.</p>
    </div>

    <div class="endpoint">
        <span class="method get">GET</span> <code>/api/iot/telemetry</code>
        <p>Returns real-time IoT soil sensor telemetry (Moisture %, Temp, pH, EC, NPK, Pump status).</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <code>/api/iot/pump</code>
        <p>JSON Body: <code>{"state": true}</code>. Controls automated solenoid irrigation pump.</p>
    </div>

    <div class="endpoint">
        <span class="method post">POST</span> <code>/api/advisor/chat</code>
        <p>JSON Body: <code>{"message": "How to make Jeevamrutha?"}</code>. Interactive AI agricultural advice.</p>
    </div>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(DOCS_HTML)

@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "online",
        "timestamp": datetime.now().isoformat(),
        "models": {
            "disease_cnn": disease_clf.model_type,
            "weather_ml": "RandomForestRegressor & GradientBoostingRegressor",
            "crop_recommender": f"RandomForestClassifier ({crop_rec.accuracy}% acc)",
            "advisor": "Google Gemini 2.5 Flash / Offline Agro-Engine"
        }
    })

@app.route("/api/weather/current", methods=["GET"])
def get_current_weather():
    city = request.args.get("city", "Bangalore")
    data = weather_svc.get_current_weather(city=city)
    return jsonify(data)

@app.route("/api/weather/predict", methods=["GET"])
def predict_weather():
    days = int(request.args.get("days", 7))
    forecast = weather_pred.predict_multi_day_trend(days=days)
    return jsonify({
        "days_projected": days,
        "metrics": weather_pred.metrics,
        "forecast": forecast
    })

@app.route("/api/crop/disease/predict", methods=["POST"])
def predict_crop_disease():
    if "image" not in request.files:
        return jsonify({"success": False, "error": "No image file provided in form-data ('image')"}), 400

    file = request.files["image"]
    try:
        img_bytes = file.read()
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        res = disease_clf.classify_image(pil_img)
        return jsonify(res)
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/crop/recommend", methods=["GET"])
def recommend_crop():
    try:
        n = float(request.args.get("n", 80))
        p = float(request.args.get("p", 40))
        k = float(request.args.get("k", 40))
        temp = float(request.args.get("temp", 25.0))
        hum = float(request.args.get("hum", 75.0))
        ph = float(request.args.get("ph", 6.5))
        rain = float(request.args.get("rain", 150.0))

        recs = crop_rec.recommend(n, p, k, temp, hum, ph, rain)
        return jsonify({"success": True, "recommendations": recs})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/crop/fertilizer/calculate", methods=["POST"])
def calculate_fertilizer():
    data = request.get_json() or {}
    soil_n = float(data.get("soil_n", 60))
    soil_p = float(data.get("soil_p", 25))
    soil_k = float(data.get("soil_k", 35))
    crop_name = data.get("target_crop", "Tomato")

    prescription = CropRecommender.calculate_fertilizer_prescription(soil_n, soil_p, soil_k, crop_name)
    return jsonify({"success": True, "prescription": prescription})

@app.route("/api/inventory", methods=["GET"])
def get_inventory():
    items = [i.to_dict() for i in inv_mgr.get_all_items()]
    return jsonify({
        "total_valuation": inv_mgr.get_total_valuation(),
        "low_stock_count": len(inv_mgr.get_low_stock_items()),
        "items": items
    })

@app.route("/api/inventory/use", methods=["POST"])
def use_inventory():
    data = request.get_json() or {}
    item_id = data.get("item_id")
    qty = float(data.get("quantity", 0))
    purpose = data.get("purpose", "Application")

    if not item_id or qty <= 0:
        return jsonify({"success": False, "error": "Item ID and positive quantity required"}), 400

    try:
        updated = inv_mgr.use_stock(item_id, qty, purpose=purpose)
        return jsonify({"success": True, "item": updated.to_dict()})
    except InsufficientStockError as e:
        return jsonify({"success": False, "error": "InsufficientStockError", "message": str(e)}), 409
    except ItemNotFoundError as e:
        return jsonify({"success": False, "error": "ItemNotFoundError", "message": str(e)}), 404
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route("/api/inventory/restock", methods=["POST"])
def restock_inventory():
    data = request.get_json() or {}
    item_id = data.get("item_id")
    qty = float(data.get("quantity", 0))
    price = data.get("price")

    try:
        price_val = float(price) if price is not None else None
        updated = inv_mgr.restock(item_id, qty, new_price=price_val)
        return jsonify({"success": True, "item": updated.to_dict()})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/iot/telemetry", methods=["GET"])
def get_iot_telemetry():
    telemetry = soil_node.read_telemetry()
    return jsonify(telemetry)

@app.route("/api/iot/pump", methods=["POST"])
def control_pump():
    data = request.get_json() or {}
    state = bool(data.get("state", False))
    soil_node.trigger_pump(state)
    return jsonify({
        "success": True,
        "pump_status": "ACTIVE" if state else "STANDBY",
        "pump_active": state
    })

@app.route("/api/advisor/chat", methods=["POST"])
def advisor_chat():
    data = request.get_json() or {}
    msg = data.get("message", "")
    if not msg:
        return jsonify({"error": "Empty message"}), 400

    reply = farm_advisor.generate_response(msg)
    return jsonify(reply)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)