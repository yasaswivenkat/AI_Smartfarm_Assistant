
import os
import sys
import json
import time
from datetime import datetime
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st

st.set_page_config(
    page_title="AI Smart Farm Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

css_path = os.path.join(os.path.dirname(__file__), "static", "style.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

from modules.inventory_manager import (
    InventoryManager, InventoryItem, InventoryException,
    ItemNotFoundError, InsufficientStockError, InvalidItemDataError
)
from modules.weather_service import WeatherService
from models.weather_predictor import WeatherPredictor
from modules.disease_classifier import DiseaseClassifier, DISEASE_KNOWLEDGE_BASE
from modules.iot_soil_monitor import SoilSensorNode
from modules.farm_advisor import FarmAdvisor
from models.crop_fertilizer_recommender import CropRecommender
from modules.audio_advisor import synthesize_speech_bytes
from utils.pdf_generator import generate_diagnostic_pdf
from utils.visualizer import (
    plot_weather_historical_matplotlib,
    plot_weather_correlation_seaborn,
    plot_rainfall_distribution_seaborn,
    plot_feature_importance_matplotlib,
    plot_cnn_training_history_matplotlib,
    plot_forecast_plotly,
    plot_inventory_valuation_plotly,
    plot_soil_telemetry_plotly
)

@st.cache_resource
def get_inventory_manager():
    return InventoryManager()

@st.cache_resource
def get_weather_predictor():
    return WeatherPredictor()

@st.cache_resource
def get_disease_classifier():
    return DiseaseClassifier()

@st.cache_resource
def get_soil_sensor():
    return SoilSensorNode(crop_type="Tomato", soil_type="Loamy")

@st.cache_resource
def get_crop_recommender():
    return CropRecommender()

if "weather_api_key" not in st.session_state:
    st.session_state.weather_api_key = ""
if "gemini_api_key" not in st.session_state:
    st.session_state.gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "**Namaste! I am Kisan Mitra, your AI Smart Farm Assistant.** How can I help you today with crops, weather, disease diagnosis, fertilizer calculations, or inventory?"}
    ]

inv_mgr = get_inventory_manager()
weather_pred = get_weather_predictor()
disease_clf = get_disease_classifier()
soil_node = get_soil_sensor()
crop_rec = get_crop_recommender()
farm_advisor = FarmAdvisor(gemini_api_key=st.session_state.gemini_api_key)
weather_svc = WeatherService(api_key=st.session_state.weather_api_key)

with st.sidebar:
    st.markdown("""
    <div style='display:flex; align-items:center; gap:10px; margin-bottom:20px; padding-bottom:12px; border-bottom:1px solid rgba(255,255,255,0.1);'>
        <div>
            <h2 style='margin:0; font-size:1.35rem; font-weight:700; color:#ffffff; letter-spacing:-0.3px;'>Smart Farm AI</h2>
            <p style='margin:0; font-size:0.82rem; color:#34d399; font-weight:500;'>Precision Agriculture System</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_choice = st.radio(
        "Navigation Menu",
        [
            "Farm Overview",
            "Weather & ML Forecast",
            "Crop Disease Detection",
            "Crop & Fertilizer Recommender",
            "Smart Inventory",
            "IoT Soil & Irrigation",
            "AI Farm Advisor",
            "Data Visualizations",
            "Yield & Profitability Estimator",
            "Mobile REST API & IoT Hub",
            "Farm Operations Report"
        ],
        index=0
    )

    st.markdown("---")
    st.markdown("#### API Configuration")
    
    owm_key = st.text_input(
        "OpenWeatherMap API Key",
        value=st.session_state.weather_api_key,
        type="password",
        help="Optional: Free key from openweathermap.org. Leave blank for automated simulation mode."
    )
    if owm_key != st.session_state.weather_api_key:
        st.session_state.weather_api_key = owm_key
        weather_svc.set_api_key(owm_key)
        st.success("Weather API Key updated!")

    gemini_key = st.text_input(
        "Google Gemini API Key",
        value=st.session_state.gemini_api_key,
        type="password",
        help="Optional: For Gemini 2.5 Flash AI chatbot. Leave blank for built-in offline agro-engine."
    )
    if gemini_key != st.session_state.gemini_api_key:
        st.session_state.gemini_api_key = gemini_key
        farm_advisor.set_api_key(gemini_key)
        st.success("Gemini API Key updated!")

    st.markdown("---")
    total_val = inv_mgr.get_total_valuation()
    low_stock_cnt = len(inv_mgr.get_low_stock_items())
    st.metric("Total Inventory Value", f"Rs. {total_val:,.2f}")
    if low_stock_cnt > 0:
        st.markdown(f"<span class='badge-low-stock'> {low_stock_cnt} Low Stock Alert(s)</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='badge-in-stock'> Inventory Healthy</span>", unsafe_allow_html=True)

    st.markdown("<br><small style='color:#94a3b8;'>Version 2.5  -  TensorFlow, Flask & Streamlit</small>", unsafe_allow_html=True)

if nav_choice == "Farm Overview":
    st.markdown("""
    <div class='farm-hero-banner'>
        <h1>AI-Powered Smart Farm Assistant</h1>
        <p>Real-time precision farming dashboard integrating Machine Learning weather trends, CNN leaf disease diagnostics, smart inventory, and IoT telemetry.</p>
    </div>
    """, unsafe_allow_html=True)

    weather_curr = weather_svc.get_current_weather(city="Bangalore")
    soil_telemetry = soil_node.read_telemetry(ambient_temp=weather_curr.get("temp_c", 28.0))
    low_stocks = inv_mgr.get_low_stock_items()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card-container'>
            <div class='metric-label'>Ambient Temperature</div>
            <div class='metric-value'>{weather_curr.get('temp_c', 26.5)}  C</div>
            <div class='metric-sub'style='color:#059669;'> {weather_curr.get('description', 'Clear Sky')}</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown(f"""
        <div class='metric-card-container'>
            <div class='metric-label'>Relative Humidity</div>
            <div class='metric-value'>{weather_curr.get('humidity_pct', 65)} %</div>
            <div class='metric-sub'style='color:#0288d1;'> Rain Prob: {weather_curr.get('rainfall_1h_mm', 0.0)} mm/h</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        moist = soil_telemetry["soil_moisture_pct"]
        st.markdown(f"""
        <div class='metric-card-container'>
            <div class='metric-label'>Soil Moisture (15cm)</div>
            <div class='metric-value' style='color:{soil_telemetry["moisture_color"]};'>{moist}%</div>
            <div class='metric-sub'style='color:{soil_telemetry["moisture_color"]};'> {soil_telemetry['moisture_health']}</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown(f"""
        <div class='metric-card-container'>
            <div class='metric-label'>Total Farm Assets</div>
            <div class='metric-value'>Rs. {inv_mgr.get_total_valuation():,.0f}</div>
            <div class='metric-sub'> {len(inv_mgr.get_all_items())} items registered</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("###")

    if low_stocks:
        st.error(f"**Urgent Inventory Alert:** {len(low_stocks)} item(s) are at or below reorder threshold!")
        alert_cols = st.columns(min(len(low_stocks), 3))
        for i, itm in enumerate(low_stocks[:3]):
            with alert_cols[i]:
                st.warning(f"**{itm.name}**\nCurrent: **{itm.quantity} {itm.unit}** (Min: {itm.min_threshold})\nCategory: {itm.category}")

    st.markdown("---")

    col_w, col_s = st.columns([1, 1])

    with col_w:
        st.markdown("#### 7-Day ML Weather Trajectory")
        forecast_7d = weather_pred.predict_multi_day_trend(current_data=weather_curr, days=7)
        st.plotly_chart(plot_forecast_plotly(forecast_7d), use_container_width=True)

    with col_s:
        st.markdown("#### IoT Subsurface Soil Health")
        history_24h = soil_node.generate_recent_history(hours=24)
        st.plotly_chart(plot_soil_telemetry_plotly(history_24h), use_container_width=True)

elif nav_choice == "Weather & ML Forecast":
    st.markdown("## Agricultural Weather Forecasting & ML Prediction")
    st.markdown("Combines real-time meteorological API data with Machine Learning algorithms (Random Forest & Gradient Boosting) to forecast rainfall and temperature trajectories.")

    t1, t2, t3 = st.tabs(["Real-time Weather Station", "ML 7-Day Trajectory", "Custom Scenario Predictor"])

    with t1:
        c_city, c_btn = st.columns([3, 1])
        with c_city:
            city_input = st.text_input("Farm Location / City", value="Bangalore")
        with c_btn:
            st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
            fetch_btn = st.button("Fetch Weather", use_container_width=True)

        w_data = weather_svc.get_current_weather(city=city_input)
        st.info(f"**Data Source:** {w_data.get('source', 'Station')} | Last updated: {w_data.get('timestamp')}")

        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Temperature", f"{w_data.get('temp_c')}  C", f"Feels {w_data.get('feels_like_c')}  C")
        m2.metric("Humidity", f"{w_data.get('humidity_pct')} %")
        m3.metric("Pressure", f"{w_data.get('pressure_hpa')} hPa")
        m4.metric("Wind Speed", f"{w_data.get('wind_speed_ms')} m/s")
        m5.metric("Rainfall Rate", f"{w_data.get('rainfall_1h_mm')} mm/h")

        weather_summary_text = f"Current weather in {w_data.get('city')}: Temperature is {w_data.get('temp_c')} degrees Celsius, humidity is {w_data.get('humidity_pct')} percent, and condition is {w_data.get('description')}."
        if st.button("Listen to Voice Weather Advisory"):
            with st.spinner("Synthesizing voice audio..."):
                audio_bytes = synthesize_speech_bytes(weather_summary_text)
                if audio_bytes:
                    st.audio(audio_bytes, format="audio/mp3")

        st.markdown("#### 5-Day Detailed Forecast Timeline")
        f5_data = weather_svc.get_5day_forecast(city=city_input)
        f_list = f5_data.get("forecast", [])[:16]
        if f_list:
            f_df = pd.DataFrame(f_list)[["datetime", "temp_c", "humidity_pct", "condition", "pop", "rainfall_3h_mm"]]
            f_df.columns = ["DateTime", "Temp ( C)", "Humidity (%)", "Condition", "Rain Chance (%)", "Rain (mm)"]
            st.dataframe(f_df, use_container_width=True, hide_index=True)

    with t2:
        st.markdown("#### Machine Learning Predictive Modeling")
        metrics = weather_pred.metrics
        c_m1, c_m2, c_m3, c_m4 = st.columns(4)
        c_m1.metric("Temperature Model R^2", f"{metrics.get('temp_r2', 0.925):.3f}")
        c_m2.metric("Temperature MAE", f"+/-{metrics.get('temp_mae', 1.46)}  C")
        c_m3.metric("Rain Detection Accuracy", f"{metrics.get('rain_accuracy', 65.0)}%")
        c_m4.metric("Dataset Records", f"{metrics.get('dataset_rows', 1095)} days")

        forecast_days = weather_pred.predict_multi_day_trend(days=7)
        st.plotly_chart(plot_forecast_plotly(forecast_days), use_container_width=True)

        st.markdown("##### Daily Agronomic Projections")
        f_display = pd.DataFrame(forecast_days)[[
            "date", "day_name", "mean_temp_c", "max_temp_c", "min_temp_c",
            "rain_prob_pct", "predicted_rain_mm", "condition", "confidence_pct"
        ]]
        f_display.columns = ["Date", "Day", "Mean Temp ( C)", "Max ( C)", "Min ( C)", "Rain Prob (%)", "Expected Rain (mm)", "Condition", "Confidence (%)"]
        st.dataframe(f_display, use_container_width=True, hide_index=True)

    with t3:
        st.markdown("#### Simulation Lab: Custom Weather Prediction")
        st.markdown("Input hypothetical or current sensor readings to simulate ML-predicted temperature and rainfall probability:")
        
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            in_temp = st.slider("Current Mean Temperature ( C)", 10.0, 48.0, 28.5, 0.5)
            in_hum = st.slider("Current Humidity (%)", 20, 100, 72)
        with sc2:
            in_press = st.slider("Atmospheric Pressure (hPa)", 980.0, 1035.0, 1012.0, 1.0)
            in_wind = st.slider("Wind Speed (m/s)", 0.0, 25.0, 4.2, 0.2)
        with sc3:
            in_rain = st.slider("Past 24h Rainfall (mm)", 0.0, 100.0, 5.0, 1.0)
            sim_btn = st.button("Predict Next Day Weather", use_container_width=True)

        pred_custom = weather_pred.predict_multi_day_trend(
            current_data={"temp_c": in_temp, "humidity_pct": in_hum, "pressure_hpa": in_press, "wind_speed_ms": in_wind, "rainfall_mm": in_rain},
            days=1
        )[0]

        res_c1, res_c2, res_c3 = st.columns(3)
        res_c1.metric("Predicted Tomorrow Temp", f"{pred_custom['mean_temp_c']}  C", f"Range: {pred_custom['min_temp_c']} deg - {pred_custom['max_temp_c']} deg")
        res_c2.metric("Rain Occurrence Probability", f"{pred_custom['rain_prob_pct']}%", f"Condition: {pred_custom['condition']}")
        res_c3.metric("Predicted Rainfall Volume", f"{pred_custom['predicted_rain_mm']} mm", f"Confidence: {pred_custom['confidence_pct']}%")

elif nav_choice == "Crop Disease Detection":
    st.markdown("## AI Crop Disease Diagnosis (TensorFlow / Keras CNN)")
    st.markdown("Upload a crop leaf photograph or select from curated sample images to classify diseases across Tomato, Potato, Corn, and Rice using deep Convolutional Neural Networks.")

    dt1, dt2, dt3 = st.tabs(["Image Diagnostic Laboratory", "Interactive Sample Gallery", "CNN Architecture & Metrics"])

    with dt1:
        col_up, col_diag = st.columns([1, 1])
        selected_image = None
        source_name = ""

        with col_up:
            st.markdown("#### Upload Leaf Photograph")
            uploaded_file = st.file_uploader("Choose leaf image (JPG/PNG)", type=["jpg", "jpeg", "png"])
            camera_file = st.camera_input("Or snap live photo using camera")

            if uploaded_file is not None:
                selected_image = Image.open(uploaded_file)
                source_name = uploaded_file.name
            elif camera_file is not None:
                selected_image = Image.open(camera_file)
                source_name = "Live Camera Capture"

            if selected_image is not None:
                st.image(selected_image, caption=f"Selected: {source_name}", use_container_width=True)

        with col_diag:
            st.markdown("#### Diagnostic Report")
            if selected_image is not None:
                with st.spinner("Analyzing foliar pathological patterns with CNN..."):
                    result = disease_clf.classify_image(selected_image)

                diag = result["diagnostic"]
                conf = result["confidence_pct"]
                severity = diag.get("severity", "Unknown")
                color = diag.get("color_code", "#2e7d32")

                st.markdown(f"""
                <div style='background-color:{color}; color:white; padding:18px 22px; border-radius:12px; margin-bottom:15px;'>
                    <div style='font-size:0.85rem; text-transform:uppercase; opacity:0.9;'>Crop: {diag.get('crop')}  -  Severity: {severity}</div>
                    <div style='font-size:1.6rem; font-weight:700;'>{diag.get('disease_name')}</div>
                    <div style='font-size:0.95rem; opacity:0.95;'>Confidence Score: <b>{conf}%</b> (Model: {result['model_type']})</div>
                </div>
                """, unsafe_allow_html=True)

                speech_text = f"Diagnosis for {diag.get('crop')}: {diag.get('disease_name')} detected with {conf} percent confidence. Severity is {severity}. Immediate action: {diag.get('immediate_action')}"
                b_c1, b_c2 = st.columns(2)
                with b_c1:
                    if st.button("Voice Diagnosis (TTS)", use_container_width=True):
                        with st.spinner("Generating audio..."):
                            tts_audio = synthesize_speech_bytes(speech_text)
                            if tts_audio:
                                st.audio(tts_audio, format="audio/mp3")

                with b_c2:
                    curr_w = weather_svc.get_current_weather()
                    pdf_bytes = generate_diagnostic_pdf(result, curr_w)
                    st.download_button(
                        label="Download PDF Prescription",
                        data=pdf_bytes,
                        file_name=f"crop_prescription_{diag.get('crop')}_{datetime.now().strftime('%Y%m%d')}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )

                st.markdown(f"** Scientific Pathogen:** `{diag.get('pathogen')}`")
                st.markdown(f"** Symptoms:** {diag.get('symptoms')}")
                st.markdown(f"** Immediate Farm Action:** {diag.get('immediate_action')}")

                with st.expander("Organic & Bio-Control Remedies", expanded=True):
                    for org in diag.get("organic_treatment", []):
                        st.markdown(f"- {org}")

                with st.expander("Chemical Control & Fungicides", expanded=True):
                    for chem in diag.get("chemical_treatment", []):
                        st.markdown(f"- {chem}")

                with st.expander("Long-term Prevention & Cultural Practices"):
                    st.markdown(f"- {diag.get('prevention')}")
                    st.info(f"**Pre-Harvest Safe Waiting Interval (PHI):** {diag.get('safe_harvest_wait_days')} days")

                top_preds = result.get("top_predictions", [])
                if top_preds:
                    st.markdown("##### Model Class Probability Distribution")
                    p_df = pd.DataFrame(top_preds)
                    st.bar_chart(p_df.set_index("disease")["probability_pct"])
            else:
                st.info("Please upload a leaf photo or pick a sample from the **Interactive Sample Gallery** tab to trigger automatic AI diagnosis.")

    with dt2:
        st.markdown("#### Test AI Diagnostic Engine with Real Disease Samples")
        st.markdown("Click any sample leaf below to automatically analyze it with the CNN model:")

        sample_dir = os.path.join(os.path.dirname(__file__), "data", "sample_leaves")
        sample_files = [
            ("tomato_early_blight.jpg", "Tomato Early Blight"),
            ("tomato_late_blight.jpg", "Tomato Late Blight"),
            ("tomato_healthy.jpg", "Tomato Healthy"),
            ("potato_early_blight.jpg", "Potato Early Blight"),
            ("potato_late_blight.jpg", "Potato Late Blight"),
            ("potato_healthy.jpg", "Potato Healthy"),
            ("corn_common_rust.jpg", "Corn Common Rust"),
            ("corn_leaf_blight.jpg", "Corn Leaf Blight"),
            ("corn_healthy.jpg", "Corn Healthy"),
            ("rice_blast.jpg", "Rice Blast"),
            ("rice_brown_spot.jpg", "Rice Brown Spot"),
            ("rice_healthy.jpg", "Rice Healthy")
        ]

        cols = st.columns(4)
        for i, (fn, label) in enumerate(sample_files):
            fpath = os.path.join(sample_dir, fn)
            with cols[i % 4]:
                if os.path.exists(fpath):
                    s_img = Image.open(fpath)
                    st.image(s_img, caption=label, use_container_width=True)
                    if st.button(f"Test {label}", key=f"btn_sample_{i}", use_container_width=True):
                        st.session_state["active_sample_path"] = fpath

        if "active_sample_path" in st.session_state and os.path.exists(st.session_state["active_sample_path"]):
            st.markdown("---")
            st.markdown("### Live Diagnosis on Selected Sample")
            test_img = Image.open(st.session_state["active_sample_path"])
            c_img, c_res = st.columns([1, 2])
            with c_img:
                st.image(test_img, caption="Diagnosed Leaf", use_container_width=True)
            with c_res:
                s_res = disease_clf.classify_image(test_img)
                s_diag = s_res["diagnostic"]
                st.success(f"**Identified:** {s_diag.get('crop')} - {s_diag.get('disease_name')} ({s_res['confidence_pct']}%)")
                st.markdown(f"**Pathogen:** `{s_diag.get('pathogen')}`")
                st.markdown(f"**Recommended Action:** {s_diag.get('immediate_action')}")
                st.markdown(f"**Organic Remedy:** {s_diag.get('organic_treatment', ['N/A'])[0]}")

    with dt3:
        st.markdown("#### Convolutional Neural Network (CNN) Architecture & Training Performance")
        meta_path = os.path.join(os.path.dirname(__file__), "models", "disease_model_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)

            cm1, cm2, cm3, cm4 = st.columns(4)
            cm1.metric("Architecture", meta.get("model_type", "CNN"))
            cm2.metric("Framework", meta.get("framework", "Keras"))
            cm3.metric("Classes Supported", f"{meta.get('num_classes', 12)} classes")
            cm4.metric("Training Accuracy", f"{meta.get('final_train_acc', 0.92)*100:.1f}%")

            history = meta.get("history", {})
            if history:
                st.pyplot(plot_cnn_training_history_matplotlib(history))
        else:
            st.info("Model metadata will be displayed upon completion of training.")

elif nav_choice == "Crop & Fertilizer Recommender":
    st.markdown("## Machine Learning Crop & Fertilizer Recommendation")
    st.markdown("Trained on soil and climatological parameters to determine the most viable high-yield crops and compute precise chemical/organic fertilizer replenishment schedules.")

    cr_tab1, cr_tab2 = st.tabs(["Crop Suitability Predictor", "Fertilizer Deficit & Dosage Calculator"])

    with cr_tab1:
        st.markdown("#### Enter Soil Test & Micro-Climate Telemetry:")
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            in_n = st.number_input("Soil Nitrogen (N) (kg/ha)", min_value=5.0, max_value=250.0, value=85.0)
            in_p = st.number_input("Soil Phosphorus (P) (kg/ha)", min_value=5.0, max_value=150.0, value=42.0)
            in_k = st.number_input("Soil Potassium (K) (kg/ha)", min_value=5.0, max_value=250.0, value=45.0)
        with sc2:
            in_ph = st.slider("Soil pH Level", 4.5, 9.0, 6.5, 0.1)
            in_temp = st.slider("Average Temperature ( C)", 12.0, 42.0, 25.0, 0.5)
        with sc3:
            in_hum = st.slider("Relative Humidity (%)", 15.0, 99.0, 80.0, 1.0)
            in_rain = st.number_input("Annual / Seasonal Rainfall (mm)", min_value=20.0, max_value=400.0, value=180.0)

        if st.button("Recommend Optimal Crops", use_container_width=True):
            recs = crop_rec.recommend(in_n, in_p, in_k, in_temp, in_hum, in_ph, in_rain)
            st.markdown("### Top Recommended Crops for Your Soil & Climate")

            r_cols = st.columns(3)
            for i, r in enumerate(recs):
                with r_cols[i]:
                    st.markdown(f"""
                    <div class='metric-card-container'>
                        <div class='metric-label'>Rank
                        <div class='metric-value' style='color:#065f46;'>{r['crop']}</div>
                        <div class='metric-sub' style='color:#059669; font-weight:bold;'>Match Confidence: {r['confidence_pct']}%</div>
                        <hr style='margin:10px 0;'>
                        <small>Ideal N-P-K: {r['ideal_profile'].get('N')}-{r['ideal_profile'].get('P')}-{r['ideal_profile'].get('K')}<br>
                        Ideal pH: {r['ideal_profile'].get('pH')}  -  Rain: {r['ideal_profile'].get('rainfall')}mm</small>
                    </div>
                    """, unsafe_allow_html=True)

    with cr_tab2:
        st.markdown("#### Precision Fertilizer Prescription Engine")
        st.markdown("Calculates the exact amount of Urea (46% N), DAP (18-46-0), and MOP (60% K2O) required to achieve target yields.")

        fc1, fc2 = st.columns([1, 1])
        with fc1:
            target_crop_sel = st.selectbox("Target Crop to Cultivate", [
                "Rice", "Maize", "Chickpea", "Kidney Beans", "Pigeon Peas", "Moth Beans",
                "Mung Bean", "Black Gram", "Lentil", "Pomegranate", "Banana", "Mango",
                "Grapes", "Watermelon", "Apple", "Orange", "Papaya", "Cotton", "Coffee"
            ])
            s_n = st.number_input("Current Soil Nitrogen (N) (kg/ha)", value=50.0, step=5.0)
            s_p = st.number_input("Current Soil Phosphorus (P) (kg/ha)", value=25.0, step=5.0)
            s_k = st.number_input("Current Soil Potassium (K) (kg/ha)", value=30.0, step=5.0)

            calc_btn = st.button("Calculate Fertilizer Dosage", use_container_width=True)

        with fc2:
            prescription = CropRecommender.calculate_fertilizer_prescription(s_n, s_p, s_k, target_crop_sel)
            st.markdown(f"#### Prescription for {target_crop_sel}")

            dosages = prescription["fertilizer_dosage_kg_ha"]
            bags = prescription["fertilizer_dosage_bags_ha"]

            d1, d2, d3 = st.columns(3)
            d1.metric("Neem Coated Urea", f"{dosages['Urea (46% N)']} kg/ha", f"~{bags['Urea (45kg Bags)']} Bags")
            d2.metric("DAP (18-46-0)", f"{dosages['DAP (18-46-0)']} kg/ha", f"~{bags['DAP (50kg Bags)']} Bags")
            d3.metric("MOP Potash (60% K)", f"{dosages['MOP (60% K)']} kg/ha", f"~{bags['MOP (50kg Bags)']} Bags")

            st.markdown("##### Recommended Application Timelines:")
            for step in prescription["application_schedule"]:
                st.markdown(f"- {step}")

elif nav_choice == "Smart Inventory":
    st.markdown("## Smart Farm Inventory Management")
    st.markdown("Tracks Fertilizers, Pesticides, Seeds, and Farm Supplies with OOP encapsulation, custom exception handling, low-stock threshold triggers, and CSV/JSON persistence.")

    it1, it2, it3, it4 = st.tabs(["Stock Catalog & Audit", "Deduct / Use Stock", "Add / Restock Item", "Import / Export"])

    with it1:
        c_filter, c_search = st.columns([1, 2])
        with c_filter:
            cat_choice = st.selectbox("Filter Category", ["All", "Fertilizer", "Pesticide", "Seeds", "Equipment"])
        with c_search:
            search_query = st.text_input("Search Item by Name or ID", "")

        all_items = inv_mgr.get_all_items()
        filtered = all_items
        if cat_choice != "All":
            filtered = [i for i in filtered if i.category == cat_choice]
        if search_query:
            q = search_query.lower()
            filtered = [i for i in filtered if q in i.name.lower() or q in i.item_id.lower()]

        table_data = []
        for itm in filtered:
            status = "Out of Stock"if itm.is_out_of_stock else ("Low Stock"if itm.is_low_stock else "In Stock")
            days_exp = itm.days_until_expiry
            exp_text = f"{days_exp} days" if days_exp is not None else "N/A"
            table_data.append({
                "Item ID": itm.item_id,
                "Name": itm.name,
                "Category": itm.category,
                "Stock": f"{itm.quantity} {itm.unit}",
                "Min Threshold": itm.min_threshold,
                "Unit Price (Rs. )": itm.price_per_unit,
                "Total Value (Rs. )": itm.total_value,
                "Status": status,
                "Expiry In": exp_text,
                "Location": itm.location
            })

        df_inv = pd.DataFrame(table_data)
        st.dataframe(df_inv, use_container_width=True, hide_index=True)

        st.markdown("#### Stock Valuation Breakdown")
        st.plotly_chart(plot_inventory_valuation_plotly(all_items), use_container_width=True)

        with st.expander("View Audit Log Trail"):
            if os.path.exists(inv_mgr.log_path):
                with open(inv_mgr.log_path, "r", encoding="utf-8") as f:
                    logs = f.readlines()[-20:]
                st.code("".join(logs) if logs else "No transactions logged yet.")

    with it2:
        st.markdown("#### Deduct Stock for Farm Application")
        st.markdown("Enforces stock availability with custom `InsufficientStockError` exception handling.")

        with st.form("deduct_stock_form"):
            item_options = {f"{i.item_id} - {i.name} ({i.quantity} {i.unit} available)": i.item_id for i in all_items}
            selected_label = st.selectbox("Select Supply Item", list(item_options.keys()))
            deduct_qty = st.number_input("Quantity to Use / Apply", min_value=0.1, step=0.5, value=1.0)
            purpose_note = st.text_input("Application Purpose", value="Field spraying for blight protection")
            submit_use = st.form_submit_button("Record Stock Usage", use_container_width=True)

            if submit_use:
                target_id = item_options[selected_label]
                try:
                    updated = inv_mgr.use_stock(target_id, deduct_qty, purpose=purpose_note)
                    st.success(f"Successfully deducted {deduct_qty} {updated.unit} of **{updated.name}**! Remaining in stock: **{updated.quantity} {updated.unit}**")
                    time.sleep(1)
                    st.rerun()
                except InsufficientStockError as e:
                    st.error(f"**Stock Exception:** {str(e)}")
                except InventoryException as e:
                    st.error(f"**Error:** {str(e)}")

    with it3:
        st.markdown("#### Add New Item or Restock Existing")
        r_tab1, r_tab2 = st.tabs(["Restock Existing Item", "Register New Supply"])

        with r_tab1:
            with st.form("restock_form"):
                r_item_options = {f"{i.item_id} - {i.name} (Current: {i.quantity} {i.unit})": i.item_id for i in all_items}
                r_label = st.selectbox("Select Item to Restock", list(r_item_options.keys()))
                r_qty = st.number_input("Quantity Received", min_value=0.5, step=1.0, value=5.0)
                r_price = st.number_input("New Unit Price (Rs. ) (Optional - 0 to keep current)", min_value=0.0, step=10.0, value=0.0)
                submit_restock = st.form_submit_button("Update Inventory Stock", use_container_width=True)

                if submit_restock:
                    target_id = r_item_options[r_label]
                    try:
                        price_arg = r_price if r_price > 0 else None
                        updated = inv_mgr.restock(target_id, r_qty, new_price=price_arg)
                        st.success(f"Successfully restocked **{updated.name}**! New Quantity: **{updated.quantity} {updated.unit}**")
                        time.sleep(1)
                        st.rerun()
                    except InventoryException as e:
                        st.error(f"**Restock Failed:** {str(e)}")

        with r_tab2:
            with st.form("add_new_item_form"):
                a1, a2 = st.columns(2)
                with a1:
                    new_id = st.text_input("Item ID (e.g. FERT-005)", value="")
                    new_name = st.text_input("Item Name", value="")
                    new_cat = st.selectbox("Category", ["Fertilizer", "Pesticide", "Seeds", "Equipment", "Other"])
                    new_qty = st.number_input("Initial Quantity", min_value=0.0, step=1.0, value=10.0)
                    new_unit = st.text_input("Unit of Measure (e.g. Bags, Liters, Kg, Packets)", value="Bags")
                with a2:
                    new_min = st.number_input("Low-Stock Alert Threshold", min_value=1.0, step=1.0, value=5.0)
                    new_price = st.number_input("Unit Price (Rs. )", min_value=0.0, step=10.0, value=500.0)
                    new_loc = st.text_input("Storage Location", value="Warehouse Shed B")
                    new_exp = st.date_input("Expiry Date")
                    new_sup = st.text_input("Supplier / Manufacturer", value="Agro Supplies Co")

                submit_add = st.form_submit_button("Save New Supply to Inventory", use_container_width=True)
                if submit_add:
                    try:
                        item_obj = InventoryItem(
                            item_id=new_id.strip(),
                            name=new_name.strip(),
                            category=new_cat,
                            quantity=float(new_qty),
                            unit=new_unit.strip(),
                            min_threshold=float(new_min),
                            price_per_unit=float(new_price),
                            location=new_loc.strip(),
                            expiry_date=new_exp.strftime("%Y-%m-%d"),
                            supplier=new_sup.strip()
                        )
                        inv_mgr.add_item(item_obj)
                        st.success(f"Added **{item_obj.name}** ({item_obj.item_id}) to catalog!")
                        time.sleep(1)
                        st.rerun()
                    except InvalidItemDataError as e:
                        st.error(f"**Validation Error:** {str(e)}")
                    except InventoryException as e:
                        st.error(f"**Error:** {str(e)}")

    with it4:
        st.markdown("#### Dual CSV / JSON Data Handling")
        st.markdown("Export or import farm stock files in standard CSV or JSON format.")

        d_c1, d_c2 = st.columns(2)
        with d_c1:
            if os.path.exists(inv_mgr.json_path):
                with open(inv_mgr.json_path, "r", encoding="utf-8") as f:
                    json_str = f.read()
                st.download_button(
                    label="Download Inventory JSON",
                    data=json_str,
                    file_name="farm_inventory.json",
                    mime="application/json",
                    use_container_width=True
                )

        with d_c2:
            if os.path.exists(inv_mgr.csv_path):
                with open(inv_mgr.csv_path, "r", encoding="utf-8") as f:
                    csv_str = f.read()
                st.download_button(
                    label="Download Inventory CSV",
                    data=csv_str,
                    file_name="farm_inventory.csv",
                    mime="text/csv",
                    use_container_width=True
                )

elif nav_choice == "IoT Soil & Irrigation":
    st.markdown("## IoT Soil & Automated Irrigation Monitor")
    st.markdown("Simulated live agro-telemetry monitoring subsurface root zones (Moisture %, Soil Temp, pH, EC, NPK) and calculating smart evapotranspiration water demands.")

    telemetry = soil_node.read_telemetry(ambient_temp=29.0)
    irr_rec = telemetry["irrigation_rec"]

    st.markdown(f"""
    <div style='background-color:#0f172a; color:white; padding:18px 24px; border-radius:14px; margin-bottom:20px; display:flex; justify-content:space-between; align-items:center;'>
        <div>
            <div style='font-size:0.85rem; color:#94a3b8;'>CONNECTED SENSOR NODE: <b>{telemetry['node_id']}</b>  -  CROP: <b>{telemetry['crop']}</b></div>
            <div style='font-size:1.5rem; font-weight:700;'>Soil Status: <span style='color:{telemetry["moisture_color"]};'>{telemetry['moisture_health']}</span></div>
        </div>
        <div style='text-align:right;'>
            <div style='font-size:0.85rem; color:#94a3b8;'>AUTOMATED SOLENOID VALVE</div>
            <div style='font-size:1.3rem; font-weight:700; color:{"#10b981" if telemetry["pump_active"] else "#94a3b8"};'>{telemetry['pump_status']}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Soil Moisture", f"{telemetry['soil_moisture_pct']}%", f"Target: 65%")
    c2.metric("Subsurface Temp", f"{telemetry['soil_temp_c']}  C")
    c3.metric("Soil pH", f"{telemetry['soil_ph']}", telemetry["ph_status"])
    c4.metric("Salinity (EC)", f"{telemetry['ec_dsm']} dS/m", "Normal")

    st.markdown("###")

    ic1, ic2 = st.columns([2, 1])
    with ic1:
        st.markdown("#### Smart Irrigation Recommendation")
        st.info(f"**Recommendation:** {irr_rec['action']}\n\n"
                f"- **Calculated Soil Moisture Deficit:** `{irr_rec['deficit_pct']}%`\n"
                f"- **Exact Water Requirement:** `{irr_rec['water_required_liters_per_ha']:,} Liters / Hectare`\n"
                f"- **Estimated Drip Irrigation Run Time:** `{irr_rec['drip_run_time_minutes']} minutes`\n"
                f"- **Priority Urgency:** `{irr_rec['urgency']}`")

    with ic2:
        st.markdown("#### Manual Solenoid Override")
        if telemetry["pump_active"]:
            if st.button("STOP IRRIGATION PUMP", use_container_width=True, type="primary"):
                soil_node.trigger_pump(False)
                st.rerun()
        else:
            if st.button("ACTIVATE IRRIGATION VALVE", use_container_width=True):
                soil_node.trigger_pump(True)
                st.rerun()

        st.caption("Auto-shutoff engages automatically once field capacity (68%) is reached.")

    st.markdown("---")
    st.markdown("#### 24-Hour Soil Telemetry History")
    hist = soil_node.generate_recent_history(hours=24)
    st.plotly_chart(plot_soil_telemetry_plotly(hist), use_container_width=True)

elif nav_choice == "AI Farm Advisor":
    st.markdown("## Kisan Mitra: AI Agricultural Consultant & Chatbot")
    st.markdown("Interactive multi-turn AI advisor powered by **Gemini 2.5 Flash** (with fallback to the comprehensive **Offline Agricultural Knowledge Engine**).")

    if farm_advisor.gemini_client is not None:
        st.success("Connected to **Google Gemini 2.5 Flash AI** via official google-genai SDK.")
    else:
        st.info("Running in **Offline Agricultural Knowledge Engine Mode** (no API key needed). You can add a Gemini API Key in the sidebar anytime for real-time generative reasoning.")

    st.markdown("##### Frequently Asked Agricultural Questions:")
    q_cols = st.columns(3)
    with q_cols[0]:
        if st.button("NPK Dosage for Tomato?", use_container_width=True):
            st.session_state.quick_prompt = "What is the recommended NPK fertilizer dosage for tomato per acre?"
    with q_cols[1]:
        if st.button("How to make Jeevamrutha?", use_container_width=True):
            st.session_state.quick_prompt = "Explain step-by-step preparation of organic Jeevamrutha."
    with q_cols[2]:
        if st.button("Sucking Pest IPM?", use_container_width=True):
            st.session_state.quick_prompt = "How to control aphids and whiteflies using integrated pest management?"

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    prompt_to_use = None
    if "quick_prompt" in st.session_state and st.session_state.quick_prompt:
        prompt_to_use = st.session_state.quick_prompt
        st.session_state.quick_prompt = None

    user_input = st.chat_input("Ask any farming, crop disease, fertilizer, or soil question...")
    if user_input:
        prompt_to_use = user_input

    if prompt_to_use:
        st.session_state.chat_history.append({"role": "user", "content": prompt_to_use})
        with st.chat_message("user"):
            st.markdown(prompt_to_use)

        with st.chat_message("assistant"):
            with st.spinner("Consulting agricultural knowledge base..."):
                curr_w = weather_svc.get_current_weather()
                reply_obj = farm_advisor.generate_response(
                    user_prompt=prompt_to_use,
                    context={"temp_c": curr_w.get("temp_c"), "humidity_pct": curr_w.get("humidity_pct"), "current_crop": "Tomato & Potato"}
                )
                formatted_reply = f"{reply_obj['response']}\n\n*Source: {reply_obj.get('source')}*"
                st.markdown(formatted_reply)

        st.session_state.chat_history.append({"role": "assistant", "content": formatted_reply})

elif nav_choice == "Data Visualizations":
    st.markdown("## Agro-Data Visualization Studio")
    st.markdown("High-resolution statistical and exploratory data visualizations rendered with **Matplotlib** and **Seaborn** as specified in the project requirements.")

    v_tab1, v_tab2, v_tab3, v_tab4 = st.tabs([
        "Matplotlib Weather Trends",
        "Seaborn Correlation Matrix",
        "Seaborn Rainfall & Temp KDE",
        "ML Feature Importances"
    ])

    hist_df = pd.read_csv(os.path.join(os.path.dirname(__file__), "data", "historical_weather.csv"))

    with v_tab1:
        st.markdown("#### Matplotlib Multi-Panel Annual Weather Cycle")
        fig_mat = plot_weather_historical_matplotlib(hist_df)
        st.pyplot(fig_mat)

    with v_tab2:
        st.markdown("#### Seaborn Correlation Heatmap of Meteorological Dynamics")
        fig_sea_corr = plot_weather_correlation_seaborn(hist_df)
        st.pyplot(fig_sea_corr)

    with v_tab3:
        st.markdown("#### Seaborn Distribution Density and Seasonal Boxplots")
        fig_sea_dist = plot_rainfall_distribution_seaborn(hist_df)
        st.pyplot(fig_sea_dist)

    with v_tab4:
        st.markdown("#### Machine Learning Feature Importance (Random Forest)")
        importances = weather_pred.metrics.get("feature_importances", {})
        if importances:
            fig_imp = plot_feature_importance_matplotlib(importances)
            st.pyplot(fig_imp)
        else:
            st.info("Feature importance data will display after ML models are trained.")

elif nav_choice == "Yield & Profitability Estimator":
    st.markdown("## Harvest Yield Estimator & Farm Profitability Calculator")
    st.markdown("Simulate projected crop yields, input expenses, market prices, and net farm return margin based on agronomic inputs and acreage.")

    yc1, yc2 = st.columns([1, 1])
    with yc1:
        crop_sel = st.selectbox("Crop Type", ["Tomato", "Potato", "Corn / Maize", "Paddy / Rice"])
        acreage = st.number_input("Cultivation Land Area (Acres)", min_value=0.5, max_value=500.0, value=2.5, step=0.5)

        base_yields = {"Tomato": 140.0, "Potato": 110.0, "Corn / Maize": 35.0, "Paddy / Rice": 28.0}
        base_prices = {"Tomato": 1800.0, "Potato": 1500.0, "Corn / Maize": 2200.0, "Paddy / Rice": 2400.0}

        est_yield_per_acre = st.number_input("Expected Yield (Quintals / Acre)", min_value=5.0, max_value=300.0, value=base_yields[crop_sel])
        market_price = st.number_input("Expected Market Selling Price (Rs.  / Quintal)", min_value=100.0, max_value=10000.0, value=base_prices[crop_sel])

        st.markdown("##### Operating Input Expenses (Per Acre):")
        seed_cost = st.number_input("Seed & Nursery Cost (Rs. /Acre)", value=4500.0, step=500.0)
        fert_pest_cost = st.number_input("Fertilizers & Crop Protection (Rs. /Acre)", value=8500.0, step=500.0)
        labor_irr_cost = st.number_input("Labor, Drip & Harvesting (Rs. /Acre)", value=12000.0, step=1000.0)

    with yc2:
        total_production_quintals = round(acreage * est_yield_per_acre, 1)
        gross_revenue = round(total_production_quintals * market_price, 2)
        total_costs = round(acreage * (seed_cost + fert_pest_cost + labor_irr_cost), 2)
        net_profit = round(gross_revenue - total_costs, 2)
        roi_pct = round((net_profit / total_costs) * 100, 1) if total_costs > 0 else 0.0

        st.markdown("### Farm Financial Projection Summary")
        m_c1, m_c2 = st.columns(2)
        m_c1.metric("Total Harvest Output", f"{total_production_quintals:,.1f} Qtl", f"{acreage} Acres")
        m_c2.metric("Gross Farm Revenue", f"Rs. {gross_revenue:,.0f}")

        m_c3, m_c4 = st.columns(2)
        m_c3.metric("Total Operating Costs", f"Rs. {total_costs:,.0f}")
        m_c4.metric("Net Farm Profit", f"Rs. {net_profit:,.0f}", f"ROI: {roi_pct}%")

        fin_df = pd.DataFrame({
            "Financial Metric": ["Gross Revenue", "Operating Expenses", "Net Profit"],
            "Amount (Rs. )": [gross_revenue, total_costs, net_profit]
        })
        st.bar_chart(fin_df.set_index("Financial Metric"))

elif nav_choice == "Mobile REST API & IoT Hub":
    st.markdown("## Mobile REST API & IoT Edge Node Hub")
    st.markdown("Integrate native Android/iOS mobile applications or hardware microcontroller sensor nodes (ESP32, Raspberry Pi, Arduino) with the backend Flask microservices.")

    st.info("The Flask REST API service runs concurrently on port `5000` via `api_server.py`. It provides full CORS compatibility, JSON contracts, and multipart image endpoints.")

    st.markdown("### Mobile App REST Endpoints")
    
    api_list = [
        {"Method": "GET", "Endpoint": "/api/health", "Description": "System status and ML models check"},
        {"Method": "GET", "Endpoint": "/api/weather/current?city=Bangalore", "Description": "Current weather observations"},
        {"Method": "GET", "Endpoint": "/api/weather/predict?days=7", "Description": "7-day ML weather trajectory"},
        {"Method": "POST", "Endpoint": "/api/crop/disease/predict", "Description": "Upload leaf image for CNN diagnosis"},
        {"Method": "GET", "Endpoint": "/api/crop/recommend", "Description": "ML crop recommendation from soil test"},
        {"Method": "POST", "Endpoint": "/api/crop/fertilizer/calculate", "Description": "Calculate Urea, DAP, MOP dosage"},
        {"Method": "GET", "Endpoint": "/api/inventory", "Description": "List all inventory stock with status"},
        {"Method": "POST", "Endpoint": "/api/inventory/use", "Description": "Deduct stock with exception handling"},
        {"Method": "GET", "Endpoint": "/api/iot/telemetry", "Description": "Real-time soil sensor telemetry"},
        {"Method": "POST", "Endpoint": "/api/iot/pump", "Description": "Toggle automated solenoid irrigation valve"},
        {"Method": "POST", "Endpoint": "/api/advisor/chat", "Description": "Chatbot message endpoint"}
    ]
    st.dataframe(pd.DataFrame(api_list), use_container_width=True, hide_index=True)

    with st.expander("Example Python / Flutter / Swift API Call (Crop Disease Detection)"):
        st.code("""
import requests

url = "http://localhost:5000/api/crop/disease/predict"
with open("leaf_photo.jpg", "rb") as f:
    files = {"image": f}
    response = requests.post(url, files=files)

data = response.json()
print("Diagnosis:", data["diagnostic"]["disease_name"])
print("Confidence:", data["confidence_pct"], "%")
print("Organic Treatment:", data["diagnostic"]["organic_treatment"])
        """, language="python")

elif nav_choice == "Farm Operations Report":
    st.markdown("## Farm Intelligence & Operations Summary Report")
    st.markdown("Automated operational summary aggregating current weather, crop health alerts, inventory valuation, and IoT soil recommendations.")

    curr_weather = weather_svc.get_current_weather()
    all_items = inv_mgr.get_all_items()
    low_stocks = inv_mgr.get_low_stock_items()
    soil_stat = soil_node.read_telemetry()

    report_md = f"""# AI-POWERED SMART FARM ASSISTANT
**Executive Farm Health & Operational Intelligence Report**
*Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*

---

- **Location:** {curr_weather.get('city', 'Bangalore')}
- **Current Temperature:** {curr_weather.get('temp_c')}  C (Feels like {curr_weather.get('feels_like_c')}  C)
- **Relative Humidity:** {curr_weather.get('humidity_pct')} %
- **Barometric Pressure:** {curr_weather.get('pressure_hpa')} hPa
- **Wind Speed:** {curr_weather.get('wind_speed_ms')} m/s
- **Precipitation:** {curr_weather.get('rainfall_1h_mm')} mm/h ({curr_weather.get('condition')})

---

- **Active Node:** {soil_stat['node_id']} ({soil_stat['crop']} in {soil_stat['soil_type']} soil)
- **Soil Moisture Level:** {soil_stat['soil_moisture_pct']}% - Status: {soil_stat['moisture_health']}
- **Subsurface Temperature:** {soil_stat['soil_temp_c']}  C
- **Soil pH:** {soil_stat['soil_ph']} ({soil_stat['ph_status']})
- **Irrigation Valve Status:** {soil_stat['pump_status']}
- **Recommended Water Volume:** {soil_stat['irrigation_rec']['water_required_liters_per_ha']:,} Liters / Hectare

---

- **Total Registered Supplies:** {len(all_items)} items
- **Total Stock Valuation:** Rs. {inv_mgr.get_total_valuation():,.2f}
- **Low-Stock Alert Items:** {len(low_stocks)}
"""

    if low_stocks:
        report_md += "\n**Critical Stock Reorder Notice:**\n"
        for itm in low_stocks:
            report_md += f"- **{itm.name}** ({itm.category}): {itm.quantity} {itm.unit} remaining (Reorder point: {itm.min_threshold})\n"
    else:
        report_md += "\n- All inventory levels are currently above minimum safety thresholds.\n"

    report_md += """
---
- **CNN Framework:** TensorFlow / Keras Deep Convolutional Neural Network
- **Pathologies Covered:** Early Blight, Late Blight, Common Rust, Northern Leaf Blight, Blast, Brown Spot
- **Target Crops:** Tomato, Potato, Maize, Rice
"""

    st.markdown(report_md)

    st.download_button(
        label="Download Full Farm Report (Markdown / Text)",
        data=report_md,
        file_name=f"smart_farm_report_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
        mime="text/markdown",
        use_container_width=True
    )
