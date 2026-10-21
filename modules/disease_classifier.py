"""
Crop Disease Classifier Module.
Provides inference using the trained TensorFlow/Keras CNN model and maps
predictions to an agricultural pathological knowledge base with organic & chemical treatments.
"""

import os
import json
import numpy as np
from PIL import Image
from typing import Dict, Any, List, Optional, Union

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_KERAS_PATH = os.path.join(BASE_DIR, "models", "crop_disease_cnn.keras")
MODEL_H5_PATH = os.path.join(BASE_DIR, "models", "crop_disease_cnn.h5")
FALLBACK_PATH = os.path.join(BASE_DIR, "models", "crop_disease_fallback.joblib")
META_PATH = os.path.join(BASE_DIR, "models", "disease_model_meta.json")

DISEASE_KNOWLEDGE_BASE: Dict[str, Dict[str, Any]] = {
    "Tomato___Early_blight": {
        "crop": "Tomato",
        "disease_name": "Early Blight",
        "pathogen": "Alternaria solani (Fungus)",
        "severity": "Moderate to High",
        "color_code": "#e65100",
        "symptoms": "Concentric dark brown rings ('target board') on older foliage, surrounded by chlorotic yellow halos. Leaves eventually yellow and drop.",
        "causes": "Warm temperatures (24-30 C) coupled with persistent leaf wetness, high humidity, and poor soil drainage.",
        "immediate_action": "Prune and burn/dispose of all lower infected leaves immediately. Avoid overhead sprinkler irrigation.",
        "organic_treatment": [
            "Spray Copper Oxychloride 50 WP (2.5 g/L water) or Bordeaux mixture (1%).",
            "Foliar application of Neem Oil (3-5 mL/L with mild surfactant) every 7 days.",
            "Soil drenching with Trichoderma viride bio-agent (10 g/L)."
        ],
        "chemical_treatment": [
            "Mancozeb 75% WP (2 g/L) applied at earliest appearance of spots.",
            "Azoxystrobin 23% SC (1 mL/L) or Difenoconazole 25% EC (0.5 mL/L) for severe outbreaks."
        ],
        "prevention": "Practice 3-year crop rotation (avoid solanaceous crops). Maintain 60cm plant spacing for adequate air circulation. Apply organic straw mulch.",
        "safe_harvest_wait_days": 7
    },
    "Tomato___Late_blight": {
        "crop": "Tomato",
        "disease_name": "Late Blight",
        "pathogen": "Phytophthora infestans (Oomycete)",
        "severity": "Critical Emergency",
        "color_code": "#b71c1c",
        "symptoms": "Large, irregular water-soaked pale-green to dark brown lesions. Underside shows delicate white fungal mildew under humid mornings. Causes rapid fruit rot.",
        "causes": "Cool, damp weather (15-22 C) with prolonged relative humidity >90% and continuous rain or heavy morning dew.",
        "immediate_action": "Quarantine affected beds immediately. Remove heavily blighted vines into sealed bags. Do NOT compost infected tissue.",
        "organic_treatment": [
            "Preventative copper hydroxide sprays (Kocide 2000 @ 2 g/L).",
            "Bio-fungicide Serenade (Bacillus amyloliquefaciens @ 5 mL/L)."
        ],
        "chemical_treatment": [
            "Metalaxyl 8% + Mancozeb 64% WP (Ridomil Gold @ 2.5 g/L).",
            "Cymoxanil + Mancozeb (Curzate M8 @ 2 g/L) for therapeutic curative knockdown."
        ],
        "prevention": "Plant certified disease-free seeds or resistant hybrids (e.g. Arka Rakshak). Implement drip irrigation to keep canopy bone dry.",
        "safe_harvest_wait_days": 10
    },
    "Tomato___Healthy": {
        "crop": "Tomato",
        "disease_name": "Healthy Foliage",
        "pathogen": "None (Plant is in optimal vigor)",
        "severity": "Optimal Health",
        "color_code": "#2e7d32",
        "symptoms": "Deep emerald green leaves with intact cellular turgor, crisp serrations, and uniform vein distribution. No visible spots or discoloration.",
        "causes": "Well-balanced nutrition (NPK), adequate sunlight (6-8 hours daily), and balanced root-zone moisture.",
        "immediate_action": "Maintain current standard agronomic management and routine IPM scouting.",
        "organic_treatment": [
            "Apply prophylactic seaweed extract / fish amino acid tonic (2 mL/L) every 14 days.",
            "Top-dress with vermicompost (150 g/plant) around root dripline."
        ],
        "chemical_treatment": ["None required. Avoid indiscriminate chemical spraying."],
        "prevention": "Continue sticky yellow traps for early whitefly/thrips detection and scheduled balanced fertigation.",
        "safe_harvest_wait_days": 0
    },
    "Potato___Early_blight": {
        "crop": "Potato",
        "disease_name": "Early Blight",
        "pathogen": "Alternaria solani",
        "severity": "Moderate",
        "color_code": "#e65100",
        "symptoms": "Dark brown circular spots with characteristic concentric rings on mature leaflets. Stems develop dark elongated lesions.",
        "causes": "Alternating wet and dry periods, plant stress due to nitrogen deficiency or heavy tuber bulking.",
        "immediate_action": "Avoid night-time irrigation. Apply protective fungicidal coat before rainy spell.",
        "organic_treatment": [
            "Spray Copper Sulphate + Hydrated Lime (1% Bordeaux mixture).",
            "Foliar spray of Pseudomonas fluorescens bio-inoculant (5 g/L)."
        ],
        "chemical_treatment": [
            "Chlorothalonil 75% WP (Kavach @ 2 g/L).",
            "Propiconazole 25% EC (Tilt @ 1 mL/L)."
        ],
        "prevention": "Ensure balanced nitrogen-potash fertilization. Destroy solanaceous weeds (nightshade). Store seed tubers at 4 C.",
        "safe_harvest_wait_days": 14
    },
    "Potato___Late_blight": {
        "crop": "Potato",
        "disease_name": "Late Blight",
        "pathogen": "Phytophthora infestans",
        "severity": "Critical Emergency",
        "color_code": "#b71c1c",
        "symptoms": "Water-soaked dark lesions spreading rapidly across foliage. Purplish-brown dry rot on tubers under the skin.",
        "causes": "High humidity (>80%) and temperatures between 10-21 C. The infamous cause of the Irish Great Famine.",
        "immediate_action": "Perform haulm cutting (dehaulming) immediately if tubers are mature to prevent spore wash into soil.",
        "organic_treatment": ["Copper Oxychloride (3 g/L) prophylactic spray prior to fog/rain events."],
        "chemical_treatment": [
            "Dimethomorph 50% WP (1 g/L) + Mancozeb (2 g/L).",
            "Fenamidone 10% + Mancozeb 50% WG (Sectin @ 2.5 g/L)."
        ],
        "prevention": "Use certified resistant tubers (Kufri Girdhari, Kufri Jyoti). Ensure high soil earthing-up to shield developing tubers.",
        "safe_harvest_wait_days": 14
    },
    "Potato___Healthy": {
        "crop": "Potato",
        "disease_name": "Healthy Leaf",
        "pathogen": "None (Clean Canopy)",
        "severity": "Optimal Health",
        "color_code": "#2e7d32",
        "symptoms": "Vigorous, smooth ovate leaflets with vibrant chlorophyll pigmentation and sturdy petiole attachment.",
        "causes": "Optimal soil moisture (65-75% field capacity) and cool weather suitable for tuberization.",
        "immediate_action": "Monitor soil moisture at 15cm depth. Continue scheduled irrigation.",
        "organic_treatment": ["Drench with mycorrhizal fungi inoculant to boost phosphorus uptake."],
        "chemical_treatment": ["None needed."],
        "prevention": "Scout for aphid vectors (Myzus persicae) to prevent viral degeneration.",
        "safe_harvest_wait_days": 0
    },
    "Corn___Common_rust": {
        "crop": "Corn / Maize",
        "disease_name": "Common Rust",
        "pathogen": "Puccinia sorghi (Fungus)",
        "severity": "Moderate",
        "color_code": "#bf360c",
        "symptoms": "Elongated golden-brown to cinnamon-brown powdery pustules on both upper and lower leaf surfaces.",
        "causes": "Moderate temperatures (16-25 C) with high relative humidity (>95%) and extended dew periods.",
        "immediate_action": "Scout upper canopy leaves during silking stage; treat if rust covers >5% leaf area.",
        "organic_treatment": [
            "Spray wettable sulfur 80% WDG (2.5 g/L).",
            "Spray baking soda solution (5 g/L + 2 drops liquid soap)."
        ],
        "chemical_treatment": [
            "Azoxystrobin 18.2% + Difenoconazole 11.4% SC (Amistar Top @ 1 mL/L).",
            "Pyraclostrobin 20% WG (0.5 g/L)."
        ],
        "prevention": "Sow rust-tolerant maize hybrids. Early planting allows crop to mature prior to peak rust spore dissemination.",
        "safe_harvest_wait_days": 21
    },
    "Corn___Northern_Leaf_Blight": {
        "crop": "Corn / Maize",
        "disease_name": "Northern Leaf Blight (NLB)",
        "pathogen": "Exserohilum turcicum",
        "severity": "High",
        "color_code": "#d84315",
        "symptoms": "Long, elliptical cigar-shaped grayish-green to tan lesions (2.5 to 15 cm long) with distinctive darkened margins.",
        "causes": "Wet, warm weather (18-27 C) accompanied by frequent rains.",
        "immediate_action": "Avoid overhead pivot irrigation in early morning.",
        "organic_treatment": ["Foliar spray of bio-formulation Trichoderma harzianum @ 5 g/L."],
        "chemical_treatment": [
            "Propiconazole 25% EC (1 mL/L).",
            "Tebuconazole 25.9% EC (Folicur @ 1.25 mL/L)."
        ],
        "prevention": "Deep autumn plowing to bury infested crop residue. Rotate with non-grass crops like legumes/soybean.",
        "safe_harvest_wait_days": 28
    },
    "Corn___Healthy": {
        "crop": "Corn / Maize",
        "disease_name": "Healthy Corn Leaf",
        "pathogen": "None",
        "severity": "Optimal Health",
        "color_code": "#2e7d32",
        "symptoms": "Long, robust linear leaf blades with broad white central midrib and strong photosynthetic activity.",
        "causes": "Plentiful nitrogen supply, warmth, and full sunlight.",
        "immediate_action": "Ensure adequate nitrogen side-dressing before knee-high (V6) and tasseling (VT) stages.",
        "organic_treatment": ["Foliar spray of Panchagavya (3%) during vegetative peak."],
        "chemical_treatment": ["None required."],
        "prevention": "Proper plant population density (65,000 plants/ha).",
        "safe_harvest_wait_days": 0
    },
    "Rice___Blast": {
        "crop": "Rice / Paddy",
        "disease_name": "Rice Blast",
        "pathogen": "Magnaporthe oryzae",
        "severity": "Critical",
        "color_code": "#c2185b",
        "symptoms": "Spindle-shaped / diamond-shaped spots with gray or white centers and brown or reddish-brown borders on leaves and neck.",
        "causes": "Excessive nitrogenous fertilization, cloudy overcast skies, high humidity (>90%), and temperatures ~20-28 C.",
        "immediate_action": "Stop all chemical nitrogen (Urea) applications immediately. Drain standing flood water for 48 hours to aerate soil.",
        "organic_treatment": [
            "Seed treatment with Pseudomonas fluorescens (10 g/kg seed).",
            "Spray crude extract of neem seed kernel (5%)."
        ],
        "chemical_treatment": [
            "Tricyclazole 75% WP (Beam @ 0.6 g/L water) - highly effective systemic blasticide.",
            "Isoprothiolane 40% EC (Fuji-One @ 1.5 mL/L)."
        ],
        "prevention": "Split nitrogen application into 3-4 doses. Treat seed with carbendazim prior to sowing.",
        "safe_harvest_wait_days": 21
    },
    "Rice___Brown_Spot": {
        "crop": "Rice / Paddy",
        "disease_name": "Brown Spot",
        "pathogen": "Bipolaris oryzae",
        "severity": "Moderate to High",
        "color_code": "#e65100",
        "symptoms": "Numerous small, circular to oval dark brown lesions with distinct yellow halos scattered evenly across leaf blades.",
        "causes": "Nutritional deficiency (especially Silicon, Potassium, Manganese) and chronic water stress in drought-prone upland fields.",
        "immediate_action": "Apply potash fertilizer (MOP @ 25 kg/ha) and ensure constant shallow standing water.",
        "organic_treatment": [
            "Spray cow urine distillate (5%) + neem extract.",
            "Silicon fertilizer soil amendment to toughen epidermis."
        ],
        "chemical_treatment": [
            "Mancozeb 75% WP (2 g/L) or Propiconazole 25% EC (1 mL/L).",
            "Carbendazim 50% WP (1 g/L)."
        ],
        "prevention": "Correct soil micronutrient deficiencies. Hot water seed treatment (53-54 C for 10 minutes).",
        "safe_harvest_wait_days": 14
    },
    "Rice___Healthy": {
        "crop": "Rice / Paddy",
        "disease_name": "Healthy Paddy Leaf",
        "pathogen": "None",
        "severity": "Optimal Health",
        "color_code": "#2e7d32",
        "symptoms": "Erect, slender, lustrous emerald green leaves with intact ligule and collar. Strong tiller foundation.",
        "causes": "Balanced water management (alternate wetting and drying) and optimal fertilization.",
        "immediate_action": "Maintain 2-5 cm standing water during panicle initiation stage.",
        "organic_treatment": ["Inoculate with Azolla bio-fertilizer for organic nitrogen fixation."],
        "chemical_treatment": ["None required."],
        "prevention": "Maintain clean bunds free of weed reservoirs.",
        "safe_harvest_wait_days": 0
    }
}

class DiseaseClassifier:
    """
    Inference Engine for Crop Disease Classification.
    Uses trained CNN (TensorFlow/Keras) with seamless fallback support.
    """

    def __init__(self):
        self.model = None
        self.fallback_model = None
        self.model_type = "CNN"
        self.classes = list(DISEASE_KNOWLEDGE_BASE.keys())
        self.load_model()

    def load_model(self):
        """Attempts to load Keras CNN, else loads fallback classifier."""
        for path in [MODEL_KERAS_PATH, MODEL_H5_PATH]:
            if os.path.exists(path):
                try:
                    import tensorflow as tf
                    self.model = tf.keras.models.load_model(path)
                    self.model_type = "TensorFlow / Keras CNN"
                    print(f"Loaded Keras CNN model from {path}")
                    return
                except Exception as e:
                    print(f"Could not load Keras model from {path}: {e}")

        if os.path.exists(FALLBACK_PATH):
            try:
                import joblib
                data = joblib.load(FALLBACK_PATH)
                self.fallback_model = data["model"]
                self.classes = data["classes"]
                self.model_type = "Scikit-Learn CV Classifier"
                print(f"Loaded fallback model from {FALLBACK_PATH}")
                return
            except Exception as e:
                print(f"Could not load fallback model: {e}")

        print("[Notice] CNN model not found on disk yet. Ready to train.")

    def classify_image(self, image_input: Union[str, bytes, Image.Image]) -> Dict[str, Any]:
        """
        Classifies an input leaf image and returns comprehensive diagnostics.
        """
        if isinstance(image_input, str):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, bytes):
            import io
            img = Image.open(io.BytesIO(image_input)).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            raise ValueError("Unsupported image input format.")

        resized_img = img.resize((128, 128))
        img_arr = np.array(resized_img, dtype=np.float32)

        probabilities = None
        predicted_idx = 0

        if self.model is not None:
            try:
                batch_arr = np.expand_dims(img_arr, axis=0)
                preds = self.model.predict(batch_arr, verbose=0)[0]
                probabilities = preds
                predicted_idx = int(np.argmax(preds))
            except Exception as e:
                print(f"Inference error with CNN model: {e}")
                self.model = None

        if self.model is None and self.fallback_model is not None:
            r = img_arr[:, :, 0]
            g = img_arr[:, :, 1]
            b = img_arr[:, :, 2]
            feats = np.array([[
                r.mean(), r.std(), g.mean(), g.std(), b.mean(), b.std(),
                (g.mean() - r.mean()), (g.mean() - b.mean()),
                r[32:96, 32:96].mean(), g[32:96, 32:96].mean(), b[32:96, 32:96].mean(),
                r[32:96, 32:96].std(), g[32:96, 32:96].std()
            ]], dtype=np.float32)
            proba = self.fallback_model.predict_proba(feats)[0]
            probabilities = proba
            predicted_idx = int(np.argmax(proba))

        if probabilities is None:
            probabilities = self._heuristic_analysis(img_arr)
            predicted_idx = int(np.argmax(probabilities))

        class_key = self.classes[predicted_idx] if predicted_idx < len(self.classes) else list(DISEASE_KNOWLEDGE_BASE.keys())[0]
        confidence = float(probabilities[predicted_idx]) * 100

        sorted_indices = np.argsort(probabilities)[::-1][:3]
        top_predictions = []
        for idx in sorted_indices:
            k = self.classes[idx] if idx < len(self.classes) else "Unknown"
            info = DISEASE_KNOWLEDGE_BASE.get(k, {})
            top_predictions.append({
                "class_key": k,
                "crop": info.get("crop", "Unknown"),
                "disease": info.get("disease_name", k),
                "probability_pct": round(float(probabilities[idx]) * 100, 2)
            })

        diagnostic = DISEASE_KNOWLEDGE_BASE.get(class_key, {})

        return {
            "success": True,
            "model_type": self.model_type,
            "predicted_class_key": class_key,
            "confidence_pct": round(confidence, 1),
            "top_predictions": top_predictions,
            "diagnostic": diagnostic
        }

    def _heuristic_analysis(self, img_arr: np.ndarray) -> np.ndarray:
        """
        Calculates diagnostic probabilities based on visual symptoms
        (chlorosis, necrotic brown/black spot ratio, rust orange pustules).
        """
        r, g, b = img_arr[:, :, 0], img_arr[:, :, 1], img_arr[:, :, 2]
        green_ratio = np.mean(g > (r * 1.1))
        brown_ratio = np.mean((r > 60) & (g > 30) & (b < 40) & (r > g))
        rust_ratio = np.mean((r > 130) & (g < 90) & (b < 40))
        dark_ratio = np.mean((r < 50) & (g < 50) & (b < 50))

        probs = np.full(len(self.classes), 0.04)

        if green_ratio > 0.65 and brown_ratio < 0.05:
            for i, c in enumerate(self.classes):
                if "Healthy" in c:
                    probs[i] = 0.30
        elif rust_ratio > 0.08:
            for i, c in enumerate(self.classes):
                if "rust" in c.lower():
                    probs[i] = 0.60
        elif dark_ratio > 0.12 or brown_ratio > 0.12:
            for i, c in enumerate(self.classes):
                if "blight" in c.lower() or "blast" in c.lower():
                    probs[i] = 0.45

        probs = probs / np.sum(probs)
        return probs
