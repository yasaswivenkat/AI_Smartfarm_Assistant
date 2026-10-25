"""
Machine Learning Crop and Fertilizer Recommendation Engine.
1. Recommends the optimal crop based on soil NPK, pH, temperature, humidity, and rainfall.
2. Calculates exact fertilizer dosage (Urea, DAP, MOP) based on nutrient deficits for target crop.
"""

import os
import math
import random
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Tuple
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "crop_recommendation.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "crop_recommender.joblib")

CROP_PROFILES = {
    "Rice": {"N": 80, "P": 40, "K": 40, "pH": 6.5, "temp": 24.0, "humidity": 82.0, "rainfall": 200.0},
    "Maize": {"N": 100, "P": 50, "K": 40, "pH": 6.2, "temp": 23.0, "humidity": 65.0, "rainfall": 90.0},
    "Chickpea": {"N": 40, "P": 60, "K": 80, "pH": 7.0, "temp": 19.0, "humidity": 18.0, "rainfall": 75.0},
    "Kidney Beans": {"N": 20, "P": 65, "K": 20, "pH": 5.8, "temp": 20.0, "humidity": 22.0, "rainfall": 105.0},
    "Pigeon Peas": {"N": 20, "P": 60, "K": 20, "pH": 6.0, "temp": 28.0, "humidity": 48.0, "rainfall": 150.0},
    "Moth Beans": {"N": 20, "P": 40, "K": 20, "pH": 7.2, "temp": 28.0, "humidity": 53.0, "rainfall": 50.0},
    "Mung Bean": {"N": 20, "P": 40, "K": 20, "pH": 6.7, "temp": 28.0, "humidity": 85.0, "rainfall": 48.0},
    "Black Gram": {"N": 40, "P": 60, "K": 20, "pH": 7.1, "temp": 30.0, "humidity": 65.0, "rainfall": 68.0},
    "Lentil": {"N": 20, "P": 60, "K": 20, "pH": 6.9, "temp": 24.0, "humidity": 64.0, "rainfall": 45.0},
    "Pomegranate": {"N": 20, "P": 10, "K": 40, "pH": 6.4, "temp": 22.0, "humidity": 90.0, "rainfall": 110.0},
    "Banana": {"N": 100, "P": 75, "K": 50, "pH": 6.0, "temp": 27.0, "humidity": 80.0, "rainfall": 100.0},
    "Mango": {"N": 20, "P": 20, "K": 30, "pH": 5.7, "temp": 31.0, "humidity": 50.0, "rainfall": 95.0},
    "Grapes": {"N": 20, "P": 125, "K": 200, "pH": 6.0, "temp": 24.0, "humidity": 82.0, "rainfall": 70.0},
    "Watermelon": {"N": 100, "P": 10, "K": 50, "pH": 6.5, "temp": 26.0, "humidity": 88.0, "rainfall": 50.0},
    "Apple": {"N": 20, "P": 125, "K": 200, "pH": 5.9, "temp": 22.0, "humidity": 92.0, "rainfall": 110.0},
    "Orange": {"N": 20, "P": 10, "K": 10, "pH": 7.0, "temp": 23.0, "humidity": 92.0, "rainfall": 110.0},
    "Papaya": {"N": 50, "P": 60, "K": 50, "pH": 6.7, "temp": 33.0, "humidity": 92.0, "rainfall": 140.0},
    "Cotton": {"N": 120, "P": 40, "K": 20, "pH": 6.9, "temp": 24.0, "humidity": 80.0, "rainfall": 80.0},
    "Coffee": {"N": 100, "P": 30, "K": 30, "pH": 6.8, "temp": 25.0, "humidity": 58.0, "rainfall": 160.0}
}

def generate_crop_dataset(num_samples_per_crop: int = 60) -> pd.DataFrame:
    """Generates synthetic multi-variable soil-climatic dataset for crop modeling."""
    np.random.seed(42)
    records = []

    for crop, profile in CROP_PROFILES.items():
        for _ in range(num_samples_per_crop):
            n = max(5, profile["N"] + np.random.normal(0, profile["N"] * 0.15))
            p = max(5, profile["P"] + np.random.normal(0, profile["P"] * 0.15))
            k = max(5, profile["K"] + np.random.normal(0, profile["K"] * 0.15))
            ph = np.clip(profile["pH"] + np.random.normal(0, 0.4), 4.5, 9.0)
            temp = profile["temp"] + np.random.normal(0, 2.5)
            hum = np.clip(profile["humidity"] + np.random.normal(0, 5.0), 15.0, 99.0)
            rain = max(10, profile["rainfall"] + np.random.normal(0, 18.0))

            records.append({
                "N": round(n, 1),
                "P": round(p, 1),
                "K": round(k, 1),
                "temperature": round(temp, 1),
                "humidity": round(hum, 1),
                "ph": round(ph, 2),
                "rainfall": round(rain, 1),
                "label": crop
            })

    df = pd.DataFrame(records)
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated crop recommendation dataset: {len(df)} rows -> {DATA_PATH}")
    return df

class CropRecommender:
    """ML Model for predicting optimal crop from soil & climate telemetry."""

    def __init__(self):
        self.model = None
        self.feature_names = ["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]
        self.accuracy = 0.0

        if os.path.exists(MODEL_PATH):
            self.load_model()
        else:
            self.train_and_save()

    def train_and_save(self) -> float:
        """Trains RandomForest on crop dataset."""
        if not os.path.exists(DATA_PATH):
            df = generate_crop_dataset()
        else:
            df = pd.read_csv(DATA_PATH)

        X = df[self.feature_names]
        y = df["label"]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
        self.model.fit(X_train, y_train)

        preds = self.model.predict(X_test)
        self.accuracy = round(float(accuracy_score(y_test, preds)) * 100, 2)

        joblib.dump({"model": self.model, "accuracy": self.accuracy, "features": self.feature_names}, MODEL_PATH)
        print(f"Crop Recommender trained with {self.accuracy}% accuracy -> {MODEL_PATH}")
        return self.accuracy

    def load_model(self) -> bool:
        """Loads serialized model."""
        try:
            data = joblib.load(MODEL_PATH)
            self.model = data["model"]
            self.accuracy = data.get("accuracy", 96.0)
            self.feature_names = data.get("features", self.feature_names)
            return True
        except Exception:
            self.train_and_save()
            return False

    def recommend(self, n: float, p: float, k: float, temp: float, hum: float, ph: float, rain: float) -> List[Dict[str, Any]]:
        """Predicts top-3 suitable crops with confidence probabilities."""
        if self.model is None:
            self.load_model()

        sample = pd.DataFrame([[n, p, k, temp, hum, ph, rain]], columns=self.feature_names)
        probs = self.model.predict_proba(sample)[0]
        top_indices = np.argsort(probs)[::-1][:3]

        recommendations = []
        for idx in top_indices:
            crop_name = self.model.classes_[idx]
            prob_pct = round(float(probs[idx]) * 100, 1)
            profile = CROP_PROFILES.get(crop_name, {})
            recommendations.append({
                "crop": crop_name,
                "confidence_pct": prob_pct,
                "ideal_profile": profile
            })
        return recommendations

    @staticmethod
    def calculate_fertilizer_prescription(soil_n: float, soil_p: float, soil_k: float, target_crop: str) -> Dict[str, Any]:
        """
        Calculates exact fertilizer quantities needed to bridge soil nutrient deficits.
        Urea contains 46% N.
        DAP contains 18% N, 46% P2O5.
        MOP (Muriate of Potash) contains 60% K2O.
        """
        profile = CROP_PROFILES.get(target_crop, {"N": 80, "P": 40, "K": 40})
        req_n = profile["N"]
        req_p = profile["P"]
        req_k = profile["K"]

        def_n = max(0.0, req_n - (soil_n * 0.4))
        def_p = max(0.0, req_p - (soil_p * 0.35))
        def_k = max(0.0, req_k - (soil_k * 0.25))

        dap_needed_kg = round((def_p / 0.46), 1) if def_p > 0 else 0.0
        n_supplied_by_dap = dap_needed_kg * 0.18

        rem_n = max(0.0, def_n - n_supplied_by_dap)
        urea_needed_kg = round((rem_n / 0.46), 1) if rem_n > 0 else 0.0

        mop_needed_kg = round((def_k / 0.60), 1) if def_k > 0 else 0.0

        return {
            "target_crop": target_crop,
            "soil_available": {"N": soil_n, "P": soil_p, "K": soil_k},
            "crop_requirement_kg_ha": {"N": req_n, "P": req_p, "K": req_k},
            "net_deficits_kg_ha": {"N": round(def_n, 1), "P": round(def_p, 1), "K": round(def_k, 1)},
            "fertilizer_dosage_kg_ha": {
                "Urea (46% N)": urea_needed_kg,
                "DAP (18-46-0)": dap_needed_kg,
                "MOP (60% K)": mop_needed_kg
            },
            "fertilizer_dosage_bags_ha": {
                "Urea (45kg Bags)": round(urea_needed_kg / 45, 1),
                "DAP (50kg Bags)": round(dap_needed_kg / 50, 1),
                "MOP (50kg Bags)": round(mop_needed_kg / 50, 1)
            },
            "application_schedule": [
                "Basal: Apply 100% DAP + 50% MOP + 25% Urea before sowing / transplanting.",
                "First Top Dressing (30 Days): Apply 50% Urea at active vegetative growth.",
                "Second Top Dressing (60 Days): Apply 25% Urea + 50% MOP before flowering stage."
            ]
        }

if __name__ == "__main__":
    recommender = CropRecommender()
    print("Crop recommender ready!")
