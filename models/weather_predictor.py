"""
Machine Learning Weather Prediction Module.
Trains and evaluates regression & classification models on historical weather data
to forecast temperature trajectories, rainfall occurrence probability, and precipitation volume.
"""

import os
import math
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from typing import Dict, Any, Tuple, List, Optional
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score, recall_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "historical_weather.csv")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "models", "weather_ml_models.joblib")
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

class WeatherPredictor:
    """
    ML Engine for agricultural weather prediction.
    Predicts:
    1. Future Mean, Min, Max Temperature Trends ( C)
    2. Rainfall probability (Classification) & Expected volume in mm (Regression)
    """

    def __init__(self, data_path: str = DATA_PATH):
        self.data_path = data_path
        self.temp_regressor: Optional[RandomForestRegressor] = None
        self.rain_classifier: Optional[RandomForestClassifier] = None
        self.rain_regressor: Optional[GradientBoostingRegressor] = None
        self.metrics: Dict[str, Any] = {}
        self.feature_names: List[str] = []
        self.df: Optional[pd.DataFrame] = None

        if os.path.exists(MODEL_SAVE_PATH):
            self.load_models()
        else:
            self.train_and_save()

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Constructs cyclical temporal and autoregressive lag features."""
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").reset_index(drop=True)

        doy = df["day_of_year"]
        df["sin_doy"] = np.sin(2 * np.pi * doy / 365.25)
        df["cos_doy"] = np.cos(2 * np.pi * doy / 365.25)

        df["temp_lag1"] = df["mean_temp_c"].shift(1)
        df["temp_lag2"] = df["mean_temp_c"].shift(2)
        df["temp_lag3"] = df["mean_temp_c"].shift(3)

        df["humidity_lag1"] = df["humidity_pct"].shift(1)
        df["rainfall_lag1"] = df["rainfall_mm"].shift(1)

        df["temp_roll7_mean"] = df["mean_temp_c"].rolling(window=7, min_periods=1).mean()
        df["rain_roll7_sum"] = df["rainfall_mm"].rolling(window=7, min_periods=1).sum()
        df["humidity_roll7_mean"] = df["humidity_pct"].rolling(window=7, min_periods=1).mean()

        df["is_rain"] = (df["rainfall_mm"] > 0.5).astype(int)

        df["target_temp_next"] = df["mean_temp_c"].shift(-1)
        df["target_rain_next"] = df["rainfall_mm"].shift(-1)
        df["target_is_rain_next"] = df["is_rain"].shift(-1)

        df = df.dropna().reset_index(drop=True)
        return df

    def train_and_save(self) -> Dict[str, Any]:
        """Trains ML models on historical dataset and calculates performance metrics."""
        if not os.path.exists(self.data_path):
            from utils.dataset_generator import generate_historical_weather
            generate_historical_weather(self.data_path)

        raw_df = pd.read_csv(self.data_path)
        processed_df = self._engineer_features(raw_df)
        self.df = processed_df

        self.feature_names = [
            "sin_doy", "cos_doy", "month",
            "mean_temp_c", "humidity_pct", "pressure_hpa", "wind_speed_ms", "solar_radiation_mj",
            "temp_lag1", "temp_lag2", "humidity_lag1", "rainfall_lag1",
            "temp_roll7_mean", "rain_roll7_sum", "humidity_roll7_mean"
        ]

        X = processed_df[self.feature_names]
        y_temp = processed_df["target_temp_next"]
        y_rain_class = processed_df["target_is_rain_next"]
        y_rain_vol = processed_df["target_rain_next"]

        split_idx = int(len(processed_df) * 0.8)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_t_train, y_t_test = y_temp.iloc[:split_idx], y_temp.iloc[split_idx:]
        y_rc_train, y_rc_test = y_rain_class.iloc[:split_idx], y_rain_class.iloc[split_idx:]
        y_rv_train, y_rv_test = y_rain_vol.iloc[:split_idx], y_rain_vol.iloc[split_idx:]

        self.temp_regressor = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
        self.temp_regressor.fit(X_train, y_t_train)
        pred_temp = self.temp_regressor.predict(X_test)
        temp_mae = mean_absolute_error(y_t_test, pred_temp)
        temp_rmse = np.sqrt(mean_squared_error(y_t_test, pred_temp))
        temp_r2 = r2_score(y_t_test, pred_temp)

        self.rain_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
        self.rain_classifier.fit(X_train, y_rc_train)
        pred_rc = self.rain_classifier.predict(X_test)
        rain_acc = accuracy_score(y_rc_test, pred_rc)
        rain_prec = precision_score(y_rc_test, pred_rc, zero_division=0)
        rain_rec = recall_score(y_rc_test, pred_rc, zero_division=0)

        self.rain_regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
        self.rain_regressor.fit(X_train, y_rv_train)
        pred_rv = self.rain_regressor.predict(X_test)
        pred_rv = np.clip(pred_rv, 0, None)
        rain_mae = mean_absolute_error(y_rv_test, pred_rv)

        self.metrics = {
            "trained_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_rows": len(processed_df),
            "temp_mae": round(float(temp_mae), 3),
            "temp_rmse": round(float(temp_rmse), 3),
            "temp_r2": round(float(temp_r2), 4),
            "rain_accuracy": round(float(rain_acc) * 100, 2),
            "rain_precision": round(float(rain_prec) * 100, 2),
            "rain_recall": round(float(rain_rec) * 100, 2),
            "rain_volume_mae_mm": round(float(rain_mae), 3),
            "feature_importances": {
                name: round(float(imp), 4)
                for name, imp in zip(self.feature_names, self.temp_regressor.feature_importances_)
            }
        }

        artifact = {
            "temp_regressor": self.temp_regressor,
            "rain_classifier": self.rain_classifier,
            "rain_regressor": self.rain_regressor,
            "feature_names": self.feature_names,
            "metrics": self.metrics
        }
        joblib.dump(artifact, MODEL_SAVE_PATH)
        print(f"Weather ML Models trained & saved to {MODEL_SAVE_PATH}")
        return self.metrics

    def load_models(self) -> bool:
        """Loads serialized ML models from disk."""
        try:
            artifact = joblib.load(MODEL_SAVE_PATH)
            self.temp_regressor = artifact["temp_regressor"]
            self.rain_classifier = artifact["rain_classifier"]
            self.rain_regressor = artifact["rain_regressor"]
            self.feature_names = artifact["feature_names"]
            self.metrics = artifact.get("metrics", {})
            return True
        except Exception as e:
            print(f"Error loading weather models: {e}. Retraining...")
            self.train_and_save()
            return False

    def predict_multi_day_trend(self, current_data: Optional[Dict[str, float]] = None, days: int = 7) -> List[Dict[str, Any]]:
        """
        Iteratively forecasts future multi-day temperature and rainfall trend
        starting from either the latest historical record or current live inputs.
        """
        if self.temp_regressor is None:
            self.load_models()

        if self.df is None and os.path.exists(self.data_path):
            raw = pd.read_csv(self.data_path)
            self.df = self._engineer_features(raw)

        last_row = self.df.iloc[-1].to_dict() if self.df is not None else {}

        curr_temp = current_data.get("temp_c", last_row.get("mean_temp_c", 26.5)) if current_data else last_row.get("mean_temp_c", 26.5)
        curr_humidity = current_data.get("humidity_pct", last_row.get("humidity_pct", 65.0)) if current_data else last_row.get("humidity_pct", 65.0)
        curr_pressure = current_data.get("pressure_hpa", last_row.get("pressure_hpa", 1012.0)) if current_data else last_row.get("pressure_hpa", 1012.0)
        curr_wind = current_data.get("wind_speed_ms", last_row.get("wind_speed_ms", 3.5)) if current_data else last_row.get("wind_speed_ms", 3.5)
        curr_rain = current_data.get("rainfall_mm", 0.0) if current_data else 0.0

        temp_lag1 = curr_temp
        temp_lag2 = last_row.get("temp_lag1", curr_temp - 0.5)
        hum_lag1 = curr_humidity
        rain_lag1 = curr_rain

        temp_roll = curr_temp
        hum_roll = curr_humidity
        rain_roll = curr_rain

        predictions = []
        today = datetime.now()

        for step in range(1, days + 1):
            target_date = today + timedelta(days=step)
            doy = target_date.timetuple().tm_yday
            sin_doy = math.sin(2 * math.pi * doy / 365.25)
            cos_doy = math.cos(2 * math.pi * doy / 365.25)
            month = target_date.month

            solar = max(5.0, 24.0 - (curr_humidity * 0.12) - (curr_rain * 0.2))

            feat_vector = pd.DataFrame([{
                "sin_doy": sin_doy,
                "cos_doy": cos_doy,
                "month": month,
                "mean_temp_c": curr_temp,
                "humidity_pct": curr_humidity,
                "pressure_hpa": curr_pressure,
                "wind_speed_ms": curr_wind,
                "solar_radiation_mj": solar,
                "temp_lag1": temp_lag1,
                "temp_lag2": temp_lag2,
                "humidity_lag1": hum_lag1,
                "rainfall_lag1": rain_lag1,
                "temp_roll7_mean": temp_roll,
                "rain_roll7_sum": rain_roll,
                "humidity_roll7_mean": hum_roll
            }])[self.feature_names]

            next_temp = float(self.temp_regressor.predict(feat_vector)[0])
            max_temp = round(next_temp + 4.2 + (step * 0.1), 1)
            min_temp = round(next_temp - 4.5 - (step * 0.1), 1)
            mean_temp = round(next_temp, 1)

            rain_prob_pct = round(float(self.rain_classifier.predict_proba(feat_vector)[0][1]) * 100, 1)
            predicted_rain_vol = float(self.rain_regressor.predict(feat_vector)[0])
            predicted_rain_mm = round(max(0.0, predicted_rain_vol if rain_prob_pct > 35 else 0.0), 1)

            if rain_prob_pct > 65:
                condition = "Rain Expected"
            elif rain_prob_pct > 35:
                condition = "Chance of Rain"
            elif curr_humidity > 70:
                condition = "Cloudy"
            else:
                condition = "Clear / Sunny"

            predictions.append({
                "date": target_date.strftime("%Y-%m-%d"),
                "day_name": target_date.strftime("%A"),
                "mean_temp_c": mean_temp,
                "max_temp_c": max_temp,
                "min_temp_c": min_temp,
                "rain_prob_pct": rain_prob_pct,
                "predicted_rain_mm": predicted_rain_mm,
                "condition": condition,
                "confidence_pct": round(max(65.0, 96.0 - (step * 2.5)), 1)
            })

            temp_lag2 = temp_lag1
            temp_lag1 = next_temp
            curr_temp = next_temp
            rain_lag1 = predicted_rain_mm
            rain_roll = (rain_roll * 0.8) + predicted_rain_mm
            hum_lag1 = curr_humidity
            curr_humidity = np.clip(curr_humidity + (5 if predicted_rain_mm > 0 else -3), 30.0, 95.0)

        return predictions

if __name__ == "__main__":
    wp = WeatherPredictor()
    print("Trained Weather Models Successfully!")
    print("Metrics:", wp.metrics)

