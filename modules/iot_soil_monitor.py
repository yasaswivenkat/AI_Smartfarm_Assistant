"""
IoT Soil & Environmental Monitoring Module.
Simulates real-time IoT agricultural sensors (Moisture, Temp, pH, N-P-K, EC)
and implements Smart Automated Irrigation decision-making logic.
"""

import math
import random
from datetime import datetime, timedelta
from typing import Dict, Any, List

class SoilSensorNode:
    """
    Simulates a multi-depth IoT soil sensor node deployed in the farm field.
    Provides realistic telemetry, diurnal fluctuations, and threshold evaluations.
    """

    def __init__(self, node_id: str = "NODE-ALPHA-01", crop_type: str = "Tomato", soil_type: str = "Loamy"):
        self.node_id = node_id
        self.crop_type = crop_type
        self.soil_type = soil_type
        self.pump_status = False
        self.pump_started_at = None
        self.soil_moisture_pct = 42.5
        self.soil_temp_c = 23.4
        self.soil_ph = 6.6
        self.ec_dsm = 1.2
        self.nitrogen_mg_kg = 145.0
        self.phosphorus_mg_kg = 38.0
        self.potassium_mg_kg = 210.0
        self.battery_pct = 94.0

    def read_telemetry(self, ambient_temp: float = 28.0, is_raining: bool = False) -> Dict[str, Any]:
        """
        Generates dynamic realistic telemetry based on farm environmental dynamics.
        """
        now = datetime.now()
        hour = now.hour

        self.soil_temp_c = round(ambient_temp * 0.75 + 6.0 + math.sin((hour - 10) * math.pi / 12) * 2.0, 1)

        if self.pump_status:
            self.soil_moisture_pct = min(88.0, self.soil_moisture_pct + random.uniform(4.0, 7.5))
            if self.soil_moisture_pct >= 68.0:
                self.pump_status = False
        elif is_raining:
            self.soil_moisture_pct = min(92.0, self.soil_moisture_pct + random.uniform(5.0, 10.0))
        else:
            self.soil_moisture_pct = max(18.0, self.soil_moisture_pct - random.uniform(0.1, 0.4))

        self.soil_moisture_pct = round(self.soil_moisture_pct, 1)

        self.soil_ph = round(max(5.2, min(8.2, self.soil_ph + random.uniform(-0.02, 0.02))), 2)
        self.ec_dsm = round(max(0.5, min(3.5, self.ec_dsm + random.uniform(-0.01, 0.01))), 2)

        if self.soil_moisture_pct < 30.0:
            moisture_health = "Critical Drought Stress"
            moisture_color = "#d32f2f"
        elif self.soil_moisture_pct < 45.0:
            moisture_health = "Irrigation Recommended"
            moisture_color = "#f57c00"
        elif self.soil_moisture_pct <= 75.0:
            moisture_health = "Optimal Field Capacity"
            moisture_color = "#2e7d32"
        else:
            moisture_health = "Waterlogged / Oversaturated"
            moisture_color = "#1565c0"

        if self.soil_ph < 6.0:
            ph_status = "Acidic (Add agricultural lime/dolomite)"
        elif self.soil_ph > 7.5:
            ph_status = "Alkaline (Apply gypsum or organic compost)"
        else:
            ph_status = "Optimal Neutral Range (6.0 - 7.5)"

        n_status = "Deficient" if self.nitrogen_mg_kg < 120 else ("Excess" if self.nitrogen_mg_kg > 250 else "Balanced")
        p_status = "Deficient" if self.phosphorus_mg_kg < 25 else ("Excess" if self.phosphorus_mg_kg > 60 else "Balanced")
        k_status = "Deficient" if self.potassium_mg_kg < 150 else ("Excess" if self.potassium_mg_kg > 300 else "Balanced")

        irrigation_rec = self.calculate_irrigation_recommendation()

        return {
            "node_id": self.node_id,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
            "crop": self.crop_type,
            "soil_type": self.soil_type,
            "soil_moisture_pct": self.soil_moisture_pct,
            "moisture_health": moisture_health,
            "moisture_color": moisture_color,
            "soil_temp_c": self.soil_temp_c,
            "soil_ph": self.soil_ph,
            "ph_status": ph_status,
            "ec_dsm": self.ec_dsm,
            "npk": {
                "nitrogen": {"value": self.nitrogen_mg_kg, "unit": "mg/kg", "status": n_status},
                "phosphorus": {"value": self.phosphorus_mg_kg, "unit": "mg/kg", "status": p_status},
                "potassium": {"value": self.potassium_mg_kg, "unit": "mg/kg", "status": k_status}
            },
            "pump_status": "ACTIVE (Irrigating)" if self.pump_status else "STANDBY (Idle)",
            "pump_active": self.pump_status,
            "battery_pct": self.battery_pct,
            "irrigation_rec": irrigation_rec
        }

    def calculate_irrigation_recommendation(self) -> Dict[str, Any]:
        """Calculates precise water requirement in Liters per Hectare."""
        target_moisture = 65.0
        deficit_pct = max(0.0, target_moisture - self.soil_moisture_pct)

        water_req_liters_per_ha = round(deficit_pct * 3200, 0)
        est_drip_duration_mins = round((water_req_liters_per_ha / 5000) * 60, 0) if deficit_pct > 0 else 0

        urgency = "HIGH" if deficit_pct > 25 else ("MEDIUM" if deficit_pct > 12 else "NONE")

        return {
            "deficit_pct": round(deficit_pct, 1),
            "water_required_liters_per_ha": int(water_req_liters_per_ha),
            "drip_run_time_minutes": int(est_drip_duration_mins),
            "urgency": urgency,
            "action": f"Run drip irrigation for ~{int(est_drip_duration_mins)} minutes." if deficit_pct > 0 else "Soil moisture is optimal; no irrigation needed."
        }

    def trigger_pump(self, state: bool):
        """Activates or stops the automated solenoid irrigation valve."""
        self.pump_status = state
        if state:
            self.pump_started_at = datetime.now()
        return self.pump_status

    def generate_recent_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Generates realistic 24-hour sensor telemetry history for time-series charts."""
        now = datetime.now()
        records = []
        base_moisture = self.soil_moisture_pct
        for i in range(hours, 0, -1):
            t = now - timedelta(hours=i)
            h = t.hour
            temp_var = math.sin((h - 11) * math.pi / 12) * 2.8
            moist_drift = math.sin(i * 0.4) * 3.5
            records.append({
                "time": t.strftime("%H:%M"),
                "datetime": t.strftime("%Y-%m-%d %H:%M"),
                "soil_moisture_pct": round(max(20.0, min(80.0, base_moisture + moist_drift)), 1),
                "soil_temp_c": round(22.0 + temp_var, 1),
                "ambient_temp_c": round(26.0 + (temp_var * 1.5), 1)
            })
        return records
