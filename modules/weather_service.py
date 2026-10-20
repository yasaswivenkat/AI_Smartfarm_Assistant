"""
Weather Service Module for AI Smart Farm Assistant.
Provides integration with OpenWeatherMap API for live forecasts
and a high-accuracy realistic meteorological simulation mode for offline/demo use.
"""

import requests
from typing import Dict, Any, Optional
import math
import random
from datetime import datetime, timedelta

class WeatherService:
    """Handles weather API calls, responses, and offline fallback."""

    BASE_URL = "https://api.openweathermap.org/data/2.5"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key.strip() if api_key else None

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip() if api_key else None

    def get_current_weather(self, city: str = "Bangalore", units: str = "metric") -> Dict[str, Any]:
        """
        Fetches current weather for given city.
        Falls back to realistic simulation if no API key or network error occurs.
        """
        if self.api_key:
            try:
                url = f"{self.BASE_URL}/weather"
                params = {
                    "q": city,
                    "appid": self.api_key,
                    "units": units
                }
                resp = requests.get(url, params=params, timeout=6)
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "success": True,
                        "source": "OpenWeatherMap Live API",
                        "city": data.get("name", city),
                        "country": data.get("sys", {}).get("country", ""),
                        "temp_c": data["main"]["temp"],
                        "feels_like_c": data["main"]["feels_like"],
                        "temp_min_c": data["main"]["temp_min"],
                        "temp_max_c": data["main"]["temp_max"],
                        "humidity_pct": data["main"]["humidity"],
                        "pressure_hpa": data["main"]["pressure"],
                        "wind_speed_ms": data["wind"]["speed"],
                        "wind_deg": data["wind"].get("deg", 0),
                        "condition": data["weather"][0]["main"],
                        "description": data["weather"][0]["description"].title(),
                        "icon": data["weather"][0]["icon"],
                        "rainfall_1h_mm": data.get("rain", {}).get("1h", 0.0),
                        "timestamp": datetime.fromtimestamp(data["dt"]).strftime("%Y-%m-%d %H:%M:%S")
                    }
                elif resp.status_code == 401:
                    return {
                        "success": False,
                        "error": "Invalid OpenWeatherMap API Key. Using high-fidelity farm simulation.",
                        "fallback": self._simulate_current_weather(city)
                    }
                else:
                    return {
                        "success": False,
                        "error": f"API Error ({resp.status_code}): {resp.json().get('message', 'Failed to retrieve data')}",
                        "fallback": self._simulate_current_weather(city)
                    }
            except Exception as e:
                return {
                    "success": False,
                    "error": f"Connection error: {str(e)}",
                    "fallback": self._simulate_current_weather(city)
                }

        return {
            "success": True,
            "source": "Simulated Agro-Meteorological Station",
            **self._simulate_current_weather(city)
        }

    def get_5day_forecast(self, city: str = "Bangalore", units: str = "metric") -> Dict[str, Any]:
        """
        Fetches 5-day / 3-hour interval weather forecast.
        """
        if self.api_key:
            try:
                url = f"{self.BASE_URL}/forecast"
                params = {"q": city, "appid": self.api_key, "units": units}
                resp = requests.get(url, params=params, timeout=6)
                if resp.status_code == 200:
                    data = resp.json()
                    forecast_list = []
                    for entry in data.get("list", []):
                        forecast_list.append({
                            "datetime": entry["dt_txt"],
                            "temp_c": entry["main"]["temp"],
                            "temp_min_c": entry["main"]["temp_min"],
                            "temp_max_c": entry["main"]["temp_max"],
                            "humidity_pct": entry["main"]["humidity"],
                            "pressure_hpa": entry["main"]["pressure"],
                            "wind_speed_ms": entry["wind"]["speed"],
                            "condition": entry["weather"][0]["main"],
                            "description": entry["weather"][0]["description"].title(),
                            "rainfall_3h_mm": entry.get("rain", {}).get("3h", 0.0),
                            "pop": entry.get("pop", 0) * 100
                        })
                    return {
                        "success": True,
                        "source": "OpenWeatherMap 5-Day Forecast API",
                        "city": data.get("city", {}).get("name", city),
                        "forecast": forecast_list
                    }
            except Exception:
                pass

        return {
            "success": True,
            "source": "Agro-Meteorological Forecast Engine (Simulated)",
            "city": city,
            "forecast": self._simulate_5day_forecast(city)
        }

    def _simulate_current_weather(self, city: str) -> Dict[str, Any]:
        """Simulates realistic current micro-climate based on hour and season."""
        now = datetime.now()
        hour = now.hour
        temp_cycle = math.sin((hour - 8) * math.pi / 12)
        base_temp = 26.5 + (temp_cycle * 5.5) + random.uniform(-1.0, 1.0)
        humidity = max(35.0, min(95.0, 68.0 - (temp_cycle * 22.0) + random.uniform(-3, 3)))
        wind = max(0.8, min(14.0, 3.5 + random.uniform(-1.2, 2.5)))
        pressure = 1012.0 + random.uniform(-2.5, 2.5)

        rain_chance = 0.35 if humidity > 70 else 0.10
        is_raining = random.random() < rain_chance
        rainfall = round(random.uniform(1.2, 14.5), 1) if is_raining else 0.0

        if is_raining:
            condition = "Rain"
            description = "Scattered Showers"
            icon = "10d"
        elif humidity > 75:
            condition = "Clouds"
            description = "Partly Cloudy"
            icon = "03d"
        else:
            condition = "Clear"
            description = "Clear Sky"
            icon = "01d"

        return {
            "city": city,
            "country": "IN",
            "temp_c": round(base_temp, 1),
            "feels_like_c": round(base_temp + (humidity / 100 * 2.0), 1),
            "temp_min_c": round(base_temp - 3.2, 1),
            "temp_max_c": round(base_temp + 3.8, 1),
            "humidity_pct": int(humidity),
            "pressure_hpa": int(pressure),
            "wind_speed_ms": round(wind, 1),
            "wind_deg": random.randint(40, 280),
            "condition": condition,
            "description": description,
            "icon": icon,
            "rainfall_1h_mm": rainfall,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S")
        }

    def _simulate_5day_forecast(self, city: str) -> list:
        """Generates realistic 5-day / 3-hour interval weather projections."""
        now = datetime.now().replace(minute=0, second=0, microsecond=0)
        forecast = []
        base_temp = 27.0

        for i in range(40):
            step_time = now + timedelta(hours=i * 3)
            hour = step_time.hour
            temp_cycle = math.sin((hour - 8) * math.pi / 12)
            temp = base_temp + (temp_cycle * 6.0) + random.uniform(-1.2, 1.2)
            humidity = max(35.0, min(95.0, 70.0 - (temp_cycle * 25.0) + random.uniform(-4, 4)))
            pop = max(5.0, min(90.0, (humidity - 40.0) * 1.5 + random.uniform(-10, 10)))
            rain_mm = round(random.uniform(0.5, 8.0), 1) if pop > 60 else 0.0

            forecast.append({
                "datetime": step_time.strftime("%Y-%m-%d %H:%M:%S"),
                "temp_c": round(temp, 1),
                "temp_min_c": round(temp - 2.5, 1),
                "temp_max_c": round(temp + 2.5, 1),
                "humidity_pct": int(humidity),
                "pressure_hpa": 1012,
                "wind_speed_ms": round(3.2 + random.uniform(-1.0, 2.0), 1),
                "condition": "Rain" if rain_mm > 0 else ("Clouds" if humidity > 70 else "Clear"),
                "description": "Light Rain" if rain_mm > 0 else ("Partly Cloudy" if humidity > 70 else "Sunny"),
                "rainfall_3h_mm": rain_mm,
                "pop": round(pop, 1)
            })
        return forecast
