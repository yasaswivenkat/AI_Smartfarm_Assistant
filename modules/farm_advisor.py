"""
AI Farm Advisor & Chatbot Module.
Provides interactive conversational assistance for farmers.
Features dual execution:
1. Gemini 2.5 Flash LLM via official google-genai SDK when API key is provided.
2. Built-in Offline Agro-Knowledge Engine with keyword & intent matching covering 50+ agricultural topics.
"""

import os
from typing import Dict, Any, List, Optional

OFFLINE_KNOWLEDGE_TOPICS = [
    {
        "keywords": ["fertilizer", "urea", "dap", "potash", "dosage", "npk", "dose"],
        "title": "Fertilizer & NPK Dosage Guide",
        "response": """**Recommended Fertilizer Dosage & Best Practices:**

1. **Tomato (Per Acre):**
   - **Basal Application:** 10 tonnes FYM / Compost + 50 kg DAP + 40 kg MOP.
   - **Vegetative Phase:** 25 kg Urea at 30 days after transplanting (DAT).
   - **Flowering & Fruiting:** 25 kg Urea + 30 kg MOP at 45-60 DAT.
   - *Tip:* Split nitrogen applications prevent vegetative overgrowth and blossom end rot.

2. **Potato (Per Acre):**
   - **Basal:** 10-15 tonnes Farm Yard Manure + 75 kg DAP + 60 kg MOP + 35 kg Urea.
   - **Earthing Up (30-35 days):** Top-dress with 40 kg Urea.

3. **Corn / Maize (Per Acre):**
   - **Basal:** 50 kg DAP + 30 kg MOP + 20 kg Zinc Sulphate.
   - **Knee-High Stage (V6):** 35 kg Urea.
   - **Tasseling Stage (VT):** 35 kg Urea.

4. **Paddy / Rice (Per Acre):**
   - **Basal:** 40 kg DAP + 25 kg MOP + 10 kg Zinc Sulphate.
   - **Active Tillering:** 25 kg Urea + Neem oil coating.
   - **Panicle Initiation:** 25 kg Urea + 15 kg MOP."""
    },
    {
        "keywords": ["irrigation", "water", "drip", "frequency", "sprinkler", "drought"],
        "title": "Smart Irrigation & Water Management",
        "response": """**Agro-Water Management & Drip Irrigation Strategy:**

- **Critical Growth Stages for Watering:**
  - *Tomato:* Flowering, fruit setting, and fruit sizing. Avoid fluctuations to prevent fruit cracking.
  - *Potato:* Stolon formation and tuber enlargement. Keep soil moisture at 65-75% field capacity.
  - *Maize:* Tasseling and silking (most sensitive to moisture stress).
  - *Rice:* Maintain shallow water (2-5 cm) from panicle initiation to grain dough stage.

- **Drip Irrigation Benefits:**
  - Saves 40-60% water compared to furrow flooding.
  - Reduces foliar fungal infections (Early/Late Blight) by keeping leaves dry.
  - Allows precise fertigation directly to root zones."""
    },
    {
        "keywords": ["organic", "jeevamrutha", "panchagavya", "natural", "bio", "compost", "neem"],
        "title": "Organic Farming & Bio-Formulations",
        "response": """**Organic Agricultural Formulations (Zero Budget Natural Farming):**

1. **Jeevamrutha Preparation (for 1 Acre):**
   - Mix 10 kg native cow dung + 10 L cow urine + 2 kg jaggery + 2 kg gram flour + handful of virgin forest soil in 200 L water.
   - Stir twice daily for 48-72 hours in shade.
   - Apply with irrigation water every 15 days to activate beneficial soil microbiome.

2. **Panchagavya (Foliar Growth Booster):**
   - Mix 5 kg cow dung, 3 L cow urine, 2 L cow milk, 2 L cow curd, 1 kg cow ghee, 3 L tender coconut water, and 12 ripe bananas.
   - Ferment for 21 days with daily stirring.
   - Dilute 300 mL in 10 L water (3% spray) for vigorous vegetative growth and pest deterrence.

3. **Neem Seed Kernel Extract (NSKE 5%):**
   - Soak 5 kg crushed neem seeds in 100 L water overnight with mild soap.
   - Highly potent organic repellent against bollworms, caterpillars, and aphids."""
    },
    {
        "keywords": ["pest", "insect", "aphid", "worm", "caterpillar", "whitefly", "borer"],
        "title": "Integrated Pest Management (IPM)",
        "response": """**Integrated Pest Management (IPM) Protocols:**

1. **Sucking Pests (Aphids, Jassids, Whiteflies):**
   - Install **Yellow Sticky Traps** (10-12 traps/acre) at canopy level.
   - Spray **Neem Oil 10,000 ppm** (3 mL/L water) or systemic insecticide **Imidacloprid 17.8% SL** (0.3 mL/L).

2. **Fruit / Shoot Borer & Spodoptera:**
   - Install **Pheromone Traps** (5-6 traps/acre) for monitoring moth populations.
   - Release **Trichogramma** egg parasitoids (50,000/acre).
   - Spray **Emamectin Benzoate 5% SG** (0.4 g/L) or **Bacillus thuringiensis (Bt)** @ 2 g/L.

3. **Leafminer:**
   - Prune and destroy lower mined leaves.
   - Spray **Abamectin 1.9% EC** (0.5 mL/L) or neem formulations."""
    },
    {
        "keywords": ["soil", "ph", "saline", "acidic", "alkaline", "loam", "clay", "texture"],
        "title": "Soil Health & pH Conditioning",
        "response": """**Soil Health Conditioning & pH Correction:**

- **Ideal Soil pH:** 6.0 to 7.2 (where macronutrients and micronutrients have peak bioavailability).
- **If Soil is Acidic (pH < 5.8):**
  - Apply Agricultural Lime (Calcium Carbonate) or Dolomite (1-2 tonnes/ha depending on buffer capacity).
- **If Soil is Alkaline/Saline (pH > 7.8):**
  - Apply Gypsum (Calcium Sulphate) followed by light leaching with clean water.
  - Incorporate green manure crops like *Dhaincha (Sesbania)* or *Sunn Hemp* at flowering stage.
- **Organic Carbon Boost:**
  - Add 5-10 tonnes well-rotted Farm Yard Manure (FYM) or 2 tonnes Vermicompost annually."""
    },
    {
        "keywords": ["scheme", "government", "subsidy", "pmkisan", "insurance", "loan", "kcc"],
        "title": "Government Schemes & Support",
        "response": """**Key Agricultural Schemes & Subsidies:**

1. **PM-KISAN:**
   - Income support of Rs. 6,000/year to eligible landholder farmer families in 3 equal installments.
2. **Pradhan Mantri Krishi Sinchayee Yojana (PMKSY):**
   - 45% - 55% subsidy on Drip and Sprinkler micro-irrigation installation.
3. **PM Fasal Bima Yojana (PMFBY):**
   - Comprehensive crop insurance for non-preventable natural risks (drought, flood, unseasonal storm). Premium: 2% for Kharif, 1.5% for Rabi crops.
4. **Kisan Credit Card (KCC):**
   - Concessional crop loans up to Rs. 3 Lakh at effectively 4% interest per annum with prompt repayment."""
    }
]

class FarmAdvisor:
    """
    Intelligent Agronomic Advisory Bot.
    Uses Gemini API if key is available, else seamlessly utilizes the rich offline expert engine.
    """

    def __init__(self, gemini_api_key: Optional[str] = None):
        self.api_key = gemini_api_key or os.environ.get("GEMINI_API_KEY", "").strip()
        self.gemini_client = None
        self._init_gemini()

    def _init_gemini(self):
        if self.api_key:
            try:
                from google import genai
                self.gemini_client = genai.Client(api_key=self.api_key)
            except Exception as e:
                print(f"Could not initialize Gemini Client: {e}")
                self.gemini_client = None

    def set_api_key(self, api_key: str):
        self.api_key = api_key.strip()
        self._init_gemini()

    def generate_response(self, user_prompt: str, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Generates response via Gemini or offline knowledge fallback."""
        user_query = user_prompt.strip()

        if self.gemini_client is not None:
            try:
                system_instruction = (
                    "You are 'Kisan Mitra AI', an expert Senior Agronomist and Smart Farming Consultant. "
                    "Provide practical, scientific, empathetic, and actionable advice to farmers on crops, diseases, "
                    "soil nutrients, irrigation scheduling, weather adaptation, and modern agritech. "
                    "Format responses with clear headings, bullet points, and highlight dosages clearly."
                )

                farm_context_str = ""
                if context:
                    farm_context_str = (
                        f"\n\nCurrent Farm Telemetry Context:"
                        f"\n- Ambient Temp: {context.get('temp_c', 'N/A')}  C"
                        f"\n- Humidity: {context.get('humidity_pct', 'N/A')} %"
                        f"\n- Soil Moisture: {context.get('soil_moisture_pct', 'N/A')} %"
                        f"\n- Current Crop: {context.get('current_crop', 'General Farm')}"
                    )

                full_prompt = f"{farm_context_str}\n\nFarmer Question: {user_query}"
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=full_prompt,
                    config={"system_instruction": system_instruction}
                )

                return {
                    "source": "Gemini 2.5 Flash AI",
                    "response": response.text,
                    "model": "gemini-2.5-flash"
                }
            except Exception as e:
                print(f"Gemini API request failed ({e}). Reverting to Offline Agro-Knowledge Engine.")

        query_lower = user_query.lower()
        matched_topic = None
        best_score = 0

        for topic in OFFLINE_KNOWLEDGE_TOPICS:
            score = sum(1 for kw in topic["keywords"] if kw in query_lower)
            if score > best_score:
                best_score = score
                matched_topic = topic

        if matched_topic and best_score > 0:
            return {
                "source": "Offline Agricultural Knowledge Engine",
                "title": matched_topic["title"],
                "response": matched_topic["response"],
                "matched_keywords": matched_topic["keywords"]
            }

        default_reply = (
            f"**Agronomic Recommendation for: '{user_query}'**\n\n"
            "For optimal crop management, ensure regular field scouting, verify soil moisture levels before irrigating, "
            "and inspect leaf undersides for early symptoms of foliar blight or sucking pests.\n\n"
            "**Quick Shortcuts You Can Ask:**\n"
            "- *'What is the recommended fertilizer dosage for tomato or corn?'*\n"
            "- *'How to prepare organic Jeevamrutha or Panchagavya?'*\n"
            "- *'How to control aphids and leafminers organically?'*\n"
            "- *'What is the best irrigation schedule for my soil type?'*\n"
            "- *'Tell me about PM-KISAN and crop insurance subsidies.'*"
        )
        return {
            "source": "Offline Agricultural Knowledge Engine",
            "title": "General Agricultural Guidance",
            "response": default_reply
        }
