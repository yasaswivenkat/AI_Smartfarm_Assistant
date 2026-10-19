"""
Dataset and Sample Asset Generator for AI-Powered Smart Farm Assistant.
Generates:
1. Multi-year realistic historical weather dataset.
2. Initial Smart Inventory records in both JSON and CSV formats.
3. Realistic sample crop leaf images representing diseases for testing & CNN training.
"""

import os
import json
import csv
import math
import random
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
SAMPLE_LEAVES_DIR = os.path.join(DATA_DIR, "sample_leaves")
TRAIN_DATASET_DIR = os.path.join(DATA_DIR, "crop_disease_dataset")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAMPLE_LEAVES_DIR, exist_ok=True)
os.makedirs(TRAIN_DATASET_DIR, exist_ok=True)

def generate_historical_weather(csv_path: str = None, num_days: int = 1095) -> pd.DataFrame:
    """
    Generates realistic daily historical weather dataset with seasonal patterns,
    trends, autocorrelation, and stochastic noise.
    1095 days = 3 full agricultural years.
    """
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "historical_weather.csv")

    random.seed(42)
    np.random.seed(42)

    start_date = datetime(2023, 1, 1)
    dates = [start_date + timedelta(days=i) for i in range(num_days)]

    records = []
    prev_temp = 24.0
    prev_humidity = 65.0

    for i, date in enumerate(dates):
        day_of_year = date.timetuple().tm_yday
        month = date.month

        seasonal_temp = 26.0 + 8.5 * math.sin(2 * math.pi * (day_of_year - 80) / 365)
        mean_temp = 0.7 * prev_temp + 0.3 * seasonal_temp + np.random.normal(0, 1.8)
        prev_temp = mean_temp
        max_temp = mean_temp + np.random.uniform(3.5, 7.5)
        min_temp = mean_temp - np.random.uniform(3.5, 7.5)

        monsoon_factor = math.exp(-((month - 7.5) ** 2) / 4.0)
        rain_prob = 0.15 + 0.60 * monsoon_factor

        is_rainy = random.random() < rain_prob
        if is_rainy:
            rainfall = round(float(np.random.exponential(scale=18.0 * (1 + monsoon_factor))), 1)
            if random.random() < 0.08:
                rainfall += round(float(np.random.uniform(40.0, 95.0)), 1)
            condition = "Heavy Rain" if rainfall > 35 else ("Moderate Rain" if rainfall > 10 else "Light Rain")
        else:
            rainfall = 0.0
            condition = "Sunny" if random.random() > 0.35 else "Partly Cloudy"

        target_humidity = 50.0 + 35.0 * monsoon_factor + (20.0 if rainfall > 0 else 0)
        humidity = np.clip(0.6 * prev_humidity + 0.4 * target_humidity + np.random.normal(0, 4.0), 20.0, 99.0)
        prev_humidity = humidity

        pressure = 1013.25 - (mean_temp - 25.0) * 0.3 - (rainfall * 0.1) + np.random.normal(0, 2.5)
        wind_speed = np.clip(np.random.gamma(shape=3.0, scale=1.4) + (2.5 if rainfall > 15 else 0), 0.5, 28.0)
        solar_radiation = np.clip(22.0 - (12.0 * (humidity / 100.0)) - (8.0 if rainfall > 0 else 0) + np.random.normal(0, 1.5), 2.0, 30.0)

        records.append({
            "date": date.strftime("%Y-%m-%d"),
            "year": date.year,
            "month": date.month,
            "day": date.day,
            "day_of_year": day_of_year,
            "mean_temp_c": round(mean_temp, 2),
            "max_temp_c": round(max_temp, 2),
            "min_temp_c": round(min_temp, 2),
            "humidity_pct": round(humidity, 1),
            "pressure_hpa": round(pressure, 1),
            "wind_speed_ms": round(wind_speed, 2),
            "rainfall_mm": round(rainfall, 1),
            "solar_radiation_mj": round(solar_radiation, 2),
            "condition": condition
        })

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Generated {len(df)} historical weather records -> {csv_path}")
    return df

def generate_initial_inventory(json_path: str = None, csv_path: str = None):
    """
    Generates rich, pre-populated Smart Farm Inventory datasets in JSON and CSV.
    Covers Fertilizers, Pesticides, Seeds, and Equipment.
    """
    if json_path is None:
        json_path = os.path.join(DATA_DIR, "inventory.json")
    if csv_path is None:
        csv_path = os.path.join(DATA_DIR, "inventory.csv")

    today = datetime.now()

    items = [
        {
            "item_id": "FERT-001",
            "name": "Neem Coated Urea (46% N)",
            "category": "Fertilizer",
            "quantity": 18.0,
            "unit": "Bags (45kg)",
            "min_threshold": 10.0,
            "price_per_unit": 270.0,
            "location": "Warehouse Shed A - Bay 1",
            "expiry_date": (today + timedelta(days=360)).strftime("%Y-%m-%d"),
            "supplier": "National Fertilizers Ltd",
            "notes": "Essential nitrogen top-dressing for vegetative growth."
        },
        {
            "item_id": "FERT-002",
            "name": "DAP (Diammonium Phosphate 18-46-0)",
            "category": "Fertilizer",
            "quantity": 4.0,
            "unit": "Bags (50kg)",
            "min_threshold": 8.0,
            "price_per_unit": 1350.0,
            "location": "Warehouse Shed A - Bay 2",
            "expiry_date": (today + timedelta(days=240)).strftime("%Y-%m-%d"),
            "supplier": "IFFCO Kisan Agro",
            "notes": "Crucial root booster during sowing stage. LOW STOCK ALERT!"
        },
        {
            "item_id": "FERT-003",
            "name": "Muriate of Potash (MOP 60% K2O)",
            "category": "Fertilizer",
            "quantity": 12.0,
            "unit": "Bags (50kg)",
            "min_threshold": 6.0,
            "price_per_unit": 1700.0,
            "location": "Warehouse Shed A - Bay 3",
            "expiry_date": (today + timedelta(days=400)).strftime("%Y-%m-%d"),
            "supplier": "Coromandel International",
            "notes": "Enhances drought tolerance and fruit firmness."
        },
        {
            "item_id": "FERT-004",
            "name": "Organic Vermicompost Bio-Enriched",
            "category": "Fertilizer",
            "quantity": 45.0,
            "unit": "Bags (25kg)",
            "min_threshold": 15.0,
            "price_per_unit": 320.0,
            "location": "Shed C - Organic Pit",
            "expiry_date": (today + timedelta(days=180)).strftime("%Y-%m-%d"),
            "supplier": "GreenEarth Agro Organics",
            "notes": "Rich humus and microbial inoculants for soil conditioning."
        },
        {
            "item_id": "PEST-001",
            "name": "Neem Oil Bio-Pesticide (10000 PPM)",
            "category": "Pesticide",
            "quantity": 3.5,
            "unit": "Liters",
            "min_threshold": 5.0,
            "price_per_unit": 650.0,
            "location": "Chemical Cabinet - Rack 1",
            "expiry_date": (today + timedelta(days=150)).strftime("%Y-%m-%d"),
            "supplier": "BioCure Pest Solutions",
            "notes": "Broad-spectrum organic repellent for aphid, thrip, whitefly."
        },
        {
            "item_id": "PEST-002",
            "name": "Mancozeb 75% WP Fungicide",
            "category": "Pesticide",
            "quantity": 6.0,
            "unit": "Packets (1kg)",
            "min_threshold": 4.0,
            "price_per_unit": 480.0,
            "location": "Chemical Cabinet - Rack 2",
            "expiry_date": (today + timedelta(days=280)).strftime("%Y-%m-%d"),
            "supplier": "UPL Crop Protection",
            "notes": "Contact fungicide for Early and Late Blight prophylaxis."
        },
        {
            "item_id": "PEST-003",
            "name": "Trichoderma Viride Bio-Fungicide",
            "category": "Pesticide",
            "quantity": 2.0,
            "unit": "Bags (2kg)",
            "min_threshold": 5.0,
            "price_per_unit": 340.0,
            "location": "Bio-Cold Storage R-1",
            "expiry_date": (today + timedelta(days=60)).strftime("%Y-%m-%d"),
            "supplier": "Agrilife Naturals",
            "notes": "Antagonistic fungus against root rot and damping-off."
        },
        {
            "item_id": "SEED-001",
            "name": "Hybrid Tomato Seeds (Arka Rakshak)",
            "category": "Seeds",
            "quantity": 8.0,
            "unit": "Packets (10g)",
            "min_threshold": 3.0,
            "price_per_unit": 420.0,
            "location": "Seed Bank Lockbox S-1",
            "expiry_date": (today + timedelta(days=450)).strftime("%Y-%m-%d"),
            "supplier": "ICAR-IIHR Certified Seeds",
            "notes": "Triple disease resistant hybrid: ToLCV, Early Blight & Bacterial Wilt."
        },
        {
            "item_id": "SEED-002",
            "name": "Certified Seed Potato (Kufri Jyoti)",
            "category": "Seeds",
            "quantity": 150.0,
            "unit": "Kg",
            "min_threshold": 200.0,
            "price_per_unit": 38.0,
            "location": "Cold Store Room B",
            "expiry_date": (today + timedelta(days=90)).strftime("%Y-%m-%d"),
            "supplier": "Central Potato Research Institute",
            "notes": "Medium maturing, wide adaptability. Stock low for upcoming rabi planting."
        },
        {
            "item_id": "SEED-003",
            "name": "Sweet Corn Hybrid Seeds (Sugar 75)",
            "category": "Seeds",
            "quantity": 14.0,
            "unit": "Packets (1kg)",
            "min_threshold": 5.0,
            "price_per_unit": 1150.0,
            "location": "Seed Bank Lockbox S-2",
            "expiry_date": (today + timedelta(days=320)).strftime("%Y-%m-%d"),
            "supplier": "Syngenta India",
            "notes": "High brix sweetness, resistant to Common Rust."
        },
        {
            "item_id": "SEED-004",
            "name": "Pusa Basmati 1121 Paddy Seeds",
            "category": "Seeds",
            "quantity": 25.0,
            "unit": "Bags (10kg)",
            "min_threshold": 10.0,
            "price_per_unit": 890.0,
            "location": "Grain Seed Storage C",
            "expiry_date": (today + timedelta(days=500)).strftime("%Y-%m-%d"),
            "supplier": "IARI New Delhi",
            "notes": "Extra-long slender grain basmati rice, high export value."
        }
    ]

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=2)

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["item_id", "name", "category", "quantity", "unit", "min_threshold", "price_per_unit", "location", "expiry_date", "supplier", "notes"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(items)

    print(f"Generated {len(items)} inventory records -> {json_path} & {csv_path}")

def draw_realistic_leaf_image(crop: str, disease: str, size=(256, 256), seed: int = None) -> Image.Image:
    """
    Renders visually distinct, realistic synthetic leaf images with specific
    symptoms (target spots, necrotic lesions, pustules, blight, healthy vein textures).
    Used both as sample images for live testing and as high-fidelity dataset for the CNN.
    """
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    img = Image.new("RGB", size, (244, 246, 240))
    draw = ImageDraw.Draw(img)
    w, h = size
    cx, cy = w // 2, h // 2

    is_healthy = "healthy" in disease.lower()

    if "tomato" in crop.lower():
        leaf_color = (46, 125, 50) if is_healthy else (76, 115, 45)
        points = [
            (cx, cy - int(h * 0.42)),
            (cx + int(w * 0.22), cy - int(h * 0.20)),
            (cx + int(w * 0.32), cy),
            (cx + int(w * 0.20), cy + int(h * 0.25)),
            (cx + int(w * 0.05), cy + int(h * 0.42)),
            (cx, cy + int(h * 0.46)),
            (cx - int(w * 0.05), cy + int(h * 0.42)),
            (cx - int(w * 0.20), cy + int(h * 0.25)),
            (cx - int(w * 0.32), cy),
            (cx - int(w * 0.22), cy - int(h * 0.20))
        ]
    elif "potato" in crop.lower():
        leaf_color = (43, 110, 48) if is_healthy else (68, 105, 42)
        points = [
            (cx, cy - int(h * 0.40)),
            (cx + int(w * 0.26), cy - int(h * 0.15)),
            (cx + int(w * 0.30), cy + int(h * 0.10)),
            (cx + int(w * 0.16), cy + int(h * 0.32)),
            (cx, cy + int(h * 0.44)),
            (cx - int(w * 0.16), cy + int(h * 0.32)),
            (cx - int(w * 0.30), cy + int(h * 0.10)),
            (cx - int(w * 0.26), cy - int(h * 0.15))
        ]
    elif "corn" in crop.lower() or "maize" in crop.lower():
        leaf_color = (60, 140, 50) if is_healthy else (85, 125, 45)
        points = [
            (cx, cy - int(h * 0.46)),
            (cx + int(w * 0.18), cy - int(h * 0.20)),
            (cx + int(w * 0.22), cy + int(h * 0.15)),
            (cx + int(w * 0.12), cy + int(h * 0.45)),
            (cx - int(w * 0.12), cy + int(h * 0.45)),
            (cx - int(w * 0.22), cy + int(h * 0.15)),
            (cx - int(w * 0.18), cy - int(h * 0.20))
        ]
    else:
        leaf_color = (55, 135, 48) if is_healthy else (75, 120, 42)
        points = [
            (cx, cy - int(h * 0.46)),
            (cx + int(w * 0.14), cy - int(h * 0.10)),
            (cx + int(w * 0.16), cy + int(h * 0.20)),
            (cx, cy + int(h * 0.46)),
            (cx - int(w * 0.16), cy + int(h * 0.20)),
            (cx - int(w * 0.14), cy - int(h * 0.10))
        ]

    draw.polygon(points, fill=leaf_color, outline=(25, 75, 25))

    vein_color = (int(leaf_color[0] * 1.2), int(leaf_color[1] * 1.15), int(leaf_color[2] * 0.9))
    draw.line([(cx, cy - int(h * 0.40)), (cx, cy + int(h * 0.43))], fill=vein_color, width=3)
    for vy in range(-int(h * 0.30), int(h * 0.35), int(h * 0.08)):
        draw.line([(cx, cy + vy), (cx + int(w * 0.18), cy + vy + int(h * 0.05))], fill=vein_color, width=1)
        draw.line([(cx, cy + vy), (cx - int(w * 0.18), cy + vy + int(h * 0.05))], fill=vein_color, width=1)

    disease_lower = disease.lower()
    if "early_blight" in disease_lower or "early blight" in disease_lower:
        num_spots = random.randint(5, 9)
        for _ in range(num_spots):
            sx = cx + random.randint(-int(w * 0.18), int(w * 0.18))
            sy = cy + random.randint(-int(h * 0.25), int(h * 0.25))
            r = random.randint(12, 24)
            draw.ellipse([sx - r - 4, sy - r - 4, sx + r + 4, sy + r + 4], fill=(185, 175, 30))
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(62, 38, 20))
            draw.ellipse([sx - r + 4, sy - r + 4, sx + r - 4, sy + r - 4], fill=(95, 60, 32))
            draw.ellipse([sx - 3, sy - 3, sx + 3, sy + 3], fill=(35, 18, 10))

    elif "late_blight" in disease_lower or "late blight" in disease_lower:
        num_blotches = random.randint(4, 7)
        for _ in range(num_blotches):
            bx = cx + random.randint(-int(w * 0.20), int(w * 0.20))
            by = cy + random.randint(-int(h * 0.30), int(h * 0.30))
            bw, bh = random.randint(22, 45), random.randint(20, 40)
            draw.ellipse([bx - bw, by - bh, bx + bw, by + bh], fill=(45, 42, 35))
            draw.ellipse([bx - int(bw * 0.7), by - int(bh * 0.7), bx + int(bw * 0.7), by + int(bh * 0.7)], fill=(28, 22, 18))
            draw.arc([bx - bw, by - bh, bx + bw, by + bh], 0, 360, fill=(160, 165, 150), width=2)

    elif "rust" in disease_lower:
        num_pustules = random.randint(25, 45)
        for _ in range(num_pustules):
            px = cx + random.randint(-int(w * 0.14), int(w * 0.14))
            py = cy + random.randint(-int(h * 0.35), int(h * 0.35))
            pr_x, pr_y = random.randint(3, 6), random.randint(2, 4)
            draw.ellipse([px - pr_x, py - pr_y, px + pr_x, py + pr_y], fill=(175, 70, 20), outline=(130, 45, 10))

    elif "leaf_blight" in disease_lower or "northern leaf blight" in disease_lower:
        num_lesions = random.randint(4, 8)
        for _ in range(num_lesions):
            lx = cx + random.randint(-int(w * 0.12), int(w * 0.12))
            ly = cy + random.randint(-int(h * 0.30), int(h * 0.30))
            lw, lh = random.randint(6, 12), random.randint(25, 55)
            draw.ellipse([lx - lw, ly - lh, lx + lw, ly + lh], fill=(140, 115, 75), outline=(95, 75, 45))

    elif "blast" in disease_lower:
        num_blast = random.randint(6, 12)
        for _ in range(num_blast):
            bx = cx + random.randint(-int(w * 0.10), int(w * 0.10))
            by = cy + random.randint(-int(h * 0.32), int(h * 0.32))
            diamond = [
                (bx, by - random.randint(12, 22)),
                (bx + random.randint(5, 9), by),
                (bx, by + random.randint(12, 22)),
                (bx - random.randint(5, 9), by)
            ]
            draw.polygon(diamond, fill=(160, 150, 140), outline=(120, 45, 30))

    elif "brown_spot" in disease_lower or "brown spot" in disease_lower:
        num_spots = random.randint(15, 30)
        for _ in range(num_spots):
            sx = cx + random.randint(-int(w * 0.11), int(w * 0.11))
            sy = cy + random.randint(-int(h * 0.34), int(h * 0.34))
            r = random.randint(3, 7)
            draw.ellipse([sx - r - 2, sy - r - 2, sx + r + 2, sy + r + 2], fill=(190, 180, 40))
            draw.ellipse([sx - r, sy - r, sx + r, sy + r], fill=(85, 45, 25))

    img = img.filter(ImageFilter.SMOOTH_MORE)
    return img

def generate_all_sample_leaves():
    """Generates a gallery of sample leaf images for interactive testing."""
    samples = [
        ("Tomato", "early_blight", "tomato_early_blight.jpg"),
        ("Tomato", "late_blight", "tomato_late_blight.jpg"),
        ("Tomato", "healthy", "tomato_healthy.jpg"),
        ("Potato", "early_blight", "potato_early_blight.jpg"),
        ("Potato", "late_blight", "potato_late_blight.jpg"),
        ("Potato", "healthy", "potato_healthy.jpg"),
        ("Corn", "common_rust", "corn_common_rust.jpg"),
        ("Corn", "northern_leaf_blight", "corn_leaf_blight.jpg"),
        ("Corn", "healthy", "corn_healthy.jpg"),
        ("Rice", "blast", "rice_blast.jpg"),
        ("Rice", "brown_spot", "rice_brown_spot.jpg"),
        ("Rice", "healthy", "rice_healthy.jpg")
    ]

    for crop, disease, filename in samples:
        filepath = os.path.join(SAMPLE_LEAVES_DIR, filename)
        img = draw_realistic_leaf_image(crop, disease, size=(256, 256), seed=42)
        img.save(filepath, "JPEG", quality=92)
    print(f"Generated {len(samples)} sample leaf images in {SAMPLE_LEAVES_DIR}")

def generate_crop_disease_dataset(samples_per_class: int = 40):
    """
    Generates balanced training/validation dataset partitioned by disease class
    folders for CNN training.
    """
    classes = [
        ("Tomato___Early_blight", "Tomato", "early_blight"),
        ("Tomato___Late_blight", "Tomato", "late_blight"),
        ("Tomato___Healthy", "Tomato", "healthy"),
        ("Potato___Early_blight", "Potato", "early_blight"),
        ("Potato___Late_blight", "Potato", "late_blight"),
        ("Potato___Healthy", "Potato", "healthy"),
        ("Corn___Common_rust", "Corn", "rust"),
        ("Corn___Northern_Leaf_Blight", "Corn", "leaf_blight"),
        ("Corn___Healthy", "Corn", "healthy"),
        ("Rice___Blast", "Rice", "blast"),
        ("Rice___Brown_Spot", "Rice", "brown_spot"),
        ("Rice___Healthy", "Rice", "healthy")
    ]

    for class_name, crop, disease in classes:
        class_dir = os.path.join(TRAIN_DATASET_DIR, class_name)
        os.makedirs(class_dir, exist_ok=True)
        for i in range(samples_per_class):
            img_path = os.path.join(class_dir, f"{class_name}_{i:03d}.jpg")
            img = draw_realistic_leaf_image(crop, disease, size=(128, 128), seed=1000 + i * 7)
            if random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                img = img.rotate(random.randint(-15, 15), fillcolor=(244, 246, 240))
            img.save(img_path, "JPEG", quality=90)

    print(f"Generated training dataset: {len(classes)} classes x {samples_per_class} images in {TRAIN_DATASET_DIR}")

if __name__ == "__main__":
    generate_historical_weather()
    generate_initial_inventory()
    generate_all_sample_leaves()
    generate_crop_disease_dataset(samples_per_class=35)
