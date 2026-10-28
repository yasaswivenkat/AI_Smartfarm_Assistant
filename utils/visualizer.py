"""
Data Visualization Utility Module.
Implements Matplotlib, Seaborn, and Plotly charting routines
for weather forecasting, ML trends, crop disease analysis, and farm inventory.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional

sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    "font.sans-serif": "DejaVu Sans",
    "axes.edgecolor": "#cccccc",
    "axes.linewidth": 0.8,
    "grid.color": "#ebebeb",
    "grid.linestyle": "--"
})

def plot_weather_historical_matplotlib(df: pd.DataFrame) -> plt.Figure:
    """
    Renders a multi-panel Matplotlib figure visualizing 1-year historical weather trends:
    1. Daily Mean Temperature with Min/Max range envelope.
    2. Daily Precipitation (mm) bars with rolling weekly average.
    """
    sample_df = df.tail(365).copy()
    sample_df["date"] = pd.to_datetime(sample_df["date"])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [2, 1.2]})

    ax1.plot(sample_df["date"], sample_df["mean_temp_c"], color="#2e7d32", label="Mean Temp ( C)", linewidth=2)
    ax1.fill_between(
        sample_df["date"],
        sample_df["min_temp_c"],
        sample_df["max_temp_c"],
        color="#a5d6a7",
        alpha=0.35,
        label="Min-Max Range"
    )
    ax1.set_ylabel("Temperature ( C)", fontsize=11, fontweight="bold", color="#1b5e20")
    ax1.set_title("Annual Agro-Meteorological Cycle & Temperature Trajectory", fontsize=13, fontweight="bold", pad=12)
    ax1.legend(loc="upper right", frameon=True)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2.bar(sample_df["date"], sample_df["rainfall_mm"], color="#0288d1", alpha=0.7, label="Daily Rainfall (mm)", width=1.0)
    if "rainfall_mm" in sample_df:
        rolling_rain = sample_df["rainfall_mm"].rolling(window=14, min_periods=1).mean()
        ax2.plot(sample_df["date"], rolling_rain, color="#01579b", linewidth=1.8, label="14-Day Trend")
    ax2.set_ylabel("Precipitation (mm)", fontsize=11, fontweight="bold", color="#01579b")
    ax2.set_xlabel("Date", fontsize=11, fontweight="bold")
    ax2.legend(loc="upper right", frameon=True)
    ax2.grid(True, linestyle=":", alpha=0.6)

    fig.tight_layout()
    return fig

def plot_weather_correlation_seaborn(df: pd.DataFrame) -> plt.Figure:
    """
    Renders a Seaborn correlation heatmap displaying inter-variable relationships
    between Temperature, Humidity, Pressure, Wind, Rainfall, and Solar Radiation.
    """
    cols = ["mean_temp_c", "humidity_pct", "pressure_hpa", "wind_speed_ms", "rainfall_mm", "solar_radiation_mj"]
    cols_exist = [c for c in cols if c in df.columns]
    corr = df[cols_exist].corr()

    labels_map = {
        "mean_temp_c": "Temperature ( C)",
        "humidity_pct": "Humidity (%)",
        "pressure_hpa": "Pressure (hPa)",
        "wind_speed_ms": "Wind Speed (m/s)",
        "rainfall_mm": "Rainfall (mm)",
        "solar_radiation_mj": "Solar Rad (MJ/m^2)"
    }
    rename_cols = [labels_map.get(c, c) for c in cols_exist]
    corr.columns = rename_cols
    corr.index = rename_cols

    fig, ax = plt.subplots(figsize=(8, 6))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(
        corr,
        mask=mask,
        cmap=cmap,
        vmax=1.0,
        vmin=-1.0,
        center=0,
        square=True,
        linewidths=1.0,
        cbar_kws={"shrink": 0.8, "label": "Pearson Correlation Coefficient"},
        annot=True,
        fmt=".2f",
        ax=ax
    )
    ax.set_title("Seaborn Correlation Matrix: Meteorological Factors", fontsize=12, fontweight="bold", pad=12)
    fig.tight_layout()
    return fig

def plot_rainfall_distribution_seaborn(df: pd.DataFrame) -> plt.Figure:
    """
    Renders Seaborn KDE and boxplot distributions of monthly rainfall and humidity.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    rainy_days = df[df["rainfall_mm"] > 0]
    sns.histplot(rainy_days["rainfall_mm"], kde=True, color="#0288d1", ax=ax1, bins=25)
    ax1.set_title("Rainfall Volume Density (Rainy Days)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Precipitation (mm)")
    ax1.set_ylabel("Frequency Count")

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    plot_df = df.copy()
    plot_df["month_name"] = plot_df["month"].apply(lambda m: month_names[int(m) - 1])
    sns.boxplot(x="month", y="mean_temp_c", data=plot_df, palette="YlOrRd", ax=ax2)
    ax2.set_title("Seasonal Temperature Distribution by Month", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Month")
    ax2.set_ylabel("Mean Temperature ( C)")

    fig.tight_layout()
    return fig

def plot_feature_importance_matplotlib(feature_importances: Dict[str, float]) -> plt.Figure:
    """Renders Matplotlib horizontal bar chart of ML model feature importances."""
    sorted_feats = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)[:10]
    names = [x[0].replace("_", " ").title() for x in sorted_feats][::-1]
    values = [x[1] for x in sorted_feats][::-1]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    bars = ax.barh(names, values, color="#388e3c", alpha=0.85, edgecolor="#1b5e20")
    ax.set_xlabel("Gini Feature Importance Score", fontsize=10, fontweight="bold")
    ax.set_title("ML Model Feature Importance (Weather Predictor)", fontsize=11, fontweight="bold")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=9, color="#2e7d32")

    ax.grid(True, linestyle=":", alpha=0.5, axis="x")
    fig.tight_layout()
    return fig

def plot_cnn_training_history_matplotlib(history: Dict[str, List[float]]) -> plt.Figure:
    """Renders dual-axis Matplotlib chart showing CNN training and validation accuracy/loss."""
    epochs = range(1, len(history.get("accuracy", [])) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, history.get("accuracy", []), "o-", color="#2e7d32", label="Train Accuracy", linewidth=2)
    ax1.plot(epochs, history.get("val_accuracy", []), "s--", color="#1565c0", label="Val Accuracy", linewidth=2)
    ax1.set_title("CNN Training vs Validation Accuracy", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0, 1.05)
    ax1.legend(loc="lower right")
    ax1.grid(True, linestyle=":")

    ax2.plot(epochs, history.get("loss", []), "o-", color="#c62828", label="Train Loss", linewidth=2)
    ax2.plot(epochs, history.get("val_loss", []), "s--", color="#e65100", label="Val Loss", linewidth=2)
    ax2.set_title("CNN Cross-Entropy Loss Convergence", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.legend(loc="upper right")
    ax2.grid(True, linestyle=":")

    fig.tight_layout()
    return fig

def plot_inventory_valuation_plotly(items: List[Any]) -> go.Figure:
    """Interactive Plotly Treemap of Farm Inventory Stock Valuation."""
    records = []
    for item in items:
        records.append({
            "name": item.name,
            "category": item.category,
            "quantity": item.quantity,
            "unit": item.unit,
            "value": item.total_value,
            "status": "Low Stock" if item.is_low_stock else "Normal"
        })
    df = pd.DataFrame(records)

    fig = px.treemap(
        df,
        path=["category", "name"],
        values="value",
        color="category",
        color_discrete_map={
            "Fertilizer": "#2e7d32",
            "Pesticide": "#c62828",
            "Seeds": "#f57f17",
            "Equipment": "#1565c0",
            "Other": "#607d8b"
        },
        title="Farm Inventory Valuation Hierarchy (Rs. )"
    )
    fig.update_layout(margin=dict(t=40, l=10, r=10, b=10))
    return fig

def plot_forecast_plotly(forecast_days: List[Dict[str, Any]]) -> go.Figure:
    """Interactive dual-axis Plotly temperature & rainfall probability forecast chart."""
    df = pd.DataFrame(forecast_days)

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["max_temp_c"],
        mode="lines+markers",
        name="Max Temp ( C)",
        line=dict(color="#d32f2f", width=2.5)
    ))

    fig.add_trace(go.Scatter(
        x=df["date"],
        y=df["min_temp_c"],
        mode="lines+markers",
        name="Min Temp ( C)",
        line=dict(color="#1976d2", width=2.5)
    ))

    fig.add_trace(go.Bar(
        x=df["date"],
        y=df["rain_prob_pct"],
        name="Rain Probability (%)",
        yaxis="y2",
        marker_color="rgba(2, 136, 209, 0.45)"
    ))

    fig.update_layout(
        title="7-Day ML Weather Trajectory & Precipitation Likelihood",
        xaxis=dict(title="Date"),
        yaxis=dict(title="Temperature ( C)", range=[10, 45]),
        yaxis2=dict(
            title="Rain Probability (%)",
            overlaying="y",
            side="right",
            range=[0, 100]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig

def plot_soil_telemetry_plotly(history_records: List[Dict[str, Any]]) -> go.Figure:
    """Interactive Plotly time-series of 24h IoT soil moisture and temperature."""
    df = pd.DataFrame(history_records)

    fig = go.Figure()

    fig.add_hrect(
        y0=45, y1=75,
        fillcolor="rgba(76, 175, 80, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="Optimal Field Capacity (45-75%)",
        annotation_position="top left"
    )

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["soil_moisture_pct"],
        mode="lines+markers",
        name="Soil Moisture (%)",
        line=dict(color="#2e7d32", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df["time"],
        y=df["soil_temp_c"],
        mode="lines",
        name="Soil Temp ( C)",
        yaxis="y2",
        line=dict(color="#f57c00", width=2, dash="dash")
    ))

    fig.update_layout(
        title="24-Hour IoT Soil Moisture & Subsurface Temperature Telemetry",
        xaxis=dict(title="Time of Day"),
        yaxis=dict(title="Soil Moisture (%)", range=[10, 95]),
        yaxis2=dict(
            title="Soil Temperature ( C)",
            overlaying="y",
            side="right",
            range=[15, 38]
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white",
        margin=dict(l=40, r=40, t=60, b=40)
    )
    return fig
