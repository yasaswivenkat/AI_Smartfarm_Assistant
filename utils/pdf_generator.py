"""
PDF Diagnostic Prescription and Farm Report Generator using ReportLab.
Produces professional, printable agronomic prescription sheets for farmers.
"""

import io
from datetime import datetime
from typing import Dict, Any
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle

def generate_diagnostic_pdf(diagnostic_data: Dict[str, Any], weather_data: Dict[str, Any]) -> bytes:
    """Generates PDF bytes of farm diagnosis and weather prescription."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=20,
        textColor=colors.HexColor("#064e3b"),
        spaceAfter=6
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=10,
        textColor=colors.HexColor("#475569"),
        spaceAfter=14
    )
    heading2_style = ParagraphStyle(
        "SectionHeading",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=13,
        textColor=colors.HexColor("#0f172a"),
        spaceBefore=10,
        spaceAfter=6
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#1e293b")
    )

    story = []

    story.append(Paragraph("AI-POWERED SMART FARM ASSISTANT", title_style))
    story.append(Paragraph(f"Official Agronomic Diagnosis & Prescription  -  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=colors.HexColor("#10b981"), spaceAfter=14))

    diag = diagnostic_data.get("diagnostic", {})
    crop = diag.get("crop", "Crop")
    disease = diag.get("disease_name", "Condition")
    severity = diag.get("severity", "Moderate")
    pathogen = diag.get("pathogen", "Pathogen")
    conf = diagnostic_data.get("confidence_pct", 0)

    story.append(Paragraph("1. CROP PATHOLOGY DIAGNOSIS", heading2_style))
    diag_table_data = [
        [Paragraph("<b>Target Crop:</b>", body_style), Paragraph(crop, body_style),
         Paragraph("<b>Diagnosis:</b>", body_style), Paragraph(f"<b>{disease}</b>", body_style)],
        [Paragraph("<b>Pathogen:</b>", body_style), Paragraph(f"<i>{pathogen}</i>", body_style),
         Paragraph("<b>Confidence:</b>", body_style), Paragraph(f"{conf}% (CNN)", body_style)],
        [Paragraph("<b>Severity Level:</b>", body_style), Paragraph(severity, body_style),
         Paragraph("<b>Safe PHI Wait:</b>", body_style), Paragraph(f"{diag.get('safe_harvest_wait_days', 7)} Days", body_style)]
    ]
    t_diag = Table(diag_table_data, colWidths=[90, 180, 90, 180])
    t_diag.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_diag)
    story.append(Spacer(1, 10))

    story.append(Paragraph("2. PRESCRIBED TREATMENT PROTOCOL", heading2_style))
    story.append(Paragraph(f"<b>Immediate Action:</b> {diag.get('immediate_action', 'Scout field and isolate infected area.')}", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Organic & Bio-Control Formulations:</b>", body_style))
    for org in diag.get("organic_treatment", []):
        story.append(Paragraph(f" -  {org}", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Chemical Fungicides & Dosage:</b>", body_style))
    for chem in diag.get("chemical_treatment", []):
        story.append(Paragraph(f" -  {chem}", body_style))
    story.append(Spacer(1, 6))

    story.append(Paragraph("<b>Preventative Management:</b>", body_style))
    story.append(Paragraph(diag.get("prevention", "Maintain crop rotation."), body_style))
    story.append(Spacer(1, 12))

    story.append(Paragraph("3. LOCAL METEOROLOGICAL CONTEXT", heading2_style))
    weather_table_data = [
        [Paragraph("<b>Location:</b>", body_style), Paragraph(weather_data.get("city", "Bangalore"), body_style),
         Paragraph("<b>Temperature:</b>", body_style), Paragraph(f"{weather_data.get('temp_c')}  C", body_style)],
        [Paragraph("<b>Humidity:</b>", body_style), Paragraph(f"{weather_data.get('humidity_pct')}%", body_style),
         Paragraph("<b>Condition:</b>", body_style), Paragraph(weather_data.get("condition", "Clear"), body_style)]
    ]
    t_w = Table(weather_table_data, colWidths=[90, 180, 90, 180])
    t_w.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f0fdf4")),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#bbf7d0")),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))
    story.append(t_w)
    story.append(Spacer(1, 14))

    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#e2e8f0"), spaceAfter=10))
    disclaimer_style = ParagraphStyle(
        "Disclaimer",
        parent=styles["Normal"],
        fontName="Helvetica-Oblique",
        fontSize=8,
        textColor=colors.HexColor("#64748b")
    )
    story.append(Paragraph(
        "Notice: This prescription was generated by the AI-Powered Smart Farm Assistant utilizing deep convolutional neural networks and agro-meteorological predictive modeling. Follow local agricultural extension safety guidelines before pesticide handling.",
        disclaimer_style
    ))

    doc.build(story)
    buffer.seek(0)
    return buffer.read()
