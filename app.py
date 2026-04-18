import torch
import torch.nn as nn
import tempfile
import os
import base64
import shutil
import numpy as np
import cv2
from datetime import datetime
from torchvision import transforms
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from ultralytics import YOLO
from PIL import Image
import gradio as gr

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image as RLImage,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.enums import TA_CENTER

from insurance_logic import get_insurance_recommendation

# ── Device ───────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🖥️ Running on: {device}")

# ── Load YOLO ────────────────────────────────────────────────────
print("Loading YOLO model...")
yolo_model = YOLO("models/yolo_best.pt")
print("✅ YOLO loaded!")

# ── Load EfficientNet ────────────────────────────────────────────
print("Loading EfficientNet model...")
effnet = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
effnet.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(effnet.classifier[1].in_features, 256),
    nn.ReLU(),
    nn.Dropout(p=0.3),
    nn.Linear(256, 3),
)
checkpoint = torch.load(
    "models/efficientnet_best.pth", map_location=device, weights_only=False
)
effnet.load_state_dict(checkpoint["model_state_dict"])
effnet.eval()
effnet = effnet.to(device)
class_names = ["minor", "moderate", "severe"]
print("✅ EfficientNet loaded!")

# ── Transforms ───────────────────────────────────────────────────
effnet_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


# ── Logo ─────────────────────────────────────────────────────────
def get_logo_base64():
    if os.path.exists("logo.png"):
        with open("logo.png", "rb") as f:
            return "data:image/png;base64," + base64.b64encode(f.read()).decode()
    return ""


logo_b64 = get_logo_base64()


# ── Pipeline ─────────────────────────────────────────────────────
def run_dentiq_pipeline(image_path):
    # Step 1: YOLO Detection
    yolo_results = yolo_model(image_path, conf=0.25, verbose=False)
    result = yolo_results[0]

    annotated_bgr = result.plot(line_width=2, font_size=12)
    annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
    annotated_pil = Image.fromarray(annotated_rgb)

    detected_parts = []
    if result.boxes is not None and len(result.boxes) > 0:
        for box in result.boxes:
            class_id = int(box.cls[0])
            detected_parts.append(yolo_model.names[class_id])

    unique_parts = list(set(detected_parts))
    n_detections = len(detected_parts)

    # Step 2: EfficientNet Classification
    img = Image.open(image_path).convert("RGB")
    tensor = effnet_transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = effnet(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = torch.argmax(probs).item()
        severity = class_names[pred_idx]
        confidence_effnet = float(probs[pred_idx]) * 100

    if not unique_parts:
        unique_parts = [severity + "_damage"]

    # Step 3: Insurance Recommendation
    insurance = get_insurance_recommendation(severity, unique_parts, confidence_effnet)

    return {
        "annotated_image": annotated_pil,
        "severity": severity,
        "confidence": confidence_effnet,
        "prob_minor": float(probs[0]) * 100,
        "prob_moderate": float(probs[1]) * 100,
        "prob_severe": float(probs[2]) * 100,
        "detected_parts": detected_parts,
        "unique_parts": unique_parts,
        "n_detections": n_detections,
        "insurance": insurance,
    }


# ── PDF Generator ────────────────────────────────────────────────
def generate_pdf_report(result, input_image_path):
    output_path = os.path.join(tempfile.gettempdir(), "dentiq_report.pdf")

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    PRIMARY = colors.HexColor("#1a1a2e")
    ACCENT = colors.HexColor("#0f3460")
    GREEN = colors.HexColor("#2ecc71")
    ORANGE = colors.HexColor("#f39c12")
    RED = colors.HexColor("#e74c3c")
    LIGHT = colors.HexColor("#f8f9fa")
    MID = colors.HexColor("#dee2e6")

    sev_color = {"minor": GREEN, "moderate": ORANGE, "severe": RED}[result["severity"]]
    ins = result["insurance"]
    story = []

    title_style = ParagraphStyle(
        "T",
        fontSize=24,
        fontName="Helvetica-Bold",
        textColor=PRIMARY,
        alignment=TA_CENTER,
        spaceAfter=4,
    )
    sub_style = ParagraphStyle(
        "S",
        fontSize=10,
        textColor=colors.HexColor("#666"),
        alignment=TA_CENTER,
        spaceAfter=16,
    )
    sec_style = ParagraphStyle(
        "Sec",
        fontSize=12,
        fontName="Helvetica-Bold",
        textColor=PRIMARY,
        spaceBefore=12,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "B", fontSize=9, textColor=colors.HexColor("#333"), spaceAfter=4, leading=14
    )
    small_style = ParagraphStyle(
        "Sm", fontSize=8, textColor=colors.HexColor("#888"), alignment=TA_CENTER
    )

    # Header
    story.append(Paragraph("DentIQ", title_style))
    story.append(Paragraph("AI-Powered Car Damage Assessment Report", sub_style))
    story.append(HRFlowable(width="100%", thickness=2, color=ACCENT))
    story.append(Spacer(1, 0.3 * cm))

    # Metadata
    now = datetime.now().strftime("%d %B %Y, %I:%M %p")
    meta = [
        [
            "Report Date",
            now,
            "Report ID",
            f'DNTIQ-{datetime.now().strftime("%Y%m%d%H%M%S")}',
        ]
    ]
    mt = Table(meta, colWidths=[3 * cm, 7 * cm, 2.5 * cm, 4.5 * cm])
    mt.setStyle(
        TableStyle(
            [
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                ("FONTNAME", (2, 0), (2, -1), "Helvetica-Bold"),
                ("BACKGROUND", (0, 0), (-1, -1), LIGHT),
                ("GRID", (0, 0), (-1, -1), 0.5, MID),
                ("PADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(mt)
    story.append(Spacer(1, 0.3 * cm))

    # Severity Banner
    sev_emoji = {"minor": "🟢", "moderate": "🟡", "severe": "🔴"}
    sev_s = ParagraphStyle(
        "Sv",
        fontSize=14,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        alignment=TA_CENTER,
        backColor=sev_color,
        borderPadding=8,
    )
    story.append(
        Paragraph(
            f"{sev_emoji[result['severity']]}  SEVERITY: "
            f"{result['severity'].upper()}  |  "
            f"Confidence: {result['confidence']:.1f}%",
            sev_s,
        )
    )
    story.append(Spacer(1, 0.3 * cm))

    # Images
    story.append(Paragraph("Damage Analysis Images", sec_style))
    img_data = []
    if os.path.exists(input_image_path):
        img_data.append(RLImage(input_image_path, width=7 * cm, height=5.5 * cm))
    ann_path = os.path.join(tempfile.gettempdir(), "dentiq_annotated.jpg")
    result["annotated_image"].save(ann_path)
    if os.path.exists(ann_path):
        img_data.append(RLImage(ann_path, width=7 * cm, height=5.5 * cm))
    if len(img_data) == 2:
        it = Table([img_data], colWidths=[8.5 * cm, 8.5 * cm])
        it.setStyle(
            TableStyle(
                [
                    ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                    ("GRID", (0, 0), (-1, -1), 0.5, MID),
                    ("PADDING", (0, 0), (-1, -1), 4),
                ]
            )
        )
        story.append(it)
    story.append(Spacer(1, 0.3 * cm))

    # Detected Parts Table
    story.append(
        Paragraph(f"Detected Damage ({result['n_detections']} regions)", sec_style)
    )
    seen = {}
    for p in result["detected_parts"]:
        seen[p] = seen.get(p, 0) + 1

    if seen:
        pd_data = [["#", "Damaged Part", "Count"]]
        for i, (p, c) in enumerate(seen.items(), 1):
            pd_data.append([str(i), p.replace("-", " ").title(), str(c)])
        pt = Table(pd_data, colWidths=[1 * cm, 12 * cm, 4 * cm])
        pt.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 8),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                    ("GRID", (0, 0), (-1, -1), 0.5, MID),
                    ("PADDING", (0, 0), (-1, -1), 6),
                ]
            )
        )
        story.append(pt)
    story.append(Spacer(1, 0.3 * cm))

    # Cost Table
    story.append(Paragraph("Repair Cost Estimate", sec_style))
    cd = [["Damaged Part", "Min (Rs.)", "Max (Rs.)"]]
    for item in ins["breakdown"]:
        cd.append(
            [
                item["part"].replace("-", " ").title(),
                f"Rs.{item['min']:,}",
                f"Rs.{item['max']:,}",
            ]
        )
    cd.append(["TOTAL ESTIMATE", f"Rs.{ins['min_cost']:,}", f"Rs.{ins['max_cost']:,}"])
    ct = Table(cd, colWidths=[10 * cm, 4 * cm, 4 * cm])
    ct.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), PRIMARY),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("BACKGROUND", (0, -1), (-1, -1), ACCENT),
                ("TEXTCOLOR", (0, -1), (-1, -1), colors.white),
                ("FONTNAME", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("FONTSIZE", (0, 0), (-1, -1), 8),
                ("ROWBACKGROUNDS", (0, 1), (-1, -2), [colors.white, LIGHT]),
                ("GRID", (0, 0), (-1, -1), 0.5, MID),
                ("PADDING", (0, 0), (-1, -1), 6),
                ("ALIGN", (1, 0), (2, -1), "CENTER"),
            ]
        )
    )
    story.append(ct)
    story.append(Spacer(1, 0.3 * cm))

    # Insurance Recommendation
    story.append(Paragraph("Insurance Recommendation", sec_style))
    rec_color = RED if ins["should_claim"] else GREEN
    rs = ParagraphStyle(
        "R",
        fontSize=12,
        fontName="Helvetica-Bold",
        textColor=colors.white,
        alignment=TA_CENTER,
        backColor=rec_color,
        borderPadding=8,
    )
    story.append(Paragraph(ins["recommendation"], rs))
    story.append(Spacer(1, 0.2 * cm))
    story.append(Paragraph(f"<b>Reason:</b> {ins['reason']}", body_style))
    story.append(Spacer(1, 0.4 * cm))

    # Footer
    story.append(HRFlowable(width="100%", thickness=1, color=MID))
    story.append(Spacer(1, 0.2 * cm))
    story.append(
        Paragraph(
            "Disclaimer: This report is generated by DentIQ AI and is for "
            "reference only. Always consult a certified mechanic and your "
            "insurance provider before making claim decisions.",
            small_style,
        )
    )
    story.append(
        Paragraph(
            "Generated by DentIQ — Powered by YOLOv8 + EfficientNetB0", small_style
        )
    )

    doc.build(story)
    return output_path


# ── Predict Function ─────────────────────────────────────────────
def dentiq_predict(image):
    if image is None:
        return None, "Please upload an image.", "", "", None

    # Save input image to temp
    temp_path = os.path.join(tempfile.gettempdir(), "dentiq_input.jpg")
    if isinstance(image, np.ndarray):
        Image.fromarray(image).save(temp_path)
    else:
        image.save(temp_path)

    # Run pipeline
    result = run_dentiq_pipeline(temp_path)
    ins = result["insurance"]

    sev_emoji = {"minor": "🟢", "moderate": "🟡", "severe": "🔴"}
    emoji = sev_emoji[result["severity"]]

    # Damage Report
    damage_report = f"""{emoji} SEVERITY: {result['severity'].upper()}
{'━' * 35}
Confidence: {result['confidence']:.1f}%

Damage Probabilities:
  Minor    : {result['prob_minor']:.1f}%
  Moderate : {result['prob_moderate']:.1f}%
  Severe   : {result['prob_severe']:.1f}%

{'━' * 35}
Detected Damage ({result['n_detections']} regions):\n"""

    seen = {}
    for p in result["detected_parts"]:
        seen[p] = seen.get(p, 0) + 1
    for p, c in seen.items():
        damage_report += f"  * {p.replace('-', ' ').title()}"
        if c > 1:
            damage_report += f" (x{c})"
        damage_report += "\n"
    if not result["detected_parts"]:
        damage_report += "  * No specific regions detected\n"

    # Cost Report
    cost_report = f"""ESTIMATED REPAIR COST
{'━' * 35}
  Minimum : Rs.{ins['min_cost']:,}
  Maximum : Rs.{ins['max_cost']:,}

Part-wise Breakdown:
{'━' * 35}\n"""
    for item in ins["breakdown"]:
        cost_report += (
            f"  * {item['part'].replace('-', ' ').title()}\n"
            f"    Rs.{item['min']:,} to Rs.{item['max']:,}\n\n"
        )

    # Insurance Report
    claim_icon = "CLAIM INSURANCE" if ins["should_claim"] else "SELF REPAIR"
    insurance_report = f"""{claim_icon}
{'━' * 35}
{ins['recommendation']}

Reason:
{ins['reason']}

{'━' * 35}
{'Filing a claim is advised.' if ins['should_claim']
 else 'Consider handling out of pocket.'}

Note: Estimates are indicative.
Always consult a certified garage
and your insurance provider.
"""

    # Generate PDF
    pdf_path = generate_pdf_report(result, temp_path)

    return (
        result["annotated_image"],
        damage_report,
        cost_report,
        insurance_report,
        pdf_path,
    )


# ── Example Images ───────────────────────────────────────────────
example_paths = []
for f in ["examples/minor.jpg", "examples/moderate.jpg", "examples/severe.jpg"]:
    if os.path.exists(f):
        example_paths.append(f)

# ── Gradio UI ────────────────────────────────────────────────────
with gr.Blocks(title="DentIQ — AI Car Damage Assessment") as app:

    gr.Markdown(
        """
    <div style='text-align:center; padding:30px 0 20px 0;'>
        <h1 style='font-size:2.8em; margin-bottom:5px;'>🧠 DentIQ</h1>
        <h3 style='color:#a0aec0; font-weight:400; margin-top:0;
                letter-spacing:1px; font-size:1em;'>
            AI-POWERED CAR DAMAGE ASSESSMENT FOR INSURANCE CLAIMS
        </h3>
        <p style='color:#718096; font-size:0.85em; max-width:550px;
                margin:8px auto 0 auto; line-height:1.6;'>
            Upload a photo of your damaged car — DentIQ detects damage
            regions, assesses severity, estimates repair costs, and
            recommends insurance action.
        </p>
        <div style='margin-top:15px; display:flex;
                    justify-content:center; gap:20px;
                    font-size:0.82em; color:#4a9eff;'>
            <span>🤖 YOLOv8 Detection</span>
            <span>|</span>
            <span>🧠 EfficientNetB0</span>
            <span>|</span>
            <span>📄 PDF Reports</span>
        </div>
    </div>
    """
    )

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Upload Car Image")
            input_image = gr.Image(label="Car Photo", type="numpy", height=320)
            submit_btn = gr.Button("Analyze Damage", variant="primary", size="lg")
            gr.Markdown(
                """
            <div style='background:#1e293b; padding:12px;
                        border-radius:8px;
                        border-left:4px solid #3b82f6;
                        font-size:0.85em; color:#94a3b8;'>
            <b style='color:#e2e8f0;'>Tips for best results:</b><br>
            * Good lighting, no blur<br>
            * Capture the full damaged area<br>
            * Avoid extreme angles
            </div>
            """
            )

        with gr.Column(scale=1):
            gr.Markdown("### Damage Detection Result")
            output_image = gr.Image(label="YOLO Bounding Box Detection", height=320)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Damage Report")
            damage_output = gr.Textbox(label="Severity and Detected Parts", lines=16)
        with gr.Column():
            gr.Markdown("### Repair Cost Estimate")
            cost_output = gr.Textbox(label="Cost Breakdown", lines=16)
        with gr.Column():
            gr.Markdown("### Insurance Recommendation")
            insurance_output = gr.Textbox(label="Should You Claim?", lines=16)

    gr.Markdown("---")

    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown(
                """
            ### Download Full Report
            After analysis, download your complete DentIQ PDF report
            with all findings, cost breakdown and recommendation.
            """
            )
        with gr.Column(scale=1):
            pdf_output = gr.File(label="Download PDF Report")

    if example_paths:
        gr.Markdown("---")
        gr.Markdown("### Try These Example Images")
        gr.Examples(
            examples=[[p] for p in example_paths],
            inputs=input_image,
            label="Click any example to load it",
        )

    gr.Markdown("---")
    gr.Markdown(
        """
    <div style='text-align:center; color:#64748b;
                font-size:0.8em; padding:10px 0;'>
        DentIQ — Powered by YOLOv8 + EfficientNetB0<br>
        Repair cost estimates are approximate.
        Always consult a certified mechanic and your insurance provider.
    </div>
    """
    )

    submit_btn.click(
        fn=dentiq_predict,
        inputs=[input_image],
        outputs=[
            output_image,
            damage_output,
            cost_output,
            insurance_output,
            pdf_output,
        ],
    )

if __name__ == "__main__":
    app.launch(
        share=False,
        theme=gr.themes.Soft(primary_hue="blue", secondary_hue="slate"),
        css="footer {display:none !important}",
    )
