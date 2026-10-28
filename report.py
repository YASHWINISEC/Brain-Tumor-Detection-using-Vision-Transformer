from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from datetime import datetime

def generate_report(filepath, prediction, confidence):
    c = canvas.Canvas(filepath, pagesize=A4)
    width, height = A4

    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height-50, "Brain Tumor AI Analysis Report")

    c.setFont("Helvetica", 12)
    c.drawString(50, height-100, f"Date: {datetime.now().strftime('%d-%m-%Y %H:%M')}")

    c.line(50, height-120, width-50, height-120)

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-160, "Prediction Summary")

    c.setFont("Helvetica", 12)
    c.drawString(50, height-190, f"Tumor Type: {prediction}")
    c.drawString(50, height-215, f"Confidence: {confidence}%")

    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, height-260, "Clinical Note")

    c.setFont("Helvetica", 11)
    c.drawString(50, height-290,
        "This AI-generated report is intended for clinical decision support only.")
    c.drawString(50, height-310,
        "Final diagnosis should be confirmed by a qualified radiologist.")

    c.showPage()
    c.save()
