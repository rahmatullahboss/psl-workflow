from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas


def generate_sample_documents(output_dir: str | Path) -> dict[str, Path]:
    """Create synthetic clean and messy legal documents for demos and tests."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    clean_pdf = output / "clean_contract.pdf"
    messy_image = output / "messy_notice.png"
    scanned_pdf = output / "scanned_notice.pdf"

    _write_clean_pdf(clean_pdf)
    _write_messy_image(messy_image)
    _write_scanned_pdf(scanned_pdf, messy_image)

    return {"clean_pdf": clean_pdf, "messy_image": messy_image, "scanned_pdf": scanned_pdf}


def _write_clean_pdf(path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    text = c.beginText(72, 720)
    text.setFont("Helvetica", 11)
    lines = [
        "Pearson Specter Litt - Internal Contract Packet",
        "Matter: Wayne Logistics acquisition review",
        "Clause 4.2 Notice Before Termination:",
        "Either party must provide ten business days written notice before termination.",
        "Clause 7.1 Termination Fee:",
        "A termination fee of USD 2,500,000 is payable if Wayne exits after diligence.",
        "Clause 9.3 Confidentiality:",
        "The parties must keep diligence materials confidential for three years.",
    ]
    for line in lines:
        text.textLine(line)
    c.drawText(text)
    c.showPage()
    text = c.beginText(72, 720)
    text.setFont("Helvetica", 11)
    for line in [
        "Operator note:",
        "The client asked whether immediate termination is available.",
        "The record supports a notice-first recommendation and a high risk label.",
    ]:
        text.textLine(line)
    c.drawText(text)
    c.save()


def _write_messy_image(path: Path) -> None:
    image = Image.new("RGB", (1250, 760), color=(246, 244, 238))
    draw = ImageDraw.Draw(image)
    lines = [
        "HANDWRITTEN CLIENT NOTE - PEARSON SPECTER LITT",
        "Need notice served before termination.",
        "Risk: High if Wayne exits before diligence closes.",
        "Ask associate to cite agreement page and fee clause.",
    ]
    y = 110
    for line in lines:
        draw.text((80, y), line, fill=(45, 45, 45))
        y += 90
    for x in range(30, 1210, 160):
        draw.line((x, 55, x + 95, 710), fill=(205, 198, 188), width=2)
    for y_line in range(80, 720, 110):
        draw.line((55, y_line, 1180, y_line + 8), fill=(218, 210, 200), width=1)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.7))
    image.save(path)


def _write_scanned_pdf(path: Path, image_path: Path) -> None:
    c = canvas.Canvas(str(path), pagesize=letter)
    c.drawImage(str(image_path), 35, 190, width=540, height=330, preserveAspectRatio=True)
    c.save()
