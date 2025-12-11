import argparse
import json
from pathlib import Path
from typing import final

from reportlab.graphics.barcode import qr
from reportlab.graphics import renderPDF
from reportlab.graphics.shapes import Drawing
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfgen import canvas as _canvas
from reportlab.platypus import (
    Flowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from exam import Exam
from util import xor_encrypt_to_hex

SECRET_KEY = "EXAMKEY"  # for encoding the exam info in the code on the 1st page

NUMERIC_SYMBOL = "[#]"  # marks numeric-only boxes
MULTI_MCQ_SYMBOL = "[*]"  # marks MCQs with multiple correct answers

MM = 72.0 / 25.4
A4_W, A4_H = A4
MARGIN = 12 * MM
GUTTER = 3 * MM

FIDUCIAL_RADIUS = 2 * MM
FIDUCIAL_OFFSET = 10 * MM

BUBBLE_RADIUS = 2 * MM

# box constants
BOX_PADDING = 3 * MM

BOX_TABLE_STYLE = TableStyle(
    [
        ("BOX", (0, 0), (-1, -1), 1, colors.black),
        ("LEFTPADDING", (0, 0), (-1, -1), BOX_PADDING),
        ("RIGHTPADDING", (0, 0), (-1, -1), BOX_PADDING),
        ("TOPPADDING", (0, 0), (-1, -1), BOX_PADDING),
        ("BOTTOMPADDING", (0, 0), (-1, -1), BOX_PADDING),
    ]
)

def draw_page_fiducials(c: _canvas.Canvas, _):
    for x, y in [
        (FIDUCIAL_OFFSET, A4_H - FIDUCIAL_OFFSET),
        (A4_W - FIDUCIAL_OFFSET, A4_H - FIDUCIAL_OFFSET),
        (A4_W - FIDUCIAL_OFFSET, FIDUCIAL_OFFSET),
        (FIDUCIAL_OFFSET, FIDUCIAL_OFFSET),
    ]:
        c.circle(x, y, FIDUCIAL_RADIUS, stroke=0, fill=1)


def create_qrcode(encoded_payload: str) -> Drawing:
    # create qrcode
    qr_size = 20 * MM 

    qr_widget = qr.QrCodeWidget(encoded_payload)
    bounds = qr_widget.getBounds()
    w = bounds[2] - bounds[0]
    h = bounds[3] - bounds[1]

    # scale qrcode
    transform = [qr_size / w, 0, 0, qr_size / h, 0, 0]
    
    # create a Drawing object to hold the qrcode
    d = Drawing(qr_size, qr_size, transform=transform)
    d.add(qr_widget)

    return d

@final
class MCQPanel(Flowable):
    def __init__(
        self,
        choices: tuple[str, ...],
        spacing_mm: int = 8,
    ):
        super().__init__()
        self.choices = choices
        self.bubble = 2 * BUBBLE_RADIUS
        self.spacing = spacing_mm * MM

        # width: one bubble + spacing per choice
        self.width = len(choices) * self.bubble + (len(choices) - 1) * self.spacing
        self.height = self.bubble

    def wrap(self, aW, aH):
        return self.width, self.height

    def draw(self):
        c = self.canv
        r = BUBBLE_RADIUS
        y = r  # center vertically

        for i, ch in enumerate(self.choices):
            x = i * (self.bubble + self.spacing)
            # bubble
            c.circle(x + r, y, r, stroke=1, fill=0)

            # label to the right of the bubble
            c.drawString(x + self.bubble + 4, y - 3, ch)


@final
class NumericPanel(Flowable):
    def __init__(self, width_mm: int = 40, height_mm: int = 7):
        super().__init__()
        self.width = width_mm * MM
        self.height = height_mm * MM
        self.spacing = 4 * MM

    def wrap(self, aW, aH):
        return self.width, self.height

    def draw(self):
        c = self.canv
        # draw the big box (starting at y=0)
        c.rect(0, 0, self.width, self.height, stroke=1, fill=0)

        # draw numeric indicator just after the box
        c.drawString(self.width + self.spacing, self.height * 0.5, NUMERIC_SYMBOL)


def generate_exam_pdf(exam: Exam, variant_idx: int, out_path: str):
    doc = SimpleDocTemplate(
        out_path,
        pagesize=A4,
        leftMargin=MARGIN + GUTTER,
        rightMargin=MARGIN,
        topMargin=MARGIN,
        bottomMargin=MARGIN,
    )

    # set the styles
    styles = getSampleStyleSheet()

    styles["BodyText"].fontName = "Times-Roman"
    styles["BodyText"].fontSize = 12
    styles["BodyText"].leading = 14

    styles["Normal"].fontName = styles["BodyText"].fontName
    styles["Normal"].fontSize = styles["BodyText"].fontSize
    styles["Normal"].leading = styles["BodyText"].leading

    # title style
    styles.add(
        ParagraphStyle(
            name="ExamTitle",
            parent=styles["Title"],
            fontName="Times-Bold",
            fontSize=24,
            alignment=TA_CENTER,
            spaceAfter=12 * MM,
        )
    )

    # informations to encode in the barcode
    payload = json.dumps(
        {
            "exam_id": exam["exam_id"],
            "variant": variant_idx,
        }
    )

    story = []

    # student identification section
    story.append(Spacer(1, 5 * MM))

    name_label = Paragraph("<b>Name:</b> ", styles["BodyText"])
    name_line = Paragraph("_" * 45, styles["BodyText"])
    id_label = Paragraph("<b>Student number:</b>", styles["BodyText"])
    id_box = NumericPanel(width_mm=60, height_mm=7)
    
    header_data = [
        [[name_label, Spacer(1, 2*MM), name_line], [id_label, Spacer(1, 2*MM), id_box]]
    ]
    
    header_table = Table(
        header_data,
        colWidths=[doc.width * 0.55, doc.width * 0.45],
    )
    
    # inner layout style
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'BOTTOM'),
        ('ALIGN', (1,0), (1,0), 'LEFT'),
    ]))
    
    header_box = Table(
        [[header_table]],
        colWidths=[doc.width]
    )
    
    header_box.setStyle(BOX_TABLE_STYLE)

    story.append(header_box)

    # exam title and qrcode on top of the page
    title_text = exam["title"] if exam["title"] is not None else ""
    title_para = Paragraph(title_text, styles["ExamTitle"])

    # prepare QR
    qr_drawing = create_qrcode(xor_encrypt_to_hex(payload, SECRET_KEY))

    # table row: title, qrcode
    qr_col_width = 25 * MM
    title_col_width = doc.width - qr_col_width

    title_table = Table(
        [[title_para, qr_drawing]], colWidths=[title_col_width, qr_col_width]
    )
    title_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                ("ALIGN", (1, 0), (1, 0), "LEFT"),  # qr align
                ("LEFTPADDING", (0, 0), (-1, -1), 0),
                ("RIGHTPADDING", (0, 0), (-1, -1), 0),
            ]
        )
    )

    story.append(title_table)

    # exam instructions
    instructions_html = (
        "Please read these instructions carefully:<br/><br/>"
        "&bull; <b>Fill the bubbles completely</b> for multiple-choice questions (MCQs).<br/>"
        f"&bull; Answers for <b>numeric questions</b> must be written in the large boxes "
        f"marked with the symbol <b>{NUMERIC_SYMBOL}</b> and should contain only numbers (and decimal points if needed). Then number should be aligned to the right.<br/>"
        f"&bull; When an MCQ is marked with the symbol <b>{MULTI_MCQ_SYMBOL}</b>, "
        "more than one answer may be correct. Choose all options that apply."
    )

    story.append(Paragraph(instructions_html, styles["BodyText"]))
    story.append(Spacer(1, 10 * MM))

    # exam questions
    order = exam["variant_ordering"][variant_idx]
    q_by_index = {q["index"]: q for q in exam["questions"]}

    for k, qidx in enumerate(order, start=1):
        q = q_by_index[qidx]

        # build inner content for one question
        inner = []

        multi_symbol = ""
        if q["type"] == "MCQ" and len(q.get("correct", [])) > 1:
            multi_symbol = f" {MULTI_MCQ_SYMBOL}"

        inner.append(
            Paragraph(
                f"<b>Q{k}.{multi_symbol}</b> {q.get('text', '')}", styles["BodyText"]
            )
        )
        inner.append(Spacer(1, 4 * MM))

        if q["type"] == "MCQ":
            inner.append(MCQPanel(tuple(q["choices"])))
        elif q["type"] == "NUM":
            inner.append(NumericPanel())

        # table with one cell spanning full available width
        box = Table(
            [[inner]],  # single cell containing a list of flowables
            colWidths=[doc.width],  # span full usable width (between margins)
        )

        box.setStyle(BOX_TABLE_STYLE)

        story.append(box)
        story.append(Spacer(1, 8 * MM))  # space between questions

    doc.build(story, onFirstPage=draw_page_fiducials, onLaterPages=draw_page_fiducials)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="Exam generator",
        description="Generate an exam from json (in our auto exam format)",
    )

    parser.add_argument("filename", type=str)

    args = parser.parse_args()

    data: Exam = json.load(open(args.filename))

    exam_id = data["exam_id"]
    out_dir = Path(f"exam_{exam_id}")
    out_dir.mkdir(parents=True, exist_ok=True)

    num_variants = data["variants"]

    for variant_idx in range(num_variants):
        out_path = out_dir / f"exam_{exam_id}_V{variant_idx + 1}.pdf"
        generate_exam_pdf(data, variant_idx=variant_idx, out_path=str(out_path))

    print(f"Generated {num_variants} variants in folder {out_dir}")
