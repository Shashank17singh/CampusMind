"""Regenerate the sample files in this folder: python tests/fixtures/make_fixtures.py"""

import io
from pathlib import Path

import pymupdf
from docx import Document
from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.util import Inches

HERE = Path(__file__).parent
HEADER = "Government College of Engineering"
BODIES = {
    1: "Normal forms reduce redundancy in relational databases. A relation is in third normal "
    "form when every non-key attribute depends only on the key.",
    2: "Decomposition must keep dependencies and remain lossless so that the original relation "
    "can be rebuilt from its parts.",
    3: "Transactions group operations into one unit of work. Isolation levels decide which "
    "anomalies a concurrent transaction can observe.",
    4: "Indexes speed up lookups at the cost of slower writes. A B-tree keeps keys sorted, so "
    "it serves range queries as well as point queries.",
}


def _draw_table(page: pymupdf.Page, x: float, y: float, rows: list[list[str]]) -> None:
    """Draw a ruled table so pdfplumber can detect it from its lines."""
    column_width, row_height = 120, 24
    for row_index, row in enumerate(rows):
        for column_index, cell in enumerate(row):
            left = x + column_index * column_width
            top = y + row_index * row_height
            page.draw_rect(pymupdf.Rect(left, top, left + column_width, top + row_height))
            page.insert_text((left + 4, top + 16), cell, fontsize=10)


def make_text_pdf() -> None:
    """A four-page PDF with a repeated header, page numbers, headings and a table."""
    document = pymupdf.open()
    for number in range(1, 5):
        page = document.new_page()
        page.insert_text((72, 40), HEADER, fontsize=9)
        page.insert_text((72, 90), f"Unit {number}: Database Design", fontsize=18)
        page.insert_textbox(pymupdf.Rect(72, 110, 520, 260), BODIES[number], fontsize=11)
        if number == 2:
            page.insert_text((72, 280), "The regula-\ntion applies to all labs.", fontsize=11)
            _draw_table(
                page,
                72,
                330,
                [["Course", "Credits"], ["DBMS", "4"], ["Operating Systems", "3"]],
            )
        page.insert_text((72, 780), f"Page {number} of 4", fontsize=9)
    document.save(HERE / "text.pdf")
    document.close()


def _text_image(lines: list[str], size: tuple[int, int] = (1240, 800)) -> Image.Image:
    """Render lines of text to an image, as a scanner would produce."""
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=40)
    for index, line in enumerate(lines):
        draw.text((60, 60 + index * 70), line, fill="black", font=font)
    return image


def make_scanned_pdf() -> None:
    """A PDF whose only page is an image, so it has no text layer."""
    image = _text_image(["NOTICE", "The library will remain", "closed on Friday."])
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    document = pymupdf.open()
    page = document.new_page()
    page.insert_image(pymupdf.Rect(0, 0, 595, 400), stream=buffer.getvalue())
    document.save(HERE / "scanned.pdf")
    document.close()


def make_notice_image() -> None:
    """A photographed notice."""
    _text_image(["NOTICE", "Fee payment closes on 15 November."]).save(HERE / "notice.png")


def make_docx() -> None:
    """A Word file with headings and a table between paragraphs."""
    document = Document()
    document.add_heading("Examination Rules", level=1)
    document.add_paragraph("Students must carry their identity cards.")
    document.add_heading("Grading", level=2)
    table = document.add_table(rows=3, cols=2)
    for row, values in zip(table.rows, [("Grade", "Marks"), ("A", "90"), ("B", "75")], strict=True):
        row.cells[0].text, row.cells[1].text = values
    document.add_paragraph("Grades are final after moderation.")
    document.save(HERE / "tables.docx")


def make_pptx() -> None:
    """A two-slide deck with titles, a table and speaker notes."""
    presentation = Presentation()
    layout = presentation.slide_layouts[1]
    first = presentation.slides.add_slide(layout)
    first.shapes.title.text = "Process Scheduling"
    first.placeholders[1].text = "Round robin gives each process a time slice."
    first.notes_slide.notes_text_frame.text = "Mention starvation."
    second = presentation.slides.add_slide(presentation.slide_layouts[5])
    second.shapes.title.text = "Comparison"
    table = second.shapes.add_table(2, 2, Inches(1), Inches(2), Inches(4), Inches(1)).table
    table.cell(0, 0).text, table.cell(0, 1).text = "Policy", "Preemptive"
    table.cell(1, 0).text, table.cell(1, 1).text = "FCFS", "No"
    presentation.save(HERE / "slides.pptx")


def make_corrupted() -> None:
    """Files whose headers look right but whose contents are garbage."""
    garbage = bytes(range(256)) * 40
    (HERE / "corrupted.pdf").write_bytes(b"%PDF-1.7\n" + garbage)
    (HERE / "corrupted.docx").write_bytes(b"PK\x03\x04" + garbage)


if __name__ == "__main__":
    make_text_pdf()
    make_scanned_pdf()
    make_notice_image()
    make_docx()
    make_pptx()
    make_corrupted()
