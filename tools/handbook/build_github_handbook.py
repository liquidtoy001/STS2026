"""Build the STS2026 beginner handbook with searchable text and vector diagrams.

Run with Python and ReportLab. The source notebook and Git history are not modified.
"""
from pathlib import Path
from html import escape

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak,
    Flowable, Preformatted,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'guides' / 'STS2026_GitHub_Team_Handbook.pdf'
NAVY = colors.HexColor('#142B43')
TEAL = colors.HexColor('#007F82')
GRAY = colors.HexColor('#526273')
PALE = colors.HexColor('#EDF5F7')
AMBER = colors.HexColor('#FFF3DA')
WIDTH = A4[0] - 92

for name, file in [('Body', 'calibri.ttf'), ('Bold', 'calibrib.ttf'), ('Italic', 'calibrii.ttf')]:
    pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / file)))
pdfmetrics.registerFontFamily('Body', normal='Body', bold='Bold', italic='Italic', boldItalic='Bold')
pdfmetrics.registerFont(TTFont('Mono', 'C:/Windows/Fonts/consola.ttf'))
S = {
    'body': ParagraphStyle('body', fontName='Body', fontSize=10.8, leading=15.2,
                           textColor=NAVY, spaceAfter=8),
    'small': ParagraphStyle('small', fontName='Body', fontSize=9.2, leading=12.5,
                            textColor=GRAY, spaceAfter=6),
    'h1': ParagraphStyle('h1', fontName='Bold', fontSize=27, leading=30,
                         textColor=NAVY, spaceAfter=12),
    'h2': ParagraphStyle('h2', fontName='Bold', fontSize=13.5, leading=17,
                         textColor=TEAL, spaceBefore=9, spaceAfter=6),
    'cover': ParagraphStyle('cover', fontName='Bold', fontSize=44, leading=47,
                            textColor=NAVY, spaceAfter=16),
    'code': ParagraphStyle('code', fontName='Mono', fontSize=9, leading=12.5,
                           textColor=NAVY),
}
story = []
chapters = []


def p(text, style='body'):
    story.append(Paragraph(text, S[style]))


def h(text):
    p(text, 'h2')


def step(n, text):
    p(f'<b>{n:02d}.</b> {text}')


def code(text):
    block = Table([[Preformatted(text.strip(), S['code'])]], colWidths=[WIDTH])
    block.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), PALE),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
        ('TOPPADDING', (0, 0), (-1, -1), 9),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 9),
    ]))
    story.extend([block, Spacer(1, 10)])


def note(label, text, warning=False):
    t = Table([[Paragraph(f'<b>{label}</b><br/>{text}', S['body'])]], colWidths=[WIDTH])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), AMBER if warning else PALE),
        ('BOX', (0, 0), (-1, -1), .5, colors.HexColor('#D4E4E7')),
        ('LEFTPADDING', (0, 0), (-1, -1), 12),
        ('RIGHTPADDING', (0, 0), (-1, -1), 12),
        ('TOPPADDING', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ]))
    story.extend([t, Spacer(1, 10)])


def table(headers, rows, widths=None):
    data = [[Paragraph(f'<b>{escape(c)}</b>', S['small']) for c in headers]]
    data += [[Paragraph(c, S['small']) for c in row] for row in rows]
    t = Table(data, colWidths=widths or [WIDTH / len(headers)] * len(headers), hAlign='LEFT')
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), PALE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('LINEBELOW', (0, 0), (-1, -1), .4, colors.HexColor('#DAE3EA')),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
    ]))
    story.extend([t, Spacer(1, 9)])


class Diagram(Flowable):
    """Simple original vector diagrams, not screenshots of an application."""

    def __init__(self, labels, captions=None):
        super().__init__()
        self.labels = labels
        self.captions = captions or [''] * len(labels)
        self.width = WIDTH
        self.height = 102

    def draw(self):
        c = self.canv
        count = len(self.labels)
        gap = 17
        box = (WIDTH - gap * (count - 1)) / count
        for i, label in enumerate(self.labels):
            x = i * (box + gap)
            c.setFillColor(PALE)
            c.setStrokeColor(TEAL)
            c.roundRect(x, 39, box, 48, 6, fill=1, stroke=1)
            c.setFillColor(NAVY)
            c.setFont('Bold', 10)
            c.drawCentredString(x + box / 2, 58, label)
            c.setFont('Body', 8.5)
            c.setFillColor(GRAY)
            c.drawCentredString(x + box / 2, 23, self.captions[i])
            if i < count - 1:
                c.setStrokeColor(TEAL)
                c.line(x + box + 3, 63, x + box + gap - 3, 63)
                c.line(x + box + gap - 7, 67, x + box + gap - 3, 63)
                c.line(x + box + gap - 7, 59, x + box + gap - 3, 63)


class BranchDiagram(Flowable):
    def __init__(self):
        super().__init__()
        self.width, self.height = WIDTH, 140

    def draw(self):
        c = self.canv
        c.setLineWidth(2)
        c.setStrokeColor(NAVY)
        c.line(35, 104, WIDTH - 25, 104)
        c.setStrokeColor(TEAL)
        c.lines([(105, 104, 155, 48), (155, 48, 330, 48), (330, 48, 400, 104)])
        for x, y in [(55, 104), (105, 104), (265, 104), (400, 104), (170, 48), (250, 48), (330, 48)]:
            c.setFillColor(TEAL if y == 48 else NAVY)
            c.circle(x, y, 5, fill=1, stroke=0)
        c.setFillColor(NAVY)
        c.setFont('Bold', 10)
        c.drawString(35, 121, 'main: shared, reviewed work')
        c.setFillColor(TEAL)
        c.drawString(150, 24, 'docs/experiment-guide: one task')
        c.setFont('Body', 9)
        c.drawString(343, 71, 'Review + merge')


def page(title, subtitle):
    if story:
        story.append(PageBreak())
    number = len(chapters) + 1
    chapters.append(title)
    p(f'<a name="p{number}"/>{number:02d} / STS2026 FIELD GUIDE', 'small')
    p(title, 'h1')
    p(subtitle, 'small')
    story.append(Spacer(1, 7))


from handbook_web_content import populate

populate(
    page, p, h, step, code, note, table,
    lambda labels, captions: story.append(Diagram(labels, captions)),
    lambda: story.append(BranchDiagram()),
)


def decorate(canvas, doc):
    n = doc.page
    canvas.saveState()
    canvas.setTitle('STS2026 GitHub Team Handbook')
    canvas.setAuthor('STS2026 Team')
    canvas.setSubject('Beginner GitHub collaboration, notebooks and recovery scenarios')
    canvas.setStrokeColor(TEAL)
    canvas.setLineWidth(2)
    canvas.line(46, A4[1] - 32, A4[0] - 46, A4[1] - 32)
    canvas.setFont('Body', 8)
    canvas.setFillColor(GRAY)
    canvas.drawString(46, 27, 'STS2026  /  WEB + macOS  /  v2.1')
    canvas.drawRightString(A4[0] - 46, 27, f'{n:02d} / {len(chapters):02d}')
    if n <= len(chapters):
        canvas.bookmarkPage(f'bookmark{n}')
        canvas.addOutlineEntry(chapters[n - 1], f'bookmark{n}', level=0)
    canvas.restoreState()


if __name__ == '__main__':
    OUT.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(str(OUT), pagesize=A4, rightMargin=46, leftMargin=46,
                            topMargin=49, bottomMargin=49, title='STS2026 GitHub Team Handbook')
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    print(OUT)
