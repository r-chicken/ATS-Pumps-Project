"""Extract structured fields + report style from ATS vibration analysis PDFs.

Each PDF page is a single equipment report with this fixed text layout:

    {Site}
    VIBRATION ANALYSIS
    Date Tested: {date}
    Recommendations
    {bullet line 1}
    {bullet line 2}
    ...
    Comments
    {narrative paragraph, wrapped across several lines}
    {Equipment ID} : Priority {N}

The recommendations/comments/priority fields are native PDF text (no OCR
needed). The three charts (spectrum, waterfall/colored-spectrum, trend) are
embedded as a single raster screenshot - "Colored Spectrum" (the label that
identifies the excluded report style) is baked into that screenshot as
pixels, not selectable PDF text, so detecting it requires OCR'ing just that
one embedded image (not the whole page, and not the main text fields).

Style detection used to be a colorfulness heuristic on the same image, but
that was calibrated on only 2 examples and started misclassifying real
waterfall-style reports once run against hundreds of real files. OCR'ing
for the literal "Colored Spectrum" label is slower per page but detects
the actual thing that makes a report that style, so it doesn't drift as
the report mix changes. colorfulness() is kept below for reference/
diagnostics but no longer drives classification.
"""
from __future__ import annotations

import io
import re
import dataclasses
from pathlib import Path

import fitz  # PyMuPDF
import numpy as np
from PIL import Image

from .graph_signals import detect_measurement_point, spectrum_priority_hint

PRIORITY_LINE_RE = re.compile(r"^(?P<equipment_id>.+?)\s*:\s*Priority\s*(?P<priority>\S+)\s*$")
DATE_RE = re.compile(r"Date Tested:\s*(?P<date>.+)")
COLORED_SPECTRUM_LABEL_RE = re.compile(r"colou?red\s+spectrum", re.IGNORECASE)

# Kept only as an optional diagnostic signal now - see module docstring.
DEFAULT_STYLE_THRESHOLD = 25.0


@dataclasses.dataclass
class ReportRecord:
    source_file: str
    page_number: int
    site: str | None
    date_tested: str | None
    equipment_id: str | None
    priority_raw: str | None
    priority_num: float | None
    recommendations: list[str]
    comments: str | None
    chart_colorfulness: float | None
    style: str  # "waterfall" | "colored_spectrum" | "unknown"
    spectrum_unit: str | None  # "in/s" | "g" | "gE" | "unknown" | None (no chart image)
    measurement_point: str | None  # sensor location/direction label off the chart title, e.g. "Mtr Shaft H IPS" - see graph_signals.detect_measurement_point
    spectrum_peak_amplitude: float | None  # tallest real Spectrum peak, linearly interpolated between the two nearest y-axis gridlines - see graph_signals.py
    spectrum_peak_amplitude_floored: float | None  # same reading floored down to the nearest printed y-axis label - kept for cross-checking against the chart's own tick text, not used for priority classification anymore
    spectrum_priority_hint: int | None  # supporting evidence only - see graph_signals.py
    spectrum_peak_error: str | None  # why peak-reading failed, if it did - see graph_signals.read_spectrum_peak
    chart_ocr_text: str | None  # cached raw OCR of the chart image - see dataset.recompute_dataset
    parse_ok: bool
    parse_notes: str


def extract_pages_text(pdf_path: str | Path) -> list[str]:
    """Return the raw text layer for each page of the PDF."""
    doc = fitz.open(pdf_path)
    try:
        return [page.get_text() for page in doc]
    finally:
        doc.close()


def parse_report_fields(raw_text: str) -> dict:
    """Regex-parse one page's text into the fields we care about.

    Defensive by design: missing sections (e.g. a healthy-equipment report
    with no Recommendations/Comments) don't raise, they just come back
    empty, with parse_notes explaining what was missing.
    """
    lines = [ln.strip() for ln in raw_text.splitlines()]
    non_empty = [(i, ln) for i, ln in enumerate(lines) if ln]

    notes = []

    site = non_empty[0][1] if non_empty else None
    if site == "VIBRATION ANALYSIS" and len(non_empty) > 1:
        # defensive: in case the site line is ever blank/missing
        site = None
        notes.append("site line missing before VIBRATION ANALYSIS")

    date_match = DATE_RE.search(raw_text)
    date_tested = date_match.group("date").strip() if date_match else None
    if not date_tested:
        notes.append("no Date Tested field found")

    def find_line_idx(target: str) -> int | None:
        for i, ln in enumerate(lines):
            if ln.strip().lower() == target.lower():
                return i
        return None

    rec_idx = find_line_idx("Recommendations")
    com_idx = find_line_idx("Comments")

    equipment_id = None
    priority_raw = None
    priority_num = None
    priority_match = None
    for ln in reversed(lines):
        m = PRIORITY_LINE_RE.match(ln)
        if m:
            priority_match = m
            break
    if priority_match:
        equipment_id = priority_match.group("equipment_id").strip()
        priority_raw = priority_match.group("priority").strip()
        try:
            priority_num = float(priority_raw)
        except ValueError:
            notes.append(f"non-numeric priority value: {priority_raw!r}")
    else:
        notes.append("no '<Equipment> : Priority <N>' line found")

    recommendations: list[str] = []
    if rec_idx is not None and com_idx is not None and com_idx > rec_idx:
        recommendations = [ln for ln in lines[rec_idx + 1 : com_idx] if ln]
    elif rec_idx is not None:
        notes.append("Recommendations section found but no Comments section after it")

    comments = None
    if com_idx is not None:
        end_idx = len(lines)
        # stop before the trailing "<Equipment> : Priority <N>" line, if present
        for i in range(len(lines) - 1, com_idx, -1):
            if PRIORITY_LINE_RE.match(lines[i]):
                end_idx = i
                break
        comment_lines = [ln for ln in lines[com_idx + 1 : end_idx] if ln]
        comments = re.sub(r"\s+", " ", " ".join(comment_lines)).strip() or None
    else:
        notes.append("no Comments section found")

    parse_ok = priority_match is not None and (com_idx is not None or rec_idx is not None)

    return {
        "site": site,
        "date_tested": date_tested,
        "equipment_id": equipment_id,
        "priority_raw": priority_raw,
        "priority_num": priority_num,
        "recommendations": recommendations,
        "comments": comments,
        "parse_ok": parse_ok,
        "parse_notes": "; ".join(notes),
    }


def colorfulness(image: Image.Image) -> float:
    """Hasler-Susstrunk colorfulness metric.

    Colored-spectrum heatmaps (blue/green/yellow/red) score much higher
    than the mostly white/grey line-chart screenshots used in the
    waterfall-style reports.
    """
    arr = np.asarray(image.convert("RGB")).astype(np.float64)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    rg = r - g
    yb = 0.5 * (r + g) - b
    std_root = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mean_root = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(std_root + 0.3 * mean_root)


def largest_embedded_image(doc: fitz.Document, page: fitz.Page) -> Image.Image | None:
    """Return the largest (by on-page area) embedded raster image on a page.

    This is the multi-panel chart screenshot in these reports (the
    equipment photo is smaller and positioned separately).
    """
    best_xref = None
    best_area = -1.0
    for img in page.get_images(full=True):
        xref = img[0]
        for rect in page.get_image_rects(xref):
            area = rect.width * rect.height
            if area > best_area:
                best_area = area
                best_xref = xref
    if best_xref is None:
        return None
    base = doc.extract_image(best_xref)
    return Image.open(io.BytesIO(base["image"]))


def classify_style_by_color(score: float | None, threshold: float = DEFAULT_STYLE_THRESHOLD) -> str:
    """Legacy colorfulness-based classifier. Kept for reference/diagnostics
    only - see module docstring for why classify_style_by_text is used
    instead."""
    if score is None:
        return "unknown"
    return "colored_spectrum" if score > threshold else "waterfall"


def ocr_image_text(image: Image.Image) -> str:
    """OCR an embedded chart screenshot. Returns "" (not an exception) if
    OCR isn't available or fails on this image - callers treat that as
    'could not determine style from text', not as a hard error, since the
    caller still has the colorfulness score as a fallback signal.
    """
    import pytesseract

    return pytesseract.image_to_string(image)


def classify_style_by_text(ocr_text: str) -> str:
    """"Colored Spectrum" is a label baked into the chart screenshot as
    pixels (see module docstring) - its presence is what actually makes a
    report that excluded style, so this looks for the words themselves
    rather than inferring it from image color statistics.
    """
    if COLORED_SPECTRUM_LABEL_RE.search(ocr_text):
        return "colored_spectrum"
    return "waterfall"


def process_pdf(
    pdf_path: str | Path,
    style_threshold: float = DEFAULT_STYLE_THRESHOLD,
    max_pages: int | None = None,
) -> list[ReportRecord]:
    """Parse each page of pdf_path into a ReportRecord.

    Pass max_pages=1 to only look at the first page of each PDF and skip
    any additional pages entirely (e.g. when later pages never contain
    equipment/priority content worth extracting).
    """
    pdf_path = Path(pdf_path)
    doc = fitz.open(pdf_path)
    records = []
    try:
        for page_number, page in enumerate(doc, start=1):
            if max_pages is not None and page_number > max_pages:
                break
            text = page.get_text()
            fields = parse_report_fields(text)

            chart_img = largest_embedded_image(doc, page)
            score = colorfulness(chart_img) if chart_img is not None else None
            ocr_text = None
            if chart_img is None:
                style = "unknown"
                fields["parse_notes"] = (fields["parse_notes"] + "; no chart image found on page").strip("; ")
                spectrum_unit = None
                measurement_point = None
                spectrum_peak_amplitude = None
                spectrum_peak_amplitude_floored = None
                spectrum_priority_hint_val = None
                spectrum_peak_error = None
            else:
                try:
                    ocr_text = ocr_image_text(chart_img)
                    style = classify_style_by_text(ocr_text)
                    measurement_point = detect_measurement_point(ocr_text)
                    hint = spectrum_priority_hint(chart_img, ocr_text)
                    spectrum_unit = hint["spectrum_unit"]
                    spectrum_peak_amplitude = hint["spectrum_peak_amplitude"]
                    spectrum_peak_amplitude_floored = hint["spectrum_peak_amplitude_floored"]
                    spectrum_priority_hint_val = hint["spectrum_priority_hint"]
                    spectrum_peak_error = hint["spectrum_peak_error"]
                except Exception as exc:  # noqa: BLE001 - e.g. tesseract binary missing
                    style = "unknown"
                    spectrum_unit = None
                    measurement_point = None
                    spectrum_peak_amplitude = None
                    spectrum_peak_amplitude_floored = None
                    spectrum_priority_hint_val = None
                    spectrum_peak_error = str(exc)
                    fields["parse_notes"] = (fields["parse_notes"] + f"; OCR failed: {exc}").strip("; ")

            records.append(
                ReportRecord(
                    source_file=pdf_path.name,
                    page_number=page_number,
                    site=fields["site"],
                    date_tested=fields["date_tested"],
                    equipment_id=fields["equipment_id"],
                    priority_raw=fields["priority_raw"],
                    priority_num=fields["priority_num"],
                    recommendations=fields["recommendations"],
                    comments=fields["comments"],
                    chart_colorfulness=score,
                    style=style,
                    spectrum_unit=spectrum_unit,
                    measurement_point=measurement_point,
                    spectrum_peak_amplitude=spectrum_peak_amplitude,
                    spectrum_peak_amplitude_floored=spectrum_peak_amplitude_floored,
                    spectrum_priority_hint=spectrum_priority_hint_val,
                    spectrum_peak_error=spectrum_peak_error,
                    chart_ocr_text=ocr_text,
                    parse_ok=fields["parse_ok"],
                    parse_notes=fields["parse_notes"],
                )
            )
    finally:
        doc.close()
    return records
