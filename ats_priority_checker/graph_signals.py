"""Supporting priority signal read directly from the Spectrum plot's pixels
(top-left panel of the embedded chart screenshot), on top of the same OCR
text already used for style detection in extract.py.

Per the domain guidance behind this module: this is not a standalone
verdict. Report writers sometimes under-state urgency for equipment that's
been high-priority before, so this exists to catch that - but it's
supporting evidence to combine with the text-based prediction, not a
replacement for it.

Status / history:
- v1 (Fund Amp text field): read the printed "Fund Amp: X" line next to the
  Spectrum plot. Retired - Fund Amp is the amplitude at whichever harmonic
  order the analyst happened to list first, not the tallest peak in the
  plot, and the two can disagree a lot (e.g. a report with Fund Amp: 0.064
  whose actual tallest Spectrum peak is ~1.85 on a 0-2 scale). Do not bring
  Fund Amp extraction back.
- v2 (this version): reads the Spectrum plot's PIXELS directly - finds the
  tallest genuine data peak, reads it off the y-axis's own printed tick
  labels (OCR'd, calibrated with RANSAC so a misread digit or two doesn't
  wreck the whole scale), and floors it to the nearest labeled gridline at
  or below the peak. Validated by hand against 3 real reports spanning two
  different y-axis label spacings (0.5 and 0.05) and three different kinds
  of on-chart marker (a fixed-height magenta "harmonic flag" bar with a
  callout line down to the real peak; a cyan zoom-selection box with no
  relationship to data height at all; small red circle/number annotations
  sitting directly on top of genuine peaks). See _find_peak_pixel for how
  markers are told apart from real data. This is a best-effort pixel
  heuristic, not pixel-perfect - the plateau-width and run-length
  thresholds below are tuned against those 3 reports, not the full ~115
  labeled set, so if you have Spectrum peak readings that look visibly
  wrong once you run this against your real data, that's the first place
  to retune (see the constants just below the imports).
- Waterfall/Trend charts: intentionally not read. The old Trend pixel
  heuristic here topped out around a 75% ceiling across five different
  techniques even for a narrower "is it escalating at all" question, well
  short of what precise point-reading would need - don't re-add
  Waterfall/Trend pixel analysis without re-reading that discussion.
- Cross-report escalation (comparing a reading against the same
  equipment's own prior dated test): tried (dataset.py's
  add_escalation_signals) and removed. This report set has multiple
  distinct measurement points per equipment_id (see
  detect_measurement_point below), often in different units, and not
  every equipment_id has a same-point history to compare against at all -
  after several rounds of trying to patch the grouping logic (by
  equipment_id alone, then equipment_id + measurement_point, relaxing the
  same-unit requirement, etc. - see the conversation this is from), the
  conclusion was that there isn't a reliably consistent per-machine
  timeline to compare against in this data, not just a bug to fix. There
  is currently NO cross-report/historical signal in this project - each
  report is judged only on its own text and its own chart reading. Don't
  re-add a cross-report escalation signal without a concrete plan for
  what makes two readings comparable (same equipment AND same
  measurement point AND same unit, at minimum) and how thin that leaves
  the usable data.
- Future idea (not implemented): compare the FREQUENCIES of each report's
  peaks across dated reports for the same equipment, to catch a resonance
  shifting frequency even when its amplitude doesn't change much. Would
  need each peak's x-axis (frequency) calibration too, not just y-axis -
  same OCR+RANSAC approach as below would likely extend to it. Would
  inherit the same "what's actually comparable across reports" problem
  escalation ran into, so worth reading that entry first.
"""
from __future__ import annotations

import re

import numpy as np
from PIL import Image

TREND_OVERALL_RE = re.compile(r"trend\s+overall[:\s]*[\d.]+\s*(?P<unit>\S+)", re.IGNORECASE)

# --- Spectrum panel geometry -------------------------------------------
# The embedded screenshot is always the same 3-panel layout: Spectrum
# top-left, Waterfall top-right, Trend across the bottom. These fractions
# crop out just the Spectrum panel (with a little slack on the bottom/right
# so its own frame border is never clipped) and were checked against 3 real
# screenshots at three different pixel resolutions (1920x1080, 1902x1045,
# 1907x995) - all three cropped correctly with these fractions.
SPECTRUM_PANEL_WFRAC = 0.52
SPECTRUM_PANEL_HFRAC = 0.72

# Left margin (as a fraction of the panel width) that contains the y-axis
# tick number labels, for OCR.
Y_LABEL_STRIP_WFRAC = 0.10
OCR_UPSCALE = 4  # upscaling the label strip before OCR fixes most misreads

# UI chrome (toolbar/title bar) sits in the first ~10% of the panel height
# in every sample seen - the frame-border search below only has to look
# past that, which is what keeps it from mistaking a toolbar separator line
# for the plot's own top border.
CHROME_SKIP_FRAC = 0.12

# Peak-pixel classification (see _find_peak_pixel for what each guards
# against):
MAX_PLATEAU_WIDTH_PX = 8    # widest a genuine single-frequency peak should look
PLATEAU_ROW_TOL_PX = 2      # row jitter allowed while still calling two columns "level"
GAP_MERGE_PX = 6            # small antialiasing/letter gaps to bridge when measuring a run's height
MAX_RUN_HEIGHT_FRAC = 0.25  # fraction of plot height a genuine peak's own column may run solid
MIN_RUN_HEIGHT_PX = 2       # below this, treat it as compression/antialiasing noise, not a mark
MAX_PEEL_PASSES = 6         # how many stacked marker layers a single column can have peeled off it
FULL_HEIGHT_TOL_PX = 2      # how close to literally touching both frame edges still counts as "spans it"


def detect_spectrum_unit(ocr_text: str) -> str:
    """Best-effort unit detection for the Spectrum plot, read from the
    chart's title text (e.g. "...Trend Overall: 1.186 in/s") rather than
    the y-axis label itself, which is usually rendered rotated 90 degrees
    and OCRs unreliably. Returns "in/s", "g", "gE", or "unknown".

    This only reads the unit STRING off the title text - it has nothing to
    do with the Trend plot's pixels/history (see module docstring on why
    those aren't read anymore).

    Anchored specifically to the token right after "Trend Overall: N" -
    confirmed against real reports this is far more reliable than
    searching the whole OCR blob. "gE" is exactly 2 characters, and OCR
    sometimes misreads the italic "E" as something else entirely (seen for
    real as "g&") - since the whole unit token is only ever 1-2 characters
    here, matching its *exact* length distinguishes a genuine bare "g"
    (1 char) from a garbled "gE" (2 chars, second char unreliable) without
    needing to know what the garbled character actually is.
    """
    match = TREND_OVERALL_RE.search(ocr_text)
    if match:
        token = match.group("unit").strip(",;:.")
        if re.search(r"in\s*/\s*s", token, re.IGNORECASE):
            return "in/s"
        if re.fullmatch(r"g.", token, re.IGNORECASE):
            return "gE"
        if re.fullmatch(r"g", token, re.IGNORECASE):
            return "g"
        return "unknown"

    # No "Trend Overall" anchor found at all - fall back to a broader,
    # less precise search over the whole OCR text.
    if re.search(r"in\s*/\s*s", ocr_text, re.IGNORECASE):
        return "in/s"
    if re.search(r"\bgE\b", ocr_text):
        return "gE"
    if re.search(r"\bg\b", ocr_text):
        return "g"
    return "unknown"


# Matches the sensor location/direction label baked into every panel
# title on the chart screenshot, e.g. "EF-3521 \ Mtr Shaft H IPS, Channel
# X" -> captures "Mtr Shaft H IPS" (equipment_id \ LOCATION, rest...).
# This is the actual measurement point (which shaft, which end, which
# direction) - equipment_id alone doesn't distinguish it, and a single
# piece of equipment can have several of these, each with its own
# separate trend history and sometimes its own unit. Useful context on its
# own (see model.priority_recommendation_table) - this is also exactly why
# a cross-report escalation signal was tried and removed (see the module
# docstring's history section) rather than a reason to bring one back.
MEASUREMENT_POINT_RE = re.compile(r"\\\s*(?P<location>[^,\n]{2,60}?)\s*,")

# The physical location itself is always "{Mtr|Fan|Pump} {Shaft|End|...}
# {H|V|A}" - Motor, Fan, or Pump, which end, then a single-letter direction
# (Horizontal, Vertical, Axial). Everything AFTER that direction letter is a
# unit designator tacked on by the ATS software ("Mtr Shaft H IPS", "Fan
# Shaft H gE3", "Pump Shaft H gE3") - not part of the physical measurement
# point, and OCR-unstable in a way the location words aren't (confirmed on
# real reports: the same real sensor read "gE3" on one visit's scan and
# "g&3" on another's). Everything BEFORE the "F"/"M"/"P" that starts
# "Fan"/"Mtr"/"Pump" is leftover cruft from the capture above (there
# normally isn't any, but nothing guarantees that on every OCR pass). This
# keeps only the part in between - "Mtr Shaft H", "Fan End H", "Pump Shaft
# H" - which is what's actually meaningful to a reviewer, so trailing OCR
# noise there can't quietly make "the same sensor" look like two different
# ones on the page.
#
# Deliberately case-sensitive (no re.IGNORECASE) on the anchors - "Fan"/
# "Mtr"/"Pump" and the direction letter are always capitalized on real
# reports, and matching lowercase too caused a real false start in testing:
# "junk before Fan End V" - re.search found the "f" inside "before" first
# and ran with it. Case-sensitive anchors can't do that; the inner
# descriptive word(s) (Shaft/End/Bearing/...) stay case-flexible since only
# the two ends need to be trustworthy.
#
# The direction letter's own end isn't always a clean word boundary either
# - confirmed on a real report, the unit suffix can end up glued directly
# onto it with no space at all ("Mtr Shaft AIPS", not "... A IPS"), so a
# plain \b there would miss it (no boundary between "A" and "I", both word
# characters). (?=[^a-z]|$) instead: stop right after the direction letter
# as long as what follows ISN'T a lowercase letter - a glued-on suffix
# like "IPS" starts uppercase, so this still cuts it off ("Mtr Shaft A"),
# while a genuine longer word that happened to start with H/V/A (lowercase
# continuing right after) would fail this and correctly NOT be treated as
# the end.
_MEASUREMENT_POINT_CORE_RE = re.compile(r"[FMP][a-z]*(?:\s+[A-Za-z]+)*?\s+[HVA](?=[^a-z]|$)")

# Known OCR misreads worth correcting outright rather than leaving to the
# repetition-voting below to (usually) outvote - confirmed on real reports:
# "Mtr" -> "Mir" often enough that a single image's OCR pass can plausibly
# get MORE than half its repeats wrong, not just the occasional one-off a
# majority vote shrugs off. Applied before the core-trim above so a
# corrected "Mtr" still matches the [FMP] anchor. "Mitr" (extra "i") is the
# same underlying misread in the other direction - same fix, same reason.
# No equivalent misread has been confirmed for "Pump" yet, so there's
# nothing to add here for it - the anchor extension below is enough on its
# own to stop "Pump ..." labels from falling through to the raw-match
# fallback the way "Mtr"/"Fan" ones no longer do.
_KNOWN_OCR_FIXES = [
    (re.compile(r"\bMir\b"), "Mtr"),
    (re.compile(r"\bMitr\b"), "Mtr"),
]


def detect_measurement_point(ocr_text: str) -> str | None:
    """Best-effort read of the sensor location/direction label from the
    chart's own panel titles (e.g. "Mtr Shaft H", "Fan End H", "Pump Shaft
    H") - returns None if nothing matched.

    The same label is printed redundantly under all three panels
    (Spectrum, Waterfall, Trend) on every real report seen, so rather than
    trusting whichever match comes first, this takes the most common
    string across all of them (after trimming each one down to its core
    "{Fan|Mtr|Pump} ... {H|V|A}" span first - see
    _MEASUREMENT_POINT_CORE_RE - so noise before/after that span doesn't
    split votes for what's really the same location across repeats within
    one image, not just across reports) - the same "let repetition outvote
    a one-off OCR slip" idea _read_y_axis_ticks uses for tick labels, just
    simpler since there's no numeric fitting involved here, only picking a
    mode. A raw match that doesn't contain a recognizable "{Fan|Mtr|Pump}
    ... {H|V|A}" span at all falls back to itself, stripped, rather than
    being dropped outright -
    better to surface an unfamiliar label than silently lose the row.
    Known OCR misreads (see _KNOWN_OCR_FIXES) are corrected explicitly
    rather than left to this voting to sort out - a real image can plausibly
    get more than half its own repeats wrong on a misread that's common
    enough to be worth naming outright, which plain majority voting alone
    wouldn't survive. Confirmed against 11 real reports spanning 6 distinct
    locations.
    """
    from collections import Counter

    matches = [m.strip() for m in MEASUREMENT_POINT_RE.findall(ocr_text) if m.strip()]
    cleaned = []
    for m in matches:
        for pattern, fix in _KNOWN_OCR_FIXES:
            m = pattern.sub(fix, m)
        core = _MEASUREMENT_POINT_CORE_RE.search(m)
        cleaned.append(core.group(0).strip() if core else m)
    cleaned = [c for c in cleaned if c]
    if not cleaned:
        return None
    return Counter(cleaned).most_common(1)[0][0]


def _crop_spectrum_panel(chart_image: Image.Image) -> Image.Image:
    w, h = chart_image.size
    return chart_image.crop((0, 0, int(w * SPECTRUM_PANEL_WFRAC), int(h * SPECTRUM_PANEL_HFRAC)))


def _read_y_axis_ticks(panel: Image.Image) -> tuple[list[tuple[float, float]], float | None]:
    """OCR the y-axis tick number labels in the panel's left margin.

    Returns (points, label_right_edge) where points is a list of
    (pixel_row, value) pairs (not yet outlier-filtered - see
    _ransac_calibration for that) and label_right_edge is the x-position
    the tick labels are right-aligned against (used later to locate the
    plot frame's left border, which sits just past it).

    Two things this works around, found by testing against real reports:
    - Tesseract regularly drops the decimal point on these small chart
      fonts ("0.5" -> "05"). Rather than trying to guess where a missing
      point goes, this only keeps tokens that already look unambiguous
      ("\\d+\\.\\d+" or a bare "\\d+") and lets RANSAC calibration (next
      step) work from whatever legible subset that leaves.
    - The axis title ("gE - Peak"), a value baked into the title text
      above the plot, and other stray OCR fragments can each produce a
      spurious numeric-looking token. Real tick labels are right-aligned
      against the axis line, so this clusters candidates by their right
      edge and keeps only the largest cluster - reliably keeps the real
      ticks and drops the rest, even when a couple of them have very low
      OCR confidence (confidence alone was tried and wasn't reliable
      enough to use as the primary filter here). It's used as a tie-
      breaker between same-size clusters, though - confirmed on a real
      report where a rotated fault-frequency callout (a "Fund Amp: ..."
      readout and a bearing-defect label, both printed in the plot's top-
      left corner, right where the y-axis's own top tick labels are)
      happened to right-align within the clustering tolerance of each
      other AND tie the real tick cluster's size, so whichever cluster's
      tokens Tesseract happened to emit first (scan order, not anything
      meaningful) silently won - here, the annotation cluster, leaving
      nothing usable to calibrate against. Every real tick label seen
      across every report tested OCRs at 88+ confidence; every spurious
      token from rotated/decorative annotation text seen OCRs at 0 - not
      close enough to call reliable on its own (hence not the primary
      filter), but a clean way to prefer the real cluster on an exact
      count tie rather than leaving it to scan order.
    - A single tick label can be missing from one psm pass entirely, or
      have its digits garbled, while a DIFFERENT pass reads it cleanly -
      confirmed on two real reports where Tesseract's default assumed-
      layout mode (--psm 6, "one uniform block of text") merged/dropped a
      label sitting close above or below another one (a "1" tick directly
      under a "1.05" one; a "0.5" tick swallowed by a nearby callout's
      dotted leader line), while --psm 6 misread a THIRD (a lone "0.2")
      that --psm 6's own neighbor-relative digit model got dragged off by
      an adjacent "92"-shaped fragment above it. --psm 11 ("sparse text,
      no assumed layout") reads each of those correctly where 6 didn't -
      but flips the failure the other way on a label 6 already had right
      (misread "1.5" as "15", decimal dropped), so it's not a strict
      upgrade to switch to on its own. Both passes run and their
      candidates POOL into one list rather than one replacing the other -
      safe to do because a wrong token from either pass (a stray "15", a
      misread "99") almost never lands on the real calibration line, so
      _ransac_calibration's existing outlier rejection (built to survive
      exactly this kind of single bad token) discards it same as always;
      meanwhile a label only one pass caught cleanly now has a chance to
      make it into the kept set at all, instead of leaving a gap
      _floor_to_axis_label has nothing to floor down to.
    """
    import pytesseract

    cw, ch = panel.size
    strip = panel.crop((0, 0, int(cw * Y_LABEL_STRIP_WFRAC), ch))
    strip_up = strip.resize((strip.size[0] * OCR_UPSCALE, strip.size[1] * OCR_UPSCALE), Image.LANCZOS)

    candidates = []
    for psm in (6, 11):
        data = pytesseract.image_to_data(
            strip_up,
            config=f"--psm {psm} -c tessedit_char_whitelist=0123456789.",
            output_type=pytesseract.Output.DICT,
        )
        for i in range(len(data["text"])):
            text = data["text"][i].strip()
            if not text or not re.fullmatch(r"\d+\.\d+|\d+", text):
                continue
            row = (data["top"][i] + data["height"][i] / 2) / OCR_UPSCALE
            right_edge = (data["left"][i] + data["width"][i]) / OCR_UPSCALE
            conf = float(data["conf"][i])
            candidates.append((row, float(text), right_edge, conf))

    if not candidates:
        return [], None

    edges = np.array([c[2] for c in candidates])
    confs = np.array([c[3] for c in candidates])
    best_center, best_score = edges[0], (0, -1.0)
    for e in edges:
        mask = np.abs(edges - e) <= 4
        score = (int(mask.sum()), float(confs[mask].sum()))
        if score > best_score:
            best_score, best_center = score, e

    points = sorted(
        {(row, val) for row, val, edge, _conf in candidates if abs(edge - best_center) <= 4},
        key=lambda p: p[1],
    )
    return points, float(best_center)


def _ransac_calibration(points: list[tuple[float, float]], row_tol: float = 6.0):
    """Fit pixel_row = a*value + b from OCR'd tick points, RANSAC-style.

    Needed because a single bad OCR token (most often two adjacent labels
    merged into one, e.g. "0.4" + "0.45" -> "20.45") is common enough that
    a plain least-squares fit (even with iterative residual trimming) can
    get pulled off badly enough to then reject the GOOD points instead.
    Trying every pair of points as a candidate line and keeping whichever
    line the most other points agree with sidesteps that.

    Returns (a, b, kept_points) or None if fewer than 2 usable points.
    """
    import itertools

    points = sorted(set(points), key=lambda p: p[1])
    if len(points) < 2:
        return None
    rows = np.array([p[0] for p in points])
    vals = np.array([p[1] for p in points])

    best_inliers = None
    for i, j in itertools.combinations(range(len(points)), 2):
        if vals[i] == vals[j]:
            continue
        a = (rows[j] - rows[i]) / (vals[j] - vals[i])
        if a >= 0:
            continue  # pixel row must decrease as the axis value increases
        b = rows[i] - a * vals[i]
        inliers = np.abs(rows - (a * vals + b)) <= row_tol
        if best_inliers is None or inliers.sum() > best_inliers.sum():
            best_inliers = inliers

    if best_inliers is None or best_inliers.sum() < 2:
        return None
    a, b = np.polyfit(vals[best_inliers], rows[best_inliers], 1)
    kept = list(zip(rows[best_inliers].tolist(), vals[best_inliers].tolist()))
    return a, b, kept


def _find_plot_frame(arr: np.ndarray, label_right_edge: float, a: float, b: float, max_tick_val: float):
    """Locate the plot's own black frame border: (left, right, top, bottom)
    in panel-pixel coordinates.

    Toolbar/title-bar chrome above the plot can be just as dark as the
    frame border, so top/bottom aren't found by "the darkest row" - they're
    found by anchoring to the y-axis calibration instead: bottom is the
    strong horizontal line closest to the calibration's own value=0 row,
    and top is the strong line closest to where the highest OCR'd tick
    should be. That estimate is enough even when OCR missed the very top
    label entirely (seen for real: a "2" tick rendered flush against a
    small triangle max-value marker, which tesseract failed to read even
    after upscaling) - the frame border pixel search recovers the exact
    row anyway.
    """
    H, W = arr.shape[0], arr.shape[1]
    r, g, bch = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    dark = (r < 140) & (g < 140) & (bch < 140)

    lo, hi = int(label_right_edge), int(label_right_edge) + 40
    col_frac = dark.mean(axis=0)
    left = lo + int(np.argmax(col_frac[lo:hi]))

    lo2, hi2 = int(W * 0.85), int(W * 0.995)
    right = hi2
    for i in range(lo2, hi2):
        if col_frac[i] > 0.5:
            right = i
            break

    row_frac = dark[:, left:right].mean(axis=1)
    strong_rows = np.where(row_frac > 0.5)[0]
    if len(strong_rows) == 0:
        return left, right, int(CHROME_SKIP_FRAC * H), H - 1

    bottom = int(strong_rows[np.argmin(np.abs(strong_rows - b))])
    est_top = a * max_tick_val + b
    below = strong_rows[strong_rows < bottom - 20]
    top = int(below[np.argmin(np.abs(below - est_top))]) if len(below) else int(max(est_top, 0))
    return left, right, top, bottom


def _ink_mask(region: np.ndarray) -> np.ndarray:
    """True where a pixel is plot "ink" (trace, marker, annotation - any
    color), False for plain white background or a gridline.

    Gridline gray is deliberately given some slack (up to a 20-point
    channel spread, not just near-equal RGB) - a plain equality check left
    a few antialiased near-gray pixels right next to the frame border
    classified as "ink", which produced a phantom 1-pixel-tall peak at the
    very top of the plot on a real report.
    """
    r, g, b = region[:, :, 0], region[:, :, 1], region[:, :, 2]
    white = (r > 225) & (g > 225) & (b > 225)
    gray = (~white) & (r > 140) & (np.abs(r - g) < 20) & (np.abs(g - b) < 20) & (np.abs(r - b) < 20)
    return ~(white | gray)



# A run whose pixels are (mostly) this saturated a blue is exempt from the
# tall-run marker/cursor-line cap in _find_peak_pixel - see that function's
# docstring point 2, and _is_blue_ish below for why this is a fixed domain
# threshold rather than a per-report "dominant color" estimate.
BLUE_ISH_MIN_MARGIN = 70  # how much B must exceed BOTH R and G by
BLUE_ISH_MIN_FRAC = 0.5   # fraction of a run's own pixels that must qualify


def _is_blue_ish(pixels: np.ndarray) -> np.ndarray:
    """True per-pixel wherever B is clearly the dominant channel - the
    genuine trace's own color on every real report seen, whether a rich,
    saturated blue (26, 6, 233) or a lighter, washed-out one (125, 128,
    253) - both comfortably clear BLUE_ISH_MIN_MARGIN=40 on both channels.

    Deliberately broad rather than a tight "must be richly saturated"
    test (tried first): on one real report, the genuine trace itself is
    that lighter, washed-out blue, and a tight saturation requirement
    wrongly excluded a real, severe peak drawn in it, right along with the
    decorative marker text that happens to share a similar hue on that
    same report. What actually needs excluding is red and magenta, the
    two on-chart marker colors confirmed on real reports (a cursor/order-
    marker line, and a harmonic-flag callout line/box) - and both fail
    this test cleanly: red (191, 0, 0) has B far below R; magenta
    (250, 0, 253) has R essentially tied with B, not clearly behind it.
    A cyan zoom-box marker color (126, 213, 253), also confirmed on a real
    report, fails too - B only edges out G by ~40, under the margin -
    though that one is caught by the width-plateau check regardless of
    color, since a box is many columns wide; this is defense in depth for
    it, not the only thing standing between it and a false read.
    """
    r, g, b = pixels[:, 0].astype(int), pixels[:, 1].astype(int), pixels[:, 2].astype(int)
    return (b - r > BLUE_ISH_MIN_MARGIN) & (b - g > BLUE_ISH_MIN_MARGIN)


def _find_peak_pixel(arr: np.ndarray, left: int, right: int, top: int, bottom: int):
    """Find the (row, col) of the Spectrum plot's tallest genuine data
    peak, in panel-pixel coordinates - the core of "read the actual graph,
    not markings that get in the way".

    On these reports, on-chart markers/annotations come in two shapes that
    both need excluding, and one shape that's fine to leave in:

    1. Wide flat-topped blocks - a colored bar flagging a fault frequency,
       a zoom-selection box, etc. Many adjacent columns share (about) the
       same topmost ink row. A real spectral peak is one frequency wide,
       at most a couple of pixels - so any run of more than
       MAX_PLATEAU_WIDTH_PX columns at the same height is a block, not
       data - EXCEPT a column that's already blue-dominant there
       (_is_blue_ish, same test point 2 uses) is excluded from that count
       and left alone entirely, not just exempted from the cap. Two real
       reports needed this: a rotated harmonic-flag label box sitting
       directly on top of a severe peak (the box's bottom edge touches the
       peak's tip with no row gap, so without the exemption the peak's own
       column reads as just more of the same plateau), and, separately, a
       severe peak with no marker involved at all, whose own several
       antialiased-width columns near its near-vertical rise were on their
       own enough columns to exceed MAX_PLATEAU_WIDTH_PX. A plain "applies
       regardless of color" version of this check (tried first) reads
       either of those as a wide block and erases the peak along with it -
       exactly backwards, same failure shape as the height cap in point 2
       ignoring blue would have, and for the same reason: the taller and
       more severe a real peak is, the more columns of its own width look
       "flat" by this test. A column inside a detected block does NOT just
       get dropped, though - it gets peeled: the block's own run in that
       column is skipped, and whatever ink comes after it (below it) in
       that same column is looked at next, same as for a tall thin mark
       below. This matters because a marker can sit directly ON TOP of
       genuine data in the same column, not just next to it - confirmed on
       a real report, a bar box + its callout line hid a real peak that
       was over a full labeled gridline taller than the peak this function
       used to report before that column was ever looked at below the box.
       See MAX_PEEL_PASSES for how many stacked layers one column can have
       peeled before giving up on it.
    2. Tall thin marks that AREN'T richly-saturated blue - rotated
       marker-label text, a UI cursor/order-marker line (e.g. red, with a
       small square handle sitting right at/above the frame's top edge -
       confirmed against real reports; the small circle sitting just below
       that handle is part of the same marker, not a peak annotation) -
       only 1-2 columns wide, so the width check above doesn't catch them.
       What does: their own column runs solid ink for most of the plot's
       height (verified on a real report: 318-474px on a ~430px-tall
       plot). Antialiasing and letter-shaped gaps inside a marker can chop
       that long run into several short ones a few pixels apart -
       GAP_MERGE_PX bridges gaps that small before measuring, so the
       marker doesn't masquerade as a short (real-looking) run. This
       height cap is deliberately NOT applied to a run that's (mostly)
       blue-dominant in hue - see _is_blue_ish - a genuine sharp, narrow-
       bandwidth resonance can legitimately run near-vertically for most
       of the plot's height in a single column (confirmed on two real
       reports, one with a richly-saturated trace and one with a lighter,
       washed-out one - the taller and more severe the true peak, the more
       likely a color-blind height cap is to wrongly exclude it, which is
       exactly backwards), and the real peak is reliably THICKER (more ink
       per row) than a hairline cursor/order-marker line even where they
       sit close together in x. _is_blue_ish uses a fixed threshold tuned
       from real reports' own baseline trace color rather than a per-
       report "what's the dominant color here" estimate - tried that
       first (twice), and both approaches picked a report's decorative
       marker-label text over its actual trace on a report where the two
       happen to be rendered in a similar hue - see that function's
       docstring for the full story before changing this.
    3. Small annotations directly on a real peak (a circle, a number, a
       triangle) - a few pixels, sitting right at or just above the
       genuine tip. These pass every filter above and are left in on
       purpose: they shift the reading by at most a few pixels, and that's
       an acceptable trade for not needing a marker color palette that
       would have to be hand-maintained per report style.

    Ahead of all three checks above, one column-local, color-blind rule
    runs first: a column whose ink runs unbroken from the very top row of
    the plot area to the very bottom row (within FULL_HEIGHT_TOL_PX of
    literally touching both edges) is thrown out outright, before
    anything else looks at it. This is the UI cursor/order-marker line
    from point 2 above, caught by its shape alone this time instead of by
    failing the blue-ish test - confirmed on a real report to run exactly
    row 0 to the last row, edge to edge, which no genuine peak does (even
    a severe one - see point 2's own genuine-tall-peak examples, neither
    reaches literally either edge). Doesn't change any known report's
    reading (the height-cap-plus-color-check in point 2 already excludes
    it) - it's a second, color-independent way to reach the same answer,
    which matters because point 2's blue-ish check is a per-pixel color
    judgment call and this one isn't.

    Returns (row, col) of the winning pixel, or (None, None) if the plot
    area has no ink at all.
    """
    y0, y1 = top + 1, bottom
    region = arr[y0:y1, left + 2 : right - 1, :]
    ink = _ink_mask(region)
    H, W = ink.shape
    run_cap = MAX_RUN_HEIGHT_FRAC * H

    # Color-blind full-height exclusion (see docstring): a column whose
    # ink is unbroken (gap-merged) from row 0 to row H-1 is entirely a
    # cursor/order-marker line, nothing left in it worth looking at.
    floor = np.zeros(W, dtype=int)
    for c in range(W):
        idx = np.where(ink[:, c])[0]
        if len(idx) == 0:
            continue
        gaps = np.where(np.diff(idx) > GAP_MERGE_PX)[0]
        first = np.split(idx, gaps + 1)[0]
        if first[0] <= FULL_HEIGHT_TOL_PX and first[-1] >= H - 1 - FULL_HEIGHT_TOL_PX:
            floor[c] = H

    def _first_run(idx: np.ndarray, col: int) -> np.ndarray:
        """Topmost run within already-floor-filtered idx (this column's
        candidate ink rows). Splits on a row gap bigger than GAP_MERGE_PX,
        AND on a sustained change between blue-dominant and not -
        confirmed necessary on a real report where a marker's callout line
        runs directly INTO a genuine peak with no row gap between them at
        all (the line terminates exactly where the peak's own trace
        begins) - row-gap-only splitting saw one continuous run the whole
        way through both, which (in stage 1 below) swallowed the real peak
        into the marker block's own measured extent, well past where the
        marker itself actually ends. A single-pixel color flicker
        (antialiasing) does NOT split - only a change that holds for at
        least MIN_RUN_HEIGHT_PX rows counts, so a genuine peak's own
        antialiased edge pixels don't fragment it.
        """
        row_gap_breaks = np.where(np.diff(idx) > GAP_MERGE_PX)[0]
        is_blue = _is_blue_ish(region[idx, col])
        color_breaks = []
        run_start = 0
        for i in range(1, len(is_blue)):
            if is_blue[i] != is_blue[run_start]:
                end = min(i + MIN_RUN_HEIGHT_PX, len(is_blue))
                if is_blue[i:end].all() if is_blue[i] else (~is_blue[i:end]).all():
                    color_breaks.append(i - 1)
                    run_start = i
        breaks = np.union1d(row_gap_breaks, np.array(color_breaks, dtype=int))
        return np.split(idx, breaks + 1)[0]

    def _topmost_at_or_below(floor: np.ndarray) -> np.ndarray:
        """Per-column topmost ink row at/below that column's floor, H if
        none. Uses ANY ink (color-blind) - purely for locating wide
        marker blocks (docstring point 1), which is color-independent by
        design."""
        result = np.full(W, H)
        for c in range(W):
            idx = np.where(ink[:, c])[0]
            idx = idx[idx >= floor[c]]
            if len(idx):
                result[c] = idx[0]
        return result

    # Stage 1: locate wide flat-topped marker blocks and skip ALL of each
    # one in a single move, advancing every column in it to ONE SHARED
    # floor - not peeling each column in the block separately at its own
    # pace. That distinction matters and was the source of two real bugs:
    # a block's internal ink is rarely uniform column-to-column (letter-
    # shaped gaps in a label, a "ladder" of small handle marks) - if each
    # column instead peels its own first run independently, columns whose
    # own first chunk happens to be short finish an early "layer" faster
    # than their neighbors, drift out of row-alignment with them on the
    # very next look, and then read as their own narrow (not-wide-
    # plateau), and often short-enough-to-look-real, group - even though
    # they're still squarely inside the same marker block. A shared floor
    # can't drift apart like that: the whole block moves together.
    # (floor was already initialized above - full-height columns start
    # this loop pre-excluded.)
    for _ in range(MAX_PEEL_PASSES):
        topmost = _topmost_at_or_below(floor)
        if (topmost == H).all():
            break
        any_wide = False
        c = 0
        while c < W:
            if topmost[c] == H:
                c += 1
                continue
            c2 = c
            while c2 + 1 < W and topmost[c2 + 1] != H and abs(int(topmost[c2 + 1]) - int(topmost[c])) <= PLATEAU_ROW_TOL_PX:
                c2 += 1
            if c2 - c + 1 > MAX_PLATEAU_WIDTH_PX:
                # A candidate plateau can accidentally rope in a genuine
                # peak's own column(s) alongside real marker ink - not
                # because the peak IS a marker, but because something
                # shares its topmost row closely enough to read as the same
                # plateau. Confirmed on two real reports: a harmonic-flag
                # label box sitting directly on top of a severe peak (the
                # box's bottom edge touches the peak's tip, no row gap
                # between them), and, separately, a severe peak with no
                # marker involved at all, whose own several antialiased-
                # width columns near its near-vertical rise were enough
                # columns on their own to exceed MAX_PLATEAU_WIDTH_PX.
                # Either way the fix is the same: a column that's already
                # blue-dominant (_is_blue_ish - the same test Stage 2 below
                # uses to exempt a tall run from its own height cap) is
                # real trace data, not marker ink, so it's excluded here
                # too - both from the count that decides whether this is
                # actually a wide block, and from the shared floor push,
                # so its own ink is left completely alone for Stage 2 to
                # read normally. Without this, the shared floor (next
                # paragraph) gets computed from - and then applied to - a
                # peak that was never part of the marker to begin with,
                # which silently erases it.
                marker_cols = []
                for cc in range(c, c2 + 1):
                    idx = np.where(ink[:, cc])[0]
                    idx = idx[idx >= floor[cc]]
                    if len(idx) == 0:
                        continue
                    fr = _first_run(idx, cc)
                    if _is_blue_ish(region[fr, cc]).mean() >= BLUE_ISH_MIN_FRAC:
                        continue
                    marker_cols.append((cc, int(fr[-1])))

                if len(marker_cols) > MAX_PLATEAU_WIDTH_PX:
                    any_wide = True
                    # Shared floor = just past the farthest this block's own
                    # (gap-merged) ink reaches, across every marker column in
                    # it - not just the shallowest one, so a stray deeper
                    # column doesn't leave the rest of the block only half-
                    # cleared. Blue-dominant columns (excluded above) never
                    # contribute to this, and never get it applied to them.
                    block_end = max(end for _, end in marker_cols)
                    for cc, _end in marker_cols:
                        floor[cc] = block_end + 1
            c = c2 + 1
        if not any_wide:
            break

    # Stage 2: within what's left (floor already past every wide block),
    # exclude a tall thin mark that ISN'T blue-dominant in hue (docstring
    # point 2) - a real cursor/order-marker line, or leftover marker-label
    # text too narrow to have been a wide plateau above. Column-
    # independent, so the drift problem stage 1 has doesn't apply here -
    # each column is judged only against its own ink, never grouped with
    # neighbors, so there's nothing to desynchronize.
    resolved = np.zeros(W, dtype=bool)
    resolved_topmost = np.full(W, H)
    for c in range(W):
        idx = np.where(ink[:, c])[0]
        idx = idx[idx >= floor[c]]
        for _ in range(MAX_PEEL_PASSES):
            if len(idx) == 0:
                break
            run = _first_run(idx, c)
            run_len = run[-1] - run[0] + 1
            if run_len >= MIN_RUN_HEIGHT_PX and (
                run_len <= run_cap or _is_blue_ish(region[run, c]).mean() >= BLUE_ISH_MIN_FRAC
            ):
                resolved[c] = True
                resolved_topmost[c] = run[0]
                break
            idx = idx[idx > run[-1]]  # too short to trust, or a tall non-blue mark - peel past it, keep looking

    valid_cols = [c for c in range(W) if resolved[c]]
    if not valid_cols:
        # Nothing ever resolved (e.g. every column is all-marker, all the
        # way down) - fall back to plain topmost-of-any-ink so this still
        # returns something rather than nothing.
        any_topmost = np.full(W, H)
        for c in range(W):
            idx = np.where(ink[:, c])[0]
            if len(idx):
                any_topmost[c] = idx[0]
        valid_cols = [c for c in range(W) if any_topmost[c] != H]
        if not valid_cols:
            return None, None
        best_col = min(valid_cols, key=lambda c: any_topmost[c])
        return y0 + int(any_topmost[best_col]), left + 2 + best_col

    best_col = min(valid_cols, key=lambda c: resolved_topmost[c])
    return y0 + int(resolved_topmost[best_col]), left + 2 + best_col


def _infer_grid_ticks(values: set[float]) -> set[float]:
    """Fill in gridline VALUES that sit strictly between two already-OCR'd
    tick labels, when doing so is essentially certain rather than a guess.

    Confirmed on two real reports: the axis had real, evenly-spaced
    gridlines at a 0.5 step (0, 0.5, 1, 1.5...), but the "1" label sat
    close enough beneath the auto-scaled top-of-axis label ("1.05") that
    Tesseract couldn't read it at any --psm mode or upscale factor tried
    (a single "1" glyph carries very little pixel information to begin
    with) - so calibration only ever recovers 0, 0.5, and 1.05, leaving
    _floor_to_axis_label nothing to floor a ~1.0 reading down to but 0.5,
    even though the real "1" gridline is sitting right there on the chart,
    unread rather than absent.

    The fix doesn't try to OCR harder - it reasons from what's already
    confirmed: these axes are evenly gridded, so the SMALLEST gap between
    two already-RANSAC-confirmed real values is the axis's own grid step
    (a smaller true step would mean two real gridlines closer together
    than anything actually observed, which is a contradiction - two
    confirmed points can't both be real and closer than the true step).
    Every multiple of that step lying strictly between the lowest and
    highest confirmed value is then a gridline that has to physically
    exist on the chart whether or not its printed label survived OCR, so
    it's added to the floor-candidate set. Nothing is ever extrapolated
    PAST the confirmed range (that would be a real guess, not an
    inference) - a step this fills gaps INSIDE, never projects outward
    from. n_steps is capped as a guard against a degenerate near-zero gap
    (e.g. two OCR reads of the same physical label a fraction of a pixel
    apart) turning this into a very long, pointless loop; if that cap
    trips, this returns the original values unchanged rather than a
    partial fill.
    """
    vals = sorted(values)
    if len(vals) < 2:
        return set(vals)
    gaps = [b - a for a, b in zip(vals, vals[1:]) if b - a > 1e-9]
    if not gaps:
        return set(vals)
    step = min(gaps)
    lo, hi = vals[0], vals[-1]
    n_steps = int((hi - lo) / step + 1e-6)
    if n_steps > 200:
        return set(vals)
    out = set(vals)
    for i in range(n_steps + 1):
        candidate = lo + i * step
        if not any(abs(candidate - v) <= step * 0.1 for v in out):
            out.add(round(candidate, 10))
    return out


def _floor_to_axis_label(value: float, kept_points: list[tuple[float, float]]) -> float:
    """Snap value down to the largest y-axis tick label at or below it -
    "read the peak, match it to the closest label rounded down" - rather
    than reporting a continuous interpolated number. This deliberately
    trades a little precision for staying anchored to a number that's
    actually printed on the chart - see _infer_grid_ticks for the one
    deliberate, narrow exception (a gridline value strictly between two
    OCR'd labels, and therefore essentially certain to be real even
    though Tesseract itself never read it).
    """
    ticks = sorted(_infer_grid_ticks({v for _, v in kept_points} | {0.0}))
    floor_val = 0.0
    for t in ticks:
        if t <= value:
            floor_val = t
        else:
            break
    return floor_val


def read_spectrum_peak(chart_image: Image.Image) -> dict:
    """Read the Spectrum plot's tallest genuine peak off its own y-axis.

    Returns a dict with:
      peak_amplitude         the peak's value (float, or None if
                              calibration or peak-finding failed) - a
                              continuous estimate from linearly
                              interpolating the peak's pixel row against
                              the calibrated y-axis (see below), NOT
                              snapped to the nearest printed gridline
      peak_amplitude_floored the same reading snapped down to the nearest
                              y-axis label at or below it (float or None) -
                              kept for cross-checking against a printed
                              number on the chart by eye; not used for
                              priority thresholds
      y_axis_ticks            the (value) labels used for calibration, for
                              sanity-checking against the actual chart - can
                              include a value strictly between two OCR'd
                              labels that Tesseract itself never read (see
                              _infer_grid_ticks), not only literal OCR output
      error                 None on success, else a short string saying
                             what failed (e.g. "could not OCR enough y-axis
                             tick labels to calibrate")

    peak_amplitude used to be the floored reading, on the reasoning that
    trading precision for staying anchored to a number actually printed on
    the chart was worth it. Changed after a direct real-report comparison
    (report_153: floored to 1.0, interpolated to 1.21, a person reading
    the same chart by eye also independently said "about 1.2") - flooring
    can only ever revise a reading DOWN, never up, so across many reports
    it's a systematic downward bias on severity, not neutral rounding.
    It also turned out to be quietly inconsistent with how this project's
    own priority thresholds are fit: velocity_priority_hint and
    acceleration_enveloping_priority_hint are refit against hand-eyeballed
    chart readings (see their docstrings) - a person reading "about 1.2"
    off a chart was never flooring to the nearest printed gridline either,
    so scoring the automated reading against thresholds fit to that kind
    of number, while itself floored, was comparing two different things.
    _floor_to_axis_label / peak_amplitude_floored are kept, not deleted -
    still useful for manually cross-checking a specific reading against
    the chart's own printed numbers - just no longer what feeds
    spectrum_priority_hint or gets used as priority thresholds are fit.

    Never raises - any failure (OCR found <2 usable tick labels, frame
    border not found, empty plot area, ...) comes back as
    peak_amplitude=None with `error` explaining why, so one bad chart image
    doesn't take down a whole build_dataset() run. Callers should treat
    peak_amplitude=None the same as "no signal available", same as the old
    Fund Amp path did below its own noise floor.
    """
    try:
        panel = _crop_spectrum_panel(chart_image)
        points, label_right_edge = _read_y_axis_ticks(panel)
        if label_right_edge is None:
            return {"peak_amplitude": None, "peak_amplitude_floored": None, "y_axis_ticks": [], "error": "no y-axis tick labels OCR'd"}

        calibration = _ransac_calibration(points)
        if calibration is None:
            return {"peak_amplitude": None, "peak_amplitude_floored": None, "y_axis_ticks": [], "error": "could not calibrate y-axis (fewer than 2 consistent tick labels)"}
        a, b, kept = calibration

        arr = np.asarray(panel.convert("RGB")).astype(int)
        max_tick_val = max(v for _, v in kept)
        left, right, top, bottom = _find_plot_frame(arr, label_right_edge, a, b, max_tick_val)
        if right <= left + 4 or bottom <= top + 4:
            return {"peak_amplitude": None, "peak_amplitude_floored": None, "y_axis_ticks": sorted(_infer_grid_ticks({v for _, v in kept})), "error": "plot frame border not found"}

        peak_row, _peak_col = _find_peak_pixel(arr, left, right, top, bottom)
        if peak_row is None:
            return {"peak_amplitude": None, "peak_amplitude_floored": None, "y_axis_ticks": sorted(_infer_grid_ticks({v for _, v in kept})), "error": "no data ink found in plot area"}

        raw_value = (peak_row - b) / a
        floored = _floor_to_axis_label(raw_value, kept)
        return {
            "peak_amplitude": round(raw_value, 4),
            "peak_amplitude_floored": floored,
            "y_axis_ticks": sorted(_infer_grid_ticks({v for _, v in kept})),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 - one bad image shouldn't kill a batch run
        return {"peak_amplitude": None, "peak_amplitude_floored": None, "y_axis_ticks": [], "error": f"unexpected error: {exc}"}


# --- Priority thresholds per unit ---------------------------------------
# All three read off the Spectrum peak amplitude (read_spectrum_peak above),
# never off Fund Amp - see module docstring.


def velocity_priority_hint(amp: float) -> int:
    """Velocity (in/s) peak amplitude -> priority, PUMP EQUIPMENT ONLY.
    >1.44 -> 1, 0.344-1.44 -> 2, 0.113-0.344 -> 3, <0.113 -> 4.

    Refit against the project owner's own hand-eyeballed amplitude
    readings (a "peak amplitude (eyeball)" column, read directly off each
    chart by a person - not this file's pixel-read
    spectrum_peak_amplitude) for 88 pump reports (priority_raw: 51 P4, 16
    P3, 17 P2, 4 P1), superseding an earlier fit against the pixel-read
    amplitude on a larger (187-report) but noisier sample. Same
    weighted-F1 grid search as before (every candidate boundary triple,
    midpoints between consecutive distinct observed amplitudes, t1<t2<t3),
    just against cleaner ground truth this time. Resulting accuracy: 75%
    (n=88), Priority-1 recall 50% (2 of 4 - a genuinely thin sample, worth
    re-checking once more Priority-1 pumps get eyeballed).

    Kept to pumps deliberately: fan equipment has different amplitude
    behavior than pumps (per the project owner) - do not port these
    numbers to ATS-Fans-Project or apply them to a mixed equipment set.

    Hand-eyeballing the amplitudes did NOT resolve the underlying overlap
    between priorities - confirmed directly on this data: "Thermal Fluid
    HX Pmp" was stated Priority 1 (severe) at a hand-verified 0.219 in/s,
    well inside where most Priority 4 (good) reports sit. Per the project
    owner, this reflects real inconsistency in how priority gets assigned
    to pump reports, not a reading-quality problem - so treat 75% as
    close to the practical ceiling for a single-amplitude threshold rule
    here, not a number a better fit would meaningfully beat.
    """
    if amp > 1.44:
        return 1
    if amp >= 0.344:
        return 2
    if amp >= 0.113:
        return 3
    return 4


def acceleration_enveloping_priority_hint(amp: float) -> int:
    """Acceleration enveloping (gE) peak amplitude -> priority, PUMP
    EQUIPMENT ONLY. >1.28 -> 1, 0.179-1.28 -> 2, 0.048-0.179 -> 3,
    <0.048 -> 4.

    Refit against the project owner's own hand-eyeballed amplitude
    readings (a "peak amplitude (eyeball)" column, read directly off each
    chart by a person - not this file's pixel-read
    spectrum_peak_amplitude) for 122 pump reports (priority_raw: 25 P4, 25
    P3, 61 P2, 11 P1), superseding an earlier fit against the pixel-read
    amplitude on a larger (187-report) but noisier sample. Same
    weighted-F1 grid search as velocity_priority_hint. Resulting accuracy:
    54% (n=122), lower than the prior pixel-read fit's 58% but with better-
    balanced recall across classes (Priority-1 recall 55%, 6 of 11, vs.
    45% before) - the search here didn't hit the earlier fit's "P2/P1 near
    20" degenerate-boundary problem (checked the top several candidates
    directly; nothing near that shape showed up this time).

    Kept to pumps deliberately: fan equipment has different amplitude
    behavior than pumps (per the project owner) - do not port these
    numbers to ATS-Fans-Project or apply them to a mixed equipment set.

    Hand-eyeballing the amplitudes did NOT resolve the underlying overlap
    between priorities - confirmed directly on this data: "Trailer Dump
    Hyd Pump" was stated Priority 4 (good) at a hand-verified 0.626 gE, a
    reading higher than the median Priority 1 or 2 report in this same
    sample. Per the project owner, this reflects real inconsistency in how
    priority gets assigned to pump reports, not a reading-quality problem
    - so treat 54% as close to the practical ceiling for a single-
    amplitude threshold rule here, not a number a better fit would
    meaningfully beat.
    """
    if amp > 1.28:
        return 1
    if amp >= 0.179:
        return 2
    if amp >= 0.048:
        return 3
    return 4


def acceleration_priority_hint(amp: float) -> int:
    """Acceleration (g) peak amplitude -> priority. >2.5 -> 1, >2 -> 2,
    >1 -> 3, else -> 4. Few reports use plain g (most are gE) - thresholds
    here are as given, not independently re-derived from a large sample."""
    if amp > 2.5:
        return 1
    if amp > 2:
        return 2
    if amp > 1:
        return 3
    return 4


_UNIT_HINT_FNS = {
    "in/s": velocity_priority_hint,
    "gE": acceleration_enveloping_priority_hint,
    "g": acceleration_priority_hint,
}


def spectrum_priority_hint(chart_image: Image.Image, ocr_text: str) -> dict:
    """Combine unit detection (OCR text) with the pixel-read Spectrum peak
    (read_spectrum_peak) into one supporting priority signal.

    Needs the chart IMAGE now, not just its OCR text - unlike the old Fund
    Amp version, the peak reading is pixel analysis, not a text field.
    """
    unit = detect_spectrum_unit(ocr_text)
    peak = read_spectrum_peak(chart_image)
    amp = peak["peak_amplitude"]
    hint_fn = _UNIT_HINT_FNS.get(unit)
    priority_hint = hint_fn(amp) if (hint_fn is not None and amp is not None) else None
    return {
        "spectrum_unit": unit,
        "spectrum_peak_amplitude": amp,
        "spectrum_peak_amplitude_floored": peak["peak_amplitude_floored"],
        "spectrum_priority_hint": priority_hint,
        "spectrum_peak_error": peak["error"],
    }
