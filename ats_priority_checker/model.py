"""Text -> priority model, and the mismatch-flagging logic built on top of it.

Given only a few hundred reports, this deliberately does NOT train a neural
net from scratch. It uses a pretrained sentence-embedding model (frozen,
not fine-tuned) to turn report text into vectors, then trains a simple
classifier on top to predict "what priority does this text imply". A
report is flagged when the stated priority disagrees with what the text
implies.

This design is meant to scale with more data later: same pipeline, same
saved artifacts format, just re-run train_priority_classifier on a bigger
dataset.csv as more labeled/parsed reports come in.

Two different questions, two different functions - do not conflate them:
  - "Should THIS report get a second look?" -> flag_mismatches(). Compares
    every signal against priority_num, the priority the REPORT ITSELF
    states - the only number that exists on a brand-new report nobody has
    reviewed yet (see dataset.build_dataset's Section 7 use case: scoring
    new PDFs with no labels at all). This is the function that has to run
    without ground truth, so it's built to never need it.
  - "Is this system actually any good?" -> priority_signal_reports() /
    priority_signal_table(). Compare every signal against true_priority,
    YOUR hand-corrected ground truth - only available for the subset
    you've manually labeled (labeling.py). On rows you've already labeled
    "mismatch", agreement with the stated priority_num is close to a BAD
    sign (it means the signal reproduced the very error you caught), so
    don't read flag_mismatch as a quality check on those rows - go
    straight to priority_signal_reports/priority_signal_table instead.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"


def report_text(row: pd.Series) -> str:
    """Combine recommendations + comments into the text the model sees."""
    recs = row.get("recommendations") or ""
    comments = row.get("comments") or ""
    return f"Recommendations: {recs}\nComments: {comments}".strip()


def embed_texts(texts: list[str], model_name: str = EMBEDDING_MODEL_NAME) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_name)
    return model.encode(list(texts), show_progress_bar=True, normalize_embeddings=True)


def cross_validated_predictions(X: np.ndarray, y: np.ndarray, n_splits: int = 5) -> dict:
    """Out-of-fold predictions: for every row, the predicted priority comes
    from a model that never saw that row during training. This is what you
    want when flagging mismatches on your existing labeled reports - using
    a model's in-sample predictions on its own training data would make
    the stated priority look "confirmed" more often than it should.
    """
    class_counts = pd.Series(y).value_counts()
    n_splits = min(n_splits, int(class_counts.min())) if len(class_counts) > 0 else n_splits
    n_splits = max(n_splits, 2)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")

    pred = cross_val_predict(clf, X, y, cv=skf, method="predict")
    proba = cross_val_predict(clf, X, y, cv=skf, method="predict_proba")
    classes = np.unique(y)

    return {"pred": pred, "proba": proba, "classes": classes, "n_splits": n_splits}


def fit_final_model(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    """Fit on ALL available data - this is the model you save and reuse for
    scoring brand-new reports (not for evaluating the training set itself,
    use cross_validated_predictions for that)."""
    clf = LogisticRegression(max_iter=2000, class_weight="balanced")
    clf.fit(X, y)
    return clf


def flag_mismatches(
    df: pd.DataFrame,
    pred: np.ndarray,
    proba: np.ndarray,
    classes: np.ndarray,
    low_confidence_threshold: float = 0.15,
) -> pd.DataFrame:
    """Compare stated priority to text-implied priority, and to the
    Spectrum peak-amplitude graph signal where available.

    Flags a row when:
      - the predicted (argmax) priority differs from the stated priority, OR
      - the model's confidence in the STATED priority is very low, even if
        it isn't the top prediction for another class (catches "text reads
        as ambiguous / doesn't clearly support this priority" cases, not
        just clean disagreements), OR
      - the Spectrum peak-amplitude hint (see graph_signals.py) suggests a
        MORE urgent priority than stated.

    Adds text_disagrees and spectrum_disagrees as their own boolean
    columns (not just folded into flag_mismatch) - given vs. text-implied,
    and given vs. spectrum-implied, each on its own, for scanning which
    source(s) actually drove a flag without re-deriving the comparison by
    hand. flag_mismatch itself is (text_disagrees | low-confidence-in-
    stated | spectrum_disagrees) - see below on why spectrum_disagrees is
    one-directional while text_disagrees isn't.

    The graph hint is allowed to flag on its own, even when the text
    agrees with the stated priority - the whole reason it exists is to
    catch cases the text can't: report writers sometimes under-state
    urgency for equipment that's been high-priority before, in which case
    the text itself may honestly match a priority that's actually too low.

    The graph hint only flags in ONE direction: hint < stated (lower number
    = more urgent). A hint suggesting LESS urgency than stated is not a
    mismatch signal - it just means the graph alone doesn't capture
    whatever else supports the higher stated priority (comments, repair
    history, other context), which is expected and not what this signal
    exists to catch. Flagging both directions was tried and produced
    disagreements on ~9-14% of otherwise-correct reports with zero gain in
    real mismatch detection (see the conversation this fix is from for the
    concrete evidence) - real prioritization is holistic, so a raw
    threshold rule landing one level more cautious than the stated
    priority is normal and not evidence of an error.

    A cross-report escalation signal (comparing against the same
    equipment's prior dated test) used to be a second graph hint here -
    tried and removed (see graph_signals.py's module docstring for why:
    this report set has multiple measurement points per equipment, often
    in different units, and there wasn't a reliably consistent per-machine
    timeline to compare against). Don't reintroduce one without reading
    that first.

    IMPORTANT - what "stated priority" means here: every comparison in this
    function is against priority_num (what the report itself says), NEVER
    against true_priority (your hand-corrected ground truth), even when
    true_priority is sitting right there in df. That's deliberate, not an
    oversight - this function's whole job is to work on a report NOBODY has
    reviewed yet, where true_priority doesn't exist. Do not change this to
    prefer true_priority when available "for accuracy" - that would make
    flag_mismatch silently behave differently on your labeled rows than on
    every future unlabeled report, which defeats the point of validating it
    on the labeled set in the first place. If you want to know how well
    flag_mismatch (or any one signal) tracks your actual corrections, use
    priority_signal_reports()/priority_signal_table() on the labeled
    subset - don't repurpose this function's output for that; a False here
    on an already-hand-labeled "mismatch" row does NOT mean the signal got
    it right, only that it agreed with the report's own (already-wrong)
    number.
    """
    out = df.copy()
    out["predicted_priority"] = pred

    class_to_idx = {c: i for i, c in enumerate(classes)}
    stated_conf = []
    for stated, row_proba in zip(out["priority_num"], proba):
        idx = class_to_idx.get(stated)
        stated_conf.append(row_proba[idx] if idx is not None else np.nan)
    out["confidence_in_stated_priority"] = stated_conf

    text_disagrees = out["predicted_priority"] != out["priority_num"]
    low_conf = out["confidence_in_stated_priority"] < low_confidence_threshold

    def _disagrees(col: str) -> pd.Series:
        if col not in out.columns:
            return pd.Series(False, index=out.index)
        hint = out[col]
        return hint.notna() & (hint < out["priority_num"])

    spectrum_disagrees = _disagrees("spectrum_priority_hint")

    # Exposed as their own columns (not just folded into flag_mismatch)
    # specifically so a caller can see WHICH source(s) drove a flag without
    # re-deriving this logic by hand - text_disagrees is any-direction
    # (predicted_priority != priority_num), spectrum_disagrees is
    # one-direction only (hint < priority_num, i.e. MORE urgent - see the
    # docstring above on why the other direction isn't a mismatch signal).
    # Plain bool, not nullable - False both when the signal is present and
    # agrees, and when spectrum_priority_hint itself is NaN (no chart
    # reading available); that's consistent with how flag_mismatch already
    # treats a missing signal (via _disagrees returning False for a
    # missing column/NaN hint), just visible per-source here instead of
    # only as the combined OR.
    out["text_disagrees"] = text_disagrees
    out["spectrum_disagrees"] = spectrum_disagrees

    out["flag_mismatch"] = text_disagrees | low_conf.fillna(False) | spectrum_disagrees

    def reason(r):
        parts = []
        if r["predicted_priority"] != r["priority_num"]:
            parts.append(f"text implies priority {r['predicted_priority']:g}, report states {r['priority_num']:g}")
        elif r["confidence_in_stated_priority"] < low_confidence_threshold:
            parts.append("text is a weak/ambiguous match for the stated priority")
        spectrum_hint = r.get("spectrum_priority_hint")
        if pd.notna(spectrum_hint) and spectrum_hint < r["priority_num"]:
            parts.append(f"Spectrum peak reading suggests priority {spectrum_hint:g}, report states {r['priority_num']:g}")
        return "; ".join(parts)

    out["flag_reason"] = out.apply(reason, axis=1)
    return out


def priority_signal_reports(df: pd.DataFrame, true_col: str = "true_priority") -> dict:
    """VALIDATION, not deployment - see module docstring. Score each priority
    SIGNAL independently against true_col (your hand-corrected ground
    truth), each with its own per-class precision/recall/f1 - not folded
    into one flag/no-flag number the way flag_mismatch is, and not scored
    against priority_num the way flag_mismatch is either.

    Only meaningful on rows where true_col is populated - i.e. your labeled
    subset. There's no equivalent of this for brand-new unlabeled reports;
    that's what flag_mismatches() is for instead.

    `predicted_priority` (text-only, from the Recommendations/Comments
    embedding model) is not the only signal worth a report card of its own:
    a report can be written in the same mild, boilerplate language every
    other Priority 4 report uses while its own Spectrum chart reads far
    more severe than that language suggests - the text model has no way to
    catch that, since it never sees the chart. Scoring spectrum_priority_hint
    the same way predicted_priority already gets scored surfaces exactly
    that gap.

    Reports on, when present in df:
      - predicted_priority: the text-only model. Broadest coverage (every
        row that went into training/cross-validation).
      - spectrum_priority_hint: THIS report's own Spectrum peak reading -
        no report history needed, so it's available on nearly every report
        with a readable chart. This is the one that catches "the writeup
        reads mild but the chart doesn't" on a single report, standalone.

    (A third, cross-report escalation signal - comparing against the same
    equipment's prior dated test - used to be scored here too. Tried and
    removed; see graph_signals.py's module docstring for why.)

    Each signal is scored ONLY on the rows where it isn't null - you can't
    score a hint that was never produced, and a signal's n here is itself
    informative about how often it actually has something to say, not just
    how accurate it is when it does.

    Returns {signal_name: {"n": int, "true_col": str, "report_text": str,
    "report_dict": dict}} for whichever of the two columns are present in
    df; a report_dict entry is None if all class-metric arrays end up
    empty (currently only ever thrown by sklearn on genuinely undefined
    input, e.g. n=0).

    report_text always leads with a "Ground truth: {true_col}" header
    line, and true_col is also returned as its own key alongside n -
    added after a real mix-up: two classification_report printouts
    (predicted_priority and spectrum_priority_hint) looked directly
    comparable, one call had been made with true_col="true_priority"
    (hand-corrected) and the other, from an earlier snapshot, effectively
    against priority_raw - same-shaped output, silently different ground
    truth, and nothing in the printed report itself said so. Comparing
    across runs (before/after a fix, old snapshot/new) is exactly when
    this matters most, so the header travels with the report text itself
    rather than living only in a caller's memory of which true_col they
    passed.
    """
    from sklearn.metrics import classification_report

    signal_cols = ["predicted_priority", "spectrum_priority_hint"]
    results = {}
    for col in signal_cols:
        if col not in df.columns:
            continue
        sub = df.dropna(subset=[col, true_col])
        header = f"Ground truth: {true_col}\n"
        if len(sub) == 0:
            results[col] = {
                "n": 0,
                "true_col": true_col,
                "report_text": header + "(no rows with both a value and a ground-truth priority)",
                "report_dict": None,
            }
            continue
        results[col] = {
            "n": len(sub),
            "true_col": true_col,
            "report_text": header + classification_report(sub[true_col], sub[col], zero_division=0),
            "report_dict": classification_report(sub[true_col], sub[col], zero_division=0, output_dict=True),
        }
    return results


def priority_signal_table(df: pd.DataFrame, true_col: str = "true_priority") -> pd.DataFrame:
    """VALIDATION, not deployment - see module docstring, and
    priority_signal_reports's docstring above. Row-level companion to
    priority_signal_reports: one row per report with predicted_priority and
    spectrum_priority_hint side by side, plus a `*_correct` bool for each
    against true_col (your hand-corrected ground truth, NOT priority_num) -
    for eyeballing exactly which signal got which report right, not just
    the aggregate precision/recall.

    flag_mismatch is deliberately NOT one of the columns here - it isn't a
    per-signal prediction, it's an OR of several signals compared against
    priority_num, so "flag_mismatch_correct" would conflate two different
    questions (see module docstring) rather than answer either one
    cleanly. If you want to know whether flag_mismatch itself is reliable,
    compare it against (true_col != priority_num) directly - the
    notebook's Section 5 already does this with precision_score/
    recall_score/confusion_matrix.
    """
    signal_cols = [c for c in ("predicted_priority", "spectrum_priority_hint") if c in df.columns]
    id_cols = [c for c in ("report_id", "equipment_id", "priority_num", true_col) if c in df.columns]
    out = df[id_cols + signal_cols].copy()
    for col in signal_cols:
        # NaN (signal didn't fire) is "not applicable", not "wrong" - keep
        # it as a null rather than letting `NaN == true_col` silently read
        # as False. Uses pandas' nullable "boolean" dtype specifically so
        # this prints as True/False/<NA> - plain np.where here upcasts the
        # whole column to float the moment a NaN is mixed in, so "correct"
        # silently renders as 1.0/0.0 sitting right next to the priority-number columns,
        # which reads at a glance like a 4th priority decision instead of a
        # true/false flag.
        correct = pd.Series(np.where(out[col].isna(), pd.NA, out[col] == out[true_col]), index=out.index)
        out[f"{col}_correct"] = correct.astype("boolean")
    return out


def priority_recommendation_table(flagged: pd.DataFrame) -> pd.DataFrame:
    """DEPLOYMENT-style comparison (see module docstring) - everything here
    is checked against priority_raw/priority_num, the priority the REPORT
    ITSELF states, same as flag_mismatches. This is flag_mismatches'
    reasoning laid out as columns instead of folded into one flag_reason
    string - same underlying logic, easier to scan or filter/sort on.

    One column per source, plus whether that source agrees with the stated
    priority:
      - text_recommended_priority / text_agrees_with_stated: predicted_priority,
        i.e. the Recommendations/Comments embedding model, on its own.
      - measurement_point / spectrum_unit / spectrum_peak_amplitude: raw
        context copied straight through from `flagged`, not a verdict of
        their own - included so a reviewer can see what was actually read
        off the chart (which sensor location/direction, which unit, what
        the pixel-read peak was) without re-opening the PDF. Present only
        when that column exists on `flagged`, same as report_id/
        equipment_id/priority_raw above.
      - graph_recommended_priority / graph_agrees_with_stated / graph_note:
        spectrum_priority_hint, i.e. THIS report's own Spectrum peak
        reading, on its own - not blended with anything else.
      - any_disagreement / disagreement_source: True + "text"/"graph"/
        "text, graph" when at least one source above doesn't agree with
        priority_num - the same condition flag_mismatch checks (modulo the
        low-confidence-in-stated-priority trigger, which has no single
        number to show in a "recommended priority" column, so it isn't
        represented here; see flag_mismatch/confidence_in_stated_priority
        on `flagged` directly if you need that specific trigger).

    Works on any DataFrame shaped like flag_mismatches()'s output, not just
    a labeled/cross-validated subset - pass one row per report you have
    text_recommended_priority (and ideally spectrum columns) for, whether
    that's every labeled report or literally every report you've ever
    extracted. See the notebook's "recommendation table for every report"
    cell for how to get predicted_priority onto rows that were never part
    of the labeled cross-validation set, using the final fitted model
    instead of cross_validated_predictions.

    This used to also fold in a cross-report escalation signal (a second
    "graph" source, comparing against the same equipment's prior dated
    test, sometimes overriding the raw spectrum reading, sometimes forcing
    disagreement even when the numbers matched) - tried, iterated on
    heavily, and removed; see graph_signals.py's module docstring for why.
    graph_recommended_priority is exactly spectrum_priority_hint again, no
    exceptions, no override.
    """
    out = pd.DataFrame(index=flagged.index)
    for col in ("report_id", "equipment_id", "priority_raw"):
        if col in flagged.columns:
            out[col] = flagged[col]

    stated = flagged["priority_num"]

    out["text_recommended_priority"] = flagged["predicted_priority"]
    text_agrees = flagged["predicted_priority"] == stated
    out["text_agrees_with_stated"] = pd.Series(
        np.where(flagged["predicted_priority"].isna(), pd.NA, text_agrees), index=flagged.index
    ).astype("boolean")

    # Raw context behind the graph columns below, not a verdict of its own -
    # included so a reviewer can see WHAT was read off the chart without
    # re-opening the PDF.
    for col in ("measurement_point", "spectrum_unit", "spectrum_peak_amplitude"):
        if col in flagged.columns:
            out[col] = flagged[col]

    spectrum = flagged.get("spectrum_priority_hint", pd.Series(np.nan, index=flagged.index))

    out["graph_recommended_priority"] = spectrum
    graph_disagrees = spectrum.notna() & (spectrum != stated)
    out["graph_agrees_with_stated"] = pd.Series(
        np.where(~spectrum.notna(), pd.NA, ~graph_disagrees), index=flagged.index
    ).astype("boolean")

    out["graph_note"] = [
        f"Spectrum reads priority {g:g} vs. stated {st:g}" if gd else ""
        for g, gd, st in zip(spectrum, graph_disagrees, stated)
    ]

    out["any_disagreement"] = text_agrees.eq(False) | graph_disagrees

    def _source(text_dis, g_dis):
        parts = []
        if text_dis:
            parts.append("text")
        if g_dis:
            parts.append("graph (spectrum)")
        return ", ".join(parts)

    out["disagreement_source"] = [
        _source(not ta, gd)
        for ta, gd in zip(text_agrees.fillna(True), graph_disagrees)
    ]

    return out


def save_bundle(clf: LogisticRegression, path: str | Path, embedding_model_name: str = EMBEDDING_MODEL_NAME) -> None:
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(clf, path)
    with open(path.with_suffix(".meta.json"), "w") as f:
        json.dump({"embedding_model_name": embedding_model_name}, f)


def load_bundle(path: str | Path) -> tuple[LogisticRegression, str]:
    import joblib

    path = Path(path)
    clf = joblib.load(path)
    with open(path.with_suffix(".meta.json")) as f:
        meta = json.load(f)
    return clf, meta["embedding_model_name"]
