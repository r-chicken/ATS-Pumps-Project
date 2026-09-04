"""Standalone scoring app - upload a report PDF, get back whether the
text and/or the Spectrum chart agree with the priority the report states.

Deliberately separate from the Colab notebook's training pipeline (see
webapp/README.md): this app never re-runs OCR across a whole PDF folder
or retrains anything. It loads the ALREADY-TRAINED model bundle
(model/priority_classifier.joblib + model/priority_classifier.meta.json -
copy these out of Drive after a Colab training run, see README) and
scores whatever PDF(s) get uploaded through the same per-report path the
notebook's own "Section 7: Scoring brand-new reports" cell uses - fast,
no batch dataset needed.
"""
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

import pandas as pd
from flask import Flask, render_template, request

# ats_priority_checker lives one directory up from this file (repo root).
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from ats_priority_checker.extract import process_pdf  # noqa: E402
from ats_priority_checker.model import load_bundle, priority_recommendation_table, report_text  # noqa: E402

MODEL_DIR = Path(__file__).resolve().parent / "model"
MODEL_PATH = MODEL_DIR / "priority_classifier.joblib"

app = Flask(__name__)

# Loaded once per running process, not once per request - both the
# classifier and the sentence-embedding model are meant to be reused, not
# rebuilt for every upload. model.py's own embed_texts() reloads the
# SentenceTransformer from scratch on every call (fine for a one-shot
# notebook cell, wasteful here), so this app calls .encode() on a
# cached instance directly instead of going through that helper.
_state: dict = {}


def _get_state() -> dict:
    if not _state:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No model found at {MODEL_PATH}. Copy priority_classifier.joblib and "
                f"priority_classifier.meta.json from your Colab run's Drive output into "
                f"webapp/model/ before building/deploying this app - see webapp/README.md."
            )
        clf, embedding_model_name = load_bundle(MODEL_PATH)
        from sentence_transformers import SentenceTransformer

        _state["clf"] = clf
        _state["embedder"] = SentenceTransformer(embedding_model_name)
    return _state


def _score_pdfs(paths: list[Path]) -> pd.DataFrame:
    """Run the same per-report extraction path as process_pdf, for every
    page of every uploaded PDF, then score each one against the loaded
    model - no batch dataset, no escalation history, just this report's
    own text and this report's own Spectrum chart against its own stated
    priority (see model.priority_recommendation_table)."""
    state = _get_state()
    rows = []
    for path in paths:
        try:
            records = process_pdf(path, max_pages=1)
        except Exception as exc:  # noqa: BLE001 - one bad upload shouldn't kill the batch
            rows.append({"report_id": path.name, "priority_raw": None, "parse_notes": f"failed to process: {exc}"})
            continue
        for page_number, rec in enumerate(records, start=1):
            d = dataclasses.asdict(rec)
            d["report_id"] = f"{path.stem}_p{page_number}"
            rows.append(d)

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    usable = df.dropna(subset=["priority_num"]).copy()
    if usable.empty:
        df["predicted_priority"] = None
        return df

    texts = usable.apply(report_text, axis=1).tolist()
    embeddings = state["embedder"].encode(texts, normalize_embeddings=True)
    usable["predicted_priority"] = state["clf"].predict(embeddings)

    table = priority_recommendation_table(usable)
    # Carry through parse_notes for any pages that failed to parse a
    # priority at all, so the upload result still explains every page,
    # not just the ones that scored successfully.
    unusable = df[df["priority_num"].isna()][["report_id"]].copy()
    if not unusable.empty:
        unusable["parse_notes"] = df.loc[unusable.index, "parse_notes"] if "parse_notes" in df.columns else "no priority found on this page"
        table = pd.concat([table, unusable], ignore_index=True)
    return table


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    uploads = [f for f in request.files.getlist("pdfs") if f.filename]
    if not uploads:
        return render_template("index.html", error="Choose at least one PDF first.")

    with tempfile.TemporaryDirectory() as tmp_dir:
        paths = []
        for f in uploads:
            if not f.filename.lower().endswith(".pdf"):
                continue
            path = Path(tmp_dir) / f.filename
            f.save(path)
            paths.append(path)

        if not paths:
            return render_template("index.html", error="Only .pdf files are supported.")

        try:
            table = _score_pdfs(paths)
        except FileNotFoundError as exc:
            return render_template("index.html", error=str(exc))

    if table.empty:
        return render_template("index.html", error="No readable report pages found in the uploaded PDF(s).")

    return render_template(
        "results.html",
        columns=table.columns.tolist(),
        rows=_format_table_for_display(table),
    )


def _format_table_for_display(table: pd.DataFrame) -> list[dict]:
    """Convert every cell to a display-ready string (blank for missing,
    Yes/No for booleans - including pandas' nullable "boolean" dtype,
    which otherwise prints as the confusing literal string "<NA>") and
    flag each row's overall disagreement state for the template to
    highlight, rather than leaving raw NaN/pd.NA/True/False to print
    as-is in HTML.
    """
    display = table.copy()
    for col in display.columns:
        if str(display[col].dtype) == "boolean":  # pandas nullable boolean
            display[col] = display[col].map({True: "Yes", False: "No"}).fillna("")
        elif display[col].dtype == bool:
            display[col] = display[col].map({True: "Yes", False: "No"})
        else:
            display[col] = display[col].fillna("")
    rows = display.to_dict(orient="records")
    for row, (_, raw_row) in zip(rows, table.iterrows()):
        row["_flagged"] = bool(raw_row.get("any_disagreement") is True)
    return rows


@app.route("/healthz", methods=["GET"])
def healthz():
    return {"status": "ok", "model_loaded": MODEL_PATH.exists()}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
