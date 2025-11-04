import os
import io
import json
import hashlib
import random
import re
import base64
from datetime import datetime
from typing import List, Dict, Any, Tuple

import pandas as pd
from dash import Dash, dcc, html, Input, Output, State, no_update, callback
from dash.exceptions import PreventUpdate

# External styles: Bootswatch Darkly and optional icons
EXTERNAL_STYLESHEETS = [
    "https://cdn.jsdelivr.net/npm/bootswatch@5.3.2/dist/darkly/bootstrap.min.css",
    "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css",
]


def normalize_token(s: str) -> str:
    """Normalize a token for comparison: lowercase, collapse whitespace, strip."""
    if s is None:
        return ""
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def detect_token_list(s: str) -> List[str]:
    """Split an options/correct cell into a list of unique tokens in original order.
    Tries separators in priority order: '|' > newline > ';' (if no commas) > '/' > ','.
    """
    if s is None:
        return []
    s = str(s)
    # Priority: pipe |, newline, semicolon within cell (Excel), slash, comma
    if "|" in s:
        parts = [p.strip() for p in s.split("|")]
    elif "\n" in s:
        parts = [p.strip() for p in s.split("\n")]
    elif ";" in s and "," not in s:  # avoid splitting typical CSV commas
        parts = [p.strip() for p in s.split(";")]
    elif "/" in s and " // " not in s:
        parts = [p.strip() for p in s.split("/")]
    else:
        parts = [p.strip() for p in s.split(",")]
    # Remove empties and duplicates preserving order
    seen = set()
    uniq: List[str] = []
    for p in parts:
        k = normalize_token(p)
        if p and k not in seen:
            uniq.append(p)
            seen.add(k)
    return uniq
@callback(
    Output("progress_plot", "figure", allow_duplicate=True),
    Input("quiz_stats", "data"),
    Input("plot_metric", "value"),
    State("quiz_data", "data"),
    prevent_initial_call=True,
)
def render_progress_plot(stats, metric, quiz_data):
    # Render attempts for current dataset hash
    def empty_fig(y_title: str):
        return {"data": [], "layout": {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "xaxis": {"title": "Attempt"}, "yaxis": {"title": y_title}}}

    if not quiz_data:
        return empty_fig("% korrekt" if (metric or "pct") == "pct" else "Poäng")
    dataset_key = quiz_data.get("hash")
    series = (stats or {}).get(dataset_key, [])
    if not series:
        return empty_fig("% korrekt" if (metric or "pct") == "pct" else "Poäng")
    x = list(range(1, len(series) + 1))
    total = len(quiz_data.get("questions", []) or [])
    if (metric or "pct") == "points":
        # Prefer stored points; fallback to pct*total/100
        y = [
            (pt.get("points") if pt.get("points") is not None else round((pt.get("pct", 0) * (pt.get("total", total) or total) / 100.0), 2))
            for pt in series
        ]
        y_title = "Poäng"
        y_range = [0, total or (max(y) if y else 1)]
    else:
        y = [pt.get("pct", 0) for pt in series]
        y_title = "% korrekt"
        y_range = [0, 100]
    fig = {
        "data": [
            {"type": "scatter", "mode": "lines+markers", "x": x, "y": y, "line": {"color": "#6f9eff"}, "marker": {"color": "#6f9eff"}},
        ],
        "layout": {
            "paper_bgcolor": "rgba(0,0,0,0)",
            "plot_bgcolor": "rgba(0,0,0,0)",
            "xaxis": {"title": "Attempt", "gridcolor": "rgba(255,255,255,.08)", "tickfont": {"color": "#e5e7eb"}, "titlefont": {"color": "#e5e7eb"}, "tickmode": "linear", "dtick": 1},
            "yaxis": {"title": y_title, "range": y_range, "gridcolor": "rgba(255,255,255,.08)", "tickfont": {"color": "#e5e7eb"}, "titlefont": {"color": "#e5e7eb"}},
            "font": {"color": "#e5e7eb"},
            "margin": {"l": 50, "r": 20, "t": 10, "b": 40},
        },
    }
    return fig


def parse_correct_indices(options: List[str], correct_cell: str) -> List[int]:
    """Resolve correct answers from a cell against options. Supports text, letters (A,B,...) or 1-based indices.
    Returns a sorted list of unique indices.
    """
    if correct_cell is None:
        return []
    tokens = detect_token_list(str(correct_cell))
    if not tokens:
        return []
    norm_map = {normalize_token(opt): i for i, opt in enumerate(options)}

    indices: List[int] = []
    seen = set()
    for t in tokens:
        tn = normalize_token(t)
        idx: int | None = None
        # Try exact text match
        if tn in norm_map:
            idx = norm_map[tn]
        else:
            # Try A/B/C... letters
            if len(tn) == 1 and tn.isalpha():
                idx = ord(tn.upper()) - ord("A")
            else:
                # Try numeric indices. Support both 1-based (common in sheets) and 0-based (robustness).
                try:
                    num = int(tn)
                    # Prefer 1-based if in range; otherwise accept 0-based if valid
                    if 1 <= num <= len(options):
                        idx = num - 1
                    elif 0 <= num < len(options):
                        idx = num
                    else:
                        idx = None
                except Exception:
                    idx = None
        if idx is not None and 0 <= idx < len(options):
            if idx not in seen:
                indices.append(idx)
                seen.add(idx)
    indices.sort()
    return indices


def parse_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    """Parse a dataframe with first three columns: question, options, correct."""
    if df.shape[1] < 3:
        raise ValueError("Expected at least 3 columns: Question, Options, Correct")

    # Normalize column names and take first 3 or 4 columns (Selected is optional 4th)
    use_cols = 4 if df.shape[1] >= 4 else 3
    df = df.iloc[:, :use_cols].copy()
    if use_cols == 3:
        df.columns = ["Question", "Options", "Correct"]
    else:
        df.columns = ["Question", "Options", "Correct", "Selected"]

    questions: List[Dict[str, Any]] = []
    prefill_selected: List[List[int]] = []
    for _, row in df.iterrows():
        q_raw = str(row["Question"]).strip()
        raw_opts_cell = row["Options"]
        opts = detect_token_list(raw_opts_cell)  # split into list
        if not q_raw:
            # skip rows without a question
            continue
        if not opts:
            # Fallback: if there is some text in the Options cell but our splitter failed, keep it as a single option
            if str(raw_opts_cell).strip():
                opts = [str(raw_opts_cell).strip()]
            else:
                # truly empty options -> skip
                continue
        correct_idx = parse_correct_indices(opts, row["Correct"])  # list of ints
        questions.append({
            "q": q_raw,
            "options": opts,
            "correct": correct_idx,
        })
        # Optional prefill of selected answers from file (if present)
        if "Selected" in df.columns:
            sel = parse_correct_indices(opts, row.get("Selected"))
        else:
            sel = []
        prefill_selected.append(sel)

    if not questions:
        raise ValueError("No valid questions found. Ensure columns are filled and options are separated by '|' or newlines.")

    # Create a deterministic hash for dataset identity
    h = hashlib.sha256(json.dumps(questions, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    payload = {
        "title": "Uploaded Quiz",
        "hash": h,
        "questions": questions,
    }
    if any(len(s) > 0 for s in prefill_selected):
        payload["prefill_selected"] = prefill_selected
    return payload


 


# Default empty state for a dataset
def default_state(dataset_hash: str) -> Dict[str, Any]:
    return {
        "dataset_hash": dataset_hash,
        "current": 0,
        "answers": {},
        "review": False,
        "created_at": datetime.utcnow().isoformat() + "Z",
    }


def dataset_from_questions(questions: List[Dict[str, Any]], title: str | None = None) -> Dict[str, Any]:
    payload = {
        "title": title or "Quiz",
        "questions": questions,
    }
    h = hashlib.sha256(json.dumps(questions, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    payload["hash"] = h
    return payload


def blank_question() -> Dict[str, Any]:
    return {"q": "", "options": ["Option 1", "Option 2"], "correct": [0]}


app = Dash(
    __name__,
    external_stylesheets=EXTERNAL_STYLESHEETS,
    suppress_callback_exceptions=True,  # allow callbacks to target components rendered by callbacks
)
server = app.server  # for deployments

app.title = "Quiz App"

# Inject our custom CSS into the HTML <head> via index_string to avoid using dcc.Markdown (which loads async-markdown)
CUSTOM_CSS = r"""
<style>
:root {
    --brand-accent: #6f9eff;
    --brand-glow: rgba(111, 158, 255, .35);
}
body {
    background: radial-gradient(1200px 600px at 20% -10%, rgba(111,158,255,.08), transparent),
                radial-gradient(900px 500px at 100% 10%, rgba(118,214,255,.06), transparent),
                #1B1F24 !important;
}
.card {
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,.08);
    background: rgba(26, 28, 33, .85);
    box-shadow: 0 6px 24px rgba(0,0,0,.35);
    transition: box-shadow .25s ease, border-color .25s ease;
}
.card-header {
    border-bottom: 1px solid rgba(255,255,255,.06);
    background: linear-gradient(180deg, rgba(255,255,255,.03), rgba(255,255,255,.00));
}
.btn {
    border-radius: 10px;
    transition: transform .08s ease, box-shadow .25s ease, background-color .2s ease, border-color .2s ease;
}
.btn:hover {
    transform: translateY(-1px);
    box-shadow: 0 8px 22px rgba(0,0,0,.35), 0 0 0 2px var(--brand-glow) inset;
    border-color: var(--brand-accent) !important;
}
.btn:active {
    transform: translateY(0);
    box-shadow: 0 2px 8px rgba(0,0,0,.4) inset;
}
#uploader {
    transition: transform .15s ease, box-shadow .25s ease, border-color .25s ease, background-color .2s ease;
    background-color: rgba(255,255,255,.02);
}
#uploader:hover {
    border-color: var(--brand-accent) !important;
    box-shadow: 0 0 0 3px var(--brand-glow) inset, 0 8px 24px rgba(0,0,0,.35);
    transform: scale(1.01);
    background-color: rgba(255,255,255,.04);
}
.progress {
    border-radius: 999px;
    background-color: rgba(255,255,255,.06);
    overflow: hidden;
}
.progress-bar {
    background-image: linear-gradient(90deg, rgba(111,158,255,.9), rgba(111,158,255,.6));
    transition: width .35s ease;
}
.form-control, .Select-control {
    border-radius: 10px !important;
    border-color: rgba(255,255,255,.12) !important;
    background-color: rgba(255,255,255,.02) !important;
    color: #e5e7eb !important; /* light text on dark bg */
}
.form-control::placeholder { color: rgba(229,231,235,.6) !important; }
.form-control:disabled { background-color: rgba(255,255,255,.06) !important; color: rgba(229,231,235,.55) !important; }
/* If an input uses an explicit light background, switch to dark text for readability */
.form-control.bg-white, .form-control.bg-light { color: #111827 !important; }
.Select-placeholder { color: rgba(229,231,235,.6) !important; }
.Select-value-label { color: #e5e7eb !important; }
.Select-menu-outer { background-color: rgba(17,24,39,.98) !important; border-color: rgba(255,255,255,.12) !important; }
.Select-option { color: #e5e7eb !important; background-color: transparent !important; }
.Select-option.is-focused { background-color: rgba(111,158,255,.18) !important; color: #fff !important; }
.Select-option.is-selected { background-color: rgba(111,158,255,.28) !important; color: #fff !important; }
.form-control:focus, .Select.is-focused > .Select-control {
    box-shadow: 0 0 0 3px var(--brand-glow) !important;
    border-color: var(--brand-accent) !important;
}
.form-check-input {
    border-radius: 6px;
    accent-color: var(--brand-accent);
    box-shadow: none !important;
}
/* Tabs */
.custom-tabs { display: flex; gap: 8px; flex-wrap: wrap; }
.custom-tabs .custom-tab { 
    background-color: rgba(255,255,255,.06); 
    color: #e5e7eb !important; 
    border-radius: 8px; 
    padding: 6px 12px; 
    border: 1px solid rgba(255,255,255,.12);
}
.custom-tabs .custom-tab:hover { 
    background-color: rgba(255,255,255,.1);
    border-color: var(--brand-accent);
    box-shadow: 0 0 0 2px var(--brand-glow) inset;
}
.custom-tabs .custom-tab--selected { 
    background-color: rgba(111,158,255,.18); 
    color: #ffffff !important; 
    border-color: rgba(111,158,255,.5);
    box-shadow: 0 0 0 2px var(--brand-glow) inset;
}
</style>
"""

app.index_string = (
    "<!DOCTYPE html>\n"
    "<html>\n"
    "    <head>\n"
    "        {%metas%}\n"
    "        <title>{%title%}</title>\n"
    "        {%favicon%}\n"
    "        {%css%}\n"
    f"        {CUSTOM_CSS}\n"
    "    </head>\n"
    "    <body>\n"
    "        {%app_entry%}\n"
    "        <footer>\n"
    "            {%config%}\n"
    "            {%scripts%}\n"
    "            {%renderer%}\n"
    "        </footer>\n"
    "    </body>\n"
    "</html>\n"
)

# ---------- Data directory helpers ----------

CSV_DIR = os.path.join(os.path.dirname(__file__), "data")
TRASH_DIR = os.path.join(CSV_DIR, "trash")
PROGRESS_SUFFIX = ".progress.csv"


def ensure_csv_dir() -> str:
    """Ensure app data directory exists; migrate legacy 'csv' folder contents into 'data' if present."""
    base_dir = os.path.dirname(__file__)
    data_dir = CSV_DIR
    legacy_dir = os.path.join(base_dir, "csv")
    try:
        os.makedirs(data_dir, exist_ok=True)
        # Migrate legacy folder contents once
        if os.path.isdir(legacy_dir):
            for name in os.listdir(legacy_dir):
                src = os.path.join(legacy_dir, name)
                dst = os.path.join(data_dir, name)
                try:
                    if not os.path.exists(dst):
                        os.replace(src, dst)
                except Exception:
                    pass
            # try to remove legacy dir if empty (ignore on failure)
            try:
                if not os.listdir(legacy_dir):
                    os.rmdir(legacy_dir)
            except Exception:
                pass
    except Exception:
        pass
    return data_dir


def ensure_trash_dir() -> str:
    ensure_csv_dir()
    try:
        os.makedirs(TRASH_DIR, exist_ok=True)
    except Exception:
        pass
    return TRASH_DIR


def slugify_filename(name: str) -> str:
    base = (name or "quiz").strip().lower()
    # replace spaces and invalid chars
    safe = "".join(ch if ch.isalnum() or ch in ("-", "_") else "-" for ch in base)
    while "--" in safe:
        safe = safe.replace("--", "-")
    safe = safe.strip("-_") or "quiz"
    return f"{safe}.json"


def _unique_path_in_dir(directory: str, basename: str) -> str:
    """Return a unique path in directory for given basename (with extension)."""
    root, ext = os.path.splitext(basename)
    candidate = os.path.join(directory, basename)
    if not os.path.exists(candidate):
        return candidate
    i = 2
    while True:
        cand = os.path.join(directory, f"{root}-{i}{ext}")
        if not os.path.exists(cand):
            return cand
        i += 1


def dataset_to_csv(dataset: Dict[str, Any], file_path: str, selected_map: Dict[int, List[int]] | None = None) -> None:
    rows = []
    for idx, q in enumerate(dataset.get("questions", [])):
        options = q.get("options", []) or []
        correct_idx = q.get("correct", []) or []
        opts_joined = "|".join(options)
        # Save correct answers as the option texts joined by '|'
        correct_texts = [options[i] for i in correct_idx if isinstance(i, int) and 0 <= i < len(options)]
        correct_joined = "|".join(correct_texts)
        row = {
            "Question": q.get("q", ""),
            "Options": opts_joined,
            "Correct": correct_joined,
        }
        if selected_map is not None:
            sel = selected_map.get(idx, []) if isinstance(selected_map, dict) else []
            # Store as pipe-joined option texts (more stable across shuffles and edits)
            try:
                sel_texts = [options[i] for i in sel if isinstance(i, int) and 0 <= i < len(options)]
            except Exception:
                sel_texts = []
            row["Selected"] = "|".join(sel_texts)
        rows.append(row)
    columns = ["Question", "Options", "Correct"] + (["Selected"] if selected_map is not None else [])
    df = pd.DataFrame(rows, columns=columns)
    df.to_csv(file_path, index=False, sep=";", encoding="utf-8-sig")


def dataset_to_json(dataset: Dict[str, Any], file_path: str, selected_map: Dict[int, List[int]] | None = None) -> None:
    """Save dataset to a JSON file, keeping selections inside each question as 'selected' indices."""
    data = {
        "title": dataset.get("title") or "Quiz",
        "hash": dataset.get("hash"),
        "questions": [],
    }
    qs = dataset.get("questions", []) or []
    for idx, q in enumerate(qs):
        rec = {
            "q": q.get("q", ""),
            "options": list(q.get("options", []) or []),
            "correct": list(q.get("correct", []) or []),
        }
        if isinstance(selected_map, dict):
            sel = selected_map.get(idx, []) or []
            rec["selected"] = list(sel)
        data["questions"].append(rec)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dataset_save_autodetect(dataset: Dict[str, Any], file_path: str, selected_map: Dict[int, List[int]] | None = None) -> None:
    """Save dataset to CSV or JSON depending on file extension."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".json":
        dataset_to_json(dataset, file_path, selected_map=selected_map)
    else:
        dataset_to_csv(dataset, file_path, selected_map=selected_map)


def _progress_path_for_selected(selected_id: str | None) -> str | None:
    if not (isinstance(selected_id, str) and selected_id.startswith("file:")):
        return None
    base_path = selected_id.split(":", 1)[1]
    root, _ = os.path.splitext(base_path)
    return f"{root}{PROGRESS_SUFFIX}"


def save_progress_to_csv(state: Dict[str, Any] | None, quiz_data: Dict[str, Any] | None, selected_id: str | None) -> None:
    try:
        if not state or not quiz_data:
            return
        path = _progress_path_for_selected(selected_id)
        if not path:
                    n = len(quiz_data.get("questions", []) or [])
        n = len(quiz_data.get("questions", []))
        rows: List[Dict[str, Any]] = []
        answers = state.get("answers", {}) if isinstance(state, dict) else {}
        for i in range(n):
            sel = answers.get(str(i), {}).get("selected", [])
            # Store indices as pipe-joined numbers
            sel_str = "|".join(str(x) for x in sel)
            rows.append({"Index": i, "Selected": sel_str, "Current": state.get("current", 0) if i == 0 else ""})
        df = pd.DataFrame(rows, columns=["Index", "Selected", "Current"])
        df.to_csv(path, index=False, sep=";", encoding="utf-8-sig")
    except Exception:
        # Best-effort persistence; ignore failures
        pass


def load_progress_from_csv(selected_id: str | None, expected_n: int) -> Tuple[Dict[str, Any], int] | None:
    try:
        path = _progress_path_for_selected(selected_id)
        if not path or not os.path.exists(path):
            return None
        df = pd.read_csv(path, sep=";", engine="python")
        answers: Dict[str, Any] = {}
        current = 0
        for _, row in df.iterrows():
            i = int(row.get("Index")) if not pd.isna(row.get("Index")) else None
            if i is None:
                continue
            if 0 <= i < expected_n:
                sel_str = str(row.get("Selected") or "")
                sel_vals = [int(x) for x in sel_str.split("|") if str(x).strip().isdigit()]
                answers[str(i)] = {"selected": sel_vals}
                # grab Current from first row if present
                if i == 0:
                    try:
                        cur_val = row.get("Current")
                        if pd.isna(cur_val):
                            cur_val = 0
                        current = int(cur_val)
                    except Exception:
                        current = 0
        return {"answers": answers}, int(current)
    except Exception:
        return None


def delete_progress_file(selected_id: str | None) -> None:
    try:
        path = _progress_path_for_selected(selected_id)
        if path and os.path.exists(path):
            os.remove(path)
    except Exception:
        pass


def _parse_json_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    questions = []
    prefill: List[List[int]] = []
    for q in raw.get("questions", []) or []:
        qtext = str(q.get("q", ""))
        opts = list(q.get("options", []) or [])
        corr = list(q.get("correct", []) or [])
        questions.append({"q": qtext, "options": opts, "correct": corr})
        sel = list(q.get("selected", []) or [])
        prefill.append(sel)
    payload = dataset_from_questions(questions, raw.get("title") or "Quiz")
    if any(len(s) > 0 for s in prefill):
        payload["prefill_selected"] = prefill
    payload["source_path"] = path
    return payload


def load_all_csv_datasets() -> List[Dict[str, Any]]:
    ensure_csv_dir()
    datasets: List[Dict[str, Any]] = []
    try:
        for name in sorted(os.listdir(CSV_DIR)):
            lower = name.lower()
            path = os.path.join(CSV_DIR, name)
            if lower.endswith(".json"):
                try:
                    data = _parse_json_dataset(path)
                    data["title"] = data.get("title") or os.path.splitext(name)[0]
                    data["source_path"] = path
                    datasets.append(data)
                except Exception:
                    continue
            elif lower.endswith(".csv"):
                # Migrate CSV -> JSON once, then move CSV to trash
                try:
                    df = pd.read_csv(path, sep=";", engine="python")
                    data = parse_dataframe(df)
                    title = data.get("title") or os.path.splitext(name)[0]
                    base = os.path.splitext(slugify_filename(title or "quiz"))[0]
                    target_json = os.path.join(CSV_DIR, f"{base}.json")
                    if os.path.exists(target_json):
                        # choose a unique JSON name
                        target_json = _unique_csv_path_for_title(title)
                    # Write JSON with any prefilled selections
                    sel_map: Dict[int, List[int]] = {}
                    pre = data.get("prefill_selected") or []
                    for i, sel in enumerate(pre):
                        sel_map[i] = list(sel or [])
                    dataset_to_json(data, target_json, selected_map=sel_map)
                    # Move CSV to trash
                    try:
                        ensure_trash_dir()
                        os.replace(path, os.path.join(TRASH_DIR, name))
                    except Exception:
                        pass
                    # Load the new JSON into list
                    data2 = _parse_json_dataset(target_json)
                    data2["title"] = data2.get("title") or os.path.splitext(os.path.basename(target_json))[0]
                    data2["source_path"] = target_json
                    datasets.append(data2)
                except Exception:
                    # If migration fails, skip this file
                    continue
    except Exception:
        pass
    return datasets

app.layout = html.Div(
    className="container py-4",
    children=[
        # Custom CSS is injected via app.index_string

        html.H2("Quiz App", className="mb-3"),
        html.P(
            "Datasets (JSON) laddas automatiskt från mappen 'csv'. JSON-formatet är robust och behåller kod och radbrytningar.",
            className="text-muted",
        ),
        # Auto-loaded datasets from folder (JSON only)
        html.Div(
            className="card mb-3",
            children=[
                html.Div(className="card-body", children=[
                    html.Div(className="row g-3", children=[
                        html.Div(className="col-md-8", children=[
                            html.Label("Dataset", className="form-label"),
                            dcc.Dropdown(id="dataset_select", options=[], value=None, placeholder="Select dataset", clearable=False),
                        ]),
                        html.Div(className="col-md-4 d-flex align-items-end justify-content-end gap-2", children=[
                            html.Button("Reload folder", id="btn_reload_csv", className="btn btn-outline-secondary btn-sm"),
                            html.Button("Delete file", id="btn_delete_csv", className="btn btn-outline-danger btn-sm"),
                        ]),
                    ]),
                    html.Small(f"Data folder: {CSV_DIR}", className="text-muted"),
                    html.Div(id="csv_load_status", className="mt-2 text-muted"),
                ]),
            ],
        ),
        dcc.ConfirmDialog(id="confirm_delete", message="Are you sure you want to delete this file?"),

        # Mode tabs
        dcc.Tabs(
            id="mode_tabs",
            value="quiz",
            children=[
                dcc.Tab(
                    label="Quiz",
                    value="quiz",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                    style={
                        "backgroundColor": "rgba(255,255,255,.06)",
                        "color": "#e5e7eb",
                        "border": "1px solid rgba(255,255,255,.12)",
                        "borderRadius": "8px",
                        "padding": "6px 12px",
                        "marginRight": "8px",
                    },
                    selected_style={
                        "backgroundColor": "rgba(111,158,255,.18)",
                        "color": "#ffffff",
                        "border": "1px solid rgba(111,158,255,.5)",
                        "borderRadius": "8px",
                        "padding": "6px 12px",
                        "marginRight": "8px",
                    },
                ),
                dcc.Tab(
                    label="Editor",
                    value="editor",
                    className="custom-tab",
                    selected_className="custom-tab--selected",
                    style={
                        "backgroundColor": "rgba(255,255,255,.06)",
                        "color": "#e5e7eb",
                        "border": "1px solid rgba(255,255,255,.12)",
                        "borderRadius": "8px",
                        "padding": "6px 12px",
                        "marginRight": "8px",
                    },
                    selected_style={
                        "backgroundColor": "rgba(111,158,255,.18)",
                        "color": "#ffffff",
                        "border": "1px solid rgba(111,158,255,.5)",
                        "borderRadius": "8px",
                        "padding": "6px 12px",
                        "marginRight": "8px",
                    },
                ),
            ],
            className="mb-3 custom-tabs",
            style={"backgroundColor": "transparent"},
            colors={
                "border": "rgba(255,255,255,.12)",
                "primary": "#6f9eff",
                "background": "#111827",
            },
        ),

        # Quiz panel
        html.Div(id="panel_quiz", children=[
            # Progress + controls
            html.Div(className="d-flex align-items-center gap-3 mb-2", children=[
                html.Div(id="progress_label", className="fw-bold"),
                html.Div(className="flex-grow-1", children=[
                    html.Div(className="progress", children=[
                        html.Div(id="progress_bar", className="progress-bar", role="progressbar", style={"width": "0%"}),
                    ]),
                ]),
            ]),

            # Question card
            html.Div(className="card mb-3", children=[
                html.Div(className="card-header d-flex justify-content-between align-items-center", children=[
                    html.Div(id="dataset_title", className="fw-semibold"),
                    html.Div(id="question_counter", className="text-muted"),
                ]),
                html.Div(className="card-body", children=[
                    html.Div(id="question_text", className="card-title"),
                    html.Div(id="answer_container", className="my-3"),
                    html.Div(id="feedback", className="mt-2"),
                ]),
                html.Div(className="card-footer d-flex gap-2", children=[
                    html.Button("Prev", id="btn_prev", className="btn btn-secondary"),
                    html.Button("Next", id="btn_next", className="btn btn-secondary"),
                    dcc.Dropdown(id="quiz_jump_select", options=[], value=None, placeholder="Gå till fråga...", clearable=False, style={"minWidth": "260px", "width": "420px"}),
                    dcc.Dropdown(
                        id="scoring_mode",
                        options=[
                            {"label": "Poäng: Helt rätt eller fel", "value": "aon"},
                            {"label": "Poäng: Delvis (utan avdrag)", "value": "partial"},
                            {"label": "Poäng: Delvis (med avdrag)", "value": "penalty"},
                        ],
                        value="aon",
                        clearable=False,
                        style={"minWidth": "260px", "width": "300px"},
                    ),
                    html.Button("Slumpa alternativ", id="btn_shuffle_options", className="btn btn-outline-info btn-sm"),
                    html.Button("Slumpa frågor", id="btn_shuffle_questions", className="btn btn-outline-info btn-sm"),
                    html.Div(className="flex-grow-1"),
                    html.Button("Restart", id="btn_reset", className="btn btn-outline-danger"),
                    html.Button("Submit", id="btn_submit", className="btn btn-primary"),
                ]),
            ]),
            # Progress over time plot
            html.Div(className="card", children=[
                html.Div(className="card-header d-flex align-items-center justify-content-between", children=[
                    html.Div("Din utveckling över tid", className="fw-semibold"),
                    dcc.Dropdown(
                        id="plot_metric",
                        options=[
                            {"label": "% korrekt", "value": "pct"},
                            {"label": "Poäng", "value": "points"},
                        ],
                        value="pct",
                        clearable=False,
                        style={"width": "160px"},
                    ),
                ]),
                html.Div(className="card-body", children=[
                    dcc.Graph(id="progress_plot", figure={"data": [], "layout": {"paper_bgcolor": "rgba(0,0,0,0)", "plot_bgcolor": "rgba(0,0,0,0)", "xaxis": {"title": "Attempt"}, "yaxis": {"title": "% korrekt", "range": [0, 100]}}})
                ])
            ], style={"marginTop": "12px"}),
        ]),

        # Editor panel
        html.Div(id="panel_editor", style={"display": "none"}, children=[
            html.Div(className="card", children=[
                html.Div(className="card-header d-flex gap-2 align-items-center", children=[
                    html.Div(className="me-2", children=[
                        html.Label("Title", className="form-label mb-0 me-2"),
                    ]),
                    dcc.Input(id="edit_title", type="text", placeholder="Quiz title", className="form-control", style={"maxWidth": "360px"}, value=""),
                    html.Div(className="flex-grow-1"),
                    html.Button("Create New Dataset", id="btn_new_dataset", className="btn btn-outline-warning btn-sm"),
                    html.Button("Export to Excel", id="btn_export_excel", className="btn btn-outline-info btn-sm"),
                ]),
                html.Div(className="card-body", children=[
                    html.Div(className="row g-3", children=[
                        html.Div(className="col-md-4", children=[
                            html.Label("Questions", className="form-label"),
                            dcc.Dropdown(id="edit_q_select", options=[], value=None, clearable=False),
                            html.Div(className="mt-2 d-flex gap-2", children=[
                                html.Button("Add Question", id="btn_add_question", className="btn btn-primary btn-sm"),
                                html.Button("Delete", id="btn_delete_question", className="btn btn-outline-danger btn-sm"),
                            ]),
                        ]),
                        html.Div(className="col-md-8", children=[
                            html.Label("Question Text", className="form-label"),
                            dcc.Textarea(id="edit_q_text", className="form-control", style={"minHeight": "100px"}, value=""),
                            html.Div(className="row g-2 mt-3", children=[
                                html.Div(className="col-12", children=[
                                    html.Label("Options (one per line)", className="form-label"),
                                    dcc.Textarea(id="edit_options_text", className="form-control", style={"minHeight": "120px"}, value=""),
                                ]),
                                html.Div(className="col-12", children=[
                                    html.Label("Correct choices", className="form-label"),
                                    dcc.Checklist(id="edit_correct_choices", options=[], value=[]),
                                ]),
                            ]),
                            # Autosave is enabled; no manual save buttons
                            html.Div(className="mt-3 text-muted", children=[
                                html.Small("All edits are saved automatically.")
                            ]),
                        ]),
                    ]),
                    html.Div(id="editor_status", className="mt-3"),
                ]),
            ]),
        ]),

        # Hidden stores
        dcc.Store(id="quiz_data", storage_type="local"),
        dcc.Store(id="quiz_state", storage_type="local"),  # persist across sessions in this browser
        dcc.Store(id="datasets_list"),
    dcc.Store(id="selected_dataset", storage_type="local"),
    dcc.Store(id="new_dataset_mode", data=False, storage_type="session"),
        dcc.Store(id="delete_target"),
        dcc.Store(id="quiz_stats", storage_type="local"),
        # Auto-clear small flash messages in feedback area
        dcc.Interval(id="feedback_auto_clear", interval=2500, n_intervals=0, disabled=True),
        # Download target for exports
        dcc.Download(id="download_export"),
    ],
)


@callback(
    Output("datasets_list", "data", allow_duplicate=True),
    Output("dataset_select", "options", allow_duplicate=True),
    Output("dataset_select", "value", allow_duplicate=True),
    Output("csv_load_status", "children", allow_duplicate=True),
    Output("quiz_data", "data", allow_duplicate=True),
    Output("selected_dataset", "data", allow_duplicate=True),
    Input("btn_reload_csv", "n_clicks"),
    State("selected_dataset", "data"),
    prevent_initial_call="initial_duplicate",
)
def scan_data_and_select(nc, selected_id):
    """Scan data folder, build options, choose selection (persisting last used), and provide quiz_data."""
    ensure_csv_dir()
    datasets = load_all_csv_datasets()
    entries = []
    for d in datasets:
        fid = f"file:{os.path.abspath(d.get('source_path'))}"
        title = d.get("title") or os.path.splitext(os.path.basename(d.get("source_path", "")))[0]
        entries.append({"id": fid, "title": title, "data": d})

    options = [{"label": e["title"], "value": e["id"]} for e in entries]

    # Determine selected id
    ids = [e["id"] for e in entries]
    if selected_id in ids:
        chosen_id = selected_id
    else:
        chosen_id = ids[0] if ids else None

    status = html.Small(f"Loaded {len(entries)} CSV file(s)", className="text-muted")
    if chosen_id is None:
        # Clear any persisted selection and dataset when folder is empty / no selection
        return entries, options, None, status, None, None
    chosen = next((e for e in entries if e["id"] == chosen_id), None)
    # Avoid re-writing quiz_data on reload if selection didn't change
    quiz_payload = no_update if (selected_id == chosen_id) else ((chosen or {}).get("data"))
    return entries, options, chosen_id, status, quiz_payload, chosen_id


@callback(
    Output("quiz_data", "data", allow_duplicate=True),
    Output("selected_dataset", "data", allow_duplicate=True),
    Output("csv_load_status", "children", allow_duplicate=True),
    Input("dataset_select", "value"),
    State("datasets_list", "data"),
    prevent_initial_call=True,
)
def on_select_dataset(selected_value, entries):
    """When user picks a dataset from dropdown, update quiz_data and persist selection."""
    if not entries or not selected_value:
        return None, None, html.Small("No dataset selected", className="text-muted")
    chosen = next((e for e in entries if e.get("id") == selected_value), None)
    if not chosen:
        return None, None, html.Small("No dataset selected", className="text-muted")
    status = html.Small(f"Selected: {chosen.get('title', 'dataset')}", className="text-muted")
    return chosen.get("data"), chosen.get("id"), status


# Removed manual Save CSV; autosave will handle persistence


@callback(
    Output("quiz_state", "data"),
    Input("quiz_data", "data"),
    State("quiz_state", "data"),
    State("selected_dataset", "data"),
)
def init_or_retain_state(quiz_data, state, selected_id):
    if not quiz_data:
        return no_update
    dataset_hash = quiz_data.get("hash")
    if state and state.get("dataset_hash") == dataset_hash:
        # keep existing
        return state
    # Prefill from dataset file if it contains a 'Selected' column
    n = len(quiz_data.get("questions", []))
    prefill = quiz_data.get("prefill_selected") or []
    if prefill:
        new_state = default_state(dataset_hash)
        answers: Dict[str, Any] = {}
        for i in range(min(n, len(prefill))):
            sel = prefill[i] or []
            # correctness can be derived for feedback display if needed
            corr = quiz_data.get("questions", [])[i].get("correct", [])
            answers[str(i)] = {"selected": sel, "correct": sorted(sel) == sorted(corr)}
        new_state["answers"] = answers
        new_state["current"] = min(max(0, new_state.get("current", 0)), max(0, n - 1))
        return new_state
    # else initialize new state for this dataset
    return default_state(dataset_hash)


def _unique_csv_path_for_title(title: str) -> str:
    ensure_csv_dir()
    base = os.path.splitext(slugify_filename(title or "quiz"))[0]
    candidate = os.path.join(CSV_DIR, f"{base}.json")
    if not os.path.exists(candidate):
        return candidate
    # add numeric suffix
    i = 2
    while True:
        cand = os.path.join(CSV_DIR, f"{base}-{i}.json")
        if not os.path.exists(cand):
            return cand
        i += 1


@callback(
    Output("datasets_list", "data", allow_duplicate=True),
    Output("dataset_select", "options", allow_duplicate=True),
    Output("dataset_select", "value", allow_duplicate=True),
    Output("csv_load_status", "children", allow_duplicate=True),
    Output("selected_dataset", "data", allow_duplicate=True),
    Output("editor_status", "children", allow_duplicate=True),
    Input("quiz_data", "data"),
    State("selected_dataset", "data"),
    State("quiz_state", "data"),
    prevent_initial_call=True,
)
def autosave_on_quiz_data(quiz_data, selected_id, state):
    """Auto-save current quiz_data to csv. Create or rename file based on title changes.
    Triggers whenever quiz_data changes (from editor edits/new/add/delete).
    """
    if not quiz_data:
        raise PreventUpdate
    # Only autosave when a concrete file dataset is selected
    if not (isinstance(selected_id, str) and selected_id.startswith("file:")):
        raise PreventUpdate
    ensure_csv_dir()
    title = (quiz_data.get("title") or "quiz").strip()
    base = os.path.splitext(slugify_filename(title))[0]
    desired_path = os.path.join(CSV_DIR, f"{base}.json")  # default to robust JSON

    current_path = selected_id.split(":", 1)[1]

    # If we have a current file, decide whether to migrate extension or rename by title
    target_path = desired_path
    renamed = False
    migrated = False
    if current_path and os.path.exists(current_path):
        current_base = os.path.splitext(os.path.basename(current_path))[0]
        desired_base = os.path.splitext(os.path.basename(desired_path))[0]
        if current_base == desired_base:
            # Same base title. If current is CSV, migrate to JSON by writing a new JSON side-by-side.
            if str(current_path).lower().endswith(".csv"):
                # choose a free JSON path
                cand = desired_path
                if os.path.exists(cand) and os.path.abspath(cand) != os.path.abspath(current_path):
                    cand = _unique_csv_path_for_title(title)
                target_path = cand
                migrated = True
            else:
                target_path = current_path
        else:
            # rename; if desired exists and is not current, choose a unique free name
            rename_target = desired_path
            if os.path.exists(desired_path) and os.path.abspath(desired_path) != os.path.abspath(current_path):
                rename_target = _unique_csv_path_for_title(title)
            try:
                os.replace(current_path, rename_target)
                target_path = rename_target
            except Exception:
                # fallback: keep current file
                target_path = current_path
            else:
                renamed = True

    # Write dataset to target_path
    try:
        # Preserve current selections from state when writing due to editor changes
        sel_map: Dict[int, List[int]] | None = None
        if isinstance(state, dict):
            sel_map = {}
            for k, v in (state.get("answers", {}) or {}).items():
                try:
                    i = int(k)
                except Exception:
                    continue
                sel_map[i] = list(v.get("selected", []))
            dataset_save_autodetect(quiz_data, target_path, selected_map=sel_map)
    except Exception:
        # if write fails, do not update lists
        raise PreventUpdate

    # Only reload dataset list if we actually renamed the file (title change)
    new_id = f"file:{os.path.abspath(target_path)}"
    editor_msg = html.Div([html.Span("Auto-saved ", className="text-success"), html.Code(os.path.basename(target_path))])
    if not renamed and not migrated:
        # Avoid reloading list to prevent transient parse races that could clear selection
        return no_update, no_update, no_update, no_update, no_update, editor_msg

    datasets = load_all_csv_datasets()
    entries = []
    for d in datasets:
        fid = f"file:{os.path.abspath(d.get('source_path'))}"
        title2 = d.get("title") or os.path.splitext(os.path.basename(d.get("source_path", "")))[0]
        entries.append({"id": fid, "title": title2, "data": d})
    options = [{"label": e["title"], "value": e["id"]} for e in entries]
    status = html.Small(f"Auto-saved to {os.path.basename(target_path)} • {len(entries)} file(s)", className="text-muted")
    return entries, options, new_id, status, new_id, editor_msg


def render_answer_inputs(options: List[str], multi: bool, selected: List[int] | None) -> html.Div:
    labels = [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)]
    items = [{"label": lbl, "value": i} for i, lbl in enumerate(labels)]
    # Always allow multi-select in the quiz regardless of number of correct answers
    control = dcc.Checklist(id="answer_input", options=items, value=selected or [], inputClassName="form-check-input")
    return html.Div(control, className="")


def compute_progress(state: Dict[str, Any], total: int) -> Tuple[int, int]:
    if not state:
        return 0, total
    # Count only questions that have a non-empty selection
    ans = state.get("answers", {}) or {}
    def _countable(v: Any) -> bool:
        return isinstance(v, dict) and isinstance(v.get("selected"), list) and len(v.get("selected")) > 0
    answered = sum(1 for v in ans.values() if _countable(v))
    return answered, total


def render_question_text(qtext: str) -> Any:
    """Render question text with support for newlines and simple code blocks.
    If the text contains triple backticks ```...```, segments inside are rendered as a monospace block.
    Otherwise, preserve newlines with pre-wrap.
    """
    if not qtext:
        return ""
    s = str(qtext).replace("\r", "")
    # Simple fence-based split
    if "```" in s:
        parts = s.split("```")
        children: List[Any] = []
        for i, part in enumerate(parts):
            if i % 2 == 0:
                # Normal text segment
                if part.strip():
                    children.append(html.Div(part, style={"whiteSpace": "pre-wrap"}))
            else:
                code = part.strip("\n")
                children.append(
                    html.Pre(
                        code,
                        style={
                            "whiteSpace": "pre-wrap",
                            "fontFamily": "SFMono-Regular,Consolas,'Liberation Mono',Menlo,monospace",
                            "backgroundColor": "rgba(255,255,255,.04)",
                            "border": "1px solid rgba(255,255,255,.12)",
                            "borderRadius": "8px",
                            "padding": "10px",
                            "marginTop": "8px",
                        },
                    )
                )
        return children if children else html.Div(s, style={"whiteSpace": "pre-wrap"})
    # No fences: preserve line breaks
    return html.Div(s, style={"whiteSpace": "pre-wrap"})


@callback(
    Output("panel_quiz", "style"),
    Output("panel_editor", "style"),
    Input("mode_tabs", "value"),
)
def toggle_panels(mode):
    if mode == "editor":
        return {"display": "none"}, {"display": "block"}
    return {"display": "block"}, {"display": "none"}


@callback(
    Output("confirm_delete", "displayed", allow_duplicate=True),
    Output("confirm_delete", "message", allow_duplicate=True),
    Output("delete_target", "data", allow_duplicate=True),
    Output("editor_status", "children", allow_duplicate=True),
    Input("btn_delete_csv", "n_clicks"),
    State("dataset_select", "value"),
    State("datasets_list", "data"),
    prevent_initial_call=True,
)
def prompt_delete_file(nc, selected_value, entries):
    if not nc:
        raise PreventUpdate
    # Only allow deleting actual files
    if not selected_value or not (isinstance(selected_value, str) and selected_value.startswith("file:")):
        return False, no_update, None, html.Div("Select a file to delete.", className="text-warning")
    # Resolve title for message
    title = None
    if entries:
        match = next((e for e in entries if e.get("id") == selected_value), None)
        if match:
            title = match.get("title")
    msg = f"Delete '{title or os.path.basename(selected_value.split(':',1)[1])}' permanently?"
    return True, msg, selected_value, no_update


@callback(
    Output("datasets_list", "data", allow_duplicate=True),
    Output("dataset_select", "options", allow_duplicate=True),
    Output("dataset_select", "value", allow_duplicate=True),
    Output("csv_load_status", "children", allow_duplicate=True),
    Output("selected_dataset", "data", allow_duplicate=True),
    Output("quiz_data", "data", allow_duplicate=True),
    Output("confirm_delete", "displayed", allow_duplicate=True),
    Output("editor_status", "children", allow_duplicate=True),
    Input("confirm_delete", "submit_n_clicks"),
    State("delete_target", "data"),
    prevent_initial_call=True,
)
def perform_delete_file(submit_clicks, target_value):
    if not submit_clicks:
        raise PreventUpdate
    # Guard
    if not target_value or not (isinstance(target_value, str) and target_value.startswith("file:")):
        raise PreventUpdate
    path = target_value.split(":", 1)[1]
    # Move to trash instead of permanent deletion
    try:
        ensure_trash_dir()
        base = os.path.basename(path)
        dest = _unique_path_in_dir(TRASH_DIR, base)
        if os.path.exists(path):
            os.replace(path, dest)
    except Exception as e:
        # Show error but still attempt to refresh list
        status = html.Small(f"Failed to move to trash: {e}", className="text-danger")
    # Reload folder
    datasets = load_all_csv_datasets()
    entries = []
    for d in datasets:
        fid = f"file:{os.path.abspath(d.get('source_path'))}"
        t2 = d.get("title") or os.path.splitext(os.path.basename(d.get("source_path", "")))[0]
        entries.append({"id": fid, "title": t2, "data": d})
    options = [{"label": e["title"], "value": e["id"]} for e in entries]
    status = html.Small(f"Loaded {len(entries)} CSV file(s)", className="text-muted")
    # Clear selection and quiz_data after deletion
    return entries, options, None, status, None, None, False, html.Div("File moved to trash.", className="text-success")


@callback(
    Output("dataset_title", "children"),
    Output("question_counter", "children"),
    Output("question_text", "children"),
    Output("answer_container", "children"),
    Output("feedback", "children"),
    Output("btn_prev", "disabled"),
    Output("btn_next", "disabled"),
    Output("btn_submit", "disabled"),
    Output("progress_label", "children"),
    Output("progress_bar", "style"),
    Input("quiz_data", "data"),
    Input("quiz_state", "data"),
)
def render_quiz(quiz_data, state):
    if not quiz_data:
        text = ""
        return text, text, text, html.Div(), html.Div(), True, True, True, "0/0", {"width": "0%"}

    qs = quiz_data["questions"]
    n = len(qs)
    idx = min(max(0, (state or {}).get("current", 0)), n - 1)
    q = qs[idx]

    # Determine if multi-select based on number of correct answers
    multi = len(q.get("correct", [])) > 1

    # Pre-select user's previous answer for this question if any
    selected = []
    # Never update the feedback area here; it's managed by Submit/shuffle/reset callbacks
    feedback = no_update
    if state:
        ans = state.get("answers", {}).get(str(idx)) or state.get("answers", {}).get(idx)
        if ans:
            selected = ans.get("selected", [])

    # Progress
    answered, total = compute_progress(state or {}, n)
    pct = int(round((answered / total) * 100)) if total else 0

    # Controls state
    disable_prev = idx <= 0
    disable_next = idx >= (n - 1)
    # Disable submit when already answered correctly? allow resubmit? We'll allow resubmit.
    disable_submit = False

    return (
        quiz_data.get("title", "Quiz"),
        f"Question {idx + 1} / {n}",
        render_question_text(q.get("q", "")),
        render_answer_inputs(q.get("options", []), multi, selected),
        feedback,
        disable_prev,
        disable_next,
        disable_submit,
        f"{answered}/{n} answered",
        {"width": f"{pct}%"},
    )


def evaluate_answer(selected: List[int] | int | None, correct: List[int]) -> Tuple[List[int], bool]:
    if selected is None:
        return [], False
    if isinstance(selected, int):
        chosen = [selected]
    else:
        chosen = sorted(list(set(selected)))
    return chosen, sorted(chosen) == sorted(correct)


def score_answer(chosen: List[int], correct: List[int], mode: str) -> Tuple[float, str]:
    """Return per-question score in [0,1] and a short message.
    Modes:
    - aon: 1.0 only if exact match, else 0.0
    - partial: TP/|Correct| (no penalty)
    - penalty: max(0, TP - FP)/|Correct|
    """
    cset = set(correct or [])
    sset = set(chosen or [])
    tp = len(cset & sset)
    fp = len(sset - cset)
    denom = max(1, len(cset))
    if mode == "partial":
        sc = tp / denom
    elif mode == "penalty":
        sc = max(0.0, (tp - fp) / denom)
    else:  # aon
        sc = 1.0 if sset == cset else 0.0
    # Build a compact message
    if sc >= 1.0:
        msg = "Rätt!"
    elif mode == "aon":
        msg = "Fel."
    else:
        msg = f"Delvis rätt: {tp}/{len(cset)}"
    return float(round(sc, 4)), msg


def render_attempt_summary(quiz_data: Dict[str, Any], state: Dict[str, Any], mode: str, show_save: bool = True, saved: bool = False) -> html.Div:
    qs = quiz_data.get("questions", [])
    answers = (state or {}).get("answers", {}) or {}
    rows: List[Any] = []
    total = len(qs)
    sum_score = 0.0
    exact_correct = 0
    for i, q in enumerate(qs):
        rec = answers.get(str(i), {}) or {}
        chosen = rec.get("selected", []) or []
        c_idx = q.get("correct", []) or []
        sc, _ = score_answer(chosen, c_idx, mode)
        sum_score += sc
        exact = (set(chosen) == set(c_idx))
        exact_correct += 1 if exact else 0
        # Labels (letters) and texts
        def fmt_labels(idxs: List[int]) -> str:
            return ", ".join([f"{chr(ord('A')+j)}" for j in idxs]) if idxs else "—"
        chosen_lbl = fmt_labels(chosen)
        correct_lbl = fmt_labels(c_idx)
        opts = q.get("options", []) or []
        def fmt_texts(idxs: List[int]) -> str:
            parts = []
            for j in idxs:
                if isinstance(j, int) and 0 <= j < len(opts):
                    parts.append(f"{chr(ord('A')+j)}. {opts[j]}")
            return " | ".join(parts) if parts else "—"
        correct_txt = fmt_texts(c_idx)
        chosen_txt = fmt_texts(chosen)
        status_badge = html.Span("Rätt" if exact else ("Delvis" if sc > 0 else "Fel"),
                                 className=f"badge {'bg-success' if exact else ('bg-info' if sc>0 else 'bg-warning')} ms-2")
        rows.append(
            html.Li([
                html.Span(f"Q{i+1}"),
                status_badge,
                html.Div(q.get("q", ""), className="small text-muted", style={"whiteSpace": "pre-wrap"}),
                html.Div([
                    html.Small(f"Dina val: {chosen_lbl}"),
                    html.Small(f" • Rätt: {correct_lbl}", className="ms-2"),
                    html.Div(html.Small(f"Rätt svar (text): {correct_txt}"), className="mt-1"),
                    html.Div(html.Small(f"Dina val (text): {chosen_txt}"), className="mt-1"),
                    html.Small(f" • Poäng: {int(round(sc*100))}%", className="ms-2" if mode!='aon' else {"display":"none"}),
                ], className="")
            ], className="mb-2")
        )
    pct = round((sum_score/total)*100, 2) if total else 0.0
    header = html.Div([
        html.Div("Resultat", className="fw-semibold mb-1"),
        html.Small(
            f"Totalt: {int(round(pct))}%  (läge: {'Helt rätt/fel' if mode=='aon' else ('Delvis utan avdrag' if mode=='partial' else 'Delvis med avdrag')})",
            className="text-muted"
        )
    ])
    controls: List[Any] = []
    if show_save and not saved:
        controls.append(html.Button("Lägg till i grafen", id="btn_save_to_plot", className="btn btn-outline-info btn-sm me-2 mt-2"))
    if saved:
        controls.append(html.Span("Tillagd i grafen", className="badge bg-info mt-2 me-2"))
    # New: allow closing the review without restarting
    controls.append(html.Button("Stäng resultat", id="btn_close_review", className="btn btn-outline-secondary btn-sm mt-2 me-2"))
    controls.append(html.Button("Starta ny runda", id="btn_start_new_run", className="btn btn-primary btn-sm mt-2"))
    return html.Div([header, html.Ul(rows, className="mt-2"), html.Div(controls)], className="alert alert-secondary")


## merged per-question submit into the single Submit callback below


@callback(
    Output("quiz_state", "data", allow_duplicate=True),
    Output("quiz_stats", "data", allow_duplicate=True),
    Output("feedback", "children", allow_duplicate=True),
    Output("quiz_data", "data", allow_duplicate=True),
    Input("btn_submit", "n_clicks"),
    State("quiz_state", "data"),
    State("quiz_data", "data"),
    State("answer_input", "value"),
    State("selected_dataset", "data"),
    State("quiz_stats", "data"),
    State("scoring_mode", "value"),
    prevent_initial_call=True,
)
def on_submit(n_clicks, state, quiz_data, selected, selected_id, stats, scoring_mode):
    if not quiz_data or not state:
        return no_update, no_update, no_update, no_update
    idx = int(state.get("current", 0))
    q = quiz_data["questions"][idx]
    chosen, is_correct = evaluate_answer(selected, q.get("correct", []))

    new_state = dict(state)
    answers = dict(new_state.get("answers", {}))
    # Compute and store per-question score per mode
    mode = scoring_mode or "aon"
    qscore, short_msg = score_answer(chosen, q.get("correct", []), mode)
    answers[str(idx)] = {"selected": chosen, "correct": is_correct, "score": qscore}
    new_state["answers"] = answers

    # Build feedback depending on mode and correctness
    if is_correct or qscore >= 1.0:
        fb = html.Div("Rätt!", className="alert alert-success")
    else:
        corr = q.get("correct", [])
        labels = ", ".join([f"{chr(ord('A') + i)}" for i in corr])
        items = [html.Li(f"{chr(ord('A') + i)}. {q.get('options', [])[i]}") for i in corr if 0 <= i < len(q.get('options', []))]
        prefix = short_msg if (mode != "aon") else "Fel."
        fb = html.Div([
            html.Div(prefix, className="fw-semibold mb-1"),
            html.Small(f"Rätt svar: {labels}", className="text-muted"),
            html.Ul(items, className="mt-2"),
        ], className="alert alert-warning")

    # Finish the attempt on submit: grade all questions based on current selections (unanswered -> 0)
    n = len(quiz_data.get("questions", []))
    total_score = 0.0
    exact_correct = 0
    full_answers: Dict[str, Any] = {}
    for i in range(n):
        rec = (new_state.get("answers", {}) or {}).get(str(i), {}) or {}
        ch_i = rec.get("selected", []) or []
        corr_i = quiz_data.get("questions", [])[i].get("correct", [])
        sc_i, _ = score_answer(ch_i, corr_i, mode)
        total_score += sc_i
        exact_correct += 1 if set(ch_i) == set(corr_i) else 0
        full_answers[str(i)] = {"selected": ch_i, "correct": set(ch_i) == set(corr_i), "score": sc_i}
    new_state["answers"] = full_answers
    new_state["review"] = True
    if mode == "aon":
        pct = round((exact_correct / n) * 100, 2) if n else 0
        points = float(exact_correct)
    else:
        pct = round((total_score / n) * 100, 2) if n else 0
        points = float(round(total_score, 4))
    # Show review with option to add to plot; do not log automatically
    fb = render_attempt_summary(quiz_data, new_state, mode, show_save=True, saved=False)
    return new_state, no_update, fb, no_update

def _summarize_question_label(text: str, idx: int) -> str:
    s = str(text or "").replace("\r", "").replace("\n", " ").strip()
    s = s[:60]
    return f"{idx+1}. {s}"


@callback(
    Output("quiz_jump_select", "options", allow_duplicate=True),
    Output("quiz_jump_select", "value", allow_duplicate=True),
    Input("quiz_data", "data"),
    Input("quiz_state", "data"),
    prevent_initial_call=True,
)
def update_quiz_jump_options(quiz_data, state):
    if not quiz_data:
        return [], None
    qs = quiz_data.get("questions", [])
    options = []
    for i, q in enumerate(qs):
        full = str(q.get("q", "") or "").replace("\r", "").replace("\n", " ").strip()
        preview = full[:40]
        label = html.Div(f"Q{i+1}: {preview}", title=full)
        options.append({"label": label, "value": i})
    cur = min(max(0, int((state or {}).get("current", 0))), max(0, len(qs) - 1)) if qs else None
    return options, cur


@callback(
    Output("quiz_data", "data", allow_duplicate=True),
    Output("quiz_state", "data", allow_duplicate=True),
    Output("feedback", "children", allow_duplicate=True),
    Output("feedback_auto_clear", "disabled", allow_duplicate=True),
    Input("btn_shuffle_options", "n_clicks"),
    State("quiz_data", "data"),
    State("quiz_state", "data"),
    State("selected_dataset", "data"),
    prevent_initial_call=True,
)
def on_shuffle_options(nc, quiz_data, state, selected_id):
    if not nc or not quiz_data:
        raise PreventUpdate
    qs = quiz_data.get("questions", [])
    # Shuffle options within each question, keep question order
    opt_maps: Dict[int, Dict[int, int]] = {}
    new_questions: List[Dict[str, Any]] = []
    for i, q in enumerate(qs):
        opts = list(q.get("options", []))
        order = list(range(len(opts)))
        random.shuffle(order)
        new_opts = [opts[k] for k in order]
        idx_map = {old: new for new, old in enumerate(order)}
        opt_maps[i] = idx_map
        corr_set = set(q.get("correct", []) or [])
        new_corr = [idx_map[c] for c in corr_set if c in idx_map]
        new_questions.append({"q": q.get("q", ""), "options": new_opts, "correct": new_corr})

    # Remap answers only within questions
    answers = (state or {}).get("answers", {}) if state else {}
    new_answers: Dict[str, Any] = {}
    for k, rec in (answers or {}).items():
        try:
            qi = int(k)
        except Exception:
            continue
        if qi < 0 or qi >= len(new_questions):
            continue
        sel = list(rec.get("selected", [])) if isinstance(rec, dict) else []
        idx_map = opt_maps.get(qi, {})
        mapped_sel = [idx_map[s] for s in sel if s in idx_map]
        # Preserve prior correctness flag; do not auto-grade on shuffle
        prev_flag = rec.get("correct") if isinstance(rec, dict) else None
        new_answers[str(qi)] = {"selected": mapped_sel, "correct": prev_flag}

    new_quiz_data = dict(quiz_data)
    new_quiz_data["questions"] = new_questions

    # Persist to CSV with mapped selections
    if isinstance(selected_id, str) and selected_id.startswith("file:"):
        current_path = selected_id.split(":", 1)[1]
        try:
            sel_map: Dict[int, List[int]] = {}
            for sk, rv in new_answers.items():
                try:
                    i = int(sk)
                except Exception:
                    continue
                sel_map[i] = list(rv.get("selected", []))
            dataset_save_autodetect(new_quiz_data, current_path, selected_map=sel_map)
        except Exception:
            pass

    new_state = dict(state or {})
    new_state["answers"] = new_answers
    new_state["review"] = False
    msg = html.Div("Alternativ slumpade (svar bevarade).", className="alert alert-info")
    # Enable auto-clear timer
    return new_quiz_data, new_state, msg, False

@callback(
    Output("quiz_data", "data", allow_duplicate=True),
    Output("quiz_state", "data", allow_duplicate=True),
    Output("feedback", "children", allow_duplicate=True),
    Output("feedback_auto_clear", "disabled", allow_duplicate=True),
    Input("btn_shuffle_questions", "n_clicks"),
    State("quiz_data", "data"),
    State("quiz_state", "data"),
    State("selected_dataset", "data"),
    prevent_initial_call=True,
)
def on_shuffle_questions(nc, quiz_data, state, selected_id):
    if not nc or not quiz_data:
        raise PreventUpdate
    qs = quiz_data.get("questions", [])
    n = len(qs)
    # Shuffle question order only
    q_order = list(range(n))
    random.shuffle(q_order)
    inv_q_order = {old_i: new_i for new_i, old_i in enumerate(q_order)}
    new_questions = [qs[old_i] for old_i in q_order]

    # Remap answers to new question indices (options unchanged)
    answers = (state or {}).get("answers", {}) if state else {}
    new_answers: Dict[str, Any] = {}
    for k, rec in (answers or {}).items():
        try:
            old_q = int(k)
        except Exception:
            continue
        if old_q < 0 or old_q >= n:
            continue
        new_q = inv_q_order.get(old_q)
        if new_q is None:
            continue
        sel = list(rec.get("selected", [])) if isinstance(rec, dict) else []
        # Preserve prior correctness flag; do not auto-grade on shuffle
        prev_flag = rec.get("correct") if isinstance(rec, dict) else None
        new_answers[str(new_q)] = {"selected": sel, "correct": prev_flag}

    # Map current pointer
    old_current = (state or {}).get("current", 0) if state else 0
    new_current = inv_q_order.get(int(old_current), 0)
    if new_current is None:
        new_current = 0

    new_quiz_data = dict(quiz_data)
    new_quiz_data["questions"] = new_questions

    # Persist to CSV
    if isinstance(selected_id, str) and selected_id.startswith("file:"):
        current_path = selected_id.split(":", 1)[1]
        try:
            sel_map: Dict[int, List[int]] = {}
            for sk, rv in new_answers.items():
                try:
                    i = int(sk)
                except Exception:
                    continue
                sel_map[i] = list(rv.get("selected", []))
            dataset_save_autodetect(new_quiz_data, current_path, selected_map=sel_map)
        except Exception:
            pass

    new_state = dict(state or {})
    new_state["answers"] = new_answers
    new_state["current"] = new_current
    new_state["review"] = False
    msg = html.Div("Frågor slumpade (svar bevarade).", className="alert alert-info")
    # Enable auto-clear timer
    return new_quiz_data, new_state, msg, False


@callback(
    Output("quiz_state", "data", allow_duplicate=True),
    Input("quiz_jump_select", "value"),
    State("quiz_state", "data"),
    State("quiz_data", "data"),
    prevent_initial_call=True,
)
def on_quiz_jump_select(target_idx, state, quiz_data):
    # Jump to a specific question. We don't depend on answer_input here to avoid missing-component errors.
    if target_idx is None or not state or not quiz_data:
        raise PreventUpdate
    n = len(quiz_data.get("questions", []))
    if not isinstance(target_idx, int) or target_idx < 0 or target_idx >= n:
        raise PreventUpdate
    new_state = dict(state)
    new_state["current"] = target_idx
    new_state["review"] = False
    return new_state


@callback(
    Output("feedback", "children", allow_duplicate=True),
    Output("feedback_auto_clear", "disabled", allow_duplicate=True),
    Input("feedback_auto_clear", "n_intervals"),
    State("feedback_auto_clear", "disabled"),
    prevent_initial_call=True,
)
def clear_feedback_on_timer(n, disabled):
    # When timer fires and it's enabled, clear the feedback and stop the timer
    if disabled:
        raise PreventUpdate
    if n is None or n == 0:
        raise PreventUpdate
    return html.Div(), True


@callback(
    Output("quiz_data", "data", allow_duplicate=True),
    Output("quiz_state", "data", allow_duplicate=True),
    Output("feedback", "children", allow_duplicate=True),
    Input("btn_start_new_run", "n_clicks"),
    State("quiz_data", "data"),
    State("selected_dataset", "data"),
    prevent_initial_call=True,
)
def on_start_new_run(nc, quiz_data, selected_id):
    if not nc or not quiz_data:
        raise PreventUpdate
    # Randomize option order per question, then shuffle questions
    opt_shuffled: List[Dict[str, Any]] = []
    for q in (quiz_data.get("questions", []) or []):
        opts = list(q.get("options", []))
        order = list(range(len(opts)))
        random.shuffle(order)
        new_opts = [opts[i] for i in order]
        correct_set = set(q.get("correct", []) or [])
        new_correct = [j for j, old_i in enumerate(order) if old_i in correct_set]
        opt_shuffled.append({"q": q.get("q", ""), "options": new_opts, "correct": new_correct})
    shuffled_questions = list(opt_shuffled)
    random.shuffle(shuffled_questions)
    new_quiz_data = dict(quiz_data)
    new_quiz_data["questions"] = shuffled_questions
    # Persist to CSV and clear Selected
    if isinstance(selected_id, str) and selected_id.startswith("file:"):
        current_path = selected_id.split(":", 1)[1]
        try:
            dataset_save_autodetect(new_quiz_data, current_path, selected_map={})
        except Exception:
            pass
    # Reset state
    new_state = default_state(new_quiz_data.get("hash"))
    msg = html.Div("Ny runda startad.", className="alert alert-info")
    return new_quiz_data, new_state, msg


@callback(
    Output("quiz_stats", "data", allow_duplicate=True),
    Output("feedback", "children", allow_duplicate=True),
    Input("btn_save_to_plot", "n_clicks"),
    State("quiz_stats", "data"),
    State("quiz_data", "data"),
    State("quiz_state", "data"),
    State("scoring_mode", "value"),
    prevent_initial_call=True,
)
def on_save_attempt_to_plot(nc, stats, quiz_data, state, scoring_mode):
    if not nc or not quiz_data or not state:
        raise PreventUpdate
    qs = quiz_data.get("questions", [])
    n = len(qs)
    mode = scoring_mode or "aon"
    # Compute pct (same rules as submit)
    if mode == "aon":
        correct = sum(1 for i in range(n) if (state.get("answers", {}) or {}).get(str(i), {}).get("correct"))
        points = float(correct)
        pct = round((correct / n) * 100, 2) if n else 0
    else:
        total_score = 0.0
        for i in range(n):
            rec = (state.get("answers", {}) or {}).get(str(i), {}) or {}
            if "score" in rec:
                total_score += float(rec.get("score", 0.0))
            else:
                ch = rec.get("selected", []) or []
                corr_i = qs[i].get("correct", [])
                s_i, _ = score_answer(ch, corr_i, mode)
                total_score += s_i
        points = float(round(total_score, 4))
        pct = round((total_score / n) * 100, 2) if n else 0
        correct = int(round(total_score))
    # Append to stats
    stats = stats or {}
    dataset_key = quiz_data.get("hash")
    arr = list(stats.get(dataset_key, []))
    arr.append({"ts": datetime.utcnow().isoformat() + "Z", "total": n, "correct": correct, "pct": pct, "points": points})
    stats[dataset_key] = arr
    # Re-render summary indicating it's saved (hide button)
    fb = render_attempt_summary(quiz_data, state, mode, show_save=False, saved=True)
    return stats, fb


@callback(
    Output("feedback", "children", allow_duplicate=True),
    Output("quiz_state", "data", allow_duplicate=True),
    Input("btn_close_review", "n_clicks"),
    State("quiz_state", "data"),
    prevent_initial_call=True,
)
def on_close_review(nc, state):
    # Hide the result panel and allow user to continue editing answers
    if not nc or not state:
        raise PreventUpdate
    new_state = dict(state)
    new_state["review"] = False
    return html.Div(), new_state

@callback(
    Output("quiz_state", "data", allow_duplicate=True),
    Input("answer_input", "value"),
    State("quiz_state", "data"),
    State("quiz_data", "data"),
    State("selected_dataset", "data"),
    prevent_initial_call=True,
)
def on_answer_change(selected, state, quiz_data, selected_id):
    # Persist selection immediately on change (no grading)
    if not state or not quiz_data:
        raise PreventUpdate
    idx = int(state.get("current", 0))
    new_state = dict(state)
    answers = dict(new_state.get("answers", {}))
    if selected is None:
        # If cleared, store empty selection
        chosen: List[int] = []
    else:
        chosen = [selected] if isinstance(selected, int) else sorted(list(set(selected)))
    prev = answers.get(str(idx), {})
    answers[str(idx)] = {"selected": chosen, "correct": prev.get("correct")}
    new_state["answers"] = answers
    new_state["review"] = False
    # Write to dataset CSV if a file is selected
    if isinstance(selected_id, str) and selected_id.startswith("file:"):
        current_path = selected_id.split(":", 1)[1]
        try:
            sel_map: Dict[int, List[int]] = {}
            for k, v in (new_state.get("answers", {}) or {}).items():
                try:
                    i = int(k)
                except Exception:
                    continue
                sel_map[i] = list(v.get("selected", []))
            dataset_save_autodetect(quiz_data, current_path, selected_map=sel_map)
        except Exception:
            pass
    return new_state


## Removed: separate Finish Attempt handler (Submit handles finish when all questions are answered)


@callback(
    Output("quiz_state", "data", allow_duplicate=True),
    Input("btn_next", "n_clicks"),
    State("quiz_state", "data"),
    State("quiz_data", "data"),
    State("answer_input", "value"),
    State("selected_dataset", "data"),
    prevent_initial_call=True,
)
def on_next(n_clicks, state, quiz_data, selected, selected_id):
    if not state or not quiz_data:
        return no_update
    idx = int(state.get("current", 0))
    n = len(quiz_data["questions"])
    # Save current selection (without grading) before moving on
    new_state = dict(state)
    answers = dict(new_state.get("answers", {}))
    if selected is not None:
        chosen = [selected] if isinstance(selected, int) else sorted(list(set(selected)))
        prev = answers.get(str(idx), {})
        answers[str(idx)] = {"selected": chosen, "correct": prev.get("correct")}
        new_state["answers"] = answers
        # Persist to dataset CSV
        if isinstance(selected_id, str) and selected_id.startswith("file:"):
            current_path = selected_id.split(":", 1)[1]
            try:
                sel_map: Dict[int, List[int]] = {}
                for k, v in (new_state.get("answers", {}) or {}).items():
                    try:
                        i = int(k)
                    except Exception:
                        continue
                    sel_map[i] = list(v.get("selected", []))
                dataset_save_autodetect(quiz_data, current_path, selected_map=sel_map)
            except Exception:
                pass
    idx2 = min(n - 1, idx + 1)
    new_state["current"] = idx2
    new_state["review"] = False
    return new_state


@callback(
    Output("quiz_state", "data", allow_duplicate=True),
    Input("btn_prev", "n_clicks"),
    State("quiz_state", "data"),
    State("quiz_data", "data"),
    State("answer_input", "value"),
    State("selected_dataset", "data"),
    prevent_initial_call=True,
)
def on_prev(n_clicks, state, quiz_data, selected, selected_id):
    if not state:
        return no_update
    idx = int(state.get("current", 0))
    # Save current selection (without grading) before moving
    new_state = dict(state)
    answers = dict(new_state.get("answers", {}))
    if selected is not None:
        chosen = [selected] if isinstance(selected, int) else sorted(list(set(selected)))
        prev = answers.get(str(idx), {})
        answers[str(idx)] = {"selected": chosen, "correct": prev.get("correct")}
        new_state["answers"] = answers
        if isinstance(selected_id, str) and selected_id.startswith("file:"):
            current_path = selected_id.split(":", 1)[1]
            try:
                sel_map: Dict[int, List[int]] = {}
                for k, v in (new_state.get("answers", {}) or {}).items():
                    try:
                        i = int(k)
                    except Exception:
                        continue
                    sel_map[i] = list(v.get("selected", []))
                dataset_save_autodetect(quiz_data, current_path, selected_map=sel_map)
            except Exception:
                pass
    idx2 = max(0, idx - 1)
    new_state["current"] = idx2
    new_state["review"] = False
    return new_state


@callback(
    Output("quiz_state", "data", allow_duplicate=True),
    Output("feedback", "children", allow_duplicate=True),
    Input("btn_reset", "n_clicks"),
    State("quiz_data", "data"),
    State("selected_dataset", "data"),
    prevent_initial_call=True,
)
def on_reset(n_clicks, quiz_data, selected_id):
    if not quiz_data:
        return no_update, no_update
    # Clear selected in file when restarting
    if isinstance(selected_id, str) and selected_id.startswith("file:"):
        current_path = selected_id.split(":", 1)[1]
        try:
            dataset_save_autodetect(quiz_data, current_path, selected_map={})
        except Exception:
            pass
    return default_state(quiz_data.get("hash")), html.Div()


# ---------- Editor helpers and callbacks ----------

def _options_from_text(text: str) -> List[str]:
    if not text:
        return []
    # one option per line
    lines = [ln.strip() for ln in str(text).replace("\r", "").split("\n")]
    return [ln for ln in lines if ln]


def _labels_for_options(options: List[str]) -> List[Dict[str, Any]]:
    return [{"label": f"{chr(ord('A')+i)}. {opt}", "value": i} for i, opt in enumerate(options)]


@callback(
    Output("download_export", "data"),
    Input("btn_export_excel", "n_clicks"),
    State("quiz_data", "data"),
    prevent_initial_call=True,
)
def export_to_excel(nc, quiz_data):
    if not nc or not quiz_data:
        raise PreventUpdate
    qs = quiz_data.get("questions", []) or []
    if not qs:
        raise PreventUpdate
    # Compute max number of options across questions
    max_opts = max((len(q.get("options", []) or []) for q in qs), default=0)
    # Build rows
    rows = []
    for q in qs:
        opts = list(q.get("options", []) or [])
        corr = list(q.get("correct", []) or [])
        row = {"Question": q.get("q", "")}
        # Option columns A..N
        for i in range(max_opts):
            label = f"Option {chr(ord('A')+i)}"
            row[label] = opts[i] if i < len(opts) else ""
        # Correct letters (A,B,...) joined by comma
        letters = [chr(ord('A')+i) for i in corr if isinstance(i, int) and 0 <= i < len(opts)]
        row["Correct"] = ", ".join(letters)
        rows.append(row)
    df = pd.DataFrame(rows)
    # Try Excel first; if engine not available, fall back to CSV
    try:
        import openpyxl  # noqa: F401
        return dcc.send_data_frame(df.to_excel, f"{(quiz_data.get('title') or 'quiz').strip() or 'quiz'}.xlsx", index=False, sheet_name="Questions")
    except Exception:
        # Fallback CSV
        return dcc.send_data_frame(df.to_csv, f"{(quiz_data.get('title') or 'quiz').strip() or 'quiz'}.csv", index=False, sep=";")


@callback(
    Output("edit_title", "value"),
    Output("edit_q_select", "options"),
    Output("edit_q_select", "value"),
    Output("edit_q_text", "value"),
    Output("edit_options_text", "value"),
    Output("edit_correct_choices", "options"),
    Output("edit_correct_choices", "value"),
    Input("quiz_data", "data"),
    Input("edit_q_select", "value"),
    prevent_initial_call=False,
)
def editor_load(quiz_data, sel_idx):
    if not quiz_data:
        return "", [], None, "", "", [], []
    qs = quiz_data.get("questions", [])
    if not qs:
        # initialize with one blank question
        qs = [blank_question()]
    # Build dropdown options
    opts = [
        {"label": f"Q{i+1}: {q.get('q','')[:40]}", "value": i}
        for i, q in enumerate(qs)
    ]
    if sel_idx is None or not isinstance(sel_idx, int) or sel_idx >= len(qs):
        sel_idx = 0
    q = qs[sel_idx]
    title = quiz_data.get("title", "Quiz")
    qtext = q.get("q", "")
    options = q.get("options", [])
    corr = q.get("correct", [])
    options_text = "\n".join(options)
    chk_opts = _labels_for_options(options)
    return title, opts, sel_idx, qtext, options_text, chk_opts, corr


# Removed: create flow is now click-first then enter title


@callback(
    Output("edit_correct_choices", "options", allow_duplicate=True),
    Output("edit_correct_choices", "value", allow_duplicate=True),
    Input("edit_options_text", "value"),
    State("edit_correct_choices", "value"),
    prevent_initial_call=True,
)
def editor_sync_choices(opt_text, current_values):
    options = _options_from_text(opt_text or "")
    chk_opts = _labels_for_options(options)
    # clamp selected values to available indices
    if not isinstance(current_values, list):
        current_values = []
    max_idx = len(options) - 1
    new_values = [v for v in current_values if isinstance(v, int) and 0 <= v <= max_idx]
    return chk_opts, new_values


@callback(
    Output("edit_q_select", "disabled", allow_duplicate=True),
    Output("edit_q_text", "disabled", allow_duplicate=True),
    Output("edit_options_text", "disabled", allow_duplicate=True),
    Output("edit_correct_choices", "style", allow_duplicate=True),
    Output("btn_add_question", "disabled", allow_duplicate=True),
    Output("btn_delete_question", "disabled", allow_duplicate=True),
    Output("editor_status", "children", allow_duplicate=True),
    Input("selected_dataset", "data"),
    Input("new_dataset_mode", "data"),
    prevent_initial_call="initial_duplicate",
)
def gate_editor_controls(selected_id, new_mode):
    has_dataset = isinstance(selected_id, str) and selected_id.startswith("file:")
    allow = has_dataset or bool(new_mode)
    disabled = not allow
    msg = no_update
    if disabled:
        msg = html.Div("Click 'Create New Dataset' and enter a title to start editing.", className="text-info")
    checklist_style = {"pointerEvents": "none", "opacity": 0.6} if disabled else {}
    return disabled, disabled, disabled, checklist_style, disabled, disabled, msg


@callback(
    Output("quiz_data", "data", allow_duplicate=True),
    Input("edit_title", "value"),
    Input("edit_q_select", "value"),
    Input("edit_q_text", "value"),
    Input("edit_options_text", "value"),
    Input("edit_correct_choices", "value"),
    State("quiz_data", "data"),
    State("selected_dataset", "data"),
    State("new_dataset_mode", "data"),
    prevent_initial_call=True,
)
def editor_live_update(title, sel_idx, qtext, opt_text, correct_vals, quiz_data, selected_id, new_mode):
    # Build updated dataset live as user edits
    # Allow live edits if file-backed OR we're in new-dataset mode (no autosave until created)
    if not quiz_data:
        raise PreventUpdate
    allow_edit = (isinstance(selected_id, str) and selected_id.startswith("file:")) or bool(new_mode)
    if not allow_edit:
        raise PreventUpdate
    current_title = quiz_data.get("title", "")
    qs = list(quiz_data.get("questions", []))
    if not qs:
        qs = [blank_question()]
    if sel_idx is None or not isinstance(sel_idx, int) or sel_idx >= len(qs):
        sel_idx = 0
    # Compare existing values to avoid loops
    existing = qs[sel_idx]
    new_options = _options_from_text(opt_text or "")
    if not isinstance(correct_vals, list):
        correct_vals = []
    new_correct = sorted({v for v in correct_vals if isinstance(v, int) and 0 <= v < len(new_options)})
    new_q = (qtext or "").strip()
    new_title = (title or current_title or "Quiz").strip()

    if (
        new_title == current_title and
        new_q == (existing.get("q") or "") and
        new_options == existing.get("options", []) and
        new_correct == existing.get("correct", [])
    ):
        return no_update

    qs[sel_idx] = {"q": new_q, "options": new_options, "correct": new_correct}
    new_data = dataset_from_questions(qs, new_title)
    return new_data


@callback(
    Output("datasets_list", "data", allow_duplicate=True),
    Output("dataset_select", "options", allow_duplicate=True),
    Output("dataset_select", "value", allow_duplicate=True),
    Output("csv_load_status", "children", allow_duplicate=True),
    Output("selected_dataset", "data", allow_duplicate=True),
    Output("editor_status", "children", allow_duplicate=True),
    Output("new_dataset_mode", "data", allow_duplicate=True),
    Input("edit_title", "value"),
    State("quiz_data", "data"),
    State("new_dataset_mode", "data"),
    prevent_initial_call=True,
)
def create_file_when_titled(title, quiz_data, new_mode):
    # Create underlying CSV when in new-dataset mode and a non-empty title is provided
    title = (title or "").strip()
    if not new_mode or not title or not quiz_data:
        raise PreventUpdate
    ensure_csv_dir()
    path = _unique_csv_path_for_title(title)
    try:
        # ensure dataset carries the title
        working = dict(quiz_data)
        working["title"] = title
        dataset_save_autodetect(working, path)
    except Exception as e:
        return no_update, no_update, no_update, html.Small(f"Kunde inte spara: {e}", className="text-danger"), no_update, no_update, no_update
    # Reload folder and select new file
    datasets = load_all_csv_datasets()
    entries = []
    for d in datasets:
        fid = f"file:{os.path.abspath(d.get('source_path'))}"
        t2 = d.get("title") or os.path.splitext(os.path.basename(d.get("source_path", "")))[0]
        entries.append({"id": fid, "title": t2, "data": d})
    options = [{"label": e["title"], "value": e["id"]} for e in entries]
    chosen_id = f"file:{os.path.abspath(path)}"
    status = html.Small(f"Skapade csv/{os.path.basename(path)}", className="text-muted")
    return entries, options, chosen_id, status, chosen_id, html.Div("Dataset created.", className="text-success"), False


@callback(
    Output("quiz_data", "data", allow_duplicate=True),
    Output("edit_q_select", "value", allow_duplicate=True),
    Output("editor_status", "children", allow_duplicate=True),
    Input("btn_add_question", "n_clicks"),
    State("quiz_data", "data"),
    State("selected_dataset", "data"),
    prevent_initial_call=True,
)
def editor_add_question(nc, quiz_data, selected_id):
    if not quiz_data:
        return no_update, no_update, html.Div("Create a dataset first (enter a title and click Create Dataset).", className="text-warning")
    if not (isinstance(selected_id, str) and selected_id.startswith("file:")):
        return no_update, no_update, html.Div("Create a dataset first.", className="text-warning")
    qs = list(quiz_data.get("questions", []))
    qs.append(blank_question())
    new_data = dataset_from_questions(qs, (quiz_data.get("title") or "Quiz").strip())
    new_idx = len(qs) - 1
    return new_data, new_idx, html.Div("Question added.", className="text-success")


@callback(
    Output("quiz_data", "data", allow_duplicate=True),
    Output("edit_q_select", "value", allow_duplicate=True),
    Output("editor_status", "children", allow_duplicate=True),
    Input("btn_delete_question", "n_clicks"),
    State("quiz_data", "data"),
    State("edit_q_select", "value"),
    State("selected_dataset", "data"),
    prevent_initial_call=True,
)
def editor_delete_question(nc, quiz_data, sel_idx, selected_id):
    if not quiz_data:
        return no_update, no_update, html.Div("No dataset loaded.", className="text-warning")
    if not (isinstance(selected_id, str) and selected_id.startswith("file:")):
        return no_update, no_update, html.Div("Create a dataset first.", className="text-warning")
    qs = list(quiz_data.get("questions", []))
    if not qs:
        return no_update, no_update, html.Div("Nothing to delete.", className="text-warning")
    if sel_idx is None or not isinstance(sel_idx, int) or sel_idx < 0 or sel_idx >= len(qs):
        sel_idx = 0
    del qs[sel_idx]
    if not qs:
        qs = [blank_question()]
        new_idx = 0
    else:
        new_idx = max(0, sel_idx - 1)
    new_data = dataset_from_questions(qs, (quiz_data.get("title") or "Quiz").strip())
    return new_data, new_idx, html.Div("Question deleted.", className="text-success")


@callback(
    Output("quiz_data", "data", allow_duplicate=True),
    Output("mode_tabs", "value", allow_duplicate=True),
    Output("editor_status", "children", allow_duplicate=True),
    Output("dataset_select", "value", allow_duplicate=True),
    Output("selected_dataset", "data", allow_duplicate=True),
    Output("csv_load_status", "children", allow_duplicate=True),
    Output("new_dataset_mode", "data", allow_duplicate=True),
    Input("btn_new_dataset", "n_clicks"),
    prevent_initial_call=True,
)
def editor_new_dataset_start(nc):
    # Enter 'new dataset' mode: clear selection and present a blank dataset for editing title/questions
    new = dataset_from_questions([blank_question()], "")
    status = html.Small("New dataset mode: enter a title to create the file.", className="text-muted")
    return new, "editor", html.Div("Fill in a title to create the dataset.", className="text-info"), None, None, status, True


## Removed: Export to JSON (JSON is now the default save format)

if __name__ == "__main__":
    # Run the Dash app (Dash 3.x+)
    # Toggle dev tools and debug via env var: set QUIZAPP_DEBUG=1 to enable
    DEBUG = os.getenv("QUIZAPP_DEBUG", "0") == "1"
    app.run(
        debug=DEBUG,
        dev_tools_ui=DEBUG,
        dev_tools_props_check=False,  # reduce noisy React/props warnings in dev
        dev_tools_hot_reload=DEBUG,
        dev_tools_silence_routes_logging=True,
    )
