"""
Build manuscript-ready behavioral analytics tables from JSONL session logs.

This exporter reads the final ``session_end`` summary from each session log,
filters out idle sessions with zero annotations, anonymizes user identifiers,
and writes both CSV and Markdown outputs for manuscript revision.

Usage:
    python analytics/export_behavioral_manuscript_tables.py

    python analytics/export_behavioral_manuscript_tables.py \
        --base-dir db/behavioral_analytics \
        --out-dir results/behavioral_analytics
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


SESSION_MARKER_TYPES = {"session_start", "session_end"}


@dataclass
class SessionRecord:
    expert_id: str
    raw_user_id: str
    session_id: str
    session_date: str
    session_duration_ms: int
    annotation_count: int
    time_on_task_per_annotation_sec: float
    interaction_events: int
    slice_navigation_count: int
    manual_annotations: int
    ai_assisted_annotations: int
    ai_runs: int
    annotation_edits: int
    annotation_deletions: int
    edit_rate: float
    deletion_rate: float
    correction_rate: float

    def to_csv_row(self) -> dict[str, Any]:
        return {
            "Expert ID": self.expert_id,
            "Session ID": self.session_id,
            "Session Date": self.session_date,
            "Session Duration (min)": round(self.session_duration_ms / 60000, 2),
            "Annotations (n)": self.annotation_count,
            "Time-on-Task / Annotation (sec)": round(self.time_on_task_per_annotation_sec, 2),
            "Interaction Events (n)": self.interaction_events,
            "Slice Navigations (n)": self.slice_navigation_count,
            "Manual Annotations (n)": self.manual_annotations,
            "AI-Assisted Annotations (n)": self.ai_assisted_annotations,
            "AI Runs (n)": self.ai_runs,
            "Annotation Edits (n)": self.annotation_edits,
            "Annotation Deletions (n)": self.annotation_deletions,
            "Edit Rate (%)": round(self.edit_rate * 100, 1),
            "Deletion Rate (%)": round(self.deletion_rate * 100, 1),
            "Correction Rate (%)": round(self.correction_rate * 100, 1),
        }


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def _load_json_lines(file_path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with open(file_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return events


def _get_last_session_end(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    session_end_event = None
    for event in events:
        if event.get("event_type") != "session_end":
            continue
        metadata = event.get("metadata") or {}
        if isinstance(metadata.get("counters_summary"), dict):
            session_end_event = event
    return session_end_event


def _format_session_date(timestamp: str | None, fallback_session_id: str) -> str:
    if timestamp:
        try:
            return datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
        except ValueError:
            pass
    return fallback_session_id


def _collect_session_record(
    file_path: Path,
    raw_user_id: str,
    expert_id: str,
    include_empty: bool,
) -> SessionRecord | None:
    events = _load_json_lines(file_path)
    if not events:
        return None

    session_end_event = _get_last_session_end(events)
    if session_end_event is None:
        return None

    metadata = session_end_event.get("metadata") or {}
    counters = metadata.get("counters_summary") or {}

    annotation_count = _safe_int(counters.get("annotation_count"))
    if annotation_count <= 0 and not include_empty:
        return None

    session_duration_ms = _safe_int(metadata.get("session_duration_ms"))
    manual_annotations = _safe_int(counters.get("manual_annotations"))
    ai_assisted_annotations = _safe_int(counters.get("ai_assisted_annotations"))
    ai_runs = _safe_int(counters.get("ai_segmentation_runs")) + _safe_int(
        counters.get("automatic_segmentation_runs")
    )
    slice_navigation_count = _safe_int(counters.get("slice_navigation_count"))
    annotation_edits = _safe_int(counters.get("annotation_edits"))
    annotation_deletions = _safe_int(counters.get("annotation_deletions"))
    interaction_events = sum(
        1 for event in events if event.get("event_type") not in SESSION_MARKER_TYPES
    )

    if annotation_count > 0:
        time_on_task_per_annotation_sec = session_duration_ms / 1000 / annotation_count
        edit_rate = annotation_edits / annotation_count
        deletion_rate = annotation_deletions / annotation_count
        correction_rate = (annotation_edits + annotation_deletions) / annotation_count
    else:
        time_on_task_per_annotation_sec = 0.0
        edit_rate = 0.0
        deletion_rate = 0.0
        correction_rate = 0.0

    return SessionRecord(
        expert_id=expert_id,
        raw_user_id=raw_user_id,
        session_id=file_path.stem,
        session_date=_format_session_date(session_end_event.get("timestamp"), file_path.stem),
        session_duration_ms=session_duration_ms,
        annotation_count=annotation_count,
        time_on_task_per_annotation_sec=time_on_task_per_annotation_sec,
        interaction_events=interaction_events,
        slice_navigation_count=slice_navigation_count,
        manual_annotations=manual_annotations,
        ai_assisted_annotations=ai_assisted_annotations,
        ai_runs=ai_runs,
        annotation_edits=annotation_edits,
        annotation_deletions=annotation_deletions,
        edit_rate=edit_rate,
        deletion_rate=deletion_rate,
        correction_rate=correction_rate,
    )


def collect_sessions(base_dir: Path, include_empty: bool = False) -> list[SessionRecord]:
    user_dirs = sorted(
        [path for path in base_dir.iterdir() if path.is_dir()],
        key=lambda path: path.name.lower(),
    )
    expert_labels = {user_dir.name: f"Expert {idx}" for idx, user_dir in enumerate(user_dirs, start=1)}

    records: list[SessionRecord] = []
    for user_dir in user_dirs:
        sessions_dir = user_dir / "sessions"
        if not sessions_dir.exists():
            continue
        for session_file in sorted(sessions_dir.glob("*.jsonl")):
            record = _collect_session_record(
                session_file,
                raw_user_id=user_dir.name,
                expert_id=expert_labels[user_dir.name],
                include_empty=include_empty,
            )
            if record is not None:
                records.append(record)

    records.sort(key=lambda rec: (rec.expert_id, rec.session_date, rec.session_id))
    return records


def build_user_summary(session_records: list[SessionRecord]) -> list[dict[str, Any]]:
    grouped: dict[str, list[SessionRecord]] = {}
    for record in session_records:
        grouped.setdefault(record.expert_id, []).append(record)

    rows: list[dict[str, Any]] = []
    for expert_id in sorted(grouped.keys()):
        records = grouped[expert_id]
        total_annotations = sum(record.annotation_count for record in records)
        total_time_ms = sum(record.session_duration_ms for record in records)
        total_manual = sum(record.manual_annotations for record in records)
        total_ai_assisted = sum(record.ai_assisted_annotations for record in records)
        total_ai_runs = sum(record.ai_runs for record in records)
        total_edits = sum(record.annotation_edits for record in records)
        total_deletions = sum(record.annotation_deletions for record in records)

        if total_annotations > 0:
            time_on_task_per_annotation_sec = total_time_ms / 1000 / total_annotations
            ai_assistance_rate = total_ai_assisted / total_annotations
            edit_rate = total_edits / total_annotations
            deletion_rate = total_deletions / total_annotations
            correction_rate = (total_edits + total_deletions) / total_annotations
        else:
            time_on_task_per_annotation_sec = 0.0
            ai_assistance_rate = 0.0
            edit_rate = 0.0
            deletion_rate = 0.0
            correction_rate = 0.0

        rows.append(
            {
                "Expert": expert_id,
                "Sessions (n)": len(records),
                "Time-on-task (min)": round(total_time_ms / 60000, 2),
                "Annotations (n)": total_annotations,
                "Time/annotation (s)": round(time_on_task_per_annotation_sec, 2),
                "Interactions/session": round(
                    _mean([record.interaction_events for record in records]), 2
                ),
                "Manual/AI-assisted annotations": f"{total_manual} / {total_ai_assisted}",
                "AI assistance (%)": round(ai_assistance_rate * 100, 1),
                "AI runs (n)": total_ai_runs,
                "Edits/deletions (n)": f"{total_edits} / {total_deletions}",
            }
        )

    return rows


def _write_csv(file_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        with open(file_path, "w", encoding="utf-8", newline="") as handle:
            handle.write("")
        return

    fieldnames = list(rows[0].keys())
    with open(file_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _escape_markdown(value: Any) -> str:
    return str(value).replace("|", "\\|")


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_No rows available._"

    headers = list(rows[0].keys())
    header_row = "| " + " | ".join(headers) + " |"
    separator_row = "| " + " | ".join(["---"] * len(headers)) + " |"
    body_rows = []
    for row in rows:
        body_rows.append(
            "| " + " | ".join(_escape_markdown(row[header]) for header in headers) + " |"
        )
    return "\n".join([header_row, separator_row] + body_rows)


def _build_markdown_report(
    user_rows: list[dict[str, Any]],
    session_rows: list[dict[str, Any]],
) -> str:
    total_experts = len(user_rows)
    total_sessions = len(session_rows)
    total_annotations = sum(int(row["Annotations (n)"]) for row in session_rows)
    total_time_minutes = sum(float(row["Session Duration (min)"]) for row in session_rows)
    total_interactions = sum(int(row["Interaction Events (n)"]) for row in session_rows)

    if total_annotations > 0:
        total_ai_assisted = sum(int(row["AI-Assisted Annotations (n)"]) for row in session_rows)
        ai_share = total_ai_assisted / total_annotations * 100
    else:
        ai_share = 0.0

    preview_rows = session_rows[:15]

    report = []
    report.append("# Behavioral Analytics Manuscript Tables")
    report.append("")
    report.append(
        "This report summarizes non-empty behavioral analytics sessions extracted from "
        "`db/behavioral_analytics`. Session summaries were read from the final `session_end` "
        "record in each JSONL log. Sessions with zero annotations were excluded. Interaction "
        "events were defined as logged events excluding `session_start` and `session_end` markers. "
        "Expert identifiers were anonymized as Expert 1, Expert 2, and so on."
    )
    report.append("")
    report.append(
        f"Across {total_experts} experts, {total_sessions} non-empty sessions, and "
        f"{total_annotations} annotations, the logs recorded {total_time_minutes:.2f} minutes "
        f"of time-on-task and {total_interactions} interaction events. AI-assisted annotations "
        f"accounted for {ai_share:.1f}% of all logged annotations. Correction activity in the "
        "session-level supplement is reported as edit and deletion event counts normalized by the "
        "number of annotations and may therefore exceed 100% when a single annotation is revised "
        "or removed multiple times."
    )
    report.append("")
    report.append("## Main Manuscript Table")
    report.append("")
    report.append(
        "The compact main table below is intended for direct insertion into the revised manuscript."
    )
    report.append("")
    report.append(_markdown_table(user_rows))
    report.append("")
    report.append("## Session-Level Supplement Preview")
    report.append("")
    report.append(
        "The first fifteen non-empty sessions are shown below for transparency. The full "
        "session-level supplementary table is available in `behavioral_session_summary.csv`."
    )
    report.append("")
    report.append(_markdown_table(preview_rows))
    report.append("")
    report.append("## Reporting Note")
    report.append("")
    report.append(
        "Time-on-task was defined as the interval between `session_start` and the final "
        "`session_end` record for each included session. Time-on-task per annotation was computed "
        "as session duration divided by annotation count. AI runs combine interactive AI segmentation "
        "and automatic segmentation invocations. The exported tables are descriptive observational "
        "summaries from the behavioral analytics layer and do not by themselves constitute a controlled "
        "paired comparison between manual-only and AI-assisted workflows."
    )
    report.append("")

    return "\n".join(report)


def _build_latex_snippet(user_rows: list[dict[str, Any]]) -> str:
    latex_headers = [
        "Expert",
        "Sessions",
        "Time-on-task (min)",
        "Annotations",
        "Time/annotation (s)",
        "Interactions/session",
        "Manual/AI-assisted ann.",
        "AI assistance (\\%)",
        "AI runs",
        "Edits/deletions",
    ]

    latex_rows: list[list[str]] = []
    for row in user_rows:
        latex_rows.append(
            [
                str(row["Expert"]),
                str(row["Sessions (n)"]),
                str(row["Time-on-task (min)"]),
                str(row["Annotations (n)"]),
                str(row["Time/annotation (s)"]),
                str(row["Interactions/session"]),
                str(row["Manual/AI-assisted annotations"]),
                str(row["AI assistance (%)"]),
                str(row["AI runs (n)"]),
                str(row["Edits/deletions (n)"]),
            ]
        )

    body_lines = []
    for values in latex_rows:
        escaped = [value.replace("&", r"\&").replace("_", r"\_") for value in values]
        body_lines.append("        " + " & ".join(escaped) + r" \\")

    lines = [
        r"\add{To complement the threshold-based taxonomy with exact measured values, Table~\ref{tab:behavioral-engine-summary} reports the quantitative workflow metrics computed by the behavioral engine developed in this study. These values were derived from non-empty annotation sessions by aggregating the event-stream and session-level logs generated within the platform. The table summarizes time-on-task, interaction intensity, manual and AI-assisted annotation counts, AI invocation frequency, and correction-related activity for each anonymized expert, thereby making explicit the quantitative basis of the behavioral categories described above.}",
        "",
        r"\begin{table}[t]",
        r"    \centering",
        r"    \caption{Behavioral workflow summary computed by the behavioral engine.}",
        r"    \label{tab:behavioral-engine-summary}",
        r"    \setlength{\tabcolsep}{3pt}",
        r"    \scriptsize",
        r"    \resizebox{\columnwidth}{!}{%",
        r"    \begin{tabular}{lccccccccc}",
        r"        \hline",
        "        " + " & ".join(latex_headers) + r" \\",
        r"        \hline",
    ]
    lines.extend(body_lines)
    lines.extend(
        [
            r"        \hline",
            r"    \end{tabular}%",
            r"    }",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def export_tables(base_dir: Path, out_dir: Path, include_empty: bool = False) -> dict[str, Path]:
    session_records = collect_sessions(base_dir, include_empty=include_empty)
    session_rows = [record.to_csv_row() for record in session_records]
    user_rows = build_user_summary(session_records)

    out_dir.mkdir(parents=True, exist_ok=True)
    session_csv = out_dir / "behavioral_session_summary.csv"
    user_csv = out_dir / "behavioral_user_summary.csv"
    markdown_path = out_dir / "behavioral_manuscript_tables.md"
    latex_path = out_dir / "behavioral_main_table_alexandria.tex"

    _write_csv(session_csv, session_rows)
    _write_csv(user_csv, user_rows)

    markdown_report = _build_markdown_report(user_rows, session_rows)
    with open(markdown_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(markdown_report)

    latex_snippet = _build_latex_snippet(user_rows)
    with open(latex_path, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(latex_snippet)

    return {
        "session_csv": session_csv,
        "user_csv": user_csv,
        "markdown": markdown_path,
        "latex": latex_path,
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Export manuscript-ready behavioral analytics tables from JSONL logs."
    )
    parser.add_argument(
        "--base-dir",
        default="db/behavioral_analytics",
        help="Behavioral analytics directory containing per-user session logs.",
    )
    parser.add_argument(
        "--out-dir",
        default="results/behavioral_analytics",
        help="Directory where CSV and Markdown outputs will be written.",
    )
    parser.add_argument(
        "--include-empty-sessions",
        action="store_true",
        help="Include sessions with zero annotations. Disabled by default for manuscript tables.",
    )
    args = parser.parse_args(argv)

    outputs = export_tables(
        base_dir=Path(args.base_dir),
        out_dir=Path(args.out_dir),
        include_empty=args.include_empty_sessions,
    )

    print("Behavioral manuscript tables written to:")
    for label, path in outputs.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()