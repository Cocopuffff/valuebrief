"""
report_writer.py
~~~~~~~~~~~~~~~~
Utility for writing per-run, per-agent markdown artifacts to logs/runs/.

File naming:  logs/runs/{TICKER}_{datetime_iso}.md
              e.g.  logs/runs/ADBE_2026-04-12T00-05-00.123456.md

Each agent node appends its section to the file as it completes.
Because the filename is unique per datetime, no section-anchoring or
idempotency logic is needed — if two runs somehow share a filename,
the second simply overwrites the first.
"""

from __future__ import annotations

import pathlib
from datetime import datetime


def _safe_iso(dt_str: str) -> str:
    """Sanitise an ISO datetime string for use as a filename component.

    Replaces colons in the time part with hyphens so the name is valid
    on all OSes.  e.g. '2026-04-12T00:05:00.123456' → '2026-04-12T00-05-00.123456'
    """
    if "T" in dt_str:
        date_part, time_part = dt_str.split("T", 1)
        return f"{date_part}T{time_part.replace(':', '-')}"
    return dt_str


class RunReportWriter:
    """
    Writes each agent node's final output into a dedicated markdown file.

    Sections are appended in the order agents complete:
        write_header  →  write_bull_thesis  →  write_bear_thesis
        →  write_judge_output  →  write_final_report

    Usage::

        writer = RunReportWriter(ticker="ADBE", run_datetime="2026-04-12T00:05:00.123456")
        writer.write_header(company="Adobe Inc.")
        writer.write_bull_thesis(text)
        writer.write_bear_thesis(text)
        writer.write_judge_output(synthesis, valuation_md, decision)
        writer.write_final_report(report_text, sources)
    """

    def __init__(self, ticker: str, company: str, run_datetime: str) -> None:
        """
        Args:
            ticker:       Stock ticker symbol, e.g. 'ABCD'.
            company:      Company name, e.g. 'ABCD Inc.'. Optional, falls back to ticker.
            run_datetime: ISO datetime string from datetime.now().isoformat(),
                          matching the thread_id used in main.py.
        """
        self.ticker = ticker.upper()
        self.run_datetime = run_datetime
        self.company = company or self.ticker

        # Resolve path relative to this file: src/ → project root → logs/runs/
        project_root = pathlib.Path(__file__).parent.parent.parent
        runs_dir = project_root / "logs" / "runs"
        runs_dir.mkdir(parents=True, exist_ok=True)

        debug_filename = f"{self.ticker}_debug_{_safe_iso(run_datetime)}.md"
        final_filename = f"{self.ticker}_{_safe_iso(run_datetime)}.md"
        self.debug_path = runs_dir / debug_filename
        self.final_path = runs_dir / final_filename

    # ── Internal helpers ──────────────────────────────────────────────────

    def _write_file(self, path: str, content: str) -> None:
        """Overwrite the file with content (used for header / full reset)."""
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)

    def _append_file(self, path: str, content: str) -> None:
        """Append content to the file."""
        with open(path, "a", encoding="utf-8") as f:
            f.write(content)

    # ── Public API ────────────────────────────────────────────────────────

    def write_header(self) -> None:
        """Create (or overwrite) the file with the run metadata header."""
        thread_id = f"{self.ticker}-{self.run_datetime}"
        header = (
            f"# {self.ticker} — {self.run_datetime[:10]}\n\n"
            f"## Run Metadata\n\n"
            f"| Field | Value |\n"
            f"| ------- | ------- |\n"
            f"| **Thread ID** | `{thread_id}` |\n"
            f"| **Started** | {self.run_datetime} |\n"
            f"| **Ticker** | {self.ticker} |\n"
            f"| **Company** | {self.company} |\n\n"
            f"---\n"
        )
        self._write_file(self.debug_path, header)

    def write_bull_thesis(self, text: str) -> None:
        """Append the Bull Analyst section."""
        self._append_file(
            self.debug_path,
            f"\n## 🐂 Bull Analyst\n\n"
            f"> Completed: {datetime.now().isoformat()}\n\n"
            f"{text}\n\n"
            f"---\n"
        )

    def write_bear_thesis(self, text: str) -> None:
        """Append the Bear Analyst section."""
        self._append_file(
            self.debug_path,
            f"\n## 🐻 Bear Analyst\n\n"
            f"> Completed: {datetime.now().isoformat()}\n\n"
            f"{text}\n\n"
            f"---\n"
        )

    def write_judge_output(
        self,
        synthesis: str,
        valuation_md: str,
        decision: str,
    ) -> None:
        """Append the Judge Analyst section (synthesis + DCF table + final decision).

        Args:
            synthesis:    Qualitative synthesis text.
            valuation_md: Pre-formatted Markdown string from _build_dcf_summary().
                          Pass an empty string if valuation was skipped.
            decision:     Final reconciled investment decision text.
        """
        valuation_block = (
            f"### Valuation (DCF)\n\n{valuation_md}\n"
            if valuation_md
            else "### Valuation (DCF)\n\n_(Valuation data unavailable)_\n"
        )
        self._append_file(
            self.debug_path,
            f"\n## ⚖️ Judge Analyst\n\n"
            f"> Completed: {datetime.now().isoformat()}\n\n"
            f"### Synthesis\n\n"
            f"{synthesis}\n\n"
            f"---\n\n"
            f"{valuation_block}\n"
            f"---\n\n"
            f"### Final Decision\n\n"
            f"{decision}\n\n"
            f"---\n"
        )

    def write_final_report(self, debug_report_text: str, report_text: str, sources: list) -> None:
        """Append the Final Report section including sources."""
        debug_content = (
            f"\n## 📄 Final Report\n\n"
            f"> Completed: {datetime.now().isoformat()}\n\n"
            f"{debug_report_text}\n"
        )
        final_content = (
            f"> Completed: {datetime.now().isoformat()}\n\n"
            f"# {self.company} - {self.ticker} Investment Report\n\n"
            f"---\n"
            f"{report_text}\n"
        )
        self._append_file(self.debug_path, debug_content)
        self._write_file(self.final_path, final_content)
