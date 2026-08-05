#!/usr/bin/env python3
"""Machine-check the documentation invariants that AGENTS.md states in prose.

The campaign is unattended, so "the agent was told to keep the index in sync" is not a
control. This is the control: it runs after every Codex writing stage in
codex/run_campaign.sh, and a stage that leaves a broken cross-reference or an unindexed
report does not get its completion marker.

Checked, all from AGENTS.md:

  §10  every relative markdown link under the subtree resolves
  §7   every report under reports/ is referenced from EXPERIMENT_REPORT_INDEX.md
  §7   every report has a "claims that do NOT hold" section -- the one section the file
       calls mandatory, because it is what the project's credibility rests on

Pre-existing violations live in codex/known_debt.txt so the checker is green on a clean
tree and only fails on damage done by the current session. Adding a line there is a
deliberate act that shows up in review; silently widening the checker is not.

Usage:
    python3 codex/check_docs.py            # errors -> exit 1
    python3 codex/check_docs.py --strict   # warnings count as errors too
    python3 codex/check_docs.py --list-debt  # print the entries that would be waived
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
INDEX = ROOT / "EXPERIMENT_REPORT_INDEX.md"
DEBT = Path(__file__).resolve().parent / "known_debt.txt"

# papers/ is a PDF corpus with its own README; archive/ is explicitly non-authoritative.
SKIP_DIRS = {"papers", ".git", "__pycache__"}

LINK_RE = re.compile(r"\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

# A heading that scopes what the data cannot support. The corpus words this three ways
# ("不能成立的结论", "能说明什么，不能说明什么"), so match the negation, not one phrasing.
NOT_HOLD_RE = re.compile(r"^#{1,4}\s.*(不能|不可|不得|NOT hold|do not hold|must not)", re.M | re.I)

# AGENTS.md §7 gives rejected reports a different obligation: say what may still be reused
# and what must not be reused as a conclusion.
REUSE_RE = re.compile(r"^#{1,4}\s.*(保留|复用|reuse|reused)", re.M | re.I)

# The corpus states the evidence grade as a header line, not as a section.
GRADE_RE = re.compile(r"(实验状态|证据等级|证据边界|evidence grade)", re.I)


def load_debt() -> set[str]:
    if not DEBT.exists():
        return set()
    out = set()
    for line in DEBT.read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.add(line)
    return out


def markdown_files() -> list[Path]:
    return sorted(
        p for p in ROOT.rglob("*.md")
        if not (SKIP_DIRS & set(p.relative_to(ROOT).parts))
    )


def check_links(errors: list[str], debt: set[str]) -> None:
    for md in markdown_files():
        rel = md.relative_to(ROOT).as_posix()
        for m in LINK_RE.finditer(md.read_text(encoding="utf-8", errors="replace")):
            href = m.group(2).split("#")[0].strip()
            if not href or href.startswith(("http://", "https://", "mailto:")):
                continue
            if (md.parent / href).exists():
                continue
            key = f"link:{rel}:{href}"
            if key in debt:
                continue
            errors.append(f"broken link  {rel}  ->  {href}")


def check_reports(errors: list[str], warnings: list[str], debt: set[str]) -> None:
    reports_dir = ROOT / "reports"
    if not reports_dir.is_dir():
        errors.append("reports/ is missing")
        return
    if not INDEX.exists():
        errors.append("EXPERIMENT_REPORT_INDEX.md is missing")
        return

    index_text = INDEX.read_text(encoding="utf-8", errors="replace")
    for report in sorted(reports_dir.rglob("*.md")):
        rel = report.relative_to(ROOT).as_posix()
        text = report.read_text(encoding="utf-8", errors="replace")

        # The index may link by full path or by filename; either is traceable.
        if rel not in index_text and report.name not in index_text:
            key = f"index:{rel}"
            if key not in debt:
                errors.append(
                    f"unindexed report  {rel}  "
                    f"(AGENTS.md §7: update EXPERIMENT_REPORT_INDEX.md in the same change)")

        rejected = rel.startswith("reports/rejected/")
        needed = (NOT_HOLD_RE.search(text) and REUSE_RE.search(text)) if rejected \
            else NOT_HOLD_RE.search(text)
        if not needed and f"nothold:{rel}" not in debt:
            what = ("what may still be reused and what must not be reused as a conclusion"
                    if rejected else "claims that do NOT hold")
            errors.append(
                f"no '{what}' section  {rel}  "
                f"(AGENTS.md §7: mandatory; a report without it is incomplete)")

        if not GRADE_RE.search(text):
            warnings.append(f"no evidence grade stated  {rel}  (AGENTS.md §7 header)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--strict", action="store_true", help="treat warnings as errors")
    ap.add_argument("--list-debt", action="store_true", help="print waived entries and exit")
    args = ap.parse_args()

    debt = load_debt()
    if args.list_debt:
        for d in sorted(debt):
            print(d)
        return 0

    errors: list[str] = []
    warnings: list[str] = []
    check_links(errors, debt)
    check_reports(errors, warnings, debt)

    for w in warnings:
        print(f"  [warn] {w}")
    for e in errors:
        print(f"  [FAIL] {e}")

    n_md = len(markdown_files())
    print(f"doc check: {n_md} markdown files, {len(errors)} errors, "
          f"{len(warnings)} warnings, {len(debt)} waived by codex/known_debt.txt")

    if errors or (args.strict and warnings):
        print("doc check FAILED — fix the above, or record a deliberate exception in "
              "codex/known_debt.txt with a reason")
        return 1
    print("doc check ok")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
