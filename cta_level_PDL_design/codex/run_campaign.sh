#!/usr/bin/env bash
# run_campaign.sh — drive one rented-GPU campaign with Codex as the judgement layer.
#
# Division of labour is deliberate:
#
#   THE SCRIPT owns everything long-running and everything that must be reproducible:
#     preflight, smoke, ./run_session.sh, ./collect.sh, the gate branch, the doc checker.
#     Those are already specified by EXPERIMENT_PLAN.md and must not vary run to run.
#
#   CODEX owns only what needs judgement and prose:
#     declaring experiment coordinates (AGENTS.md §5), reading the gate verdict against its
#     caveats, writing reports in the §7 structure, and keeping EXPERIMENT_REPORT_INDEX.md
#     in sync.
#
# Keeping a two-hour GPU sweep inside an agent turn is how sessions get lost to a dropped
# connection. The sweep is a script; the agent is invoked before and after it.
#
# Properties, matching the driver contract in EXPERIMENT_PLAN.md §13.1:
#   - unattended        : no stage asks a person anything, including the §6 branch
#   - resumable         : codex/state/<stage>.done; re-running skips finished stages
#   - fail-soft         : a failed measurement stage is recorded and the campaign continues,
#                         EXCEPT self-check failures (preflight/smoke), which abort — burning
#                         GPU hours on a broken harness is the failure this exists to prevent
#   - every number on disk: nothing is produced only in an agent's context window
#
# Usage:
#   ./codex/run_campaign.sh                    # full campaign
#   ./codex/run_campaign.sh --fresh            # ignore previous stage markers
#   ./codex/run_campaign.sh <stage> [stage...] # run named stages only
#   DRY_RUN=1 ./codex/run_campaign.sh          # print what each stage would do
#
# Stages:
#   audit    codex  pre-run audit: machine role, harness admissibility, declared coordinates
#   smoke    script FAST=1 ./run_session.sh into bench/results_smoke — plumbing proof
#   measure  script ./run_session.sh into bench/${RESULTS} — Tier 0 + Tier 1p + gate
#   tier1    codex  write the Tier 1 multi-wave report and update the index
#   branch   script read gate.json, write codex/state/branch.json, decide what may run next
#   tier4    script+codex  Tier 4 LLM bracket, only if the branch allows it
#   wrapup   codex  umbrella campaign report, collect.sh, final doc check
#
# Environment:
#   RESULTS=results_<tag>   results directory under bench/ (default: results)
#   STEP_TIMEOUT=1800       per-benchmark-step timeout in seconds; see run_all.sh
#   CODEX_MODEL=...         override the model (default: whatever ~/.codex/config.toml says)
#   CODEX_SANDBOX=...       read-only | workspace-write | danger-full-access
#                           (default workspace-write; Tier 4 needs network for the model pull)
#   SKIP_CODEX=1            run only the script stages; useful to test the plumbing

set -uo pipefail

SELF_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "${SELF_DIR}/.." && pwd)"
cd "${ROOT}"

STATE="${SELF_DIR}/state"
PROMPTS="${SELF_DIR}/prompts"
mkdir -p "${STATE}"

RESULTS="${RESULTS:-results}"
STEP_TIMEOUT="${STEP_TIMEOUT:-1800}"
CODEX_SANDBOX="${CODEX_SANDBOX:-workspace-write}"
DRY_RUN="${DRY_RUN:-0}"
SKIP_CODEX="${SKIP_CODEX:-0}"
export STEP_TIMEOUT

CAMPAIGN_LOG="${STATE}/campaign.log"
START_EPOCH=$(date +%s)

say() {
    local now elapsed
    now=$(date +%s); elapsed=$((now - START_EPOCH))
    printf '[%02d:%02d:%02d] %s\n' $((elapsed/3600)) $((elapsed%3600/60)) $((elapsed%60)) "$*" \
        | tee -a "${CAMPAIGN_LOG}"
}
rule() { say "════════════════════════════════════════════════════════════════"; }
record_failure() { echo "$(date -Iseconds) $*" >> "${STATE}/failures.log"; say "FAILED: $*"; }
# failures.log accumulates across resumes on purpose, so the end-of-run summary reports only
# what this invocation added -- otherwise a clean resume looks like it failed again.
FAILURES_AT_START=$(wc -l < "${STATE}/failures.log" 2>/dev/null | tr -d ' ' || echo 0)
FAILURES_AT_START=${FAILURES_AT_START:-0}

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
    awk 'NR>1 && /^#/ { sub(/^# ?/, ""); print; next } NR>1 { exit }' "$0"
    exit 0
fi
if [ "${1:-}" = "--fresh" ]; then rm -f "${STATE}"/*.done "${STATE}/failures.log"; shift; fi

# ---------------------------------------------------------------- machine role
# AGENTS.md §3: detect, do not assume. The measurement stages are meaningless off a GPU box
# and the audit stage must know which set of rules it is under.
if command -v nvidia-smi >/dev/null 2>&1; then MACHINE="GPU_BOX"; else MACHINE="DEV_BOX"; fi
export CTA_MACHINE_ROLE="${MACHINE}"

say "campaign start $(date -Iseconds)  machine=${MACHINE}  RESULTS=${RESULTS}"

# ---------------------------------------------------------------- codex invocation
codex_available() { command -v codex >/dev/null 2>&1; }

# codex_stage <name> <prompt-file> — one agent turn, with the runtime context prepended so the
# prompt files stay static and reviewable. The doc checker runs afterwards: a stage that wrote
# a broken link or an unindexed report is not finished, so it does not get a .done marker.
codex_stage() {
    local name="$1" prompt="${PROMPTS}/$2"
    [ -f "${STATE}/${name}.done" ] && { say "skip ${name} (already done)"; return 0; }
    if [ "${SKIP_CODEX}" = "1" ]; then say "skip ${name} (SKIP_CODEX=1)"; return 0; fi
    if ! codex_available; then record_failure "${name}: codex CLI not found on PATH"; return 1; fi
    if [ ! -f "${prompt}" ]; then record_failure "${name}: missing prompt ${prompt}"; return 1; fi

    local -a args=(exec --skip-git-repo-check -C "${ROOT}" -s "${CODEX_SANDBOX}"
                   --output-last-message "${STATE}/${name}.out.md")
    [ -n "${CODEX_MODEL:-}" ] && args+=(-m "${CODEX_MODEL}")
    # Tier 4 pulls a model from the network; workspace-write blocks that by default.
    [ "${CODEX_SANDBOX}" = "workspace-write" ] && args+=(-c sandbox_workspace_write.network_access=true)

    say "run  ${name} (codex, sandbox=${CODEX_SANDBOX})"
    if [ "${DRY_RUN}" = "1" ]; then say "  DRY_RUN: codex ${args[*]} < ${prompt}"; return 0; fi

    {
        echo "# Runtime context (generated by codex/run_campaign.sh — do not edit)"
        echo
        echo "- subtree root : ${ROOT}"
        echo "- machine role : ${MACHINE}"
        echo "- results dir  : bench/${RESULTS}"
        echo "- stage        : ${name}"
        echo "- date         : $(date -Iseconds)"
        echo
        cat "${prompt}"
    } | codex "${args[@]}" - >> "${STATE}/${name}.log" 2>&1
    local rc=$?

    if [ ${rc} -ne 0 ]; then record_failure "${name}: codex exited ${rc} (see ${STATE}/${name}.log)"; return 1; fi
    if ! python3 "${SELF_DIR}/check_docs.py" >> "${STATE}/${name}.log" 2>&1; then
        record_failure "${name}: doc check failed after the stage (see ${STATE}/${name}.log)"
        return 1
    fi
    touch "${STATE}/${name}.done"
    say "ok   ${name}"
}

# script_stage <name> <command...> — same bookkeeping, no agent.
script_stage() {
    local name="$1"; shift
    [ -f "${STATE}/${name}.done" ] && { say "skip ${name} (already done)"; return 0; }
    say "run  ${name} ($*)"
    if [ "${DRY_RUN}" = "1" ]; then say "  DRY_RUN: $*"; return 0; fi
    if "$@" >> "${STATE}/${name}.log" 2>&1; then
        touch "${STATE}/${name}.done"; say "ok   ${name}"; return 0
    fi
    local rc=$?
    record_failure "${name}: exit ${rc} (see ${STATE}/${name}.log)"
    return ${rc}
}

require_gpu() {
    if [ "${MACHINE}" != "GPU_BOX" ]; then
        say "SKIP $1 — no nvidia-smi on this machine. AGENTS.md §3: never present a GPU number"
        say "     that is not traceable to a file under bench/results*/."
        return 1
    fi
    return 0
}

# ---------------------------------------------------------------- stages
stage_audit() {
    rule; say "STAGE audit — pre-run audit and declared coordinates"
    codex_stage audit 10_preflight_audit.md
}

stage_smoke() {
    rule; say "STAGE smoke — FAST=1 whole session, proves plumbing before spending hours"
    require_gpu smoke || return 0
    FAST=1 RESULTS=results_smoke script_stage smoke ./run_session.sh --fresh
}

stage_measure() {
    rule; say "STAGE measure — ./run_session.sh: Tier 0, Tier 1p multi-wave, gate"
    require_gpu measure || return 0
    # run_session.sh exit codes: 1 = self-check failed (nothing measured), 2 = no gate.
    RESULTS="${RESULTS}" script_stage measure ./run_session.sh
    local rc=$?
    if [ ${rc} -eq 1 ]; then
        say "ABORT: preflight or smoke failed inside run_session.sh. Nothing was measured."
        return 1
    fi
    return 0
}

stage_tier1() {
    rule; say "STAGE tier1 — write the Tier 1 multi-wave report, update the index"
    if [ ! -s "bench/${RESULTS}/gate.json" ]; then
        record_failure "tier1: bench/${RESULTS}/gate.json missing — nothing to report on"
        return 1
    fi
    codex_stage tier1 30_tier1_report.md
}

# The §6 branch, computed rather than eyeballed. gate.py already owns the thresholds; this
# only turns its verdict plus the §5.3 coverage flag into "which stage may run next".
stage_branch() {
    rule; say "STAGE branch — resolve the §6 verdict into the next admissible stage"
    local gate="bench/${RESULTS}/gate.json"
    if [ ! -s "${gate}" ]; then record_failure "branch: ${gate} missing"; return 1; fi
    python3 - "${gate}" "${STATE}/branch.json" <<'PY' | tee -a "${CAMPAIGN_LOG}"
import json, sys
gate = json.load(open(sys.argv[1]))
v = gate["verdict"]
multi = bool(gate.get("plan_multi_complete"))
allow_tier4 = v in ("GO", "LLM_ONLY", "STOP")   # plan §8: STOP still runs the three rungs
allow_tier23 = v == "GO" and multi
why = {
    "INVALID": "a configuration failed validation; no timing from this run is usable",
    "STOP":    "median space% < 2; only the Tier 4 three rungs, then stop",
    "LLM_ONLY":"median space% in 2..8; skip Tier 2/3, confirm on a real workload",
    "GO":      "median space% >= 8",
}[v]
if v == "INVALID":
    allow_tier4 = False
if v == "GO" and not multi:
    why += "; but plan §5.3's 2x/8x/32x SM set is incomplete, so Tier 2/3 stays closed"
out = {"verdict": v, "plan_multi_complete": multi, "allow_tier4": allow_tier4,
       "allow_tier23": allow_tier23, "reason": why,
       "median_space_pct": gate.get("median_space_pct")}
json.dump(out, open(sys.argv[2], "w"), indent=2, sort_keys=True)
print(f"verdict={v} multi_complete={multi} tier4={allow_tier4} tier23={allow_tier23}")
print(f"reason: {why}")
PY
    touch "${STATE}/branch.done"
}

stage_tier4() {
    rule; say "STAGE tier4 — LLM end-to-end: Ceiling − PDL_grid"
    local branch="${STATE}/branch.json"
    if [ ! -s "${branch}" ]; then say "skip tier4 — no branch decision yet"; return 0; fi
    if ! python3 -c "import json,sys;sys.exit(0 if json.load(open(sys.argv[1]))['allow_tier4'] else 1)" \
         "${branch}"; then
        say "skip tier4 — the §6 branch does not allow it"; return 0
    fi
    require_gpu tier4 || return 0
    script_stage tier4_sweep bash -c 'cd bench/llm && ./run_llm_sweep.sh'
    codex_stage tier4 50_tier4_report.md
}

stage_wrapup() {
    rule; say "STAGE wrapup — umbrella report, collect, final doc check"
    codex_stage wrapup 90_campaign_wrapup.md
    if [ "${MACHINE}" = "GPU_BOX" ] && [ "${DRY_RUN}" != "1" ]; then
        ./collect.sh >> "${STATE}/wrapup.log" 2>&1 || say "collect.sh reported a problem"
    fi
    # A rented box can disappear. CAMPAIGN_COMMIT=1 saves the prose so the session's
    # conclusions survive the machine. Markdown only: AGENTS.md §9 keeps binaries, tarballs
    # and bench/results*/ out of the repo, and the raw evidence travels in collect.sh's
    # tarball instead.
    if [ "${CAMPAIGN_COMMIT:-0}" = "1" ] && [ "${DRY_RUN}" != "1" ]; then
        git add -- '*.md' >> "${STATE}/wrapup.log" 2>&1
        if git diff --cached --quiet; then
            say "nothing to commit"
        elif git commit -m "docs: campaign results and index update ($(date +%Y-%m-%d))" \
             >> "${STATE}/wrapup.log" 2>&1; then
            say "committed markdown changes"
        else
            say "commit failed, see ${STATE}/wrapup.log"
        fi
    fi
}

# ---------------------------------------------------------------- driver
STAGES=("$@")
[ ${#STAGES[@]} -eq 0 ] && STAGES=(audit smoke measure tier1 branch tier4 wrapup)

for s in "${STAGES[@]}"; do
    case "${s}" in
        audit)   stage_audit ;;
        smoke)   stage_smoke ;;
        measure) stage_measure || break ;;
        tier1)   stage_tier1 ;;
        branch)  stage_branch ;;
        tier4)   stage_tier4 ;;
        wrapup)  stage_wrapup ;;
        *) say "unknown stage '${s}'"; exit 1 ;;
    esac
done

rule
ELAPSED=$(( $(date +%s) - START_EPOCH ))
say "campaign finished in $((ELAPSED/3600))h$((ELAPSED%3600/60))m"
NEW_FAILURES=$(tail -n "+$((FAILURES_AT_START + 1))" "${STATE}/failures.log" 2>/dev/null)
if [ -n "${NEW_FAILURES}" ]; then
    say "stages that failed in this invocation (recorded, campaign continued):"
    printf '%s\n' "${NEW_FAILURES}" | sed 's/^/    /' | tee -a "${CAMPAIGN_LOG}"
    exit 1
fi
say "no stage failures in this invocation"
exit 0
