#!/usr/bin/env python3
"""Fail-closed PTX, resource, NVTX, and real sm_100 cubin proof for Tier 5."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


WORKERS = ("dsaIndexer", "dsaTopk", "dsaAttention")
PTX_EXACT = {
    "dsaIndexer": {
        "launch_dependents": 2, "wait": 0, "membar_gl": 1,
        "acquire": 0, "release": 1,
    },
    "dsaTopk": {
        "launch_dependents": 2, "wait": 1, "membar_gl": 1,
        "acquire": 1, "release": 1,
    },
    "dsaAttention": {
        "launch_dependents": 2, "wait": 1, "membar_gl": 1,
        "acquire": 1, "release": 0, "global_load": 3,
    },
}
ATTENTION_HISTORY_STATIC_SITES = 1


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temp, path)
    finally:
        try:
            os.unlink(temp)
        except FileNotFoundError:
            pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def extract_ptx_sections(ptx: str, errors: list[str]) -> dict[str, str]:
    found: dict[str, list[str]] = {name: [] for name in WORKERS}
    entries = list(re.finditer(r"(?m)^\.entry\s+([^\s(]+)", ptx))
    for index, match in enumerate(entries):
        end = entries[index + 1].start() if index + 1 < len(entries) else len(ptx)
        mangled = match.group(1)
        for name in WORKERS:
            if name in mangled:
                found[name].append(ptx[match.start():end])
    sections: dict[str, str] = {}
    for name, matches in found.items():
        if len(matches) != 1:
            errors.append(f"PTX entries for {name}={len(matches)}, expected=1")
        if matches:
            sections[name] = matches[0]
    return sections


def ordered(section: str, tokens: list[str]) -> bool:
    cursor = -1
    for token in tokens:
        cursor = section.find(token, cursor + 1)
        if cursor < 0:
            return False
    return True


def regex_positions(section: str, pattern: str) -> list[int]:
    return [match.start() for match in re.finditer(pattern, section)]


_PTX_REGISTER = r"%(?:p|r|rd)\d+"
_PTX_LABEL = r"\$L__BB[A-Za-z0-9_]+"
# The forward-progress entry marker is emitted by a deliberately named inline-PTX
# predicate.  It is an auditable predicated atomic instruction, not a control-flow guard.
_PTX_INSTRUCTION_GUARD = r"(?:%p\d+|dsa_entry_thread0)"


def _parse_ptx_instruction_cfg(section: str) -> dict[str, Any]:
    """Parse the entry into an instruction CFG, rejecting unknown control flow.

    This deliberately accepts only the direct PTX control-flow forms emitted by the
    audited build.  A new or malformed branch form is evidence failure, not something
    to approximate with textual fallthrough.
    """
    instructions: list[dict[str, Any]] = []
    label_occurrences: dict[str, list[int]] = {}
    errors: list[str] = []
    offset = 0
    for line_number, raw_line in enumerate(section.splitlines(keepends=True), 1):
        code_without_comment = raw_line.split("//", 1)[0]
        stripped = code_without_comment.strip()
        position = offset + (len(code_without_comment) - len(code_without_comment.lstrip()))
        offset += len(raw_line)
        if not stripped or stripped in {"{", "}", "(", ")"}:
            continue
        label = re.fullmatch(rf"(?P<label>{_PTX_LABEL}):", stripped)
        if label:
            label_occurrences.setdefault(label.group("label"), []).append(position)
            continue
        # Entry signatures, declarations, line tables, and inline-asm declarations do
        # not alter control flow.  Executable PTX in this cubin is one statement/line.
        if stripped.startswith("."):
            continue
        if stripped.count(";") != 1 or not stripped.endswith(";"):
            errors.append(f"unparsed executable line {line_number}: {stripped}")
            continue
        parsed = re.fullmatch(
            rf"(?:(?P<guard>@!?{_PTX_INSTRUCTION_GUARD})\s+)?"
            r"(?P<opcode>[A-Za-z_$][A-Za-z0-9_.$]*)"
            r"(?:\s+(?P<operands>.*?))?;",
            stripped,
        )
        if not parsed:
            errors.append(f"malformed PTX instruction line {line_number}: {stripped}")
            continue
        instructions.append({
            "index": len(instructions),
            "line": line_number,
            "position": position,
            "text": stripped,
            "guard": parsed.group("guard"),
            "opcode": parsed.group("opcode"),
            "operands": parsed.group("operands") or "",
        })

    duplicate_labels = sorted(
        label for label, positions in label_occurrences.items() if len(positions) != 1
    )
    if duplicate_labels:
        errors.append("duplicate PTX labels: " + ",".join(duplicate_labels))

    label_nodes: dict[str, int] = {}
    for label, positions in label_occurrences.items():
        if len(positions) != 1:
            continue
        target = next(
            (item["index"] for item in instructions if item["position"] > positions[0]),
            None,
        )
        if target is None:
            errors.append(f"label has no executable target: {label}")
        else:
            label_nodes[label] = target

    branch_count = 0
    unsupported_control: list[str] = []
    successors: dict[int, set[int]] = {item["index"]: set() for item in instructions}
    for item in instructions:
        index = item["index"]
        text = item["text"]
        conditional = re.fullmatch(
            rf"@(?P<neg>!?)?(?P<pred>%p\d+)\s+bra\s+"
            rf"(?P<target>{_PTX_LABEL});",
            text,
        )
        unconditional = re.fullmatch(
            rf"bra(?:\.uni)?\s+(?P<target>{_PTX_LABEL});", text
        )
        if conditional:
            branch_count += 1
            item["branch_kind"] = "conditional"
            item["branch_negated"] = conditional.group("neg") == "!"
            item["branch_predicate"] = conditional.group("pred")
            item["branch_target"] = conditional.group("target")
            target_node = label_nodes.get(item["branch_target"])
            if target_node is None:
                errors.append(
                    f"unresolved conditional branch target at line {item['line']}: "
                    f"{item['branch_target']}"
                )
            else:
                successors[index].add(target_node)
                item["branch_target_node"] = target_node
            if index + 1 >= len(instructions):
                errors.append(f"conditional branch has no fallthrough at line {item['line']}")
            else:
                successors[index].add(index + 1)
                item["fallthrough_node"] = index + 1
        elif unconditional:
            branch_count += 1
            item["branch_kind"] = "unconditional"
            item["branch_target"] = unconditional.group("target")
            target_node = label_nodes.get(item["branch_target"])
            if target_node is None:
                errors.append(
                    f"unresolved unconditional branch target at line {item['line']}: "
                    f"{item['branch_target']}"
                )
            else:
                successors[index].add(target_node)
                item["branch_target_node"] = target_node
        elif item["opcode"].startswith(("bra", "brx")):
            unsupported_control.append(text)
            errors.append(f"unsupported or malformed branch at line {item['line']}: {text}")
        elif item["opcode"].startswith(("call", "vcall", "jmp")):
            unsupported_control.append(text)
            errors.append(f"unsupported control flow at line {item['line']}: {text}")
        elif item["opcode"] in {"ret", "exit", "trap", "brkpt"}:
            item["branch_kind"] = "terminator"
        elif index + 1 < len(instructions):
            successors[index].add(index + 1)
        else:
            errors.append(f"final instruction is not a terminator: {text}")

    node_labels: dict[int, list[str]] = {}
    for label, node in label_nodes.items():
        node_labels.setdefault(node, []).append(label)
    return {
        "instructions": instructions,
        "label_occurrences": label_occurrences,
        "label_nodes": label_nodes,
        "node_labels": node_labels,
        "successors": successors,
        "branch_count": branch_count,
        "duplicate_labels": duplicate_labels,
        "unsupported_control": unsupported_control,
        "errors": errors,
    }


def _reachable(successors: dict[int, set[int]], start: int | None) -> set[int]:
    if start is None or start not in successors:
        return set()
    seen: set[int] = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(successors[node] - seen)
    return seen


def _dominators(successors: dict[int, set[int]], start: int | None) -> dict[int, set[int]]:
    reachable = _reachable(successors, start)
    if start is None or not reachable:
        return {}
    predecessors = {node: set() for node in reachable}
    for node in reachable:
        for successor in successors[node] & reachable:
            predecessors[successor].add(node)
    dominators = {
        node: ({start} if node == start else set(reachable)) for node in reachable
    }
    changed = True
    while changed:
        changed = False
        for node in reachable - {start}:
            incoming = predecessors[node]
            updated = {node}
            if incoming:
                updated |= set.intersection(*(dominators[parent] for parent in incoming))
            if updated != dominators[node]:
                dominators[node] = updated
                changed = True
    return dominators


def _register_definition_nodes(
    instructions: list[dict[str, Any]], register: str | None,
) -> list[int]:
    if not register:
        return []
    no_destination_prefixes = (
        "st.", "sust.", "red.", "bra", "brx", "bar.", "membar.",
        "fence.", "griddepcontrol.", "nanosleep.", "ret", "exit", "trap",
        "brkpt", "call", "vcall", "jmp",
    )
    result = []
    for item in instructions:
        if item["opcode"].startswith(no_destination_prefixes):
            continue
        first_operand = item["operands"].split(",", 1)[0].strip()
        if first_operand == register:
            result.append(item["index"])
    return result


def _node_name(cfg: dict[str, Any], node: int | None) -> str | None:
    if node is None:
        return None
    labels = cfg["node_labels"].get(node, [])
    if labels:
        return "|".join(sorted(labels))
    instructions = cfg["instructions"]
    if 0 <= node < len(instructions):
        return f"line:{instructions[node]['line']}"
    return None


def attention_acquire_cfg_proof(
    section: str, acquire_position: int | None = None,
    history_load_position: int | None = None,
) -> dict[str, Any]:
    """Prove Impl's acquire/dataflow and CFG dominance for every history load.

    CUDA 13 places the acquire loop after the history loop textually.  This proof parses
    every direct branch plus fallthrough, rejects any unknown/ambiguous control flow, binds
    the selector/epoch/address registers to the kernel parameters, and computes dominance
    from the Impl selector target rather than relying on textual order.
    """
    cfg = _parse_ptx_instruction_cfg(section)
    instructions: list[dict[str, Any]] = cfg["instructions"]
    successors: dict[int, set[int]] = cfg["successors"]
    proof_errors = list(cfg["errors"])

    def fullmatch_nodes(pattern: str) -> list[tuple[int, re.Match[str]]]:
        result = []
        compiled = re.compile(pattern)
        for item in instructions:
            match = compiled.fullmatch(item["text"])
            if match:
                result.append((item["index"], match))
        return result

    def require_one(name: str, values: list[Any]) -> Any | None:
        if len(values) != 1:
            proof_errors.append(f"{name} count={len(values)}, expected=1")
            return None
        return values[0]

    def parameter_load(index: int, width: int) -> tuple[int, re.Match[str]] | None:
        return require_one(
            f"param_{index} u{width} load",
            fullmatch_nodes(
                rf"ld\.param\.u{width}\s+(?P<dst>{_PTX_REGISTER}),\s*"
                rf"\[[^\]]*_param_{index}\];"
            ),
        )

    mode_load = parameter_load(13, 32)
    mode_register = mode_load[1].group("dst") if mode_load else None
    selectors = fullmatch_nodes(
        rf"setp\.eq\.s32\s+(?P<pred>%p\d+),\s*"
        rf"{re.escape(mode_register) if mode_register else r'%rIMPOSSIBLE'},\s*1;"
    )
    selector_setp = require_one("Impl mode selector compare", selectors)
    selector_branch = None
    if selector_setp:
        selector_index, selector_match = selector_setp
        next_index = selector_index + 1
        if next_index < len(instructions):
            candidate = instructions[next_index]
            if (
                candidate.get("branch_kind") == "conditional"
                and not candidate.get("branch_negated")
                and candidate.get("branch_predicate") == selector_match.group("pred")
            ):
                selector_branch = candidate
        if selector_branch is None:
            proof_errors.append("Impl selector is not immediately followed by its positive branch")
    selector_entry = selector_branch.get("branch_target_node") if selector_branch else None
    tid_load = require_one(
        "tid.x load",
        fullmatch_nodes(r"mov\.u32\s+(?P<dst>%r\d+),\s*%tid\.x;"),
    )
    tid_register = tid_load[1].group("dst") if tid_load else None
    thread_gate_compare = None
    thread_gate_branch = None
    thread_gate_linear = False
    if selector_setp and selector_setp[0] > 0:
        candidate = instructions[selector_setp[0] - 1]
        if (
            candidate.get("branch_kind") == "conditional"
            and not candidate.get("branch_negated")
            and candidate.get("fallthrough_node") == selector_setp[0]
        ):
            thread_gate_branch = candidate
            gate_compares = fullmatch_nodes(
                rf"setp\.ne\.s32\s+{re.escape(candidate['branch_predicate'])},\s*"
                rf"{re.escape(tid_register or '%rIMPOSSIBLE')},\s*0;"
            )
            thread_gate_compare = require_one(
                "thread-nonzero gate compare", gate_compares
            )
            if thread_gate_compare and thread_gate_compare[0] < candidate["index"]:
                thread_gate_linear = all(
                    successors[node] == {node + 1}
                    for node in range(thread_gate_compare[0], candidate["index"])
                )
    if thread_gate_branch is None:
        proof_errors.append(
            "Impl selector is not reached only by thread-zero gate fallthrough"
        )
    if thread_gate_compare is None:
        proof_errors.append("thread-nonzero gate is not bound to tid.x")
    if not thread_gate_linear:
        proof_errors.append("thread-nonzero compare-to-branch path is not linear")
    nonzero_compare = None
    nonzero_branch = None
    if selector_branch:
        compare_index = selector_branch["index"] + 1
        branch_index = compare_index + 1
        if branch_index < len(instructions):
            compare_candidate = instructions[compare_index]
            compare_match = re.fullmatch(
                rf"setp\.ne\.s32\s+(?P<pred>%p\d+),\s*"
                rf"{re.escape(mode_register or '%rIMPOSSIBLE')},\s*0;",
                compare_candidate["text"],
            )
            branch_candidate = instructions[branch_index]
            if compare_match:
                nonzero_compare = compare_candidate
                if (
                    branch_candidate.get("branch_kind") == "conditional"
                    and not branch_candidate.get("branch_negated")
                    and branch_candidate.get("branch_predicate")
                    == compare_match.group("pred")
                ):
                    nonzero_branch = branch_candidate
        if nonzero_compare is None:
            proof_errors.append(
                "Impl fallthrough is not immediately followed by mode!=0 compare"
            )
        if nonzero_branch is None:
            proof_errors.append(
                "mode!=0 compare is not immediately followed by its positive branch"
            )

    epoch_pointer_load = parameter_load(6, 64)
    epoch_pointer_register = (
        epoch_pointer_load[1].group("dst") if epoch_pointer_load else None
    )
    epoch_address = require_one(
        "epoch pointer cvta",
        fullmatch_nodes(
            rf"cvta\.to\.global\.u64\s+(?P<dst>%rd\d+),\s*"
            rf"{re.escape(epoch_pointer_register) if epoch_pointer_register else r'%rdIMPOSSIBLE'};"
        ),
    )
    epoch_address_register = epoch_address[1].group("dst") if epoch_address else None
    epoch_load = require_one(
        "epoch value load",
        fullmatch_nodes(
            rf"ld\.global(?:\.[A-Za-z0-9_]+)*\.u32\s+(?P<dst>%r\d+),\s*"
            rf"\[{re.escape(epoch_address_register) if epoch_address_register else r'%rdIMPOSSIBLE'}\];"
        ),
    )
    epoch_register = epoch_load[1].group("dst") if epoch_load else None

    topk_pointer_load = parameter_load(3, 64)
    topk_pointer_register = (
        topk_pointer_load[1].group("dst") if topk_pointer_load else None
    )
    ctaid_load = require_one(
        "ctaid.x load",
        fullmatch_nodes(r"mov\.u32\s+(?P<dst>%r\d+),\s*%ctaid\.x;"),
    )
    ctaid_register = ctaid_load[1].group("dst") if ctaid_load else None
    query_base_load = parameter_load(16, 32)
    query_base_register = (
        query_base_load[1].group("dst") if query_base_load else None
    )
    query_definitions = fullmatch_nodes(
        rf"add\.s32\s+(?P<dst>%r\d+),\s*"
        rf"(?:{re.escape(query_base_register or '%rIMPOSSIBLE')},\s*"
        rf"{re.escape(ctaid_register or '%rIMPOSSIBLE')}|"
        rf"{re.escape(ctaid_register or '%rIMPOSSIBLE')},\s*"
        rf"{re.escape(query_base_register or '%rIMPOSSIBLE')});"
    )
    query_definition = require_one("global query base+ctaid", query_definitions)
    query_register = query_definition[1].group("dst") if query_definition else None
    flag_offset = require_one(
        "topk flag global-query offset",
        fullmatch_nodes(
            rf"mul\.wide\.(?:s|u)32\s+(?P<dst>%rd\d+),\s*"
            rf"{re.escape(query_register) if query_register else r'%rIMPOSSIBLE'},\s*4;"
        ),
    )
    flag_offset_register = flag_offset[1].group("dst") if flag_offset else None
    flag_address = require_one(
        "topk flag address",
        fullmatch_nodes(
            rf"add\.s64\s+(?P<dst>%rd\d+),\s*"
            rf"{re.escape(topk_pointer_register) if topk_pointer_register else r'%rdIMPOSSIBLE'},\s*"
            rf"{re.escape(flag_offset_register) if flag_offset_register else r'%rdIMPOSSIBLE'};"
        ),
    )
    flag_address_register = flag_address[1].group("dst") if flag_address else None
    acquire = require_one(
        "bound acquire load",
        fullmatch_nodes(
            rf"ld\.acquire\.gpu\.b32\s+(?P<dst>%r\d+),\s*"
            rf"\[{re.escape(flag_address_register) if flag_address_register else r'%rdIMPOSSIBLE'}\];"
        ),
    )
    acquire_node = acquire[0] if acquire else None
    acquire_register = acquire[1].group("dst") if acquire else None

    bound_register_nodes = [
        ("mode", mode_register, mode_load[0] if mode_load else None),
        ("epoch_pointer", epoch_pointer_register,
         epoch_pointer_load[0] if epoch_pointer_load else None),
        ("epoch_address", epoch_address_register, epoch_address[0] if epoch_address else None),
        ("epoch", epoch_register, epoch_load[0] if epoch_load else None),
        ("topk_pointer", topk_pointer_register,
         topk_pointer_load[0] if topk_pointer_load else None),
        ("ctaid", ctaid_register, ctaid_load[0] if ctaid_load else None),
        ("query_base", query_base_register,
         query_base_load[0] if query_base_load else None),
        ("global_query", query_register,
         query_definition[0] if query_definition else None),
        ("tid", tid_register, tid_load[0] if tid_load else None),
        ("flag_offset", flag_offset_register, flag_offset[0] if flag_offset else None),
        ("flag_address", flag_address_register, flag_address[0] if flag_address else None),
        ("acquire_value", acquire_register, acquire_node),
    ]
    unique_register_definitions = True
    for name, register, expected_node in bound_register_nodes:
        definitions = _register_definition_nodes(instructions, register)
        if register is None or definitions != [expected_node]:
            unique_register_definitions = False
            proof_errors.append(
                f"{name} register definition mismatch: register={register} "
                f"definitions={definitions} expected={[expected_node]}"
            )

    success_compare = None
    success_branch = None
    if acquire_node is not None and acquire_node + 2 < len(instructions):
        compare_candidate = instructions[acquire_node + 1]
        compare_match = re.fullmatch(
            rf"setp\.eq\.s32\s+(?P<pred>%p\d+),\s*"
            rf"{re.escape(acquire_register or '%rIMPOSSIBLE')},\s*"
            rf"{re.escape(epoch_register or '%rIMPOSSIBLE')};",
            compare_candidate["text"],
        )
        branch_candidate = instructions[acquire_node + 2]
        if compare_match:
            success_compare = compare_candidate
            if (
                branch_candidate.get("branch_kind") == "conditional"
                and not branch_candidate.get("branch_negated")
                and branch_candidate.get("branch_predicate") == compare_match.group("pred")
            ):
                success_branch = branch_candidate
        if success_compare is None:
            proof_errors.append("acquire value is not compared directly with loaded epoch")
        if success_branch is None:
            proof_errors.append("acquire equality is not immediately followed by a positive branch")
    else:
        proof_errors.append("acquire success sequence is incomplete")

    marker_positions = [
        match.start() for match in re.finditer(
            r"(?m)^\s*\.reg\s+\.u32\s+dsa_history_loaded_value\s*;", section
        )
    ]
    count_marker_positions = [
        match.start() for match in re.finditer(
            r"(?m)^\s*\.reg\s+\.u64\s+dsa_history_load_count\s*;", section
        )
    ]
    semantic_marker_positions = [
        match.start() for match in re.finditer(
            r"(?m)^\s*\.reg\s+\.u32\s+dsa_semantic_index\s*;", section
        )
    ]
    count_dependency_marker_positions = [
        match.start() for match in re.finditer(
            r"(?m)^\s*\.reg\s+\.u32\s+dsa_history_count_dependency\s*;",
            section,
        )
    ]
    semantic_dependency_marker_positions = [
        match.start() for match in re.finditer(
            r"(?m)^\s*\.reg\s+\.u32\s+dsa_history_dependency\s*;",
            section,
        )
    ]
    count_add_nodes = [
        item["index"] for item in instructions
        if re.fullmatch(
            r"add\.u64\s+dsa_history_load_count,\s*"
            r"dsa_history_load_count,\s*1;",
            item["text"],
        )
    ]
    history_site_cardinality = all(
        count == ATTENTION_HISTORY_STATIC_SITES
        for count in (
            len(marker_positions), len(count_marker_positions),
            len(count_dependency_marker_positions), len(count_add_nodes),
            len(semantic_marker_positions), len(semantic_dependency_marker_positions),
        )
    )
    if not history_site_cardinality:
        proof_errors.append(
            "history static-site cardinality mismatch: "
            f"loaded={len(marker_positions)} count_decl={len(count_marker_positions)} "
            f"count_dependency={len(count_dependency_marker_positions)} "
            f"count_add={len(count_add_nodes)} semantic={len(semantic_marker_positions)} "
            f"semantic_dependency={len(semantic_dependency_marker_positions)} "
            f"expected={ATTENTION_HISTORY_STATIC_SITES}"
        )

    history_pointer_load = parameter_load(1, 64)
    history_pointer_register = (
        history_pointer_load[1].group("dst") if history_pointer_load else None
    )
    history_base = require_one(
        "history pointer cvta",
        fullmatch_nodes(
            rf"cvta\.to\.global\.u64\s+(?P<dst>%rd\d+),\s*"
            rf"{re.escape(history_pointer_register or '%rdIMPOSSIBLE')};"
        ),
    )
    history_base_register = history_base[1].group("dst") if history_base else None
    indices_pointer_load = parameter_load(0, 64)
    indices_pointer_register = (
        indices_pointer_load[1].group("dst") if indices_pointer_load else None
    )
    indices_base = require_one(
        "indices pointer cvta",
        fullmatch_nodes(
            rf"cvta\.to\.global\.u64\s+(?P<dst>%rd\d+),\s*"
            rf"{re.escape(indices_pointer_register or '%rdIMPOSSIBLE')};"
        ),
    )
    indices_base_register = indices_base[1].group("dst") if indices_base else None
    key_tiles_load = parameter_load(10, 32)
    key_tiles_register = key_tiles_load[1].group("dst") if key_tiles_load else None
    topk_load = parameter_load(12, 32)
    topk_register = topk_load[1].group("dst") if topk_load else None
    query_widen = require_one(
        "global query widen",
        fullmatch_nodes(
            rf"cvt\.(?:s64\.s32|u64\.u32)\s+(?P<dst>%rd\d+),\s*"
            rf"{re.escape(query_register or '%rIMPOSSIBLE')};"
        ),
    )
    rank_widen = require_one(
        "initial rank widen",
        [
            item for item in fullmatch_nodes(
                rf"cvt\.u64\.u32\s+(?P<dst>%rd\d+),\s*"
                rf"{re.escape(tid_register or '%rIMPOSSIBLE')};"
            )
            if marker_positions and instructions[item[0]]["position"] < marker_positions[0]
        ],
    )
    topk_widen = require_one(
        "topk widen",
        fullmatch_nodes(
            rf"cvt\.s64\.s32\s+(?P<dst>%rd\d+),\s*"
            rf"{re.escape(topk_register or '%rIMPOSSIBLE')};"
        ),
    )
    topk_wide_register = topk_widen[1].group("dst") if topk_widen else None
    query_wide_register = query_widen[1].group("dst") if query_widen else None
    rank_wide_register = rank_widen[1].group("dst") if rank_widen else None
    initial_linear_candidates = []
    for left, right in (
        (topk_wide_register, query_wide_register),
        (query_wide_register, topk_wide_register),
    ):
        initial_linear_candidates.extend(fullmatch_nodes(
            rf"mad\.lo\.s64\s+(?P<dst>%rd\d+),\s*"
            rf"{re.escape(left or '%rdIMPOSSIBLE')},\s*"
            rf"{re.escape(right or '%rdIMPOSSIBLE')},\s*"
            rf"{re.escape(rank_wide_register or '%rdIMPOSSIBLE')};"
        ))
    initial_linear_index = require_one(
        "initial global-query row-plus-rank index", initial_linear_candidates
    )
    initial_index_byte_offset = require_one(
        "initial index byte offset",
        fullmatch_nodes(
            rf"shl\.b64\s+(?P<dst>%rd\d+),\s*"
            rf"{re.escape(initial_linear_index[1].group('dst') if initial_linear_index else '%rdIMPOSSIBLE')},\s*2;"
        ),
    )
    initial_index_address = require_one(
        "initial indices address",
        fullmatch_nodes(
            rf"add\.s64\s+(?P<dst>%rd\d+),\s*"
            rf"{re.escape(indices_base_register or '%rdIMPOSSIBLE')},\s*"
            rf"{re.escape(initial_index_byte_offset[1].group('dst') if initial_index_byte_offset else '%rdIMPOSSIBLE')};"
        ),
    )
    index_address_register = (
        initial_index_address[1].group("dst") if initial_index_address else None
    )
    index_load = require_one(
        "rank index load",
        fullmatch_nodes(
            rf"ld\.global(?:\.[A-Za-z0-9_]+)*\.u32\s+(?P<dst>%r\d+),\s*"
            rf"\[{re.escape(index_address_register or '%rdIMPOSSIBLE')}\];"
        ),
    )
    index_register = index_load[1].group("dst") if index_load else None
    valid_compare = require_one(
        "index/key_tiles validity compare",
        fullmatch_nodes(
            rf"setp\.lt\.u32\s+(?P<pred>%p\d+),\s*"
            rf"{re.escape(index_register or '%rIMPOSSIBLE')},\s*"
            rf"{re.escape(key_tiles_register or '%rIMPOSSIBLE')};"
        ),
    )
    valid_predicate = valid_compare[1].group("pred") if valid_compare else None
    fallback_modulo = require_one(
        "fallback modulo key_tiles",
        fullmatch_nodes(
            rf"rem\.u32\s+(?P<fallback>%r\d+),\s*%r\d+,\s*"
            rf"{re.escape(key_tiles_register or '%rIMPOSSIBLE')};"
        ),
    )
    fallback_register = (
        fallback_modulo[1].group("fallback") if fallback_modulo else None
    )
    safe_index_definition = require_one(
        "safe-index select",
        fullmatch_nodes(
            rf"selp\.b32\s+(?P<safe>%r\d+),\s*"
            rf"{re.escape(index_register or '%rIMPOSSIBLE')},\s*"
            rf"{re.escape(fallback_register or '%rIMPOSSIBLE')},\s*"
            rf"{re.escape(valid_predicate or '%pIMPOSSIBLE')};"
        ),
    )
    safe_index_register = (
        safe_index_definition[1].group("safe") if safe_index_definition else None
    )
    history_address_chains: list[dict[str, Any]] = []
    for offset_node, offset_match in fullmatch_nodes(
        r"mul\.wide\.u32\s+(?P<offset>%rd\d+),\s*"
        r"(?P<safe>%r\d+),\s*4;"
    ):
        offset_register = offset_match.group("offset")
        safe_register = offset_match.group("safe")
        if safe_register != safe_index_register:
            continue
        for address_node, address_match in fullmatch_nodes(
            rf"add\.s64\s+(?P<address>%rd\d+),\s*"
            rf"{re.escape(history_base_register or '%rdIMPOSSIBLE')},\s*"
            rf"{re.escape(offset_register)};"
        ):
            address_register = address_match.group("address")
            for load_node, load_match in fullmatch_nodes(
                rf"ld\.global(?:\.[A-Za-z0-9_]+)*\.u32\s+"
                rf"(?P<loaded>%r\d+),\s*\[{re.escape(address_register)}\];"
            ):
                history_address_chains.append({
                    "safe_register": safe_register,
                    "offset_register": offset_register,
                    "offset_node": offset_node,
                    "address_register": address_register,
                    "address_node": address_node,
                    "load_register": load_match.group("loaded"),
                    "load_node": load_node,
                })
    history_address_chain = require_one(
        "history param/address/load chain", history_address_chains
    )
    history_load_node = (
        history_address_chain["load_node"] if history_address_chain else None
    )
    history_load_register = (
        history_address_chain["load_register"] if history_address_chain else None
    )

    def instruction_match(index: int | None, pattern: str) -> re.Match[str] | None:
        if index is None or not 0 <= index < len(instructions):
            return None
        return re.fullmatch(pattern, instructions[index]["text"])

    loaded_marker_source = instruction_match(
        history_load_node + 1 if history_load_node is not None else None,
        rf"mov\.u32\s+dsa_history_loaded_value,\s*"
        rf"{re.escape(history_load_register or '%rIMPOSSIBLE')};",
    )
    loaded_marker_output = instruction_match(
        history_load_node + 2 if history_load_node is not None else None,
        r"mov\.u32\s+(?P<loaded_value>%r\d+),\s*dsa_history_loaded_value;",
    )
    loaded_value_register = (
        loaded_marker_output.group("loaded_value") if loaded_marker_output else None
    )
    if loaded_marker_source is None or loaded_marker_output is None:
        proof_errors.append("history load result is not bound through loaded-value marker")

    count_dependency = instruction_match(
        history_load_node + 3 if history_load_node is not None else None,
        rf"mov\.u32\s+dsa_history_count_dependency,\s*"
        rf"{re.escape(loaded_value_register or '%rIMPOSSIBLE')};",
    )
    count_input = instruction_match(
        history_load_node + 4 if history_load_node is not None else None,
        r"mov\.u64\s+dsa_history_load_count,\s*(?P<count>%rd\d+);",
    )
    count_increment = instruction_match(
        history_load_node + 5 if history_load_node is not None else None,
        r"add\.u64\s+dsa_history_load_count,\s*"
        r"dsa_history_load_count,\s*1;",
    )
    count_output = instruction_match(
        history_load_node + 6 if history_load_node is not None else None,
        rf"mov\.u64\s*"
        rf"{re.escape(count_input.group('count') if count_input else '%rdIMPOSSIBLE')},\s*"
        rf"dsa_history_load_count;",
    )
    count_chain_bound = all(
        item is not None
        for item in (count_dependency, count_input, count_increment, count_output)
    )
    if not count_chain_bound:
        proof_errors.append("history loaded value is not bound to exact count increment chain")

    semantic_dependencies = fullmatch_nodes(
        rf"mov\.u32\s+dsa_history_dependency,\s*"
        rf"{re.escape(loaded_value_register or '%rIMPOSSIBLE')};"
    )
    semantic_dependency = require_one(
        "history-to-semantic dependency", semantic_dependencies
    )
    semantic_candidate_marker = None
    semantic_output_marker = None
    semantic_candidate_register = None
    semantic_output_register = None
    if semantic_dependency:
        semantic_candidate_marker = instruction_match(
            semantic_dependency[0] + 1,
            r"mov\.u32\s+dsa_semantic_index,\s*(?P<candidate>%r\d+);",
        )
        semantic_candidate_register = (
            semantic_candidate_marker.group("candidate")
            if semantic_candidate_marker else None
        )
        semantic_output_marker = instruction_match(
            semantic_dependency[0] + 2,
            r"mov\.u32\s+(?P<semantic>%r\d+),\s*dsa_semantic_index;",
        )
        semantic_output_register = (
            semantic_output_marker.group("semantic")
            if semantic_output_marker else None
        )
    if semantic_candidate_marker is None or semantic_output_marker is None:
        proof_errors.append("semantic candidate/output marker chain is incomplete")

    semantic_candidate_definition = None
    semantic_invalid_definition = None
    semantic_invalid_register = None
    if semantic_dependency and semantic_candidate_register:
        candidate_node = semantic_dependency[0] - 1
        invalid_match = instruction_match(
            candidate_node - 1,
            rf"add\.s32\s+(?P<invalid>%r\d+),\s*"
            rf"{re.escape(fallback_register or '%rIMPOSSIBLE')},\s*"
            rf"{re.escape(key_tiles_register or '%rIMPOSSIBLE')};",
        )
        if invalid_match:
            semantic_invalid_register = invalid_match.group("invalid")
            candidate_match = instruction_match(
                candidate_node,
                rf"selp\.b32\s+{re.escape(semantic_candidate_register)},\s*"
                rf"{re.escape(index_register or '%rIMPOSSIBLE')},\s*"
                rf"{re.escape(semantic_invalid_register)},\s*"
                rf"{re.escape(valid_predicate or '%pIMPOSSIBLE')};",
            )
            if candidate_match:
                semantic_invalid_definition = candidate_node - 1
                semantic_candidate_definition = candidate_node
    if semantic_candidate_definition is None:
        proof_errors.append(
            "semantic marker input is not the index/fallback/key_tiles candidate"
        )

    contribution_bound = False
    contribution_end_node = None
    if semantic_dependency and semantic_output_register and loaded_value_register:
        output_node = semantic_dependency[0] + 2
        value_widen = instruction_match(
            output_node + 1,
            rf"cvt\.u64\.u32\s+(?P<wide_value>%rd\d+),\s*"
            rf"{re.escape(loaded_value_register)};",
        )
        semantic_scale = instruction_match(
            output_node + 2,
            rf"mul\.wide\.u32\s+(?P<wide_semantic>%rd\d+),\s*"
            rf"{re.escape(semantic_output_register)},\s*65537;",
        )
        partial_add = None
        final_add = None
        if value_widen and semantic_scale:
            partial_add = instruction_match(
                output_node + 3,
                rf"add\.s64\s+(?P<partial>%rd\d+),\s*(?P<acc>%rd\d+),\s*"
                rf"{re.escape(value_widen.group('wide_value'))};",
            )
            if partial_add:
                final_add = instruction_match(
                    output_node + 4,
                    rf"add\.s64\s*{re.escape(partial_add.group('acc'))},\s*"
                    rf"{re.escape(partial_add.group('partial'))},\s*"
                    rf"{re.escape(semantic_scale.group('wide_semantic'))};",
                )
        contribution_bound = all(
            item is not None
            for item in (value_widen, semantic_scale, partial_add, final_add)
        )
        if contribution_bound:
            contribution_end_node = output_node + 4
    if not contribution_bound:
        proof_errors.append("loaded value and semantic index are not bound to contribution")

    rank_loop_bound = False
    rank_loop_register = None
    rank_loop_branch = None
    index_address_update_node = None
    rank_update_node = None
    initial_rank = None
    ntid_definitions: list[tuple[int, re.Match[str]]] = []
    ntid_register = None
    stride = None
    loop_compare = None
    if contribution_end_node is not None:
        rank_update = instruction_match(
            contribution_end_node + 1,
            r"add\.s32\s+(?P<rank>%r\d+),\s*(?P=rank),\s*"
            r"(?P<ntid>%r\d+);",
        )
        if rank_update:
            rank_loop_register = rank_update.group("rank")
            ntid_register = rank_update.group("ntid")
            rank_update_node = contribution_end_node + 1
            ntid_definitions = fullmatch_nodes(
                rf"mov\.u32\s+{re.escape(ntid_register)},\s*%ntid\.x;"
            )
            initial_rank = require_one(
                "initial dynamic rank",
                fullmatch_nodes(
                    rf"mov\.u32\s+{re.escape(rank_loop_register)},\s*"
                    rf"{re.escape(tid_register or '%rIMPOSSIBLE')};"
                ),
            )
            stride = instruction_match(
                contribution_end_node + 2,
                rf"mul\.wide\.u32\s+(?P<stride>%rd\d+),\s*"
                rf"{re.escape(ntid_register)},\s*4;",
            )
            address_update = None
            loop_compare = None
            loop_branch = None
            if stride:
                address_update = instruction_match(
                    contribution_end_node + 3,
                    rf"add\.s64\s*{re.escape(index_address_register or '%rdIMPOSSIBLE')},\s*"
                    rf"{re.escape(index_address_register or '%rdIMPOSSIBLE')},\s*"
                    rf"{re.escape(stride.group('stride'))};",
                )
            if address_update:
                index_address_update_node = contribution_end_node + 3
                loop_compare = instruction_match(
                    contribution_end_node + 4,
                    rf"setp\.lt\.s32\s+(?P<pred>%p\d+),\s*"
                    rf"{re.escape(rank_loop_register)},\s*"
                    rf"{re.escape(topk_register or '%rIMPOSSIBLE')};",
                )
            if loop_compare and contribution_end_node + 5 < len(instructions):
                candidate = instructions[contribution_end_node + 5]
                if (
                    candidate.get("branch_kind") == "conditional"
                    and not candidate.get("branch_negated")
                    and candidate.get("branch_predicate") == loop_compare.group("pred")
                    and candidate.get("branch_target_node")
                    == (index_load[0] if index_load else None)
                ):
                    loop_branch = candidate
                    rank_loop_branch = candidate
            rank_loop_bound = bool(
                len(ntid_definitions) == 1
                and initial_rank
                and stride
                and address_update
                and loop_compare
                and loop_branch
            )
    if not rank_loop_bound:
        proof_errors.append(
            "dynamic rank/index address stride and loop backedge are not bound"
        )

    index_to_contribution_branch_free = bool(
        index_load
        and contribution_end_node is not None
        and index_load[0] < contribution_end_node
        and not any(
            item.get("branch_kind") in {
                "conditional", "unconditional", "terminator"
            }
            for item in instructions[index_load[0]:contribution_end_node + 1]
        )
    )
    if not index_to_contribution_branch_free:
        proof_errors.append("index load-to-history contribution path contains control flow")

    history_dataflow_unique_definitions = False
    if history_address_chain and history_pointer_load and history_base:
        safe_register = history_address_chain["safe_register"]
        safe_definitions = _register_definition_nodes(instructions, safe_register)
        expected_definitions = [
            (history_pointer_register, history_pointer_load[0]),
            (history_base_register, history_base[0]),
            (safe_register, safe_definitions[0] if len(safe_definitions) == 1 else None),
            (history_address_chain["offset_register"],
             history_address_chain["offset_node"]),
            (history_address_chain["address_register"],
             history_address_chain["address_node"]),
            (history_load_register, history_load_node),
            (loaded_value_register,
             history_load_node + 2 if history_load_node is not None else None),
            (semantic_output_register,
             semantic_dependency[0] + 2 if semantic_dependency else None),
        ]
        history_dataflow_unique_definitions = all(
            register is not None
            and expected is not None
            and _register_definition_nodes(instructions, register) == [expected]
            for register, expected in expected_definitions
        )
        history_dataflow_unique_definitions = bool(
            history_dataflow_unique_definitions
            and len(safe_definitions) == 1
            and instructions[safe_definitions[0]]["opcode"] == "selp.b32"
        )
    if not history_dataflow_unique_definitions:
        proof_errors.append("history address/value/semantic registers are not single-definition")

    index_semantic_unique_definitions = False
    if all(
        item is not None
        for item in (
            indices_pointer_load, indices_base, key_tiles_load, topk_load,
            query_widen, rank_widen, topk_widen, initial_linear_index,
            initial_index_byte_offset, initial_index_address, index_load,
            valid_compare, fallback_modulo, safe_index_definition,
            semantic_invalid_definition, semantic_candidate_definition,
            initial_rank, stride, loop_compare, rank_loop_branch,
        )
    ):
        simple_definition_pairs = [
            (indices_pointer_register, indices_pointer_load[0]),
            (indices_base_register, indices_base[0]),
            (key_tiles_register, key_tiles_load[0]),
            (topk_register, topk_load[0]),
            (query_widen[1].group("dst"), query_widen[0]),
            (rank_widen[1].group("dst"), rank_widen[0]),
            (topk_widen[1].group("dst"), topk_widen[0]),
            (initial_linear_index[1].group("dst"), initial_linear_index[0]),
            (initial_index_byte_offset[1].group("dst"),
             initial_index_byte_offset[0]),
            (index_register, index_load[0]),
            (valid_predicate, valid_compare[0]),
            (fallback_register, fallback_modulo[0]),
            (safe_index_register, safe_index_definition[0]),
            (semantic_invalid_register, semantic_invalid_definition),
            (semantic_candidate_register, semantic_candidate_definition),
            (stride.group("stride"), contribution_end_node + 2),
            (rank_loop_branch.get("branch_predicate"),
             contribution_end_node + 4),
        ]
        simple_definitions_ok = all(
            register is not None
            and _register_definition_nodes(instructions, register) == [expected]
            for register, expected in simple_definition_pairs
        )
        rank_definitions_ok = bool(
            rank_loop_register
            and initial_rank
            and rank_update_node is not None
            and _register_definition_nodes(instructions, rank_loop_register)
            == [initial_rank[0], rank_update_node]
        )
        address_definitions_ok = bool(
            index_address_register
            and initial_index_address
            and index_address_update_node is not None
            and _register_definition_nodes(instructions, index_address_register)
            == [initial_index_address[0], index_address_update_node]
        )
        ntid_definitions_ok = bool(
            len(ntid_definitions) == 1
            and rank_loop_register
            and instruction_match(
                rank_update_node,
                rf"add\.s32\s+{re.escape(rank_loop_register)},\s*"
                rf"{re.escape(rank_loop_register)},\s*"
                rf"{re.escape(ntid_register or '%rIMPOSSIBLE')};",
            )
        )
        index_semantic_unique_definitions = bool(
            simple_definitions_ok and rank_definitions_ok
            and address_definitions_ok and ntid_definitions_ok
        )
    if not index_semantic_unique_definitions:
        proof_errors.append(
            "indices/rank/safe-index/semantic registers are not single-definition"
        )
    history_nodes: list[int] = []
    for marker_position in marker_positions:
        preceding = [
            item for item in instructions if item["position"] < marker_position
        ]
        if not preceding or not re.fullmatch(
            r"ld\.global(?:\.[A-Za-z0-9_]+)*\.u32\s+%r\d+,\s*\[[^\]]+\];",
            preceding[-1]["text"],
        ):
            proof_errors.append(
                "history marker is not immediately preceded by a global u32 load"
            )
        else:
            history_nodes.append(preceding[-1]["index"])
    if not marker_positions:
        proof_errors.append("no history-load marker declarations found")
    if len(set(history_nodes)) != len(marker_positions):
        proof_errors.append("history-load marker/load mapping is not one-to-one")
    history_nodes = sorted(set(history_nodes))
    history_marker_matches_bound_load = bool(
        history_load_node is not None and history_nodes == [history_load_node]
    )
    if not history_marker_matches_bound_load:
        proof_errors.append("history marker does not identify the bound param_1 load")
    history_sites_branch_free = False
    if history_site_cardinality and len(history_nodes) == ATTENTION_HISTORY_STATIC_SITES:
        history_sites_branch_free = True
        for load_node, loaded_position, count_position, add_node, semantic_position in zip(
            history_nodes, marker_positions, count_marker_positions,
            count_add_nodes, semantic_marker_positions,
        ):
            load_position = instructions[load_node]["position"]
            add_position = instructions[add_node]["position"]
            ordered_site = (
                load_position < loaded_position < count_position
                < add_position < semantic_position
            )
            branch_in_site = any(
                item.get("branch_kind") in {"conditional", "unconditional", "terminator"}
                and load_position < item["position"] < semantic_position
                for item in instructions
            )
            if not ordered_site or branch_in_site:
                history_sites_branch_free = False
                break
    if not history_sites_branch_free:
        proof_errors.append(
            "not every history static site is a branch-free load/count-add/semantic chain"
        )
    history_full_chain_branch_free = bool(
        history_load_node is not None
        and contribution_end_node is not None
        and history_load_node < contribution_end_node
        and not any(
            item.get("branch_kind") in {
                "conditional", "unconditional", "terminator"
            }
            for item in instructions[history_load_node:contribution_end_node + 1]
        )
    )
    if not history_full_chain_branch_free:
        proof_errors.append("history load-to-contribution chain contains control flow")

    waits = [
        item["index"] for item in instructions
        if item["text"] == "griddepcontrol.wait;"
    ]
    wait_node = require_one("griddepcontrol.wait CFG node", waits)
    floor_join_node = wait_node + 1 if wait_node is not None else None
    if floor_join_node is not None and floor_join_node >= len(instructions):
        proof_errors.append("griddepcontrol.wait has no fallthrough join")
        floor_join_node = None
    success_target_node = (
        success_branch.get("branch_target_node") if success_branch else None
    )
    success_targets_floor_join = bool(
        success_target_node is not None and success_target_node == floor_join_node
    )
    if not success_targets_floor_join:
        proof_errors.append("acquire success does not target floor-wait fallthrough join")
    nonzero_targets_floor_join = bool(
        nonzero_branch
        and nonzero_branch.get("branch_target_node") == floor_join_node
    )
    floor_selector_falls_through_to_wait = bool(
        nonzero_branch
        and wait_node is not None
        and nonzero_branch.get("fallthrough_node") == wait_node
    )
    if not nonzero_targets_floor_join:
        proof_errors.append("mode!=0 branch does not target floor-wait fallthrough join")
    if not floor_selector_falls_through_to_wait:
        proof_errors.append("mode==0 selector fallthrough is not the unique floor wait")

    full_dominators = _dominators(successors, 0 if instructions else None)
    dataflow_definition_nodes = [
        mode_load[0] if mode_load else None,
        epoch_pointer_load[0] if epoch_pointer_load else None,
        epoch_address[0] if epoch_address else None,
        epoch_load[0] if epoch_load else None,
        topk_pointer_load[0] if topk_pointer_load else None,
        ctaid_load[0] if ctaid_load else None,
        flag_offset[0] if flag_offset else None,
        flag_address[0] if flag_address else None,
        tid_load[0] if tid_load else None,
    ]
    dataflow_use_nodes = [
        selector_setp[0] if selector_setp else None,
        success_compare["index"] if success_compare else None,
        success_compare["index"] if success_compare else None,
        success_compare["index"] if success_compare else None,
        acquire_node, acquire_node, acquire_node, acquire_node,
        thread_gate_compare[0] if thread_gate_compare else None,
    ]
    dataflow_definitions_dominate_uses = all(
        definition is not None
        and use is not None
        and definition in full_dominators.get(use, set())
        for definition, use in zip(dataflow_definition_nodes, dataflow_use_nodes)
    )
    if not dataflow_definitions_dominate_uses:
        proof_errors.append("bound parameter/address definitions do not dominate their uses")
    history_definitions_dominate_uses = False
    if (
        history_address_chain and history_pointer_load and history_base
        and history_load_node is not None and semantic_dependency
        and semantic_candidate_definition is not None
        and contribution_end_node is not None
    ):
        safe_definitions = _register_definition_nodes(
            instructions, history_address_chain["safe_register"]
        )
        history_definition_use_pairs = [
            (history_pointer_load[0], history_load_node),
            (history_base[0], history_load_node),
            (safe_definitions[0] if len(safe_definitions) == 1 else None,
             history_load_node),
            (history_address_chain["offset_node"], history_load_node),
            (history_address_chain["address_node"], history_load_node),
            (history_load_node, contribution_end_node),
            (history_load_node + 2, contribution_end_node),
            (semantic_candidate_definition, semantic_dependency[0] + 1),
            (semantic_dependency[0] + 2, contribution_end_node),
        ]
        history_definitions_dominate_uses = all(
            definition is not None
            and definition in full_dominators.get(use, set())
            for definition, use in history_definition_use_pairs
        )
    if not history_definitions_dominate_uses:
        proof_errors.append("history address/value definitions do not dominate their uses")
    index_semantic_definitions_dominate_uses = False
    if all(
        item is not None
        for item in (
            history_load_node, semantic_dependency, contribution_end_node,
            indices_pointer_load, indices_base, key_tiles_load, topk_load,
            query_widen, rank_widen, topk_widen, initial_linear_index,
            initial_index_byte_offset, initial_index_address, index_load,
            valid_compare, fallback_modulo, safe_index_definition,
            semantic_invalid_definition, semantic_candidate_definition,
        )
    ):
        index_semantic_definition_use_pairs = [
            (indices_pointer_load[0], history_load_node),
            (indices_base[0], history_load_node),
            (key_tiles_load[0], history_load_node),
            (topk_load[0], history_load_node),
            (query_widen[0], history_load_node),
            (rank_widen[0], history_load_node),
            (topk_widen[0], history_load_node),
            (initial_linear_index[0], history_load_node),
            (initial_index_byte_offset[0], history_load_node),
            (initial_index_address[0], history_load_node),
            (index_load[0], history_load_node),
            (valid_compare[0], history_load_node),
            (fallback_modulo[0], history_load_node),
            (safe_index_definition[0], history_load_node),
            (semantic_invalid_definition, semantic_dependency[0] + 1),
            (semantic_candidate_definition, contribution_end_node),
        ]
        index_semantic_definitions_dominate_uses = all(
            definition in full_dominators.get(use, set())
            for definition, use in index_semantic_definition_use_pairs
        )
    if not index_semantic_definitions_dominate_uses:
        proof_errors.append("indices/safe-index/semantic definitions do not dominate uses")

    impl_reachable = _reachable(successors, selector_entry)
    impl_dominators = _dominators(successors, selector_entry)
    all_history_reachable = bool(history_nodes) and all(
        node in impl_reachable for node in history_nodes
    )
    acquire_dominates_all_history = bool(
        acquire_node is not None
        and all_history_reachable
        and all(acquire_node in impl_dominators.get(node, set()) for node in history_nodes)
    )
    if not all_history_reachable:
        proof_errors.append("not every marker-bound history load is reachable from Impl entry")
    if not acquire_dominates_all_history:
        proof_errors.append("acquire does not dominate every Impl-reachable history load")

    join_reachable = _reachable(successors, success_target_node)
    join_reaches_all_history = bool(history_nodes) and all(
        node in join_reachable for node in history_nodes
    )
    if not join_reaches_all_history:
        proof_errors.append("acquire success join does not reach every history load")

    consumer_barrier_node = (
        thread_gate_branch.get("branch_target_node") if thread_gate_branch else None
    )
    consumer_join_is_cta_barrier = bool(
        consumer_barrier_node is not None
        and instructions[consumer_barrier_node]["text"] == "bar.sync 0;"
    )
    if not consumer_join_is_cta_barrier:
        proof_errors.append("nonzero threads do not target a bar.sync 0 consumer join")
    floor_join_to_barrier_linear = False
    if floor_join_node is not None and consumer_barrier_node is not None:
        current = floor_join_node
        seen: set[int] = set()
        floor_join_to_barrier_linear = True
        while current != consumer_barrier_node:
            if current in seen or current in history_nodes or len(successors[current]) != 1:
                floor_join_to_barrier_linear = False
                break
            seen.add(current)
            current = next(iter(successors[current]))
    if not floor_join_to_barrier_linear:
        proof_errors.append("floor/acquire join does not linearly converge on consumer barrier")
    consumer_barrier_dominates_all_history = bool(
        consumer_join_is_cta_barrier
        and history_nodes
        and all(
            consumer_barrier_node in full_dominators.get(node, set())
            for node in history_nodes
        )
    )
    if not consumer_barrier_dominates_all_history:
        proof_errors.append("consumer bar.sync 0 does not dominate every history load")

    retry_has_nanosleep = False
    retry_backedge = False
    retry_target_node = None
    failure_path: list[int] = []
    if success_branch and acquire_node is not None:
        current = success_branch.get("fallthrough_node")
        seen: set[int] = set()
        while current is not None and current != acquire_node and current not in seen:
            seen.add(current)
            failure_path.append(current)
            item = instructions[current]
            if item["opcode"].startswith("nanosleep."):
                retry_has_nanosleep = True
            if current in history_nodes or len(successors[current]) != 1:
                break
            next_node = next(iter(successors[current]))
            if next_node == acquire_node:
                retry_target_node = next_node
                retry_backedge = item.get("branch_kind") == "unconditional"
                break
            current = next_node
    if not retry_has_nanosleep:
        proof_errors.append("acquire failure path has no nanosleep")
    if not retry_backedge:
        proof_errors.append("acquire failure path is not a deterministic explicit backedge")

    positions_match = True
    if acquire_position is not None:
        positions_match = bool(
            acquire_node is not None
            and instructions[acquire_node]["position"] == acquire_position
        )
    if history_load_position is not None:
        positions_match = bool(
            positions_match
            and history_nodes
            and instructions[history_nodes[0]]["position"] == history_load_position
        )
    if not positions_match:
        proof_errors.append("caller-selected acquire/history positions do not match CFG proof")

    selector_predicate_unique = bool(
        selector_setp
        and _register_definition_nodes(
            instructions, selector_setp[1].group("pred")
        ) == [selector_setp[0]]
    )
    success_predicate_unique = bool(
        success_compare
        and _register_definition_nodes(
            instructions,
            re.fullmatch(
                r"setp\.eq\.s32\s+(%p\d+),.*;", success_compare["text"]
            ).group(1),
        ) == [success_compare["index"]]
    )
    nonzero_predicate_unique = bool(
        nonzero_compare
        and _register_definition_nodes(
            instructions,
            re.fullmatch(
                r"setp\.ne\.s32\s+(%p\d+),.*;", nonzero_compare["text"]
            ).group(1),
        ) == [nonzero_compare["index"]]
    )
    thread_gate_predicate_unique = bool(
        thread_gate_compare
        and thread_gate_branch
        and _register_definition_nodes(
            instructions, thread_gate_branch.get("branch_predicate")
        ) == [thread_gate_compare[0]]
    )
    if not selector_predicate_unique:
        proof_errors.append("Impl selector predicate does not have one bound definition")
    if not success_predicate_unique:
        proof_errors.append("acquire success predicate does not have one bound definition")
    if not nonzero_predicate_unique:
        proof_errors.append("mode!=0 predicate does not have one bound definition")
    if not thread_gate_predicate_unique:
        proof_errors.append("thread-nonzero predicate does not have one bound definition")

    passed = not proof_errors
    return {
        "pass": passed,
        "errors": proof_errors,
        "cfg_instruction_count": len(instructions),
        "cfg_branch_count": cfg["branch_count"],
        "labels_unique": not cfg["duplicate_labels"],
        "all_branch_targets_resolved": not any(
            "branch target" in error for error in cfg["errors"]
        ),
        "unsupported_control_flow": cfg["unsupported_control"],
        "selector_count": len(selectors),
        "selector_mode_parameter": 13,
        "selector_mode_register": mode_register,
        "selector_target": selector_branch.get("branch_target") if selector_branch else None,
        "selector_entry_block": _node_name(cfg, selector_entry),
        "tid_register": tid_register,
        "thread_gate_block": _node_name(
            cfg, thread_gate_branch["index"] if thread_gate_branch else None
        ),
        "thread_gate_target": (
            thread_gate_branch.get("branch_target") if thread_gate_branch else None
        ),
        "thread_gate_linear": thread_gate_linear,
        "nonzero_selector_register": mode_register,
        "nonzero_selector_target": (
            nonzero_branch.get("branch_target") if nonzero_branch else None
        ),
        "nonzero_selector_target_block": _node_name(
            cfg,
            nonzero_branch.get("branch_target_node") if nonzero_branch else None,
        ),
        "epoch_pointer_parameter": 6,
        "epoch_pointer_register": epoch_pointer_register,
        "epoch_register": epoch_register,
        "topk_flags_parameter": 3,
        "topk_flags_register": topk_pointer_register,
        "ctaid_register": ctaid_register,
        "query_base_parameter": 16,
        "query_base_register": query_base_register,
        "global_query_register": query_register,
        "global_query_bound": bool(query_definition),
        "acquire_address_register": flag_address_register,
        "acquire_block": _node_name(cfg, acquire_node),
        "acquire_success_target": (
            success_branch.get("branch_target") if success_branch else None
        ),
        "acquire_success_block": _node_name(cfg, success_target_node),
        "floor_join_block": _node_name(cfg, floor_join_node),
        "consumer_barrier_block": _node_name(cfg, consumer_barrier_node),
        "consumer_join_is_cta_barrier": consumer_join_is_cta_barrier,
        "floor_join_to_barrier_linear": floor_join_to_barrier_linear,
        "consumer_barrier_dominates_all_history": (
            consumer_barrier_dominates_all_history
        ),
        "acquire_failure_block": (
            _node_name(cfg, failure_path[0]) if failure_path else None
        ),
        "acquire_failure_target": _node_name(cfg, retry_target_node),
        "history_load_blocks": [_node_name(cfg, node) for node in history_nodes],
        "history_load_count": len(history_nodes),
        "history_static_site_expected": ATTENTION_HISTORY_STATIC_SITES,
        "history_site_cardinality": history_site_cardinality,
        "history_sites_branch_free": history_sites_branch_free,
        "history_full_chain_branch_free": history_full_chain_branch_free,
        "history_parameter": 1,
        "history_pointer_register": history_pointer_register,
        "history_base_register": history_base_register,
        "history_safe_index_register": (
            history_address_chain["safe_register"] if history_address_chain else None
        ),
        "history_address_register": (
            history_address_chain["address_register"] if history_address_chain else None
        ),
        "history_loaded_register": history_load_register,
        "history_value_register": loaded_value_register,
        "semantic_candidate_register": semantic_candidate_register,
        "semantic_output_register": semantic_output_register,
        "history_marker_matches_bound_load": history_marker_matches_bound_load,
        "history_count_chain_bound": count_chain_bound,
        "history_semantic_chain_bound": bool(
            semantic_candidate_marker and semantic_output_marker
        ),
        "history_contribution_bound": contribution_bound,
        "history_dataflow_unique_definitions": history_dataflow_unique_definitions,
        "history_definitions_dominate_uses": history_definitions_dominate_uses,
        "indices_parameter": 0,
        "indices_pointer_register": indices_pointer_register,
        "indices_base_register": indices_base_register,
        "key_tiles_parameter": 10,
        "key_tiles_register": key_tiles_register,
        "topk_parameter": 12,
        "topk_register": topk_register,
        "rank_index_address_register": index_address_register,
        "rank_index_register": index_register,
        "validity_predicate": valid_predicate,
        "fallback_register": fallback_register,
        "rank_loop_register": rank_loop_register,
        "rank_loop_bound": rank_loop_bound,
        "index_to_contribution_branch_free": index_to_contribution_branch_free,
        "index_semantic_unique_definitions": index_semantic_unique_definitions,
        "index_semantic_definitions_dominate_uses": (
            index_semantic_definitions_dominate_uses
        ),
        "static_history_proof_scope": (
            "one_static_unrolled1_loop_site; param0_rank_index_to_safe_index; "
            "param1_safe_address_to_load_count_semantic_contribution"
        ),
        "runtime_reference_complement_required": True,
        "runtime_reference_complement": (
            "same-binary full output/history-sum/history-load-count validation"
        ),
        "all_history_reachable_from_impl": all_history_reachable,
        "acquire_dominates_all_history": acquire_dominates_all_history,
        "success_targets_floor_join": success_targets_floor_join,
        "nonzero_targets_floor_join": nonzero_targets_floor_join,
        "floor_selector_falls_through_to_wait": floor_selector_falls_through_to_wait,
        "success_join_reaches_all_history": join_reaches_all_history,
        "retry_has_nanosleep": retry_has_nanosleep,
        "retry_backedge_to_acquire": retry_backedge,
        "unique_register_definitions": unique_register_definitions,
        "dataflow_definitions_dominate_uses": dataflow_definitions_dominate_uses,
        "selector_predicate_unique": selector_predicate_unique,
        "nonzero_predicate_unique": nonzero_predicate_unique,
        "thread_gate_predicate_unique": thread_gate_predicate_unique,
        "success_predicate_unique": success_predicate_unique,
        "positions_match": positions_match,
    }


def verify_ptx(ptx: str, errors: list[str]) -> dict[str, dict[str, Any]]:
    """Verify exact worker-local PDL/atomic counts and their semantic order."""
    sections = extract_ptx_sections(ptx, errors)
    kernels: dict[str, dict[str, Any]] = {}
    tokens = {
        "launch_dependents": "griddepcontrol.launch_dependents",
        "wait": "griddepcontrol.wait",
        "membar_gl": "membar.gl;",
        "acquire": "ld.acquire.gpu.b32",
        "release": "st.release.gpu.b32",
    }
    for name, exact in PTX_EXACT.items():
        section = sections.get(name, "")
        counts = {key: section.count(token) for key, token in tokens.items()}
        global_loads = regex_positions(
            section, r"(?m)\bld(?:\.[a-z0-9_]+)*\.global(?:\.[a-z0-9_]+)*\b"
        )
        shared_loads = regex_positions(
            section, r"(?m)\bld(?:\.[a-z0-9_]+)*\.shared(?:\.[a-z0-9_]+)*\b"
        )
        shared_stores = regex_positions(
            section, r"(?m)\bst(?:\.[a-z0-9_]+)*\.shared(?:\.[a-z0-9_]+)*\b"
        )
        local_accesses = regex_positions(
            section, r"(?m)\b(?:ld|st)(?:\.[a-z0-9_]+)*\.local(?:\.[a-z0-9_]+)*\b"
        )
        counts.update({
            "global_load": len(global_loads),
            "shared_load": len(shared_loads),
            "shared_store": len(shared_stores),
            "local_access": len(local_accesses),
        })
        consumer_entry_proof: dict[str, Any] = {}
        if name in {"dsaTopk", "dsaAttention"}:
            entry_parameter = 16 if name == "dsaTopk" else 17
            entry_pointer_loads = list(re.finditer(
                rf"(?m)^ld\.param\.u64\s+(?P<ptr>%rd\d+),\s*"
                rf"\[[^\]]*_param_{entry_parameter}\];$",
                section,
            ))
            entry_decl = regex_positions(
                section, r"(?m)^\.reg \.pred dsa_entry_thread0;$"
            )
            entry_setp = list(re.finditer(
                r"(?m)^setp\.eq\.u32 dsa_entry_thread0,\s*%r\d+,\s*0;$",
                section,
            ))
            entry_atomic = list(re.finditer(
                r"(?m)^@dsa_entry_thread0 atom\.global\.sys\.add\.u32 "
                r"dsa_entry_old,\s*\[(?P<address>%rd\d+)\],\s*1;$",
                section,
            ))
            pointer_bound = False
            entry_pointer_register = (
                entry_pointer_loads[0].group("ptr")
                if len(entry_pointer_loads) == 1 else None
            )
            entry_address_register = (
                entry_atomic[0].group("address") if len(entry_atomic) == 1 else None
            )
            if entry_pointer_register and entry_address_register:
                if name == "dsaTopk":
                    pointer_bound = entry_address_register == entry_pointer_register
                else:
                    pointer_bound = bool(re.search(
                        rf"(?m)^add\.s64\s+{re.escape(entry_address_register)},\s*"
                        rf"{re.escape(entry_pointer_register)},\s*4;$",
                        section,
                    ))
            marker_order = bool(
                len(entry_setp) == 1 and len(entry_atomic) == 1
                and entry_setp[0].start() < entry_atomic[0].start()
                < section.find(tokens["acquire"])
            )
            entry_marker_pass = bool(
                len(entry_pointer_loads) == 1
                and len(entry_decl) == 1
                and len(entry_setp) == 1
                and len(entry_atomic) == 1
                and pointer_bound and marker_order
            )
            counts["consumer_entry_marker"] = len(entry_atomic)
            consumer_entry_proof = {
                "consumer_entry_parameter": entry_parameter,
                "consumer_entry_pointer_register": entry_pointer_register,
                "consumer_entry_address_register": entry_address_register,
                "consumer_entry_marker_count": len(entry_atomic),
                "consumer_entry_pointer_bound": pointer_bound,
                "consumer_entry_before_acquire": marker_order,
                "consumer_entry_marker_pass": entry_marker_pass,
            }
        if name == "dsaIndexer":
            pair_markers = regex_positions(section, r"\bdsa_pair_term\b")
            pair_adds = regex_positions(
                section, r"(?m)\badd\.u32\b[^;\n]*\bdsa_pair_term\b"
            )
            order_label = "launch<global_to_shared_lut<pair_add<fence<release<floor_launch"
            positions = {
                key: [match.start() for match in re.finditer(re.escape(tokens[key]), section)]
                for key in ("launch_dependents", "membar_gl", "release")
            }
            order_ok = (
                len(positions["launch_dependents"]) == 2
                and len(global_loads) >= 3
                and len(shared_stores) >= 2
                and len(shared_loads) >= 2
                and len(pair_markers) >= 1
                and len(pair_adds) >= 1
                and len(positions["membar_gl"]) == 1
                and len(positions["release"]) == 1
                and positions["launch_dependents"][0] < shared_stores[0]
                and shared_stores[-1] < pair_markers[0]
                and shared_loads[-1] < positions["membar_gl"][0]
                and pair_adds[-1] < positions["membar_gl"][0]
                < positions["release"][0] < positions["launch_dependents"][1]
            )
            counts.update({
                "pair_marker": len(pair_markers),
                "explicit_pair_add": len(pair_adds),
            })
            extra_proof = {
                "pair_iteration": "explicit_inline_ptx_add_u32_per_pair",
                "pair_add_present": bool(pair_adds),
                "query_key_cache": "global_to_shared_once_per_cta",
                "shared_cache_present": len(shared_stores) >= 2 and len(shared_loads) >= 2,
                "register_tile_no_local_spill": len(local_accesses) == 0,
            }
        elif name == "dsaTopk":
            order_tokens = [
                tokens["launch_dependents"], tokens["wait"], tokens["acquire"],
                tokens["membar_gl"], tokens["release"], tokens["launch_dependents"],
            ]
            order_label = "launch<wait<acquire<fence<release<floor_launch"
            order_ok = ordered(section, order_tokens)
            extra_proof = {}
        else:
            order_tokens = [
                tokens["launch_dependents"], tokens["wait"], tokens["acquire"],
                tokens["membar_gl"], tokens["launch_dependents"],
            ]
            history_loaded = regex_positions(section, r"\bdsa_history_loaded_value\b")
            history_count = regex_positions(section, r"\bdsa_history_load_count\b")
            history_count_add = regex_positions(
                section, r"(?m)\badd\.u64\b[^;\n]*\bdsa_history_load_count\b"
            )
            semantic_index = regex_positions(section, r"\bdsa_semantic_index\b")
            history_load_before_semantic = False
            no_branch_between_load_and_semantic = False
            history_load_after_acquire = False
            acquire_cfg: dict[str, Any] = {}
            if history_loaded and history_count and history_count_add and semantic_index:
                preceding_loads = [pos for pos in global_loads if pos < history_loaded[0]]
                if preceding_loads:
                    history_load = preceding_loads[-1]
                    history_load_before_semantic = (
                        history_load < history_loaded[0]
                        < history_count_add[0] < semantic_index[0]
                    )
                    acquire_position = section.find(tokens["acquire"])
                    fence_position = section.find(tokens["membar_gl"])
                    if acquire_position >= 0 and fence_position > history_load:
                        acquire_cfg = attention_acquire_cfg_proof(
                            section, acquire_position, history_load
                        )
                        history_load_after_acquire = bool(acquire_cfg.get("pass"))
                    segment = section[history_load:semantic_index[0]]
                    no_branch_between_load_and_semantic = not re.search(r"\b(?:bra|brx)\b", segment)
            order_label = (
                "launch<wait<acquire<history_load<history_marker<semantic_index"
                "<fence<floor_launch"
            )
            order_ok = (
                ordered(section, order_tokens)
                and history_load_before_semantic
                and history_load_after_acquire
                and no_branch_between_load_and_semantic
            )
            counts.update({
                "history_loaded_marker": len(history_loaded),
                "history_load_count_marker": len(history_count),
                "explicit_history_count_add": len(history_count_add),
                "semantic_index_marker": len(semantic_index),
            })
            extra_proof = {
                "history_load_before_semantic_propagation": history_load_before_semantic,
                "history_load_after_dependency_acquire": history_load_after_acquire,
                "history_load_to_semantic_straight_line": no_branch_between_load_and_semantic,
                "history_load_shape": (
                    "one_unrolled1_static load/count/semantic chain per dynamic rank loop; "
                    "same-binary runtime reference validation required"
                ),
                "dynamic_history_count": "explicit_add_u64_after_loaded_value_per_rank",
                "acquire_cfg_proof": acquire_cfg,
            }
        kernels[name] = {
            **counts, "exact_required": exact,
            "ordering": order_label, "ordering_pass": order_ok,
            **extra_proof, **consumer_entry_proof,
        }
        for key, expected in exact.items():
            if counts[key] != expected:
                errors.append(f"{name} {key}={counts[key]} expected exactly {expected}")
        if name == "dsaIndexer" and not extra_proof["pair_add_present"]:
            errors.append("dsaIndexer explicit per-pair PTX add marker missing")
        if name == "dsaIndexer" and not extra_proof["shared_cache_present"]:
            errors.append("dsaIndexer global-to-shared LUT cache proof missing")
        if name == "dsaIndexer" and not extra_proof["register_tile_no_local_spill"]:
            errors.append("dsaIndexer key register tile spilled to local memory")
        if name == "dsaAttention" and not extra_proof["history_load_before_semantic_propagation"]:
            errors.append("dsaAttention history load is not proven before semantic propagation")
        if name == "dsaAttention" and not extra_proof["history_load_after_dependency_acquire"]:
            errors.append("dsaAttention history load is not proven after dependency acquire")
        if name == "dsaAttention" and not extra_proof["history_load_to_semantic_straight_line"]:
            errors.append("dsaAttention history load/semantic propagation contains a branch")
        if name in {"dsaTopk", "dsaAttention"} and not consumer_entry_proof.get(
            "consumer_entry_marker_pass", False
        ):
            errors.append(f"{name} consumer entry marker is not parameter-bound before acquire")
        if not order_ok:
            errors.append(f"{name} PTX ordering proof failed: {order_label}")
    return kernels


def find_nvdisasm() -> str | None:
    candidates = (
        shutil.which("nvdisasm"),
        "/usr/local/cuda-13.0/bin/nvdisasm",
        "/usr/local/cuda/bin/nvdisasm",
    )
    return next((str(path) for path in candidates if path and Path(path).is_file()), None)


def extract_sass(
    binary: Path, target: str, output: Path, errors: list[str]
) -> dict[str, Any]:
    nvdisasm = find_nvdisasm()
    proof: dict[str, Any] = {
        "target": target,
        "nvdisasm": nvdisasm,
        "semantic_scope": (
            "real_cubin_target_and_worker_machine_code; "
            "PDL ordering authority is exact PTX, not SASS mnemonic inference"
        ),
    }
    if nvdisasm is None:
        errors.append("nvdisasm unavailable")
        return proof
    try:
        version = subprocess.run(
            [nvdisasm, "--version"], check=False, text=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        )
        proof["nvdisasm_version"] = version.stdout.strip()
        with tempfile.TemporaryDirectory(prefix="dsa-cubin-proof-") as temp_name:
            temp = Path(temp_name)
            extract = subprocess.run(
                ["cuobjdump", "-xelf", "all", str(binary.resolve())],
                cwd=temp, check=False, text=True,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            )
            proof["cuobjdump_extract_returncode"] = extract.returncode
            proof["cuobjdump_extract_stdout"] = extract.stdout.strip()
            if extract.returncode != 0:
                errors.append(f"cuobjdump cubin extraction returncode={extract.returncode}")
                return proof
            candidates = sorted(temp.glob("*.cubin"))
            proof["extracted_cubins"] = [path.name for path in candidates]
            selected: tuple[Path, str] | None = None
            diagnostics: list[str] = []
            for cubin in candidates:
                run = subprocess.run(
                    [nvdisasm, str(cubin)], check=False, text=True,
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                )
                diagnostics.append(f"{cubin.name}:rc={run.returncode}")
                if run.returncode == 0 and re.search(
                    rf"(?m)^\s*\.target\s+{re.escape(target)}\s*$", run.stdout
                ):
                    selected = (cubin, run.stdout)
                    break
            proof["candidate_diagnostics"] = diagnostics
            if selected is None:
                errors.append(f"no decodable {target} cubin extracted")
                return proof
            cubin, sass = selected
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(sass, encoding="utf-8")
            sections: dict[str, int] = {}
            local_memory_instructions: dict[str, int] = {}
            for name in WORKERS:
                match = re.search(
                    rf"(?ms)^//-+ \.text\.[^\n]*{name}[^\n]* -+\n(.*?)"
                    rf"(?=^//-+|\Z)",
                    sass,
                )
                body = match.group(1) if match else ""
                count = len(re.findall(r"(?m)^\s*/\*[0-9a-fA-F]+\*/", body)) \
                    if match else 0
                sections[name] = count
                local_memory_instructions[name] = len(
                    re.findall(r"(?m)\b(?:LDL|STL)(?:\.[A-Z0-9]+)*\b", body)
                )
                if count <= 0:
                    errors.append(f"{target} SASS has no machine-code section for {name}")
            if local_memory_instructions.get("dsaIndexer", 0) != 0:
                errors.append(
                    f"{target} dsaIndexer SASS contains LDL/STL local-memory instructions"
                )
            cubin_bytes = cubin.read_bytes()
            sass_bytes = sass.encode("utf-8")
            proof.update({
                "cubin_name": cubin.name,
                "cubin_bytes": len(cubin_bytes),
                "cubin_sha256": sha256_bytes(cubin_bytes),
                "sass_path": str(output),
                "sass_bytes": len(sass_bytes),
                "sass_sha256": sha256_bytes(sass_bytes),
                "worker_instruction_counts": sections,
                "worker_local_memory_instructions": local_memory_instructions,
            })
    except OSError as exc:
        errors.append(f"cubin/SASS proof failed: {exc}")
    return proof


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("binary", type=Path)
    parser.add_argument("--ptx", type=Path, required=True)
    parser.add_argument("--resources", type=Path, required=True)
    parser.add_argument("--sass", type=Path, required=True)
    parser.add_argument("--target", default="sm_100")
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    errors: list[str] = []
    try:
        ptx_run = subprocess.run(
            ["cuobjdump", "--dump-ptx", str(args.binary)], check=False,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        resource_run = subprocess.run(
            ["cuobjdump", "--dump-resource-usage", str(args.binary)], check=False,
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        args.ptx.parent.mkdir(parents=True, exist_ok=True)
        args.resources.parent.mkdir(parents=True, exist_ok=True)
        args.ptx.write_text(ptx_run.stdout + ptx_run.stderr, encoding="utf-8")
        args.resources.write_text(
            resource_run.stdout + resource_run.stderr, encoding="utf-8"
        )
    except OSError as exc:
        payload = {"schema": 2, "status": "FAIL", "errors": [str(exc)]}
        atomic_json(args.json, payload)
        return 2
    if ptx_run.returncode != 0:
        errors.append(f"cuobjdump PTX returncode={ptx_run.returncode}")
    if resource_run.returncode != 0:
        errors.append(f"cuobjdump resources returncode={resource_run.returncode}")

    kernels = verify_ptx(ptx_run.stdout, errors)

    resources: dict[str, dict[str, int]] = {}
    for name in WORKERS:
        match = re.search(
            rf"(?m)^ Function [^\n]*{name}[^\n]*:\n\s+REG:(\d+) "
            rf"STACK:(\d+) SHARED:(\d+) LOCAL:(\d+)",
            resource_run.stdout,
        )
        if not match:
            errors.append(f"resource report missing {name}")
            continue
        resources[name] = {
            key: int(value) for key, value in zip(
                ("registers", "stack", "static_shared", "local"), match.groups()
            )
        }
        if resources[name]["registers"] <= 0:
            errors.append(f"resource report has non-positive registers for {name}")
        if name == "dsaIndexer" and (
            resources[name]["stack"] != 0 or resources[name]["local"] != 0
        ):
            errors.append(
                "dsaIndexer register-tile proof requires resource STACK=0 and LOCAL=0"
            )

    binary_bytes = args.binary.read_bytes()
    required_nvtx = (
        "dsa.poison", "dsa.floor", "dsa.wave_floor", "dsa.impl", "dsa.ceiling",
        "dsa.validate.floor", "dsa.validate.wave_floor", "dsa.validate.impl",
        "dsa.validate.ceiling_wrongness",
    )
    nvtx_ranges = {
        name: int(name.encode("ascii") + b"\0" in binary_bytes) for name in required_nvtx
    }
    missing_nvtx = [name for name, present in nvtx_ranges.items() if not present]
    if missing_nvtx:
        errors.append("missing NVTX ranges: " + ",".join(missing_nvtx))

    sass_proof = extract_sass(args.binary, args.target, args.sass, errors)
    indexer_kernel = kernels.setdefault("dsaIndexer", {})
    indexer_resources = resources.get("dsaIndexer", {})
    indexer_sass_local = sass_proof.get(
        "worker_local_memory_instructions", {}
    ).get("dsaIndexer")
    indexer_kernel["resource_stack_bytes"] = indexer_resources.get("stack")
    indexer_kernel["resource_local_bytes"] = indexer_resources.get("local")
    indexer_kernel["sass_local_memory_instructions"] = indexer_sass_local
    indexer_kernel["register_tile_no_spill_complete"] = bool(
        indexer_kernel.get("register_tile_no_local_spill")
        and indexer_resources.get("stack") == 0
        and indexer_resources.get("local") == 0
        and indexer_sass_local == 0
    )
    payload = {
        "schema": 2,
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "binary": str(args.binary),
        "binary_sha256": sha256_bytes(binary_bytes),
        "ptx_path": str(args.ptx),
        "ptx_sha256": sha256_bytes(args.ptx.read_bytes()),
        "resource_path": str(args.resources),
        "resource_sha256": sha256_bytes(args.resources.read_bytes()),
        "ptx_semantic_authority": True,
        "kernels": kernels,
        "resources": resources,
        "nvtx_ranges": nvtx_ranges,
        "sass_proof": sass_proof,
    }
    atomic_json(args.json, payload)
    print(
        "DSA_BINARY_PROOF "
        f"schema=2 status={payload['status']} errors={len(errors)} "
        f"sha256={payload['binary_sha256']} target={args.target}"
    )
    return 0 if not errors else 2


if __name__ == "__main__":
    raise SystemExit(main())
