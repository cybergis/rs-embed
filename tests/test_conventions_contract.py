"""Convention contract tests — the repo's hard rules, enforced by CI.

Each test here encodes a maintainability convention (see
``MAINTAINABILITY_REVIEW.md`` and the add-model skill) so a violation fails
loudly during development instead of accruing silently:

1. generic layers must not hardcode model names (policy lives on the
   embedder class — flags, hooks, capabilities);
2. on-the-fly embedders resolve temporal through the shared
   ``temporal_to_range`` entry point;
3. ``describe()`` keeps its minimal schema promise (``type`` + ``output``);
4. "must match / keep in sync" comments are a ratchet: the count may only
   go down — extracting the shared function is the fix, not documenting
   the duplication.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

from rs_embed.core.registry import get_embedder_cls
from rs_embed.embedders.catalog import MODEL_SPECS

SRC = Path(__file__).resolve().parents[1] / "src" / "rs_embed"

# Layers that must stay model-agnostic. embedders/ is exempt (models live
# there); core/registry + embedders/catalog hold the id->class mapping itself.
_GENERIC_LAYER_FILES = sorted(
    p
    for d in ("tools", "pipelines", "providers", "core")
    for p in (SRC / d).rglob("*.py")
    if "_vendor" not in p.parts and p.name != "registry.py"
) + [SRC / "api.py", SRC / "model.py", SRC / "load.py"]


def _code_string_literals(path: Path) -> list[tuple[int, str]]:
    """All string constants in *path*, excluding docstrings."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                docstrings.add(id(body[0].value))
    out: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            out.append((node.lineno, node.value))
    return out


def test_generic_layers_contain_no_model_name_literals():
    """Model-specific behavior belongs on the embedder class (flag / hook /
    capability), never as a name check inside tools/pipelines/providers/core.
    """
    model_ids = sorted(MODEL_SPECS, key=len, reverse=True)
    patterns = {mid: re.compile(rf"\b{re.escape(mid)}\b") for mid in model_ids}
    violations: list[str] = []
    for path in _GENERIC_LAYER_FILES:
        for lineno, text in _code_string_literals(path):
            for mid, pat in patterns.items():
                if pat.search(text):
                    violations.append(
                        f"{path.relative_to(SRC.parent.parent)}:{lineno}: "
                        f"string literal mentions model id {mid!r}: {text[:80]!r}"
                    )
    assert not violations, (
        "Generic layers must not branch on model names — move the policy to a "
        "class flag/hook on the embedder (see EmbedderBase). Violations:\n  "
        + "\n  ".join(violations)
    )


def test_onthefly_embedders_resolve_temporal_via_shared_helper():
    """Every on-the-fly embedder funnels temporal through temporal_to_range
    (None -> package default window). Raising a bespoke 'temporal required'
    error or hand-rolling the range is a convention violation. Precomputed
    embedders are exempt (year semantics).
    """
    missing = []
    for path in sorted((SRC / "embedders").glob("onthefly_*.py")):
        if "temporal_to_range" not in path.read_text(encoding="utf-8"):
            missing.append(path.name)
    assert not missing, f"on-the-fly embedders not using temporal_to_range: {missing}"


@pytest.mark.parametrize("model_id", sorted(MODEL_SPECS))
def test_describe_minimal_schema(model_id):
    """describe() always includes 'type' and 'output', and declared outputs
    are drawn from the supported set (documented in api.describe_model)."""
    desc = get_embedder_cls(model_id)().describe()
    assert isinstance(desc, dict)
    assert "type" in desc, f"{model_id}: describe() missing 'type'"
    assert "output" in desc, f"{model_id}: describe() missing 'output'"
    outputs = desc["output"]
    assert isinstance(outputs, list) and outputs, f"{model_id}: 'output' must be a non-empty list"
    assert set(outputs) <= {"pooled", "grid"}, f"{model_id}: unknown outputs {outputs}"


# ── sync-comment ratchet ────────────────────────────────────────────

# Phrases that admit "this code must be kept equivalent to code elsewhere by
# hand". Each one is duplication debt: the fix is extracting one shared
# function, not writing the comment.
_SYNC_DEBT_PATTERNS = [
    r"must match",
    r"keep in sync",
    r"kept in sync",
    r"must stay identical",
    r"mirror(?:s)? get_embedding",
    r"mirroring the single-point",
]

# Current debt level. This number may only DECREASE: when you remove one of
# these comments by extracting shared code, lower the baseline in the same
# commit. Raising it requires deliberately accepting new hand-maintained
# duplication — don't.
_SYNC_DEBT_BASELINE = 7


def test_sync_comment_ratchet():
    pat = re.compile("|".join(_SYNC_DEBT_PATTERNS), re.IGNORECASE)
    hits: list[str] = []
    for path in sorted(SRC.rglob("*.py")):
        if "_vendor" in path.parts:
            continue
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if pat.search(line):
                hits.append(f"{path.relative_to(SRC.parent.parent)}:{lineno}: {line.strip()[:90]}")
    count = len(hits)
    assert count <= _SYNC_DEBT_BASELINE, (
        f"Hand-maintained-equivalence comments went UP ({count} > baseline "
        f"{_SYNC_DEBT_BASELINE}). Extract the shared function instead of "
        "documenting the duplication:\n  " + "\n  ".join(hits)
    )
    assert count == _SYNC_DEBT_BASELINE, (
        f"Debt went down ({count} < {_SYNC_DEBT_BASELINE}) — nice. Lower "
        "_SYNC_DEBT_BASELINE to match so the improvement is locked in."
    )
