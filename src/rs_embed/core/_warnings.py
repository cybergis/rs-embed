"""Pretty, structured rendering for rs-embed's own warnings.

The default :func:`warnings.showwarning` prints a terse two-line blob
(``path:lineno: Category: message`` followed by the source line) that is hard
to scan, especially when several warnings stack up during a batch export.

This module installs a drop-in replacement for ``warnings.showwarning`` that
renders warnings *originating from inside the rs_embed package* as a small,
colourised block::

    ⚠ rs-embed · UserWarning
    │ model='gse' manages spatial tiling automatically based on request
    │ size; input_prep is ignored.
    ╰ rs_embed/api.py:116

Warnings from other libraries are passed through untouched to whatever handler
was installed before us, so importing rs_embed never hijacks unrelated output.

Behaviour is intentionally conservative:

* Colour is emitted to a TTY or a Jupyter/IPython kernel, and is suppressed when
  ``NO_COLOR`` is set or ``TERM=dumb``. The structured layout is used regardless.
* The message hard-wraps to the terminal width in a real terminal; in a notebook
  (no readable terminal size) it is left unwrapped so the cell soft-wraps it to
  the live window width.
* Set ``RS_EMBED_PLAIN_WARNINGS=1`` to opt out entirely and keep Python's
  default formatting.

Public entry points: :func:`enable_pretty_warnings` and
:func:`disable_pretty_warnings`.
"""

from __future__ import annotations

import os
import re
import shutil
import sys
import textwrap
import warnings
from typing import TextIO

# Root of the installed ``rs_embed`` package (``.../site-packages/rs_embed``).
# ``__file__`` is ``.../rs_embed/core/_warnings.py`` -> two dirnames up.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PKG_PARENT = os.path.dirname(_PKG_ROOT)

# ANSI styling.
_RESET = "\x1b[0m"
_BOLD = "\x1b[1m"
_DIM = "\x1b[2m"

# Per-category accent colour + human label. Anything not listed falls back to
# the yellow "warning" styling.
_YELLOW = "\x1b[33m"
_MAGENTA = "\x1b[35m"
_RED = "\x1b[31m"

# Inline highlighting for code-like tokens inside the message body.
_GREEN = "\x1b[32m"  # quoted values, e.g. 'tile'
_CYAN = "\x1b[36m"  # parameter / identifier names, e.g. input_prep

_CATEGORY_STYLE = {
    "DeprecationWarning": _MAGENTA,
    "PendingDeprecationWarning": _MAGENTA,
    "FutureWarning": _MAGENTA,
    "RuntimeWarning": _RED,
    "ResourceWarning": _RED,
}

# Compiled token regex for inline highlighting, built lazily so we only pay for
# (and import) the model catalogue the first time a warning is actually shown.
_token_re: re.Pattern[str] | None = None


def _build_token_re() -> re.Pattern[str]:
    """Compile the alternation that finds highlightable tokens in a message.

    Recognises, in priority order at each position:

    * ``quoted``  — single/double-quoted values such as ``'tile'`` or ``"grid"``
    * ``assign``  — an identifier immediately before ``=`` (the param name in
      ``model=`` / ``input_prep='tile'``)
    * ``ident``   — bare snake_case identifiers such as ``input_prep``
    * ``model``   — known model ids (``remoteclip``, ``gse``, ...) from the catalog
    """
    try:
        from rs_embed.embedders.catalog import MODEL_SPECS

        # Longest-first so e.g. ``satmaepp`` is preferred over ``satmae``.
        models = sorted(MODEL_SPECS.keys(), key=len, reverse=True)
    except Exception:
        models = []

    parts = [
        r"(?P<quoted>'[^'\n]*'|\"[^\"\n]*\")",
        r"(?P<assign>[A-Za-z_][\w.]*)(?==)",
        r"(?P<ident>[A-Za-z_]\w*_\w*)",
    ]
    if models:
        model_alt = "|".join(re.escape(m) for m in models)
        parts.append(rf"(?P<model>\b(?:{model_alt})\b)")

    return re.compile("|".join(parts))


def _highlight_token(match: re.Match[str]) -> str:
    """Wrap a single matched token in its accent colour."""
    group = match.lastgroup
    token = match.group()
    if group == "quoted":
        return f"{_GREEN}{token}{_RESET}"
    if group == "model":
        return f"{_BOLD}{_CYAN}{token}{_RESET}"
    return f"{_CYAN}{token}{_RESET}"  # assign / ident


def _highlight(line: str) -> str:
    """Colourise code-like tokens within one already-wrapped line of text."""
    global _token_re
    if _token_re is None:
        _token_re = _build_token_re()
    return _token_re.sub(_highlight_token, line)

# Saved handler we replace, so we can both delegate non-rs_embed warnings and
# cleanly restore on disable. ``None`` means we are not currently installed.
_previous_showwarning = None


def _default_show(
    message: object,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """Python's stock warning output, built only from public ``warnings`` API."""
    stream = file if file is not None else sys.stderr
    try:
        stream.write(warnings.formatwarning(message, category, filename, lineno, line))
    except (OSError, ValueError):
        pass  # stream gone (e.g. closed during interpreter shutdown)


def _is_rs_embed_warning(filename: str) -> bool:
    """True when ``filename`` lives inside the rs_embed package tree."""
    if not filename:
        return False
    try:
        return os.path.abspath(filename).startswith(_PKG_ROOT + os.sep)
    except (OSError, ValueError):
        return False


def _friendly_path(filename: str) -> str:
    """Render an rs_embed source path relative to the package parent.

    ``.../site-packages/rs_embed/api.py`` -> ``rs_embed/api.py``. Falls back to
    the original path if it does not sit under the package parent.
    """
    abs_path = os.path.abspath(filename)
    if abs_path.startswith(_PKG_PARENT + os.sep):
        return os.path.relpath(abs_path, _PKG_PARENT)
    return filename


def _in_notebook() -> bool:
    """True when running inside a Jupyter/IPython kernel (notebook, lab, qtconsole).

    Such frontends render ANSI colour and soft-wrap output to the cell width, but
    report ``isatty() == False`` and expose no usable terminal size.
    """
    try:
        from IPython import get_ipython
    except Exception:
        return False
    ip = get_ipython()
    # ``ZMQInteractiveShell`` is the kernel-backed frontend; a plain terminal
    # IPython REPL is ``TerminalInteractiveShell`` and is already a real TTY.
    return ip is not None and type(ip).__name__ == "ZMQInteractiveShell"


def _terminal_width() -> int:
    """Detected terminal width, clamped to a readable range."""
    width = shutil.get_terminal_size((80, 24)).columns
    return max(40, min(width, 100))


def _display_mode(stream: TextIO) -> tuple[bool, int | None]:
    """Resolve ``(color, wrap_width)`` for ``stream``.

    ``wrap_width`` is an int to hard-wrap to, or ``None`` to skip hard-wrapping
    and let the surface soft-wrap to its own (live) width — used in notebooks,
    where there is no terminal size to read.
    """
    try:
        is_tty = bool(stream.isatty())
    except Exception:
        is_tty = False
    notebook = _in_notebook()

    color = True
    if os.environ.get("NO_COLOR") is not None or os.environ.get("TERM") == "dumb":
        color = False
    elif not (is_tty or notebook):
        color = False  # plain pipe/file: no colour

    if notebook and not is_tty:
        return color, None  # let the cell flow to its own width
    return color, _terminal_width()


def _render(
    message: object,
    category: type[Warning],
    filename: str,
    lineno: int,
    color: bool,
    wrap_width: int | None,
) -> str:
    """Build the structured (optionally colourised) warning block.

    ``wrap_width`` is the column to hard-wrap the message at, or ``None`` to
    leave each paragraph as a single line so the display soft-wraps it itself.
    """
    cat_name = getattr(category, "__name__", str(category))
    text = str(message).strip()
    location = f"{_friendly_path(filename)}:{lineno}"

    if color:
        accent = _CATEGORY_STYLE.get(cat_name, _YELLOW)
        header = f"{_BOLD}{accent}⚠ rs-embed{_RESET} {_DIM}{cat_name}{_RESET}"
        bar = f"{accent}│{_RESET}"
        tail_glyph = f"{accent}╰{_RESET}"
        loc = f"{tail_glyph} {_DIM}{location}{_RESET}"
    else:
        header = f"⚠ rs-embed · {cat_name}"
        bar = "│"
        loc = f"╰ {location}"

    paragraphs = text.splitlines() or [""]
    if wrap_width is None:
        # Soft-flow: one line per paragraph; the terminal/notebook wraps it to
        # its own live width. Long lines have no hard breaks to go stale.
        wrapped = paragraphs
    else:
        # Hard-wrap to the detected width, leaving room for the "│ " gutter.
        wrapped = []
        for paragraph in paragraphs:
            wrapped.extend(textwrap.wrap(paragraph, width=wrap_width - 2) or [""])

    # Highlight code-like tokens *after* wrapping: ANSI escapes would otherwise
    # count toward textwrap's width budget and break lines in the wrong place.
    if color:
        wrapped = [_highlight(line) for line in wrapped]

    body = "\n".join(f"{bar} {line}" for line in wrapped)
    return f"{header}\n{body}\n{loc}\n"


def _showwarning(
    message: object,
    category: type[Warning],
    filename: str,
    lineno: int,
    file: TextIO | None = None,
    line: str | None = None,
) -> None:
    """Replacement for :func:`warnings.showwarning`.

    Restyles rs_embed-originating warnings; everything else is delegated to the
    handler that was installed before us (falling back to Python's default).
    """
    if not _is_rs_embed_warning(filename):
        delegate = _previous_showwarning or _default_show
        delegate(message, category, filename, lineno, file, line)
        return

    stream = file if file is not None else sys.stderr
    try:
        color, wrap_width = _display_mode(stream)
        block = _render(message, category, filename, lineno, color, wrap_width)
        stream.write(block)
    except Exception:
        # Never let pretty-printing swallow a warning: fall back to default.
        _default_show(message, category, filename, lineno, file, line)


# ---------------------------------------------------------------------------
# Jupyter/IPython: keep our warnings from being swallowed on cell re-run.
#
# IPython clears only ``__main__``'s ``__warningregistry__`` between cells, so a
# warning emitted from inside rs_embed shows once per *kernel session* instead of
# once per cell run (the second run is silently de-duplicated). We mirror what
# IPython does for user code by clearing our own modules' registries before each
# cell — scoped to rs_embed, so other libraries' de-duplication is untouched and
# warnings still de-duplicate within a single cell execution.
# ---------------------------------------------------------------------------
_notebook_hook_installed = False


def _reset_rs_embed_warning_registry() -> None:
    """Drop stale per-module warning de-dup state for all ``rs_embed`` modules."""
    for name, module in list(sys.modules.items()):
        if module is None:
            continue
        if name == "rs_embed" or name.startswith("rs_embed."):
            try:
                module.__dict__.pop("__warningregistry__", None)
            except Exception:
                pass


def _pre_run_cell(*_args, **_kwargs) -> None:
    """IPython ``pre_run_cell`` callback (accepts the optional info argument)."""
    _reset_rs_embed_warning_registry()


def _install_notebook_registry_reset() -> None:
    """Register the per-cell registry reset when running under IPython/Jupyter."""
    global _notebook_hook_installed
    if _notebook_hook_installed:
        return
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is None or not hasattr(ip, "events"):
            return
        ip.events.register("pre_run_cell", _pre_run_cell)
        _notebook_hook_installed = True
    except Exception:
        pass


def _uninstall_notebook_registry_reset() -> None:
    global _notebook_hook_installed
    if not _notebook_hook_installed:
        return
    try:
        from IPython import get_ipython

        ip = get_ipython()
        if ip is not None and hasattr(ip, "events"):
            ip.events.unregister("pre_run_cell", _pre_run_cell)
    except Exception:
        pass
    _notebook_hook_installed = False


def enable_pretty_warnings() -> None:
    """Install the structured warning renderer (idempotent).

    No-op when ``RS_EMBED_PLAIN_WARNINGS`` is set to a truthy value. Under
    IPython/Jupyter this also registers a ``pre_run_cell`` hook so rs_embed
    warnings are not silently swallowed when a cell is re-run.
    """
    global _previous_showwarning

    if os.environ.get("RS_EMBED_PLAIN_WARNINGS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }:
        return

    # Install the cell hook first (and unconditionally): the renderer may have
    # been installed at import time before the IPython shell existed, so we still
    # want a later call to attach the hook. Both steps are individually idempotent.
    _install_notebook_registry_reset()

    if getattr(warnings.showwarning, "_rs_embed_pretty", False):
        return  # renderer already installed

    _previous_showwarning = warnings.showwarning
    _showwarning._rs_embed_pretty = True  # type: ignore[attr-defined]
    warnings.showwarning = _showwarning


def disable_pretty_warnings() -> None:
    """Restore the warning handler that was active before :func:`enable_pretty_warnings`."""
    global _previous_showwarning

    if getattr(warnings.showwarning, "_rs_embed_pretty", False):
        warnings.showwarning = _previous_showwarning or _default_show
    _previous_showwarning = None
    _uninstall_notebook_registry_reset()
