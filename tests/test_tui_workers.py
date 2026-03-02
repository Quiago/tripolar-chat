"""Tests for Textual TUI worker correctness.

In Textual 8.x, `call_from_thread` lives on the *App* instance, not on
Screen.  These tests inspect the source code to catch the common mistake of
writing `self.call_from_thread(...)` inside a Screen subclass.

Expected status before fix: FAIL
Expected status after fix:  PASS
"""

import ast
import pathlib
import textwrap


_TUI_PATH = pathlib.Path(__file__).parent.parent / "client" / "tui.py"


def _get_class_source(class_name: str) -> str:
    """Extract the source of a named class from tui.py using the AST."""
    source = _TUI_PATH.read_text()
    tree = ast.parse(source)
    lines = source.splitlines()

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            start = node.lineno - 1
            end = node.end_lineno
            return "\n".join(lines[start:end])

    raise AssertionError(f"Class {class_name!r} not found in {_TUI_PATH}")


def _bare_call_from_thread_count(class_source: str) -> int:
    """Return number of bare 'self.call_from_thread' occurrences.

    'self.app.call_from_thread' does NOT match because 'self.app.' and
    'self.' are different prefixes; no subtraction is needed.
    """
    import re

    return len(re.findall(r"self\.call_from_thread", class_source))


# ── ModelSelectScreen ─────────────────────────────────────────────────────────

def test_model_select_screen_no_bare_call_from_thread():
    """ModelSelectScreen must use self.app.call_from_thread, not self.call_from_thread.

    Textual 8.x removed call_from_thread from Screen; it now lives on App.
    Calling self.call_from_thread inside a Screen raises AttributeError at runtime.
    """
    source = _get_class_source("ModelSelectScreen")
    bad_count = _bare_call_from_thread_count(source)
    assert bad_count == 0, (
        f"Found {bad_count} bare self.call_from_thread call(s) in ModelSelectScreen. "
        "Replace with self.app.call_from_thread(...)."
    )


# ── HistoryListScreen ─────────────────────────────────────────────────────────

def test_history_list_screen_no_bare_call_from_thread():
    """HistoryListScreen must use self.app.call_from_thread, not self.call_from_thread.

    Same Textual 8.x constraint as ModelSelectScreen.
    """
    source = _get_class_source("HistoryListScreen")
    bad_count = _bare_call_from_thread_count(source)
    assert bad_count == 0, (
        f"Found {bad_count} bare self.call_from_thread call(s) in HistoryListScreen. "
        "Replace with self.app.call_from_thread(...)."
    )
