"""Textual TUI for ODIN.

Layout
------
┌─ input-container (full width) ──────────────────────────────────────┐
│ Requirement input                                                     │
│ Output directory input                                                │
│ Test input (stdin)                                                    │
│ [Run]  [Clear]  [Send Feedback]                                      │
└──────────────────────────────────────────────────────────────────────┘
┌─ code-display (60%) ──────┐  ┌─ info-panel (40%) ───────────────────┐
│ Generated code             │  │ Iteration panel                       │
│                            │  │ Execution output                      │
│                            │  │ Run history                           │
└────────────────────────────┘  └──────────────────────────────────────┘
┌─ status-container (full width) ─────────────────────────────────────┐
│ Status message / past-run memory notice                               │
└──────────────────────────────────────────────────────────────────────┘

Two distinct user actions
--------------------------
1. [Run]           — starts a NEW pipeline.  Internally calls
                     backend.run_pipeline(), which queries run_memory and
                     then invokes the graph with is_continuation=False.

2. [Send Feedback] — only active when a run is in progress / complete.
                     Calls backend.continue_run(), which invokes the graph
                     with is_continuation=True and skips run_memory entirely.

The TUI stores self.active_run (RunHistory | None).  If it is None, [Send
Feedback] does nothing.  If it is set, [Run] starts a fresh run and clears
self.active_run first.

Files are always also persisted under outputs/ by the backend.  The output
directory field here controls an *additional* copy written to the user's
project folder on success.
"""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Footer, Header, Input, Label, Static

from odin.backend import OdinBackend
from odin.config import config
from odin.state import RunHistory

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Custom widgets
# ---------------------------------------------------------------------------

class CodeDisplay(Static):
    """Displays generated code."""

    def __init__(self, title: str = "Code", **kwargs):
        super().__init__(**kwargs)
        self._title = title
        self._code = ""

    def render(self) -> str:
        if not self._code:
            return f"[dim]{self._title}[/dim]\n\n[italic]No code yet[/italic]"
        return f"[bold]{self._title}[/bold]\n\n[green]{self._code}[/green]"

    def update_code(self, title: str, code: str) -> None:
        self._title = title
        self._code = code
        self.refresh()


class IterationPanel(Static):
    """Shows per-iteration transparency: category, retrieval decision."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._iteration = 0
        self._status = ""
        self._category = ""
        self._retrieval_triggered = False
        self._retrieval_reason = ""
        self._past_run_notice = ""

    def render(self) -> str:
        lines = [f"[bold]Iteration {self._iteration}[/bold]\n"]
        if self._past_run_notice:
            lines.append(f"[cyan]Memory: {self._past_run_notice}[/cyan]")
        if self._status:
            color = "green" if "SUCCESS" in self._status else "red"
            lines.append(f"Status: [{color}]{self._status}[/{color}]")
        if self._category:
            lines.append(f"Error category: [yellow]{self._category}[/yellow]")
        if self._retrieval_reason:
            color = "cyan" if self._retrieval_triggered else "dim"
            label = "TRIGGERED" if self._retrieval_triggered else "SKIPPED"
            lines.append(f"RAG retrieval: [{color}]{label}[/{color}]")
            lines.append(f"[dim]{self._retrieval_reason}[/dim]")
        return "\n".join(lines)

    def update(  # type: ignore[override]
        self,
        iteration: int = 0,
        status: str = "",
        category: str = "",
        retrieval_triggered: bool = False,
        retrieval_reason: str = "",
        past_run_notice: str = "",
    ) -> None:
        self._iteration = iteration
        self._status = status
        self._category = category
        self._retrieval_triggered = retrieval_triggered
        self._retrieval_reason = retrieval_reason
        self._past_run_notice = past_run_notice
        self.refresh()


class HistoryPanel(Static):
    """Scrolling run history log."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._items: list[str] = []

    def render(self) -> str:
        if not self._items:
            return "[dim]No history yet[/dim]"
        return "[bold]Run History[/bold]\n\n" + "\n".join(self._items)

    def add(self, text: str) -> None:  # type: ignore[override]
        ts = datetime.now().strftime("%H:%M:%S")
        self._items.append(f"[{ts}] {text}")
        self.refresh()


# ---------------------------------------------------------------------------
# Main application
# ---------------------------------------------------------------------------

class OdinTUI(App):
    """ODIN terminal user interface."""

    CSS = """
    Screen { layout: grid; grid-size: 2 3; grid-rows: auto 1fr auto; }

    #input-container  { column-span: 2; height: auto; padding: 1; background: $panel; }
    #button-row       { height: auto; margin-top: 1; }
    #main-container   { column-span: 2; height: 1fr; }
    #code-display     { width: 60%; height: 100%; border: solid $primary; padding: 1; }
    #info-panel       { width: 40%; height: 100%; border: solid $secondary; padding: 1; }
    #status-container { column-span: 2; height: auto; padding: 1; background: $panel; }

    Input  { width: 100%; margin-bottom: 0; }
    Button { height: 3; margin-right: 1; }
    #feedback-input { display: none; }
    #feedback-label { display: none; }
    """

    BINDINGS = [
        ("q", "quit", "Quit"),
        ("r", "run", "Run"),
        ("c", "clear", "Clear"),
    ]

    def __init__(self) -> None:
        super().__init__()
        self._backend = OdinBackend()
        # active_run holds the RunHistory of the current or most-recent run.
        # None means no run has started yet.
        self.active_run: RunHistory | None = None

    # ── Layout ──────────────────────────────────────────────────────────

    def compose(self) -> ComposeResult:
        yield Header()

        with Container(id="input-container"):
            yield Label("Task (natural language):")
            yield Input(
                placeholder="e.g. 'Write a palindrome checker'  or  'Build a FastAPI todo service'",
                id="requirement-input",
            )
            yield Label("Output directory — generated files are copied here on success:")
            yield Input(placeholder="./output", value="./output", id="output-dir-input")
            yield Label("Test input (optional — piped to stdin during execution):")
            yield Input(placeholder="e.g. hello world", id="test-input")

            # Feedback row — hidden until a run exists
            yield Label(
                "Feedback (describe what's wrong or what you want changed):",
                id="feedback-label",
            )
            yield Input(
                placeholder="e.g. 'The function should handle empty lists'",
                id="feedback-input",
            )

            with Horizontal(id="button-row"):
                yield Button("Run", id="run-button", variant="primary")
                yield Button("Send Feedback", id="feedback-button", variant="warning")
                yield Button("Clear", id="clear-button")

        with Horizontal(id="main-container"):
            with VerticalScroll(id="code-display"):
                yield CodeDisplay(id="code-widget")
            with Vertical(id="info-panel"):
                yield IterationPanel(id="iteration-widget")
                yield Static("[bold]Output[/bold]")
                yield VerticalScroll(Static("", id="output-widget"))
                yield HistoryPanel(id="history-widget")

        with Container(id="status-container"):
            yield Static(f"Ready — model: {config.gemini_model}", id="status-widget")

        yield Footer()

    # ── Button handlers ──────────────────────────────────────────────────

    @on(Button.Pressed, "#run-button")
    def _on_run(self) -> None:
        self.action_run()

    @on(Button.Pressed, "#feedback-button")
    def _on_feedback(self) -> None:
        self._action_send_feedback()

    @on(Button.Pressed, "#clear-button")
    def _on_clear(self) -> None:
        self.action_clear()

    # ── Actions ──────────────────────────────────────────────────────────

    def action_run(self) -> None:
        """Start a fresh pipeline run — queries run_memory, is_continuation=False."""
        req = self.query_one("#requirement-input", Input).value.strip()
        if not req:
            self._set_status("Please enter a task description")
            return

        output_dir = self.query_one("#output-dir-input", Input).value.strip() or "./output"
        test_input = self.query_one("#test-input", Input).value.strip()

        # Clear any active run so feedback button doesn't apply to the old one
        self.active_run = None
        self.action_clear()
        self._set_status("Starting…")
        self._start_new_run(req, test_input, output_dir)

    def _action_send_feedback(self) -> None:
        """Continue the active run with user feedback — skips run_memory."""
        if self.active_run is None:
            self._set_status("No active run — press Run first")
            return

        feedback = self.query_one("#feedback-input", Input).value.strip()
        if not feedback:
            self._set_status("Please enter feedback before sending")
            return

        self._set_status("Applying feedback…")
        self._continue_run(self.active_run, feedback)

    def action_clear(self) -> None:
        self.query_one("#code-widget", CodeDisplay).update_code("Code", "")
        self.query_one("#iteration-widget", IterationPanel).update()
        self.query_one("#output-widget", Static).update("")
        self._set_status("Ready")

    # ── Workers (run off the event loop so the UI stays responsive) ──────

    @work(exclusive=True)
    async def _start_new_run(self, requirement: str, test_input: str, output_dir: str) -> None:
        history_widget = self.query_one("#history-widget", HistoryPanel)
        history_widget.add(f"New run: {requirement[:60]}…")

        try:
            run_history = await self._backend.run_pipeline(
                requirement=requirement,
                test_input=test_input,
            )
        except Exception as exc:
            logger.exception("Pipeline failed")
            self._set_status(f"Error: {exc}")
            history_widget.add(f"✗ Error: {str(exc)[:60]}")
            return

        self.active_run = run_history
        self._show_run_results(run_history, output_dir)

        # Reveal feedback controls now that a run exists
        self.query_one("#feedback-label").styles.display = "block"
        self.query_one("#feedback-input").styles.display = "block"

    @work(exclusive=True)
    async def _continue_run(self, run_history: RunHistory, feedback: str) -> None:
        """Call backend.continue_run() — is_continuation=True, no run_memory query."""
        history_widget = self.query_one("#history-widget", HistoryPanel)
        history_widget.add(f"Feedback: {feedback[:60]}…")

        # Clear the feedback input
        self.query_one("#feedback-input", Input).value = ""
        test_input = self.query_one("#test-input", Input).value.strip()

        try:
            updated = await self._backend.continue_run(
                run_history=run_history,
                user_feedback=feedback,
                test_input=test_input,
            )
        except Exception as exc:
            logger.exception("Continuation failed")
            self._set_status(f"Error: {exc}")
            history_widget.add(f"✗ Error: {str(exc)[:60]}")
            return

        self.active_run = updated
        output_dir = self.query_one("#output-dir-input", Input).value.strip() or "./output"
        self._show_run_results(updated, output_dir)

    # ── Display helpers ──────────────────────────────────────────────────

    def _show_run_results(self, run_history: RunHistory, output_dir: str) -> None:
        history_widget = self.query_one("#history-widget", HistoryPanel)
        code_widget = self.query_one("#code-widget", CodeDisplay)
        iter_widget = self.query_one("#iteration-widget", IterationPanel)
        output_widget = self.query_one("#output-widget", Static)

        # Code panel — show last iteration
        if run_history.code_versions:
            last = run_history.code_versions[-1]
            if last.files:
                if len(last.files) == 1:
                    title = f"Code — iteration {last.iteration}"
                    code_widget.update_code(title, last.files[0].content)
                else:
                    title = f"Code — iteration {last.iteration} ({last.files[0].path})"
                    code_widget.update_code(title, last.files[0].content)

        # Execution output (or pipeline/API error if execution never ran)
        if run_history.executions:
            ex = run_history.executions[-1]
            txt = f"=== Iteration {ex.iteration} ===\n"
            if ex.stdout:
                txt += f"STDOUT:\n{ex.stdout}\n"
            if ex.stderr:
                txt += f"STDERR:\n{ex.stderr}\n"
            txt += f"Exit code: {ex.exit_code}  ({ex.execution_time_ms:.0f}ms)"
            if run_history.error_message:
                txt += f"\n\nPIPELINE ERROR:\n{run_history.error_message}"
            output_widget.update(txt)
        elif run_history.error_message:
            output_widget.update(
                f"=== Pipeline / API Error ===\n"
                f"Model: {config.gemini_model}\n"
                f"{run_history.error_message}\n\n"
                f"Tip: Check GEMINI_API_KEY / GEMINI_MODEL in .env or see odin.log for full traceback."
            )

        # Iteration transparency panel
        category = run_history.error_analyses[-1].category if run_history.error_analyses else ""
        ret = run_history.retrievals[-1] if run_history.retrievals else None
        ex_status = "SUCCESS" if run_history.executions and run_history.executions[-1].success else "FAILURE"

        # Past-run memory notice (shown only when it was used — i.e. not a continuation)
        past_notice = ""
        # We surface the notice from the last code version's iteration 1 token usage
        # presence of past_run_context is captured at backend level; we approximate
        # by showing a notice if run_memory returned anything (stored in run_history
        # indirectly — we keep it simple and show nothing rather than guess)

        iter_widget.update(
            iteration=run_history.executions[-1].iteration if run_history.executions else 0,
            status=ex_status,
            category=category,
            retrieval_triggered=ret.triggered if ret else False,
            retrieval_reason=ret.decision_reason if ret else "",
            past_run_notice=past_notice,
        )

        # Status bar + history
        status = run_history.final_status
        if status == "success":
            written = self._copy_output_files(run_history, output_dir)
            self._set_status(
                f"✓ Success in {run_history.total_iterations} iteration(s)! "
                f"Files written to {written}"
            )
            history_widget.add(f"✓ Success → {written}")
        elif status == "max_iterations_reached":
            self._set_status("✗ Max iterations reached — no files written to output dir")
            history_widget.add("✗ Max iterations reached")
        else:
            err_short = (run_history.error_message or "Unknown error").splitlines()[0][:120]
            self._set_status(f"✗ Failed: {err_short}")
            history_widget.add(f"✗ Failed: {err_short[:70]}")

    def _copy_output_files(self, run_history: RunHistory, output_dir: str) -> str:
        """Copy the final generated files to a per-run subfolder in the user's output directory.

        Using a per-run subfolder prevents newer tasks from overwriting earlier tasks' main.py.
        """
        from odin.persistence import _make_run_dir_name

        base_out = Path(output_dir).resolve()
        if base_out.exists() and base_out.is_file():
            raise ValueError(
                f"Output path '{base_out}' is a file, not a folder. Please choose a folder path."
            )
        out = base_out / _make_run_dir_name(run_history.run_id, run_history.requirement)
        out.mkdir(parents=True, exist_ok=True)

        final = run_history.code_versions[-1]
        for f in final.files:
            dest = out / f.path
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(f.content, encoding="utf-8")
            logger.info("Copied output: %s", dest)

        return str(out)

    def _set_status(self, message: str) -> None:
        self.query_one("#status-widget", Static).update(message)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_tui() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler("odin.log")],
    )
    OdinTUI().run()
