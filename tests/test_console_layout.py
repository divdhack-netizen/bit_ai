"""Unit tests for the responsive TTY layout: ANSI truncation, metrics, sync."""

from __future__ import annotations

import os
import sys
import threading
import time
from io import StringIO
from unittest.mock import Mock, call, patch

import pytest
from rich.cells import cell_len
from rich.console import Console
from rich.markdown import Markdown
from rich.text import Text

from assistant.console import (
    ThermalBar,
    ConsoleUI,
    RendererPanelStream,
    TurnTimeline,
    tool_activity_sentence,
)
from assistant.splash import (
    FRAME_INTERVAL_SEC,
    SPLASH_MESSAGE,
    POST_ANIMATION_PAUSE_SEC,
    GREETING_PAUSE_SEC,
)
from assistant.metrics import LLMTurnStats, format_stats_segment
from assistant.thermal import ThermalState, status_bar_line
from assistant.tty_layout import (
    MARKDOWN_READING_WIDTH,
    TTY_WIDTH_MARGIN,
    COMPOSER_PREFIX,
    composer_height,
    sticky_composer_height,
    clear_lines_above,
    panel_cursor_column,
    cuu_clear_composer,
    cuu_clear_panel_input,
    content_cursor_geometry,
    panel_inner_width,
    reading_width,
    visible_width,
    session_header_markup,
    sticky_composer_start_line,
    lines_from_content_to_top_border,
    panel_rise_lines,
    visual_content_lines,
    scroll_region_sequence,
    sync_dimensions,
    composer_line,
    adaptive_panel_title,
    truncate_visible,
    tty_size,
)

pytestmark = pytest.mark.unit


def _sample_metrics() -> LLMTurnStats:
    """Realistic snapshot for the bar's progressive layout."""
    return LLMTurnStats(
        token_input=12_345,
        token_output=678,
        tokens_per_second=42.5,
        used_context=3_072,
        max_context=32_768,
        llm_calls=2,
        ttft_ms=120.0,
        llm_duration_ms=1500.0,
        tool_duration_ms=80.0,
    )


def _create_isolated_renderer(
    on_render: Mock | None = None,
) -> tuple[RendererPanelStream, Mock]:
    """
    Create a renderer with in-memory output and an observable timeline.

    Args:
        on_render: Optional callback used to count the actual refreshes.

    Returns:
        Renderer under test and the ReAct timeline mock.
    """
    console = Console(
        file=StringIO(),
        width=80,
        height=24,
        force_terminal=False,
    )
    timeline = Mock(spec=TurnTimeline)
    return RendererPanelStream(console, timeline, on_render=on_render), timeline


def _live_body(renderable: object) -> object:
    """Extract Text/Markdown from the stream's Group(title, Constrain(body))."""
    renderables = getattr(renderable, "renderables", None)
    if not renderables:
        return getattr(renderable, "renderable", renderable)
    inner = renderables[-1]
    return getattr(inner, "renderable", inner)


def test_start_direct_answer_opens_panel_without_react_markers() -> None:
    """The direct pipeline must enter Final Answer mode immediately."""
    live = Mock()
    renderer, _ = _create_isolated_renderer()

    with (
        patch("assistant.console.Live", return_value=live),
        patch("assistant.console.sync_dimensions"),
        patch("assistant.console.time.monotonic", return_value=0.0),
    ):
        renderer.start_direct_answer()
        renderer.write_token("Risposta senza Thought")

    assert live.start.called
    assert "".join(renderer._final_buffer) == "Risposta senza Thought"


def test_stream_shows_first_plain_text_chunk_immediately() -> None:
    """The first content opens Live already populated, without waiting for a periodic refresh."""
    live = Mock()
    clock = Mock(return_value=10.0)
    renderer, _ = _create_isolated_renderer()

    with (
        patch("assistant.console.Live", return_value=live) as create_live,
        patch("assistant.console.sync_dimensions"),
        patch("assistant.console.time.monotonic", clock),
    ):
        renderer.write_token("Final Ans")
        create_live.assert_not_called()

        renderer.write_token("wer: **ciao**")

    initial_panel = create_live.call_args.args[0]
    body = _live_body(initial_panel)
    assert isinstance(body, Text)
    assert body.plain == "**ciao**"
    assert create_live.call_args.kwargs["auto_refresh"] is False
    assert isinstance(renderer._final_buffer, list)
    assert renderer._final_buffer == ["**ciao**"]
    live.start.assert_called_once_with(refresh=True)
    live.update.assert_not_called()


def test_stream_limits_refresh_and_bar_without_sleep() -> None:
    """Chunks on the decode thread do not call Live.refresh: the worker does."""
    live = Mock()
    on_render = Mock()
    renderer, _ = _create_isolated_renderer(on_render)

    with (
        patch("assistant.console.Live", return_value=live),
        patch("assistant.console.sync_dimensions"),
    ):
        renderer.write_token("Final Answer: a")
        for _ in range(40):
            renderer.write_token("x")
        time.sleep(0.12)
        assert "".join(renderer._final_buffer) == f"a{'x' * 40}"
        renderer.finalize("ax")

    assert live.refresh.call_count >= 1
    assert live.refresh.call_count <= 8
    live.update.assert_called()


def test_stream_creates_markdown_once_on_finalize() -> None:
    """Chunks stay Text and the full content becomes Markdown only at the end."""
    live = Mock()
    clock = Mock(return_value=0.0)
    renderer, _ = _create_isolated_renderer()

    with (
        patch("assistant.console.Live", return_value=live) as create_live,
        patch("assistant.console.sync_dimensions"),
        patch("assistant.console.time.monotonic", clock),
        patch("assistant.console.Markdown", wraps=Markdown) as create_markdown,
    ):
        renderer.write_token("Final Answer: **ciao")
        clock.return_value = 0.01
        renderer.write_token("** mondo")

        assert create_markdown.call_count == 0
        assert isinstance(_live_body(create_live.call_args.args[0]), Text)

        renderer.finalize("Final Answer: fallback")

    create_markdown.assert_called_once()
    final_panel = live.update.call_args.args[0]
    body = _live_body(final_panel)
    assert isinstance(body, Markdown)
    assert body.markup == "**ciao** mondo"
    assert live.update.call_args.kwargs["refresh"] is True
    live.stop.assert_called_once()
    assert renderer._final_buffer == []


def _visible_screen(raw: str) -> str:
    """Apply CR, erase-line and SGR as the TTY would on the test capture."""
    lines = [""]
    index = 0
    while index < len(raw):
        if raw.startswith("\033[K", index):
            lines[-1] = ""
            index += 3
            continue
        if raw.startswith("\r", index):
            lines[-1] = ""
            index += 1
            continue
        if raw.startswith("\n", index):
            lines.append("")
            index += 1
            continue
        if raw.startswith("\033[", index):
            end = raw.find("m", index)
            if end != -1:
                index = end + 1
                continue
        lines[-1] += raw[index]
        index += 1
    return "\n".join(lines)


def test_stream_overflow_switches_to_free_scroll() -> None:
    """If Live fills the TTY, later tokens go to the scrollback."""
    live = Mock()
    live._lock = threading.RLock()
    live._nested = False
    live._started = True
    live.console.is_terminal = False
    live._live_render.last_render_height = 24
    renderer, _ = _create_isolated_renderer()

    with (
        patch("assistant.console.Live", return_value=live),
        patch("assistant.console.sync_dimensions"),
        patch("assistant.console.time.monotonic", return_value=0.0),
    ):
        renderer.write_token("Final Answer: inizio")
        renderer._run_live_refresh()
        renderer.write_token(" coda")
        renderer.finalize("Final Answer: inizio coda")

    assert live.stop.called
    assert renderer._live is None
    assert renderer._free_scroll is False
    capture = _visible_screen(renderer._console.file.getvalue())
    assert "inizio" in capture
    assert "coda" in capture
    live.update.assert_not_called()


def test_short_finalize_stays_in_live() -> None:
    """An answer that fits the viewport does not suspend the scroll region."""
    live = Mock()
    order: list[str] = []
    live.stop.side_effect = lambda: order.append("stop")
    console = Console(file=StringIO(), width=80, height=24, force_terminal=False)
    timeline = Mock(spec=TurnTimeline)
    renderer = RendererPanelStream(
        console,
        timeline,
        suspend_scroll=lambda: order.append("sospendi"),
        restore_scroll=lambda: order.append("ripristina"),
    )
    with (
        patch("assistant.console.Live", return_value=live),
        patch("assistant.console.sync_dimensions"),
        patch("assistant.console.time.monotonic", return_value=0.0),
    ):
        renderer.write_token("Final Answer: testo")
        renderer.finalize("Final Answer: testo")
    assert order == ["stop"]
    live.update.assert_called()


def test_tall_finalize_prints_outside_scroll_region() -> None:
    """If the markdown does not fit, Live.stop() is not used: the start would stay cut."""
    live = Mock()
    live._lock = threading.RLock()
    live.console.is_terminal = False
    order: list[str] = []
    console = Console(file=StringIO(), width=20, height=3, force_terminal=False)
    timeline = Mock(spec=TurnTimeline)
    renderer = RendererPanelStream(
        console,
        timeline,
        suspend_scroll=lambda: order.append("sospendi"),
        restore_scroll=lambda: order.append("ripristina"),
    )
    with (
        patch("assistant.console.Live", return_value=live),
        patch("assistant.console.sync_dimensions"),
        patch("assistant.console.time.monotonic", return_value=0.0),
    ):
        renderer.write_token("Final Answer: uno due tre quattro cinque sei sette otto")
        renderer.finalize(
            "Final Answer: uno due tre quattro cinque sei sette otto nove dieci"
        )
    assert order == ["sospendi", "ripristina"]
    assert "uno" in console.file.getvalue()


def test_thinking_wraps_on_words() -> None:
    """CoT does not split words at the TTY edge."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=12,
        height=24,
        force_terminal=True,
        color_system=None,
    )
    timeline = TurnTimeline(console)
    timeline.write_thinking("alfa bravo charlie")
    timeline.close_thinking()
    text = buffer.getvalue()
    assert "alfabravo" not in text.replace("\n", "")
    assert "charlie" in text


def test_free_scroll_does_not_cut_words_at_edge() -> None:
    """Answers taller than the TTY stay whole: wrapped on words, within the width."""
    live = Mock()
    live._lock = threading.RLock()
    live._nested = False
    live._started = True
    live.console.is_terminal = False
    live._live_render.last_render_height = 24
    renderer, _ = _create_isolated_renderer()
    words = [f"w{index:02d}" for index in range(40)]
    paragraph = " ".join(words)

    with (
        patch("assistant.console.Live", return_value=live),
        patch("assistant.console.sync_dimensions"),
        patch("assistant.console.time.monotonic", return_value=0.0),
    ):
        renderer.write_token("Final Answer: " + paragraph)
        renderer._run_live_refresh()
        renderer.finalize("Final Answer: " + paragraph)

    visible = _visible_screen(renderer._console.file.getvalue())
    lines = [line for line in visible.splitlines() if line.strip()]
    assert lines
    assert all(cell_len(line) <= renderer._console.width for line in lines)
    for word in words:
        assert word in visible
    for line in lines:
        for token in line.split():
            if token.startswith("w") and token[1:].isdigit():
                assert token in words


def test_stream_preserves_react_parser_with_split_markers() -> None:
    """Thought, Action, Action Input and Final Answer stay valid across chunks."""
    live = Mock()
    renderer, timeline = _create_isolated_renderer()

    with (
        patch("assistant.console.Live", return_value=live) as create_live,
        patch("assistant.console.sync_dimensions"),
        patch("assistant.console.time.monotonic", return_value=0.0),
    ):
        # The chosen boundaries intentionally split all the main markers.
        for react_fragment in (
            "Thou",
            "ght: analisi\nAct",
            "ion: cerca\nAction In",
            "put: query\nObservation: ok\nFinal Ans",
            "wer: fatto",
        ):
            renderer.write_token(react_fragment)

    timeline.emit_thinking.assert_has_calls([call("Thought: analisi")])
    timeline.emit_action_from_stream.assert_called_once_with("cerca")
    timeline.emit_action_input_from_stream.assert_called_once_with("query")
    timeline.flush_pending_action_input.assert_called_once()
    assert create_live.call_args.args[0].renderables[-1].renderable.plain == "fatto"


def test_truncate_visible_respects_cells_without_ansi() -> None:
    assert truncate_visible("abcdef", 3) == "abc\033[0m"
    assert visible_width(truncate_visible("abcdef", 3)) == 3


def test_truncate_visible_does_not_split_ansi_sequences() -> None:
    colored = "\033[32mNominal\033[0m-extra"
    result = truncate_visible(colored, 7)
    assert "\033[3" not in result.replace("\033[32m", "").replace("\033[0m", "")
    assert result.endswith("\033[0m")
    assert visible_width(result) <= 7


def test_truncate_visible_zero_or_negative_budget() -> None:
    assert truncate_visible("abc", 0) == ""
    assert truncate_visible("abc", -1) == ""


def test_full_metrics_without_cap() -> None:
    formatted = format_stats_segment(_sample_metrics())
    assert "42.5 tok/s" in formatted
    assert "token: 13 023" in formatted
    assert "ctx: 9%" in formatted
    assert "token in:" not in formatted
    assert "token out:" not in formatted
    assert "TTFT:" not in formatted
    verbose = format_stats_segment(_sample_metrics(), verbose=True)
    assert "LLM: 2/1500 ms" in verbose
    assert "TTFT: 120 ms" in verbose
    assert "tool: 80 ms" in verbose


def test_progressive_metrics_across_widths() -> None:
    stats = _sample_metrics()
    narrow = format_stats_segment(stats, max_width=40)
    average = format_stats_segment(stats, max_width=80)
    wide = format_stats_segment(stats, max_width=120)
    assert "tok/s" in narrow
    assert visible_width(narrow) <= 40
    assert visible_width(average) <= 80
    assert len(average) >= len(narrow)
    assert "ctx:" in wide
    assert "token in:" not in wide
    assert visible_width(wide) <= 120


def test_status_bar_line_respects_max_columns() -> None:
    line = status_bar_line(
        ThermalState.NOMINAL, 55.0, _sample_metrics(),
        ansi=False, max_columns=50,
    )
    assert visible_width(line) <= 50
    assert line.startswith("tmp: Nominal")


def test_status_bar_line_shows_low_power_mode() -> None:
    from assistant.host import PowerState

    line = status_bar_line(
        ThermalState.NOMINAL, 55.0, None,
        ansi=False, power=PowerState(True, True),
    )
    assert line.startswith("pwr: LPM")


def test_bar_line_prefix_only_if_too_narrow() -> None:
    line = status_bar_line(
        ThermalState.NOMINAL, 55.0, _sample_metrics(),
        ansi=False, max_columns=20,
    )
    assert "·" not in line
    assert visible_width(line) <= 20


def test_sync_dimensions_uses_anti_wrap_margin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(width=80, height=24, force_terminal=True)

    def _fake_size(_console: Console) -> tuple[int, int]:
        return 100, 40

    monkeypatch.setattr("assistant.tty_layout.tty_size", _fake_size)
    columns, lines = sync_dimensions(console, reserve_bar=True)
    assert columns == 100
    assert lines == 40
    assert console.width == 100 - TTY_WIDTH_MARGIN
    assert console.height == 39


def test_tty_size_prefers_ioctl_over_columns_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(width=80, height=24, force_terminal=True)

    def _fake_ioctl() -> tuple[int, int] | None:
        return 79, 34

    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setattr("assistant.tty_layout.read_ioctl_tty_size", _fake_ioctl)
    columns, lines = tty_size(console)
    assert columns == 79
    assert lines == 34


def test_tty_size_enlarge_ignores_narrow_columns(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(width=40, height=24, force_terminal=True)

    def _wide_ioctl() -> tuple[int, int] | None:
        return 93, 34

    monkeypatch.setenv("COLUMNS", "40")
    monkeypatch.setattr("assistant.tty_layout.read_ioctl_tty_size", _wide_ioctl)
    columns, lines = tty_size(console)
    assert columns == 93
    assert lines == 34


def test_sync_sets_atomic_size_and_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(width=80, height=24, force_terminal=True)

    def _fake_size(_console: Console) -> tuple[int, int]:
        return 79, 34

    monkeypatch.setattr("assistant.tty_layout.tty_size", _fake_size)
    columns, lines = sync_dimensions(console, reserve_bar=True)
    assert columns == 79
    assert lines == 34
    assert console.width == 79 - TTY_WIDTH_MARGIN
    assert console.height == 33
    assert console._width is not None and console._height is not None
    assert os.environ.get("COLUMNS") == "79"
    assert os.environ.get("LINES") == "34"


def test_adaptive_panel_title_truncates_long() -> None:
    markup = "[user.label]Tu[/] [dim]· default · q4 · gemma4:12b[/]"
    budget = 20
    result = adaptive_panel_title(markup, budget)
    assert cell_len(Text.from_markup(result).plain) <= budget


def test_adaptive_panel_title_unchanged_if_short() -> None:
    markup = "[agent.label]Assistente[/]"
    assert adaptive_panel_title(markup, 40) == markup


def test_apply_resize_invalidates_last_line_and_reserves() -> None:
    console = Console(width=80, height=24, force_terminal=True)
    bar = ThermalBar(console)
    bar._enable = True
    bar._last_line = "stale"
    bar._reserved_lines = 24
    calls: list[str] = []

    def _reserve() -> None:
        calls.append("reserve")
        bar._reserved_lines = 30

    def _redraw(**_kwargs) -> None:
        calls.append("redraw")

    with patch.object(bar, "_sync_console"), \
            patch.object(bar, "_reserve_last_line", side_effect=_reserve), \
            patch.object(bar, "redraw", side_effect=_redraw):
        bar._apply_resize()

    assert bar._last_line == ""
    assert "reserve" in calls
    assert "redraw" in calls


def test_bar_after_clear_invalidates_line() -> None:
    console = Console(width=80, height=24, force_terminal=True)
    bar = ThermalBar(console)
    bar._enable = True
    bar._last_line = "stale"
    with patch.object(bar, "_reserve_last_line") as reserve:
        bar.after_clear()
    reserve.assert_called_once()
    assert bar._last_line == ""


def test_handle_resize_sets_ui_flag() -> None:
    """A resize without Live signals the prompt reflow, without wiping the history."""
    ui = ConsoleUI()
    ui._ui_needs_redraw.clear()
    ui._stream._live_open = False
    ui._handle_resize()
    assert ui._ui_needs_redraw.is_set()


def test_handle_resize_live_queues_flag_without_reflow() -> None:
    """The bar thread queues the Live resize without calling Rich directly."""
    ui = ConsoleUI()
    ui._ui_needs_redraw.clear()
    ui._live_resize_requested.clear()
    ui._stream._live_open = True
    with patch.object(ui._stream, "reflow_on_resize") as reflow:
        ui._handle_resize()
    reflow.assert_not_called()
    assert not ui._ui_needs_redraw.is_set()
    assert ui._live_resize_requested.is_set()


def test_write_token_consumes_live_resize_on_main_thread() -> None:
    """The flag produced by a worker is consumed by write_token on the main thread."""
    ui = ConsoleUI()
    ui._stream._live_open = True
    thread_reflow: list[int | None] = []

    def _record_thread_reflow() -> None:
        """Store the thread that actually performs the Live reflow."""
        thread_reflow.append(threading.current_thread().ident)

    with (
        patch.object(ui._stream, "write_token") as write_stream,
        patch.object(
            ui._stream,
            "reflow_on_resize",
            side_effect=_record_thread_reflow,
        ),
    ):
        worker = threading.Thread(target=ui._handle_resize)
        worker.start()
        worker.join()

        assert thread_reflow == []
        ui.write_token("chunk")

    write_stream.assert_called_once_with("chunk")
    assert thread_reflow == [threading.main_thread().ident]
    assert not ui._live_resize_requested.is_set()


def test_finalize_consumes_resize_before_closing_live() -> None:
    """finalize applies the pending resize before replacing the final Panel."""
    ui = ConsoleUI()
    operation_order: list[str] = []
    ui._live_resize_requested.set()

    def _record_reflow() -> None:
        """Record the reflow to check its order relative to closing."""
        operation_order.append("resize")

    def _record_finalization(_result: str) -> None:
        """Record the simulated finalization without starting a real Rich Live."""
        operation_order.append("finalize")

    with (
        patch.object(
            ui._stream,
            "reflow_on_resize",
            side_effect=_record_reflow,
        ),
        patch.object(
            ui._stream,
            "finalize",
            side_effect=_record_finalization,
        ),
    ):
        ui.finalize("Final Answer: fatto")

    assert operation_order == ["resize", "finalize"]
    assert not ui._live_resize_requested.is_set()


def test_print_splash_centers_message() -> None:
    """The splash prints the greeting; the fake clock does not wait on the wall clock."""
    buffer = StringIO()
    ui = ConsoleUI()
    ui.console = Console(
        file=buffer,
        width=80,
        height=24,
        force_terminal=True,
        color_system=None,
    )
    fake_time = 0.0
    expected: list[float] = []

    def _monotonic() -> float:
        return fake_time

    def _recorded_pause(seconds: float) -> None:
        """Advance the fake clock and record the duration."""
        nonlocal fake_time
        expected.append(seconds)
        fake_time += seconds

    ui.print_splash(pause=_recorded_pause, monotonic=_monotonic)

    capture = buffer.getvalue()
    assert SPLASH_MESSAGE in capture
    assert expected[-2:] == [POST_ANIMATION_PAUSE_SEC, GREETING_PAUSE_SEC]
    assert all(delta == FRAME_INTERVAL_SEC for delta in expected[:-2])
    assert expected[:-2]


def test_print_splash_skips_non_terminal_console() -> None:
    """Pipes and captures get neither the splash nor the wait."""
    buffer = StringIO()
    ui = ConsoleUI()
    ui.console = Console(
        file=buffer,
        width=80,
        height=24,
        force_terminal=False,
        color_system=None,
    )
    expected: list[float] = []

    def _recorded_pause(seconds: float) -> None:
        """Must not be called outside a TTY."""
        expected.append(seconds)

    ui.print_splash(pause=_recorded_pause)

    assert buffer.getvalue() == ""
    assert expected == []


def test_welcome_body_is_compact() -> None:
    """The startup banner has intro, badges and /help, without the keyboard manual."""
    ui = ConsoleUI()
    body = ui._welcome_body({
        "verbose": True,
        "full_verification": False,
        "burn_protection": True,
        "thinking": False,
    })
    assert "Assistente pronto" in body
    assert "/help" in body
    assert "Stato:" not in body
    assert "Comandi:\n" not in body
    assert "/clear" not in body
    assert "Ctrl+W" not in body
    assert "solo dai documenti della cartella" in body
    assert "Verbose: [badge.on]on[/]" in body
    assert "Thinking: [badge.off]off[/]" in body
    assert "Web:" not in body


def test_help_body_contains_commands() -> None:
    """`/help` lists status, slash commands and the keyboard guide."""
    ui = ConsoleUI()
    body = ui._help_body({
        "verbose": True,
        "full_verification": False,
        "burn_protection": True,
        "thinking": False,
    })
    assert "Stato:" in body
    assert "Comandi:" in body
    assert "/clear" in body
    assert "/web" not in body
    assert "/archivio" not in body
    assert "Cartella documenti:" in body
    assert "/model" in body
    assert "/quant" not in body
    assert "/thinking" in body
    assert "/help" in body
    assert "Ctrl+W" in body
    assert "Ctrl+C" in body
    assert "Ctrl+V" in body
    assert "Option+frecce" in body
    assert "Selezione:" in body
    assert "Appunti:" in body
    assert "Verbose: [badge.on]on[/]" in body
    assert "Thinking: [badge.off]off[/]" in body
    assert "Web:" not in body


def test_reprint_chrome_does_not_clear_history(monkeypatch: pytest.MonkeyPatch) -> None:
    """The input reflow does not call console.clear() and does not reprint the welcome."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    ui = ConsoleUI()
    ui.console = Console(
        file=StringIO(),
        width=80,
        height=24,
        force_terminal=True,
        color_system=None,
    )
    ui._current_prompt_title = "Tu"
    ui._drawn_input_width = 80
    ui._welcome_state = {"verbose": False}
    with (
        patch.object(ui.console, "clear") as clear,
        patch("assistant.console.clear_lines_above") as delete,
        patch("assistant.console.print_composer") as composer,
        patch("assistant.console.sync_dimensions"),
        patch.object(ui.thermal_bar, "redraw"),
    ):
        ui._reprint_chrome("ciao", anchor=None, cursor=4)
    clear.assert_not_called()
    delete.assert_called_once()
    composer.assert_called_once()
    title = composer.call_args.args[1]
    assert "Assistente pronto" not in str(title)


def test_redraw_shows_metrics_even_without_verbose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tok/s and total tokens stay in the bar; verbose only adds the latencies."""
    console = Console(width=80, height=24, force_terminal=True)
    bar = ThermalBar(console)
    bar._enable = True
    bar._show_metrics = False
    bar._metrics = _sample_metrics()
    capture: dict[str, object] = {}

    def _fake_line(
        _state: object,
        _temp: object,
        metrics: object = None,
        **kwargs: object,
    ) -> str:
        capture["metriche"] = metrics
        capture["verbose"] = kwargs.get("verbose")
        return "tmp: Nominal"

    monkeypatch.setattr("assistant.console.status_bar_line", _fake_line)
    monkeypatch.setattr("assistant.console.tty_size", lambda _c: (80, 24))
    monkeypatch.setattr("assistant.console.read_thermal_state", lambda: ThermalState.NOMINAL)
    monkeypatch.setattr("assistant.console.read_cpu_temperature_celsius", lambda: 40.0)
    monkeypatch.setattr("assistant.console.detect_power", lambda: None)
    monkeypatch.setattr(bar, "_sync_console", lambda: None)
    monkeypatch.setattr(bar, "_draw", lambda _line: None)
    bar.redraw()
    assert capture["metriche"] is bar._metrics
    assert capture["verbose"] is False

    bar.set_show_metrics(True)
    bar.redraw()
    assert capture["metriche"] is bar._metrics
    assert capture["verbose"] is True


def test_thinking_closes_progress_line() -> None:
    """«Comprendo la richiesta» does not stay glued to the Pensiero label."""
    ui = ConsoleUI()
    buffer = StringIO()
    ui.console.file = buffer
    ui.console._force_terminal = True
    ui.thermal_bar.redraw = lambda *args, **kwargs: None
    ui.update_progress("Comprendo la richiesta")
    ui.close_progress()
    ui.record_tool_action("cerca_testo", "problemi odierni")
    ui.write_thinking("Thinking Process:")
    text = buffer.getvalue()
    assert "Comprendo la richiesta\n" in text
    assert "Comprendo la richiestaCerco" not in text
    assert "Comprendo la richiestaPensiero" not in text
    assert "Pensiero Thinking Process:" in text


def test_native_thinking_label_and_outside_panel() -> None:
    """The dim `Pensiero` stream is one continuous flow, outside the answer panel."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=80,
        height=24,
        force_terminal=True,
        color_system=None,
    )
    timeline = TurnTimeline(console)
    renderer = RendererPanelStream(console, timeline)
    timeline.write_thinking("ana")
    timeline.write_thinking("lisi")
    timeline.close_thinking()
    renderer.start_direct_answer()
    live = Mock()
    with (
        patch("assistant.console.Live", return_value=live),
        patch("assistant.console.sync_dimensions"),
        patch("assistant.console.time.monotonic", return_value=0.0),
    ):
        renderer.write_token("risposta")
    assert "".join(renderer._final_buffer) == "risposta"
    capture = buffer.getvalue()
    assert "Pensiero" in capture
    assert "analisi" in capture
    assert "caratteri" not in capture


def test_input_panel_cursor_geometry() -> None:
    assert panel_cursor_column(1) == 3
    assert panel_rise_lines(1, 0) == 2
    # While editing, the cursor is INSIDE the content: just 1 line up to the top border.
    assert lines_from_content_to_top_border(0) == 1
    assert lines_from_content_to_top_border(0, content_line=3) == 3
    assert panel_inner_width(119, 1) == 115


def test_clear_lines_above_emits_cr_before_cuu() -> None:
    """CR is mandatory: without it, ED 0 leaves the title and duplicates the Panel."""
    stdout = Mock()
    stdout.isatty.return_value = True

    with patch("assistant.tty_layout.sys.stdout", stdout):
        clear_lines_above(2, is_terminal=True)

    stdout.write.assert_called_once_with("\r\033[2A\033[J")
    stdout.flush.assert_called_once()


def test_content_geometry_and_cuu_clear() -> None:
    """A single API: visual lines, column at the end of the text, CUU on Enter."""
    assert visual_content_lines("", 16) == 1
    assert visual_content_lines("a" * 16, 16) == 1
    assert visual_content_lines("a" * 17, 16) == 2
    assert visual_content_lines("a" * 33, 16) == 3

    column, lines = content_cursor_geometry("a" * 20, 16)
    assert lines == 2
    assert column == 4

    # Console width 20 → inner 16; 33 cells → 3 CUU from the content to the border.
    assert cuu_clear_panel_input("a" * 33, 20) == 3
    assert cuu_clear_panel_input("", 20) == 1

    # Word wrap: last line = last word, not `len % width`.
    sentence = (
        "riusciresti a fornirmi i link per prendere una scheda grafica "
        "di ultima generazione e un buon processore? Forniscimi a"
    )
    words_column, word_lines = content_cursor_geometry(sentence, 111)
    assert word_lines == 2
    assert words_column == 12
    assert words_column != (cell_len(sentence) % 111)

    # Word wrap can take more lines than a fold on whole cells.
    ragged_column, ragged_lines = content_cursor_geometry(
        "uno due tre quattro",
        10,
    )
    assert ragged_lines == 3
    assert ragged_column == 7


def test_clipboard_local_cut_and_paste(monkeypatch: pytest.MonkeyPatch) -> None:
    from assistant import prompt_tty

    monkeypatch.setattr(prompt_tty.sys, "platform", "darwin")

    def boom(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(prompt_tty.subprocess, "run", boom)
    prompt_tty.write_clipboard("hello")
    assert prompt_tty.read_clipboard() == "hello"
    prompt_tty.write_clipboard("a\nb")
    assert prompt_tty.read_clipboard() == "a b"


def test_update_progress_overwrites_same_line() -> None:
    """Pull percentages stay on one line: EL+CR, with no newline per tick."""
    ui = ConsoleUI()
    buffer = StringIO()
    ui.console.file = buffer
    ui.console._force_terminal = True
    ui.thermal_bar.redraw = lambda *args, **kwargs: None
    ui.update_progress("gemma4:12b: downloading 10%")
    ui.update_progress("gemma4:12b: downloading 20%")
    text = buffer.getvalue()
    assert "\033[2K" in text
    assert text.count("\n") == 0
    assert "20%" in text
    ui.print_system_message("Modello impostato")
    closed = buffer.getvalue()
    assert closed.count("\n") >= 1
    assert "Modello impostato" in closed


def test_update_progress_non_tty_wraps_line() -> None:
    """Outside a TTY each tick is a complete system message (capturable log)."""
    ui = ConsoleUI()
    buffer = StringIO()
    ui.console.file = buffer
    ui.console._force_terminal = False
    ui.thermal_bar.redraw = lambda *args, **kwargs: None
    ui.update_progress("gemma4:12b: downloading 10%")
    ui.update_progress("gemma4:12b: downloading 20%")
    text = buffer.getvalue()
    assert text.count("10%") == 1
    assert text.count("20%") == 1
    assert text.count("\n") >= 2


def test_reading_width_respects_cap() -> None:
    """On wide terminals the markdown does not exceed 88 columns."""
    assert reading_width(160) == MARKDOWN_READING_WIDTH
    assert reading_width(40) == 40
    assert reading_width(0) == 1


def test_cuu_clear_composer_counts_prefix() -> None:
    """The boxless composer rises by the visual lines of `› ` + text."""
    assert composer_line("ciao").startswith(COMPOSER_PREFIX)
    assert cuu_clear_composer("", 80) == 1
    long = "a" * 90
    assert cuu_clear_composer(long, 40) == visual_content_lines(
        composer_line(long), 40
    )


def test_tool_activity_sentence_in_italian() -> None:
    """Known tools have a human sentence; the others stay readable."""
    assert tool_activity_sentence("cerca_testo") == "Cerco nei documenti"
    assert tool_activity_sentence("scrivi_file") == "Scrivo"
    assert tool_activity_sentence("tool_sconosciuto") == "tool sconosciuto"


def test_timeline_prints_activity_on_one_line() -> None:
    """The timeline does not use ⚡/↳ glyphs; in verbose it adds the technical name."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=80,
        height=24,
        force_terminal=True,
        color_system=None,
    )
    timeline = TurnTimeline(console)
    timeline.record_tool_action("cerca_testo", "balenottera azzurra")
    text = buffer.getvalue()
    assert "Cerco nei documenti" in text
    assert "balenottera azzurra" in text
    assert "⚡" not in text
    assert "cerca_testo" not in text

    timeline.reset_turn()
    timeline.set_verbose(True)
    timeline.record_tool_action("scrivi_file", "note.txt")
    verbose = buffer.getvalue()
    assert "Scrivo" in verbose
    assert "note.txt" in verbose
    assert "(scrivi_file)" in verbose


def test_stream_uses_reading_cap_on_wide_console() -> None:
    """Live gets a Constrain at 88 columns even if the TTY is wider."""
    live = Mock()
    renderer, _ = _create_isolated_renderer()
    renderer._console.width = 160

    with (
        patch("assistant.console.Live", return_value=live) as create_live,
        patch("assistant.console.sync_dimensions"),
        patch("assistant.console.time.monotonic", return_value=0.0),
    ):
        renderer.start_direct_answer()
        renderer.write_token("ciao")

    constrain = create_live.call_args.args[0].renderables[-1]
    assert constrain.width == MARKDOWN_READING_WIDTH


def test_session_header_markup_includes_model_and_help() -> None:
    """The sticky header is one line: model, documents folder, thinking, /help."""
    markup = session_header_markup(
        {
            "verbose": False,
            "full_verification": False,
            "burn_protection": True,
            "thinking": False,
        },
        "default · q4 · gemma4:12b",
        "/Users/x/balena",
    )
    assert "Assistente" in markup
    assert "default · q4 · gemma4:12b" in markup
    assert "cartella /Users/x/balena" in markup
    assert "web" not in markup
    assert "thinking off" in markup
    assert "/help" in markup
    off = session_header_markup(
        {"thinking": True},
        "fast · q4 · gemma4:e4b",
    )
    assert "cartella" not in off
    assert "thinking" in off
    assert "thinking off" not in off


def test_sticky_composer_geometry() -> None:
    """The sticky composer sits above the status and does not eat header and transcript."""
    assert composer_height("", 80) == 2
    assert sticky_composer_height("", 80, 24, header=True) == 2
    assert sticky_composer_start_line(24, 2) == 22
    assert scroll_region_sequence(2, 21) == "\033[2;21r"
    high = sticky_composer_height("a" * 400, 40, 24, header=True)
    assert 2 <= high <= 6


def test_sync_subtracts_explicit_chrome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Header + status reduce the Live height, not just the bar line."""
    console = Console(width=80, height=24, force_terminal=True)

    def _fake_size(_console: Console) -> tuple[int, int]:
        return 100, 40

    monkeypatch.setattr("assistant.tty_layout.tty_size", _fake_size)
    _columns, _lines = sync_dimensions(
        console,
        reserve_bar=True,
        reserved_lines=3,
    )
    assert console.height == 37


def test_chrome_lines_counts_header_and_composer() -> None:
    """With the bar active, the transcript loses header, composer and status."""
    console = Console(width=80, height=24, force_terminal=True)
    bar = ThermalBar(console)
    assert bar.chrome_lines() == 1
    bar._enable = True
    bar._header_lines = 1
    bar._composer_lines = 2
    assert bar.chrome_lines() == 4
    assert bar.header_active is True


def test_reprint_sticky_chrome_does_not_use_cuu(monkeypatch: pytest.MonkeyPatch) -> None:
    """With sticky chrome, the reflow does not clear the history with CUU+ED."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    ui = ConsoleUI()
    ui.console = Console(
        file=StringIO(),
        width=80,
        height=24,
        force_terminal=True,
        color_system=None,
    )
    ui.thermal_bar._enable = True
    ui.thermal_bar._header_lines = 1
    ui._current_prompt_title = "Tu"
    with (
        patch.object(ui.console, "clear") as clear,
        patch("assistant.console.clear_lines_above") as delete,
        patch("assistant.console.print_composer") as composer,
        patch.object(ui, "_draw_sticky_header") as header,
        patch.object(ui, "_draw_sticky_composer") as sticky,
        patch.object(ui, "_sync"),
        patch.object(ui.thermal_bar, "redraw"),
    ):
        ui._reprint_chrome("ciao", anchor=None, cursor=4)
    clear.assert_not_called()
    delete.assert_not_called()
    composer.assert_not_called()
    header.assert_called_once()
    sticky.assert_called_once()
