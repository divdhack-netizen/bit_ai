"""TTY layout: dimension sync, visible truncation and anti-wrap Panel."""

from __future__ import annotations

import os
import re
import shutil
import sys

from rich._wrap import divide_line
from rich.cells import cell_len
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.text import Text

# Anti-wrap margin: Terminal.app wraps on the last column
TTY_WIDTH_MARGIN = 1
# Corners + Panel title spaces
PANEL_TITLE_MARGIN = 4
# User input Panel padding: (vertical, horizontal)
PADDING_PANEL_INPUT = (0, 1)
# Prefix of the boxless composer
COMPOSER_PREFIX = "› "
# Markdown/stream reading cap: do not stretch lines on ultrawide terminals
MARKDOWN_READING_WIDTH = 88
# Sticky composer: title + body, never more than the usable viewport
MAX_STICKY_COMPOSER_LINES = 6
# Sticky header: one line outside the scroll region
STICKY_HEADER_LINES = 1
# Shared border style (theme key in ConsoleUI)
PANEL_BORDER_STYLE = "panel.border"
# SGR sequences used by the thermal bar
_PATTERN_ANSI_SGR = re.compile(r"\033\[[0-9;]*m")


def read_ioctl_tty_size() -> tuple[int, int] | None:
    """
    Read columns/rows through ioctl on the TTY file descriptor.

    Returns:
        (columns, rows) if available, otherwise None (pipe/redirect).
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            fileno = stream.fileno()
        except (AttributeError, OSError, ValueError):
            continue
        try:
            size = os.get_terminal_size(fileno)
        except OSError:
            continue
        if size.columns > 0 and size.lines > 0:
            return size.columns, size.lines
    return None


def _tty_size_without_env() -> tuple[int, int] | None:
    """shutil fallback ignoring COLUMNS/LINES (often stale after a shrink)."""
    saved: dict[str, str] = {}
    for key in ("COLUMNS", "LINES"):
        if key in os.environ:
            saved[key] = os.environ.pop(key)
    try:
        try:
            size = shutil.get_terminal_size()
        except OSError:
            return None
        if size.columns > 0 and size.lines > 0:
            return size.columns, size.lines
        return None
    finally:
        os.environ.update(saved)


def tty_size(console: Console) -> tuple[int, int]:
    """
    Read TTY columns/rows, preferring ioctl (follows enlarge and shrink).

    Does not use min(ioctl, COLUMNS): a small COLUMNS after a shrink
    would block the enlarge.
    """
    ioctl = read_ioctl_tty_size()
    if ioctl is not None:
        return ioctl
    without_env = _tty_size_without_env()
    if without_env is not None:
        return without_env
    return max(1, console.width), max(1, console.height)


def sync_dimensions(
    console: Console,
    *,
    reserve_bar: bool = False,
    reserved_lines: int = 0,
) -> tuple[int, int]:
    """
    Align the Console size with the TTY (width = columns - 1 for anti-wrap).

    Args:
        reserve_bar: If True and `reserved_lines` is 0, subtract 1 line (status).
        reserved_lines: Header + composer + status already computed from the chrome.

    Returns:
        TTY dimensions (columns, rows) before the margin.
    """
    columns, lines = tty_size(console)
    width = max(1, columns - TTY_WIDTH_MARGIN)
    extra = reserved_lines
    if extra <= 0 and reserve_bar:
        extra = 1
    if extra > 0 and lines > extra:
        height = max(1, lines - extra)
    else:
        height = max(1, lines)
    console.size = (width, height)
    os.environ["COLUMNS"] = str(columns)
    os.environ["LINES"] = str(lines)
    return columns, lines


def adaptive_panel_title(markup: str, max_cells: int) -> str:
    """Truncate a Rich Panel title to the usable width (ellipsis if needed)."""
    if max_cells <= 0:
        return ""
    text = Text.from_markup(markup)
    if cell_len(text.plain) <= max_cells:
        return markup
    truncated = text.copy()
    truncated.truncate(max_cells, overflow="ellipsis")
    return truncated.markup


def strip_ansi_sgr(text: str) -> str:
    """Remove SGR sequences from an ANSI string."""
    return _PATTERN_ANSI_SGR.sub("", text)


def visible_width(text: str) -> int:
    """Width in cells, ignoring ANSI SGR."""
    return cell_len(strip_ansi_sgr(text))


def truncate_visible(text: str, max_columns: int) -> str:
    """Truncate to the visible width without splitting ANSI sequences."""
    if max_columns <= 0:
        return ""
    if visible_width(text) <= max_columns:
        return text
    result: list[str] = []
    used_cells = 0
    index = 0
    length = len(text)
    while index < length:
        if text[index] == "\033":
            match = _PATTERN_ANSI_SGR.match(text, index)
            if match:
                result.append(match.group(0))
                index = match.end()
                continue
        char = text[index]
        cost = cell_len(char)
        if used_cells + cost > max_columns:
            break
        result.append(char)
        used_cells += cost
        index += 1
    return "".join(result) + "\033[0m"


def panel_cursor_column(horizontal_padding: int) -> int:
    """1-based column of the first content character in a Rich Panel."""
    return 2 + horizontal_padding


def panel_inner_width(
    console_width: int,
    horizontal_padding: int = PADDING_PANEL_INPUT[1],
) -> int:
    """Usable cells in the Panel (no borders or padding); at least 1."""
    # 2 cells: left/right vertical borders.
    return max(1, console_width - 2 - (2 * horizontal_padding))


def panel_content_wrap_segments(
    text: str,
    usable_width: int,
) -> list[str]:
    """
    Wrap `text` like the body of a Rich Panel (word wrap + fold).

    Whole-cell geometry (`len % width`) is not enough: Rich wraps on
    words, so the last line is shorter than the division remainder.
    """
    if not text:
        return [""]
    useful = max(1, usable_width)
    cuts = divide_line(text, useful, fold=True)
    if not cuts:
        return [text]
    segments: list[str] = []
    start = 0
    for cut in cuts:
        segments.append(text[start:cut])
        start = cut
    segments.append(text[start:])
    if segments and segments[-1] == "" and len(segments) > 1:
        segments.pop()
    return segments or [""]


def content_cursor_geometry(
    text: str,
    usable_width: int,
) -> tuple[int, int]:
    """
    Column offset (0-based in the content) and visual lines at the end of `text`.

    Returns:
        `(extra_column, visual_lines)`. On a full line `extra_column == usable_width`.
    """
    if not text:
        return 0, 1
    useful = max(1, usable_width)
    segments = panel_content_wrap_segments(text, useful)
    last = segments[-1]
    extra_column = cell_len(last)
    if extra_column == 0:
        return useful, max(1, len(segments) - 1)
    return extra_column, len(segments)


def visual_content_lines(text: str, usable_width: int) -> int:
    """Visual lines of `text` within the Panel's usable width (at least 1)."""
    return content_cursor_geometry(text, usable_width)[1]


def panel_rise_lines(content_lines: int, vertical_padding: int) -> int:
    """CUU from the cursor below the Panel up to the first content line."""
    return 1 + vertical_padding + max(1, content_lines) + vertical_padding


def lines_from_content_to_top_border(
    vertical_padding: int,
    *,
    content_line: int = 1,
) -> int:
    """
    CUU from the cursor INSIDE the content up to the Panel's top border.

    While editing in cbreak use this geometry (not `panel_rise_lines`).
    """
    return 1 + vertical_padding + max(0, content_line - 1)


def cuu_clear_panel_input(
    text: str,
    console_width: int,
    padding: tuple[int, int] = PADDING_PANEL_INPUT,
) -> int:
    """
    CUU from the end of content (wrap included) to the input Panel's top border.

    Used on Enter / reprint: a single source of truth for the climb.
    """
    padding_v, padding_h = padding
    useful = panel_inner_width(console_width, padding_h)
    return lines_from_content_to_top_border(
        padding_v,
        content_line=visual_content_lines(text, useful),
    )


def print_panel(
    console: Console,
    body: object,
    title_markup: str,
    *,
    border_style: str = PANEL_BORDER_STYLE,
    padding: tuple[int, int] = (0, 1),
    sync: bool = True,
) -> None:
    """Print a Panel with a truncated title and an explicit width."""
    if sync:
        sync_dimensions(console, reserve_bar=True)
    width = console.width
    title = adaptive_panel_title(
        title_markup,
        max(1, width - PANEL_TITLE_MARGIN),
    )
    console.print(Panel(
        body,
        title=title,
        title_align="left",
        border_style=border_style,
        padding=padding,
        width=width,
        expand=True,
    ))


def position_cursor_in_input_panel(
    *,
    content_lines: int = 1,
    padding: tuple[int, int] = PADDING_PANEL_INPUT,
    extra_column: int = 0,
    is_terminal: bool = True,
) -> None:
    """Move the cursor to the content line of a just-printed Panel."""
    if not is_terminal or not sys.stdout.isatty():
        return
    padding_v, padding_h = padding
    lines_up = panel_rise_lines(content_lines, padding_v)
    column = panel_cursor_column(padding_h) + max(0, extra_column)
    sys.stdout.write(f"\033[{lines_up}A\033[{column}G")
    sys.stdout.flush()


def position_cursor_at_index(
    text: str,
    index: int,
    usable_width: int,
    *,
    padding: tuple[int, int] = PADDING_PANEL_INPUT,
    is_terminal: bool = True,
) -> None:
    """
    After `print_panel`, put the cursor on the Unicode index in `text`.

    From the bottom of the box: `CUU` grows if the index is not on the last wrapped line.
    """
    if not is_terminal or not sys.stdout.isatty():
        return
    index = max(0, min(index, len(text)))
    col_pref, pref_lines = content_cursor_geometry(text[:index], usable_width)
    _col_full, full_lines = content_cursor_geometry(text, usable_width)
    padding_v, padding_h = padding
    lines_up = 2 + padding_v + (full_lines - pref_lines)
    column = panel_cursor_column(padding_h) + col_pref
    sys.stdout.write(f"\033[{lines_up}A\033[{column}G")
    sys.stdout.flush()


def reading_width(console_width: int) -> int:
    """Usable width for markdown and stream: never beyond the reading cap."""
    return max(1, min(int(console_width), MARKDOWN_READING_WIDTH))


def composer_line(text: str) -> str:
    """Editing line: prefix plus user content."""
    return f"{COMPOSER_PREFIX}{text}"


def cuu_clear_composer(text: str, console_width: int) -> int:
    """
    CUU from the composer's last line to the title, title included.

    Same approximation as `cuu_clear_panel_input`: it uses the content
    height, not the cursor's exact line.
    """
    useful = max(1, console_width)
    return visual_content_lines(composer_line(text), useful)


def print_composer(
    console: Console,
    title_markup: str,
    body: Text,
    *,
    sync: bool = False,
) -> None:
    """Print title + `› ` line without a box, for editing the request."""
    if sync:
        sync_dimensions(console, reserve_bar=True)
    title = adaptive_panel_title(title_markup, max(1, console.width))
    console.print(Text.from_markup(title), highlight=False)
    line = Text()
    line.append(COMPOSER_PREFIX, style="user.label")
    if body.plain != " ":
        line.append_text(body)
    console.print(line, highlight=False, soft_wrap=True)


def print_user_message(
    console: Console,
    title_markup: str,
    text: str,
    *,
    sync: bool = True,
) -> None:
    """Confirmed user turn: title and body, without the editing prefix."""
    if sync:
        sync_dimensions(console, reserve_bar=True)
    title = adaptive_panel_title(title_markup, max(1, console.width))
    console.print(Text.from_markup(title), highlight=False)
    console.print(Text(text) if text else Text(" ", style="dim"), highlight=False)


def position_cursor_in_composer(*, is_terminal: bool = True) -> None:
    """After an empty `print_composer`, put the cursor after `› `."""
    if not is_terminal or not sys.stdout.isatty():
        return
    column = 1 + cell_len(COMPOSER_PREFIX)
    sys.stdout.write(f"\033[1A\033[{column}G")
    sys.stdout.flush()


def position_composer_cursor(
    text: str,
    index: int,
    console_width: int,
    *,
    is_terminal: bool = True,
) -> None:
    """After `print_composer`, put the cursor on the Unicode index in `text`."""
    if not is_terminal or not sys.stdout.isatty():
        return
    useful = max(1, console_width)
    index = max(0, min(index, len(text)))
    composite = composer_line(text)
    vis_index = len(COMPOSER_PREFIX) + index
    col_pref, pref_lines = content_cursor_geometry(composite[:vis_index], useful)
    _col_full, full_lines = content_cursor_geometry(composite, useful)
    lines_up = 1 + (full_lines - pref_lines)
    column = 1 + col_pref
    sys.stdout.write(f"\033[{lines_up}A\033[{column}G")
    sys.stdout.flush()


def clear_lines_above(line_count: int, *, is_terminal: bool = True) -> None:
    """
    Go back to column 1, move up N lines and erase to the end of screen (`ED 0`).

    The CR is mandatory: `CUU` keeps the column; without it, a reprint in the
    middle of a line duplicates the Panel title.
    """
    if line_count < 1 or not is_terminal or not sys.stdout.isatty():
        return
    sys.stdout.write(f"\r\033[{line_count}A\033[J")
    sys.stdout.flush()


def session_header_markup(
    state: dict[str, bool],
    model_label: str,
    folder: str = "",
) -> str:
    """One sticky line: model, documents folder, thinking and the `/help` hint."""
    thinking = "thinking" if state.get("thinking", False) else "thinking off"
    dim: list[str] = []
    model = model_label.strip()
    if model:
        dim.append(model)
    if folder:
        dim.append(escape(f"cartella {folder}"))
    dim.extend([thinking, "/help"])
    return f"[agent.label]Assistente[/] [dim]· {' · '.join(dim)}[/]"


def composer_height(text: str, console_width: int) -> int:
    """Title plus the visual lines of the `› ` + text body."""
    useful = max(1, console_width)
    return 1 + visual_content_lines(composer_line(text), useful)


def sticky_composer_height(
    text: str,
    console_width: int,
    tty_lines: int,
    *,
    header: bool,
) -> int:
    """Height reserved for the composer, limited to the viewport below header and status."""
    full = composer_height(text, console_width)
    chrome = (STICKY_HEADER_LINES if header else 0) + 1
    maximum = min(MAX_STICKY_COMPOSER_LINES, max(2, tty_lines - chrome - 1))
    return max(2, min(full, maximum))


def sticky_scroll_start_line(*, header: bool) -> int:
    """1-based first line of the scroll region below the header."""
    return 1 + (STICKY_HEADER_LINES if header else 0)


def sticky_composer_start_line(tty_lines: int, composer_lines: int) -> int:
    """1-based line of the first composer line (the status takes the last one)."""
    return max(1, int(tty_lines) - max(1, composer_lines))


def sticky_scroll_end_line(
    tty_lines: int,
    *,
    header: bool,
    composer_lines: int,
) -> int:
    """1-based last line of the scroll region (above composer and status)."""
    low = 1 + max(0, composer_lines)
    top = 1 + (STICKY_HEADER_LINES if header else 0)
    return max(top, int(tty_lines) - low)


def scroll_region_sequence(line_before: int, last_line: int) -> str:
    """DECSTBM: freeze the header and the bottom band outside the scroll."""
    before = max(1, line_before)
    last = max(before, last_line)
    return f"\033[{before};{last}r"


def _render_text_ansi(console: Console, text: Text, max_columns: int) -> str:
    """Rich render without writing to the TTY; ignores control sequences."""
    useful = max(1, max_columns)
    options = console.options.update(width=useful, no_wrap=True)
    pieces: list[str] = []
    for segment in console.render(text, options):
        if segment.control:
            continue
        style = segment.style
        if style is None:
            pieces.append(segment.text)
            continue
        pieces.append(style.render(segment.text, color_system=console._color_system))
    return truncate_visible("".join(pieces).replace("\n", ""), useful)


def render_line_markup(console: Console, markup: str, max_columns: int) -> str:
    """Rich markup → ANSI for one line, truncated to the visible cells."""
    return _render_text_ansi(console, Text.from_markup(markup), max_columns)


def clear_absolute_lines(
    start_line: int,
    number: int,
    *,
    is_terminal: bool = True,
) -> None:
    """Clear N lines at absolute coordinates, without touching the scrollback."""
    if number < 1 or not is_terminal or not sys.stdout.isatty():
        return
    parts = [f"\033[{start_line + index};1H\033[2K" for index in range(number)]
    sys.stdout.write("".join(parts))
    sys.stdout.flush()


def draw_absolute_line(
    line: int,
    ansi_text: str,
    *,
    is_terminal: bool = True,
    restore_cursor: bool = True,
) -> None:
    """Write a line at an absolute CUP; by default it does not steal the cursor."""
    if not is_terminal or not sys.stdout.isatty() or line < 1:
        return
    # ?7l turns autowrap off: a wider header does not spill onto the line below.
    body = f"\033[{line};1H\033[?7l\033[2K{ansi_text}\033[?7h"
    if restore_cursor:
        sys.stdout.write(f"\0337{body}\0338")
    else:
        sys.stdout.write(body)
    sys.stdout.flush()


def position_cursor_absolute(
    line: int,
    column: int,
    *,
    is_terminal: bool = True,
) -> None:
    """1-based CUP; no-op outside a TTY."""
    if not is_terminal or not sys.stdout.isatty():
        return
    sys.stdout.write(f"\033[{max(1, line)};{max(1, column)}H")
    sys.stdout.flush()


def position_transcript_start(
    *,
    header: bool,
    is_terminal: bool = True,
) -> None:
    """Cursor on the first line of the scroll region, column 1."""
    position_cursor_absolute(
        sticky_scroll_start_line(header=header),
        1,
        is_terminal=is_terminal,
    )


def position_transcript_end(
    tty_lines: int,
    *,
    header: bool,
    composer_lines: int = 0,
    is_terminal: bool = True,
) -> None:
    """Cursor on the last line of the scroll region, column 1."""
    line = sticky_scroll_end_line(
        tty_lines,
        header=header,
        composer_lines=composer_lines,
    )
    position_cursor_absolute(line, 1, is_terminal=is_terminal)


def print_sticky_header(
    console: Console,
    markup: str,
    *,
    is_terminal: bool = True,
) -> None:
    """Draw the header on line 1 without wiping the history."""
    if not is_terminal or not sys.stdout.isatty():
        return
    columns, _lines = tty_size(console)
    text = render_line_markup(console, markup, max(1, columns - 1))
    draw_absolute_line(1, text, is_terminal=True, restore_cursor=True)


def print_sticky_composer(
    console: Console,
    title_markup: str,
    body: Text,
    text: str,
    cursor: int,
    tty_lines: int,
    composer_lines: int,
    *,
    is_terminal: bool = True,
) -> None:
    """Draw title + `› ` above the bar, at fixed coordinates."""
    if not is_terminal or not sys.stdout.isatty():
        print_composer(console, title_markup, body, sync=False)
        return
    useful = max(1, console.width)
    start = sticky_composer_start_line(tty_lines, composer_lines)
    title = render_line_markup(console, title_markup, useful)
    body_line = Text()
    body_line.append(COMPOSER_PREFIX, style="user.label")
    if body.plain != " ":
        body_line.append_text(body)
    wrapped = body_line.wrap(console, useful) or [Text("")]
    body_lines = max(1, composer_lines - 1)
    visible = wrapped[-body_lines:]
    while len(visible) < body_lines:
        visible.append(Text(""))
    draw_absolute_line(start, title, is_terminal=True, restore_cursor=False)
    for offset, piece in enumerate(visible, start=1):
        ansi = _render_text_ansi(console, piece, useful)
        draw_absolute_line(
            start + offset,
            ansi,
            is_terminal=True,
            restore_cursor=False,
        )
    composite = composer_line(text)
    index = max(0, min(cursor, len(text)))
    vis_index = len(COMPOSER_PREFIX) + index
    col_pref, rel_line = content_cursor_geometry(composite[:vis_index], useful)
    body_start_line = start + 1
    cursor_line = body_start_line + rel_line - 1
    max_line = start + composer_lines - 1
    cursor_line = min(max(body_start_line, cursor_line), max_line)
    position_cursor_absolute(cursor_line, 1 + col_pref, is_terminal=True)
