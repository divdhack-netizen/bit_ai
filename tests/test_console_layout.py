"""Test unitari per layout TTY responsive: troncamento ANSI, metriche, sync."""

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

from assistente.console import (
    BarraTermica,
    ConsoleUI,
    RendererPanelStream,
    TimelineTurno,
    frase_attivita_tool,
)
from assistente.splash import (
    INTERVALLO_FRAME_SEC,
    MESSAGGIO_SPLASH,
    PAUSA_DOPO_ANIMAZIONE_SEC,
    PAUSA_SALUTO_SEC,
)
from assistente.metriche import StatisticheTurnoLLM, formatta_segmento_statistiche
from assistente.thermal import StatoTermico, riga_barra_stato
from assistente.tty_layout import (
    LARGHEZZA_LETTURA_MARKDOWN,
    MARGINE_LARGHEZZA_TTY,
    PREFISSO_COMPOSER,
    altezza_composer,
    altezza_composer_sticky,
    cancella_righe_sopra,
    colonna_cursore_panel,
    cuu_clear_composer,
    cuu_clear_panel_input,
    geometria_cursore_contenuto,
    larghezza_interna_panel,
    larghezza_lettura,
    larghezza_visibile,
    markup_header_sessione,
    riga_inizio_composer_sticky,
    righe_da_contenuto_a_bordo_superiore,
    righe_salita_panel,
    righe_visive_contenuto,
    sequenza_regione_scroll,
    sincronizza_dimensioni,
    riga_composer,
    titolo_panel_adattivo,
    tronca_visibile,
    tty_size,
)

pytestmark = pytest.mark.unit


def _metriche_esempio() -> StatisticheTurnoLLM:
    """Snapshot realistico per il layout progressivo della barra."""
    return StatisticheTurnoLLM(
        token_input=12_345,
        token_output=678,
        token_al_secondo=42.5,
        contesto_usato=3_072,
        contesto_massimo=32_768,
        chiamate_llm=2,
        ttft_ms=120.0,
        durata_llm_ms=1500.0,
        durata_tool_ms=80.0,
    )


def _crea_renderer_isolato(
    on_render: Mock | None = None,
) -> tuple[RendererPanelStream, Mock]:
    """
    Crea un renderer con output in memoria e timeline osservabile.

    Args:
        on_render: Callback opzionale usata per contare i refresh effettivi.

    Returns:
        Renderer sotto test e mock della timeline ReAct.
    """
    console = Console(
        file=StringIO(),
        width=80,
        height=24,
        force_terminal=False,
    )
    timeline = Mock(spec=TimelineTurno)
    return RendererPanelStream(console, timeline, on_render=on_render), timeline


def _corpo_live(renderable: object) -> object:
    """Estrae Text/Markdown dal Group(titolo, Constrain(corpo)) dello stream."""
    renderables = getattr(renderable, "renderables", None)
    if not renderables:
        return getattr(renderable, "renderable", renderable)
    inner = renderables[-1]
    return getattr(inner, "renderable", inner)


def test_inizia_risposta_diretta_apre_panel_senza_marcatori_react() -> None:
    """La pipeline direct deve entrare subito in Final Answer mode."""
    live = Mock()
    renderer, _ = _crea_renderer_isolato()

    with (
        patch("assistente.console.Live", return_value=live),
        patch("assistente.console.sincronizza_dimensioni"),
        patch("assistente.console.time.monotonic", return_value=0.0),
    ):
        renderer.inizia_risposta_diretta()
        renderer.scrivi_token("Risposta senza Thought")

    assert live.start.called
    assert "".join(renderer._buffer_finale) == "Risposta senza Thought"


def test_stream_mostra_primo_chunk_plain_text_subito() -> None:
    """Il primo contenuto apre Live già popolato, senza attendere un refresh periodico."""
    live = Mock()
    orologio = Mock(return_value=10.0)
    renderer, _ = _crea_renderer_isolato()

    with (
        patch("assistente.console.Live", return_value=live) as crea_live,
        patch("assistente.console.sincronizza_dimensioni"),
        patch("assistente.console.time.monotonic", orologio),
    ):
        renderer.scrivi_token("Final Ans")
        crea_live.assert_not_called()

        renderer.scrivi_token("wer: **ciao**")

    panel_iniziale = crea_live.call_args.args[0]
    corpo = _corpo_live(panel_iniziale)
    assert isinstance(corpo, Text)
    assert corpo.plain == "**ciao**"
    assert crea_live.call_args.kwargs["auto_refresh"] is False
    assert isinstance(renderer._buffer_finale, list)
    assert renderer._buffer_finale == ["**ciao**"]
    live.start.assert_called_once_with(refresh=True)
    live.update.assert_not_called()


def test_stream_limita_refresh_e_barra_senza_sleep() -> None:
    """I chunk sul thread di decode non chiamano Live.refresh: lo fa il worker."""
    live = Mock()
    on_render = Mock()
    renderer, _ = _crea_renderer_isolato(on_render)

    with (
        patch("assistente.console.Live", return_value=live),
        patch("assistente.console.sincronizza_dimensioni"),
    ):
        renderer.scrivi_token("Final Answer: a")
        for _ in range(40):
            renderer.scrivi_token("x")
        time.sleep(0.12)
        assert "".join(renderer._buffer_finale) == f"a{'x' * 40}"
        renderer.finalizza("ax")

    assert live.refresh.call_count >= 1
    assert live.refresh.call_count <= 8
    live.update.assert_called()


def test_stream_crea_markdown_una_sola_volta_in_finalizzazione() -> None:
    """I chunk restano Text e il contenuto completo diventa Markdown solo alla fine."""
    live = Mock()
    orologio = Mock(return_value=0.0)
    renderer, _ = _crea_renderer_isolato()

    with (
        patch("assistente.console.Live", return_value=live) as crea_live,
        patch("assistente.console.sincronizza_dimensioni"),
        patch("assistente.console.time.monotonic", orologio),
        patch("assistente.console.Markdown", wraps=Markdown) as crea_markdown,
    ):
        renderer.scrivi_token("Final Answer: **ciao")
        orologio.return_value = 0.01
        renderer.scrivi_token("** mondo")

        assert crea_markdown.call_count == 0
        assert isinstance(_corpo_live(crea_live.call_args.args[0]), Text)

        renderer.finalizza("Final Answer: fallback")

    crea_markdown.assert_called_once()
    panel_finale = live.update.call_args.args[0]
    corpo = _corpo_live(panel_finale)
    assert isinstance(corpo, Markdown)
    assert corpo.markup == "**ciao** mondo"
    assert live.update.call_args.kwargs["refresh"] is True
    live.stop.assert_called_once()
    assert renderer._buffer_finale == []


def _schermo_visibile(raw: str) -> str:
    """Applica CR, cancella-riga e SGR come farebbe il TTY sulla cattura di test."""
    righe = [""]
    indice = 0
    while indice < len(raw):
        if raw.startswith("\033[K", indice):
            righe[-1] = ""
            indice += 3
            continue
        if raw.startswith("\r", indice):
            righe[-1] = ""
            indice += 1
            continue
        if raw.startswith("\n", indice):
            righe.append("")
            indice += 1
            continue
        if raw.startswith("\033[", indice):
            fine = raw.find("m", indice)
            if fine != -1:
                indice = fine + 1
                continue
        righe[-1] += raw[indice]
        indice += 1
    return "\n".join(righe)


def test_stream_overflow_passa_a_scroll_libero() -> None:
    """Se Live satura il TTY, i token successivi vanno nello scrollback."""
    live = Mock()
    live._lock = threading.RLock()
    live._nested = False
    live._started = True
    live.console.is_terminal = False
    live._live_render.last_render_height = 24
    renderer, _ = _crea_renderer_isolato()

    with (
        patch("assistente.console.Live", return_value=live),
        patch("assistente.console.sincronizza_dimensioni"),
        patch("assistente.console.time.monotonic", return_value=0.0),
    ):
        renderer.scrivi_token("Final Answer: inizio")
        renderer._esegui_refresh_live()
        renderer.scrivi_token(" coda")
        renderer.finalizza("Final Answer: inizio coda")

    assert live.stop.called
    assert renderer._live is None
    assert renderer._scroll_libero is False
    cattura = _schermo_visibile(renderer._console.file.getvalue())
    assert "inizio" in cattura
    assert "coda" in cattura
    live.update.assert_not_called()


def test_finalizza_corta_resta_nel_live() -> None:
    """Una risposta che entra nel viewport non sospende la scroll-region."""
    live = Mock()
    ordine: list[str] = []
    live.stop.side_effect = lambda: ordine.append("stop")
    console = Console(file=StringIO(), width=80, height=24, force_terminal=False)
    timeline = Mock(spec=TimelineTurno)
    renderer = RendererPanelStream(
        console,
        timeline,
        sospendi_scroll=lambda: ordine.append("sospendi"),
        ripristina_scroll=lambda: ordine.append("ripristina"),
    )
    with (
        patch("assistente.console.Live", return_value=live),
        patch("assistente.console.sincronizza_dimensioni"),
        patch("assistente.console.time.monotonic", return_value=0.0),
    ):
        renderer.scrivi_token("Final Answer: testo")
        renderer.finalizza("Final Answer: testo")
    assert ordine == ["stop"]
    live.update.assert_called()


def test_finalizza_alta_stampa_fuori_dalla_scroll_region() -> None:
    """Se il markdown non entra, non si usa Live.stop(): l'inizio resterebbe tagliato."""
    live = Mock()
    live._lock = threading.RLock()
    live.console.is_terminal = False
    ordine: list[str] = []
    console = Console(file=StringIO(), width=20, height=3, force_terminal=False)
    timeline = Mock(spec=TimelineTurno)
    renderer = RendererPanelStream(
        console,
        timeline,
        sospendi_scroll=lambda: ordine.append("sospendi"),
        ripristina_scroll=lambda: ordine.append("ripristina"),
    )
    with (
        patch("assistente.console.Live", return_value=live),
        patch("assistente.console.sincronizza_dimensioni"),
        patch("assistente.console.time.monotonic", return_value=0.0),
    ):
        renderer.scrivi_token("Final Answer: uno due tre quattro cinque sei sette otto")
        renderer.finalizza(
            "Final Answer: uno due tre quattro cinque sei sette otto nove dieci"
        )
    assert ordine == ["sospendi", "ripristina"]
    assert "uno" in console.file.getvalue()


def test_pensiero_va_a_capo_sulle_parole() -> None:
    """Il CoT non spezza le parole sul bordo del TTY."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=12,
        height=24,
        force_terminal=True,
        color_system=None,
    )
    timeline = TimelineTurno(console)
    timeline.scrivi_pensiero("alfa bravo charlie")
    timeline.chiudi_pensiero()
    testo = buffer.getvalue()
    assert "alfabravo" not in testo.replace("\n", "")
    assert "charlie" in testo


def test_scroll_libero_non_taglia_le_parole_sul_bordo() -> None:
    """Le risposte più alte del TTY restano intere: a capo sulle parole, dentro la larghezza."""
    live = Mock()
    live._lock = threading.RLock()
    live._nested = False
    live._started = True
    live.console.is_terminal = False
    live._live_render.last_render_height = 24
    renderer, _ = _crea_renderer_isolato()
    parole = [f"w{indice:02d}" for indice in range(40)]
    paragrafo = " ".join(parole)

    with (
        patch("assistente.console.Live", return_value=live),
        patch("assistente.console.sincronizza_dimensioni"),
        patch("assistente.console.time.monotonic", return_value=0.0),
    ):
        renderer.scrivi_token("Final Answer: " + paragrafo)
        renderer._esegui_refresh_live()
        renderer.finalizza("Final Answer: " + paragrafo)

    visibile = _schermo_visibile(renderer._console.file.getvalue())
    righe = [riga for riga in visibile.splitlines() if riga.strip()]
    assert righe
    assert all(cell_len(riga) <= renderer._console.width for riga in righe)
    for parola in parole:
        assert parola in visibile
    for riga in righe:
        for token in riga.split():
            if token.startswith("w") and token[1:].isdigit():
                assert token in parole


def test_stream_preserva_parser_react_con_marcatori_spezzati() -> None:
    """Thought, Action, Action Input e Final Answer restano validi tra chunk."""
    live = Mock()
    renderer, timeline = _crea_renderer_isolato()

    with (
        patch("assistente.console.Live", return_value=live) as crea_live,
        patch("assistente.console.sincronizza_dimensioni"),
        patch("assistente.console.time.monotonic", return_value=0.0),
    ):
        # I confini scelti spezzano intenzionalmente tutti i marcatori principali.
        for frammento_react in (
            "Thou",
            "ght: analisi\nAct",
            "ion: cerca\nAction In",
            "put: query\nObservation: ok\nFinal Ans",
            "wer: fatto",
        ):
            renderer.scrivi_token(frammento_react)

    timeline.emetti_pensiero.assert_has_calls([call("Thought: analisi")])
    timeline.emetti_azione_da_stream.assert_called_once_with("cerca")
    timeline.emetti_input_azione_da_stream.assert_called_once_with("query")
    timeline.flush_input_azione_pendente.assert_called_once()
    assert crea_live.call_args.args[0].renderables[-1].renderable.plain == "fatto"


def test_tronca_visibile_rispetta_celle_senza_ansi() -> None:
    assert tronca_visibile("abcdef", 3) == "abc\033[0m"
    assert larghezza_visibile(tronca_visibile("abcdef", 3)) == 3


def test_tronca_visibile_non_spezza_sequenze_ansi() -> None:
    colorato = "\033[32mNominal\033[0m-extra"
    risultato = tronca_visibile(colorato, 7)
    assert "\033[3" not in risultato.replace("\033[32m", "").replace("\033[0m", "")
    assert risultato.endswith("\033[0m")
    assert larghezza_visibile(risultato) <= 7


def test_tronca_visibile_budget_zero_o_negativo() -> None:
    assert tronca_visibile("abc", 0) == ""
    assert tronca_visibile("abc", -1) == ""


def test_metriche_complete_senza_tetto() -> None:
    formatted = formatta_segmento_statistiche(_metriche_esempio())
    assert "42.5 tok/s" in formatted
    assert "token: 13 023" in formatted
    assert "ctx: 9%" in formatted
    assert "token in:" not in formatted
    assert "token out:" not in formatted
    assert "TTFT:" not in formatted
    verbose = formatta_segmento_statistiche(_metriche_esempio(), verbose=True)
    assert "LLM: 2/1500 ms" in verbose
    assert "TTFT: 120 ms" in verbose
    assert "tool: 80 ms" in verbose


def test_metriche_progressive_su_larghezze() -> None:
    stats = _metriche_esempio()
    stretto = formatta_segmento_statistiche(stats, max_larghezza=40)
    medio = formatta_segmento_statistiche(stats, max_larghezza=80)
    largo = formatta_segmento_statistiche(stats, max_larghezza=120)
    assert "tok/s" in stretto
    assert larghezza_visibile(stretto) <= 40
    assert larghezza_visibile(medio) <= 80
    assert len(medio) >= len(stretto)
    assert "ctx:" in largo
    assert "token in:" not in largo
    assert larghezza_visibile(largo) <= 120


def test_riga_barra_stato_rispetta_max_colonne() -> None:
    riga = riga_barra_stato(
        StatoTermico.NOMINALE, 55.0, _metriche_esempio(),
        ansi=False, max_colonne=50,
    )
    assert larghezza_visibile(riga) <= 50
    assert riga.startswith("tmp: Nominal")


def test_riga_barra_stato_mostra_low_power_mode() -> None:
    from assistente.host import PowerState

    riga = riga_barra_stato(
        StatoTermico.NOMINALE, 55.0, None,
        ansi=False, power=PowerState(True, True),
    )
    assert riga.startswith("pwr: LPM")


def test_riga_barra_solo_prefisso_se_troppo_stretto() -> None:
    riga = riga_barra_stato(
        StatoTermico.NOMINALE, 55.0, _metriche_esempio(),
        ansi=False, max_colonne=20,
    )
    assert "·" not in riga
    assert larghezza_visibile(riga) <= 20


def test_sincronizza_dimensioni_usa_margine_anti_wrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(width=80, height=24, force_terminal=True)

    def _size_finto(_console: Console) -> tuple[int, int]:
        return 100, 40

    monkeypatch.setattr("assistente.tty_layout.tty_size", _size_finto)
    colonne, righe = sincronizza_dimensioni(console, riserva_barra=True)
    assert colonne == 100
    assert righe == 40
    assert console.width == 100 - MARGINE_LARGHEZZA_TTY
    assert console.height == 39


def test_tty_size_preferisce_ioctl_a_columns_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(width=80, height=24, force_terminal=True)

    def _ioctl_finto() -> tuple[int, int] | None:
        return 79, 34

    monkeypatch.setenv("COLUMNS", "120")
    monkeypatch.setattr("assistente.tty_layout.leggi_ioctl_tty_size", _ioctl_finto)
    colonne, righe = tty_size(console)
    assert colonne == 79
    assert righe == 34


def test_tty_size_enlarge_ignora_columns_stretto(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(width=40, height=24, force_terminal=True)

    def _ioctl_largo() -> tuple[int, int] | None:
        return 93, 34

    monkeypatch.setenv("COLUMNS", "40")
    monkeypatch.setattr("assistente.tty_layout.leggi_ioctl_tty_size", _ioctl_largo)
    colonne, righe = tty_size(console)
    assert colonne == 93
    assert righe == 34


def test_sincronizza_imposta_size_atomico_e_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    console = Console(width=80, height=24, force_terminal=True)

    def _size_finto(_console: Console) -> tuple[int, int]:
        return 79, 34

    monkeypatch.setattr("assistente.tty_layout.tty_size", _size_finto)
    colonne, righe = sincronizza_dimensioni(console, riserva_barra=True)
    assert colonne == 79
    assert righe == 34
    assert console.width == 79 - MARGINE_LARGHEZZA_TTY
    assert console.height == 33
    assert console._width is not None and console._height is not None
    assert os.environ.get("COLUMNS") == "79"
    assert os.environ.get("LINES") == "34"


def test_titolo_panel_adattivo_tronca_lungo() -> None:
    markup = "[user.label]Tu[/] [dim]· default · q4 · gemma4:12b[/]"
    budget = 20
    risultato = titolo_panel_adattivo(markup, budget)
    assert cell_len(Text.from_markup(risultato).plain) <= budget


def test_titolo_panel_adattivo_invariato_se_corto() -> None:
    markup = "[agent.label]Assistente[/]"
    assert titolo_panel_adattivo(markup, 40) == markup


def test_applica_resize_invalida_ultima_riga_e_riserva() -> None:
    console = Console(width=80, height=24, force_terminal=True)
    barra = BarraTermica(console)
    barra._attiva = True
    barra._ultima_riga = "stale"
    barra._righe_riservate = 24
    chiamate: list[str] = []

    def _riserva() -> None:
        chiamate.append("riserva")
        barra._righe_riservate = 30

    def _ridisegna(**_kwargs) -> None:
        chiamate.append("ridisegna")

    with patch.object(barra, "_sincronizza_console"), \
            patch.object(barra, "_riserva_ultima_riga", side_effect=_riserva), \
            patch.object(barra, "ridisegna", side_effect=_ridisegna):
        barra._applica_resize()

    assert barra._ultima_riga == ""
    assert "riserva" in chiamate
    assert "ridisegna" in chiamate


def test_barra_dopo_clear_invalida_riga() -> None:
    console = Console(width=80, height=24, force_terminal=True)
    barra = BarraTermica(console)
    barra._attiva = True
    barra._ultima_riga = "stale"
    with patch.object(barra, "_riserva_ultima_riga") as riserva:
        barra.dopo_clear()
    riserva.assert_called_once()
    assert barra._ultima_riga == ""


def test_gestisci_resize_imposta_flag_ui() -> None:
    """Un resize senza Live segnala il reflow del prompt, senza wipe dello storico."""
    ui = ConsoleUI()
    ui._necessita_ridisegno_ui.clear()
    ui._stream._live_aperto = False
    ui._gestisci_resize()
    assert ui._necessita_ridisegno_ui.is_set()


def test_gestisci_resize_live_accoda_flag_senza_reflow() -> None:
    """Il thread barra accoda il resize Live senza chiamare direttamente Rich."""
    ui = ConsoleUI()
    ui._necessita_ridisegno_ui.clear()
    ui._resize_live_richiesto.clear()
    ui._stream._live_aperto = True
    with patch.object(ui._stream, "reflow_su_resize") as reflow:
        ui._gestisci_resize()
    reflow.assert_not_called()
    assert not ui._necessita_ridisegno_ui.is_set()
    assert ui._resize_live_richiesto.is_set()


def test_scrivi_token_consuma_resize_live_sul_main_thread() -> None:
    """Il flag prodotto da un worker viene consumato da scrivi_token sul main thread."""
    ui = ConsoleUI()
    ui._stream._live_aperto = True
    thread_reflow: list[int | None] = []

    def _registra_thread_reflow() -> None:
        """Memorizza il thread che esegue concretamente il reflow Live."""
        thread_reflow.append(threading.current_thread().ident)

    with (
        patch.object(ui._stream, "scrivi_token") as scrivi_stream,
        patch.object(
            ui._stream,
            "reflow_su_resize",
            side_effect=_registra_thread_reflow,
        ),
    ):
        worker = threading.Thread(target=ui._gestisci_resize)
        worker.start()
        worker.join()

        assert thread_reflow == []
        ui.scrivi_token("chunk")

    scrivi_stream.assert_called_once_with("chunk")
    assert thread_reflow == [threading.main_thread().ident]
    assert not ui._resize_live_richiesto.is_set()


def test_finalizza_consuma_resize_prima_di_chiudere_live() -> None:
    """finalizza applica il resize pendente prima di sostituire il Panel conclusivo."""
    ui = ConsoleUI()
    ordine_operazioni: list[str] = []
    ui._resize_live_richiesto.set()

    def _registra_reflow() -> None:
        """Registra il reflow per verificare l'ordine rispetto alla chiusura."""
        ordine_operazioni.append("resize")

    def _registra_finalizzazione(_risultato: str) -> None:
        """Registra la finalizzazione simulata senza avviare un vero Rich Live."""
        ordine_operazioni.append("finalizza")

    with (
        patch.object(
            ui._stream,
            "reflow_su_resize",
            side_effect=_registra_reflow,
        ),
        patch.object(
            ui._stream,
            "finalizza",
            side_effect=_registra_finalizzazione,
        ),
    ):
        ui.finalizza("Final Answer: fatto")

    assert ordine_operazioni == ["resize", "finalizza"]
    assert not ui._resize_live_richiesto.is_set()


def test_stampa_splash_centra_il_messaggio() -> None:
    """La splash stampa il saluto; l'orologio fake non attende il wall-clock."""
    buffer = StringIO()
    ui = ConsoleUI()
    ui.console = Console(
        file=buffer,
        width=80,
        height=24,
        force_terminal=True,
        color_system=None,
    )
    tempo = 0.0
    attese: list[float] = []

    def _monotonic() -> float:
        return tempo

    def _pausa_registrata(secondi: float) -> None:
        """Avanza l'orologio fake e registra la durata."""
        nonlocal tempo
        attese.append(secondi)
        tempo += secondi

    ui.stampa_splash(pausa=_pausa_registrata, monotonic=_monotonic)

    cattura = buffer.getvalue()
    assert MESSAGGIO_SPLASH in cattura
    assert attese[-2:] == [PAUSA_DOPO_ANIMAZIONE_SEC, PAUSA_SALUTO_SEC]
    assert all(delta == INTERVALLO_FRAME_SEC for delta in attese[:-2])
    assert attese[:-2]


def test_stampa_splash_salta_console_non_terminale() -> None:
    """Pipe e capture non ricevono splash né attesa."""
    buffer = StringIO()
    ui = ConsoleUI()
    ui.console = Console(
        file=buffer,
        width=80,
        height=24,
        force_terminal=False,
        color_system=None,
    )
    attese: list[float] = []

    def _pausa_registrata(secondi: float) -> None:
        """Non deve essere invocata fuori dal TTY."""
        attese.append(secondi)

    ui.stampa_splash(pausa=_pausa_registrata)

    assert buffer.getvalue() == ""
    assert attese == []


def test_corpo_benvenuto_e_compatto() -> None:
    """Il banner di avvio ha intro, badge e /help, senza il manuale tastiera."""
    ui = ConsoleUI()
    corpo = ui._corpo_benvenuto({
        "verbose": True,
        "verifica_completa": False,
        "burn_protection": True,
        "thinking": False,
    })
    assert "Assistente pronto" in corpo
    assert "/help" in corpo
    assert "Stato:" not in corpo
    assert "Comandi:\n" not in corpo
    assert "/clear" not in corpo
    assert "Ctrl+W" not in corpo
    assert "solo dai documenti della cartella" in corpo
    assert "Verbose: [badge.on]on[/]" in corpo
    assert "Thinking: [badge.off]off[/]" in corpo
    assert "Web:" not in corpo


def test_corpo_aiuto_contiene_comandi() -> None:
    """`/help` elenca stato, slash command e guida tastiera."""
    ui = ConsoleUI()
    corpo = ui._corpo_aiuto({
        "verbose": True,
        "verifica_completa": False,
        "burn_protection": True,
        "thinking": False,
    })
    assert "Stato:" in corpo
    assert "Comandi:" in corpo
    assert "/clear" in corpo
    assert "/web" not in corpo
    assert "/archivio" not in corpo
    assert "Cartella documenti:" in corpo
    assert "/model" in corpo
    assert "/quant" not in corpo
    assert "/thinking" in corpo
    assert "/help" in corpo
    assert "Ctrl+W" in corpo
    assert "Ctrl+C" in corpo
    assert "Ctrl+V" in corpo
    assert "Option+frecce" in corpo
    assert "Selezione:" in corpo
    assert "Appunti:" in corpo
    assert "Verbose: [badge.on]on[/]" in corpo
    assert "Thinking: [badge.off]off[/]" in corpo
    assert "Web:" not in corpo


def test_ristampa_chrome_non_cancella_lo_storico(monkeypatch: pytest.MonkeyPatch) -> None:
    """Il reflow dell'input non chiama console.clear() e non ristampa il welcome."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    ui = ConsoleUI()
    ui.console = Console(
        file=StringIO(),
        width=80,
        height=24,
        force_terminal=True,
        color_system=None,
    )
    ui._titolo_prompt_corrente = "Tu"
    ui._larghezza_input_disegnata = 80
    ui._stato_benvenuto = {"verbose": False}
    with (
        patch.object(ui.console, "clear") as clear,
        patch("assistente.console.cancella_righe_sopra") as cancella,
        patch("assistente.console.stampa_composer") as composer,
        patch("assistente.console.sincronizza_dimensioni"),
        patch.object(ui.barra_termica, "ridisegna"),
    ):
        ui._ristampa_chrome("ciao", ancora=None, cursore=4)
    clear.assert_not_called()
    cancella.assert_called_once()
    composer.assert_called_once()
    titolo = composer.call_args.args[1]
    assert "Assistente pronto" not in str(titolo)


def test_ridisegna_mostra_metriche_anche_senza_verbose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tok/s e token totali restano in barra; verbose aggiunge solo le latenze."""
    console = Console(width=80, height=24, force_terminal=True)
    barra = BarraTermica(console)
    barra._attiva = True
    barra._mostra_metriche = False
    barra._metriche = _metriche_esempio()
    cattura: dict[str, object] = {}

    def _riga_finta(
        _stato: object,
        _temp: object,
        metriche: object = None,
        **kwargs: object,
    ) -> str:
        cattura["metriche"] = metriche
        cattura["verbose"] = kwargs.get("verbose")
        return "tmp: Nominal"

    monkeypatch.setattr("assistente.console.riga_barra_stato", _riga_finta)
    monkeypatch.setattr("assistente.console.tty_size", lambda _c: (80, 24))
    monkeypatch.setattr("assistente.console.leggi_stato_termico", lambda: StatoTermico.NOMINALE)
    monkeypatch.setattr("assistente.console.leggi_temperatura_cpu_celsius", lambda: 40.0)
    monkeypatch.setattr("assistente.console.detect_power", lambda: None)
    monkeypatch.setattr(barra, "_sincronizza_console", lambda: None)
    monkeypatch.setattr(barra, "_disegna", lambda _riga: None)
    barra.ridisegna()
    assert cattura["metriche"] is barra._metriche
    assert cattura["verbose"] is False

    barra.imposta_mostra_metriche(True)
    barra.ridisegna()
    assert cattura["metriche"] is barra._metriche
    assert cattura["verbose"] is True


def test_pensiero_chiude_la_riga_di_progresso() -> None:
    """«Comprendo la richiesta» non resta attaccato all'etichetta Pensiero."""
    ui = ConsoleUI()
    buffer = StringIO()
    ui.console.file = buffer
    ui.console._force_terminal = True
    ui.barra_termica.ridisegna = lambda *args, **kwargs: None
    ui.aggiorna_progresso("Comprendo la richiesta")
    ui.chiudi_progresso()
    ui.registra_azione_tool("cerca_testo", "problemi odierni")
    ui.scrivi_pensiero("Thinking Process:")
    testo = buffer.getvalue()
    assert "Comprendo la richiesta\n" in testo
    assert "Comprendo la richiestaCerco" not in testo
    assert "Comprendo la richiestaPensiero" not in testo
    assert "Pensiero Thinking Process:" in testo


def test_pensiero_nativo_etichetta_e_fuori_dal_panel() -> None:
    """Lo stream dim `Pensiero` è un flusso continuo, fuori dal panel risposta."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=80,
        height=24,
        force_terminal=True,
        color_system=None,
    )
    timeline = TimelineTurno(console)
    renderer = RendererPanelStream(console, timeline)
    timeline.scrivi_pensiero("ana")
    timeline.scrivi_pensiero("lisi")
    timeline.chiudi_pensiero()
    renderer.inizia_risposta_diretta()
    live = Mock()
    with (
        patch("assistente.console.Live", return_value=live),
        patch("assistente.console.sincronizza_dimensioni"),
        patch("assistente.console.time.monotonic", return_value=0.0),
    ):
        renderer.scrivi_token("risposta")
    assert "".join(renderer._buffer_finale) == "risposta"
    cattura = buffer.getvalue()
    assert "Pensiero" in cattura
    assert "analisi" in cattura
    assert "caratteri" not in cattura


def test_geometria_cursore_panel_input() -> None:
    assert colonna_cursore_panel(1) == 3
    assert righe_salita_panel(1, 0) == 2
    # In editing il cursore è DENTRO il contenuto: 1 sola riga fino al bordo alto.
    assert righe_da_contenuto_a_bordo_superiore(0) == 1
    assert righe_da_contenuto_a_bordo_superiore(0, riga_contenuto=3) == 3
    assert larghezza_interna_panel(119, 1) == 115


def test_cancella_righe_sopra_emette_cr_prima_di_cuu() -> None:
    """CR obbligatorio: senza, ED 0 lascia il titolo e duplica il Panel."""
    stdout = Mock()
    stdout.isatty.return_value = True

    with patch("assistente.tty_layout.sys.stdout", stdout):
        cancella_righe_sopra(2, is_terminal=True)

    stdout.write.assert_called_once_with("\r\033[2A\033[J")
    stdout.flush.assert_called_once()


def test_geometria_contenuto_e_cuu_clear() -> None:
    """Una sola API: righe visive, colonna a fine testo, CUU Enter."""
    assert righe_visive_contenuto("", 16) == 1
    assert righe_visive_contenuto("a" * 16, 16) == 1
    assert righe_visive_contenuto("a" * 17, 16) == 2
    assert righe_visive_contenuto("a" * 33, 16) == 3

    colonna, righe = geometria_cursore_contenuto("a" * 20, 16)
    assert righe == 2
    assert colonna == 4

    # width console 20 → interna 16; 33 celle → 3 CUU dal contenuto al bordo.
    assert cuu_clear_panel_input("a" * 33, 20) == 3
    assert cuu_clear_panel_input("", 20) == 1

    # Word wrap: ultima riga = ultima parola, non `len % larghezza`.
    frase = (
        "riusciresti a fornirmi i link per prendere una scheda grafica "
        "di ultima generazione e un buon processore? Forniscimi a"
    )
    colonna_parole, righe_parole = geometria_cursore_contenuto(frase, 111)
    assert righe_parole == 2
    assert colonna_parole == 12
    assert colonna_parole != (cell_len(frase) % 111)

    # Word wrap può occupare più righe di un fold a celle intere.
    colonna_ragged, righe_ragged = geometria_cursore_contenuto(
        "uno due tre quattro",
        10,
    )
    assert righe_ragged == 3
    assert colonna_ragged == 7


def test_clipboard_taglia_e_incolla_locale(monkeypatch: pytest.MonkeyPatch) -> None:
    from assistente import prompt_tty

    monkeypatch.setattr(prompt_tty.sys, "platform", "darwin")

    def boom(*_args, **_kwargs):
        raise FileNotFoundError

    monkeypatch.setattr(prompt_tty.subprocess, "run", boom)
    prompt_tty.scrivi_clipboard("hello")
    assert prompt_tty.leggi_clipboard() == "hello"
    prompt_tty.scrivi_clipboard("a\nb")
    assert prompt_tty.leggi_clipboard() == "a b"


def test_aggiorna_progresso_sovrascrive_stessa_riga() -> None:
    """Le percentuali del pull restano su una riga: EL+CR, senza newline per tick."""
    ui = ConsoleUI()
    buffer = StringIO()
    ui.console.file = buffer
    ui.console._force_terminal = True
    ui.barra_termica.ridisegna = lambda *args, **kwargs: None
    ui.aggiorna_progresso("gemma4:12b: downloading 10%")
    ui.aggiorna_progresso("gemma4:12b: downloading 20%")
    testo = buffer.getvalue()
    assert "\033[2K" in testo
    assert testo.count("\n") == 0
    assert "20%" in testo
    ui.stampa_messaggio_sistema("Modello impostato")
    chiuso = buffer.getvalue()
    assert chiuso.count("\n") >= 1
    assert "Modello impostato" in chiuso


def test_aggiorna_progresso_non_tty_va_a_capo() -> None:
    """Fuori dal TTY ogni tick è un messaggio sistema completo (log catturabile)."""
    ui = ConsoleUI()
    buffer = StringIO()
    ui.console.file = buffer
    ui.console._force_terminal = False
    ui.barra_termica.ridisegna = lambda *args, **kwargs: None
    ui.aggiorna_progresso("gemma4:12b: downloading 10%")
    ui.aggiorna_progresso("gemma4:12b: downloading 20%")
    testo = buffer.getvalue()
    assert testo.count("10%") == 1
    assert testo.count("20%") == 1
    assert testo.count("\n") >= 2


def test_larghezza_lettura_rispetta_il_cap() -> None:
    """Su terminali larghi il markdown non supera 88 colonne."""
    assert larghezza_lettura(160) == LARGHEZZA_LETTURA_MARKDOWN
    assert larghezza_lettura(40) == 40
    assert larghezza_lettura(0) == 1


def test_cuu_clear_composer_conta_prefisso() -> None:
    """Il composer senza box sale delle righe visive di `› ` + testo."""
    assert riga_composer("ciao").startswith(PREFISSO_COMPOSER)
    assert cuu_clear_composer("", 80) == 1
    lungo = "a" * 90
    assert cuu_clear_composer(lungo, 40) == righe_visive_contenuto(
        riga_composer(lungo), 40
    )


def test_frase_attivita_tool_in_italiano() -> None:
    """I tool noti hanno una frase umana; gli altri restano leggibili."""
    assert frase_attivita_tool("cerca_testo") == "Cerco nei documenti"
    assert frase_attivita_tool("scrivi_file") == "Scrivo"
    assert frase_attivita_tool("tool_sconosciuto") == "tool sconosciuto"


def test_timeline_stampa_attivita_su_una_riga() -> None:
    """La timeline non usa glifi ⚡/↳; in verbose aggiunge il nome tecnico."""
    buffer = StringIO()
    console = Console(
        file=buffer,
        width=80,
        height=24,
        force_terminal=True,
        color_system=None,
    )
    timeline = TimelineTurno(console)
    timeline.registra_azione_tool("cerca_testo", "balenottera azzurra")
    testo = buffer.getvalue()
    assert "Cerco nei documenti" in testo
    assert "balenottera azzurra" in testo
    assert "⚡" not in testo
    assert "cerca_testo" not in testo

    timeline.reset_turno()
    timeline.imposta_verbose(True)
    timeline.registra_azione_tool("scrivi_file", "note.txt")
    verbose = buffer.getvalue()
    assert "Scrivo" in verbose
    assert "note.txt" in verbose
    assert "(scrivi_file)" in verbose


def test_stream_usa_cap_lettura_su_console_larga() -> None:
    """Live riceve un Constrain a 88 colonne anche se il TTY è più largo."""
    live = Mock()
    renderer, _ = _crea_renderer_isolato()
    renderer._console.width = 160

    with (
        patch("assistente.console.Live", return_value=live) as crea_live,
        patch("assistente.console.sincronizza_dimensioni"),
        patch("assistente.console.time.monotonic", return_value=0.0),
    ):
        renderer.inizia_risposta_diretta()
        renderer.scrivi_token("ciao")

    constrain = crea_live.call_args.args[0].renderables[-1]
    assert constrain.width == LARGHEZZA_LETTURA_MARKDOWN


def test_markup_header_sessione_include_modello_e_help() -> None:
    """L'header sticky è una riga: modello, cartella documenti, thinking, /help."""
    markup = markup_header_sessione(
        {
            "verbose": False,
            "verifica_completa": False,
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
    spento = markup_header_sessione(
        {"thinking": True},
        "fast · q4 · gemma4:e4b",
    )
    assert "cartella" not in spento
    assert "thinking" in spento
    assert "thinking off" not in spento


def test_geometria_composer_sticky() -> None:
    """Il composer sticky sta sopra lo stato e non mangia header e transcript."""
    assert altezza_composer("", 80) == 2
    assert altezza_composer_sticky("", 80, 24, header=True) == 2
    assert riga_inizio_composer_sticky(24, 2) == 22
    assert sequenza_regione_scroll(2, 21) == "\033[2;21r"
    alto = altezza_composer_sticky("a" * 400, 40, 24, header=True)
    assert 2 <= alto <= 6


def test_sincronizza_sottrae_chrome_esplicito(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Header + stato riducono l'altezza Live, non solo la riga della barra."""
    console = Console(width=80, height=24, force_terminal=True)

    def _size_finto(_console: Console) -> tuple[int, int]:
        return 100, 40

    monkeypatch.setattr("assistente.tty_layout.tty_size", _size_finto)
    _colonne, _righe = sincronizza_dimensioni(
        console,
        riserva_barra=True,
        righe_riservate=3,
    )
    assert console.height == 37


def test_righe_chrome_conta_header_e_composer() -> None:
    """Con barra attiva il transcript perde header, composer e stato."""
    console = Console(width=80, height=24, force_terminal=True)
    barra = BarraTermica(console)
    assert barra.righe_chrome() == 1
    barra._attiva = True
    barra._header_righe = 1
    barra._righe_composer = 2
    assert barra.righe_chrome() == 4
    assert barra.header_attivo is True


def test_ristampa_chrome_sticky_non_usa_cuu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Con chrome sticky il reflow non cancella lo storico con CUU+ED."""
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    ui = ConsoleUI()
    ui.console = Console(
        file=StringIO(),
        width=80,
        height=24,
        force_terminal=True,
        color_system=None,
    )
    ui.barra_termica._attiva = True
    ui.barra_termica._header_righe = 1
    ui._titolo_prompt_corrente = "Tu"
    with (
        patch.object(ui.console, "clear") as clear,
        patch("assistente.console.cancella_righe_sopra") as cancella,
        patch("assistente.console.stampa_composer") as composer,
        patch.object(ui, "_disegna_header_sticky") as header,
        patch.object(ui, "_disegna_composer_sticky") as sticky,
        patch.object(ui, "_sincronizza"),
        patch.object(ui.barra_termica, "ridisegna"),
    ):
        ui._ristampa_chrome("ciao", ancora=None, cursore=4)
    clear.assert_not_called()
    cancella.assert_not_called()
    composer.assert_not_called()
    header.assert_called_once()
    sticky.assert_called_once()
