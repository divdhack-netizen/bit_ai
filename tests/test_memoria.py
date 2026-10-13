"""Test unit: memoria conversazionale e blocco storico nel prompt ReAct."""

from __future__ import annotations

from assistente.memoria import MemoriaConversazione, blocco_storico_prompt
from assistente.prompt_budget import estimate_tokens
from assistente.politica_turno import decisione_da_schema
from assistente.schema_turno import SchemaTurno
from assistente.session import _costruisci_descrizione_task

_DOMANDA = decisione_da_schema(SchemaTurno(kind="explain"))


def test_memoria_registra_e_formatta() -> None:
    """Dopo un turno, formatta espone Utente e Assistente."""
    memoria = MemoriaConversazione()
    memoria.registra("domanda uno", "risposta uno")
    testo = memoria.formatta()
    assert "Utente: domanda uno" in testo
    assert "Assistente: risposta uno" in testo
    assert memoria.ha_turni() is True


def test_memoria_ignora_turni_vuoti() -> None:
    """Registrazioni incomplete non entrano nello storico."""
    memoria = MemoriaConversazione()
    memoria.registra("  ", "ok")
    memoria.registra("ok", "")
    assert memoria.ha_turni() is False


def test_memoria_tronca_risposta_lunga() -> None:
    """Risposte oltre il limite tengono la coda e segnalano il taglio."""
    memoria = MemoriaConversazione(max_caratteri_risposta=90)
    memoria.registra("q", "inizio\n" + ("x" * 400) + "\nfine esito")
    testo = memoria.formatta()
    assert "contenuto ridotto" in testo
    assert "fine esito" in testo


def test_memoria_compatto_include_esito_quando_il_budget_basta() -> None:
    """I turni omessi restano nel compatto, con una riga di esito."""
    memoria = MemoriaConversazione(max_turni=6, max_caratteri=50_000)
    memoria.registra("prima domanda", "elenco\n- alfa\n- beta")
    memoria.registra("seconda " + ("y" * 2000), "fatto\npath: note.md")
    testo = memoria.formatta(max_tokens=220)
    assert "Compact storico" in testo
    assert "Esiti:" in testo
    assert "elenco" in testo
    assert estimate_tokens(testo) <= 220


def test_memoria_numerazione_assoluta_dopo_riduzione() -> None:
    """I turni restano numerati con l'indice di sessione, non rinumerati da 1."""
    memoria = MemoriaConversazione(max_turni=2)
    memoria.registra("u1", "a1")
    memoria.registra("u2", "a2")
    memoria.registra("u3", "a3")
    testo = memoria.formatta()
    assert "Turno 2" in testo
    assert "Turno 3" in testo
    assert "Turno 1" not in testo


def test_memoria_limite_turni() -> None:
    """Solo gli ultimi N turni restano nel buffer."""
    memoria = MemoriaConversazione(max_turni=2)
    memoria.registra("u1", "a1")
    memoria.registra("u2", "a2")
    memoria.registra("u3", "a3")
    testo = memoria.formatta()
    assert "u1" not in testo
    assert "u2" in testo
    assert "u3" in testo


def test_memoria_limite_caratteri_riduce_senza_formatta_ripetuto() -> None:
    """
    Con budget basso, _riduci elimina i turni vecchi rispettando max_caratteri.

    Verifica anche che la lunghezza formattata resti ≤ budget dopo molte registrazioni.
    """
    memoria = MemoriaConversazione(max_turni=6, max_caratteri=400, max_caratteri_risposta=200)
    # Molti turni lunghi: deve scartare i più vecchi senza superare il budget.
    for indice in range(12):
        memoria.registra(f"domanda-{indice}", "x" * 180)
    assert len(memoria.turni) < 12
    assert len(memoria.formatta()) <= 400


def test_memoria_svuota() -> None:
    """svuota rimuove tutto lo storico."""
    memoria = MemoriaConversazione()
    memoria.registra("q", "a")
    memoria.svuota()
    assert memoria.ha_turni() is False
    assert memoria.formatta() == ""


def test_descrizione_task_include_storico() -> None:
    """Il prompt ReAct include lo storico, ma i fatti restano nei documenti."""
    task = _costruisci_descrizione_task(
        "approfondisci il punto 1",
        verifica_completa=False,
        decisione=_DOMANDA,
        storico="Utente: elenca ricerche\nAssistente: 1. Miller 2. Unit-A",
    )
    assert "STORICO CONVERSAZIONE" in task
    assert "Miller" in task
    assert "approfondisci il punto 1" in task
    assert "verificati nei documenti" in task
    assert "cerca_web" not in task
    assert "esplicita i passaggi essenziali" in task
    assert task.index("Workflow:") < task.index("STORICO CONVERSAZIONE")
    assert task.index("STORICO CONVERSAZIONE") < task.index("RICHIESTA:")


def test_descrizione_task_listing_cita_cartella() -> None:
    """Il task ReAct di listing impone l'Action Input della cartella nominata."""
    task = _costruisci_descrizione_task(
        "Elenca i file nella cartella assistente senza modificarli.",
        verifica_completa=False,
        decisione=_DOMANDA,
    )
    assert "Action Input obbligatorio: assistente" in task
    assert "cerca_web" not in task
    assert "elenca i nomi osservati" in task


def test_blocco_storico_prompt_mantiene_non_negarlo() -> None:
    """Direct/ReAct conservano il prefisso classico con «non negarlo»."""
    blocco = blocco_storico_prompt("Utente: ciao\nAssistente: ciao")
    assert "non negarlo" in blocco
    assert blocco_storico_prompt("") == ""
