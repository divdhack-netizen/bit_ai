"""Route del turno dalla scheda, saluto locale, riversamento dello storico e scope lock."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

from assistente.schema_turno import SchemaTurno

PercorsoTurno = Literal["direct", "react"]

ANTI_GOLD_PLATING = (
    "Scope lock: fai solo quanto chiesto. "
    "Non aggiungere feature, refactor o miglioramenti non richiesti. "
    "Non creare helper/utilità per operazioni una tantum. "
    "Tre righe simili sono meglio di un'astrazione prematura."
)

MOTIVO_CHAT_LOCALE = "chat breve senza contenuto da ricercare"

_PUNTEGGIATURA_CHAT = re.compile(r"[,;:]+")
_INTENT_CHAT_LOCALE = re.compile(
    r"^(?:"
    r"ciao(?:\s+ciao)*|hey|hi|hello|salve|"
    r"buongiorno|buonasera|buonanotte|"
    r"grazie(?:\s+mille)?|thanks|thank\s+you|"
    r"prego|di\s+niente|"
    r"(?:ok(?:ay)?|va\s+bene|perfetto|ottimo|capito)(?:\s+grazie(?:\s+mille)?)?|"
    r"come\s+stai|how\s+are\s+you"
    r")[\s!.?]*$",
    re.IGNORECASE,
)

_KINDS_SENZA_TOOL = frozenset({"chat", "explain"})


@dataclass(frozen=True, slots=True)
class AnalisiRichiesta:
    """Segnali del turno presi dallo schema, non dal lessico della frase."""

    scrittura: bool
    workspace: bool
    spiegazione: bool


@dataclass(frozen=True, slots=True)
class DecisioneTurno:
    """Esito usato da SessioneAgente per routing e gate di scrittura."""

    percorso: PercorsoTurno
    scrittura_consentita: bool
    analisi: AnalisiRichiesta
    motivo_percorso: str = ""


def analisi_neutra() -> AnalisiRichiesta:
    """Nessun segnale: il testo non viene classificato."""
    return AnalisiRichiesta(scrittura=False, workspace=False, spiegazione=False)


def analisi_da_schema(schema: SchemaTurno) -> AnalisiRichiesta:
    """Traduce la scheda in segnali per il prompt, senza leggere la frase."""
    return AnalisiRichiesta(
        scrittura=schema.write or schema.deliver == "file",
        workspace=schema.kind == "workspace" or bool(schema.paths or schema.folder),
        spiegazione=schema.kind == "explain",
    )


def decisione_da_schema(schema: SchemaTurno) -> DecisioneTurno:
    """
    Route solo dai campi della scheda.

    Closed-corpus: ogni domanda di contenuto passa dai documenti (ReAct).
    Solo la chat pura, senza destinazione, resta una generazione diretta.
    """
    analisi = analisi_da_schema(schema)
    ha_destinazione = bool(
        schema.paths
        or schema.folder
        or schema.write
        or schema.deliver == "file"
    )
    if schema.kind == "chat" and not ha_destinazione:
        percorso: PercorsoTurno = "direct"
        motivo = "chat senza contenuto da cercare"
    elif schema.kind in _KINDS_SENZA_TOOL and not ha_destinazione:
        percorso = "react"
        motivo = "domanda da verificare nei documenti"
    else:
        percorso = "react"
        motivo = "compito con tool"
    scrittura = _scrittura_consentita(schema, percorso)
    return DecisioneTurno(
        percorso=percorso,
        scrittura_consentita=scrittura,
        analisi=analisi,
        motivo_percorso=motivo,
    )


def _lettura_file_senza_scrittura(schema: SchemaTurno) -> bool:
    """Un path senza write, cartella o consegna è il file da leggere."""
    return bool(
        schema.paths
        and not schema.write
        and schema.deliver != "file"
        and not schema.folder.strip()
    )


def _scrittura_consentita(schema: SchemaTurno, percorso: PercorsoTurno) -> bool:
    """Il ReAct scrive, tranne letture di file e domande sui documenti."""
    if schema.write or schema.deliver == "file":
        return True
    if _lettura_file_senza_scrittura(schema):
        return False
    if schema.kind in _KINDS_SENZA_TOOL:
        return False
    return percorso == "react"


def decisione_react_aperta() -> DecisioneTurno:
    """Schema assente: tool completi, scrittura dentro la sandbox."""
    return DecisioneTurno(
        percorso="react",
        scrittura_consentita=True,
        analisi=analisi_neutra(),
        motivo_percorso="schema assente: ReAct completo",
    )


def decisione_saluto() -> DecisioneTurno:
    """Saluto esatto: una generazione locale, senza schema e senza tool."""
    return DecisioneTurno(
        percorso="direct",
        scrittura_consentita=False,
        analisi=analisi_neutra(),
        motivo_percorso=MOTIVO_CHAT_LOCALE,
    )


def _e_chat_locale(richiesta: str) -> bool:
    """
    True se il turno è solo saluto, ringraziamento o conferma.

    «ciao, chi è il presidente?» resta False: non è solo un saluto.
    """
    testo = (richiesta or "").strip()
    if not testo:
        return True
    if _INTENT_CHAT_LOCALE.fullmatch(testo):
        return True
    normalizzato = re.sub(r"\s+", " ", _PUNTEGGIATURA_CHAT.sub(" ", testo)).strip()
    return bool(_INTENT_CHAT_LOCALE.fullmatch(normalizzato))


def formatta_contesto_intent(decisione: DecisioneTurno) -> str:
    """Serializza la route già scelta, per il prompt ReAct."""
    analysis = decisione.analisi
    intent_flags = [
        name
        for flag, name in (
            (analysis.scrittura, "scrittura"),
            (analysis.workspace, "workspace"),
            (analysis.spiegazione, "concettuale"),
        )
        if flag
    ] or ["chat"]
    return "\n".join(
        (
            "INTENT CONTEXT:",
            f"- Intent: {', '.join(intent_flags)}",
            f"- Percorso: {decisione.percorso} ({decisione.motivo_percorso})",
            f"- Scrittura consentita: {'sì' if decisione.scrittura_consentita else 'no'}",
        )
    )


def capabilities_react() -> tuple[str, ...]:
    """Tool del ReAct: file della cartella, riassunto e data. Nessuna rete."""
    return ("file", "summary", "date")


# Riversamento della chat su file: nessun tema nuovo, solo lo storico.
_ESTENSIONI_DOCUMENTO = r"txt|md|json|ya?ml|html?"
_PATH_DOCUMENTO = re.compile(
    rf"[\w./-]+\.(?:{_ESTENSIONI_DOCUMENTO})\b",
    re.IGNORECASE,
)
_STILE_CODA = re.compile(
    r"\.?\s*sii?\s+prolisso(?:\s+e\s+approfondisci)?\.?\s*$",
    re.IGNORECASE,
)
_SCAFFOLD_CLAUSE = re.compile(
    r"(?:"
    r"(?:nella|dentro|in)\s+(?:la\s+)?cartella\s+[^\s.,;:]+|"
    r"(?:crea(?:re|ndo)?|realizza(?:re)?)\s+(?:una\s+|la\s+)?cartella"
    r"(?:\s+dal\s+nome)?\s+[^\s.,;:]+|"
    r"al\s+suo\s+interno|"
    r"in\s+quella\s+cartella|"
    r"all['’]interno\s+(?:di(?:lla)?|della\s+cartella|di\s+essa)\s*[^\s.,;:]*|"
    r"crea(?:re|ndo)?\s+(?:un\s+|il\s+)?file\s+[\w./-]+\.\w+|"
    r"(?:un|il)\s+documento\s+[\w./-]+\.\w+|"
    r"in\s+questo\s+documento"
    r")",
    re.IGNORECASE,
)
_TRAILING_FILE_INSTRUCTION = re.compile(
    r"\s+(?:e\s+)?(?:scriv\w*|salv\w*).{0,100}"
    rf"[\w./ -]+\.(?:{_ESTENSIONI_DOCUMENTO})\s*$",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W\d_]{3,}", re.UNICODE)
_META_STOPWORDS = frozenset(
    {
        "all",
        "approfondisci",
        "bene",
        "cartella",
        "che",
        "chi",
        "cio",
        "ciò",
        "come",
        "con",
        "consultare",
        "cosa",
        "crea",
        "creare",
        "dal",
        "del",
        "della",
        "delle",
        "degli",
        "directory",
        "documents",
        "dunque",
        "esclusivamente",
        "file",
        "fonti",
        "fornisci",
        "interno",
        "nella",
        "nome",
        "online",
        "per",
        "più",
        "prolisso",
        "qual",
        "quale",
        "quali",
        "quelle",
        "quindi",
        "riesci",
        "scrivere",
        "scrivi",
        "sono",
        "sul",
        "suo",
        "trovare",
        "tutto",
        "una",
        "uno",
    }
)
# Verbi e deissi del riversamento: non sono un argomento da cercare.
_PAROLE_SENZA_TEMA = _META_STOPWORDS | frozenset(
    {
        "adesso",
        "avevi",
        "chat",
        "conversazione",
        "detta",
        "detto",
        "documento",
        "gia",
        "già",
        "hai",
        "memoria",
        "nel",
        "nelle",
        "nello",
        "negli",
        "ora",
        "precedente",
        "precedenti",
        "prima",
        "quanto",
        "qui",
        "quella",
        "quelli",
        "quello",
        "questa",
        "queste",
        "questi",
        "questo",
        "riporta",
        "riportala",
        "riportalo",
        "riportami",
        "riportare",
        "risultato",
        "risultati",
        "salva",
        "salvala",
        "salvalo",
        "salvami",
        "salvare",
        "scritta",
        "scritti",
        "scritto",
        "sopra",
        "storico",
        "turno",
        "turni",
    }
)
_CUE_RIVERSAMENTO = re.compile(
    r"(?:"
    r"\briporta(?:mi|re|lo|la)?\s+tutto\b|"
    r"\bsul\s+file\b|"
    r"\bsalva(?:re|lo|la|mi)?\b"
    r")",
    re.IGNORECASE,
)


def _pulisci_tema(tema: str) -> str:
    """Toglie code di stile («sii prolisso») dal tema."""
    return _STILE_CODA.sub("", (tema or "").strip()).strip(" .;,")


def path_documento_nominato(testo: str) -> str | None:
    """Primo nome di documento citato nella frase, oppure None."""
    paths = tuple(dict.fromkeys(_PATH_DOCUMENTO.findall((testo or "").strip())))
    return paths[0] if paths else None


def e_riversamento_storico(testo: str) -> bool:
    """
    True se la frase chiede di riversare la chat, senza un tema nuovo.

    «riportami tutto», «sul file» e «salva» contano solo se, tolti il documento
    e i verbi di salvataggio, non resta un argomento da cercare. Due path
    restano ambigui e non prendono questa scorciatoia.
    """
    grezzo = (testo or "").strip()
    if not grezzo or _CUE_RIVERSAMENTO.search(grezzo) is None:
        return False
    paths = tuple(dict.fromkeys(_PATH_DOCUMENTO.findall(grezzo)))
    if len(paths) > 1:
        return False
    return not _ha_tema_nuovo(grezzo)


def _ha_tema_nuovo(testo: str) -> bool:
    """True se, tolti path e verbi di salvataggio, resta un argomento."""
    senza_path = _PATH_DOCUMENTO.sub(" ", testo or "")
    ripulito = _strip_scaffold(senza_path)
    return any(
        termine not in _PAROLE_SENZA_TEMA
        for termine in _WORD.findall(ripulito.casefold())
    )


def _strip_scaffold(query: str) -> str:
    """Rimuove clausole di cartella/file senza toccare il tema."""
    stripped = _SCAFFOLD_CLAUSE.sub(" ", query or "")
    stripped = _TRAILING_FILE_INSTRUCTION.sub("", stripped)
    return _pulisci_tema(" ".join(stripped.split()))
