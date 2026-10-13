"""Comprensione deterministica: regole del testo, scheda locale e destinazioni.

La scheda locale decide senza LLM i turni chiari (domande, letture, una sola
destinazione nuova). Nel dubbio torna None e la sessione chiede la scheda al modello.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

from assistente.schema_turno import MAX_PATHS, MAX_SUBJECT_CARATTERI, SchemaTurno

# Unica fonte delle estensioni file per comprensione, validatore e sessione.
# Ogni nuovo tipo di file si aggiunge solo qui.
ESTENSIONI_FILE: tuple[str, ...] = (
    "py", "js", "ts", "tsx", "jsx",
    "htm", "html", "css",
    "json", "md", "txt",
    "java", "toml", "yml", "yaml",
    "pdf", "docx", "pages",
    "png", "jpg", "jpeg", "webp",
)
_ESTENSIONI_FILE = "|".join(ESTENSIONI_FILE)

_SPAZI = re.compile(r"\s+")
# Path file citati nel testo, nell'ordine di apparizione.
PATH_FILE = re.compile(
    rf"[\w./-]+\.(?:{_ESTENSIONI_FILE})\b",
    re.IGNORECASE,
)
_CARTELLA = re.compile(
    r"\b(?:cartella|directory|dir)\s+[`'\"]?([A-Za-z0-9_./-]+)[`'\"]?",
    re.IGNORECASE,
)
_CARTELLA_DAL_NOME = re.compile(
    r"\b(?:cartella|directory|dir)\s+dal\s+nome\s+[`'\"]?([A-Za-z0-9_./-]+)[`'\"]?",
    re.IGNORECASE,
)
_CARTELLA_CREATA = re.compile(
    r"\bcrea(?:re|ndo)?\s+(?:una\s+|la\s+)?(?:cartella|directory|dir)\s+"
    r"(?:dal\s+nome\s+)?[`'\"]?/?([A-Za-z0-9_./-]+)[`'\"]?",
    re.IGNORECASE,
)
_CARTELLA_CONTENITORE = re.compile(
    r"\b(?:nella|dentro(?:\s+la)?|in)\s+(?:la\s+)?"
    r"(?:cartella|directory|dir)\s+[`'\"]?/?([A-Za-z0-9_./-]+)[`'\"]?",
    re.IGNORECASE,
)
_CARTELLA_STOP = frozenset(
    {
        "dal", "di", "nome", "una", "la", "un", "il",
        # «cartella attuale/corrente/presente» è la root, non una cartella di nome «attuale».
        "attuale", "corrente", "presente", "questa", "stessa", "principale",
    }
)
# Tipo documentale senza nome file (es. «documento .txt»).
_DOCUMENTO_SENZA_NOME = re.compile(
    rf"(?:"
    rf"(?:un\s+)?documento\s*\.?\s*(?:{_ESTENSIONI_FILE})\b|"
    rf"(?:in\s+)?(?:un\s+)?file\s+(?:di\s+testo|txt)\b|"
    rf"\bfile\s+\.(?:{_ESTENSIONI_FILE})\b"
    rf")",
    re.IGNORECASE,
)
_GERUNDIO_SCRITTURA = re.compile(
    r"(?:"
    r"scrivend\w*|salvand\w*|creand\w*|generand\w*|"
    r"mettendol[oaie]|riassumendol[oaie]|"
    r"writing\s+(?:it\s+)?(?:to|in|into)|"
    r"saving\s+(?:it\s+)?(?:to|as|in)"
    r")",
    re.IGNORECASE,
)
VINCOLO_NO_FILE = re.compile(
    r"(?:"
    r"\bniente\s+file\b|"
    r"\bnessun\s+file\b|"
    r"\bsenza\s+(?:creare\s+|salvare\s+|scrivere\s+)?file\b|"
    r"\bnon\s+(?:creare|salvare|scrivere|toccare)\s+"
    r"(?:un\s+|il\s+|alcun\s+)?file\b|"
    r"\bsolo\s+(?:in\s+)?chat\b|"
    r"\bno\s+files?\b|"
    r"\bdon'?t\s+(?:write|save|create)\s+(?:a\s+)?files?\b"
    r")",
    re.IGNORECASE,
)
# Follow-up deittico corto: «e il secondo?» non è una domanda autonoma.
_DEITTICO_CORTO = re.compile(
    r"^(?:"
    r"(?:e\s+)?(?:il|la|lo|i|le)\s+"
    r"(?:prim[oaie]|second[oaie]|terz[oaie]|quart[oaie]|quint[oaie]|ultim[oaie]?)"
    r"|"
    r"(?:e\s+)?(?:quello|quella|questo|questa|quelli|quelle)"
    r"|"
    r"the\s+(?:first|second|third|fourth|fifth|last)(?:\s+one)?"
    r")\s*\??$",
    re.IGNORECASE,
)
CUE_LETTURA = re.compile(
    r"\b(?:leggi|leggere|riassumi|riassumere|dimmi|spiega|riporta)\b",
    re.IGNORECASE,
)
# I clitici («salvalo», «scrivimi») restano sullo stesso verbo.
CUE_SALVATAGGIO = re.compile(
    r"\b(?:scrivi\w*|salva\w*|crea\w*|accoda\w*)\b",
    re.IGNORECASE,
)
# «riassumi il documento in y.txt» ha un verbo di lettura, ma il file è la destinazione.
CUE_DESTINAZIONE_FILE = re.compile(
    rf"\b(?:in|su|sul|sulla|dentro|nel|nello|nella)\s+"
    rf"(?:(?:un|il|lo|la)\s+)?(?:file\s+)?"
    rf"[\w./-]+\.(?:{_ESTENSIONI_FILE})\b",
    re.IGNORECASE,
)
# Mutazioni di file esistenti e scritture implicite («appuntami», «in un txt»):
# la scheda locale non le decide, passa all'LLM.
_CUE_AMBIGUO = re.compile(
    r"\b(?:modific\w*|aggiorn\w*|aggiung\w*|corregg\w*|corress\w*|sostituisc\w*|"
    r"cambi\w*|rinomin\w*|elimin\w*|cancell\w*|rimuov\w*|spost\w*|"
    r"inserisc\w*|mett\w*|append\w*|"
    r"appunt\w*|annot\w*|archivi\w*|memorizz\w*|registr\w*|"
    r"segna(?:mi|lo|la|li|le|telo)?)\b"
    r"|(?<![.\w])txt\b",
    re.IGNORECASE,
)
# Sotto questa soglia, senza path né cartella, il testo è troppo corto per deciderlo.
_PAROLE_MINIME_LOCALI = 3
# Azioni del mondo reale che l'agente non può compiere: meglio dirlo che fingere.
# Il prefisso evita i falsi positivi descrittivi («spiegami come mandare…»,
# «una guida per prenotare…»), che sono normali domande informative.
_NON_ISTRUZIONE = (
    r"(?<!\bcome\s)(?<!\bper\s)(?<!\bspiegami\s)(?<!\bspiega\s)(?<!\bsi\s)"
)
_AZIONI_NON_SUPPORTATE: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "email",
        re.compile(
            _NON_ISTRUZIONE
            + r"\b(?:manda|mandami|mandagli|mandale|mandi|mandare|"
            r"invia|inviami|inviagli|inviare|inviarlo|inviarla|spedisci|spedire)\b"
            r"[\s\S]{0,40}\b(?:e-?mail|mail|messaggio|messaggi|sms|whatsapp|telegram)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "generazione",
        re.compile(
            _NON_ISTRUZIONE
            + r"\b(?:genera|generami|generare|crea|creami|creare|disegna|"
            r"disegnami|disegnare|produci|renderizza)\b[\s\S]{0,40}"
            r"(?:\b(?:immagine|foto|fotografia|illustrazione|disegno|"
            r"render|video|logo)\b|\bpdf\b(?!\.))",
            re.IGNORECASE,
        ),
    ),
    (
        "shell",
        re.compile(
            _NON_ISTRUZIONE
            + r"\b(?:esegui|eseguilo|eseguila|eseguire|lancia|lanciare|"
            r"avvia|avviare)\b[\s\S]{0,40}"
            r"\b(?:comando|command|shell|terminale|script)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "prenotazione",
        re.compile(
            _NON_ISTRUZIONE
            + r"\b(?:prenota|prenotami|prenotare|prenotiamo)\b[\s\S]{0,40}"
            r"\b(?:volo|treno|hotel|albergo|tavolo|ristorante|camera|"
            r"biglietto|biglietti|noleggio|auto)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "audio",
        re.compile(
            _NON_ISTRUZIONE
            + r"(?:\b(?:ascolta|ascoltare|trascrivi|trascrivere)\b[\s\S]{0,40}"
            r"\baudio\b"
            r"|\baudio\b[\s\S]{0,40}\b(?:trascrivi|trascrivere|ascolta)\b)",
            re.IGNORECASE,
        ),
    ),
)


@dataclass(frozen=True, slots=True)
class ComprensioneTurno:
    """Tema e path del turno, già fissati dalla scheda."""

    compito: str
    paths: tuple[str, ...]


def formatta_blocco_comprensione(comprensione: ComprensioneTurno) -> str:
    """
    Serializza COMPITO per il prompt, o stringa vuota se non serve iniezione.

    Returns:
        Una riga «COMPITO: …» oppure "".
    """
    if not comprensione.compito:
        return ""
    return f"COMPITO: {comprensione.compito}"


def normalizza(richiesta: str) -> str:
    """Collassa whitespace; non riscrive il lessico (niente spellcheck)."""
    return _SPAZI.sub(" ", str(richiesta or "")).strip()


def estrai_paths(testo: str) -> tuple[str, ...]:
    """Path file espliciti, senza duplicati, nell'ordine di apparizione."""
    return tuple(dict.fromkeys(PATH_FILE.findall(testo or "")))


def azione_esterna_non_supportata(richiesta: str) -> str | None:
    """
    Capacità esterna richiesta che l'agente non possiede, o None.

    Returns:
        Una fra email, generazione, shell, prenotazione, audio; oppure None.
    """
    testo = richiesta or ""
    for capability, pattern in _AZIONI_NON_SUPPORTATE:
        if pattern.search(testo):
            return capability
    return None


def scheda_locale(
    richiesta: str,
    esiste: Callable[[str], bool],
) -> SchemaTurno | None:
    """
    Scheda certa senza LLM, oppure None se il testo è ambiguo.

    Decide solo letture e domande, e scritture con un verbo esplicito e una
    sola destinazione nuova. Modifiche, follow-up deittici e scritture senza
    destinazione restano al modello.

    Args:
        richiesta: Messaggio originale del turno.
        esiste: True se il path esiste già nella cartella documenti (sorgente).

    Returns:
        SchemaTurno da passare al validatore, oppure None.
    """
    testo = normalizza(richiesta)
    if not testo or _DEITTICO_CORTO.fullmatch(testo):
        return None
    subject = testo[:MAX_SUBJECT_CARATTERI]
    if azione_esterna_non_supportata(testo) is not None:
        # Il chiarimento onesto lo scrive valida_turno.
        return SchemaTurno(kind="explain", subject=subject)
    paths = estrai_paths(testo)
    cartella = estrai_cartella_nominata(testo) or ""
    if not paths and not cartella and len(testo.split()) < _PAROLE_MINIME_LOCALI:
        return None
    no_file = VINCOLO_NO_FILE.search(testo) is not None
    if not no_file and _CUE_AMBIGUO.search(testo):
        return None
    verbo_scrittura = bool(
        CUE_SALVATAGGIO.search(testo) or _GERUNDIO_SCRITTURA.search(testo)
    )
    indizio_file = bool(
        _DOCUMENTO_SENZA_NOME.search(testo) or CUE_DESTINAZIONE_FILE.search(testo)
    )
    if no_file or not (verbo_scrittura or indizio_file):
        return SchemaTurno(
            kind="workspace" if paths or cartella else "explain",
            subject=subject,
            paths=paths[:MAX_PATHS],
            folder=cartella,
        )
    nuovi = [path for path in paths if not esiste(path)]
    if verbo_scrittura and len(nuovi) == 1 and len(paths) <= MAX_PATHS:
        return SchemaTurno(
            kind="workspace",
            write=True,
            deliver="file",
            subject=subject,
            paths=paths,
            folder=cartella,
        )
    return None


def _pulisci_nome_cartella(raw: str | None) -> str | None:
    """Normalizza un nome cartella e scarta token grammaticali."""
    if not raw:
        return None
    candidate = raw.strip().rstrip("/.").lstrip("/")
    if not candidate or candidate == ".":
        return None
    if candidate.casefold() in _CARTELLA_STOP:
        return None
    return candidate


def estrai_cartella_nominata(testo: str) -> str | None:
    """Cartella da creare, eventuale contenitore, o prima menzione utile."""
    dal_match = _CARTELLA_DAL_NOME.search(testo)
    crea_match = _CARTELLA_CREATA.search(testo)
    box_match = _CARTELLA_CONTENITORE.search(testo)
    child = _pulisci_nome_cartella(
        (dal_match or crea_match).group(1) if (dal_match or crea_match) else None
    )
    parent = _pulisci_nome_cartella(
        box_match.group(1) if box_match else None
    )
    if parent and child and parent.casefold() != child.casefold():
        return f"{parent}/{child}"
    if child:
        return child
    if parent:
        return parent
    fall_match = _CARTELLA.search(testo)
    return _pulisci_nome_cartella(
        fall_match.group(1) if fall_match else None
    )


def componi_path_in_cartella(file_path: str, cartella: str | None) -> str:
    """Annida il file nella cartella nominata, senza duplicare il prefisso."""
    if not cartella:
        return file_path
    if file_path == cartella or file_path.startswith(f"{cartella}/"):
        return file_path
    nome = file_path.rsplit("/", 1)[-1]
    return f"{cartella}/{nome}"


def risolvi_destinazione(
    estratto: str | None,
    cartella: str | None,
    precedente: str | None,
) -> str | None:
    """
    Destinazione di scrittura: path del turno, nidificato, o documento precedente.

    Un follow-up che cita solo il basename riusa il path nidificato già scritto.
    """
    if not estratto:
        return precedente
    nested = componi_path_in_cartella(estratto, cartella)
    if (
        precedente
        and not cartella
        and "/" not in estratto
        and precedente.rsplit("/", 1)[-1] == estratto.rsplit("/", 1)[-1]
    ):
        return precedente
    return nested
