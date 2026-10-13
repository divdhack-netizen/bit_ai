"""Sessione agente: cache LLM/executor, prompt e orchestrazione turni."""

from __future__ import annotations

import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_core.agents import AgentAction
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.exceptions import OutputParserException
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_ollama import OllamaLLM

from assistente.config import (
    ALIAS_MODELLO_DEFAULT,
    CONTESTO_LLM_TOKEN,
    ESTENSIONI_SCRIVIBILI,
    ConfigRuntime,
    MAX_ITERAZIONI_AGENTE,
    MAX_TOKEN_GENERATI,
    SEPARATORE_FILE,
    risolvi_tag_ollama,
)
from assistente.console import ConsoleUI
from assistente.data_tempo import contesto_temporale, crea_tool_data
from assistente.direct_pipeline import PREFISSO_SISTEMA_DIRECT, DirectAnswerPipeline
from assistente.evaluate_turn import evaluate_turn
from assistente.file_service import WorkspaceFileService
from assistente.host import HostCapabilities, can_expand_context
from assistente.inference_profiles import (
    InferenceRoute,
    contesto_espanso,
    profile_for_route,
)
from assistente.memoria import (
    REGOLA_QA_SOLO_CHAT,
    MemoriaConversazione,
    _riga_esito,
    blocco_storico_prompt,
)
from assistente.metriche import AccumulatoreMetricheLLM, StatisticheTurnoLLM
from assistente.ollama_runtime import OllamaRuntime, StreamingOllamaLLM
from assistente.politica_energia import (
    alias_piu_leggero,
    prepara_turno_energia,
)
from assistente.comprensione import (
    PATH_FILE,
    ComprensioneTurno,
    estrai_cartella_nominata,
    formatta_blocco_comprensione,
    normalizza,
    risolvi_destinazione,
    scheda_locale,
)
from assistente.politica_turno import (
    ANTI_GOLD_PLATING,
    DecisioneTurno,
    _e_chat_locale,
    capabilities_react,
    decisione_da_schema,
    decisione_react_aperta,
    decisione_saluto,
    e_riversamento_storico,
    formatta_contesto_intent,
    path_documento_nominato,
)
from assistente.schema_turno import (
    NUM_PREDICT_SCHEMA,
    PREFISSO_SCHEMA,
    PacchettoFollowup,
    SchemaTurno,
    formatta_prompt_schema,
    ripara_json_schema,
    schema_json_ollama,
)
from assistente.validatore_turno import (
    MESSAGGIO_FILE_MANCANTE,
    ContestoValidazione,
    EsitoValidazione,
    valida_turno,
)
from assistente.project_memory import format_project_memory_block
from assistente.prefisso_kv import ContinuazionePrefisso
from assistente.prompt_budget import (
    PromptBudget,
    estimate_tokens,
    limita_passi_scratchpad,
)
from assistente.react_parser import AssistenteReActParser
from assistente.registro_processo import (
    RegistroEventi,
    classifica_esito_turno,
)
from assistente.lettori import ISTRUZIONE_FORMATO_NON_SUPPORTATO
from assistente.riassunto import crea_tool_riassunto
from assistente.tools import AVVISO_LETTURA_RIPETUTA
from assistente.thermal import (
    ContestoTurnoTermico,
    StatoTermico,
    pausa_adattiva_tra_iterazioni,
    prepara_turno,
)
from assistente.tools import (
    applica_fallback,
    crea_tool_file,
)

_MARCATORI_LIMITE = ("iteration limit", "time limit")
_MESSAGGIO_ERRORE_FORMATO = (
    "Formato non valido. Esempio corretto senza tool:\n"
    "Thought: ho la risposta\n"
    "Final Answer: testo per l'utente\n"
    "Esempio con tool:\n"
    "Thought: elenco la cartella richiesta\n"
    "Action: elenca_cartella\n"
    "Action Input: assistente\n"
    "Non usare Action: Nessuna/None. Action deve essere un tool reale."
)


def _path_da_input_tool(tool_input: object) -> str:
    """Prima riga o path esplicito, per il placeholder di un'observation omessa."""
    if isinstance(tool_input, dict):
        for key in ("path", "file_path", "sorgente"):
            if key in tool_input:
                return str(tool_input[key])[:180]
        return ""
    text = str(tool_input or "").strip()
    if not text:
        return ""
    return text.splitlines()[0][:180]


def _limita_passi_react(
    steps: list[tuple[AgentAction, str]],
    max_tokens: int,
) -> list[tuple[AgentAction, str]]:
    """Applica microcompact e il tetto token prima della chiamata LLM successiva."""
    packed = [
        (
            str(action.tool),
            _path_da_input_tool(action.tool_input),
            str(action.log),
            str(observation),
        )
        for action, observation in steps
    ]
    limited = limita_passi_scratchpad(packed, max_tokens)
    trimmed: list[tuple[AgentAction, str]] = []
    for (action, observation), (log, content) in zip(steps, limited, strict=True):
        if log == str(action.log) and content == str(observation):
            trimmed.append((action, observation))
            continue
        trimmed.append((AgentAction(action.tool, action.tool_input, log), content))
    return trimmed


# Tool il cui risultato riuscito è già una risposta utile per l'utente.
_TOOL_RISPOSTA_DIRETTA = frozenset(
    # cerca_testo no: i suoi match per radice sono spesso rumore, non una risposta.
    {"riassumi_documento", "elenca_cartella", "elenca_albero", "trova_file"}
)
_ESTRATTO_DOCUMENTO_CARATTERI = 1500
NOTA_RIPIEGO_LIMITE = (
    "(Limite di passi raggiunto: risposta ricavata dall'ultimo risultato utile.)"
)


def _risposta_da_passi(steps: list[tuple[AgentAction, str]]) -> str | None:
    """
    Al limite iterazioni, ricava la risposta dall'ultimo passo utile invece di
    mostrare «Agent stopped…»: spesso il dato era già stato letto.

    Returns:
        Testo per l'utente, oppure None se nessun passo ha prodotto qualcosa.
    """
    for action, observation in reversed(steps):
        tool = str(action.tool)
        testo = str(observation).removeprefix(AVVISO_LETTURA_RIPETUTA).strip()
        if not testo:
            continue
        errore = testo.startswith("Errore")
        # Un formato non supportato è la risposta giusta, non un fallimento da saltare.
        if tool == "leggi_documento" and errore and "non supportato" in testo:
            return testo.removeprefix("Errore:").replace(
                ISTRUZIONE_FORMATO_NON_SUPPORTATO, ""
            ).strip()
        if errore:
            continue
        if tool in _TOOL_RISPOSTA_DIRETTA and not testo.startswith("Nessun"):
            return testo
        if tool == "leggi_documento":
            path = str(action.tool_input).strip()
            estratto = testo[:_ESTRATTO_DOCUMENTO_CARATTERI].rstrip()
            coda = "…" if len(testo) > _ESTRATTO_DOCUMENTO_CARATTERI else ""
            return f"Ho letto {path}. Inizio del documento:\n\n{estratto}{coda}"
    return None


def _osservazione_errore_parsing(errore: OutputParserException) -> str:
    """Observation di retry: payload scrittura troncato vs formato ReAct generico."""
    osservazione = str(getattr(errore, "observation", "") or "")
    if osservazione:
        return osservazione
    if "thought incompleto" in str(errore).casefold():
        return "Thought incompleto: emetti Action con un tool reale oppure Final Answer breve."
    return _MESSAGGIO_ERRORE_FORMATO


_MESSAGGIO_STORICO_VUOTO = "Nessun turno precedente da salvare."
_PAROLE_VERIFICA = re.compile(
    r"\b(?:verifica|verificalo|controlla|controllalo|ricontrolla|ricontrollalo)\b",
    re.IGNORECASE,
)
_STORICO_MIN_TOKEN = 128
_STORICO_MAX_TOKEN_LEGGERO = 1024
_STORICO_QUOTA_LEGGERA = 5
_STORICO_QUOTA_PESANTE = 3
# Allineato al refresh barra durante Live: evita I/O termico su ogni token.
_INTERVALLO_METRICHE_LIVE_SEC = 0.25
# Closed-corpus: l'unica fonte sono i documenti della cartella di lancio.
REGOLA_CORPUS = (
    "Unica fonte: i documenti della cartella. Non usare conoscenze tue né Internet. "
    "Se l'utente nomina un file: leggi_documento su quel file (trova_file se il "
    "path non è esatto); se il tool dà errore, riporta l'errore. "
    "Per una domanda: cerca_testo con le parole chiave, poi leggi_documento sui file "
    "trovati. Se non trova nulla, al massimo un secondo cerca_testo con sinonimi "
    "e le sole immagini pertinenti; poi fermati. Cita il file (e la pagina) "
    "di ogni informazione. Se i documenti non contengono la risposta, dillo "
    "chiaramente: non è presente nei documenti."
)
_IDENTITA_AGENTE = (
    "Parli in italiano informale, dando del tu. Lavori solo sui documenti della "
    "cartella corrente (txt, md, pdf, docx, pages, immagini) e scrivi solo file .txt. "
    f"{REGOLA_CORPUS} Non inventare fatti, path o operazioni su disco. "
    f"Usa i tool che servono e fermati appena il risultato è verificato. {ANTI_GOLD_PLATING}"
)
_TEMPLATE_REACT = """{identita}
Tool disponibili:
{tools}

Protocollo:
Thought: piano essenziale
Action: uno tra [{tool_names}]
Action Input: input del tool
Observation: risultato
Thought: completato
Final Answer: risposta

Se non serve un tool, dopo Thought scrivi subito Final Answer.
Non usare Action: Nessuna/None. Mai Action: Final Answer.

Question: {input}
Thought:{agent_scratchpad}"""

CapabilitySet = tuple[str, ...]
EXECUTOR_CACHE_LIMIT = 4


@dataclass(frozen=True, slots=True)
class ExecutorKey:
    """Identifica un AgentExecutor senza confonderlo con i pesi Ollama."""

    alias: str
    max_iter: int
    num_predict: int
    context_tokens: int
    capabilities: CapabilitySet
    scrittura_consentita: bool
    reasoning: bool


class TurnoAnnullato(RuntimeError):
    """Segnala una cancellazione richiesta dall'utente durante lo stream."""

def _costruisci_descrizione_task(
    richiesta: str,
    verifica_completa: bool,
    decisione: DecisioneTurno,
    storico: str = "",
    *,
    project_memory: str = "",
    comprensione: ComprensioneTurno | None = None,
    nota: str = "",
) -> str:
    """
    Costruisce istruzioni ReAct compatte usando la decisione già presa dal routing.

    La regola del corpus sta nell'identità, nel prefisso stabile: qui non si ripete.

    Returns:
        Richiesta arricchita con workflow, gate file e storico necessari.
    """
    write_allowed = decisione.scrittura_consentita
    instructions: list[str] = []
    if project_memory:
        instructions.append(project_memory.strip())
    instructions.append("Workflow: minimo di Action, poi Final Answer.")
    if storico:
        instructions.append(
            "Lo STORICO serve al filo del discorso; i fatti vanno verificati nei documenti."
        )
    cartella = estrai_cartella_nominata(richiesta)
    if write_allowed:
        destinazione = (
            _destinazione_scrittura(comprensione.paths)
            if comprensione is not None
            else ""
        )
        if destinazione:
            instructions.append(
                f"Destinazione già nota: {destinazione}. "
                "Usa scrivi_file su quel path, senza chiedere il nome."
            )
            if nota:
                instructions.append(f"{nota} Non creare cartelle per questo.")
        elif cartella is not None:
            instructions.append(
                f"Cartella già nominata: {cartella}. "
                "Usa crea_cartella se manca, poi scrivi_file su un .txt. "
                "Niente elenca_cartella obbligatorio. Action Input senza slash iniziale."
            )
        else:
            instructions.append(
                "Path incerto: trova_file o elenca_cartella, poi leggi_documento."
            )
        instructions.append(
            "crea_cartella solo se l'utente chiede una cartella. File nuovo: scrivi_file "
            f"(path relativo .txt SENZA slash, riga {SEPARATORE_FILE}, body). "
            "Si scrivono solo file .txt: mai sovrascrivere i documenti sorgente. "
            "Patch su un .txt già letto: modifica_file. Un path per Action. "
            "Niente virgole nel path. Non dichiarare successo senza Observation positiva."
        )
    else:
        if cartella is not None:
            instructions.append(
                f"elenca_cartella Action Input obbligatorio: {cartella}. "
                "Mai '.' o la root del workspace. Per più livelli: elenca_albero."
            )
        else:
            instructions.append(
                "Contenuto: cerca_testo. Nome incerto: trova_file. "
                "Cartella attuale: elenca_cartella con Action Input . "
                "Panoramica: elenca_albero."
            )
        instructions.append(REGOLA_QA_SOLO_CHAT)

    # Il tool dedicato evita di copiare l'intero documento nello scratchpad.
    instructions.append("Per un documento usa riassumi_documento una sola volta.")
    # La modalità check rilegge il file per verificare il contenuto realmente persistito.
    if verifica_completa and write_allowed:
        instructions.append(
            "Dopo ogni scrittura rileggi lo stesso path e cita soltanto file verificati. "
            "La verifica inline è obbligatoria prima di Final Answer."
        )
    instructions.append(
        "Final Answer: esito breve. "
        "Per elenca_cartella o elenca_albero: elenca i nomi osservati, "
        "senza riassumere e senza ricopiare metadati o dimensioni."
    )
    instructions.append(
        "Thought: esplicita i passaggi essenziali prima di ogni Action."
    )
    compito_block = (
        formatta_blocco_comprensione(comprensione) if comprensione is not None else ""
    )
    return "\n".join(
        part
        for part in (
            *instructions,
            blocco_storico_prompt(storico).rstrip(),
            formatta_contesto_intent(decisione),
            compito_block,
            f"RICHIESTA:\n{richiesta.strip()}",
            (
                "Action Input di scrivi_file: path relativo .txt, "
                f"riga {SEPARATORE_FILE}, body grezzo. "
                "Nel body niente fence markdown né Thought/Action."
                if write_allowed
                else ""
            ),
        )
        if part
    ).strip()


def _richiede_verifica_completa(richiesta: str, verifica_completa_attiva: bool) -> bool:
    return verifica_completa_attiva or bool(_PAROLE_VERIFICA.search(richiesta))


# Il ReAct aperto non accorcia il budget in base al lessico della frase.
ITERAZIONI_REACT_APERTO = 7


def _estrai_unico_path_richiesto(
    richiesta: str,
    service: WorkspaceFileService,
) -> str | None:
    """
    Estrae e valida l'unica destinazione documentale nominata nella richiesta.

    Args:
        richiesta: Testo originale che autorizza esplicitamente la scrittura.
        service: Boundary workspace usata per normalizzare ogni candidato.

    Returns:
        Path relativo quando esiste una sola destinazione valida, altrimenti None.
    """
    normalized_paths: list[str] = []
    # Valida tutti i candidati per non scegliere silenziosamente un path ambiguo.
    for raw_path in PATH_FILE.findall(richiesta):
        normalized_path = service.normalize_file_path(raw_path)
        # I path non consentiti non devono raggiungere la scrittura diretta.
        if normalized_path is not None:
            normalized_paths.append(normalized_path)
    unique_paths = list(dict.fromkeys(normalized_paths))
    # Il percorso deterministico gestisce un solo documento per richiesta.
    if len(unique_paths) != 1:
        return None
    return unique_paths[0]


def _destinazione_scrittura(paths: tuple[str, ...]) -> str:
    """Primo path scrivibile (.txt): gli altri sono documenti sorgente da leggere."""
    for path in paths:
        if Path(path).suffix.casefold() in ESTENSIONI_SCRIVIBILI:
            return path
    return ""


def _adatta_destinazioni_txt(
    schema: SchemaTurno | None,
    esiste: Callable[[str], bool],
) -> tuple[SchemaTurno | None, str]:
    """
    Converte in .txt le destinazioni nuove con un'altra estensione.

    Un path che esiste già è un documento da leggere e resta com'è; uno che
    non esiste, in un turno di scrittura, è la destinazione richiesta.

    Returns:
        Schema adattato e nota per l'utente (vuota se nulla è cambiato).
    """
    if schema is None or not (schema.write or schema.deliver == "file"):
        return schema, ""
    adattati: list[str] = []
    cambi: list[tuple[str, str]] = []
    for path in schema.paths:
        suffix = Path(path).suffix.casefold()
        if suffix and suffix not in ESTENSIONI_SCRIVIBILI and not esiste(path):
            base = Path(path).with_suffix(".txt")
            nuovo = base.as_posix()
            indice = 2
            # Il .txt omonimo può essere un documento sorgente: mai sovrascriverlo.
            while esiste(nuovo):
                nuovo = base.with_name(f"{base.stem}_{indice}.txt").as_posix()
                indice += 1
            cambi.append((path, nuovo))
            adattati.append(nuovo)
        else:
            adattati.append(path)
    if not cambi:
        return schema, ""
    elenco = ", ".join(f"{vecchio} → {nuovo}" for vecchio, nuovo in cambi)
    nota = f"Nota: scrivo solo file .txt, quindi uso {elenco}."
    subject = schema.subject
    for vecchio, nuovo in cambi:
        subject = subject.replace(vecchio, nuovo)
    return replace(schema, paths=tuple(adattati), subject=subject), nota


def _comprensione_dallo_schema(
    schema: SchemaTurno | None,
    richiesta: str = "",
) -> ComprensioneTurno:
    """
    Tema e path dello schema, senza compilare un COMPITO dal lessico.

    Un subject che ripete l'inizio della richiesta (scheda locale) non diventa
    COMPITO: duplicherebbe la riga RICHIESTA del prompt.
    """
    if schema is None:
        return ComprensioneTurno("", ())
    compito = schema.subject.strip()
    if compito and normalizza(richiesta).casefold().startswith(compito.casefold()):
        compito = ""
    return ComprensioneTurno(compito=compito, paths=schema.paths)


class CallbackUI(BaseCallbackHandler):
    """Callback unico per streaming, timeline e metriche dei due percorsi agente."""

    def __init__(
        self,
        ui: ConsoleUI,
        pausa_tra_iterazioni_sec: float = 0.0,
        *,
        stato_iniziale: StatoTermico = StatoTermico.NOMINALE,
        burn_protection: bool = False,
        contesto_massimo: int = CONTESTO_LLM_TOKEN,
        cancel_event: threading.Event | None = None,
        registro: RegistroEventi | None = None,
    ) -> None:
        """
        Inizializza stato per-turno e accumulatore del profilo LLM effettivo.

        Args:
            ui: Console destinataria di token, timeline e statistiche.
            pausa_tra_iterazioni_sec: Backoff termico di base tra generazioni ReAct.
            stato_iniziale: Snapshot termico usato per adattare le pause.
            burn_protection: Abilita il ricalcolo dinamico del backoff.
            contesto_massimo: Finestra token del modello usato nel percorso corrente.
            cancel_event: Segnale condiviso per interrompere il turno.
            registro: Osservabilità di crash/errori; None usa un registro in memoria.
        """
        super().__init__()
        self._ui = ui
        self._pausa_sec = pausa_tra_iterazioni_sec
        self._stato_iniziale = stato_iniziale
        self._burn_protection = burn_protection
        self._cancel_event = cancel_event
        self._registro = registro if registro is not None else RegistroEventi()
        self._metriche = AccumulatoreMetricheLLM(contesto_massimo)
        self._prossimo_ridisegno_metriche = 0.0
        self._pensiero_aperto = False
        self._tool_corrente: str | None = None
        self._ui.imposta_sorgente_metriche(self._metriche.snapshot)

    def _alza_se_annullato(self) -> None:
        """Interrompe il turno se l'utente ha richiesto la cancellazione."""
        if self._cancel_event is not None and self._cancel_event.is_set():
            raise TurnoAnnullato("Turno annullato dall'utente.")

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        """Applica la pausa termica e apre il timer TTFT della chiamata."""
        del serialized, kwargs
        self._alza_se_annullato()
        pausa = self._pausa_sec
        # Con burn protection la pausa riflette lo stato termico aggiornato.
        if self._burn_protection:
            pausa = pausa_adattiva_tra_iterazioni(self._stato_iniziale, self._pausa_sec)
        # Il timer LLM parte dopo il throttling per misurare il provider, non l'attesa termica.
        if pausa > 0:
            wait_started = time.perf_counter()
            time.sleep(pausa)
            self._metriche.registra_attesa_termica(
                time.perf_counter() - wait_started
            )
        self._metriche.inizia_chiamata()
        self._registro.heartbeat()
        if prompts:
            self._metriche.stima_prompt(prompts[0])
        self._pubblica_metriche_subito()

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        """Instrada CoT nativo sulla timeline e il testo utile sul panel."""
        self._alza_se_annullato()
        self._registro.heartbeat()
        if kwargs.get("thinking"):
            if not token:
                return
            self._pensiero_aperto = True
            self._emetti_token_pensiero(token)
            return
        if not token:
            return
        self._chiudi_pensiero_aperto()
        self._emetti_token_contenuto(token)

    def _prepara_emissione_token(self, token: str) -> float:
        """Registra TTFT/conteggio e restituisce l'istante di inizio render."""
        self._metriche.registra_primo_token()
        self._metriche.registra_token_stream(token)
        self._aggiorna_metriche_live()
        return time.perf_counter()

    def _emetti_token_pensiero(self, token: str) -> None:
        """Scrive un frammento di CoT nativo e ne misura il rendering."""
        started = self._prepara_emissione_token(token)
        self._ui.scrivi_pensiero(token)
        self._metriche.registra_rendering(time.perf_counter() - started)

    def _emetti_token_contenuto(self, token: str) -> None:
        """Scrive un frammento di risposta e ne misura il rendering."""
        started = self._prepara_emissione_token(token)
        self._ui.scrivi_token(token)
        self._metriche.registra_rendering(time.perf_counter() - started)

    def _chiudi_pensiero_aperto(self) -> None:
        """Chiude lo stream Pensiero se ancora visibile."""
        if not self._pensiero_aperto:
            return
        self._ui.chiudi_pensiero()
        self._pensiero_aperto = False

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """Chiude una generazione riuscita e pubblica lo snapshot aggiornato."""
        del kwargs
        self._registro.heartbeat()
        self._chiudi_pensiero_aperto()
        self._ui.reset_chiamata_llm()
        self._ui.imposta_metriche_turno(self._metriche.registra(response))

    def on_llm_error(self, error: BaseException, **kwargs: Any) -> None:
        """Chiude timer e stream quando il provider LLM solleva un errore."""
        del kwargs
        if not isinstance(error, (KeyboardInterrupt, TurnoAnnullato)):
            self._registro.registra_errore_llm(error)
        self._chiudi_pensiero_aperto()
        self._ui.reset_chiamata_llm()
        self._ui.imposta_metriche_turno(self._metriche.registra_errore_llm())

    def on_agent_action(self, action: AgentAction, **kwargs: Any) -> None:
        """Registra l'azione ReAct prima dell'invocazione del tool."""
        del kwargs
        # Le pseudo-azioni parser non rappresentano un tool utile per la timeline.
        if action.tool != "_Exception":
            self._tool_corrente = action.tool
            self._ui.registra_azione_tool(action.tool, str(action.tool_input))

    def on_tool_start(self, serialized: dict[str, Any], input_str: str, **kwargs: Any) -> None:
        """Avvia timer e timeline di un tool lanciato da AgentExecutor."""
        del input_str, kwargs
        nome = str(serialized.get("name", "tool"))
        self._tool_corrente = nome
        self._metriche.inizia_tool()
        self._registro.heartbeat()
        self._ui.inizia_esecuzione_tool(nome)

    def on_tool_end(self, output: str, **kwargs: Any) -> None:
        """Termina un tool ReAct e aggiorna timeline e metriche."""
        del kwargs
        osservazione = str(output)
        self._tool_corrente = None
        self._registro.heartbeat()
        self._ui.termina_esecuzione_tool()
        self._ui.emetti_risultato_tool(osservazione)
        self._ui.imposta_metriche_turno(self._metriche.termina_tool())

    def on_tool_error(self, error: BaseException, **kwargs: Any) -> None:
        """Termina timer e timeline quando un tool ReAct solleva un errore."""
        del kwargs
        nome = self._tool_corrente or "tool"
        self._registro.registra_errore_tool(nome, error)
        self._tool_corrente = None
        self._ui.termina_esecuzione_tool()
        self._ui.emetti_errore_tool(str(error))
        self._ui.imposta_metriche_turno(self._metriche.termina_tool())

    def registra_diagnostica(
        self,
        messaggio: str,
        tipo_eccezione: str | None = None,
    ) -> None:
        """Propaga una diagnostica pipeline (testo utente invariato) al registro."""
        self._registro.registra_diagnostica(messaggio, tipo_eccezione=tipo_eccezione)

    def ultime_metriche(self) -> StatisticheTurnoLLM | None:
        """Snapshot LLM del turno, oppure None se non c'è ancora una chiamata."""
        return self._metriche.snapshot()

    def _pubblica_metriche_subito(self) -> None:
        """Forza un ridisegno della barra con lo snapshot corrente."""
        self._prossimo_ridisegno_metriche = (
            time.monotonic() + _INTERVALLO_METRICHE_LIVE_SEC
        )
        self._ui.imposta_metriche_turno(self._metriche.snapshot())

    def _aggiorna_metriche_live(self) -> None:
        """
        Aggiorna i numeri della barra senza saturare I/O termico su ogni token.

        Con Live attivo il ridisegno segue già il refresh del panel; senza Live
        (Thought ReAct) si ridisegna a 4 Hz.
        """
        snapshot = self._metriche.snapshot()
        if self._ui.ha_stream_live:
            self._ui.aggiorna_metriche_turno(snapshot)
            return
        ora = time.monotonic()
        if ora < self._prossimo_ridisegno_metriche:
            self._ui.aggiorna_metriche_turno(snapshot)
            return
        self._prossimo_ridisegno_metriche = ora + _INTERVALLO_METRICHE_LIVE_SEC
        self._ui.imposta_metriche_turno(snapshot)


def _arricchisci_risposta(risultato: str, path_scritti: list[str]) -> str:
    """
    Allinea la Final Answer ai file realmente persistiti nel turno.

    Args:
        path_scritti: Solo file, mai cartelle: una directory creata non è un
            «file scritto» e non va elencata al posto del contenuto mancante.

    Returns:
        Risposta originale, oppure arricchita/corretta se i path citati non combaciano.
    """
    cited_paths = list(dict.fromkeys(PATH_FILE.findall(risultato)))
    written_set = set(path_scritti)
    missing_cited = [path for path in cited_paths if path not in written_set]
    # La FA ha nominato file che il tool non ha materializzato: correggi il claim.
    if missing_cited:
        scritti = ", ".join(path_scritti) if path_scritti else "(nessuno)"
        mancanti = ", ".join(missing_cited)
        return (
            f"{risultato.rstrip()}\n\n"
            f"Attenzione: path citati ma NON presenti su disco: {mancanti}. "
            f"File effettivamente scritti in questo turno: {scritti}."
        )
    if not path_scritti:
        return risultato
    elenco = ", ".join(path_scritti)
    basso = risultato.lower()
    if any(marcatore in basso for marcatore in _MARCATORI_LIMITE):
        return f"Operazione interrotta per limite iterazioni. File scritti su disco: {elenco}"
    if set(cited_paths) != written_set:
        return f"{risultato.rstrip()}\n\nFile verificati su disco: {elenco}"
    return risultato

class SessioneAgente:
    """Orchestratore di sessione: policy, memoria, documenti della cartella e ReAct."""

    def __init__(
        self,
        ui: ConsoleUI,
        file_service: WorkspaceFileService | None = None,
        llm_factory: Callable[..., OllamaLLM] | None = None,
        ollama_runtime: OllamaRuntime | None = None,
        host: HostCapabilities | None = None,
        registro: RegistroEventi | None = None,
    ) -> None:
        """
        Costruisce una sessione con dipendenze riusabili e isolate.

        Args:
            ui: Interfaccia console della sessione.
            file_service: Boundary disco isolata; None usa la cartella corrente.
            llm_factory: Costruttore Ollama sostituibile senza patch globali nei test.
            ollama_runtime: Runtime centralizzato opzionale, utile nei test integrazione.
            host: Snapshot macchina; None non amplia i budget (test e default 16 GB).
            registro: Osservabilità persistente; None usa un registro in memoria.
        """
        self._ui = ui
        self._host = host
        self._registro = registro if registro is not None else RegistroEventi()
        self._llm_factory = llm_factory or StreamingOllamaLLM
        self._runtime = ollama_runtime or OllamaRuntime(
            self._llm_factory,
            # Una factory mockata non deve aprire il canale di controllo Ollama.
            manage_residency=llm_factory is None,
            host=host,
        )
        self._executor_per_chiave: OrderedDict[ExecutorKey, AgentExecutor] = (
            OrderedDict()
        )
        self._file_service = file_service or WorkspaceFileService()
        # Le immagini passano dalla vision del modello del turno corrente.
        self._alias_turno = ALIAS_MODELLO_DEFAULT
        # Avviso deterministico del turno (es. destinazione convertita in .txt).
        self._nota_turno = ""
        self._file_service.imposta_descrittore_immagine(self._descrivi_immagine)
        self._cancel_event = threading.Event()
        self._active_callback: CallbackUI | None = None
        self._ultimo_callback: CallbackUI | None = None
        provider_iniziale = self._bind_llm_provider(True)
        self._direct_pipeline = DirectAnswerPipeline(
            provider_iniziale,
            self._ui,
        )
        # Lo storico è locale alla sessione e non persiste su disco.
        self._memoria = MemoriaConversazione()
        self._continuazione = ContinuazionePrefisso()
        # Memoria progetto (AGENTS.md) caricata una volta all'avvio sessione.
        self._project_memory = format_project_memory_block(self._file_service.workspace)
        self._path_fallback_turno: list[str] = []
        self._ultimo_documento: str | None = None
        self._ultimo_schema: SchemaTurno | None = None

    @property
    def registro(self) -> RegistroEventi:
        """Registro di processo della sessione, iniettato dalla CLI o in memoria."""
        return self._registro

    def svuota_memoria(self) -> None:
        """Azzera lo storico conversazione (comando /clear)."""
        self._memoria.svuota()
        self._continuazione.invalida()
        self._ultimo_documento = None
        self._ultimo_schema = None
        # Invalida anche la cache file: /clear riparte da contesto fresco.
        self._file_service.invalidate_read_cache()

    def ha_storico(self) -> bool:
        """True se la sessione ha almeno un turno memorizzato."""
        return self._memoria.ha_turni()

    def registra_turno_storico(self, utente: str, assistente: str) -> None:
        """Inserisce uno scambio pregresso nello storico (benchmark e test di follow-up)."""
        self._memoria.registra(utente, assistente)

    @property
    def path_fallback_turno(self) -> list[str]:
        """Path creati dal fallback create-only, non dai tool ReAct."""
        return list(self._path_fallback_turno)

    @property
    def workspace(self) -> Path:
        """Cartella documenti confinata usata dai tool file."""
        return self._file_service.workspace

    def _esiste_nel_corpus(self, path: str) -> bool:
        """True se il path esiste già nella cartella documenti."""
        resolved = self._file_service.resolve_file(path)
        return resolved is not None and resolved.absolute.exists()

    def _descrivi_immagine(self, immagine: bytes, nome: str) -> str:
        """Vision sul modello del turno: nessun secondo modello caricato."""
        self._controlla_annullamento()
        self._registro.registra_diagnostica(f"vision: {nome}")
        return self._runtime.descrivi_immagine(self._alias_turno, immagine, nome)

    def scarica_pesi_attivi(self) -> None:
        """Scarica il modello residente per misurare un cold start."""
        self._runtime.unload_active()

    def close(self) -> None:
        """Rilascia connessione e modello Ollama posseduti dalla sessione."""
        self._runtime.close()

    def annulla_turno(self) -> None:
        """Richiede la cancellazione cooperativa della generazione corrente."""
        self._cancel_event.set()

    def _bind_llm_provider(
        self,
        reasoning: bool,
    ) -> Callable[[str, int, int], OllamaLLM]:
        """Chiude `reasoning` del turno senza stato mutabile sulla sessione."""

        def provider(alias: str, num_ctx: int, num_predict: int) -> OllamaLLM:
            llm = self._runtime.get_llm(
                alias,
                num_ctx,
                num_predict,
                reasoning=reasoning,
            )
            if isinstance(llm, StreamingOllamaLLM):
                llm.collega_continuazione(self._continuazione)
            return llm

        return provider

    def nomi_modelli_installati(self) -> frozenset[str]:
        """Catalogo Ollama normalizzato visibile da `/model`."""
        return self._runtime.list_available_model_names()

    def scarica_modello(self, tag: str) -> None:
        """Pull dell'artefatto con progresso in console, poi invalida il catalogo."""
        self._runtime.pull_model(tag, self._ui.aggiorna_progresso)

    def preriscalda_modello(self, alias_modello: str) -> None:
        """
        Precarica i pesi e stampa subito gli esiti in console.

        Args:
            alias_modello: Chiave presente in MODELLI_DISPONIBILI.

        Returns:
            None. Usato da `/model` e dal wizard, quando stdout è libero.
        """
        self._runtime.preload(alias_modello)
        self.annuncia_preriscaldamento(alias_modello)

    def annuncia_preriscaldamento(self, alias_modello: str) -> None:
        """
        Stampa gli esiti già raccolti da un preload, senza ripetere la richiesta HTTP.

        Args:
            alias_modello: Alias usato nel preload, per il tag di fallback nel messaggio.

        Returns:
            None. Non richiama `preload`: un secondo POST scaricherebbe di nuovo i pesi.
        """
        fallback = self._runtime.last_tag_fallback
        # Mapping tag profilo → alias libreria installato (es. gemma4).
        if fallback is not None:
            self._ui.stampa_messaggio_sistema(fallback)
        warning = self._runtime.last_memory_warning
        # Profilo esplicito consentito, ma lo swap va reso visibile.
        if warning is not None:
            self._ui.stampa_messaggio_sistema(f"Avviso memoria: {warning}")
        cpu_fallback = self._runtime.last_cpu_fallback
        # GPU attesa ma Ollama ha caricato sulla CPU: non blocca il turno.
        if cpu_fallback is not None:
            self._ui.stampa_messaggio_sistema(cpu_fallback)
        server_env = self._runtime.last_server_env_warning
        if server_env is not None:
            self._ui.stampa_messaggio_sistema(server_env)
        control_error = self._runtime.last_control_error
        # Ollama assente non impedisce l'avvio CLI: si avvisa prima del turno.
        if control_error is not None:
            preferred = risolvi_tag_ollama(alias_modello)
            active = self._runtime.active_tag or preferred
            self._ui.stampa_messaggio_sistema(
                "Preriscaldamento non riuscito: verificare Ollama "
                f"({control_error}). Se il modello manca: "
                f"ollama pull {active}"
            )

    def _scrivi_su_file(self, richiesta: str, risposta: str) -> str:
        """
        Persiste un testo nella singola destinazione .txt richiesta.

        Args:
            richiesta: Prompt originale dal quale ricavare il path autorizzato.
            risposta: Testo da salvare (lo storico della conversazione).

        Returns:
            Testo con conferma del path, oppure diagnostica senza falsa scrittura.
        """
        extracted = _estrai_unico_path_richiesto(richiesta, self._file_service)
        cartella = estrai_cartella_nominata(richiesta)
        destination = risolvi_destinazione(
            extracted,
            cartella,
            self._ultimo_documento,
        )
        if destination is not None:
            destination = (
                self._file_service.normalize_file_path(destination) or destination
            )
        # Più destinazioni o path mancanti richiedono disambiguazione esplicita.
        if destination is None:
            return (
                f"{risposta}\n\n"
                "Errore scrittura: indica un unico path file relativo e valido."
            )
        if cartella:
            mkdir_result = self._file_service.create_directory(cartella)
            if not mkdir_result.succeeded:
                return f"{risposta}\n\nErrore scrittura: {mkdir_result.error}"
        content = risposta.rstrip() + "\n"
        write_result = self._file_service.write_text(destination, content)
        # Il risultato tipizzato impedisce di dichiarare successo dopo un errore del boundary.
        if not write_result.succeeded:
            return f"{risposta}\n\nErrore scrittura: {write_result.error}"
        self._ultimo_documento = write_result.path
        return f"{risposta}\n\nFile scritto e verificato: {write_result.path}"

    def _riversa_storico(self, richiesta: str) -> tuple[str, list[str]]:
        """
        Scrive i turni già in memoria sul documento nominato, senza cercare.

        Senza un path chiede il nome del file. Il testo è quello memorizzato
        per intero: il formatter del prompt taglia le risposte lunghe.
        """
        self._registro.registra_diagnostica("riversamento storico senza ricerca")
        if path_documento_nominato(richiesta) is None:
            return self._finalizza_turno(
                richiesta,
                MESSAGGIO_FILE_MANCANTE,
                scrittura_richiesta=False,
            ), []
        self._file_service.begin_turn(True)
        risultato = self._scrivi_su_file(richiesta, self._testo_turni_memorizzati())
        scritti = self._file_service.written_paths
        return self._finalizza_turno(
            richiesta,
            risultato,
            scrittura_richiesta=True,
            path_scritti=scritti,
        ), scritti

    def _testo_turni_memorizzati(self) -> str:
        """Concatena gli scambi già registrati, senza il taglio del prompt."""
        blocchi = [
            (
                f"--- Turno {turno.indice} ---\n"
                f"Utente: {turno.utente}\n"
                f"Assistente: {turno.assistente}"
            )
            for turno in self._memoria.turni
        ]
        if not blocchi:
            return _MESSAGGIO_STORICO_VUOTO
        return "\n\n".join(blocchi)

    def esegui_richiesta(self, richiesta: str, config: ConfigRuntime) -> tuple[str, list[str]]:
        """
        Esegue un turno completo applicando una sola policy e servizi isolati.

        Returns:
            Risposta testuale e path materializzati nel turno (tool e/o fallback).
        """
        self._cancel_event.clear()
        self._path_fallback_turno = []
        verifica = _richiede_verifica_completa(richiesta, config.verifica_completa)
        energia = prepara_turno_energia(config, self._ui)
        contesto = prepara_turno(config, self._ui) if config.burn_protection else None
        alias = energia.alias_modello
        if contesto is not None:
            alias = alias_piu_leggero(alias, contesto.alias_modello)
            self._registra_throttle_se_taglia(contesto)
        self._alias_turno = alias
        thinking = config.thinking and (contesto.thinking if contesto else True)
        provider = self._bind_llm_provider(thinking)
        self._direct_pipeline._llm_provider = provider
        if e_riversamento_storico(richiesta):
            # Riversare la chat non richiede né scheda né ricerca.
            self._ui.prepara_nuovo_turno_stream()
            self._registro.inizia_turno("direct", alias, richiesta)
            return self._riversa_storico(richiesta)
        origine, validazione, decisione = self._classifica_turno(
            richiesta,
            alias,
            contesto,
        )
        if validazione is not None and not validazione.schema.clarify:
            self._ultimo_schema = validazione.schema
        schema_corrente = validazione.schema if validazione is not None else None
        schema_corrente, self._nota_turno = _adatta_destinazioni_txt(
            schema_corrente,
            self._esiste_nel_corpus,
        )
        if self._nota_turno:
            self._ui.stampa_messaggio_sistema(self._nota_turno)
        comprensione = _comprensione_dallo_schema(schema_corrente, richiesta)
        self._file_service.begin_turn(decisione.scrittura_consentita)
        self._ui.prepara_nuovo_turno_stream()
        self._registro.inizia_turno(decisione.percorso, alias, richiesta)
        self._registro.registra_diagnostica(
            f"schema origine={origine} percorso={decisione.percorso}"
        )
        if validazione is not None and validazione.schema.clarify:
            return self._finalizza_turno(
                richiesta,
                validazione.schema.clarify,
                scrittura_richiesta=False,
            ), []
        if decisione.percorso == "direct":
            return self._esegui_turno_direct(
                richiesta,
                alias,
                thinking,
                config,
                contesto,
                comprensione,
            ), []
        return self._esegui_turno_react(
            richiesta,
            alias,
            thinking,
            verifica,
            decisione,
            config,
            contesto,
            comprensione,
        )

    def _classifica_turno(
        self,
        richiesta: str,
        alias: str,
        contesto: ContestoTurnoTermico | None,
    ) -> tuple[str, EsitoValidazione | None, DecisioneTurno]:
        """
        Saluto e scheda locale senza LLM; solo il testo ambiguo chiede la scheda al modello.

        Termica seria e scheda illeggibile aprono il ReAct completo.
        """
        if _e_chat_locale(richiesta):
            return "schema_skipped_saluto", None, decisione_saluto()
        locale = scheda_locale(richiesta, self._esiste_nel_corpus)
        if locale is not None:
            validazione = self._valida_schema(locale, richiesta)
            return "locale", validazione, decisione_da_schema(validazione.schema)
        if contesto is not None and contesto.stato >= StatoTermico.SERIO:
            self._registro.registra_diagnostica("schema_skipped_thermal")
            return "schema_skipped_thermal", None, decisione_react_aperta()
        self._ui.aggiorna_progresso("Comprendo la richiesta")
        self._controlla_annullamento()
        try:
            validazione, repair, retry = self._schema_da_modello(richiesta, alias)
        finally:
            self._ui.chiudi_progresso()
        if validazione is None:
            self._registro.registra_diagnostica("schema_fallback")
            return "schema_fallback", None, decisione_react_aperta()
        schema = validazione.schema
        self._registro.registra_diagnostica(
            "schema "
            f"kind={schema.kind} deliver={schema.deliver} "
            f"repair={int(repair)} retry={int(retry)}"
        )
        return "schema", validazione, decisione_da_schema(schema)

    def _schema_da_modello(
        self,
        richiesta: str,
        alias: str,
    ) -> tuple[EsitoValidazione | None, bool, bool]:
        """
        Una chiamata e un solo retry, solo per JSON illeggibile.

        Un errore di Ollama non ripete la chiamata: il turno passa subito al ReAct.
        Il prefisso KV è PREFISSO_SCHEMA, sulla stessa finestra dei turni.
        """
        context_tokens = self._runtime.contesto_sessione(alias)
        self._assicura_prefisso_schema(alias, context_tokens)
        llm = self._runtime.get_llm(
            alias,
            context_tokens,
            NUM_PREDICT_SCHEMA,
            reasoning=False,
        )
        if hasattr(llm, "collega_continuazione"):
            llm.collega_continuazione(self._continuazione)
        followup = self._pacchetto_followup()
        iso = contesto_temporale().iso
        formato = schema_json_ollama()
        errore = ""
        for tentativo in range(2):
            testo = self._completa_llm(
                llm,
                formatta_prompt_schema(richiesta, iso, followup, errore=errore),
                format=formato,
                num_predict=NUM_PREDICT_SCHEMA,
                reasoning=False,
                temperature=0.0,
            )
            if testo is None:
                self._registro.registra_diagnostica("schema: errore Ollama, niente retry")
                return None, False, tentativo > 0
            repair = ripara_json_schema(testo)
            if repair.schema is not None:
                riparato = repair.riparato or tentativo > 0
                return self._valida_schema(repair.schema, richiesta), riparato, tentativo > 0
            errore = repair.errore or "JSON non valido."
        return None, True, True

    def _valida_schema(
        self,
        schema: SchemaTurno,
        richiesta: str,
    ) -> EsitoValidazione:
        """Applica deissi e gate di scrittura sullo schema (locale o del modello)."""
        return valida_turno(
            schema,
            richiesta,
            ContestoValidazione(
                ultimo=self._ultimo_schema,
                documento_precedente=self._ultimo_documento,
            ),
        )

    def _pacchetto_followup(self) -> PacchettoFollowup:
        """Ultimo schema e una riga di esito, senza lo storico intero."""
        esito = ""
        if self._memoria.ha_turni():
            esito = _riga_esito(self._memoria.turni[-1].assistente)
        ultimo = self._ultimo_schema
        if ultimo is None:
            return PacchettoFollowup(esito=esito)
        return PacchettoFollowup(
            kind=ultimo.kind,
            subject=ultimo.subject,
            path=ultimo.paths[0] if ultimo.paths else "",
            esito=esito,
        )

    def _assicura_prefisso_schema(self, alias: str, context_tokens: int) -> None:
        """Prefill del solo prefisso schema, così il turno non ripaga gli esempi."""
        chiave = (alias, context_tokens, False, PREFISSO_SCHEMA)
        if self._continuazione.ha_prefisso(chiave):
            return
        self._continuazione.imposta_chiave(chiave, PREFISSO_SCHEMA)
        ids = self._runtime.context_prefisso_precaricato(PREFISSO_SCHEMA, context_tokens)
        if ids is None:
            ids = self._runtime.prefill_prefisso_raw(alias, context_tokens, PREFISSO_SCHEMA)
        if ids:
            self._continuazione.registra_prefisso(ids)

    def _completa_llm(self, llm: Any, prompt: str, **kwargs: Any) -> str | None:
        """
        Generazione senza pannello risposta.

        Returns:
            Il testo (vuoto se il mock non restituisce una stringa), oppure None
            se Ollama ha sollevato un errore: in quel caso ripetere è inutile.
        """
        self._controlla_annullamento()
        try:
            risultato = llm.invoke(prompt, **kwargs)
        except TurnoAnnullato:
            raise
        except Exception:
            return None
        if isinstance(risultato, str):
            return risultato
        return ""

    def _controlla_annullamento(self) -> None:
        """Ctrl+C cooperativo fra schema, ricerche e scritture."""
        if self._cancel_event.is_set():
            raise TurnoAnnullato("Turno annullato dall'utente.")

    def _crea_callback_turno(
        self,
        config: ConfigRuntime,
        contesto: ContestoTurnoTermico | None,
        context_tokens: int,
    ) -> CallbackUI:
        """
        Costruisce l'observer condiviso da una route.

        Args:
            config: Flag runtime, incluso burn protection.
            contesto: Snapshot termico opzionale prodotto prima del turno.
            context_tokens: Finestra usata per la percentuale contesto.

        Returns:
            Callback con cancellazione e metriche per-turno.
        """
        throttle = contesto.throttle if contesto else None
        callback = CallbackUI(
            self._ui,
            throttle.pausa_tra_iterazioni_sec if throttle else 0.0,
            stato_iniziale=contesto.stato if contesto else StatoTermico.NOMINALE,
            burn_protection=config.burn_protection,
            contesto_massimo=context_tokens,
            cancel_event=self._cancel_event,
            registro=self._registro,
        )
        self._ultimo_callback = callback
        return callback

    def ultime_metriche(self) -> StatisticheTurnoLLM | None:
        """Snapshot LLM dell'ultimo turno, oppure None."""
        if self._ultimo_callback is None:
            return None
        return self._ultimo_callback.ultime_metriche()

    def _memoria_pool_bytes(self) -> int:
        """Pool di inferenza dello snapshot host; 0 se la sessione non lo conosce."""
        if self._host is None:
            return 0
        return self._host.inference_memory_bytes or self._host.memory_bytes

    def _profilo_con_storico(
        self,
        alias: str,
        route: InferenceRoute,
        output_tokens: int | None,
        richiesta: str,
        scratchpad_tokens: int,
    ) -> tuple[Any, str]:
        """
        Profilo sulla finestra di sessione, con storico entro budget.

        Tutte le route usano `contesto_sessione`: cambiare `num_ctx` costringe
        Ollama a ricaricare i pesi. Si sale di uno scalino solo se il prompt
        riempie la finestra.
        """
        profile = profile_for_route(
            alias,
            route,
            output_tokens=output_tokens,
            host=self._host,
        )
        sessione = self._runtime.contesto_sessione(alias)
        if sessione != profile.context_tokens:
            profile = replace(profile, context_tokens=sessione)
        storico = self._storico_per_profilo(
            profile.context_tokens,
            profile.output_tokens,
            scratchpad_tokens=scratchpad_tokens,
            route=route,
        )
        stima = (
            estimate_tokens(richiesta)
            + estimate_tokens(self._project_memory)
            + estimate_tokens(storico)
        )
        chosen = contesto_espanso(
            profile.context_tokens,
            alias=alias,
            kv_cache_type=self._runtime.kv_cache_type,
            memory_bytes=self._memoria_pool_bytes(),
            weight_gb=profile.variant.size_gb,
            can_expand=self._host is not None and can_expand_context(self._host),
            tight=False,
            prompt_tokens=stima,
            model_ceiling=self._runtime.tetto_contesto_modello(alias),
        )
        if chosen == profile.context_tokens:
            return profile, storico
        profile = replace(profile, context_tokens=chosen)
        return profile, self._storico_per_profilo(
            profile.context_tokens,
            profile.output_tokens,
            scratchpad_tokens=scratchpad_tokens,
            route=route,
        )

    def _prefisso_direct(self) -> str:
        """Testa stabile del prompt direct, identica a quella costruita dalla pipeline."""
        memory = self._project_memory.strip()
        if memory:
            return f"{PREFISSO_SISTEMA_DIRECT}\n\n{memory}\n\n"
        return f"{PREFISSO_SISTEMA_DIRECT}\n\n"

    def _assicura_prefisso_direct(
        self,
        alias: str,
        context_tokens: int,
        reasoning: bool,
    ) -> None:
        """Usa gli id del preload, oppure un prefill corto se il prefisso è diverso."""
        prefix = self._prefisso_direct()
        chiave = (alias, context_tokens, reasoning, prefix)
        if self._continuazione.ha_prefisso(chiave):
            return
        self._continuazione.imposta_chiave(chiave, prefix)
        ids = self._runtime.context_prefisso_precaricato(prefix, context_tokens)
        if ids is None:
            ids = self._runtime.prefill_prefisso_raw(alias, context_tokens, prefix)
        if ids:
            self._continuazione.registra_prefisso(ids)

    def _storico_per_profilo(
        self,
        context_tokens: int,
        output_tokens: int,
        *,
        scratchpad_tokens: int,
        route: InferenceRoute = "direct",
    ) -> str:
        """
        Seleziona i turni recenti entro una quota del budget input.

        Args:
            context_tokens: Finestra totale della route.
            output_tokens: Riserva di generazione.
            scratchpad_tokens: Riserva per tool o struttura del prompt.
            route: Route usata per dimensionare lo storico.

        Returns:
            Storico formattato e limitato in modo conservativo.
        """
        budget = PromptBudget(
            context_tokens=context_tokens,
            output_tokens=output_tokens,
            scratchpad_tokens=scratchpad_tokens,
        )
        if route == "direct":
            history_tokens = min(
                max(_STORICO_MIN_TOKEN, budget.input_tokens // _STORICO_QUOTA_LEGGERA),
                _STORICO_MAX_TOKEN_LEGGERO,
            )
        else:
            history_tokens = max(
                _STORICO_MIN_TOKEN,
                budget.input_tokens // _STORICO_QUOTA_PESANTE,
            )
        return self._memoria.formatta(max_tokens=history_tokens)

    def _finalizza_turno(
        self,
        richiesta: str,
        risposta: str,
        *,
        scrittura_richiesta: bool,
        path_scritti: list[str] | None = None,
        max_iterations_hit: bool = False,
    ) -> str:
        """
        Applica evaluate_turn e registra lo storico in un solo punto.

        ``scrittura_richiesta`` riflette l'intento dell'utente, non il permesso dei
        tool: la route ReAct apre la scrittura anche per una semplice lettura, e in
        quel caso mancare file non è un problema da segnalare.
        """
        evaluated = evaluate_turn(
            risposta,
            scrittura_richiesta=scrittura_richiesta,
            path_scritti=path_scritti or [],
            max_iterations_hit=max_iterations_hit,
        )
        metriche = self.ultime_metriche()
        self._registro.termina_turno(
            classifica_esito_turno(
                incompleto=evaluated.incompleto,
                risposta=risposta,
            ),
            note=evaluated.note,
            metriche=metriche.as_dict() if metriche else None,
        )
        self._memoria.registra(richiesta, evaluated.risposta)
        return evaluated.risposta

    def _registra_throttle_se_taglia(self, contesto: ContestoTurnoTermico) -> None:
        """Registra i tagli termici che spiegano turni tronchi senza essere un crash."""
        throttle = contesto.throttle
        taglia = (
            throttle.num_predict < MAX_TOKEN_GENERATI
            or throttle.max_iterazioni < MAX_ITERAZIONI_AGENTE
            or throttle.disabilita_thinking
        )
        if not taglia:
            return
        self._registro.registra_termica(
            throttle.stato.name,
            num_predict=throttle.num_predict,
            max_iterazioni=throttle.max_iterazioni,
            alias=throttle.alias_modello,
        )

    def _esegui_turno_direct(
        self,
        richiesta: str,
        alias: str,
        thinking: bool,
        config: ConfigRuntime,
        contesto: ContestoTurnoTermico | None,
        comprensione: ComprensioneTurno,
    ) -> str:
        """
        Esegue chat/follow-up con una sola generazione.

        Returns:
            Risposta completa già registrata nella memoria di sessione.
        """
        throttle = contesto.throttle if contesto else None
        profile, storico = self._profilo_con_storico(
            alias,
            "direct",
            throttle.num_predict if throttle else None,
            richiesta,
            128,
        )
        callback = self._crea_callback_turno(
            config,
            contesto,
            profile.context_tokens,
        )
        self._assicura_prefisso_direct(alias, profile.context_tokens, thinking)
        # Direct: prefisso KV senza INTENT CONTEXT, al più la riga COMPITO.
        risultato = self._direct_pipeline.answer(
            richiesta,
            alias,
            storico,
            profile.context_tokens,
            profile.output_tokens,
            callback,
            intent_context=formatta_blocco_comprensione(comprensione),
            project_memory=self._project_memory,
            reasoning=thinking,
        )
        return self._finalizza_turno(
            richiesta,
            risultato,
            scrittura_richiesta=False,
        )

    def _esegui_turno_react(
        self,
        richiesta: str,
        alias: str,
        thinking: bool,
        verifica: bool,
        decisione: DecisioneTurno,
        config: ConfigRuntime,
        contesto: ContestoTurnoTermico | None,
        comprensione: ComprensioneTurno,
    ) -> tuple[str, list[str]]:
        """
        Esegue il percorso workspace con tool minimi e budget limitati.

        Returns:
            Risposta finale e path eventualmente materializzati dal fallback.
        """
        scrittura_consentita = decisione.scrittura_consentita
        throttle = contesto.throttle if contesto else None
        max_iterations = ITERAZIONI_REACT_APERTO
        if throttle is not None:
            if scrittura_consentita:
                max_iterations = max(max_iterations, throttle.max_iterazioni)
            else:
                max_iterations = min(max_iterations, throttle.max_iterazioni)
        profile, storico = self._profilo_con_storico(
            alias,
            "react",
            throttle.num_predict if throttle else None,
            richiesta,
            768,
        )
        executor = self._ottieni_executor(
            alias,
            max_iterations,
            profile.output_tokens,
            profile.context_tokens,
            scrittura_consentita=scrittura_consentita,
            reasoning=thinking,
        )
        callback = self._crea_callback_turno(
            config,
            contesto,
            profile.context_tokens,
        )
        self._active_callback = callback
        try:
            esito_executor = executor.invoke(
                    {
                        "input": _costruisci_descrizione_task(
                            richiesta,
                            verifica,
                            decisione,
                            storico,
                            project_memory=self._project_memory,
                            comprensione=comprensione,
                            nota=self._nota_turno,
                        ),
                    },
                    RunnableConfig(callbacks=[callback]),
                )
        finally:
            self._active_callback = None
        risultato = str(esito_executor["output"])
        hit_limit = any(marker in risultato.casefold() for marker in _MARCATORI_LIMITE)
        if hit_limit:
            ripiego = _risposta_da_passi(esito_executor.get("intermediate_steps", []))
            if ripiego is not None:
                self._registro.registra_diagnostica("limite iterazioni: ripiego sull'ultimo passo")
                risultato = f"{ripiego}\n\n{NOTA_RIPIEGO_LIMITE}"
                hit_limit = False
        path_scritti = self._file_service.written_paths
        if scrittura_consentita:
            # Solo i file: una cartella creata non è un «file scritto» da elencare.
            risultato = _arricchisci_risposta(
                risultato,
                self._file_service.written_files,
            )
            self._path_fallback_turno = applica_fallback(
                risultato,
                path_scritti,
                service=self._file_service,
            )
            path_scritti = self._file_service.written_paths
        finale = self._finalizza_turno(
            richiesta,
            risultato,
            scrittura_richiesta=decisione.analisi.scrittura,
            path_scritti=path_scritti,
            max_iterations_hit=hit_limit,
        )
        return finale, path_scritti

    def _ottieni_executor(
        self,
        alias: str,
        max_iter: int,
        num_predict: int,
        context_tokens: int,
        *,
        scrittura_consentita: bool,
        reasoning: bool,
    ) -> AgentExecutor:
        """Restituisce un executor cacheato sull'intero profilo operativo."""
        chiave = ExecutorKey(
            alias,
            max_iter,
            num_predict,
            context_tokens,
            capabilities_react(),
            scrittura_consentita,
            reasoning,
        )
        cached = self._executor_per_chiave.get(chiave)
        if cached is not None:
            self._executor_per_chiave.move_to_end(chiave)
            return cached
        created = self._crea_executor(
            alias,
            max_iter,
            num_predict,
            context_tokens,
            chiave.capabilities,
            scrittura_consentita=scrittura_consentita,
            reasoning=reasoning,
        )
        self._executor_per_chiave[chiave] = created
        while len(self._executor_per_chiave) > EXECUTOR_CACHE_LIMIT:
            self._executor_per_chiave.popitem(last=False)
        return created

    def _crea_executor(
        self,
        alias: str,
        max_iter: int,
        num_predict: int,
        context_tokens: int,
        capabilities: CapabilitySet,
        *,
        scrittura_consentita: bool,
        reasoning: bool,
    ) -> AgentExecutor:
        tools = self._crea_tools_per_capability(
            alias,
            capabilities,
            scrittura_consentita=scrittura_consentita,
            reasoning=reasoning,
        )
        prompt = PromptTemplate.from_template(_TEMPLATE_REACT).partial(
            identita=_IDENTITA_AGENTE
        )
        # Parser custom: evita loop su "Action: Final Answer" (tool fantasma).
        agente = create_react_agent(
            llm=self._runtime.get_llm(
                alias,
                context_tokens,
                num_predict,
                reasoning=reasoning,
            ),
            tools=tools,
            prompt=prompt,
            output_parser=AssistenteReActParser(),
        )
        budget = PromptBudget(
            context_tokens=context_tokens,
            output_tokens=num_predict,
        )
        return AgentExecutor(
            agent=agente,
            tools=tools,
            max_iterations=max_iter,
            return_intermediate_steps=True,
            handle_parsing_errors=_osservazione_errore_parsing,
            trim_intermediate_steps=lambda steps: _limita_passi_react(
                steps,
                budget.scratchpad_tokens,
            ),
            verbose=False,
        )

    def _crea_tools_per_capability(
        self,
        alias: str,
        capabilities: CapabilitySet,
        *,
        scrittura_consentita: bool = False,
        reasoning: bool = True,
    ) -> list[Any]:
        """
        Costruisce soltanto i tool pertinenti alla richiesta.

        Args:
            alias: Modello attivo condiviso col riassunto.
            capabilities: Famiglie ordinate prodotte dalla policy.
            scrittura_consentita: Gate usato nelle descrizioni dinamiche.

        Returns:
            Lista LangChain deterministica usata da prompt ed executor.
        """
        tools: list[Any] = []
        # I tool file sono necessari anche al riassunto per leggere e salvare.
        if "file" in capabilities:
            tools.extend(
                crea_tool_file(
                    self._file_service,
                    write_allowed=scrittura_consentita,
                    workspace_label=self._file_service.workspace.name,
                )
            )
        # Il provider chiuso sull'alias impedisce il caricamento implicito del 12B.
        if "summary" in capabilities:
            tools.extend(
                crea_tool_riassunto(
                    self._file_service,
                    alias=alias,
                    # La finestra del profilo summary è solo budget: num_ctx resta
                    # quello di sessione, così il riassunto non ricarica i pesi.
                    llm_provider=lambda _context, output: self._runtime.get_llm(
                        alias,
                        self._runtime.contesto_sessione(alias),
                        output,
                        reasoning=reasoning,
                    ),
                    callback_provider=lambda: self._active_callback,
                )
            )
        # La data serve a calcoli su date citate nei documenti.
        if "date" in capabilities:
            tools.extend(crea_tool_data())
        return tools
