"""Prova di comprensione end-to-end del classificatore di turno.

Esercita il percorso reale di comprensione — prompt schema → Ollama → reparo
JSON → validatore → routing — su un corpus di richieste scritte da un utente.
Misura quante dimensioni (kind, scrittura, path, route) il modello azzecca,
distinguendo accuratezza per categoria e costo di riparazione JSON.

Uso:
    python -m benchmarks.comprensione_probe --alias default
    python -m benchmarks.comprensione_probe --alias default fast --output output/...
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from assistente.comprensione import scheda_locale
from assistente.ollama_runtime import OllamaRuntime
from assistente.politica_turno import _e_chat_locale, decisione_da_schema
from assistente.schema_turno import (
    NUM_PREDICT_SCHEMA,
    formatta_prompt_schema,
    ripara_json_schema,
    schema_json_ollama,
)
from assistente.validatore_turno import ContestoValidazione, valida_turno


@dataclass(frozen=True, slots=True)
class Caso:
    """Una richiesta utente con l'esito di comprensione atteso dalla semantica."""

    categoria: str
    testo: str
    kind_attesi: frozenset[str] | None = None
    write_atteso: bool | None = None
    route_attesa: str | None = None
    path_atteso: str | None = None
    cartella_attesa: str | None = None


# Corpus: richieste come le scriverebbe una persona, con l'esito inteso.
# kind_attesi=None significa «non valutato» (es. saluto, che non chiama lo schema).
CORPUS: tuple[Caso, ...] = (
    Caso("saluto", "ciao", route_attesa="direct", write_atteso=False),
    Caso("saluto", "grazie mille", route_attesa="direct", write_atteso=False),
    Caso(
        "domanda",
        "quanto pesa una balenottera azzurra adulta?",
        kind_attesi=frozenset({"explain"}),
        route_attesa="react",
        write_atteso=False,
    ),
    Caso(
        "domanda",
        "qual è la differenza tra misticeti e odontoceti?",
        kind_attesi=frozenset({"explain"}),
        route_attesa="react",
        write_atteso=False,
    ),
    Caso(
        "domanda",
        "cosa dicono i documenti sulle rotte migratorie?",
        kind_attesi=frozenset({"explain", "workspace"}),
        route_attesa="react",
        write_atteso=False,
    ),
    Caso(
        "lettura",
        "leggi il file avvistamenti.pdf e dimmi di cosa parla",
        kind_attesi=frozenset({"explain", "workspace", "chat"}),
        route_attesa="react",
        write_atteso=False,
        path_atteso="avvistamenti.pdf",
    ),
    Caso(
        "lettura",
        "cosa mostra la foto coda.jpg?",
        kind_attesi=frozenset({"explain", "workspace"}),
        route_attesa="react",
        write_atteso=False,
        path_atteso="coda.jpg",
    ),
    Caso(
        "scrittura",
        "riassumi report.docx e salva il riassunto in sintesi.txt",
        kind_attesi=frozenset({"workspace"}),
        route_attesa="react",
        write_atteso=True,
        path_atteso="sintesi.txt",
    ),
    Caso(
        "scrittura",
        "nella cartella note crea il file domande.txt con le domande aperte",
        kind_attesi=frozenset({"workspace"}),
        route_attesa="react",
        write_atteso=True,
        path_atteso="domande.txt",
        cartella_attesa="note",
    ),
    Caso(
        "vincolo_negativo",
        "dimmi cosa mangiano le orche, niente file, solo in chat",
        kind_attesi=frozenset({"chat", "explain"}),
        route_attesa="react",
        write_atteso=False,
    ),
)


@dataclass
class Esito:
    """Comprensione di un singolo caso su un singolo alias."""

    categoria: str
    testo: str
    esito: str
    kind_detto: str = ""
    kind_ok: bool | None = None
    route_detta: str = ""
    route_ok: bool | None = None
    write_detto: bool = False
    write_ok: bool | None = None
    path_ok: bool | None = None
    cartella_ok: bool | None = None
    riparato: bool = False
    latenza_s: float = 0.0
    superato: bool = False
    dettaglio: str = ""

    def valutata(self) -> bool:
        """True se almeno una dimensione era attesa (caso scored)."""
        return any(
            flag is not None
            for flag in (
                self.kind_ok,
                self.route_ok,
                self.write_ok,
                self.path_ok,
                self.cartella_ok,
            )
        )


def _valuta(
    caso: Caso,
    esito: str,
    schema: object | None,
    route: str,
    riparato: bool,
    latenza: float,
) -> Esito:
    """Confronta l'esito osservato con le attese del caso, dimensione per dimensione."""
    esito_caso = Esito(
        categoria=caso.categoria,
        testo=caso.testo,
        esito=esito,
        route_detta=route,
        riparato=riparato,
        latenza_s=round(latenza, 2),
    )
    if schema is None:
        esito_caso.route_ok = (
            None if caso.route_attesa is None else route == caso.route_attesa
        )
        esito_caso.write_ok = None if caso.write_atteso is None else caso.write_atteso is False
        esito_caso.superato = bool(esito_caso.route_ok) and (
            caso.write_atteso is None or caso.write_atteso is False
        )
        return esito_caso

    kind = str(getattr(schema, "kind", ""))
    write = bool(getattr(schema, "write", False) or getattr(schema, "deliver", "") == "file")
    paths = tuple(getattr(schema, "paths", ()))
    folder = str(getattr(schema, "folder", ""))
    esito_caso.kind_detto = kind
    esito_caso.write_detto = write

    if caso.kind_attesi is not None:
        esito_caso.kind_ok = kind in caso.kind_attesi
    if caso.route_attesa is not None:
        esito_caso.route_ok = route == caso.route_attesa
    if caso.write_atteso is not None:
        esito_caso.write_ok = write == caso.write_atteso
    if caso.path_atteso is not None:
        atteso = caso.path_atteso.casefold()
        esito_caso.path_ok = any(p.casefold().endswith(atteso) for p in paths)
    if caso.cartella_attesa is not None:
        esito_caso.cartella_ok = folder.replace("\\", "/").casefold() == (
            caso.cartella_attesa.casefold()
        ) or caso.cartella_attesa.casefold() in folder.casefold()

    valutate = [
        flag
        for flag in (
            esito_caso.kind_ok,
            esito_caso.route_ok,
            esito_caso.write_ok,
            esito_caso.path_ok,
            esito_caso.cartella_ok,
        )
        if flag is not None
    ]
    fallite = [i for i, flag in enumerate(valutate) if not flag]
    esito_caso.superato = all(valutate) and bool(valutate)
    if fallite:
        esito_caso.dettaglio = "fallite: " + ", ".join(str(i) for i in fallite)
    return esito_caso


def esegui_caso(
    runtime: OllamaRuntime,
    alias: str,
    caso: Caso,
) -> Esito:
    """Compila la scheda di un caso con il modello reale e la valuta."""
    # Il saluto esatto salta lo schema anche in produzione: route locale diretta.
    if _e_chat_locale(caso.testo):
        return _valuta(caso, "schema_skipped_saluto", None, "direct", False, 0.0)
    # Come in produzione: la scheda locale decide i casi chiari senza LLM.
    inizio = time.perf_counter()
    locale = scheda_locale(caso.testo, lambda _path: False)
    if locale is not None:
        schema = valida_turno(locale, caso.testo, ContestoValidazione()).schema
        return _valuta(
            caso,
            "locale",
            schema,
            decisione_da_schema(schema).percorso,
            False,
            time.perf_counter() - inizio,
        )

    llm = runtime.get_llm(
        alias,
        runtime.contesto_sessione(alias),
        NUM_PREDICT_SCHEMA,
        reasoning=False,
    )
    prompt = formatta_prompt_schema(caso.testo, "2026-09-28")
    inizio = time.perf_counter()
    risposta = llm.invoke(
        prompt,
        format=schema_json_ollama(),
        num_predict=NUM_PREDICT_SCHEMA,
        reasoning=False,
        temperature=0.0,  # come la produzione (Fase 7): misura il campionamento effettivo
    )
    latenza = time.perf_counter() - inizio
    testo = risposta if isinstance(risposta, str) else ""
    repair = ripara_json_schema(testo)
    if repair.schema is None:
        return _valuta(caso, "schema_fallback", None, "", True, latenza)

    validazione = valida_turno(repair.schema, caso.testo, ContestoValidazione())
    schema = validazione.schema
    decisione = decisione_da_schema(schema)
    return _valuta(
        caso,
        "schema",
        schema,
        decisione.percorso,
        repair.riparato,
        latenza,
    )


def _riepilogo(alias: str, esiti: list[Esito], repeat: int) -> dict[str, object]:
    """Aggrega accuratezza per dimensione, categoria e stabilità del kind."""
    scored = [e for e in esiti if e.valutata()]
    per_dimensione: dict[str, dict[str, int]] = {}
    for nome, campo in (
        ("kind", "kind_ok"),
        ("route", "route_ok"),
        ("scrittura", "write_ok"),
        ("path", "path_ok"),
        ("cartella", "cartella_ok"),
    ):
        valutati = [getattr(e, campo) for e in esiti if getattr(e, campo) is not None]
        if valutati:
            per_dimensione[nome] = {
                "ok": sum(1 for v in valutati if v),
                "tot": len(valutati),
            }
    per_categoria: dict[str, dict[str, int]] = {}
    for esito in scored:
        voce = per_categoria.setdefault(esito.categoria, {"ok": 0, "tot": 0})
        voce["tot"] += 1
        if esito.superato:
            voce["ok"] += 1
    # Stabilità per richiesta: tutte le ripetizioni uguali, e kind costante
    per_testo: dict[str, list[Esito]] = {}
    for esito in scored:
        per_testo.setdefault(esito.testo, []).append(esito)
    stabili = sum(
        1
        for run in per_testo.values()
        if sum(1 for e in run if e.superato) == len(run)
    )
    kind_costante = sum(
        1
        for run in per_testo.values()
        if len({e.kind_detto for e in run}) == 1
    )
    totale_ok = sum(1 for e in scored if e.superato)
    return {
        "alias": alias,
        "ripetizioni": repeat,
        "casi": len(per_testo),
        "scored": len(scored),
        "superati": totale_ok,
        "accuratezza": round(totale_ok / len(scored), 3) if scored else 0.0,
        "casi_stabili": stabili,
        "kind_costante": kind_costante,
        "fallback_schema": sum(1 for e in esiti if e.esito == "schema_fallback"),
        "risolti_in_locale": sum(1 for e in esiti if e.esito == "locale"),
        "riparati": sum(1 for e in esiti if e.riparato),
        "latenza_media_s": round(
            sum(e.latenza_s for e in esiti) / len(esiti), 2
        )
        if esiti
        else 0.0,
        "per_dimensione": per_dimensione,
        "per_categoria": per_categoria,
    }


def main() -> None:
    """Esegue il corpus sugli alias richiesti e stampa/reporta il riepilogo."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alias",
        nargs="+",
        default=["default"],
        help="Alias modello da provare (fast, default, quality).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Tronca il corpus.")
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Ripetizioni per caso: misura la stabilità della comprensione.",
    )
    parser.add_argument("--output", type=Path, default=None, help="JSON di output.")
    args = parser.parse_args()

    corpus = CORPUS[: args.limit] if args.limit else CORPUS
    runtime = OllamaRuntime(manage_residency=False)
    report: dict[str, object] = {
        "corpus": len(corpus),
        "repeat": args.repeat,
        "alias": {},
    }

    for alias in args.alias:
        esiti: list[Esito] = []
        print(f"\n=== alias {alias} (repeat={args.repeat}) ===")
        for giro in range(args.repeat):
            for caso in corpus:
                esito = esegui_caso(runtime, alias, caso)
                esiti.append(esito)
                stato = "OK " if esito.superato else "KO "
                print(
                    f"  {stato}[r{giro}][{caso.categoria}] "
                    f"{esito.kind_detto or esito.esito} "
                    f"route={esito.route_detta} {esito.latenza_s:.1f}s :: {caso.testo[:50]}"
                )
        riepilogo = _riepilogo(alias, esiti, args.repeat)
        report["alias"][alias] = {
            "riepilogo": riepilogo,
            "esiti": [asdict(e) for e in esiti],
        }
        print(
            f"  -> accuratezza {riepilogo['accuratezza']} "
            f"({riepilogo['superati']}/{riepilogo['scored']}) "
            f"| casi stabili {riepilogo['casi_stabili']}/{riepilogo['casi']}"
        )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\nReport scritto in {args.output}")


if __name__ == "__main__":
    main()
