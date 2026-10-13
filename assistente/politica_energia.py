"""Policy energetica: Low Power Mode usa l'alias fast senza mutare la preferenza."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from assistente.config import ConfigRuntime
from assistente.host import PowerState, detect_power
from assistente.inference_profiles import MODEL_ALIAS_ORDER

_MESSAGGIO_LPM = (
    "Low Power Mode: uso fast (E4B); collega l'alimentatore o "
    "disattiva Risparmio energetico."
)
_MESSAGGIO_BATTERIA = (
    "Alimentazione a batteria: i token/s possono calare. "
    "Collega l'alimentatore per prestazioni piene."
)


class EnergiaUI(Protocol):
    """Superficie minima per gli avvisi energetici."""

    def stampa_messaggio_sistema(self, messaggio: str) -> None: ...


@dataclass(frozen=True, slots=True)
class ContestoEnergia:
    """Alias effettivo e snapshot alimentazione del turno."""

    alias_modello: str
    power: PowerState
    messaggio: str | None


_ultimo_stato_annunciato: str | None = None


def reset_energia_per_test() -> None:
    """Riporta l'annuncio one-shot allo stato iniziale."""
    global _ultimo_stato_annunciato
    _ultimo_stato_annunciato = None


def _indice_alias(alias: str) -> int:
    """Posizione in MODEL_ALIAS_ORDER; alias sconosciuto = default."""
    try:
        return MODEL_ALIAS_ORDER.index(alias)
    except ValueError:
        return MODEL_ALIAS_ORDER.index("default")


def alias_piu_leggero(primo: str, secondo: str) -> str:
    """Sceglie l'alias più a sinistra nella matrice fast < default."""
    return MODEL_ALIAS_ORDER[min(_indice_alias(primo), _indice_alias(secondo))]


def alias_per_alimentazione(
    alias_richiesto: str,
    config: ConfigRuntime,
    *,
    power: PowerState | None = None,
) -> str:
    """
    Applica il downgrade a fast solo con Low Power Mode e flag attivo.
    """
    stato = power if power is not None else detect_power()
    if config.auto_fast_low_power and stato.low_power_mode:
        return "fast"
    return alias_richiesto


def prepara_turno_energia(
    config: ConfigRuntime,
    ui: EnergiaUI,
    *,
    power: PowerState | None = None,
) -> ContestoEnergia:
    """
    Deriva l'alias effettivo e annuncia un cambio di alimentazione una sola volta.
    """
    global _ultimo_stato_annunciato
    stato = power if power is not None else detect_power()
    alias = alias_per_alimentazione(config.modello, config, power=stato)
    chiave = stato.chiave()
    messaggio: str | None = None
    if chiave == "lpm" and alias == "fast" and config.modello != "fast":
        messaggio = _MESSAGGIO_LPM
    elif chiave == "battery":
        messaggio = _MESSAGGIO_BATTERIA
    if messaggio is not None and chiave != _ultimo_stato_annunciato:
        ui.stampa_messaggio_sistema(messaggio)
    _ultimo_stato_annunciato = chiave
    return ContestoEnergia(alias, stato, messaggio)
