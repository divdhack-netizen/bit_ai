# Procedura guidata di primo avvio

Questo documento è la spec del wizard eseguito da `main.py`.
Il codice in `assistente/startup.py` e `assistente/bootstrap.py` deve seguirlo;
in caso di dubbio vince questo file.

Nessun download o installazione avviene senza **consenso esplicito** (sì).
Un rifiuto (no, invio vuoto, Ctrl+C) interrompe l’avvio con codice di uscita diverso da zero.
Non esiste un marker su disco che nasconda dipendenze mancanti: ogni lancio **ri-verifica** lo stato reale.

## 0. Prerequisiti (fuori dal wizard)

Il wizard **non** installa Python né crea il virtualenv.

- macOS (tutti i SoC Apple Silicon; Intel è avvisato e prosegue) **oppure** Linux
- Python 3.12 o successivo
- repository clonato e dipendenze:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cd ~/Documents/balena   # la cartella dei documenti
~/Documents/bit_ai/venv/bin/python ~/Documents/bit_ai/main.py
```

L'agente lavora solo sui documenti della cartella di lancio (o di `--cartella`).

Windows non è un target: se Ollama è già in ascolto l’agente può funzionare in degrado
(niente wizard, niente burn protection nativa). Altrimenti il processo esce con codice 1.

## 1. Modalità interattiva e non interattiva

All’avvio l’agente rileva, in quest’ordine:

1. CLI Ollama (`ollama` nel `PATH`; su macOS anche nella `bin/` di Homebrew se brew c’è)
2. API Ollama (`GET` sull’URL da `OLLAMA_HOST`, default `http://127.0.0.1:11434/api/version`)
3. Almeno un artefatto Gemma 4 della matrice (`fast` / `default`, inclusi i fallback di tag in `inference_profiles.py`)

Homebrew è richiesto **solo su macOS e solo se manca Ollama**. Su Linux non è un prerequisito,
anche se Linuxbrew è installato.

**Ambiente pronto:** CLI+API Ollama e almeno un Gemma 4 della matrice. Il wizard non parte.
Il rilevamento e il preriscaldamento del modello partono **insieme** allo splash HUD
(se la sessione è interattiva). Il panel di benvenuto compare a preload concluso
(o a splash concluso, se il load è già finito).

**Manca qualcosa e lo stdin è un TTY** (nessun `--prompt`): parte il wizard a passi, solo per i pezzi assenti.

**Manca qualcosa e la sessione non è interattiva** (`--prompt` e/o stdin non TTY): **nessun wizard**. Messaggio che rimanda a questo file e uscita con codice `1`.

## 2. macOS: Homebrew (solo se Ollama manca)

Rilevamento:

- `which brew`
- path Apple Silicon: `/opt/homebrew/bin/brew`
- path Intel: `/usr/local/bin/brew`

Se il binario esiste ma non è nel `PATH` del processo, il wizard lo antepone al `PATH` e applica `brew shellenv` (variabili `HOMEBREW_*` e `PATH`).

Se manca **e** manca Ollama:

1. Spiega che serve Homebrew e che l’installer ufficiale chiederà la **password amministratore**.
2. Chiede consenso. Rifiuto → stop.
3. Esegue in foreground, con stdin/stdout/stderr del terminale:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Il processo Python **non** wrappa `sudo` e **non** imposta un timeout: l’installer è interattivo.

4. Dopo exit `0`, applica `brew shellenv` al processo corrente. Se `brew` è ancora assente → errore e stop.
5. Consiglia (senza eseguirlo) di aggiungere a `~/.zprofile`:

```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Se `curl` o la rete falliscono: messaggio con https://brew.sh, nessun retry automatico.

## 3. Ollama

Rilevamento: CLI `ollama` **e** API in ascolto.

### 3.1 macOS

Se la CLI manca:

1. Chiede consenso a `brew install ollama`. Rifiuto → stop.
2. Esegue `brew install ollama`.
3. Avvia il servizio: `brew services start ollama`.
4. Attende l’API fino a circa 20 secondi.
5. Se la porta non risponde: avvia `ollama serve` in background (con env server: flash attention, KV `q4_0`, un modello, `num_parallel=1`, keep-alive `30m`) e ritenta l’attesa.
6. Se l’API resta irraggiungibile → errore e stop.

Se la CLI c’è ma l’API è giù: **non** reinstallare. Solo avvio servizio / `ollama serve` e healthcheck.

Non si impostano `launchctl setenv` sul daemon già avviato. Se Ollama era già un servizio, il client stampa un avviso: le env restano opzionali nel README.

Non si usa l’installer `.dmg` / `.app` dal sito Ollama.

### 3.2 Linux

Se la CLI manca:

1. Chiede consenso all’installer ufficiale `https://ollama.com/install.sh`. Rifiuto → stop.
2. Esegue in foreground:

```bash
/bin/sh -c "$(curl -fsSL https://ollama.com/install.sh)"
```

3. Tenta `systemctl --user start ollama`, poi `systemctl start ollama`.
4. Se l’API non risponde: `ollama serve` in background con le stesse env server del caso macOS.
5. Timeout API → errore e stop.

Se la CLI c’è ma l’API è giù: solo avvio systemd / `serve`, senza reinstallare.

## 4. Modelli Gemma 4

Famiglia unica, tutto on-device (niente tag `:cloud`). Matrice fissa (pesi dichiarati, esclusa KV cache):

| Alias    | Tag Ollama           | Quant | Pesi ~ |
|----------|----------------------|-------|--------|
| `fast`   | `gemma4:e4b`         | q4    | 6.6 GB |
| `default`| `gemma4:12b`         | q4    | 8.0 GB |

Fallback di catalogo (come in `TAG_INSTALL_CANDIDATES`): `gemma4:e4b-it-q4_K_M` e `gemma4` per fast; `gemma4:12b-it-q4_K_M` per default. Il suffisso `:latest` è normalizzato a `gemma4` (che in libreria punta a E4B).

Fasce in base al pool di inferenza (RAM unificata su Apple; VRAM su Linux NVIDIA) e all’headroom già usato da `fits_memory`:

- **sconsigliato** — i pesi (più 6 GB di headroom sulla memoria unificata) non entrano.
- **stretto** — entra, ma il residuo dopo i pesi è inferiore a 8 GB (6 GB di headroom + 2 GB di margine)
- **consigliato** — altrimenti. Su 16 GB: `fast` e `default`. Su 8 GB: entrambi sconsigliati. Su 64 GB: entrambi consigliati.

RAM/VRAM non rilevabile: nessuna fascia punitiva (si può pullare).

Il wizard elenca le due righe (indice, alias, tag, GB, fascia, già presente o no). Evidenzia un suggerimento in base alla memoria, **senza** preselezionarlo. Se `default` è consigliato (tipico su 16 GB e oltre) è quello evidenziato.

L’utente indica quali pullare: indici (`1 2`), alias (`fast default`), anche uno solo o entrambi. Invio vuoto = nessuno.
Una selezione illeggibile viene riproposta (fino a 5 tentativi), non fa abortire subito il wizard.

- Nessuna scelta e **nessun** Gemma 4 della matrice già in Ollama → stop (codice `1`).
- Nessuna scelta ma **almeno uno** è già installato → avviso e si prosegue.
- Scelte: `POST /api/pull` in streaming verso Ollama; progresso in console; dopo ogni pull si invalida il catalogo.

## 4.1 Visione per le immagini

I Gemma 4 della matrice sono multimodali: le immagini della cartella (png, jpg,
webp) vengono lette dallo stesso modello già caricato. Non serve alcun modello
aggiuntivo da scaricare.

## 5. Fine e sessione

Quando API Ollama e almeno un Gemma 4 della matrice sono presenti:

1. Se `--model` / alias di sessione non è tra quelli installati, si usa il primo alias della matrice che lo è (`fast`, poi `default`) e si informa l’utente (`/model` per cambiarlo).
2. Il thread di avvio precarica i pesi in parallelo allo splash (probe + `OllamaRuntime.preload`). La sessione agente si apre a HUD concluso, con `num_gpu` / `num_thread` derivati dal host (Metal su Apple Silicon, CUDA se c’è NVIDIA, CPU-only altrimenti). I messaggi di esito restano sul thread principale.
3. Se il load dura più dell’animazione, la CLI attende la fine del preload prima del panel «Assistente pronto».

## 6. Lancio successivo

Ogni avvio ripete il rilevamento del passo 1. Ollama spento o modelli cancellati fanno ripartire solo i passi mancanti.

Non esiste un file “setup completato” che salti i controlli.

## 7. Limiti

- Il wizard non bypassa `sudo` né simula la password.
- Non installa Python, pip o il venv.
- Non configura `launchctl` per un daemon Ollama già in esecuzione.
- Non propone famiglie diverse da Gemma 4.
- Non esegue `ollama signin` e non scarica modelli cloud.
- Windows è fuori scope (degrado se Ollama è già avviato).
