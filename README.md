# Agente closed-corpus locale (LangChain + Ollama)

Assistente da terminale che lavora **solo sui documenti della cartella da cui viene lanciato**: niente ricerca web, nessuna conoscenza esterna. Legge **txt, md, pdf, docx e immagini** (png, jpg, webp, tramite la visione di Gemma 4), cita il file da cui prende ogni informazione e, se la risposta non è nei documenti, lo dice. Scrive **solo file `.txt`**: i documenti sorgente non vengono mai modificati. Usa un agente ReAct, un percorso **direct** solo per la chat e modelli **Gemma 4** locali via Ollama. Il runtime resta Ollama; l’app rileva il host (macOS Apple Silicon / Intel, Linux CPU o NVIDIA) e regola budget, `num_gpu`, `num_thread` e termica. Sul profilo **16 GB unificato** i budget KV restano 4096/6144.

## Avvio

Si lancia **dentro** la cartella dei documenti (es. `balena/`):

```bash
cd ~/Documents/balena
~/Documents/bit_ai/venv/bin/python ~/Documents/bit_ai/main.py
```

In alternativa `--cartella PATH` indica la cartella senza spostarsi. Opzioni complete:

```bash
./venv/bin/python main.py [--cartella PATH] [--verbose/--no-verbose] [--check] [--burn-protection] [--model fast|default] [--thinking/--no-thinking] [--auto-fast/--no-auto-fast] [--prompt "testo"]
```

Al **primo avvio** (e ogni volta che manca una dipendenza) un wizard chiede il consenso per installare Ollama (Homebrew su macOS, installer ufficiale su Linux) e per il pull dei Gemma 4. La procedura completa è in [STARTUP.MD](STARTUP.MD). Con `--prompt` il wizard non parte: se l'ambiente è incompleto il processo esce con codice 1.

Alias retrocompatibile: `python agent.py` (stesso comportamento di `main.py`).

Endpoint Ollama: variabile `OLLAMA_HOST` (anche senza schema, es. `127.0.0.1:11434`). Default `http://127.0.0.1:11434`.

## Struttura

```
main.py
assistente/
  host.py                # OS, SoC, RAM/VRAM, opzioni Ollama
  bootstrap.py           # installer Darwin/Linux e avvio daemon
  config.py              # alias derivati dalla matrice e config file
  inference_profiles.py  # profili hardware-aware (tag, ctx, output, keep-alive)
  ollama_runtime.py      # lifecycle Ollama: cache, preload, unload, pull
  startup.py             # wizard primo avvio (Ollama, Gemma 4)
  prompt_budget.py       # stima token e trimming anti-overflow
  direct_pipeline.py     # chat/spiegazioni a una sola generazione
  file_service.py        # sandbox sulla cartella documenti, ricerca, scritture .txt atomiche
  lettori.py             # estrazione testo: txt/md, pdf (pypdf), docx (stdlib), immagini (vision)
  tools.py               # adapter LangChain file e fallback controllato
  riassunto.py           # riassumi_documento sul runtime della sessione
  comprensione.py        # scheda locale (regole, zero LLM), path, cartelle, destinazioni
  schema_turno.py        # scheda JSON via LLM, solo per le richieste ambigue
  validatore_turno.py    # corregge scrittura, destinazione e chiarimenti della scheda
  politica_turno.py      # routing direct (chat) / react (documenti)
  data_tempo.py          # data_oggi
  memoria.py             # storico conversazione entro budget profilo
  session.py             # orchestrazione, capability tool, metriche
  metriche.py            # TTFT, load/prefill/decode, rendering, e2e
  thermal.py             # throttling termico workload-first + isteresi
  registro_processo.py   # JSONL crash, turni incompleti, stall
  scripts/
    thermal_monitor.swift  # monitor persistente Darwin (anche mode once)
  cli.py                 # argparse e loop interattivo
  console.py             # console Rich, streaming lineare e barra termica
benchmarks/
  benchmark_hot_paths.py   # microbench Python senza Ollama
  benchmark_inference.py   # baseline Ollama/agente (JSON confrontabile)
```

## Tool agente

| Tool | Uso |
|------|-----|
| `cerca_testo` | Cerca parole nel contenuto di tutti i documenti (txt, md, pdf, docx); restituisce file, riga o pagina ed estratto |
| `leggi_documento` | Legge un documento: txt/md, pdf (con `[pagina N]`), docx, immagini (trascrizione e descrizione via vision) |
| `elenca_cartella` | Elenca un livello di una cartella |
| `elenca_albero` | Elenca fino a due livelli, saltando venv/.git |
| `trova_file` | Cerca per nome (glob o frammento) |
| `riassumi_documento` | Riassume un documento via Ollama; opzionale `path_in` / `---` / `path_out.txt` |
| `scrivi_file` | Crea o sovrascrive atomicamente un file **`.txt`** (`path`, `---`, contenuto) |
| `modifica_file` | Patch esatta a match unico su un **`.txt`** (`path`, `<<<OLD` / `<<<NEW`) |
| `crea_cartella` | Crea una cartella relativa (`exist_ok` se già presente) |
| `data_oggi` | Data di calendario del sistema (per calcoli su date dei documenti) |

Nessun tool di rete. Le estensioni scrivibili stanno in `config.ESTENSIONI_SCRIVIBILI`.
Le immagini vengono lette dallo **stesso modello** già caricato (Gemma 4 è multimodale):
nessun secondo modello. Un PDF senza testo (scansione) passa dalla vision sulle prime
immagini incorporate. `cerca_testo` esclude le immagini, troppo costose da descrivere tutte.

### Routing dei turni

Prima di ogni turno la richiesta viene compresa così, dal più economico al più costoso:

1. **saluto** o **riversamento dello storico** («riportami tutto sul file x.txt»): regole locali, nessuna scheda
2. **scheda locale** (`comprensione.scheda_locale`, ~0,04 ms): decide da sola domande, letture
   e scritture con un verbo esplicito e una sola destinazione nuova
3. **scheda LLM** (JSON vincolato) solo per i casi ambigui: modifiche di file esistenti,
   «salvalo» senza destinazione, follow-up deittici («e il secondo?»), scritture implicite

In entrambi i casi la scheda passa dal validatore deterministico. Nel registro la riga
`schema origine=locale|schema|…` dice quale strada ha preso ogni turno. Tutte le chiamate
(preload, scheda, direct, ReAct, riassunto, vision) usano un unico `num_ctx` di sessione
(`OllamaRuntime.contesto_sessione`), così Ollama non ricarica i pesi fra un passo e l'altro.

- **direct** — solo saluti e commenti sulla conversazione: una generazione, senza fatti presi dai pesi
- **react** — ogni domanda di contenuto e ogni operazione su file: `cerca_testo` → `leggi_documento` → risposta con citazione del file, oppure «non presente nei documenti»

Il riassunto riusa il runtime/modello della sessione: non carica un secondo 12B implicito.

## Permessi

Le regole allow/deny stanno in `.assistente/permissions.json` **dentro la cartella
documenti**; se manca, vale `assistente/permissions.json` del progetto (deny su `.env`,
credenziali e segreti). La deny vale anche per letture, elenchi e `cerca_testo`.

## Registro errori

Crash di processo, eccezioni di turno, richieste incomplete, stall Ollama e diagnostiche LLM vengono scritti in `output/logs/` (JSONL giornaliero, dump `crash-{pid}.txt`, indice `ultimi_errori.md`). In sessione interattiva: `/errori` oppure `/errori 20`.

## Host e sfruttamento hardware

`assistente/host.py` classifica la macchina e `OllamaRuntime` applica le opzioni:

| Host | Offload | Thread | Pool pesi |
|------|---------|--------|-----------|
| Darwin `arm64` (M1–M4) | `num_gpu=-1` (Metal) | auto Ollama (non pin P-core) | RAM unificata |
| Darwin Intel | default Ollama | P-core se rilevati | RAM |
| Linux NVIDIA | `num_gpu=-1` | auto | VRAM |
| Linux CPU-only | `num_gpu=0` | auto | RAM di sistema |

Su SoC Pro/Max/Ultra **oppure** pool ≥ 32 GiB il `num_ctx` di sessione è 8192. Su 16 GB M-base è 6144 per tutte le route (evita reload a cambio percorso). Con KV `q4_0`, se pesi e cache stanno nel residuo, `fast` e `default` su 16 GB salgono a 8192 e le macchine già ampie a 12288; un turno che riempie la finestra può fare un altro scalino, fino al contesto dichiarato dal modello.

Burn protection: su macOS resta Swift + `ProcessInfo` + SMC; soglie soft più basse sui chip fanless, più alte su Pro/Linux. Su Linux si legge `/sys/class/thermal`. Se il preload finisce sulla CPU mentre si aspettava GPU, la CLI avvisa senza bloccare il turno. In stato **Serious/Critical** la burn protection può sospendere il CoT nativo (`thinking disattivato`) per tagliare il carico.

Il ragionamento nativo del modello (`think`) è **spento di default**. Si riaccende con `--thinking` o `/thinking on` (alias `/t`): in streaming il CoT è un flusso continuo dim (`Pensiero` + token), poi compare la risposta. Lo storico non salva il pensiero. In Low Power Mode l'agente usa automaticamente `fast` (E4B) se `--auto-fast` è attivo (default), senza cambiare la preferenza `/model`.

## Benchmark

Micro-benchmark dei percorsi Python (senza rete né Ollama):

```bash
./venv/bin/python -m benchmarks.benchmark_hot_paths
```

Benchmark di inferenza Ollama/agente (cold/warm, TTFT, load/prefill/decode, termica, qualità):

```bash
# Matrice pianificata senza chiamare Ollama
./venv/bin/python -m benchmarks.benchmark_inference --dry-run

# Suite rapida (fast + default, rubrica qualità, 2 ripetizioni)
./venv/bin/python -m benchmarks.benchmark_inference \
  --alias fast default \
  --case chat coding tool_selection file_write notes_write \
         react_multistep follow_up_storico evidenze_fornite \
  --mode ollama_raw agent_no_ui \
  --repeat 2 --cooldown 5 --think off \
  --output output/iterazioni/iterNN.json

# Confronto prima/dopo con verdetto ACCETTA/RIFIUTA
./venv/bin/python -m benchmarks.compare_benchmarks \
  output/iterazioni/prima.json output/iterazioni/dopo.json --markdown
```

Gli alias fuori headroom (es. `fast` su macchine molto strette) vengono saltati.
Il registro delle iterazioni di ottimizzazione sta in `output/iterazioni/REGISTRO.md`
(cartella locale, non versionata: `output/` è in `.gitignore`).
Sampler Gemma 4 non-thinking: `temperature=0.3`, `top_k=64`, `top_p=0.95`,
`repeat_penalty=1.0`, `num_batch=512`. `num_predict` è per-chiamata e non
invalida il client Ollama.

## Profili modello e quantizzazione

Matrice **fissa** (nessun flag `--quant` / `/quant`): ogni alias ha un solo artefatto.

| Alias | Modello | Quant | Tag effettivo | Pesi ~ |
|-------|---------|-------|---------------|--------|
| `fast` | E4B | q4 | `gemma4:e4b` | ~6.6 GB |
| `default` | 12B | q4 | `gemma4:12b` | ~8.0 GB |

Budget di route tipici sul profilo 16 GB (`num_ctx` di sessione = 6144):

| Route | `num_ctx` sessione | `num_predict` |
|-------|--------------------|---------------|
| direct / summary | 6144 | 640 / 512 |
| react | 6144 | 1024 |
| react (Pro/Max/Ultra o ≥32 GiB) | 8192 | 1536 |

Pull esempio:

```bash
ollama pull gemma4:e4b
ollama pull gemma4:12b
```

La quantizzazione della **KV cache** (`OLLAMA_KV_CACHE_TYPE`) è separata dai pesi del modello.

## Comandi runtime

- `/help` (o `/?`) — stato, slash command e guida tastiera
- `/verbose`, `/check`, `/burn_protection` (o `/bp`), `/thinking` (o `/t`) — `[on|off|toggle]`
- `/model [fast|default]`
- `/clear` (o `/reset`, `/nuova`) — azzera lo storico conversazione
- `/errori` — ultimi errori di processo
- `exit` / `quit` / `esci`

Con `/verbose` (o `--verbose`) la barra aggiunge TTFT, prefill e le altre latenze; di default restano tok/s, token totali (in+out) e percentuale di contesto, più alimentazione e termica.

Ctrl+C al prompt interrompe sempre; con una selezione esplicita copia. Durante una generazione annulla il turno in modo cooperativo e riporta al prompt.

## Ollama (env del daemon)

Il client applica keep-alive e `num_gpu` su ogni generazione. Su Apple Silicon
con Metal `num_thread` è omesso (auto Ollama); su Darwin senza GPU si pinna ai
P-core. Le variabili **server** (flash attention, tipo KV, modelli caricati)
vengono impostate **solo** se il wizard avvia `ollama serve` nel processo figlio.

Se Ollama è già un servizio (`brew services` / systemd), configurale sul daemon e riavvialo. Esempio macOS:

```bash
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q4_0
launchctl setenv OLLAMA_KEEP_ALIVE 30m
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
launchctl setenv OLLAMA_NUM_PARALLEL 1
```

`Ollama.app` come login item (SMAppService) avvia un `serve` **senza** queste env: flash attention resta false e la KV è f16. Disattiva l'avvio al login dell'app e lascia al wizard lo spawn di `ollama serve`, oppure imposta `launchctl setenv` **prima** che l'app parta e riavvia Ollama. Il client segnala all'avvio se FA o `NUM_PARALLEL` non coincidono.

Su Linux le stesse chiavi vanno nel unit file systemd o nell’ambiente del processo `ollama serve`. Il client usa keep-alive `30m` in sessione (i pesi si scaricano all'uscita).

### Rollback rapido

1. Torna al profilo leggero su 16 GB: `--model fast` oppure `/model fast` (E4B q4).
2. Disattiva burn protection solo per diagnosi: `--no-burn-protection`.
3. Confronta prima/dopo con `benchmark_inference.py` sullo stesso set di `--case`.
