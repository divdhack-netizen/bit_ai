# Local closed-corpus agent (LangChain + Ollama)

A terminal assistant that works **only on the documents in the folder it is launched from**: no web search, no outside knowledge. It reads **txt, md, pdf, docx and images** (png, jpg, webp, through Gemma 4's vision), cites the file each piece of information comes from and, if the answer is not in the documents, says so. It writes **only `.txt` files**: source documents are never modified. It uses a ReAct agent, a **direct** path for chat only, and local **Gemma 4** models through Ollama. The runtime is always Ollama; the app detects the host (macOS Apple Silicon / Intel, Linux CPU or NVIDIA) and tunes budgets, `num_gpu`, `num_thread` and thermals. On the **16 GB unified** profile the KV budgets stay at 4096/6144.

The agent talks to the user in Italian; the code, comments and documentation are in English.

## Getting started

Launch it **inside** the documents folder (e.g. `balena/`):

```bash
cd ~/Documents/balena
~/Documents/bit_ai/venv/bin/python ~/Documents/bit_ai/main.py
```

Alternatively, `--folder PATH` points at the folder without changing directory (`--cartella` still works as an alias). Full options:

```bash
./venv/bin/python main.py [--folder PATH] [--verbose/--no-verbose] [--check] [--burn-protection] [--model fast|default] [--thinking/--no-thinking] [--auto-fast/--no-auto-fast] [--prompt "testo"]
```

On the **first run** (and whenever a dependency is missing) a wizard asks for consent to install Ollama (Homebrew on macOS, the official installer on Linux) and to pull the Gemma 4 models. The full procedure is in [STARTUP.MD](STARTUP.MD). With `--prompt` the wizard does not start: if the environment is incomplete the process exits with code 1.

Backward-compatible alias: `python agent.py` (same behaviour as `main.py`).

Ollama endpoint: the `OLLAMA_HOST` variable (a scheme is optional, e.g. `127.0.0.1:11434`). Default `http://127.0.0.1:11434`.

## Layout

```
main.py
assistant/
  host.py                # OS, SoC, RAM/VRAM, Ollama options
  bootstrap.py           # Darwin/Linux installers and daemon startup
  config.py              # aliases derived from the matrix, file config
  inference_profiles.py  # hardware-aware profiles (tag, ctx, output, keep-alive)
  ollama_runtime.py      # Ollama lifecycle: cache, preload, unload, pull
  startup.py             # first-run wizard (Ollama, Gemma 4)
  prompt_budget.py       # token estimates and anti-overflow trimming
  direct_pipeline.py     # chat/explanations in a single generation
  file_service.py        # sandbox on the documents folder, search, atomic .txt writes
  readers.py             # text extraction: txt/md, pdf (pypdf), docx (stdlib), images (vision)
  tools.py               # LangChain file adapters and controlled fallback
  summary.py             # riassumi_documento on the session runtime
  comprehension.py       # local card (rules, zero LLM), paths, folders, destinations
  turn_schema.py         # JSON card via the LLM, only for ambiguous requests
  turn_validator.py      # fixes the card's write, destination and clarifications
  turn_policy.py         # routing: direct (chat) / react (documents)
  date_time.py           # data_oggi
  memory.py              # conversation history within the profile budget
  session.py             # orchestration, tool capabilities, metrics
  metrics.py             # TTFT, load/prefill/decode, rendering, e2e
  thermal.py             # workload-first thermal throttling + hysteresis
  process_log.py         # JSONL crashes, incomplete turns, stalls
  scripts/
    thermal_monitor.swift  # persistent Darwin monitor (also `once` mode)
  cli.py                 # argparse and interactive loop
  console.py             # Rich console, linear streaming and thermal bar
benchmarks/
  benchmark_hot_paths.py   # Python microbenchmarks without Ollama
  benchmark_inference.py   # Ollama/agent baseline (comparable JSON)
  compare_benchmarks.py    # before/after comparison with a verdict
  comprehension_probe.py   # comprehension accuracy on a well-written corpus
  hard_comprehension.py    # comprehension stress test and safety boundaries
```

## Agent tools

Tool names stay in Italian because the model sees them in its (Italian) prompt.

| Tool | Purpose |
|------|---------|
| `cerca_testo` | Searches for words in the content of every document (txt, md, pdf, docx); returns file, line or page, and an excerpt |
| `leggi_documento` | Reads a document: txt/md, pdf (with `[pagina N]`), docx, images (transcription and description via vision) |
| `elenca_cartella` | Lists one level of a folder |
| `elenca_albero` | Lists up to two levels, skipping venv/.git |
| `trova_file` | Searches by name (glob or fragment) |
| `riassumi_documento` | Summarizes a document via Ollama; optionally `path_in` / `---` / `path_out.txt` |
| `scrivi_file` | Atomically creates or overwrites a **`.txt`** file (`path`, `---`, content) |
| `modifica_file` | Exact single-match patch on a **`.txt`** (`path`, `<<<OLD` / `<<<NEW`) |
| `crea_cartella` | Creates a relative folder (`exist_ok` if it already exists) |
| `data_oggi` | The system calendar date (for calculations on dates in the documents) |

There are no network tools. The writable extensions live in `config.WRITABLE_EXTENSIONS`.
Images are read by the **same model** that is already loaded (Gemma 4 is multimodal):
no second model. A PDF without text (a scan) goes through vision on its first
embedded images. `cerca_testo` skips images, which would be too expensive to describe one by one.

### Turn routing

Before each turn the request is understood in this order, from cheapest to most expensive:

1. **greeting** or **history spill** («riportami tutto sul file x.txt»): local rules, no card
2. **local card** (`comprehension.local_card`, ~0.04 ms): decides questions, reads and
   writes with an explicit verb and a single new destination on its own
3. **LLM card** (constrained JSON) only for ambiguous cases: edits to existing files,
   «salvalo» without a destination, deictic follow-ups («e il secondo?»), implicit writes

In both cases the card goes through the deterministic validator. In the log, the
`schema origine=locale|schema|…` line tells which path each turn took. Every call
(preload, card, direct, ReAct, summary, vision) uses a single session `num_ctx`
(`OllamaRuntime.session_context`), so Ollama does not reload the weights between steps.

- **direct**: only greetings and comments on the conversation; one generation, with no facts taken from the weights
- **react**: every content question and every file operation; `cerca_testo` → `leggi_documento` → answer citing the file, or «non presente nei documenti»

The summary reuses the session's runtime and model: it does not implicitly load a second 12B.

## Permissions

The allow/deny rules live in `.assistant/permissions.json` **inside the documents
folder** (the older `.assistente/permissions.json` is still read). If it is missing,
the project's `assistant/permissions.json` applies (deny on `.env`, credentials and
secrets). Deny also applies to reads, listings and `cerca_testo`.

## Error log

Process crashes, turn exceptions, incomplete requests, Ollama stalls and LLM diagnostics are written to `output/logs/` (daily JSONL, `crash-{pid}.txt` dumps, an `ultimi_errori.md` index). In an interactive session: `/errori` or `/errori 20`.

## Host and hardware use

`assistant/host.py` classifies the machine and `OllamaRuntime` applies the options:

| Host | Offload | Threads | Weight pool |
|------|---------|---------|-------------|
| Darwin `arm64` (M1–M4) | `num_gpu=-1` (Metal) | Ollama auto (no P-core pinning) | Unified RAM |
| Darwin Intel | Ollama default | P-cores if detected | RAM |
| Linux NVIDIA | `num_gpu=-1` | auto | VRAM |
| Linux CPU-only | `num_gpu=0` | auto | System RAM |

On Pro/Max/Ultra SoCs **or** a pool ≥ 32 GiB the session `num_ctx` is 8192. On a 16 GB base M chip it is 6144 for every route (avoids reloads when the path changes). With a `q4_0` KV, if the weights and the cache fit in what is left, `fast` and `default` on 16 GB go up to 8192 and already ample machines to 12288; a turn that fills the window can take one more step, up to the context declared by the model.

Burn protection: on macOS it uses Swift + `ProcessInfo` + SMC; the soft thresholds are lower on fanless chips and higher on Pro/Linux. On Linux it reads `/sys/class/thermal`. If the preload lands on the CPU when a GPU was expected, the CLI warns without blocking the turn. In the **Serious/Critical** states burn protection can suspend the native CoT (`thinking disattivato`) to cut the load.

The model's native reasoning (`think`) is **off by default**. Turn it back on with `--thinking` or `/thinking on` (alias `/t`): while streaming, the CoT is a continuous dim flow (`Pensiero` + tokens), then the answer appears. The history does not keep the thinking. In Low Power Mode the agent automatically uses `fast` (E4B) if `--auto-fast` is on (the default), without changing the `/model` preference.

## Benchmarks

Microbenchmark of the Python paths (no network, no Ollama):

```bash
./venv/bin/python -m benchmarks.benchmark_hot_paths
```

Ollama/agent inference benchmark (cold/warm, TTFT, load/prefill/decode, thermals, quality):

```bash
# Planned matrix without calling Ollama
./venv/bin/python -m benchmarks.benchmark_inference --dry-run

# Quick suite (fast + default, quality rubric, 2 repetitions)
./venv/bin/python -m benchmarks.benchmark_inference \
  --alias fast default \
  --case chat coding tool_selection file_write notes_write \
         react_multistep follow_up_storico evidenze_fornite \
  --mode ollama_raw agent_no_ui \
  --repeat 2 --cooldown 5 --think off \
  --output output/iterazioni/iterNN.json

# Before/after comparison with an ACCETTA/RIFIUTA verdict
./venv/bin/python -m benchmarks.compare_benchmarks \
  output/iterazioni/prima.json output/iterazioni/dopo.json --markdown
```

Comprehension benchmarks (the `--without-llm` run only executes the deterministic probes):

```bash
./venv/bin/python -m benchmarks.comprehension_probe --alias default
./venv/bin/python -m benchmarks.hard_comprehension --alias default --generation
./venv/bin/python -m benchmarks.hard_comprehension --without-llm
```

Aliases without enough headroom (e.g. `fast` on very tight machines) are skipped.
The log of optimization iterations lives in `output/iterazioni/REGISTRO.md`
(a local, unversioned folder: `output/` is in `.gitignore`).
Gemma 4 non-thinking sampler: `temperature=0.3`, `top_k=64`, `top_p=0.95`,
`repeat_penalty=1.0`, `num_batch=512`. `num_predict` is per call and does not
invalidate the Ollama client.

## Model profiles and quantization

**Fixed** matrix (no `--quant` / `/quant` flag): each alias has a single artifact.

| Alias | Model | Quant | Effective tag | Weights ~ |
|-------|-------|-------|---------------|-----------|
| `fast` | E4B | q4 | `gemma4:e4b` | ~6.6 GB |
| `default` | 12B | q4 | `gemma4:12b` | ~8.0 GB |

Typical route budgets on the 16 GB profile (session `num_ctx` = 6144):

| Route | Session `num_ctx` | `num_predict` |
|-------|-------------------|---------------|
| direct / summary | 6144 | 640 / 512 |
| react | 6144 | 1024 |
| react (Pro/Max/Ultra or ≥32 GiB) | 8192 | 1536 |

Example pulls:

```bash
ollama pull gemma4:e4b
ollama pull gemma4:12b
```

The **KV cache** quantization (`OLLAMA_KV_CACHE_TYPE`) is separate from the model weights.

## Runtime commands

- `/help` (or `/?`): status, slash commands and keyboard guide
- `/verbose`, `/check`, `/burn_protection` (or `/bp`), `/thinking` (or `/t`): `[on|off|toggle]`
- `/model [fast|default]`
- `/clear` (or `/reset`, `/nuova`): clears the conversation history
- `/errori`: latest process errors
- `exit` / `quit` / `esci`

With `/verbose` (or `--verbose`) the bar adds TTFT, prefill and the other latencies; by default it shows tok/s, total tokens (in+out) and context percentage, plus power and thermal state.

Ctrl+C at the prompt always interrupts; with an explicit selection it copies. During a generation it cooperatively cancels the turn and returns to the prompt.

## Ollama (daemon env)

The client applies keep-alive and `num_gpu` to every generation. On Apple Silicon
with Metal, `num_thread` is omitted (Ollama auto); on Darwin without a GPU it is pinned to
the P-cores. The **server** variables (flash attention, KV type, loaded models)
are set **only** if the wizard starts `ollama serve` as a child process.

If Ollama already runs as a service (`brew services` / systemd), configure them on the daemon and restart it. macOS example:

```bash
launchctl setenv OLLAMA_FLASH_ATTENTION 1
launchctl setenv OLLAMA_KV_CACHE_TYPE q4_0
launchctl setenv OLLAMA_KEEP_ALIVE 30m
launchctl setenv OLLAMA_MAX_LOADED_MODELS 1
launchctl setenv OLLAMA_NUM_PARALLEL 1
```

`Ollama.app` as a login item (SMAppService) starts a `serve` **without** these env vars: flash attention stays false and the KV is f16. Turn off the app's launch at login and let the wizard spawn `ollama serve`, or run `launchctl setenv` **before** the app starts and restart Ollama. At startup the client reports whether FA or `NUM_PARALLEL` do not match.

On Linux the same keys go into the systemd unit file or the environment of the `ollama serve` process. The client uses a `30m` keep-alive during the session (the weights are unloaded on exit).

### Quick rollback

1. Go back to the light profile on 16 GB: `--model fast` or `/model fast` (E4B q4).
2. Turn off burn protection for diagnosis only: `--no-burn-protection`.
3. Compare before/after with `benchmark_inference.py` on the same `--case` set.
