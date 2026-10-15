# First-run guided setup

This document is the spec of the wizard run by `main.py`.
The code in `assistant/startup.py` and `assistant/bootstrap.py` must follow it;
when in doubt, this file wins.

No download or installation happens without **explicit consent** (yes).
A refusal (no, empty Enter, Ctrl+C) stops startup with a non-zero exit code.
There is no on-disk marker that hides missing dependencies: every launch **re-checks** the real state.

The wizard's prompts and messages are in Italian; quoted labels below are shown as the user sees them.

## 0. Prerequisites (outside the wizard)

The wizard does **not** install Python or create the virtualenv.

- macOS (every Apple Silicon SoC; Intel gets a warning and continues) **or** Linux
- Python 3.12 or later
- cloned repository and dependencies:

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
cd ~/Documents/balena   # the documents folder
~/Documents/bit_ai/venv/bin/python ~/Documents/bit_ai/main.py
```

The agent works only on the documents in the launch folder (or in `--folder`).

Windows is not a target: if Ollama is already listening the agent can run in degraded mode
(no wizard, no native burn protection). Otherwise the process exits with code 1.

## 1. Interactive and non-interactive mode

At startup the agent detects, in this order:

1. The Ollama CLI (`ollama` on `PATH`; on macOS also in Homebrew's `bin/` if brew is there)
2. The Ollama API (`GET` on the URL from `OLLAMA_HOST`, default `http://127.0.0.1:11434/api/version`)
3. At least one Gemma 4 artifact from the matrix (`fast` / `default`, including the tag fallbacks in `inference_profiles.py`)

Homebrew is required **only on macOS and only if Ollama is missing**. On Linux it is not a prerequisite,
even if Linuxbrew is installed.

**Environment ready:** Ollama CLI+API and at least one Gemma 4 from the matrix. The wizard does not start.
Detection and model prewarm start **together** with the HUD splash
(if the session is interactive). The welcome panel appears once the preload is done
(or once the splash is done, if the load has already finished).

**Something is missing and stdin is a TTY** (no `--prompt`): the step-by-step wizard starts, only for the missing pieces.

**Something is missing and the session is not interactive** (`--prompt` and/or a non-TTY stdin): **no wizard**. A message pointing to this file and exit code `1`.

## 2. macOS: Homebrew (only if Ollama is missing)

Detection:

- `which brew`
- Apple Silicon path: `/opt/homebrew/bin/brew`
- Intel path: `/usr/local/bin/brew`

If the binary exists but is not on the process `PATH`, the wizard prepends it to `PATH` and applies `brew shellenv` (the `HOMEBREW_*` and `PATH` variables).

If it is missing **and** Ollama is missing:

1. Explain that Homebrew is needed and that the official installer will ask for the **administrator password**.
2. Ask for consent. Refusal → stop.
3. Run in the foreground, with the terminal's stdin/stdout/stderr:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

The Python process does **not** wrap `sudo` and does **not** set a timeout: the installer is interactive.

4. After exit `0`, apply `brew shellenv` to the current process. If `brew` is still missing → error and stop.
5. Suggest (without running it) adding to `~/.zprofile`:

```bash
eval "$(/opt/homebrew/bin/brew shellenv)"
```

If `curl` or the network fails: a message pointing to https://brew.sh, no automatic retry.

## 3. Ollama

Detection: the `ollama` CLI **and** a listening API.

### 3.1 macOS

If the CLI is missing:

1. Ask for consent to `brew install ollama`. Refusal → stop.
2. Run `brew install ollama`.
3. Start the service: `brew services start ollama`.
4. Wait for the API for about 20 seconds.
5. If the port does not answer: start `ollama serve` in the background (with the server env: flash attention, `q4_0` KV, one model, `num_parallel=1`, `30m` keep-alive) and wait again.
6. If the API stays unreachable → error and stop.

If the CLI is there but the API is down: do **not** reinstall. Only start the service / `ollama serve` and run the healthcheck.

No `launchctl setenv` is applied to an already running daemon. If Ollama was already a service, the client prints a warning: the env vars stay optional, as described in the README.

The `.dmg` / `.app` installer from the Ollama website is not used.

### 3.2 Linux

If the CLI is missing:

1. Ask for consent to the official installer `https://ollama.com/install.sh`. Refusal → stop.
2. Run in the foreground:

```bash
/bin/sh -c "$(curl -fsSL https://ollama.com/install.sh)"
```

3. Try `systemctl --user start ollama`, then `systemctl start ollama`.
4. If the API does not answer: `ollama serve` in the background with the same server env as on macOS.
5. API timeout → error and stop.

If the CLI is there but the API is down: only start systemd / `serve`, without reinstalling.

## 4. Gemma 4 models

A single family, everything on-device (no `:cloud` tags). Fixed matrix (declared weights, KV cache excluded):

| Alias    | Ollama tag           | Quant | Weights ~ |
|----------|----------------------|-------|-----------|
| `fast`   | `gemma4:e4b`         | q4    | 6.6 GB    |
| `default`| `gemma4:12b`         | q4    | 8.0 GB    |

Catalog fallbacks (as in `TAG_INSTALL_CANDIDATES`): `gemma4:e4b-it-q4_K_M` and `gemma4` for fast; `gemma4:12b-it-q4_K_M` for default. The `:latest` suffix is normalized to `gemma4` (which points to E4B in the library).

Tiers based on the inference pool (unified RAM on Apple; VRAM on Linux NVIDIA) and on the headroom already used by `fits_memory`:

- **sconsigliato** (not recommended): the weights (plus 6 GB of headroom on unified memory) do not fit.
- **stretto** (tight): it fits, but what is left after the weights is under 8 GB (6 GB of headroom + 2 GB of margin).
- **consigliato** (recommended): otherwise. On 16 GB: `fast` and `default`. On 8 GB: both not recommended. On 64 GB: both recommended.

RAM/VRAM that cannot be detected: no penalizing tier (pulling is allowed).

The wizard lists the two rows (index, alias, tag, GB, tier, already present or not). It highlights a suggestion based on memory, **without** preselecting it. If `default` is recommended (typical on 16 GB and above) it is the highlighted one.

The user picks which to pull: indexes (`1 2`), aliases (`fast default`), just one or both. Empty Enter = none.
An unreadable selection is asked again (up to 5 attempts); it does not abort the wizard right away.

- No choice and **no** Gemma 4 from the matrix already in Ollama → stop (code `1`).
- No choice but **at least one** is already installed → warning and continue.
- Choices: streaming `POST /api/pull` to Ollama; progress in the console; after each pull the catalog is invalidated.

## 4.1 Vision for images

The Gemma 4 models in the matrix are multimodal: the images in the folder (png, jpg,
webp) are read by the same model that is already loaded. No additional model
needs to be downloaded.

## 5. Finish and session

When the Ollama API and at least one Gemma 4 from the matrix are present:

1. If `--model` / the session alias is not among the installed ones, the first installed alias of the matrix is used (`fast`, then `default`) and the user is told (`/model` to change it).
2. The startup thread preloads the weights in parallel with the splash (probe + `OllamaRuntime.preload`). The agent session opens once the HUD is done, with `num_gpu` / `num_thread` derived from the host (Metal on Apple Silicon, CUDA if there is NVIDIA, CPU-only otherwise). Outcome messages stay on the main thread.
3. If the load lasts longer than the animation, the CLI waits for the preload to finish before the «Assistente pronto» panel.

## 6. Later launches

Every start repeats the detection of step 1. A stopped Ollama or deleted models restart only the missing steps.

There is no "setup completed" file that skips the checks.

## 7. Limits

- The wizard does not bypass `sudo` or fake the password.
- It does not install Python, pip or the venv.
- It does not configure `launchctl` for an Ollama daemon that is already running.
- It does not offer families other than Gemma 4.
- It does not run `ollama signin` and does not download cloud models.
- Windows is out of scope (degraded mode if Ollama is already running).
