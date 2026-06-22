# Experiment: Piper (VITS) lecturer voice — the *fast* one

Fine-tune a **Piper** voice on the lecturer — entirely on Kaggle, **no local
computation**. Companion to `kaggle_finetune_piper.ipynb`.

## Why Piper

XTTS v2 and F5-TTS are the **quality** voices, but both are too slow for a
realtime phone/Telegram voice bot (XTTS is autoregressive; F5 is multi-step
flow matching). Piper is a **VITS** model: non-autoregressive, RTF ≪ 1, ~tens of
ms to first audio on CPU *or* GPU. It is the realtime tier.

This is **not a replacement** for F5/XTTS — it's the low-latency voice for the
`clone-telegram/voice-agent` call pipeline, which already loads a `PiperVoice`
in `voice-agent/tts.py`. Train this → that bot is plug-in.

Licensing: Piper is MIT/GPL, fine for the bot (unlike XTTS's CPML).

## Key difference vs the F5/XTTS notebooks

Those notebooks start from a raw `lecture.wav` and run whisper + pyannote. **This
one does not.** The dataset already exists on HuggingFace, already diarized to
the single target voice:

- `cyttic/audio-kri-russian`   — 52 clips, 16 kHz  (`SPEAKER_01`)
- `cyttic/audio-kri-russian-2` — 138 clips, 22 kHz (`SPEAKER_03`)

So we **pull both parquets, resample everything to 22.05 kHz mono / 16-bit, and
fine-tune** — no transcription, no diarization. ~15 min of target speech total.

## Storage policy: HuggingFace only

**No Kaggle Datasets (input), no Kaggle output/versioning, no gdrive.** Every
persistent artifact is on HF:

- **data in** ← the two HF datasets above
- **model out** → `HF_MODEL_REPO`: `lecturer_ru.onnx` + `lecturer_ru.onnx.json`
- **resume state** ↔ `resume.ckpt` in that same model repo

`/kaggle/working` is throwaway scratch, rebuilt from HF on every run. **Crash-resume
goes through HF, not Kaggle:** section 2 pulls `resume.ckpt` if it exists (else the
base), and section 6 pushes the voice *and* a fresh `resume.ckpt`. So a dead
session loses nothing — re-run top-to-bottom and it continues. To train more,
raise `MAX_EPOCHS` (absolute — see gotchas) and re-run; each run resumes from HF.

## Training stack: `OHF-Voice/piper1-gpl` (and a hard env conflict)

We use the **maintained Piper** (`OHF-Voice/piper1-gpl`), not the old
`rhasspy/piper`. It folds phonemize/caching into the `fit` command (no separate
preprocess).

**The env conflict that bit twice:** `datasets`/`pandas` (Kaggle's Python-3.12
image) need **numpy 2**; piper1-gpl's training deps pin **numpy <2**. Installing
piper downgrades numpy and then `import datasets` dies with
`numpy.dtype size changed ... Expected 96, got 88`. They cannot coexist in one
kernel. **Resolution = cell order:** pull + prep the data on the stock numpy-2
image first, write wavs to disk, and install piper **last**. After the piper
install `datasets`/`pandas` are broken in-kernel — fine, because training runs in
a `!python` subprocess and the HF upload uses pure-python `huggingface_hub`.
**Do not reorder the cells.**

## Pipeline (one notebook, in order — order is load-bearing)

| # | Step | Tooling | numpy |
|---|------|---------|-------|
| 1 | Pull both HF datasets → `wav/` + `metadata.csv` (`file.wav\|text`), resampled | `datasets` + `soundfile`/`librosa` | 2.x |
| 2 | Start ckpt: **resume from HF `resume.ckpt`**, else base Russian `.ckpt` | `huggingface_hub` | 2.x |
| 3 | **Install** `piper1-gpl[train]` + build ext (espeak-ng, cmake, ninja) | git + pip | →<2 |
| 4 | Fine-tune (`piper.train fit`; phonemize/cache folded in) | piper1-gpl | <2 (subproc) |
| 5 | Export to `.onnx` (+ copy config → `.onnx.json`) | `piper.train.export_onnx` | <2 (subproc) |
| 6 | Push voice **+ fresh `resume.ckpt`** to the HF model repo | `huggingface_hub.HfApi` | n/a |
| 7 | Quick listen | `piper` CLI | n/a |

We fine-tune **from an existing Russian checkpoint** (same principle as F5): the
espeak-ng `ru` phoneme set is already learned, so no training from scratch and
15 min is enough to adapt timbre/prosody. Only **medium**-quality base checkpoints
load without modification.

## Prerequisites

- Kaggle notebook with **GPU** (T4 / P100).
- Kaggle Secret **`HF_TOKEN`** — a HuggingFace **write** token (the datasets are
  private; same token reads them and writes the model repo).
- `espeak-ng` (installed by the notebook via apt).

## Configure (CONFIG cell)

| Variable | Meaning |
|---|---|
| `HF_DATASETS` | list of source dataset repos to combine (both, by default) |
| `HF_MODEL_REPO` | where the voice (`.onnx`) **and** `resume.ckpt` are stored |
| `BASE_REPO` / `BASE_CKPT` | base Russian Piper checkpoint (first run only) |
| `ESPEAK_VOICE` | espeak-ng voice for phonemization (`ru`) |
| `SR` | 22050 — must match the medium base checkpoint |
| `MAX_EPOCHS`, `BATCH` | absolute epoch target / batch (lower batch if OOM) |

## Base checkpoints (Russian)

Piper publishes **training** checkpoints (`.ckpt`, not the released `.onnx`) at
**`rhasspy/piper-checkpoints`** — a **dataset** repo, so `hf_hub_download(...,
repo_type='dataset')` (default `model` → `RepositoryNotFoundError`). Verified RU
medium checkpoints (22.05 kHz):

- `ru/ru_RU/irina/medium/epoch=4139-step=929464.ckpt`  — **female** → default.
- `ru/ru_RU/denis/medium/epoch=4474-step=1521860.ckpt` — male.
- `ru/ru_RU/dmitri/medium/epoch=5589-step=1478840.ckpt` — male.
- `ru/ru_RU/ruslan/medium/epoch=2436-step=1724372.ckpt` — male.

Each dir also has `config.json` + `dataset.jsonl.gz` + `train.sh`. (Checkpoint
filenames are pinned to a commit; re-list if the repo updates.)

Pick the base whose gender/pitch is closest to the target; you're adapting, not
overwriting, so a closer start = fewer steps to a good voice. Confirm the exact
`epoch=*-step=*.ckpt` filename from the repo's file list and put it in `BASE_CKPT`.

## Outputs

All in the HF **model** repo (`HF_MODEL_REPO`):
- `lecturer_ru.onnx` + `lecturer_ru.onnx.json` — everything Piper needs to serve.
- `resume.ckpt` — training checkpoint for resuming / re-exporting later.

The two `.onnx` files drop straight into `voice-agent/` — set `VA_TTS_VOICE` to
the `.onnx` and the call bot speaks in this voice.

## Gotchas (expected, not yet run end-to-end)

- **numpy 2 vs <2 is unavoidable; isolate by cell order, not by pinning.**
  `datasets` needs numpy 2; piper1-gpl's install forces numpy <2. Pull data
  first, install piper last (see "env conflict" above). Both a manual `numpy<2`
  pin *and* installing piper before the data pull reproduce
  `numpy.dtype size changed ... Expected 96, got 88`.
- **Stale checkpoint hparams (`sample_bytes`).** `fit --ckpt_path` replays the
  checkpoint's saved `hyper_parameters` through the parser; old rhasspy checkpoints
  carry keys the current `VitsModel` rejects → *"does not accept option
  'model.sample_bytes' … Parsing of ckpt_path hyperparameters failed!"*. Section 3b
  re-saves the ckpt keeping only hparams in `VitsModel`'s signature. The model is
  still built from the CLI `--model.*` args, so this is purely cosmetic surgery.
- **Epoch counter continues from the base.** `fit --ckpt_path` restores the
  checkpoint's epoch (irina = 4139), so `--trainer.max_epochs` is absolute (set to
  6000 → ~1860 epochs of finetuning). If `fit` exits immediately, raise it.
- **Sample-rate match.** Base `medium` = 22.05 kHz; all clips (incl. the 16 kHz
  dataset) are resampled to `SR`. Mismatched SR vs the checkpoint wrecks quality.
- **Medium base only.** Only medium-quality base checkpoints load without code
  changes — keep `ru/ru_RU/<voice>/medium`.
- **espeak-ng required** at train *and* serve time (`ESPEAK_VOICE='ru'`).
- **espeakbridge build needs cmake ≥ 3.26.** `setup.py build_ext` CMake-downloads
  and compiles espeak-ng (`CMakeLists` requires 3.26); Kaggle's apt cmake is older,
  so pip-install `cmake>=3.26` + `scikit-build` into the runtime env (build_ext runs
  un-isolated). Symptom of skipping this: `ImportError: cannot import name
  'espeakbridge' from 'piper'` at `fit`. Don't pipe the build log to `tail` — the
  cell gates on `from piper import espeakbridge`.
- **Tiny data → overfit/robotic** if trained too long. Watch the sample audio and
  stop early; for ~15 min, a few thousand steps is plenty.

## Status & next steps

- ✅ Notebook written, mirrors the F5/XTTS cloud workflow; starts from the
  existing HF datasets (no whisper/diarize).
- ⏳ Not yet run end-to-end on Kaggle (needs the `HF_TOKEN` secret).
- 🔜 Serving: drop `<voice>.onnx(.json)` into `voice-agent/`, point
  `VA_TTS_VOICE` at it. A `telegram_bot_piper.py` (mirroring `telegram_bot_f5.py`)
  is trivial since `voice-agent/tts.py` already wraps `PiperVoice` — deferred
  until the voice is trained and auditioned.
