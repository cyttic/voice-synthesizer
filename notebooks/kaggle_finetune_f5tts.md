# Experiment: F5-TTS lecturer voice (all-cloud)

Fine-tune a second voice model on a new Russian lecture — entirely on Kaggle,
**no local computation**. Companion to `kaggle_finetune_f5tts.ipynb`.

## Why F5-TTS

The first model is **XTTS v2**, licensed under CPML (non-commercial, R&D only).
This experiment switches to **F5-TTS** (MIT) so the result is free to use.

To avoid extending the base ZH/EN vocab for Cyrillic, we fine-tune **from an
existing Russian F5-TTS checkpoint** whose vocab already covers Russian.

## Pipeline (one notebook, in order)

| # | Step | Tooling |
|---|------|---------|
| 1 | Download `lecture.wav` from Google Drive → 24 kHz mono | `gdown` + ffmpeg |
| 2 | Transcribe | `openai-whisper` large-v3 (CUDA) |
| 2 | Diarize, keep lecturer's clean clips | `pyannote/speaker-diarization-3.1` |
| 3 | Build Parquet → push to HF dataset repo | `datasets.push_to_hub` |
| 4 | Download base Russian checkpoint | `huggingface_hub.hf_hub_download` |
| 5 | Prepare data + fine-tune | F5-TTS `prepare_csv_wavs.py` + `finetune_cli.py` |
| 6 | Upload fine-tuned model to HF model repo | `huggingface_hub.HfApi` |

Clips are cut at pyannote speaker-turn boundaries (single-speaker by
construction), filtered by duration (3–15 s) and Whisper confidence
(`avg_logprob ≥ -0.5`, `no_speech_prob ≤ 0.3`), and the speaker with the most
total speech time is taken as the lecturer.

## Prerequisites

- Kaggle notebook with **GPU** accelerator (T4 / P100).
- Kaggle Secret **`HF_TOKEN`** — a HuggingFace **write** token.
- Accept the [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)
  license with that same HF account.

## Configure (CONFIG cell)

| Variable | Meaning |
|---|---|
| `GDRIVE_ID` | file id of `data/lecture.wav` on Google Drive (share → copy link → id) |
| `HF_DATASET_REPO` | where the Parquet dataset is pushed |
| `HF_MODEL_REPO` | where the fine-tuned model is uploaded |
| `PRETRAIN_REPO` / `PRETRAIN_CKPT` / `PRETRAIN_VOCAB` | base Russian checkpoint |
| `EXP_NAME` | F5-TTS architecture — must match the checkpoint |
| `WHISPER_MODEL`, `SR`, `MIN_DUR`/`MAX_DUR`, filters | prep params |

## Base checkpoints (Russian)

- `Misha24-10/F5-TTS_RUSSIAN` — default; `model_last_inference.safetensors` + `vocab.txt`.
- `hotstone228/F5-TTS-Russian` — alternative; trained on Common Voice 17.

Both use the `char` tokenizer with Cyrillic in the vocab.

## Outputs

- HF **dataset** repo: the 24 kHz mono Parquet (`audio`, `text`, `duration`).
- HF **model** repo: `model_last.pt` + `vocab.txt`.
- Fine-tune checkpoints during the run: `<F5_ROOT>/ckpts/lecturer_ru/`.

## Gotchas

- **Disk-full hang**: /kaggle/working caps at ~20 GB; hoarding checkpoints
  (~2.5 GB each) fills it and training freezes mid-epoch with the GPU idle and
  disk I/O pegged. Mitigated: `--keep_last_n_checkpoints 1`, 45 epochs,
  dataloader workers capped at 4 (Kaggle has 4 CPUs), and a § 3b fast-restart
  cell that reloads the pushed dataset from HF instead of redoing whisper +
  diarization. `model_last.pt` (every 500 updates) auto-resumes on relaunch.
- **Dataset dir suffix must match the tokenizer flag.** `finetune_cli.py` loads
  `data/<DATASET_NAME>_<tokenizer>`; we train with `--tokenizer custom`, so the
  prepared data goes in `..._custom` (not `..._char`).
- **`prepare_csv_wavs.py` arg style drifted across versions.** Current main takes
  the **metadata.csv path** (used here); older builds want the **directory** —
  swap `{meta}` → `{WORK}` if it errors.
- **state_dict size mismatch on checkpoint load** → `EXP_NAME` doesn't match the
  checkpoint architecture. Switch `F5TTS_v1_Base` ↔ `F5TTS_Base`.
- **CUDA OOM on T4** → lower `--batch_size_per_gpu` (e.g. 1200) and raise
  `--grad_accumulation_steps`.
- **torchcodec is unreliable** (CUDA mismatch) → all audio is loaded/decoded via
  `soundfile`, never the `datasets` Audio decoder or torchaudio's default loader.
- For ~30–60 min of clean speech, a few thousand updates is usually enough.
  Watch the `--log_samples` audio and stop early to avoid overfitting.

## Status & next steps

- ✅ Notebook written and JSON-valid; flags checked against current F5-TTS source.
- ⏳ Not yet run end-to-end on Kaggle (needs the `HF_TOKEN` secret + a source URL).
- 🔜 Serving: `inference/telegram_bot.py` loads **XTTS**, not F5-TTS. Playing this
  model through the bot needs an F5-TTS inference adapter (F5-TTS conditions on a
  reference wav + its transcript). Deferred.
