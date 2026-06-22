# Experiment: RVC v2 realtime voice model (all-cloud)

Train a **real-time voice-conversion** model on Kaggle. Companion to
`kaggle_finetune_rvc.ipynb`. This is a different task from the TTS notebooks:

- **F5-TTS / XTTS** turn *text* into speech — useless for a live call.
- **RVC v2** converts *speech to speech* (keep words + prosody, swap identity).
  Its `.pth` + `.index` output is exactly what a realtime engine like
  `w-okada/voice-changer` loads. This is the right family for a VoIP voice changer.

## Why this is cheap

The dataset is **already built** — we reuse `cyttic/lecturer-ru-dataset` (the
clean single-speaker clips the F5 pipeline produced). RVC needs **no transcripts**,
just clean audio of one speaker, so the whole transcribe/diarize stage is skipped.

## Pipeline (one notebook, in order)

| # | Step | Tooling |
|---|------|---------|
| 0 | Clone RVC + download pretrained assets | RVC-WebUI + `lj1995/VoiceConversionWebUI` |
| 1 | Pull dataset from HF -> wav folder | `datasets` (raw bytes via soundfile) |
| 2 | Preprocess (slice 3 s, resample 40 k) | `preprocess.py` |
| 3 | Pitch (RMVPE) + content features (ContentVec v2) | `extract_f0_rmvpe.py`, `extract_feature_print.py` |
| 4 | Build `filelist.txt` + `config.json` | replicates the WebUI "Train" button |
| 5 | Train from v2 40k pretrained G/D | `train.py` |
| 6 | Build faiss retrieval `.index` | `faiss` |
| 7 | Upload `.pth` + `.index` to HF | `huggingface_hub` |

## Model choice rationale

- **40k, v2** is the realtime sweet spot: low latency, 768-dim ContentVec features,
  good quality. 48k is higher fidelity but heavier; 32k is lighter but rougher.
- **RMVPE** pitch extractor: fast + accurate, the realtime-friendly default
  (Harvest is higher quality but far too slow; CREPE-tiny is fast but noisier).

## Prerequisites

- Kaggle notebook with **GPU** (T4 / P100).
- Kaggle Secret **`HF_TOKEN`** — a HuggingFace **write** token.

## Outputs

- HF **model** repo `cyttic/lecturer-ru-rvc`: `lecturer_ru.pth` + `lecturer_ru.index`.
- During the run: checkpoints in `logs/lecturer_ru/`, inference weight in
  `assets/weights/lecturer_ru.pth`.

## Gotchas

- **fairseq won't build on Kaggle's Python 3.12.** RVC pins `fairseq==0.12.2`,
  which fails to compile. The install cell strips it from `requirements.txt` and
  installs the `One-sixth/fairseq` fork instead. That fork's `setup.py` imports
  torch/cython at metadata-generation time, so under pip's **build isolation**
  (a clean env without them) it dies with `egg_info did not run successfully` /
  `metadata-generation-failed`. Fix, already applied: install it with
  `--no-build-isolation` (plus `cython omegaconf`) so it sees Kaggle's torch.
  If it still fights, the fallback is the **Applio** RVC distribution (cleaner
  headless installs, no fairseq at all).
- **Don't let requirements reinstall torch.** The install cell also strips
  `torch/torchaudio/torchvision` so Kaggle's CUDA build is preserved (same caution
  as the F5 notebook — a clobbered torch breaks everything downstream).
- **`pip install -r requirements.txt` is all-or-nothing.** If any single package
  fails metadata generation, pip aborts the whole file and installs *none* of it,
  so RVC's runtime deps go missing and you hit `ModuleNotFoundError` one at a time
  (`ffmpeg` -> `parselmouth` -> ...). The install cell therefore also installs the
  runtime set explicitly: `ffmpeg-python praat-parselmouth pyworld torchcrepe
  faiss-cpu tensorboardX`. Symptom of a half-done preprocess: `logs/<name>/` has
  no `0_gt_wavs` / `1_16k_wavs`.
- **Extractor arg order drifts across RVC versions.** If `extract_f0_rmvpe.py` or
  `extract_feature_print.py` errors on args, the current WebUI builds them as
  `extract_f0_rmvpe.py {n_part} {i_part} {i_gpu} {exp_dir} {is_half}` and
  `extract_feature_print.py {device} {n_part} {i_part} {i_gpu} {exp_dir} {version} {is_half}`.
  Check `infer/modules/train/` in the cloned repo if the paths moved.
- **`No module named 'av'`** — this RVC build decodes audio with PyAV, not just
  ffmpeg-python. `av` is in the explicit install list.
- **`Weights only load failed ... fairseq.data.dictionary.Dictionary`** — torch>=2.6
  made `torch.load` default to `weights_only=True`, which rejects fairseq's hubert
  checkpoint and RVC's pretrained G/D. The install cell sets
  `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` in `os.environ` (inherited by every `!python`
  subprocess, including training) to flip it back.
- **`'FigureCanvasAgg' object has no attribute 'tostring_rgb'`** — matplotlib>=3.8
  removed `tostring_rgb()`, which RVC's tensorboard spectrogram logging calls; it
  crashes the training subprocess at the first log step (after epoch 1 prints
  losses). The train cell patches `infer/lib/train/utils.py` to use `buffer_rgba()`
  before launching. Crash happens before any checkpoint saves, so a re-run is clean.
- **CUDA OOM on T4** -> lower `BATCH` (6, then 4) in CONFIG.
- **Output sounds noisy/robotic** -> usually too little clean audio for the
  speaker, or pitch mismatch at inference (set a pitch shift in the realtime app).
- RVC epochs are cheap (seconds each for ~300 short clips); 100-300 epochs is
  plenty. Watch for overfitting past that (it starts to reproduce dataset noise).

## Realtime use

Load `lecturer_ru.pth` + `lecturer_ru.index` into `w-okada/voice-changer`, route
its output to a **virtual microphone** (PipeWire/PulseAudio null-sink on Linux,
VB-CABLE on Windows), and select that as the mic in the VoIP app. Tune chunk size
(latency vs quality), index ratio (timbre strength), and pitch shift. ~100-200 ms
on an RTX 2080 Super.

## Status & next steps

- Notebook written and JSON-valid; mirrors the canonical RVC training flow.
- Not yet run end-to-end on Kaggle (needs `HF_TOKEN`); like the F5 notebook,
  expect 1-2 interactive fixes on first run (fairseq install, extractor arg drift).
- On-device Android realtime (the eventual Telegram goal) is a separate, harder
  problem — RVC realtime is GPU-class; deferred.
