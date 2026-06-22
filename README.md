# 🎙️ LecturevoiceClone

A pipeline for building a high-quality TTS voice model from lecture recordings. Automatically extracts, cleans, diarizes, and segments audio — producing a ready-to-use dataset for fine-tuning XTTS v2.

---

## 💡 Idea

University lecture recordings are an ideal source for voice cloning:
- Long monologue speech from a single professor
- Varied intonation and natural prosody
- Hours of material available

The challenge: raw lecture recordings contain pauses, student questions, background noise, and other speakers. This pipeline solves all of that automatically.

---

## 🔁 Pipeline Overview

```
lecture.mp4 / .wav
      │
      ▼
┌─────────────────┐
│  Extract Audio  │  ffmpeg
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Transcribe + Diarize       │  WhisperX + pyannote.audio
│  (who said what, when)      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Filter by Speaker          │  Keep only target speaker (professor)
│  Remove student speech      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Split into Chunks          │  3–15 second clips, sentence-aligned
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Quality Filter             │  Remove too-short, noisy, or bad clips
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Export Dataset             │  wavs/ + metadata.csv
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Fine-tune XTTS v2          │  Coqui TTS trainer
└─────────────────────────────┘
```

---

## 📁 Project Structure

```
lecturevoiceclone/
├── README.md
├── requirements.txt
├── config.yaml                  # Main config (paths, speaker ID, thresholds)
│
├── pipeline/
│   ├── 01_extract_audio.py      # ffmpeg: video → wav
│   ├── 02_diarize.py            # WhisperX + pyannote: transcribe + label speakers
│   ├── 03_filter_speaker.py     # Keep only target speaker segments
│   ├── 04_split_chunks.py       # Split audio into short clips
│   ├── 05_quality_filter.py     # Remove bad clips
│   └── 06_export_dataset.py     # Build wavs/ + metadata.csv
│
├── train/
│   └── finetune_xtts.py         # XTTS v2 fine-tuning script
│
├── inference/
│   ├── generate.py              # Generate speech from fine-tuned model
│   ├── zero_shot.py             # Zero-shot voice cloning from a reference clip
│   └── telegram_bot.py          # Telegram bot serving the fine-tuned voice
│
└── dataset/                     # Output dataset (auto-generated)
    ├── wavs/
    │   ├── clip_0001.wav
    │   ├── clip_0002.wav
    │   └── ...
    └── metadata.csv
```

---

## ⚙️ Requirements

- Python 3.10+
- CUDA GPU (8GB+ VRAM recommended for training)
- ffmpeg installed on system
- Hugging Face account (for pyannote diarization model)

### Install dependencies

```bash
pip install -r requirements.txt
```

**requirements.txt includes:**
- `whisperx`
- `pyannote.audio`
- `pydub`
- `TTS` (Coqui)
- `torch`
- `torchaudio`
- `pandas`

---

## 🚀 Usage

### 1. Configure

Edit `config.yaml`:

```yaml
input_video: "lecture.mp4"
output_dir: "dataset/"
hf_token: "hf_your_token_here"     # Hugging Face token for pyannote
target_speaker: "SPEAKER_00"       # Will be auto-detected or set manually
language: "en"                     # Lecture language
min_clip_duration: 3.0             # seconds
max_clip_duration: 15.0            # seconds
```

### 2. Run the full pipeline

```bash
python pipeline/01_extract_audio.py
python pipeline/02_diarize.py
python pipeline/03_filter_speaker.py
python pipeline/04_split_chunks.py
python pipeline/05_quality_filter.py
python pipeline/06_export_dataset.py
```

Or run all steps at once:

```bash
python run_pipeline.py
```

### 3. Fine-tune XTTS v2

```bash
python train/finetune_xtts.py --dataset dataset/ --output model/
```

### 4. Generate speech

```bash
python inference/generate.py \
  --model model/ \
  --text "Hello, today we will discuss neural networks." \
  --output output.wav
```

---

## 🤖 Telegram Bot

`inference/telegram_bot.py` serves the fine-tuned voice over Telegram: a user
sends text, the bot replies with a voice message. It talks to the Telegram Bot
API directly via `requests` (no bot framework), and runs the model on your own
machine (GPU or CPU). Each user can tune the voice with `/temp`, `/speed`, and
`/reppen`; their settings persist in `user_settings.json`.

### Prerequisites

- A bot token from [@BotFather](https://t.me/BotFather)
- `ffmpeg` on the system (used to encode OGG/Opus voice messages)
- A Python environment with `TTS`, `torch`, `torchaudio`, `soundfile`,
  `requests`, and `numpy` installed (the same env used for `generate.py`)
- A model directory containing `model.pth`, `config.json`, `vocab.json`, and
  `reference.wav` (e.g. `model-v2/`)

### Configuration (environment variables)

| Variable | Default | Description |
|---|---|---|
| `TELEGRAM_TOKEN` | — (required) | Bot token from @BotFather |
| `XTTS_MODEL_DIR` | `model-v2` | Dir with `model.pth`, `config.json`, `vocab.json`, `reference.wav` |
| `XTTS_DEVICE` | `cuda` | `cuda` or `cpu` |
| `XTTS_LANGUAGE` | `ru` | Generation language |
| `XTTS_MAX_CHARS` | `1000` | Max characters per message |
| `XTTS_SETTINGS_FILE` | `user_settings.json` | Where per-user settings are stored |

### Run

```bash
export TELEGRAM_TOKEN=123456:ABC-your-token
export XTTS_MODEL_DIR=model-v2
export XTTS_DEVICE=cuda            # or cpu

python inference/telegram_bot.py
```

On startup the bot loads the model (~10–20s), registers its command menu, and
prints `Bot online: <username>`. Open the bot in Telegram and send any text to
get a voice reply. `/help` lists all commands.

To keep it running after you log out, launch it under `tmux`/`nohup` or as a
systemd user service.

> ⚠️ XTTS v2 is licensed under the Coqui Public Model License (non-commercial,
> R&D use only).

---

## 🏎️ Variant: Piper voice (fast / realtime)

XTTS v2 is the *quality* voice but autoregressive — too slow for a realtime voice
bot. **Piper** (VITS) is the low-latency alternative: non-autoregressive, faster
than realtime on CPU, ~tens-of-ms to first audio. It's the voice that feeds the
realtime Telegram **call** agent in `../clone-telegram/voice-agent` (whose
`tts.py` already wraps `PiperVoice`).

- **Train:** `notebooks/kaggle_finetune_piper.ipynb` (+ `.md`). Runs entirely on
  Kaggle; fine-tunes from the Russian Piper **Irina** base
  (`rhasspy/piper-checkpoints`, a *dataset* repo) using the existing HF datasets
  (`cyttic/audio-kri-russian` + `-2`) — no whisper/diarize re-run.
- **Storage: HuggingFace only** (no Kaggle Datasets/output, no `/kaggle` paths).
  Data in ← HF; voice + resume checkpoint out → `cyttic/lecturer-ru-piper`
  (`lecturer_ru.onnx` + `lecturer_ru.onnx.json` + `resume.ckpt`). Crash-resume
  round-trips through HF, so a dead Kaggle session loses nothing.
- **Local model dir:** `model-piper/` (gitignored; pull from HF).

### Test it locally

```bash
pip install piper-tts          # bundles espeak-ng; CPU inference
echo "Привет! Это тест клонированного голоса." \
  | piper -m lecturer-ru-piper/lecturer_ru.onnx -f /tmp/test.wav
ffplay -autoexit -nodisp /tmp/test.wav        # or: aplay /tmp/test.wav
```

Needs **both** `lecturer_ru.onnx` and `lecturer_ru.onnx.json` in the same folder.
Tune naturalness without retraining via the `"inference"` block in the `.onnx.json`
(`length_scale` = rate, `noise_scale` / `noise_w` = expressiveness).

### Serve in the realtime call bot

Copy the two `.onnx` files into `clone-telegram/voice-agent/voices/` and set
`VA_TTS_VOICE` to the `.onnx` — no extra serving code needed.

> ✅ Piper is MIT/GPL (fine to use/redistribute), unlike XTTS's CPML.

---

## 📊 Dataset Format

The pipeline produces a dataset compatible with Coqui TTS trainer:

```
dataset/
├── wavs/
│   ├── clip_0001.wav
│   ├── clip_0002.wav
│   └── ...
└── metadata.csv
```

`metadata.csv` format:

```
clip_0001|Today we will discuss the basics of machine learning.
clip_0002|The first concept we need to understand is a neural network.
clip_0003|Let me show you a simple example on the board.
```

---

## 🎯 Recommended Audio Amount

| Quality | Audio needed |
|---|---|
| Decent clone | ~30 minutes of clean speech |
| Good quality | ~1–3 hours |
| High quality | 3+ hours, varied content |

Lecture recordings work great because they provide long, natural monologue speech — exactly what TTS models need.

---

## 🧠 Models Used

| Task | Tool |
|---|---|
| Transcription | [WhisperX](https://github.com/m-bain/whisperX) / openai-whisper |
| Speaker Diarization | [pyannote.audio](https://github.com/pyannote/pyannote-audio) |
| Voice cloning — quality | [Coqui XTTS v2](https://github.com/coqui-ai/TTS) · [F5-TTS](https://github.com/SWivid/F5-TTS) |
| Voice cloning — fast/realtime | [Piper](https://github.com/OHF-Voice/piper1-gpl) (VITS) |
| Voice conversion | [RVC](https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI) |
| Audio processing | [pydub](https://github.com/jiaaro/pydub) + [ffmpeg](https://ffmpeg.org) |

---

## ⚠️ Notes

- **pyannote** requires accepting the model license on Hugging Face and providing an access token
- Diarization accuracy is ~90–95%; a quick manual review of the dataset is recommended
- XTTS v2 supports ~17 languages — set the correct language in `config.yaml`
- For best results, use recordings where the professor's voice is clear and dominant

---

## 📄 License

MIT
