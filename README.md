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
│   └── generate.py              # Generate speech from fine-tuned model
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
| Transcription | [WhisperX](https://github.com/m-bain/whisperX) |
| Speaker Diarization | [pyannote.audio](https://github.com/pyannote/pyannote-audio) |
| Voice Cloning / Fine-tuning | [Coqui XTTS v2](https://github.com/coqui-ai/TTS) |
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
