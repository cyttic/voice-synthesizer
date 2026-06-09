"""
Transcribe a Russian audio file with Whisper + pyannote diarization.
Clips are cut at pyannote speaker-turn boundaries (not Whisper boundaries),
so each output clip contains exactly one speaker's voice.

By default ALL speakers are exported into separate subfolders:
    dataset/clips/SPEAKER_00/clip_0001.wav + clip_0001.txt
    dataset/clips/SPEAKER_01/clip_0001.wav + clip_0001.txt
    ...
Listen to a few clips from each folder, find which one is the lecturer,
then re-run with --target-speaker SPEAKER_XX to export only that speaker
(or just keep the folder you want and delete the rest).

segments.json records the speaker label for every clip.

Install dependencies:
    pip install openai-whisper pydub pyannote.audio
    # also needs ffmpeg on PATH

Diarization requires a Hugging Face token with access to:
    https://huggingface.co/pyannote/speaker-diarization-3.1
"""

import argparse
import json
import os
from pathlib import Path

import whisper
from pydub import AudioSegment


# --- Transcription ---

def transcribe(audio_path: str, model_size: str, device: str = "cpu") -> list[dict]:
    print(f"Loading Whisper model '{model_size}' on {device}...")
    model = whisper.load_model(model_size, device=device)

    print(f"Transcribing {audio_path} (language: ru)...")
    result = model.transcribe(audio_path, language="ru", verbose=False)

    segments = []
    for seg in result["segments"]:
        text = seg["text"].strip()
        if not text:
            continue
        segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": text,
            "avg_logprob": seg.get("avg_logprob", 0.0),
            "no_speech_prob": seg.get("no_speech_prob", 0.0),
        })

    print(f"Whisper: {len(segments)} segments.")
    return segments


# --- Diarization ---

def load_diarization_pipeline(hf_token: str):
    """Load (and on first run, download) the pyannote pipeline. Done BEFORE
    transcription so token/license/download failures surface immediately."""
    from pyannote.audio import Pipeline

    print("Loading pyannote diarization pipeline (validates token + model)...")
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        token=hf_token,
    )
    if pipeline is None:
        raise RuntimeError(
            "pyannote returned no pipeline. Usually means the HF token is invalid "
            "or you haven't accepted the model license at "
            "https://huggingface.co/pyannote/speaker-diarization-3.1"
        )
    return pipeline


def run_diarization(pipeline, audio_path: str):
    import soundfile as sf
    import torch

    print("Running speaker diarization...")
    # Load audio in-memory ourselves (via libsndfile) and hand it to the
    # pipeline as a {waveform, sample_rate} dict. This bypasses pyannote's
    # torchcodec-based loader, which is broken under the current CUDA setup.
    data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    # soundfile gives (frames, channels); pyannote wants (channels, frames)
    waveform = torch.from_numpy(data.T)
    output = pipeline({"waveform": waveform, "sample_rate": sample_rate})

    # pyannote 4.x returns a DiarizeOutput dataclass; older versions return an
    # Annotation directly. We want an Annotation with .itertracks().
    # Prefer exclusive_speaker_diarization: it drops overlapping speech turns
    # (two people at once), which is exactly what we want for clean clips.
    if hasattr(output, "exclusive_speaker_diarization"):
        return output.exclusive_speaker_diarization
    if hasattr(output, "speaker_diarization"):
        return output.speaker_diarization
    return output  # already an Annotation (legacy)


def get_turns(diarization) -> tuple[list[dict], str]:
    """
    Returns (turns, lecturer) where turns is every speaker turn as
    {start, end, speaker} sorted by time, and lecturer is the speaker label
    with the most total speech time (the lecturer in a lecture recording).
    """
    turns = [
        {"start": turn.start, "end": turn.end, "speaker": speaker}
        for turn, _, speaker in diarization.itertracks(yield_label=True)
    ]
    turns.sort(key=lambda t: t["start"])

    totals: dict[str, float] = {}
    for t in turns:
        totals[t["speaker"]] = totals.get(t["speaker"], 0.0) + (t["end"] - t["start"])
    print(f"Speakers found (sec of speech): { {k: round(v, 1) for k, v in sorted(totals.items())} }")

    lecturer = max(totals, key=totals.get)
    print(f"Auto-detected lecturer (most speech): {lecturer}")
    return turns, lecturer


# --- Match Whisper text to pyannote turns ---

def build_clips(
    turns: list[dict],
    whisper_segments: list[dict],
    min_logprob: float,
    max_no_speech_prob: float,
    min_duration: float,
    max_duration: float,
) -> list[dict]:
    """
    For each pyannote turn, collect overlapping Whisper text. Audio is cut at
    the pyannote turn boundaries, so each clip is guaranteed single-speaker.
    Each clip keeps its speaker label.
    """
    clips = []

    for turn in turns:
        turn_start, turn_end, speaker = turn["start"], turn["end"], turn["speaker"]
        duration = turn_end - turn_start

        if duration < min_duration or duration > max_duration:
            continue

        overlapping = []
        for seg in whisper_segments:
            overlap = min(seg["end"], turn_end) - max(seg["start"], turn_start)
            if overlap <= 0:
                continue
            if seg["avg_logprob"] < min_logprob or seg["no_speech_prob"] > max_no_speech_prob:
                continue
            overlapping.append(seg)

        if not overlapping:
            continue

        text = " ".join(s["text"] for s in overlapping).strip()
        if not text:
            continue

        clips.append({
            "start": turn_start,
            "end": turn_end,
            "speaker": speaker,
            "text": text,
            "avg_logprob": min(s["avg_logprob"] for s in overlapping),
            "no_speech_prob": max(s["no_speech_prob"] for s in overlapping),
        })

    print(f"Built {len(clips)} single-speaker clips with text.")
    return clips


# --- Audio splitting ---

def split_audio(
    audio_path: str,
    clips: list[dict],
    output_dir: str,
    target_speaker: str | None,
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"\nLoading audio and exporting clips...")
    audio = AudioSegment.from_file(audio_path)

    # Per-speaker clip counters so filenames restart in each folder
    counters: dict[str, int] = {}

    for clip in clips:
        speaker = clip["speaker"]
        if target_speaker and speaker != target_speaker:
            continue

        counters[speaker] = counters.get(speaker, 0) + 1
        clip_id = f"clip_{counters[speaker]:04d}"

        # When exporting a single target speaker, put files flat in output_dir.
        # Otherwise separate each speaker into its own subfolder for review.
        clip_dir = out if target_speaker else out / speaker
        clip_dir.mkdir(parents=True, exist_ok=True)

        start_ms = int(clip["start"] * 1000)
        end_ms = int(clip["end"] * 1000)
        segment = audio[start_ms:end_ms]

        segment.export(clip_dir / f"{clip_id}.wav", format="wav")
        (clip_dir / f"{clip_id}.txt").write_text(clip["text"], encoding="utf-8")

        print(f"  {speaker}/{clip_id}.wav  [{clip['start']:.1f}s – {clip['end']:.1f}s]  {clip['text'][:60]}")

    # Always save the full index (every clip + its speaker label)
    index_path = out / "segments.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(clips, f, ensure_ascii=False, indent=2)

    exported = sum(counters.values())
    print(f"\nDone. Exported {exported} clips to: {out}")
    if not target_speaker:
        print("Clips are separated by speaker into subfolders.")
        print("Listen to each folder, find the lecturer, then re-run with")
        print("  --target-speaker SPEAKER_XX")
        print("or simply keep that folder and delete the others.")
    print(f"Full index (all speakers): {index_path}")


# --- Preflight ---

def preflight(audio_path: str, output_dir: str, hf_token: str | None):
    """Validate everything needed to finish the run, BEFORE the slow
    transcription starts. Raises SystemExit with a clear message on failure."""
    errors = []

    # Audio file exists and is readable
    if not os.path.isfile(audio_path):
        errors.append(f"audio file not found: {audio_path}")
    else:
        try:
            import soundfile as sf
            sf.info(audio_path)  # reads header only
        except ImportError:
            errors.append("soundfile not installed — pip install soundfile")
        except Exception as e:
            errors.append(f"cannot read audio file {audio_path}: {e}")

    # HF token present
    if not hf_token:
        errors.append("HF_TOKEN not set — export HF_TOKEN=hf_xxx (required for diarization)")

    # Required imports for the later stages
    for module, pip_name in [("whisper", "openai-whisper"),
                             ("pyannote.audio", "pyannote.audio"),
                             ("pydub", "pydub"),
                             ("torch", "torch")]:
        try:
            __import__(module)
        except ImportError:
            errors.append(f"{module} not installed — pip install {pip_name}")

    # Output directory is writable (create it now)
    try:
        from pathlib import Path
        Path(output_dir).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        errors.append(f"cannot create output dir {output_dir}: {e}")

    if errors:
        print("\nPreflight FAILED — fix these before running:")
        for e in errors:
            print(f"  - {e}")
        raise SystemExit(1)

    print("Preflight OK — all dependencies present, audio readable, token set.")


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description="Transcribe, diarize, and split audio into single-speaker clips.")
    parser.add_argument("audio", help="Path to input audio file (wav, flac, mp3, etc.)")
    parser.add_argument("--output", "-o", default="dataset/clips", help="Output directory (default: dataset/clips)")
    parser.add_argument("--model", "-m", default="large-v3", help="Whisper model: tiny/base/small/medium/large-v2/large-v3 (default: large-v3)")
    parser.add_argument("--device", default="cpu", help="Device for Whisper: cpu or cuda (default: cpu — GPU currently broken by CUDA/driver mismatch)")
    parser.add_argument("--min-logprob", type=float, default=-0.5, help="Min Whisper avg_logprob confidence (default: -0.5)")
    parser.add_argument("--max-no-speech-prob", type=float, default=0.3, help="Max Whisper no_speech_prob (default: 0.3)")
    parser.add_argument("--min-duration", type=float, default=3.0, help="Min clip duration in seconds (default: 3.0)")
    parser.add_argument("--max-duration", type=float, default=15.0, help="Max clip duration in seconds (default: 15.0)")
    parser.add_argument("--target-speaker", default=None, help="Export only this speaker label (e.g. SPEAKER_01). Default: auto-detected lecturer (most frequent voice).")
    parser.add_argument("--all-speakers", action="store_true", help="Export every speaker into separate subfolders for review instead of just the lecturer.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token (defaults to HF_TOKEN env var)")
    args = parser.parse_args()

    hf_token = args.hf_token or os.environ.get("HF_TOKEN")

    # Validate everything up front so a long transcription never gets wasted.
    preflight(args.audio, args.output, hf_token)

    # Load the diarization model BEFORE transcribing. First run downloads the
    # model (~hundreds of MB) and validates the token/license — if that fails,
    # it fails now in seconds rather than after minutes of transcription.
    pipeline = load_diarization_pipeline(hf_token)

    whisper_segments = transcribe(args.audio, model_size=args.model, device=args.device)

    diarization = run_diarization(pipeline, args.audio)
    turns, lecturer = get_turns(diarization)

    # Default: keep only the lecturer (most frequent voice). --all-speakers
    # exports everyone; --target-speaker overrides the auto-detection.
    if args.all_speakers:
        target_speaker = None
    else:
        target_speaker = args.target_speaker or lecturer

    clips = build_clips(
        turns,
        whisper_segments,
        min_logprob=args.min_logprob,
        max_no_speech_prob=args.max_no_speech_prob,
        min_duration=args.min_duration,
        max_duration=args.max_duration,
    )

    split_audio(args.audio, clips, args.output, target_speaker)


if __name__ == "__main__":
    main()
