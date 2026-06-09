"""
Zero-shot voice cloning with base XTTS v2 — no training required.

Clones a voice from a single short reference clip (6-30s) of the speaker and
synthesizes Russian text in that voice. Runs on CPU.

Install (CPU):
    pip install coqui-tts

Usage:
    python inference/zero_shot.py \\
        --reference dataset/krimba_25022025/clip_0001.wav \\
        --text "Здравствуйте, сегодня мы поговорим о криминалистике." \\
        --output out.wav

The first run downloads the base XTTS v2 model (~1.8 GB).
XTTS v2 is non-commercial (CPML) — R&D use only.
"""

import argparse
import os

# Auto-accept the Coqui model license prompt (non-interactive).
os.environ.setdefault("COQUI_TOS_AGREED", "1")


def main():
    parser = argparse.ArgumentParser(description="Zero-shot XTTS v2 voice cloning (CPU).")
    parser.add_argument("--reference", "-r", required=True,
                        help="Reference wav of the speaker (6-30s, clean). e.g. a dataset clip.")
    parser.add_argument("--text", required=True, help="Text to synthesize (Russian)")
    parser.add_argument("--output", "-o", default="out.wav", help="Output wav path (default: out.wav)")
    parser.add_argument("--language", default="ru", help="Language code (default: ru)")
    args = parser.parse_args()

    if not os.path.isfile(args.reference):
        raise SystemExit(f"Reference wav not found: {args.reference}")

    import torch
    from TTS.api import TTS

    torch.set_num_threads(os.cpu_count() or 4)  # use all CPU cores

    print("Loading base XTTS v2 (downloads ~1.8 GB on first run)...")
    tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

    print(f"Cloning from {args.reference}")
    print(f"Generating: {args.text!r}")
    tts.tts_to_file(
        text=args.text,
        speaker_wav=args.reference,
        language=args.language,
        file_path=args.output,
    )
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
