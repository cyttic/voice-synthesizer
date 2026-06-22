#!/usr/bin/env python3
"""
RVC voice conversion — convert an input speech clip into the fine-tuned voice
(model-rvc/lecturer_ru.pth + .index).

This is Step 0 for the KrimbaGram "voice changer" feature: it proves the trained
RVC model works on real audio, independent of any Telegram integration.

Uses rvc-python (https://pypi.org/project/rvc-python/), which bundles the RVC
inference stack and auto-downloads the HuBERT (content) and RMVPE (pitch) base
models on first run.

Setup (see commands the assistant gave you):
    python3 -m venv .venv-rvc && source .venv-rvc/bin/activate
    pip install --upgrade pip
    pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    pip install rvc-python

Run (from the voice-synthesizer project root):
    python inference/rvc_convert.py -i input.wav -o rvc_out.wav

`input.wav` should be a short speech clip in ANY voice (record yourself). The
output will be that same speech re-voiced as the lecturer.
"""

import argparse
import os
import sys


def main():
    p = argparse.ArgumentParser(description="RVC speech-to-speech voice conversion.")
    p.add_argument("--model", default="model-rvc/lecturer_ru.pth",
                   help="Path to the trained RVC generator (.pth).")
    p.add_argument("--index", default="model-rvc/lecturer_ru.index",
                   help="Path to the FAISS feature index (.index). Optional.")
    p.add_argument("--input", "-i", required=True,
                   help="Input speech wav (any voice).")
    p.add_argument("--output", "-o", default="rvc_out.wav",
                   help="Output wav path.")
    p.add_argument("--device", default="cpu",
                   help='"cpu" or "cuda:0". CPU is safest given this machine\'s CUDA mismatch.')
    p.add_argument("--pitch", type=int, default=0,
                   help="Pitch shift in semitones (e.g. +12 / -12 for octave).")
    p.add_argument("--f0method", default="rmvpe",
                   help="Pitch extraction: rmvpe (best) | harvest | crepe | pm.")
    p.add_argument("--index-rate", type=float, default=0.5,
                   help="0..1 timbre retrieval strength from the .index (0 = ignore index).")
    p.add_argument("--protect", type=float, default=0.33,
                   help="0..0.5 protect voiceless consonants/breath (lower = stronger conversion).")
    a = p.parse_args()

    for f in (a.model, a.input):
        if not os.path.exists(f):
            sys.exit(f"ERROR: not found: {f}\n(run from the voice-synthesizer project root)")

    have_index = os.path.exists(a.index)
    if not have_index:
        print(f"NOTE: index not found at {a.index} — converting without retrieval (index_rate=0).")

    try:
        from rvc_python.infer import RVCInference
    except ImportError:
        sys.exit("ERROR: rvc-python not installed. Run: pip install rvc-python")

    print(f"Loading RVC model on {a.device} ...")
    rvc = RVCInference(device=a.device)

    # Load model (+ index if available). API has shifted across versions, so be defensive.
    try:
        rvc.load_model(a.model, index_path=a.index if have_index else "")
    except TypeError:
        rvc.load_model(a.model)

    params = dict(
        f0up_key=a.pitch,
        f0method=a.f0method,
        index_rate=a.index_rate if have_index else 0.0,
        protect=a.protect,
    )
    try:
        rvc.set_params(**params)
    except Exception as e:
        print(f"set_params warning ({e}); continuing with defaults.")

    print(f"Converting {a.input} -> {a.output} ...")
    rvc.infer_file(a.input, a.output)
    print(f"Done. Wrote {a.output}")


if __name__ == "__main__":
    main()
