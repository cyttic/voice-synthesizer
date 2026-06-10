"""
Generate Russian speech in the fine-tuned lecturer voice — CPU only.

First, download the fine-tuned model from the HF Hub (one time):
    huggingface-cli download <user>/xtts-v2-kri-russian --local-dir model/
    # gives model/model.pth, model/config.json, model/vocab.json, model/reference.wav

Install (CPU):
    pip install coqui-tts soundfile

Usage:
    python inference/generate.py \\
        --model-dir model/ \\
        --text "Здравствуйте, сегодня мы поговорим о криминалистике." \\
        --output out.wav

XTTS v2 is non-commercial (CPML) — R&D use only.
"""

import argparse
import os

import numpy as np
import soundfile as sf
import torch
import torchaudio

# torchcodec (torchaudio's default decoder) is broken under this machine's
# CUDA/driver mismatch. Replace torchaudio.load with a soundfile-based loader
# so XTTS's internal load_audio() never touches torchcodec. Must run before
# any torchaudio.load call.
def _sf_load(path, *args, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)  # (frames, ch)
    return torch.from_numpy(data.T), sr                             # (ch, frames)

torchaudio.load = _sf_load

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def load_model(model_dir: str, device: str = "cpu") -> Xtts:
    config = XttsConfig()
    config.load_json(os.path.join(model_dir, "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=os.path.join(model_dir, "model.pth"),
        vocab_path=os.path.join(model_dir, "vocab.json"),
        use_deepspeed=False,
    )
    model.to(device)
    model.eval()
    return model


def generate(model: Xtts, text: str, reference_wav: str, output: str,
             language: str = "ru", **gen_kwargs) -> None:
    print("Computing speaker conditioning from reference...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[reference_wav]
    )

    print(f"Generating: {text!r}")
    print(f"  params: {gen_kwargs}")
    out = model.inference(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        **gen_kwargs,
    )

    # Save via soundfile (also avoids torchcodec on the write path)
    wav = np.asarray(out["wav"], dtype="float32")
    sf.write(output, wav, 24000)
    print(f"Saved {output}  ({len(wav) / 24000:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Generate Russian speech in the cloned lecturer voice (CPU).")
    parser.add_argument("--model-dir", default="model", help="Dir with model.pth, config.json, vocab.json, reference.wav")
    parser.add_argument("--text", required=True, help="Text to synthesize (Russian)")
    parser.add_argument("--output", "-o", default="out.wav", help="Output wav path (default: out.wav)")
    parser.add_argument("--reference", default=None, help="Reference wav for the voice (default: <model-dir>/reference.wav)")
    parser.add_argument("--language", default="ru", help="Language code (default: ru)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Randomness; lower=flatter, higher=more varied (default: 0.7)")
    parser.add_argument("--repetition-penalty", type=float, default=5.0, help="Penalize repeats/stutter; raise if it loops (default: 5.0)")
    parser.add_argument("--length-penalty", type=float, default=1.0, help="Bias toward shorter/longer output (default: 1.0)")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k sampling (default: 50)")
    parser.add_argument("--top-p", type=float, default=0.85, help="Top-p / nucleus sampling (default: 0.85)")
    parser.add_argument("--speed", type=float, default=1.0, help="Speaking speed, 0.8=slower 1.2=faster (default: 1.0)")
    parser.add_argument("--device", default="cpu", help="cpu or cuda (default: cpu)")
    args = parser.parse_args()

    reference = args.reference or os.path.join(args.model_dir, "reference.wav")
    if not os.path.isfile(reference):
        raise SystemExit(f"Reference wav not found: {reference}")

    if args.device == "cpu":
        torch.set_num_threads(os.cpu_count() or 4)  # use all CPU cores

    print(f"Loading model on {args.device}...")
    model = load_model(args.model_dir, device=args.device)
    generate(
        model, args.text, reference, args.output, args.language,
        temperature=args.temperature,
        repetition_penalty=args.repetition_penalty,
        length_penalty=args.length_penalty,
        top_k=args.top_k,
        top_p=args.top_p,
        speed=args.speed,
    )


if __name__ == "__main__":
    main()
