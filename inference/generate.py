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

import torch
import torchaudio
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts


def load_model(model_dir: str) -> Xtts:
    config = XttsConfig()
    config.load_json(os.path.join(model_dir, "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(
        config,
        checkpoint_path=os.path.join(model_dir, "model.pth"),
        vocab_path=os.path.join(model_dir, "vocab.json"),
        use_deepspeed=False,   # no GPU / DeepSpeed on CPU
    )
    model.cpu()  # force CPU
    model.eval()
    return model


def generate(model: Xtts, text: str, reference_wav: str, output: str,
             language: str = "ru", temperature: float = 0.7) -> None:
    print("Computing speaker conditioning from reference...")
    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=[reference_wav]
    )

    print(f"Generating: {text!r}")
    out = model.inference(
        text,
        language,
        gpt_cond_latent,
        speaker_embedding,
        temperature=temperature,
    )

    wav = torch.tensor(out["wav"]).unsqueeze(0)  # (1, samples)
    torchaudio.save(output, wav, 24000)
    duration = wav.shape[1] / 24000
    print(f"Saved {output}  ({duration:.1f}s)")


def main():
    parser = argparse.ArgumentParser(description="Generate Russian speech in the cloned lecturer voice (CPU).")
    parser.add_argument("--model-dir", default="model", help="Dir with model.pth, config.json, vocab.json, reference.wav")
    parser.add_argument("--text", required=True, help="Text to synthesize (Russian)")
    parser.add_argument("--output", "-o", default="out.wav", help="Output wav path (default: out.wav)")
    parser.add_argument("--reference", default=None, help="Reference wav for the voice (default: <model-dir>/reference.wav)")
    parser.add_argument("--language", default="ru", help="Language code (default: ru)")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    args = parser.parse_args()

    reference = args.reference or os.path.join(args.model_dir, "reference.wav")
    if not os.path.isfile(reference):
        raise SystemExit(f"Reference wav not found: {reference}")

    torch.set_num_threads(os.cpu_count() or 4)  # use all CPU cores

    model = load_model(args.model_dir)
    generate(model, args.text, reference, args.output, args.language, args.temperature)


if __name__ == "__main__":
    main()
