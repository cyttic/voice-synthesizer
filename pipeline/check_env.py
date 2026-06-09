"""
Preflight check for the transcription pipeline.

Run this BEFORE transcribing to confirm every dependency is present and
working. It checks Python packages, ffmpeg, the GPU/CUDA situation, the
Hugging Face token, and that audio can actually be loaded.

    python3 pipeline/check_env.py
    python3 pipeline/check_env.py /path/to/audio.flac   # also test-load a file
"""

import importlib
import os
import shutil
import subprocess
import sys

OK = "\033[92m  OK \033[0m"
WARN = "\033[93m WARN\033[0m"
FAIL = "\033[91m FAIL\033[0m"

problems = 0
warnings = 0


def status(level: str, label: str, detail: str = "") -> None:
    global problems, warnings
    if level == FAIL:
        problems += 1
    elif level == WARN:
        warnings += 1
    line = f"[{level}] {label}"
    if detail:
        line += f"\n         {detail}"
    print(line)


def check_python() -> None:
    v = sys.version_info
    if v >= (3, 10):
        status(OK, f"Python {v.major}.{v.minor}.{v.micro}")
    else:
        status(FAIL, f"Python {v.major}.{v.minor}", "Need Python 3.10+")


def check_package(module: str, pip_name: str | None = None) -> bool:
    pip_name = pip_name or module
    try:
        mod = importlib.import_module(module)
        ver = getattr(mod, "__version__", "?")
        status(OK, f"{module} ({ver})")
        return True
    except Exception as e:
        status(FAIL, f"{module} missing", f"pip install {pip_name}   ({type(e).__name__}: {e})")
        return False


def check_ffmpeg() -> None:
    path = shutil.which("ffmpeg")
    if not path:
        status(FAIL, "ffmpeg not on PATH", "sudo apt install ffmpeg")
        return
    try:
        out = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True, timeout=10)
        first = out.stdout.splitlines()[0] if out.stdout else "ffmpeg"
        status(OK, first)
    except Exception as e:
        status(WARN, "ffmpeg found but failed to run", str(e))


def check_torch_cuda() -> None:
    try:
        import torch
    except ImportError:
        status(FAIL, "torch missing", "pip install torch")
        return

    status(OK, f"torch ({torch.__version__}), built for CUDA {torch.version.cuda}")

    if torch.cuda.is_available():
        try:
            name = torch.cuda.get_device_name(0)
            status(OK, f"CUDA available — GPU: {name}")
        except Exception as e:
            status(WARN, "CUDA reports available but GPU query failed", str(e))
    else:
        status(WARN, "CUDA NOT available — will run on CPU (slow)",
               "Common cause: torch CUDA version newer than the NVIDIA driver. "
               "Check `nvidia-smi` (driver's max CUDA) vs torch.version.cuda above.")


def check_torchcodec_note() -> None:
    """torchcodec is often broken under CUDA mismatch; we bypass it with soundfile."""
    try:
        import torchcodec  # noqa: F401
        status(OK, "torchcodec importable")
    except Exception:
        status(WARN, "torchcodec not usable",
               "Not fatal — diarize() loads audio via soundfile and passes it "
               "in-memory to pyannote, bypassing torchcodec.")


def check_hf_token() -> None:
    token = os.environ.get("HF_TOKEN")
    if token:
        masked = token[:6] + "..." + token[-4:] if len(token) > 12 else "***"
        status(OK, f"HF_TOKEN set ({masked})")
    else:
        status(FAIL, "HF_TOKEN not set",
               "export HF_TOKEN=hf_xxx  — required for pyannote diarization. "
               "Also accept the license at huggingface.co/pyannote/speaker-diarization-3.1")


def check_audio_load(audio_path: str) -> None:
    try:
        import soundfile as sf
    except ImportError:
        status(FAIL, "cannot test audio load — soundfile missing")
        return

    if not os.path.exists(audio_path):
        status(FAIL, f"audio file not found: {audio_path}")
        return

    try:
        info = sf.info(audio_path)
        status(OK, f"audio loads: {audio_path}",
               f"{info.duration:.1f}s, {info.samplerate} Hz, {info.channels} ch, {info.format}")
    except Exception as e:
        status(FAIL, f"failed to read audio: {audio_path}", str(e))


def main() -> None:
    print("=" * 60)
    print("  Voice-synthesizer pipeline — environment check")
    print("=" * 60)

    print("\n-- Core --")
    check_python()
    check_ffmpeg()

    print("\n-- Python packages --")
    check_package("whisper", "openai-whisper")
    check_package("pyannote.audio", "pyannote.audio")
    check_package("pydub")
    check_package("soundfile")
    check_package("numpy")

    print("\n-- PyTorch / GPU --")
    check_torch_cuda()
    check_torchcodec_note()

    print("\n-- Hugging Face --")
    check_hf_token()

    if len(sys.argv) > 1:
        print("\n-- Audio file --")
        check_audio_load(sys.argv[1])

    print("\n" + "=" * 60)
    if problems:
        print(f"  {problems} problem(s), {warnings} warning(s) — fix FAILs before running.")
        sys.exit(1)
    elif warnings:
        print(f"  0 problems, {warnings} warning(s) — pipeline will run (CPU/bypass).")
        sys.exit(0)
    else:
        print("  All checks passed. Ready to transcribe.")
        sys.exit(0)


if __name__ == "__main__":
    main()
