"""
Download the fine-tuned F5-TTS lecturer model from HuggingFace into model-f5/.

The repo is private, so an HF token with read access is required:
    export HF_TOKEN=hf_...          # or `huggingface-cli login` once

Run:
    python scripts/download_f5tts.py
"""

import os

from huggingface_hub import snapshot_download

REPO = "cyttic/lecturer-ru-f5tts"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "model-f5")

path = snapshot_download(
    REPO,
    local_dir=DEST,
    token=os.environ.get("HF_TOKEN"),  # None -> falls back to the cached login
)
print("Downloaded to:", path)
for name in sorted(os.listdir(path)):
    full = os.path.join(path, name)
    if os.path.isfile(full):
        print(f"  {name}  ({os.path.getsize(full) / 1e6:.1f} MB)")
