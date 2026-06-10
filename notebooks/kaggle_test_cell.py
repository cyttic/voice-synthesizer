# Paste this into a Kaggle cell AFTER training (uses in-memory: trainer, TOKENIZER_FILE, SPEAKER_REFERENCE).
# Tests the fine-tuned model on GPU, plays each clip inline, and prints a download link per file.

import os, glob, torch, torchaudio, soundfile as sf
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from IPython.display import Audio, FileLink, display

# soundfile loader (avoids torchcodec when reading reference wavs)
def _sf_load(path, *a, **k):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr
torchaudio.load = _sf_load

# Load best checkpoint onto the GPU
run_dir = trainer.output_path
ckpt    = sorted(glob.glob(f"{run_dir}/best_model*.pth"))[-1]
config  = XttsConfig(); config.load_json(f"{run_dir}/config.json")
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_path=ckpt, vocab_path=TOKENIZER_FILE, use_deepspeed=False)
model.cuda()
print("Model device:", next(model.parameters()).device)

# Generation helper: saves a wav, plays it inline, prints a download link
OUT_DIR = "/kaggle/working/generated"
os.makedirs(OUT_DIR, exist_ok=True)
_n = 0

def speak(text, temperature=0.7, reference=None):
    global _n
    _n += 1
    g, s = model.get_conditioning_latents(audio_path=reference or SPEAKER_REFERENCE)
    out = model.inference(text, "ru", g, s, temperature=temperature)
    path = os.path.join(OUT_DIR, f"gen_{_n:02d}.wav")
    sf.write(path, out["wav"], 24000)
    print(f"[{_n}] {text}")
    display(Audio(out["wav"], rate=24000))                       # play inline
    display(FileLink(os.path.relpath(path, "/kaggle/working")))  # download link
    return path

# Try it — call speak() as many times as you like:
speak("Здравствуйте, сегодня мы поговорим о криминалистике.")
speak("Это пример синтеза речи нашей моделью.", temperature=0.6)
