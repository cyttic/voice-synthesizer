"""
Telegram bot: user sends text -> bot replies with a voice message in the
fine-tuned lecturer voice. Backend runs on your machine (GPU or CPU).

No bot framework needed — talks to the Telegram Bot API directly via requests,
to avoid dependency conflicts.

Setup:
    pip install requests            # (usually already installed)
    export TELEGRAM_TOKEN=123:ABC   # from @BotFather
    export XTTS_MODEL_DIR=model-v2  # dir with model.pth, config.json, vocab.json, reference.wav
    export XTTS_DEVICE=cuda         # or cpu

Run:
    python inference/telegram_bot.py

XTTS v2 is non-commercial (CPML) — R&D use only.
"""

import json
import os
import re
import subprocess
import tempfile
import time

import numpy as np
import requests
import soundfile as sf
import torch
import torchaudio

# torchcodec (torchaudio's default decoder) is broken under this machine's
# CUDA setup. Load audio via soundfile so XTTS's load_audio never touches it.
def _sf_load(path, *args, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr

torchaudio.load = _sf_load

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

# --- config ---
TOKEN     = os.environ.get("TELEGRAM_TOKEN")
MODEL_DIR = os.environ.get("XTTS_MODEL_DIR", "model-v2")
DEVICE    = os.environ.get("XTTS_DEVICE", "cuda")
LANGUAGE  = os.environ.get("XTTS_LANGUAGE", "ru")
MAX_CHARS = int(os.environ.get("XTTS_MAX_CHARS", "1000"))

API = f"https://api.telegram.org/bot{TOKEN}"
REFERENCE = os.path.join(MODEL_DIR, "reference.wav")
SETTINGS_FILE = os.environ.get("XTTS_SETTINGS_FILE", "user_settings.json")

# Fixed params (not user-tunable) + default values for tunable ones.
BASE_GEN = dict(top_k=50, top_p=0.85, enable_text_splitting=True)
DEFAULTS = dict(temperature=0.7, repetition_penalty=5.0, speed=1.0)

# User command -> (param name, min, max). These are the knobs users can change.
TUNABLE = {
    "temp":   ("temperature", 0.0, 0.9),
    "speed":  ("speed", 0.7, 1.3),
    "reppen": ("repetition_penalty", 2.0, 12.0),
}

# --- per-user settings (persisted to JSON) ---
user_settings: dict = {}   # {"<chat_id>": {"temperature": 0.6, ...}}

def load_settings():
    global user_settings
    if os.path.isfile(SETTINGS_FILE):
        try:
            with open(SETTINGS_FILE) as f:
                user_settings = json.load(f)
            print(f"Loaded settings for {len(user_settings)} users.")
        except Exception as e:
            print("settings load error:", e)

def save_settings():
    try:
        with open(SETTINGS_FILE, "w") as f:
            json.dump(user_settings, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("settings save error:", e)

def get_gen(chat_id) -> dict:
    """Merge fixed params + defaults + this user's overrides."""
    g = {**BASE_GEN, **DEFAULTS}
    g.update(user_settings.get(str(chat_id), {}))
    return g


# --- model (loaded once at startup) ---

def load_model(model_dir: str, device: str) -> Xtts:
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


print(f"Loading model from {MODEL_DIR} on {DEVICE}...")
model = load_model(MODEL_DIR, DEVICE)
gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(audio_path=[REFERENCE])
print("Model ready. Actual device:", next(model.parameters()).device)


# --- synthesis ---

def synth_to_ogg(text: str, gen: dict) -> str:
    """Generate speech and return a path to a Telegram-ready OGG/Opus file."""
    t0 = time.time()
    out = model.inference(text, LANGUAGE, gpt_cond_latent, speaker_embedding, **gen)
    wav = np.asarray(out["wav"], dtype="float32")
    gen_s = time.time() - t0
    audio_s = len(wav) / 24000
    ratio = gen_s / audio_s if audio_s else 0
    print(f"  chars={len(text)}  gen={gen_s:.1f}s  audio={audio_s:.1f}s  "
          f"(x{ratio:.1f} realtime)")

    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    ogg_fd, ogg_path = tempfile.mkstemp(suffix=".ogg")
    os.close(wav_fd); os.close(ogg_fd)

    sf.write(wav_path, wav, 24000)
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "32k", "-ac", "1", ogg_path],
        check=True, capture_output=True,
    )
    os.remove(wav_path)
    return ogg_path


# --- Telegram API helpers ---

# All send calls have timeouts so a stalled network never freezes the bot.
def send_message(chat_id, text):
    requests.post(f"{API}/sendMessage", data={"chat_id": chat_id, "text": text},
                  timeout=(10, 20))

def send_voice(chat_id, ogg_path):
    with open(ogg_path, "rb") as f:
        requests.post(f"{API}/sendVoice", data={"chat_id": chat_id},
                      files={"voice": f}, timeout=(10, 60))  # file upload

def chat_action(chat_id, action="record_voice"):
    requests.post(f"{API}/sendChatAction", data={"chat_id": chat_id, "action": action},
                  timeout=(10, 15))

def register_commands():
    """Populate the Telegram command menu (the / autocomplete + Menu button)."""
    commands = [
        {"command": "temp",     "description": "Выразительность 0.0–0.9"},
        {"command": "speed",    "description": "Скорость речи 0.7–1.3"},
        {"command": "reppen",   "description": "Против повторов 2–12"},
        {"command": "settings", "description": "Показать мои настройки"},
        {"command": "reset",    "description": "Сбросить настройки"},
        {"command": "help",     "description": "Помощь"},
    ]
    try:
        requests.post(f"{API}/setMyCommands", json={"commands": commands}, timeout=(10, 15))
        print("Command menu registered.")
    except requests.RequestException as e:
        print("setMyCommands failed:", type(e).__name__)


# --- commands ---

HELP = (
    "Пришли мне текст — отвечу голосовым сообщением 🎙️\n\n"
    "Настройки голоса (у каждого свои):\n"
    "• /temp 0.7 — выразительность (0.0–0.9)\n"
    "• /speed 1.0 — скорость речи (0.7–1.3)\n"
    "• /reppen 5 — против заиканий/повторов (2–12)\n"
    "• /settings — показать твои настройки\n"
    "• /reset — сбросить к стандартным"
)

def fmt_settings(chat_id) -> str:
    g = get_gen(chat_id)
    return (
        "Твои текущие настройки:\n"
        f"• temperature = {g['temperature']}  (/temp 0.0–0.9)\n"
        f"• speed = {g['speed']}  (/speed 0.7–1.3)\n"
        f"• repetition_penalty = {g['repetition_penalty']}  (/reppen 2–12)\n"
        "/reset — сбросить к стандартным"
    )

def handle_command(chat_id, text) -> None:
    # Accept "/temp 0.9", "/temp=0.9", "/temp:0.9", "/temp@bot 0.9"
    m = re.match(r"^/([A-Za-z]+)(?:@\S+)?[\s=:]*(.*)$", text.strip())
    if not m:
        send_message(chat_id, "Неизвестная команда. /help")
        return
    cmd = m.group(1).lower()
    arg = m.group(2).strip()

    if cmd in ("start", "help"):
        send_message(chat_id, HELP)
    elif cmd == "settings":
        send_message(chat_id, fmt_settings(chat_id))
    elif cmd == "reset":
        user_settings.pop(str(chat_id), None)
        save_settings()
        send_message(chat_id, "Настройки сброшены к стандартным ✅")
    elif cmd in TUNABLE:
        key, lo, hi = TUNABLE[cmd]
        if not arg:
            send_message(chat_id, f"Укажи значение: /{cmd} {lo}–{hi}")
            return
        try:
            val = float(arg.replace(",", "."))
        except ValueError:
            send_message(chat_id, "Нужно число, например /%s %s" % (cmd, lo))
            return
        clamped = max(lo, min(hi, val))
        user_settings.setdefault(str(chat_id), {})[key] = clamped
        save_settings()
        note = "" if clamped == val else f" (ограничено диапазоном {lo}–{hi})"
        send_message(chat_id, f"{key} = {clamped}{note} ✅")
    else:
        send_message(chat_id, "Неизвестная команда. /help")


# --- main polling loop ---

def main():
    if not TOKEN:
        raise SystemExit("TELEGRAM_TOKEN not set. export TELEGRAM_TOKEN=... (from @BotFather)")

    load_settings()
    register_commands()
    me = requests.get(f"{API}/getMe").json()
    print("Bot online:", me.get("result", {}).get("username"))
    print("Polling for messages... (Ctrl+C to stop)")

    offset = None
    while True:
        try:
            # Short long-poll + tight read timeout so a hung connection on a
            # flaky network is abandoned and retried quickly (was 30s/40s).
            resp = requests.get(f"{API}/getUpdates",
                                params={"timeout": 15, "offset": offset},
                                timeout=(10, 20)).json()  # (connect, read)
        except requests.RequestException as e:
            print("poll error (retrying):", type(e).__name__)
            continue

        for upd in resp.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message") or {}
            chat_id = msg.get("chat", {}).get("id")
            text = msg.get("text")
            if not chat_id or not text:
                continue

            if text.startswith("/"):
                try:
                    handle_command(chat_id, text)
                except Exception as e:
                    print("command error:", e)
                continue

            text = text[:MAX_CHARS]
            print(f"[{chat_id}] {text[:60]}")
            try:
                chat_action(chat_id)
                ogg = synth_to_ogg(text, get_gen(chat_id))
                send_voice(chat_id, ogg)
                os.remove(ogg)
            except Exception as e:
                print("synth error:", e)
                send_message(chat_id, f"Ошибка генерации: {e}")


if __name__ == "__main__":
    main()
