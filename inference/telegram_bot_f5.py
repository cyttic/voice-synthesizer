"""
Telegram bot serving the fine-tuned F5-TTS lecturer voice (MIT-licensed model).
Same plumbing as telegram_bot.py (XTTS), different backend: F5-TTS is a flow
matching model, so it conditions on a reference wav + its transcript instead of
a speaker embedding, and its knobs are speed / nfe (denoising steps) / cfg.

Setup:
    export TELEGRAM_TOKEN=123:ABC    # from @BotFather
    export F5_MODEL_DIR=model-f5     # model_inference.pt, vocab.txt,
                                     # reference.wav, reference.txt

Run:
    python inference/telegram_bot_f5.py
"""

import json
import os
import re
import subprocess
import tempfile
import time

import requests
import soundfile as sf
import torch
import torchaudio

# torchcodec (torchaudio's default decoder) is broken under this machine's
# CUDA setup. Load audio via soundfile so F5-TTS never touches it.
def _sf_load(path, *args, **kwargs):
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    return torch.from_numpy(data.T), sr

torchaudio.load = _sf_load

from f5_tts.api import F5TTS

# --- config ---
TOKEN     = os.environ.get("TELEGRAM_TOKEN")
MODEL_DIR = os.environ.get("F5_MODEL_DIR", "model-f5")
DEVICE    = os.environ.get("F5_DEVICE", "cuda")
EXP_NAME  = os.environ.get("F5_EXP_NAME", "F5TTS_v1_Base")  # must match the ckpt
MAX_CHARS = int(os.environ.get("F5_MAX_CHARS", "1000"))

API = f"https://api.telegram.org/bot{TOKEN}"
REF_WAV = os.path.join(MODEL_DIR, "reference.wav")
REF_TXT = open(os.path.join(MODEL_DIR, "reference.txt"), encoding="utf-8").read().strip()
SETTINGS_FILE = os.environ.get("F5_SETTINGS_FILE", "user_settings_f5.json")

DEFAULTS = dict(speed=1.0, nfe_step=32, cfg_strength=2.0)

# User command -> (param name, min, max).
TUNABLE = {
    "speed": ("speed", 0.5, 1.5),
    "nfe":   ("nfe_step", 16, 64),       # more steps = cleaner, slower
    "cfg":   ("cfg_strength", 1.0, 4.0), # text adherence vs naturalness
}

# --- per-user settings (persisted to JSON) ---
user_settings: dict = {}

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
    g = dict(DEFAULTS)
    g.update(user_settings.get(str(chat_id), {}))
    g["nfe_step"] = int(g["nfe_step"])
    return g


# --- model (loaded once at startup) ---

print(f"Loading F5-TTS from {MODEL_DIR} on {DEVICE}...")
tts = F5TTS(
    model=EXP_NAME,
    ckpt_file=os.path.join(MODEL_DIR, "model_inference.pt"),
    vocab_file=os.path.join(MODEL_DIR, "vocab.txt"),
    device=DEVICE,
)
print("Model ready.")


# --- synthesis ---

def synth_to_ogg(text: str, gen: dict) -> str:
    """Generate speech and return a path to a Telegram-ready OGG/Opus file."""
    t0 = time.time()
    wav, sr, _ = tts.infer(
        ref_file=REF_WAV,
        ref_text=REF_TXT,
        gen_text=text,
        speed=gen["speed"],
        nfe_step=gen["nfe_step"],
        cfg_strength=gen["cfg_strength"],
        remove_silence=True,
    )
    gen_s = time.time() - t0
    audio_s = len(wav) / sr
    ratio = gen_s / audio_s if audio_s else 0
    print(f"  chars={len(text)}  gen={gen_s:.1f}s  audio={audio_s:.1f}s  "
          f"(x{ratio:.1f} realtime)")

    wav_fd, wav_path = tempfile.mkstemp(suffix=".wav")
    ogg_fd, ogg_path = tempfile.mkstemp(suffix=".ogg")
    os.close(wav_fd); os.close(ogg_fd)

    sf.write(wav_path, wav, sr)
    subprocess.run(
        ["ffmpeg", "-y", "-i", wav_path, "-c:a", "libopus", "-b:a", "32k", "-ac", "1", ogg_path],
        check=True, capture_output=True,
    )
    os.remove(wav_path)
    return ogg_path


# --- Telegram API helpers ---

def send_message(chat_id, text):
    requests.post(f"{API}/sendMessage", data={"chat_id": chat_id, "text": text},
                  timeout=(10, 20))

def send_voice(chat_id, ogg_path):
    with open(ogg_path, "rb") as f:
        requests.post(f"{API}/sendVoice", data={"chat_id": chat_id},
                      files={"voice": f}, timeout=(10, 60))

def chat_action(chat_id, action="record_voice"):
    requests.post(f"{API}/sendChatAction", data={"chat_id": chat_id, "action": action},
                  timeout=(10, 15))

def register_commands():
    commands = [
        {"command": "speed",    "description": "Скорость речи 0.5–1.5"},
        {"command": "nfe",      "description": "Качество/скорость 16–64 шагов"},
        {"command": "cfg",      "description": "Чёткость текста 1–4"},
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
    "• /speed 1.0 — скорость речи (0.5–1.5)\n"
    "• /nfe 32 — шагов генерации, больше = чище но медленнее (16–64)\n"
    "• /cfg 2 — чёткость следования тексту (1–4)\n"
    "• /settings — показать твои настройки\n"
    "• /reset — сбросить к стандартным"
)

def fmt_settings(chat_id) -> str:
    g = get_gen(chat_id)
    return (
        "Твои текущие настройки:\n"
        f"• speed = {g['speed']}  (/speed 0.5–1.5)\n"
        f"• nfe_step = {g['nfe_step']}  (/nfe 16–64)\n"
        f"• cfg_strength = {g['cfg_strength']}  (/cfg 1–4)\n"
        "/reset — сбросить к стандартным"
    )

def handle_command(chat_id, text) -> None:
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
    me = requests.get(f"{API}/getMe", timeout=(10, 20)).json()
    print("Bot online:", me.get("result", {}).get("username"))
    print("Polling for messages... (Ctrl+C to stop)")

    offset = None
    while True:
        try:
            resp = requests.get(f"{API}/getUpdates",
                                params={"timeout": 15, "offset": offset},
                                timeout=(10, 20)).json()
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
            ogg = None
            try:
                chat_action(chat_id)
                ogg = synth_to_ogg(text, get_gen(chat_id))
                send_voice(chat_id, ogg)
            except Exception as e:
                print("synth error:", e)
                send_message(chat_id, f"Ошибка генерации: {e}")
            finally:
                if ogg and os.path.isfile(ogg):
                    os.remove(ogg)


if __name__ == "__main__":
    main()
