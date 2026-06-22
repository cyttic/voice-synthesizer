# model-piper

Output of `notebooks/kaggle_finetune_piper.ipynb` — the **fast / realtime** voice.

After the Kaggle run, the trained Piper voice lives on HF
(`cyttic/lecturer-ru-piper`) as two files; drop them here to serve locally:

```
model-piper/
├── lecturer_ru.onnx        # the voice
└── lecturer_ru.onnx.json   # its config (sample rate, phoneme map, ...)
```

Both are gitignored (large/binary; the source of truth is the HF repo).

## Serve
This voice plugs straight into the realtime call pipeline:

```bash
# in clone-telegram/voice-agent/
cp /path/model-piper/lecturer_ru.onnx*  voices/
export VA_TTS_VOICE=voices/lecturer_ru.onnx
python server.py
```

`voice-agent/tts.py` already wraps `PiperVoice`, so no new serving code is
needed. A `telegram_bot_piper.py` (mirroring `telegram_bot_f5.py`) can be added
once the voice is auditioned.

> Needs `espeak-ng` installed on the serving machine (Russian phonemization).
