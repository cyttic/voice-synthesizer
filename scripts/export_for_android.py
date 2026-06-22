#!/usr/bin/env python3
"""
Export the RVC model to the two ONNX files KrimbaGram's on-device voice changer needs:
  generator.onnx  - your fine-tuned voice (from lecturer_ru.pth)
  hubert.onnx     - the shared ContentVec/HuBERT v2 encoder (vec-768-layer-12, downloaded)

Run inside the working RVC environment (the Kaggle Python-3.10 env, or a local uv 3.10 env):

    /kaggle/working/rvc310/bin/python scripts/export_for_android.py \
        --rvc /kaggle/working/rvc \
        --pth /kaggle/working/rvc/assets/weights/lecturer_ru.pth \
        --out /kaggle/working/android_models

Then copy the two files in --out to the phone at: <app filesDir>/voicechanger/
(hubert.onnx + generator.onnx). On the device that's typically
/data/data/org.telegram.messenger.beta/files/voicechanger/ — push via:
    adb push generator.onnx /sdcard/Download/   (then move in-app or via a file manager)
"""

import argparse
import os
import sys
import urllib.request

VEC_URL = "https://huggingface.co/NaruseMioShirakana/MoeSS-SUBModel/resolve/main/vec-768-layer-12.onnx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rvc", default="/kaggle/working/rvc", help="RVC-WebUI repo root")
    ap.add_argument("--pth", default="/kaggle/working/rvc/assets/weights/lecturer_ru.pth")
    ap.add_argument("--out", default="/kaggle/working/android_models")
    a = ap.parse_args()

    os.makedirs(a.out, exist_ok=True)
    os.chdir(a.rvc)
    sys.path.insert(0, a.rvc)

    # RVC checkpoints predate PyTorch 2.6's weights_only=True default
    import torch
    _ol = torch.load
    torch.load = lambda *args, **kw: _ol(*args, **{**kw, "weights_only": False})

    # 1) generator -> ONNX
    gen = os.path.join(a.out, "generator.onnx")
    from infer.modules.onnx.export import export_onnx
    print("Exporting generator ->", gen)
    print(" ", export_onnx(a.pth, gen))

    # 2) hubert / ContentVec v2 (768, layer 12) -> download prebuilt ONNX
    hub = os.path.join(a.out, "hubert.onnx")
    if not os.path.exists(hub):
        print("Downloading ContentVec ONNX ->", hub)
        urllib.request.urlretrieve(VEC_URL, hub)

    # 3) print signatures so the Android side can be matched/verified
    import onnxruntime as ort
    for name, p in [("hubert", hub), ("generator", gen)]:
        s = ort.InferenceSession(p, providers=["CPUExecutionProvider"])
        print(f"\n== {name}.onnx ==")
        print("  INPUTS :", [(i.name, i.shape, i.type) for i in s.get_inputs()])
        print("  OUTPUTS:", [(o.name, o.shape, o.type) for o in s.get_outputs()])

    print("\nDONE. Copy these to the phone's <filesDir>/voicechanger/:")
    print("   ", gen)
    print("   ", hub)


if __name__ == "__main__":
    main()
