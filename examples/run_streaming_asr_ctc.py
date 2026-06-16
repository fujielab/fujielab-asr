"""Streaming CTC ASR demo where the auxiliary information is in the tokens.

For the composite / span encodings, fillers (F) and repairs (D) are part of the
recognized token sequence (no separate auxiliary head), so a plain CTC model is
enough. This demo recognizes Japanese kana and extracts the marked fillers/repairs.

  - composite tokens: "あ+F" (filler) / "チ+D" (repair)
  - span tokens:      "<F> あ の </F>" / "<D> ... </D>"
"""

import numpy as np
import soundfile as sf
from os import path
from scipy.signal import resample

from fujielab.asr.espnet_ext.espnet2.bin.asr_ctc_inference_cbs import Speech2TextCTC

# Composite-token model (markers are token suffixes "+F"/"+D"). CSJ kana, CBS
# Transformer encoder (left context 12 / main block 3 / look-ahead 0).
model_name = "fujie/espnet_asr_csj_pron_comp_cbs_ctc_120300_hop132"
# Span-token model alternative:
# model_name = "fujie/espnet_asr_csj_pron_span_cbs_ctc_120300_hop132"

s2t = Speech2TextCTC.from_pretrained(model_name, streaming=True)

if not path.exists("aps-smp.mp3"):
    import requests

    url = "https://clrd.ninjal.ac.jp/csj/sound-f/aps-smp.mp3"
    r = requests.get(url)
    r.raise_for_status()
    open("aps-smp.mp3", "wb").write(r.content)

audio, fs = sf.read("aps-smp.mp3")
if fs != 16000:
    audio = resample(audio, int(len(audio) * 16000 / fs))
    fs = 16000

chunk = int(16000 * 0.1)  # 100 ms
final = None
for i in range(0, len(audio), chunk):
    c = audio[i : i + chunk]
    is_final = False
    if len(c) < chunk:
        c = np.pad(c, (0, chunk - len(c)), "constant")
        is_final = True
    res = s2t.streaming_decode(c, is_final=is_final)
    if res:
        final = res[0]
        print("".join(final.tokens))

if final is not None:
    print("\n=== Final ===")
    print("Tokens:", " ".join(final.tokens))
    fillers = [t for t in final.tokens if t.endswith("+F") or t in ("<F>",)]
    repairs = [t for t in final.tokens if t.endswith("+D") or t in ("<D>",)]
    print("Fillers (F):", " ".join(fillers) if fillers else "(none)")
    print("Repairs (D):", " ".join(repairs) if repairs else "(none)")
