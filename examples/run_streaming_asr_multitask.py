"""Streaming CTC multitask ASR demo (recognition + auxiliary-information labels).

Recognizes Japanese kana tokens with a streaming CTC model and, for every token,
predicts an auxiliary-information label:
  N = normal, F = filler (フィラー), D = repair / disfluency (言い直し).

The model is the CTC-only multitask counterpart of the transducer models used in
``run_streaming_asr.py``; instead of a beam search it runs chunk-by-chunk greedy
CTC and reads the auxiliary head at each token's emission frames.
"""

import numpy as np
import soundfile as sf
from os import path
from scipy.signal import resample

from fujielab.asr.espnet_ext.espnet2.bin.asr_multitask_ctc_inference_cbs import (
    Speech2TextMultitaskCTC,
)

# Japanese Kana + per-token auxiliary information (N/F/D) model, trained on CSJ.
# CBS Transformer encoder (left context 12 / main block 3 / look-ahead 0).
model_name = "fujie/espnet_asr_csj_pron_aux_cbs_ctc_120300_hop132"

s2t = Speech2TextMultitaskCTC.from_pretrained(model_name, streaming=True)

# Download a sample audio file if it does not exist.
# The audio file is from https://clrd.ninjal.ac.jp/csj/sound-f/aps-smp.mp3
if not path.exists("aps-smp.mp3"):
    import requests

    url = "https://clrd.ninjal.ac.jp/csj/sound-f/aps-smp.mp3"
    response = requests.get(url)
    response.raise_for_status()
    with open("aps-smp.mp3", "wb") as f:
        f.write(response.content)

audio, fs = sf.read("aps-smp.mp3")
if fs != 16000:
    num_samples = len(audio)
    audio = resample(audio, int(num_samples * 16000 / fs))
    fs = 16000

num_samples = len(audio)
chunk_size = int(16000 * 1 / 10)  # 100 ms

final_result = None
for i in range(0, num_samples, chunk_size):
    chunk = audio[i : i + chunk_size]
    is_final = False
    if len(chunk) < chunk_size:
        chunk = np.pad(chunk, (0, chunk_size - len(chunk)), "constant")
        is_final = True
    results = s2t.streaming_decode(chunk, is_final=is_final)
    if results:
        r = results[0]
        # token[aux] view (only show non-normal aux to keep it readable)
        view = " ".join(
            tok if aux == "N" else f"{tok}[{aux}]"
            for tok, aux in zip(r.tokens, r.aux_labels)
        )
        print(view)
        final_result = r

if final_result is not None:
    print("\n=== Final ===")
    print("Text:", "".join(final_result.tokens))
    fillers = [t for t, a in zip(final_result.tokens, final_result.aux_labels) if a == "F"]
    repairs = [t for t, a in zip(final_result.tokens, final_result.aux_labels) if a == "D"]
    print("Fillers (F):", " ".join(fillers) if fillers else "(none)")
    print("Repairs (D):", " ".join(repairs) if repairs else "(none)")
