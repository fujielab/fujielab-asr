"""Streaming multitask RNN-Transducer: transcript + per-token filler/repair labels.

Unlike the CTC multitask model (see run_streaming_asr_multitask.py), the base
tokens here come from a Transducer beam search, and each token's auxiliary
label is read from the joint network's hidden state at the exact lattice node
the search emitted that token at. Partial results arrive on every chunk.
"""

from os import path

import numpy as np
import soundfile as sf
from scipy.signal import resample

from fujielab.asr.espnet_ext.espnet2.bin.asr_multitask_transducer_inference_cbs import (
    Speech2TextMultitaskTransducer,
)

# Japanese kana ASR + auxiliary-information head (N/F/D), CSJ + CEJC,
# CSJ-aligned filler criterion, 50 epochs.
model_name = "fujielab/espnet_asr_csj_cejc_pron_aux_cbs_transducer_120300_hop132"

s2t = Speech2TextMultitaskTransducer.from_pretrained(
    model_name,
    beam_size=20,
    beam_search_config=dict(search_type="maes"),
)

# Sample audio from https://clrd.ninjal.ac.jp/csj/sound-f/aps-smp.mp3
if not path.exists("aps-smp.mp3"):
    import requests

    r = requests.get("https://clrd.ninjal.ac.jp/csj/sound-f/aps-smp.mp3")
    r.raise_for_status()
    with open("aps-smp.mp3", "wb") as f:
        f.write(r.content)

audio, fs = sf.read("aps-smp.mp3")
if audio.ndim > 1:
    audio = audio.mean(axis=1)
if fs != 16000:
    audio = resample(audio, int(len(audio) * 16000 / fs))
    fs = 16000

chunk = int(fs * 0.1)  # 100 ms
n = len(audio)
result = None

for i in range(0, n, chunk):
    c = audio[i : i + chunk]
    # The last chunk must be flagged even when it happens to be exactly full,
    # otherwise the recognizer is never told to flush.
    is_final = i + chunk >= n
    if len(c) < chunk:
        c = np.pad(c, (0, chunk - len(c)))

    results = s2t.streaming_decode(c, is_final=is_final)
    if results:
        result = results[0]

if result is None:
    print("(nothing recognized)")
else:
    # tokens and aux_labels are aligned 1:1; aux is one of N / F / D
    print(
        " ".join(
            f"{t}[{a}]" if a != "N" else t
            for t, a in zip(result.tokens, result.aux_labels)
        )
    )
    n_f = result.aux_labels.count("F")
    n_d = result.aux_labels.count("D")
    print(f"\n{len(result.tokens)} tokens, {n_f} filler, {n_d} repair")
