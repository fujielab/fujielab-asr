"""Streaming Zipformer transducer (icefall) via sherpa-onnx: CPU real-time, 320 ms chunks.

    pip install "fujielab-asr[sherpa]"
    python examples/run_streaming_asr_sherpa.py [audio.wav] [--int8] [--chunk_ms 320|640] [--threads 1]

Prints the partial hypothesis after every 100 ms of audio and the final one at the end.
"""
import argparse
import sys
import time
from os import path

import numpy as np
import soundfile as sf

from fujielab.asr.sherpa_onnx_ext import Speech2TextSherpaOnnx

ap = argparse.ArgumentParser()
ap.add_argument("audio", nargs="?", default="aps-smp.mp3")
ap.add_argument("--model", default="fujie/sherpa_onnx_asr_csj_cejc_pron_comp_zipformer_stream_c320_la0")
ap.add_argument("--chunk_ms", type=int, default=320, help="encoder chunk of the exported model: 320 or 640")
ap.add_argument("--int8", action="store_true")
ap.add_argument("--threads", type=int, default=1)
ap.add_argument("--beam", action="store_true", help="modified beam search (4 paths) instead of greedy")
a = ap.parse_args()

s2t = Speech2TextSherpaOnnx.from_pretrained(
    a.model,
    chunk_ms=a.chunk_ms,
    int8=a.int8,
    num_threads=a.threads,
    decoding_method="modified_beam_search" if a.beam else "greedy_search",
)

# Sample audio (CSJ excerpt) if nothing else is given.
if a.audio == "aps-smp.mp3" and not path.exists(a.audio):
    import requests

    r = requests.get("https://clrd.ninjal.ac.jp/csj/sound-f/aps-smp.mp3")
    r.raise_for_status()
    open(a.audio, "wb").write(r.content)

audio, fs = sf.read(a.audio, dtype="float32")
if audio.ndim == 2:
    audio = audio.mean(axis=1)
if fs != 16000:
    from scipy.signal import resample

    audio = resample(audio, int(len(audio) * 16000 / fs)).astype(np.float32)
    fs = 16000

step = int(fs * 0.1)  # 100 ms
n = len(audio)
t_compute = 0.0
result = None
for i in range(0, n, step):
    chunk = audio[i : i + step]
    is_final = i + step >= n  # flag the last chunk by position, not by its length
    t0 = time.perf_counter()
    result = s2t.streaming_decode(chunk, is_final=is_final)[0]
    t_compute += time.perf_counter() - t0
    sys.stdout.write("\r" + result.text[-100:].ljust(100))
    sys.stdout.flush()
print()
print("tokens     :", " ".join(result.tokens))
print("aux labels :", " ".join(f"{t}[{l}]" if l != "N" else t for t, l in zip(result.base_tokens, result.aux_labels)))
print("transcript :", result.transcript)
print(f"audio {n / fs:.1f} s, compute {t_compute:.2f} s, RTF {t_compute / (n / fs):.3f} "
      f"({'int8' if a.int8 else 'fp32'}, {a.threads} thread(s), chunk {a.chunk_ms} ms)")
