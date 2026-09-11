#!/usr/bin/env python3
"""CPU real-time-factor benchmark for the streaming Transducer (CBS) models.

Feeds 100 ms chunks through Speech2Text.streaming_decode exactly as a live client
would, and reports RTF (= compute time / audio time) and per-chunk latency.

    python examples/bench_streaming_rtf.py --config exp/.../config.yaml --model exp/.../valid.cer_transducer.ave_10best.pth \
        --wav_scp dump/raw/cejc_eval10m/wav.scp --n_utts 30 --threads 1 --beam_size 20
"""
import argparse, time, statistics as st
import numpy as np, soundfile as sf, torch

from fujielab.asr.espnet_ext.espnet2.bin.asr_transducer_inference_cbs import Speech2Text

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True); ap.add_argument("--model", required=True)
ap.add_argument("--wav_scp", action="append", required=True); ap.add_argument("--n_utts", type=int, default=30)
ap.add_argument("--threads", type=int, default=1); ap.add_argument("--beam_size", type=int, default=20)
ap.add_argument("--chunk_ms", type=int, default=100); ap.add_argument("--text", action="append", default=[])
a = ap.parse_args()
torch.set_num_threads(a.threads); torch.set_num_interop_threads(1)
s2t = Speech2Text(asr_train_config=a.config, asr_model_file=a.model, device="cpu", streaming=True,
                  lm_weight=0.0, beam_size=a.beam_size, beam_search_config=dict(search_type="maes"))
utts = []
for scp in a.wav_scp:
    utts += [l.split() for l in open(scp)][: a.n_utts]
refs = {}
for t in a.text:
    refs.update({l.split()[0]: l.split()[1:] for l in open(t, encoding="utf-8")})
chunk = int(16000 * a.chunk_ms / 1000)
audio_s = 0.0; compute_s = 0.0; lat = []; hyps_out = {}
# warm-up
x, _ = sf.read(utts[0][1]); s2t.reset_streaming_cache()
for i in range(0, len(x), chunk):
    c = x[i:i+chunk]; c = np.pad(c, (0, chunk-len(c))) if len(c) < chunk else c
    s2t.streaming_decode(c, is_final=i+chunk >= len(x))
for u, p in utts:
    x, fs = sf.read(p); assert fs == 16000
    s2t.reset_streaming_cache(); audio_s += len(x) / 16000; result = None
    for i in range(0, len(x), chunk):
        c = x[i:i+chunk]; c = np.pad(c, (0, chunk-len(c))) if len(c) < chunk else c
        t0 = time.perf_counter(); h = s2t.streaming_decode(c, is_final=i+chunk >= len(x)); dt = time.perf_counter() - t0
        compute_s += dt; lat.append(dt)
        if h: result = s2t.hypotheses_to_results(h)
    hyps_out[u] = result[0][1] if result else []
lat_ms = sorted(v * 1000 for v in lat)
print(f"threads={a.threads} beam={a.beam_size} chunk={a.chunk_ms}ms utts={len(utts)} audio={audio_s:.1f}s compute={compute_s:.1f}s "
      f"RTF={compute_s/audio_s:.3f} chunk latency ms: mean {st.mean(lat_ms):.1f} p50 {lat_ms[len(lat_ms)//2]:.1f} p95 {lat_ms[int(len(lat_ms)*0.95)]:.1f} max {lat_ms[-1]:.1f}")
if refs:
    import editdistance
    def strip(t): return "".join(x.split("+")[0] for x in t if x != "<sp>")
    e = n = 0
    for u, h in hyps_out.items():
        r = strip(refs[u]); e += editdistance.eval(r, strip(h)); n += len(r)
    print(f"  CER on these utts (aux stripped): {100*e/n:.2f}%")
