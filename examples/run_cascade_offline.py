#!/usr/bin/env python3
"""Transcribe audio files with the offline two-layer Cascade model (X15ev) on the CPU.

The file is cut at pauses into segments of at most --max-len seconds (a simple
energy-based splitter; the model was trained on utterances of up to 30 s), and each
segment is recognized as a whole. Per segment it prints layer 1 (phonemes as
katakana, fillers / repairs in brackets) and layer 2 (the written form in CSJ
notation: fillers (F えー), repairs and word fragments (D こ), laughter <笑> / (笑 …)).

    pip install "fujielab-asr[cascade]"
    python examples/run_cascade_offline.py examples/aps-smp.mp3
    python examples/run_cascade_offline.py a.wav b.flac --out result.jsonl   # also save every symbol with its time
    python examples/run_cascade_offline.py a.wav --model <local model directory>
"""
import argparse
import json
import os
from math import gcd

import numpy as np
import torch

from fujielab.asr.icefall_ext import Speech2TextCascadeOffline
from fujielab.asr.icefall_ext.cascade_zipformer import SR

MODEL = "fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_offline_writaux_proncond"


def read_audio(path):
    """Mono float32 at 16 kHz."""
    import soundfile as sf

    x, fs = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != SR:
        from scipy.signal import resample_poly

        g = gcd(fs, SR)
        x = resample_poly(x, SR // g, fs // g).astype(np.float32)
    return x


def split_on_silence(x, max_len=20.0, min_sil=0.3, margin=0.2, frame=0.01):
    """Cut at pauses of min_sil s or more (10 ms frame power against a threshold set from the file itself);
    a segment longer than max_len s is cut again at its quietest frame in the middle half.
    -> [(start sample, end sample)], each widened by `margin` s on both sides."""
    n = int(SR * frame)
    m = len(x) // n
    if m == 0:
        return [(0, len(x))] if len(x) else []
    db = 10 * np.log10(np.mean(x[:m * n].reshape(m, n) ** 2, axis=1) + 1e-10)
    voiced = db > max(np.percentile(db, 10) + 10, db.max() - 50)
    segs, start, sil = [], None, 0
    for i, v in enumerate(voiced):
        if v:
            start = i if start is None else start
            sil = 0
        elif start is not None:
            sil += 1
            if sil * frame >= min_sil:
                segs.append((start, i - sil + 1))
                start, sil = None, 0
    if start is not None:
        segs.append((start, m - sil))

    def cut(a, b):
        if (b - a) * frame <= max_len:
            return [(a, b)]
        lo, hi = a + (b - a) // 4, b - (b - a) // 4
        c = lo + int(np.argmin(db[lo:hi]))
        return cut(a, c) + cut(c, b)

    pad = int(margin * SR)
    return [(max(0, a * n - pad), min(len(x), b * n + pad))
            for s in segs for a, b in cut(*s) if (b - a) * frame >= 0.1]


def kana_tagged(s2t, r):
    """Layer 1 as katakana with fillers / repairs in (F …) / (D …)."""
    items, _ = s2t.p2k.convert(r.tokens)
    out, cur = [], ""
    for s, lab in items:
        lab = lab if lab in ("F", "D") else ""
        if lab != cur:
            out.append(")" if cur else "")
            out.append(f"({lab} " if lab else "")
            cur = lab
        out.append(s)
    if cur:
        out.append(")")
    return "".join(out).strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+", help="audio files (wav / flac / mp3 / ...; resampled to 16 kHz)")
    ap.add_argument("--model", default=MODEL, help="Hugging Face tag or local model directory")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads")
    ap.add_argument("--max-len", type=float, default=20.0, help="longest segment (s)")
    ap.add_argument("--out", help="write every segment's symbols and times to this JSONL file")
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    s2t = Speech2TextCascadeOffline.from_pretrained(args.model, num_threads=args.threads)
    fo = open(args.out, "w", encoding="utf-8") if args.out else None
    total = 0.0
    for path in args.audio:
        x = read_audio(path)
        segs = split_on_silence(x, max_len=args.max_len)
        print(f"{path}: {len(x) / SR:.1f} s, {len(segs)} segments")
        for s, e in segs:
            r = s2t(x[s:e])[0]
            total += (e - s) / SR
            print(f"\n[{s / SR:7.2f} - {e / SR:7.2f}]")
            print(f"  layer 1: {kana_tagged(s2t, r)}")
            print(f"  layer 2: {r.tagged_text}")
            if fo:
                t0 = s / SR
                fo.write(json.dumps({
                    "file": os.path.basename(path), "start": round(t0, 3), "end": round(e / SR, 3),
                    "text": r.text, "tagged_text": r.tagged_text, "kana": r.kana,
                    "tokens": r.tokens, "timestamps": [round(t0 + t, 3) for t in r.timestamps],
                    "graphemes": r.graphemes, "grapheme_timestamps": [round(t0 + t, 3) for t in r.grapheme_timestamps],
                }, ensure_ascii=False) + "\n")
        print()
    if fo:
        fo.close()
    if total:
        print(f"{total:.1f} s of speech in {s2t.compute:.1f} s (real-time factor {s2t.compute / total:.3f}, {args.threads} threads)")


if __name__ == "__main__":
    main()
