"""``fujielab-asr``: transcribe audio files with the two-layer Zipformer models.

    fujielab-asr audio.wav [more.wav ...]           # offline model (default)
    fujielab-asr --model streaming audio.wav        # low-latency streaming model, fed 100 ms at a time
    fujielab-asr --model offline,streaming a.wav    # both, for comparison
    fujielab-asr --model <tag or local dir> a.wav   # any other two-layer model (offline / streaming from its config)
    fujielab-asr --format jsonl a.wav > out.jsonl   # one JSON object per segment

Each file is converted to 16 kHz mono, split at pauses into segments of at most
``--max-segment`` seconds, and recognized segment by segment. For each segment the
written form (``text``) and the pronunciation in katakana (``kana``) are printed,
with fillers as ``(F …)``, repairs / word fragments as ``(D …)`` and laughter as
``<笑>`` (CSJ notation).

The default models are gated on the Hugging Face Hub (research use only): request
access on their model pages and log in (``hf auth login``) before the first run.
Needs the ``cascade`` extra: ``pip install "fujielab-asr[cascade]"``.
"""
import argparse
import json
import os
import sys
import time
from math import gcd

import numpy as np

SR = 16000
CHUNK = SR // 10  # streaming models are fed 100 ms at a time
ALIASES = {
    "offline": "fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_offline_writaux_proncond",
    "streaming": "fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_stream_c80_c160_la0_writaux",
}


def load_audio(path):
    """-> 16 kHz mono float32 (wav / flac / mp3 / ... as far as soundfile reads them)."""
    import soundfile as sf

    x, fs = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if fs != SR:
        from scipy.signal import resample_poly

        g = gcd(fs, SR)
        x = resample_poly(x, SR // g, fs // g).astype(np.float32)
    return x


def split_on_silence(x, max_len=15.0, min_sil=0.3, margin=0.2, frame=0.01):
    """Simple pause-based segmentation -> [(start sample, end sample)].

    10 ms frames above a power threshold are speech; a pause of ``min_sil`` s or more ends
    a segment, and a segment longer than ``max_len`` s is cut at the quietest frame of its
    middle half. Each segment gets ``margin`` s of context on both sides.
    """
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


def load_model(tag, threads):
    """-> (name, recognizer, is_streaming). Offline or streaming is read from the model's config.json."""
    from fujielab.asr.icefall_ext import Speech2TextCascade, Speech2TextCascadeOffline
    from fujielab.asr.icefall_ext.cascade_zipformer import resolve_model_dir

    model_dir = resolve_model_dir(ALIASES.get(tag, tag))
    with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
        offline = json.load(f).get("mode") == "offline"
    cls = Speech2TextCascadeOffline if offline else Speech2TextCascade
    return cls.from_pretrained(model_dir, num_threads=threads), not offline


def recognize(s2t, streaming, seg):
    if not streaming:
        return s2t(seg)[0]
    r = None
    for i in range(0, len(seg), CHUNK):  # the last chunk (is_final=True) also resets the recognizer
        r = s2t.streaming_decode(seg[i:i + CHUNK], is_final=i + CHUNK >= len(seg))[0]
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(prog="fujielab-asr", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("audio", nargs="+", help="audio files (wav / flac / mp3 / ...)")
    ap.add_argument("--model", default="offline",
                    help="offline (default) / streaming / a Hugging Face tag / a local model directory; "
                         "several separated by commas")
    ap.add_argument("--max-segment", type=float, default=15.0, help="longest segment in seconds (default 15)")
    ap.add_argument("--format", choices=["text", "jsonl"], default="text")
    ap.add_argument("--threads", type=int, default=4, help="CPU threads (default 4)")
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(args.threads)
    names = [m.strip() for m in args.model.split(",") if m.strip()]
    models = {}
    for name in names:
        print(f"loading {ALIASES.get(name, name)}", file=sys.stderr)
        try:
            models[name] = load_model(name, args.threads)
        except PermissionError as e:
            sys.exit(f"error: {e}")
    width = max(len(n) for n in names)
    sec, total = dict.fromkeys(names, 0.0), 0.0

    for path in args.audio:
        x = load_audio(path)
        segs = split_on_silence(x, max_len=args.max_segment)
        total += sum(e - s for s, e in segs) / SR
        if args.format == "text":
            print(f"{path}: {len(x) / SR:.1f} s, {len(segs)} segments")
        for k, (s, e) in enumerate(segs, 1):
            if args.format == "text":
                print(f"\n#{k}  {s / SR:.2f}-{e / SR:.2f} s")
            for name in names:
                s2t, streaming = models[name]
                t = time.perf_counter()
                r = recognize(s2t, streaming, x[s:e])
                sec[name] += time.perf_counter() - t
                if args.format == "jsonl":
                    print(json.dumps({"file": path, "segment": k, "start": round(s / SR, 2), "end": round(e / SR, 2),
                                      "model": name, "text": r.tagged_text, "kana": r.tagged_kana,
                                      "text_plain": r.text, "kana_plain": r.kana}, ensure_ascii=False))
                else:
                    label = f"{name:<{width}s} " if len(names) > 1 else ""
                    print(f"  {label}text: {r.tagged_text}")
                    print(f"  {label}kana: {r.tagged_kana}")
        if args.format == "text":
            print()
    if total > 0:
        for name in names:
            print(f"{name}: {sec[name]:.1f} s for {total:.1f} s of speech (RTF {sec[name] / total:.2f})", file=sys.stderr)


if __name__ == "__main__":
    main()
