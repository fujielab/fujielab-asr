#!/usr/bin/env python3
"""Live demo of the two-layer, two-rate Cascade recognizer on the CPU (microphone or audio file).

Layer 1 (phonemes, confirmed every 160 ms) is shown as katakana as it arrives,
next to layer 2 (kana-kanji, every 640 ms). Fillers (+F) are orange, repairs and
word fragments (+D) magenta, laughter (Event-DP model only) green.

    pip install "fujielab-asr[cascade]" sounddevice soxr rich
    python examples/run_cascade_live.py                    # default input device (microphone)
    python examples/run_cascade_live.py --list-devices     # list input devices
    python examples/run_cascade_live.py --device 2         # pick a device by number or part of its name
    python examples/run_cascade_live.py --wav a.wav        # stream a file in real time (--fast: full speed)
    python examples/run_cascade_live.py --wav a.wav --plain   # just print phonemes / kana / text per file
    python examples/run_cascade_live.py --bench            # measure speed only (20 s of noise at full speed)

On a Mac, the terminal app asks for microphone access on the first run (System
Settings > Privacy & Security > Microphone). If the status line says processing
is falling behind, check the real-time factor with --bench first.
"""
import argparse
import os
import queue
import subprocess
import sys
import time

import numpy as np
import torch

from fujielab.asr.icefall_ext import P2Kana, Speech2TextCascade
from fujielab.asr.icefall_ext.cascade_zipformer import FRAME_S, SR

MODEL = "fujielab/icefall_asr_csj_cejc_reazon_medium_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0"  # Cascade+Event-DP + ReazonSpeech
MODEL_X10A = "fujielab/icefall_asr_csj_cejc_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0"  # CSJ+CEJC only
MODEL_X6 = "fujielab/icefall_asr_csj_cejc_phn_writ_comp_cascade_zipformer_stream_c160_c640_la0"  # no events


# ----------------------------------------------------------------------------- display
STYLE = {"": "", "F": "orange1", "D": "magenta", "laugh": "bold green", "L": "green", "raw": "dim"}


class Segments:
    """Collects both layers' symbols per stretch of speech (split at pauses)."""

    def __init__(self, p2k: P2Kana, chunk2: int, pause: float):
        self.p2k, self.chunk2, self.pause = p2k, chunk2, pause
        self.p, self.g = [], []
        self.t_start = None
        self.last_p = -10 ** 9

    def add(self, ev1, ev2):
        for f, tok, _ in ev1:
            if self.t_start is None:
                self.t_start = f * FRAME_S
            self.p.append(tok)
            self.last_p = f
        self.g += [tok for _, tok, _ in ev2]

    def should_close(self, l1_frames: int, l2_frames: int) -> bool:
        """Close when layer 1 has been silent for `pause` s and layer 2 has read the chunk at that position."""
        return bool(self.p) and (l1_frames - self.last_p) * FRAME_S >= self.pause and l2_frames >= self.last_p + self.chunk2

    def kana(self):
        from rich.text import Text

        items, pending = self.p2k.convert(self.p)
        t = Text()
        for s, lab in items:
            t.append(s, style=STYLE.get(lab, ""))
        if pending:
            t.append("".join(pending), style="dim italic")
        return t

    def grapheme(self):
        return "".join(t for t in self.g if not t.startswith("<"))

    def reset(self):
        self.p, self.g, self.t_start = [], [], None


def tail(text, n):
    """Last n characters of a rich.Text / str (long utterances lose their head in the live view)."""
    return text if len(text) <= n else text[len(text) - n:]


def run_live(s2t: Speech2TextCascade, blocks, pause=1.2, realtime_src=True):
    from rich.console import Console, Group
    from rich.live import Live
    from rich.table import Table
    from rich.text import Text

    con = Console()
    seg = Segments(s2t.p2k, s2t.chunk2, pause)
    n_seg = 0
    level = -90.0
    t_wall0 = time.perf_counter()

    def rows(kana, graph):
        # Japanese has no spaces: labels and text in separate columns, wrapped inside the text column
        t = Table.grid(padding=(0, 1))
        t.add_column(no_wrap=True, width=5)
        t.add_column(overflow="fold")
        t.add_row(Text(" kana", style="bold"), kana)
        t.add_row(Text(" text", style="bold"), Text(graph))
        return t

    def finished_lines(s, k):
        head = Text(f"#{k}  {s.t_start:7.1f} s", style="bold cyan")
        return Group(head, rows(s.kana(), s.grapheme()), Text(""))

    def render():
        w = max(20, con.width - 8)
        audio = s2t.samples / SR
        rtf = s2t.compute / audio if audio > 0 else 0.0
        backlog = (time.perf_counter() - t_wall0) - audio if realtime_src else 0.0
        bar = "█" * int(max(0, min(20, (level + 60) / 3)))
        status = Text(f"● {audio:7.1f} s   RTF {rtf:4.2f}   input {bar:<20s} {level:5.1f} dBFS", style="dim")
        if realtime_src and backlog > 0.5:
            status.append(f"   processing is {backlog:.1f} s behind (check speed with --bench)", style="bold red")
        return Group(
            Text("─" * min(con.width, 100), style="dim"),
            rows(tail(seg.kana(), 2 * w), tail(seg.grapheme(), 2 * w)),
            status,
        )

    con.print(f"[bold]Cascade live demo[/bold]  model: {s2t.cfg['name']}  "
              f"(layer 1 every {s2t.cfg['chunk_size'] * 20} ms, layer 2 every {s2t.chunk2 * 40} ms)")
    con.print("[dim]Layer-1 phonemes shown as katakana ([orange1]filler[/orange1], [magenta]repair[/magenta], "
              "[green]laughter[/green]). Ctrl+C to quit.[/dim]\n")
    try:
        with Live(render(), console=con, refresh_per_second=12, transient=True) as live:
            for x in blocks:
                if x is None:
                    live.update(render())
                    continue
                level = 20 * np.log10(np.sqrt(np.mean(x.astype(np.float64) ** 2)) + 1e-9)
                ev1, ev2 = s2t.accept(x)
                seg.add(ev1, ev2)
                if seg.should_close(s2t.l1_frames, s2t.l2_frames):
                    n_seg += 1
                    live.console.print(finished_lines(seg, n_seg))
                    seg.reset()
                live.update(render())
    except KeyboardInterrupt:
        pass
    seg.add(*s2t.flush())
    if seg.p or seg.g:
        n_seg += 1
        con.print(finished_lines(seg, n_seg))
    audio = s2t.samples / SR
    if audio > 0:
        con.print(f"[dim]audio {audio:.1f} s, compute {s2t.compute:.1f} s (RTF {s2t.compute / audio:.3f})[/dim]")


# ----------------------------------------------------------------------------- input
def mic_blocks(device, block_sec=0.1):
    import sounddevice as sd

    info = sd.query_devices(device, "input")
    rate = int(info["default_samplerate"])
    try:
        sd.check_input_settings(device=device, samplerate=SR, channels=1, dtype="float32")
        rate = SR
    except Exception:
        pass
    rs = None
    if rate != SR:
        import soxr

        rs = soxr.ResampleStream(rate, SR, 1, dtype="float32")
    print(f"input: {info['name']} ({rate} Hz{' -> 16 kHz' if rs else ''})", file=sys.stderr)
    q = queue.Queue()

    def cb(indata, frames, t, status):
        q.put(indata[:, 0].copy())

    with sd.InputStream(device=device, channels=1, samplerate=rate, dtype="float32",
                        blocksize=int(rate * block_sec), callback=cb):
        while True:
            try:
                x = q.get(timeout=0.2)
            except queue.Empty:
                yield None
                continue
            yield rs.resample_chunk(x) if rs else x


def read_audio(path):
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        import soxr

        x = soxr.resample(x, sr, SR).astype(np.float32)
    return x


def wav_blocks(x, fast, block_sec=0.1):
    """Yield the audio 100 ms at a time, in real time unless `fast` (the caller flushes at the end)."""
    b = int(SR * block_sec)
    t0 = time.perf_counter()
    for i in range(0, len(x), b):
        if not fast:
            wait = t0 + i / SR - time.perf_counter()
            if wait > 0:
                time.sleep(wait)
        yield x[i:i + b]


def default_threads():
    """Apple silicon: the number of performance cores (waiting for efficiency cores slows everything down). Else up to 4."""
    if sys.platform == "darwin":
        try:
            n = int(subprocess.run(["sysctl", "-n", "hw.perflevel0.physicalcpu"], capture_output=True, text=True).stdout)
            if n > 0:
                return n
        except (OSError, ValueError):
            pass
    return max(1, min(4, os.cpu_count() or 1))


def bench(s2t: Speech2TextCascade, sec=20.0):
    """Stream noise (about -40 dBFS) at full speed and report compute time per layer."""
    rng = np.random.default_rng(0)
    x = (rng.standard_normal(int(sec * SR)) * 0.01).astype(np.float32)
    s2t.accept(x[:SR])  # the first calls are slow: leave the first second out
    s2t.compute = 0.0
    s2t.parts = {k: 0.0 for k in s2t.parts}
    s0 = s2t.samples
    for blk in wav_blocks(x[SR:], fast=True):
        s2t.accept(blk)
    audio = (s2t.samples - s0) / SR
    print(f"audio {audio:.1f} s, compute {s2t.compute:.2f} s (RTF {s2t.compute / audio:.3f}; below 1 runs in real time)")
    for k, v in s2t.parts.items():
        print(f"  {v:6.2f} s (RTF {v / audio:.3f})  {k}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL,
                    help=f"Hugging Face tag or local model directory (default: Cascade+Event-DP trained with "
                         f"ReazonSpeech; CSJ+CEJC only: {MODEL_X10A}; without events: {MODEL_X6})")
    ap.add_argument("--device", default=None, help="input device number or part of its name")
    ap.add_argument("--list-devices", action="store_true")
    ap.add_argument("--wav", nargs="*", default=None, help="audio file(s) to stream instead of the microphone")
    ap.add_argument("--fast", action="store_true", help="stream --wav at full speed instead of real time")
    ap.add_argument("--plain", action="store_true", help="just print phonemes / kana / text for each --wav file")
    ap.add_argument("--pause", type=float, default=1.2, help="start a new line after layer 1 has been silent this long (s)")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads (default: performance cores on Apple silicon, else 4)")
    ap.add_argument("--bench", action="store_true", help="measure speed only (20 s of noise at full speed)")
    args = ap.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        return
    torch.set_num_threads(args.threads or default_threads())
    torch.set_num_interop_threads(1)

    print(f"loading {args.model} ({torch.get_num_threads()} threads)", file=sys.stderr)
    s2t = Speech2TextCascade.from_pretrained(args.model)

    if args.wav is not None and args.plain:
        for path in args.wav:
            r = s2t(read_audio(path))[0]
            print(f"{path}\tP\t{' '.join(r.tokens)}")
            print(f"{path}\tK\t{r.kana}")
            print(f"{path}\tG\t{r.text}")
        return
    if args.bench:
        bench(s2t)
        return
    if args.wav is not None:
        x = np.concatenate([read_audio(p) for p in args.wav])
        run_live(s2t, wav_blocks(x, args.fast), pause=args.pause, realtime_src=not args.fast)
    else:
        dev = args.device
        if dev is not None and dev.isdigit():
            dev = int(dev)
        run_live(s2t, mic_blocks(dev), pause=args.pause)


if __name__ == "__main__":
    main()
