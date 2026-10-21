"""Two-layer, two-rate streaming Zipformer transducer ("Cascade", icefall) run as TorchScript on the CPU.

The models of this family (``fujielab/icefall_asr_csj_cejc_phn_writ_comp_*cascade_zipformer_stream_c160_c640_la0``)
were trained with icefall on CSJ + CEJC and exported as one TorchScript module.
One audio stream gives two outputs:

- **layer 1**: phoneme symbols (IPA-based P notation: ``ː`` long vowel, ``N``
  moraic nasal, ``Q`` geminate, ``ʲ`` palatalization; ``|`` word boundary,
  ``<sp>`` pause, composite ``a+F`` filler and ``a+D`` repair phonemes and, on the
  Event-DP models, ``<laugh>`` and ``<L>`` ... ``</L>`` for laughter), confirmed
  every 160 ms;
- **layer 2**: kana-kanji characters (the written form), computed from the
  layer-1 encoder output every 640 ms.

Both layers run in real time on a laptop CPU (RTF ≈ 0.12 with 4 threads).
Only ``torch``, ``numpy``, ``kaldi-native-fbank`` and ``huggingface_hub`` are
needed; ESPnet and icefall are not imported by this module.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch

SR = 16000
LOG_EPS = -23.025850929940457  # log(1e-10): the fbank value of the silence padding used in training and evaluation
FRAME_S = 0.04  # output frame of both layers (25 Hz)

_COMPOSITE = re.compile(r"^(?P<base>.+?)\+(?P<mark>[FD])$")
_SKIP = {"[ja]"}  # language tag at the start of every layer-1 hypothesis


# ----------------------------------------------------------------------------- phonemes -> kana
class P2Kana:
    """Turns layer-1 symbols (possibly with ``+F`` / ``+D``) into katakana morae by longest match.

    Trailing symbols that do not form a mora yet (a consonant still waiting for its
    vowel) are returned as ``pending``; they are completed by the next chunk.
    """

    def __init__(self, table: Dict[str, str]):
        self.table = {tuple(k.split()): v for k, v in table.items()}
        self.maxlen = max(len(k) for k in self.table)
        self.prefixes = {k[:i] for k in self.table for i in range(1, len(k))}

    def convert(self, toks: Sequence[str]) -> Tuple[List[Tuple[str, str]], List[str]]:
        """-> ([(string, label)], pending symbols). label is "" / "F" / "D" / "laugh" / "L" / "raw"."""
        base, labs = [], []
        for t in toks:
            if t.startswith("<") or "+" not in t:
                base.append(t)
                labs.append("")
            else:
                b, _, lab = t.partition("+")
                base.append(b)
                labs.append(lab)
        out, i, n, inL = [], 0, len(base), False
        while i < n:
            b = base[i]
            if b in ("[ja]", "<blk>", "<unk>", "<mask>"):
                i += 1
                continue
            if b == "|":
                out.append((" ", ""))
                i += 1
                continue
            if b == "<sp>":
                out.append(("、", ""))
                i += 1
                continue
            if b == "<laugh>":
                out.append(("（笑）", "laugh"))
                i += 1
                continue
            if b in ("<L>", "</L>"):
                inL = b == "<L>"
                i += 1
                continue
            for L in range(min(self.maxlen, n - i), 0, -1):
                key = tuple(base[i:i + L])
                if key in self.table:
                    lab = "L" if inL else ("F" if "F" in labs[i:i + L] else ("D" if "D" in labs[i:i + L] else ""))
                    out.append((self.table[key], lab))
                    i += L
                    break
            else:
                if tuple(base[i:]) in self.prefixes:  # unfinished mora at the end
                    return out, base[i:]
                out.append((b, "raw"))
                i += 1
        return out, []

    def to_text(self, toks: Sequence[str]) -> str:
        """Plain katakana string (``|`` -> space, ``<sp>`` -> 、, ``<laugh>`` -> （笑）), pending symbols appended as is."""
        items, pending = self.convert(toks)
        return "".join(s for s, _ in items).strip() + "".join(pending)


# ----------------------------------------------------------------------------- search
class _Greedy:
    """Transducer greedy search, at most ``max_sym`` symbols per frame (initialized as icefall's greedy_search)."""

    def __init__(self, decoder, joiner, blank: int, context: int, max_sym: int):
        self.decoder, self.joiner = decoder, joiner
        self.blank, self.context, self.max_sym = blank, context, max_sym
        self.hyp = [blank] * context
        init = torch.tensor([[-1] * (context - 1) + [blank]], dtype=torch.int64)
        self.dec = self.joiner.decoder_proj(self.decoder(init, False)).squeeze(1)

    def step(self, enc):
        """enc: (T, C) encoder output -> [(frame, symbol id, confidence)]; confidence is the joiner's posterior."""
        enc = self.joiner.encoder_proj(enc)
        out = []
        for t in range(enc.size(0)):
            e = enc[t:t + 1]
            for _ in range(self.max_sym):
                logits = self.joiner(e, self.dec, False)
                y = int(logits.argmax(dim=-1))
                if y == self.blank:
                    break
                conf = float(torch.exp(logits[0, y] - torch.logsumexp(logits[0], dim=0)))
                self.hyp.append(y)
                out.append((t, y, conf))
                ctx = torch.tensor([self.hyp[-self.context:]], dtype=torch.int64)
                self.dec = self.joiner.decoder_proj(self.decoder(ctx, False)).squeeze(1)
        return out


def _load_tokens(path: str) -> List[str]:
    d = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line:
                tok, i = line.rsplit(" ", 1)
                d[int(i)] = tok
    return [d[i] for i in range(len(d))]


# ----------------------------------------------------------------------------- results
@dataclass
class CascadeResult:
    """Both layers' hypotheses of the current utterance, refreshed after every chunk.

    Layer 1: ``tokens``, ``aux_labels``, ``base_tokens``, ``timestamps`` and
    ``confidences`` are aligned 1:1. ``tokens`` are the model's phoneme symbols
    (``a+F`` = phoneme in a filler, ``a+D`` = in a repair / word fragment, ``|`` word
    boundary, ``<sp>`` pause, ``<laugh>``, ``<L>`` ... ``</L>``; the ``[ja]`` language
    tag is dropped), ``aux_labels`` the marker per token (``N`` / ``F`` / ``D``) and
    ``base_tokens`` the tokens with the marker removed, as in the other results of
    this package. ``kana`` is layer 1 converted to katakana.

    Layer 2: ``text`` is the written form; ``graphemes`` / ``grapheme_timestamps`` /
    ``grapheme_confidences`` are its symbols. Layer 2 runs on 640 ms chunks, so it
    lags layer 1 by 0.3-0.6 s.

    Timestamps are seconds from the start of the utterance (start of the 40 ms
    frame the symbol was emitted at); confidences are the joiner posteriors.
    """

    text: str
    kana: str
    tokens: List[str]
    aux_labels: List[str]
    base_tokens: List[str]
    timestamps: List[float] = field(default_factory=list)
    confidences: List[float] = field(default_factory=list)
    graphemes: List[str] = field(default_factory=list)
    grapheme_timestamps: List[float] = field(default_factory=list)
    grapheme_confidences: List[float] = field(default_factory=list)


def _split_marker(token: str):
    m = _COMPOSITE.match(token)
    if m:
        return m.group("base"), m.group("mark")
    return token, "N"


# ----------------------------------------------------------------------------- recognizer
class Speech2TextCascade:
    """Chunk-by-chunk recognizer for the two-layer Cascade models.

    Usage::

        s2t = Speech2TextCascade.from_pretrained(
            "fujielab/icefall_asr_csj_cejc_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0")
        for chunk in chunks_of_100ms:                # float32 / float64 in [-1, 1] or int16, 16 kHz
            result = s2t.streaming_decode(chunk, is_final=is_last)[0]
        print(result.kana)                           # layer 1 as katakana
        print(result.text)                           # layer 2 (kana-kanji)

    ``streaming_decode`` keeps one utterance open until ``is_final=True`` (or
    :meth:`reset`); the next call starts a new utterance. For incremental use,
    :meth:`accept` / :meth:`flush` return only the symbols that are new since the
    previous call, as ``(frame, symbol, confidence)`` per layer.

    Loading turns off TorchScript's profiling executor for the whole process
    (``torch._C._jit_set_profiling_executor(False)`` and friends): with it on, the
    first decoder call fails with "Inference tensors cannot be saved for backward".
    This does not affect eager-mode (ESPnet) models.
    """

    frame_s = FRAME_S

    def __init__(self, model_dir: str, num_threads: Optional[int] = None):
        import kaldi_native_fbank  # local import: keep the package importable without it

        self._knf = kaldi_native_fbank
        if num_threads is not None:
            torch.set_num_threads(num_threads)
        torch._C._jit_set_profiling_executor(False)
        torch._C._jit_set_profiling_mode(False)
        torch._C._set_graph_executor_optimize(False)

        self.model_dir = model_dir
        with open(os.path.join(model_dir, "config.json"), encoding="utf-8") as f:
            self.cfg = cfg = json.load(f)
        self.m = torch.jit.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        self.m.eval()
        self.tok_p = _load_tokens(os.path.join(model_dir, "tokens_p.txt"))
        self.tok_g = _load_tokens(os.path.join(model_dir, "tokens_g.txt"))
        with open(os.path.join(model_dir, "p2kana.json"), encoding="utf-8") as f:
            self.p2k = P2Kana(json.load(f))
        self.T = cfg["chunk_size"] * 2 + cfg["pad_length"]  # fbank frames per encoder call (look-ahead included)
        self.shift = cfg["chunk_size"] * 2  # fbank frames the encoder advances per call (160 ms)
        self.chunk2 = cfg["chunk_size2"]  # layer-2 chunk in layer-1 output frames (640 ms)
        self.compute = 0.0
        self.parts = {"layer-1 encoder": 0.0, "layer-1 search": 0.0, "layer-2 encoder": 0.0, "layer-2 search": 0.0}
        self.reset()

    @classmethod
    def from_pretrained(
        cls,
        model_tag: str = "fujielab/icefall_asr_csj_cejc_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0",
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "Speech2TextCascade":
        """Download from the Hugging Face Hub (345 MB), or use a local directory."""
        if os.path.isdir(model_tag):
            model_dir = model_tag
        else:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(model_tag, cache_dir=cache_dir)
        return cls(model_dir, **kwargs)

    # -------------------------------------------------------------- low level
    def reset(self) -> None:
        """Back to the start of an utterance (fbank, encoder states of both layers, search, frame counters)."""
        cfg = self.cfg
        o = self._knf.FbankOptions()
        o.frame_opts.dither = 0
        o.frame_opts.snip_edges = False
        o.frame_opts.samp_freq = SR
        o.mel_opts.num_bins = cfg["fbank"]["num_bins"]
        o.mel_opts.high_freq = cfg["fbank"]["high_freq"]
        self.fbank = self._knf.OnlineFbank(o)
        cpu = torch.device("cpu")
        self.states1 = self.m.encoder.get_init_states(1, cpu)
        self.states2 = self.m.encoder2.get_init_states(1, cpu)
        ms = cfg["max_sym_per_frame"]
        self.g1 = _Greedy(self.m.decoder, self.m.joiner, 0, cfg["context_size"], ms)
        self.g2 = _Greedy(self.m.decoder2, self.m.joiner2, 0, cfg["context_size"], ms)
        self.frames = []  # fbank frames not yet consumed by layer 1 (look-ahead included)
        self.fetched = 0  # fbank frames taken out of the extractor
        self.l1_frames = 0  # layer-1 output frames (25 Hz)
        self.l2_frames = 0  # layer-1 output frames read by layer 2
        self.buf2 = []
        self.samples = 0
        self._ev1, self._ev2 = [], []  # symbols of the open utterance (streaming_decode)

    def _as_float32(self, speech: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        x = np.asarray(speech)
        if x.ndim == 2:  # (T, C) -> mono
            x = x.mean(axis=1)
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float32) / 32768.0
        return np.ascontiguousarray(x, dtype=np.float32)

    def _pull(self):
        n = self.fbank.num_frames_ready
        if n > self.fetched:
            # get_frame points into knf's buffer, so copy before pop
            self.frames += [np.array(self.fbank.get_frame(i), dtype=np.float32, copy=True) for i in range(self.fetched, n)]
            self.fbank.pop(n - self.fetched)
            self.fetched = n

    def accept(self, speech: Union[np.ndarray, Sequence[float]]):
        """Feed audio (16 kHz, any length) -> (new layer-1 symbols, new layer-2 symbols), each [(frame, symbol, confidence)].

        Frames count 40 ms output frames from the start of the utterance.
        """
        x = self._as_float32(speech)
        with torch.inference_mode():
            self.fbank.accept_waveform(SR, x)
            self.samples += len(x)
            self._pull()
            return self._run()

    def flush(self):
        """Emit the rest at the end of an utterance: silence frames (LOG_EPS) are appended as in evaluation.

        Returns the new symbols like :meth:`accept`. Call :meth:`reset` before the next utterance.
        """
        with torch.inference_mode():
            self.fbank.input_finished()
            self._pull()
            target = ((self.fetched - 7) // 2 + 1) // 2  # layer-1 output frames covering the real audio
            ev1, ev2, added = [], [], 0
            while (added < 30 or self.l2_frames < target) and added < 400:
                self.frames += [np.full(self.frames[0].shape if self.frames else (80,), LOG_EPS, dtype=np.float32)] * self.shift
                added += self.shift
                a, b = self._run()
                ev1 += a
                ev2 += b
            return ev1, ev2

    def _run(self):
        ev1, ev2 = [], []
        while len(self.frames) >= self.T:
            t0 = time.perf_counter()
            feats = np.stack(self.frames[:self.T])
            del self.frames[:self.shift]
            enc, _, self.states1 = self.m.encoder(
                torch.from_numpy(feats).unsqueeze(0), torch.tensor([self.T], dtype=torch.int32), self.states1)
            t1 = time.perf_counter()
            ev1 += [(self.l1_frames + t, self.tok_p[y], c) for t, y, c in self.g1.step(enc[0])]
            t2 = time.perf_counter()
            self.parts["layer-1 encoder"] += t1 - t0
            self.parts["layer-1 search"] += t2 - t1
            self.l1_frames += enc.size(1)
            self.buf2.append(enc)
            while sum(b.size(1) for b in self.buf2) >= self.chunk2:
                cat = torch.cat(self.buf2, dim=1)
                self.buf2 = [cat[:, self.chunk2:]] if cat.size(1) > self.chunk2 else []
                t3 = time.perf_counter()
                enc2, _, self.states2 = self.m.encoder2(
                    cat[:, :self.chunk2], torch.tensor([self.chunk2], dtype=torch.int32), self.states2)
                t4 = time.perf_counter()
                ev2 += [(self.l2_frames + t, self.tok_g[y], c) for t, y, c in self.g2.step(enc2[0])]
                self.parts["layer-2 encoder"] += t4 - t3
                self.parts["layer-2 search"] += time.perf_counter() - t4
                self.l2_frames += self.chunk2
            self.compute += time.perf_counter() - t0
        return ev1, ev2

    # -------------------------------------------------------------- decoding
    def make_result(self, ev1, ev2) -> CascadeResult:
        """Build a :class:`CascadeResult` from accumulated ``(frame, symbol, confidence)`` lists."""
        ev1 = [e for e in ev1 if e[1] not in _SKIP]
        tokens = [t for _, t, _ in ev1]
        base, aux = [], []
        for t in tokens:
            b, a = _split_marker(t)
            base.append(b)
            aux.append(a)
        graphemes = [t for _, t, _ in ev2]
        return CascadeResult(
            text="".join(t for t in graphemes if not t.startswith("<")),
            kana=self.p2k.to_text(tokens),
            tokens=tokens,
            aux_labels=aux,
            base_tokens=base,
            timestamps=[round(f * FRAME_S, 3) for f, _, _ in ev1],
            confidences=[c for _, _, c in ev1],
            graphemes=graphemes,
            grapheme_timestamps=[round(f * FRAME_S, 3) for f, _, _ in ev2],
            grapheme_confidences=[c for _, _, c in ev2],
        )

    def streaming_decode(
        self, speech: Union[np.ndarray, Sequence[float]], is_final: bool = False
    ) -> List[CascadeResult]:
        """Feed one chunk (any length; 100 ms is typical) and return both layers' hypotheses so far.

        With ``is_final=True`` the utterance is flushed (silence is appended so that
        both layers emit their last symbols) and closed.
        """
        a, b = self.accept(speech)
        self._ev1 += a
        self._ev2 += b
        if is_final:
            a, b = self.flush()
            self._ev1 += a
            self._ev2 += b
        result = self.make_result(self._ev1, self._ev2)
        if is_final:
            self.reset()
        return [result]

    def __call__(self, speech: Union[np.ndarray, Sequence[float]]) -> List[CascadeResult]:
        """Recognize a whole utterance (resets any open utterance first)."""
        self.reset()
        return self.streaming_decode(speech, is_final=True)
