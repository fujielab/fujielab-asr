"""Two-layer offline (full right context) Zipformer transducer ("Cascade", icefall) run as TorchScript on the CPU.

The offline model of this family
(``fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_offline_writaux_proncond``,
"X15ev") sees the whole utterance in both layers. It is the most accurate model of the
package: repairs in particular can only be decided after hearing what follows. It is
not streaming; give it one utterance (or one VAD segment of up to 20-30 s) at a time.

- **layer 1**: phoneme symbols with filler / repair markers and laughter events, as in
  the streaming Cascade models;
- **layer 2**: the written form, with the filler / repair markers on the characters as
  well (``え+F``, ``こ+D``; "writing layer with auxiliary marks"). Its encoder reads the
  layer-1 encoder output *and* the symbols the layer-1 greedy search emitted at each
  frame ("pronunciation information input").

Only ``torch``, ``numpy``, ``kaldi-native-fbank`` and ``huggingface_hub`` are needed;
the results are those of the model's reference decoder (``offline_ref_decode.py``),
which match icefall's evaluation decode.
"""

from __future__ import annotations

import json
import os
import time
from typing import List, Optional, Sequence, Union

import numpy as np
import torch

from .cascade_zipformer import (
    FRAME_S,
    LOG_EPS,
    SR,
    CascadeResult,
    P2Kana,
    _as_float32,
    _Greedy,
    _load_tokens,
    make_cascade_result,
)

DEFAULT_OFFLINE_MODEL = "fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_offline_writaux_proncond"
TAIL_PAD = 30  # log(1e-10) fbank frames appended to every utterance, as in evaluation (so that the last word is emitted)


class Speech2TextCascadeOffline:
    """Whole-utterance recognizer for the offline two-layer Cascade model.

    Usage::

        s2t = Speech2TextCascadeOffline.from_pretrained()   # the X15ev model (345 MB)
        result = s2t(speech)[0]          # one utterance: float32 / float64 in [-1, 1] or int16, 16 kHz
        print(result.kana)               # layer 1 as katakana
        print(result.tagged_text)        # layer 2 in CSJ notation: (F えー)昨日(D こ)今度の話
        print(result.text)               # layer 2 without the markers

    The result is a :class:`CascadeResult`, as for the streaming models; timestamps
    are seconds from the start of the utterance.

    Cut long recordings into utterances first, at pauses: the model was trained on
    utterances of at most 30 s (``max_segment_s``; 99 % within 17.7 s, median 3.6 s),
    and full attention gets slower with the square of the length.

    Loading turns off TorchScript's profiling executor for the whole process, as
    :class:`Speech2TextCascade` does.
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
        if cfg.get("mode") != "offline":
            raise ValueError(f"{model_dir} holds a streaming model: load it with Speech2TextCascade")
        self.m = torch.jit.load(os.path.join(model_dir, "model.pt"), map_location="cpu")
        self.m.eval()
        self.tok_p = _load_tokens(os.path.join(model_dir, "tokens_p.txt"))
        self.tok_g = _load_tokens(os.path.join(model_dir, "tokens_g.txt"))
        with open(os.path.join(model_dir, "p2kana.json"), encoding="utf-8") as f:
            self.p2k = P2Kana(json.load(f))
        self.max_sym = cfg["max_sym_per_frame"]
        self.context = cfg["context_size"]
        self.blank = cfg["blank_id"]
        self.max_segment_s = cfg.get("train_max_duration_s", 30.0)
        self.compute = 0.0
        self.parts = {"fbank": 0.0, "layer-1 encoder": 0.0, "layer-1 search": 0.0,
                      "layer-2 encoder": 0.0, "layer-2 search": 0.0}

    @classmethod
    def from_pretrained(
        cls,
        model_tag: str = DEFAULT_OFFLINE_MODEL,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "Speech2TextCascadeOffline":
        """Download from the Hugging Face Hub (345 MB), or use a local directory."""
        if os.path.isdir(model_tag):
            model_dir = model_tag
        else:
            from huggingface_hub import snapshot_download

            model_dir = snapshot_download(model_tag, cache_dir=cache_dir)
        return cls(model_dir, **kwargs)

    # -------------------------------------------------------------- pieces
    def fbank(self, speech: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        """80-dim log-mel fbank (T, 80) of one utterance, as in training (no dither, snip_edges=False)."""
        fc = self.cfg["fbank"]
        o = self._knf.FbankOptions()
        o.frame_opts.dither = fc.get("dither", 0.0)
        o.frame_opts.snip_edges = fc.get("snip_edges", False)
        o.frame_opts.samp_freq = SR
        o.mel_opts.num_bins = fc["num_bins"]
        o.mel_opts.high_freq = fc["high_freq"]
        fb = self._knf.OnlineFbank(o)
        fb.accept_waveform(SR, _as_float32(speech))
        fb.input_finished()
        n = fb.num_frames_ready
        feats = np.empty((n, fc["num_bins"]), dtype=np.float32)
        for i in range(n):
            feats[i] = fb.get_frame(i)
        return feats

    def l1_symbols(self, ev1, num_frames: int) -> torch.Tensor:
        """Layer-1 greedy hypothesis -> (1, T, max_sym) symbol ids per output frame in emission order, -1 where none."""
        toks = torch.full((1, num_frames, self.max_sym), -1, dtype=torch.int64)
        k = {}
        for t, y, _ in ev1:
            j = k.get(t, 0)
            toks[0, t, j] = y
            k[t] = j + 1
        return toks

    def decode_fbank(self, feats: np.ndarray) -> CascadeResult:
        """Recognize one utterance given as fbank (T, 80) (e.g. lhotse features)."""
        with torch.inference_mode():
            t0 = time.perf_counter()
            x = torch.from_numpy(np.ascontiguousarray(feats, dtype=np.float32)).unsqueeze(0)
            x = torch.nn.functional.pad(x, (0, 0, 0, TAIL_PAD), value=LOG_EPS)
            enc, enc_lens = self.m.encoder(x, torch.tensor([x.size(1)], dtype=torch.int64))
            T = int(enc_lens[0])
            t1 = time.perf_counter()
            ev1 = _Greedy(self.m.decoder, self.m.joiner, self.blank, self.context, self.max_sym).step(enc[0, :T])
            t2 = time.perf_counter()
            enc2, enc2_lens = self.m.encoder2(enc, enc_lens, self.l1_symbols(ev1, enc.size(1)))
            t3 = time.perf_counter()
            ev2 = _Greedy(self.m.decoder2, self.m.joiner2, self.blank, self.context, self.max_sym).step(
                enc2[0, :int(enc2_lens[0])])
            t4 = time.perf_counter()
        for k, (a, b) in zip(list(self.parts)[1:], [(t0, t1), (t1, t2), (t2, t3), (t3, t4)]):
            self.parts[k] += b - a
        self.compute += t4 - t0
        return make_cascade_result(
            self.p2k,
            [(t, self.tok_p[y], c) for t, y, c in ev1],
            [(t, self.tok_g[y], c) for t, y, c in ev2],
        )

    # -------------------------------------------------------------- decoding
    def __call__(self, speech: Union[np.ndarray, Sequence[float]]) -> List[CascadeResult]:
        """Recognize one utterance (16 kHz) -> [CascadeResult]."""
        t0 = time.perf_counter()
        feats = self.fbank(speech)
        dt = time.perf_counter() - t0
        self.parts["fbank"] += dt
        self.compute += dt
        if len(feats) == 0:
            return [make_cascade_result(self.p2k, [], [])]
        return [self.decode_fbank(feats)]
