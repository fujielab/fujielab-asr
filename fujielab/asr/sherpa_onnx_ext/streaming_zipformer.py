"""Streaming Zipformer transducer (icefall) run by sherpa-onnx on the CPU.

The models of this family (``fujielab/sherpa_onnx_asr_csj_cejc_pron_*_zipformer_*``)
are streaming Zipformer transducers trained with icefall on CSJ + CEJC and
exported to ONNX. They emit the same mora + marker tokens as the ESPnet models of
this package (``|`` word boundary, ``<sp>`` pause, composite ``ア+F`` filler and
``ン+D`` repair morae) and run in real time on a single CPU thread
(RTF ≈ 0.08 fp32 / 0.05 int8 at 320 ms chunks).

Only ``sherpa-onnx``, ``numpy`` and ``huggingface_hub`` are needed at run time;
``torch`` and ESPnet are not imported by this module.
"""

from __future__ import annotations

import glob
import os
import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

import numpy as np

_COMPOSITE = re.compile(r"^(?P<base>.+?)\+(?P<mark>[FD])$")
_STRUCTURAL = {"|", "<sp>"}


@dataclass
class SherpaOnnxResult:
    """One hypothesis, refreshed after every chunk.

    ``tokens``, ``aux_labels`` and ``timestamps`` are aligned 1:1. ``tokens`` are
    the model's own output (composite encoding: ``ア+F`` = filler mora, ``ン+D`` =
    repair mora, ``|`` = word boundary, ``<sp>`` = pause, ``<mask>`` = masked
    region); ``aux_labels`` hold the marker per token (``N`` / ``F`` / ``D``) and
    ``base_tokens`` the tokens with the marker removed, so the result can be
    consumed exactly like the multitask Transducer results of this package.
    """

    text: str
    tokens: List[str]
    aux_labels: List[str]
    base_tokens: List[str]
    timestamps: List[float] = field(default_factory=list)  # seconds, emission frame start

    @property
    def transcript(self) -> str:
        """Plain mora string: markers, word boundaries and pauses removed."""
        return "".join(t for t in self.base_tokens if t not in _STRUCTURAL)


def _split_marker(token: str):
    m = _COMPOSITE.match(token)
    if m:
        return m.group("base"), m.group("mark")
    return token, "N"


class Speech2TextSherpaOnnx:
    """Chunk-by-chunk recognizer around ``sherpa_onnx.OnlineRecognizer``.

    Usage::

        s2t = Speech2TextSherpaOnnx.from_pretrained(
            "fujielab/sherpa_onnx_asr_csj_cejc_pron_comp_zipformer_stream_c320_la0")
        for chunk in chunks_of_100ms:                # float32 / float64 in [-1, 1] or int16
            result = s2t.streaming_decode(chunk, is_final=is_last)[0]
        print(result.text)                           # "ア+F ノ+F | イ ッ パ ツ メ ..."

    ``streaming_decode`` keeps one utterance open until ``is_final=True`` (or
    :meth:`reset`), then the next call starts a new utterance. Every call returns
    the current best hypothesis (partial results are usable immediately).
    """

    def __init__(
        self,
        model_dir: str,
        chunk_ms: int = 320,
        int8: bool = False,
        num_threads: int = 1,
        decoding_method: str = "greedy_search",
        max_active_paths: int = 4,
        tail_padding_s: float = 0.6,
        strip_trailing_boundary: bool = True,
        sample_rate: int = 16000,
        enable_endpoint_detection: bool = False,
        provider: str = "cpu",
        **recognizer_kwargs,
    ):
        """
        ``tail_padding_s`` seconds of silence are appended when an utterance is
        finished so that the encoder (which needs a full chunk plus its right
        context) emits the last tokens: with 0.6 s the final hypotheses match the
        offline icefall decode on 56 / 60 evaluation utterances, with 0.3 s only 31.
        The silence itself may produce a trailing ``|`` / ``<sp>``; those carry no
        content and are removed from the final result when
        ``strip_trailing_boundary`` is True.
        """
        import sherpa_onnx  # local import: keep the package importable without it

        self.model_dir = model_dir
        self.sample_rate = sample_rate
        self.chunk_ms = chunk_ms
        self.int8 = int8
        self.tail_padding_s = tail_padding_s
        self.strip_trailing_boundary = strip_trailing_boundary
        enc, dec, joi = self.resolve_files(model_dir, chunk_ms=chunk_ms, int8=int8)
        self.recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=os.path.join(model_dir, "tokens.txt"),
            encoder=enc,
            decoder=dec,
            joiner=joi,
            num_threads=num_threads,
            sample_rate=sample_rate,
            feature_dim=80,
            decoding_method=decoding_method,
            max_active_paths=max_active_paths,
            enable_endpoint_detection=enable_endpoint_detection,
            provider=provider,
            **recognizer_kwargs,
        )
        self.stream = None

    # ------------------------------------------------------------------ files
    @staticmethod
    def resolve_files(model_dir: str, chunk_ms: int = 320, int8: bool = False):
        """Pick encoder / decoder / joiner ONNX files for the chunk size.

        The icefall export names files ``<part>-epoch-E-avg-A-chunk-C-left-L[.int8].onnx``
        where ``C`` is the chunk size in 50 Hz encoder frames (16 = 320 ms, 32 = 640 ms).
        """
        chunk_frames = int(round(chunk_ms / 20))
        sfx = ".int8.onnx" if int8 else ".onnx"
        out = []
        for part in ("encoder", "decoder", "joiner"):
            cands = [
                f for f in glob.glob(os.path.join(model_dir, f"{part}-*-chunk-{chunk_frames}-left-*.onnx"))
                if f.endswith(sfx) and (int8 or not f.endswith(".int8.onnx"))
            ]
            if not cands:
                avail = sorted(os.path.basename(f) for f in glob.glob(os.path.join(model_dir, f"{part}-*.onnx")))
                raise FileNotFoundError(
                    f"no {part} model for chunk_ms={chunk_ms} (chunk {chunk_frames} frames), int8={int8} "
                    f"in {model_dir}; available: {avail}"
                )
            out.append(sorted(cands)[0])
        return tuple(out)

    @classmethod
    def from_pretrained(
        cls,
        model_tag: str = "fujielab/sherpa_onnx_asr_csj_cejc_pron_comp_zipformer_stream_c320_la0",
        chunk_ms: int = 320,
        int8: bool = False,
        cache_dir: Optional[str] = None,
        **kwargs,
    ) -> "Speech2TextSherpaOnnx":
        """Download (only the files for ``chunk_ms`` / ``int8``) from the Hugging Face Hub, or use a local directory."""
        if os.path.isdir(model_tag):
            model_dir = model_tag
        else:
            from huggingface_hub import snapshot_download

            chunk_frames = int(round(chunk_ms / 20))
            sfx = ".int8.onnx" if int8 else ".onnx"
            model_dir = snapshot_download(
                model_tag,
                cache_dir=cache_dir,
                allow_patterns=["tokens.txt", "*.json", "README.md", f"*-chunk-{chunk_frames}-left-*{sfx}"],
                ignore_patterns=None if int8 else ["*.int8.onnx"],
            )
        return cls(model_dir, chunk_ms=chunk_ms, int8=int8, **kwargs)

    # -------------------------------------------------------------- decoding
    def reset(self) -> None:
        """Drop the current utterance; the next chunk starts a new one."""
        self.stream = self.recognizer.create_stream()

    def _as_float32(self, speech: Union[np.ndarray, Sequence[float]]) -> np.ndarray:
        x = np.asarray(speech)
        if x.ndim == 2:  # (T, C) -> mono
            x = x.mean(axis=1)
        if np.issubdtype(x.dtype, np.integer):
            x = x.astype(np.float32) / 32768.0
        return np.ascontiguousarray(x, dtype=np.float32)

    def _decode_pending(self) -> None:
        while self.recognizer.is_ready(self.stream):
            self.recognizer.decode_stream(self.stream)

    def _result(self, final: bool = False) -> SherpaOnnxResult:
        tokens = list(self.recognizer.tokens(self.stream))
        try:
            timestamps = [float(t) for t in self.recognizer.timestamps(self.stream)]
        except Exception:  # pragma: no cover - older sherpa-onnx without timestamps
            timestamps = []
        if final and self.strip_trailing_boundary:
            while tokens and tokens[-1] in _STRUCTURAL:
                tokens.pop()
            timestamps = timestamps[: len(tokens)]
        base, aux = [], []
        for t in tokens:
            b, a = _split_marker(t)
            base.append(b)
            aux.append(a)
        return SherpaOnnxResult(
            text=" ".join(tokens), tokens=tokens, aux_labels=aux, base_tokens=base, timestamps=timestamps
        )

    def streaming_decode(
        self, speech: Union[np.ndarray, Sequence[float]], is_final: bool = False
    ) -> List[SherpaOnnxResult]:
        """Feed one chunk (any length; 100 ms is typical) and return the current hypothesis.

        With ``is_final=True`` the utterance is flushed (``tail_padding_s`` of silence
        is appended so the last tokens are emitted, matching how the model was
        trained with trailing silence) and closed.
        """
        if self.stream is None:
            self.reset()
        x = self._as_float32(speech)
        if len(x):
            self.stream.accept_waveform(self.sample_rate, x)
        if is_final:
            if self.tail_padding_s > 0:
                self.stream.accept_waveform(
                    self.sample_rate, np.zeros(int(self.sample_rate * self.tail_padding_s), dtype=np.float32)
                )
            self.stream.input_finished()
        self._decode_pending()
        result = self._result(final=is_final)
        if is_final:
            self.stream = None
        return [result]

    def __call__(self, speech: Union[np.ndarray, Sequence[float]]) -> List[SherpaOnnxResult]:
        """Recognize a whole utterance (resets any open stream first)."""
        self.reset()
        return self.streaming_decode(speech, is_final=True)

    # ------------------------------------------------------------ endpointing
    def is_endpoint(self) -> bool:
        """True when sherpa-onnx's endpoint rules fired (needs ``enable_endpoint_detection=True``)."""
        return self.stream is not None and self.recognizer.is_endpoint(self.stream)

    def finish_segment(self) -> SherpaOnnxResult:
        """Return the hypothesis of the current segment and start a new one (for endpoint-driven use)."""
        if self.stream is None:
            self.reset()
        self._decode_pending()
        result = self._result(final=True)
        self.recognizer.reset(self.stream)
        return result
