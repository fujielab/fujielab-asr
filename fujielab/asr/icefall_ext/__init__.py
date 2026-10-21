"""Streaming recognizers trained with `icefall <https://github.com/k2-fsa/icefall>`_ and run as TorchScript
(CPU real-time; icefall itself is not needed at inference time)."""

from .cascade_zipformer import CascadeResult, P2Kana, Speech2TextCascade  # noqa: F401
