"""Streaming recognizers served by `sherpa-onnx <https://github.com/k2-fsa/sherpa-onnx>`_
(CPU real-time, no PyTorch at inference time)."""

from .streaming_zipformer import Speech2TextSherpaOnnx, SherpaOnnxResult  # noqa: F401
