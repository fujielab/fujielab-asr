#!/usr/bin/env python3

"""Streaming inference for plain CTC-only ASR models.

This is the no-auxiliary-head counterpart of
:class:`fujielab.asr.espnet_ext.espnet2.bin.asr_multitask_ctc_inference_cbs.Speech2TextMultitaskCTC`.
It serves the auxiliary-information encodings where the F/D markers are part of
the recognized TOKEN sequence itself (so a standard CTC model suffices):

  - composite : tokens carry a suffix, e.g. "あ+F" / "チ+D".
  - span      : range markers appear as tokens, e.g. "<F> あ の </F>".

The base tokens (with whatever markers) are recognized by CTC alone on a
Contextual Block Streaming encoder. Decoding is chunk-by-chunk greedy CTC; the
contextual-block encoder emits each output frame exactly once across the stream,
so streaming greedy CTC equals single-pass greedy CTC — we only carry the last
frame's argmax id across chunk boundaries for blank/duplicate collapse.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch

from fujielab.asr.espnet_ext.espnet2.asr.frontend.online_audio_processor import (
    OnlineAudioProcessor,
)
from espnet2.tasks.asr import ASRTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter


@dataclass
class CTCResult:
    """One greedy-CTC hypothesis (tokens may include +F/+D or <F></F> markers)."""

    text: Optional[str]
    tokens: List[str]
    token_ids: List[int]
    score: float


class Speech2TextCTC:
    """Speech2Text for plain CTC-only models (markers embedded in the tokens).

    Args:
        asr_train_config: ASR model training config path.
        asr_model_file: ASR model path.
        device: Device to use for inference.
        dtype: Data type.
        token_type: Type of token units (defaults to the training args).
        bpemodel: BPE model path (defaults to the training args).
        streaming: Whether to prepare chunk-by-chunk streaming inference.
    """

    def __init__(
        self,
        asr_train_config: Union[Path, str] = None,
        asr_model_file: Union[Path, str] = None,
        device: str = "cpu",
        dtype: str = "float32",
        token_type: Optional[str] = None,
        bpemodel: Optional[str] = None,
        streaming: bool = True,
    ) -> None:
        super().__init__()

        asr_model, asr_train_args = ASRTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        token_list = asr_model.token_list
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = getattr(asr_train_args, "bpemodel", None)

        if token_type is None:
            tokenizer = None
        elif token_type == "bpe":
            tokenizer = (
                build_tokenizer(token_type=token_type, bpemodel=bpemodel)
                if bpemodel is not None
                else None
            )
        else:
            tokenizer = build_tokenizer(token_type=token_type)

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.device = device
        self.dtype = dtype
        self.converter = TokenIDConverter(token_list=token_list)
        self.tokenizer = tokenizer
        self.blank_id = asr_model.blank_id

        self.streaming = streaming
        self.asr_model.encoder.dynamic_chunk_training = False

        if streaming:
            self.audio_processor = OnlineAudioProcessor(
                asr_model._extract_feats,
                asr_model.normalize,
                asr_train_args.frontend_conf,
                device,
            )
            self.reset_streaming_cache()

    def reset_streaming_cache(self) -> None:
        """Reset streaming state (encoder cache + greedy-CTC collapse state)."""
        self.encoder_states = None
        self.audio_processor.reset_cache()
        self._prev_argmax = self.blank_id
        self._token_ids: List[int] = []
        self._score = 0.0

    @torch.no_grad()
    def streaming_decode(
        self, speech: Union[torch.Tensor, np.ndarray], is_final: bool = False
    ) -> List[CTCResult]:
        """Decode a chunk of speech, accumulating greedy-CTC state."""
        if isinstance(speech, np.ndarray):
            speech = torch.as_tensor(speech)
        speech = speech.to(device=self.device)

        feats, feats_length = self.audio_processor.compute_features(
            speech.to(getattr(torch, self.dtype)), is_final
        )
        enc_out, _, self.encoder_states = self.asr_model.encoder(
            feats, feats_length, self.encoder_states, is_final=is_final, infer_mode=True
        )
        if isinstance(enc_out, tuple):
            enc_out = enc_out[0]
        enc = enc_out[0]  # (T_chunk, D)
        if enc.size(0) > 0:
            self._consume_frames(enc)

        results = self._build_results()
        if is_final:
            self.reset_streaming_cache()
        return results

    def _consume_frames(self, enc: torch.Tensor) -> None:
        log_probs = self.asr_model.ctc.log_softmax(enc.unsqueeze(0))[0]  # (T, V)
        best_lp, ids = log_probs.max(dim=-1)
        self._score += float(best_lp.sum().item())
        blank = self.blank_id
        for t in range(ids.size(0)):
            cur = int(ids[t].item())
            if cur != self._prev_argmax and cur != blank:
                self._token_ids.append(cur)
            self._prev_argmax = cur

    def _build_results(self) -> List[CTCResult]:
        tokens = self.converter.ids2tokens(self._token_ids)
        text = self.tokenizer.tokens2text(tokens) if self.tokenizer is not None else None
        return [CTCResult(text, tokens, list(self._token_ids), self._score)]

    @torch.no_grad()
    def __call__(self, speech: Union[torch.Tensor, np.ndarray]) -> List[CTCResult]:
        """Offline (whole-utterance) greedy CTC decoding."""
        if isinstance(speech, np.ndarray):
            speech = torch.as_tensor(speech)
        speech = speech.to(getattr(torch, self.dtype)).unsqueeze(0).to(self.device)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        enc, enc_lens = self.asr_model.encode(speech, lengths)
        if isinstance(enc, tuple):
            enc = enc[0]
        logp = self.asr_model.ctc.log_softmax(enc[:1])[0]
        best_lp, ids = logp.max(dim=-1)
        score = float(best_lp.sum().item())
        collapsed, prev = [], None
        for i in ids.tolist():
            if i != prev:
                collapsed.append(i)
                prev = i
        tids = [i for i in collapsed if i != self.blank_id]
        tokens = self.converter.ids2tokens(tids)
        text = self.tokenizer.tokens2text(tokens) if self.tokenizer is not None else None
        return [CTCResult(text, tokens, tids, score)]

    def hypotheses_to_results(self, results: List[CTCResult]) -> List[CTCResult]:
        """Identity pass-through (API parity)."""
        return results

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None, **kwargs: Optional[Any]
    ) -> "Speech2TextCTC":
        """Build a Speech2TextCTC from a pretrained model tag."""
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader
            except ImportError:
                raise ImportError(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
            d = ModelDownloader()
            kwargs.update(**d.download_and_unpack(model_tag))
        return Speech2TextCTC(**kwargs)
