#!/usr/bin/env python3

"""Streaming inference for the CTC-only multitask ASR model.

The base tokens are recognized by CTC alone (no decoder) on a Contextual Block
Streaming encoder, and an auxiliary-information label (N/F/D = none / filler /
repair) is predicted per token by a head on the encoder output.

This mirrors the API of
:class:`fujielab.asr.espnet_ext.espnet2.bin.asr_transducer_inference_cbs.Speech2Text`
(``from_pretrained`` / ``streaming_decode`` / ``hypotheses_to_results``) but
replaces the transducer beam search with chunk-by-chunk greedy CTC decoding,
returning both the recognized tokens and their per-token auxiliary labels.

Because the contextual-block encoder emits each output frame exactly once across
the stream (its internal overlap / look-ahead buffering is carried in the
encoder state), streaming greedy CTC is identical to running greedy CTC once on
the concatenated encoder output: we only need to carry the last frame's argmax id
across chunk boundaries for blank/duplicate collapse.
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
from fujielab.asr.espnet_ext.espnet2.tasks.asr_multitask_ctc import ASRMultitaskCTCTask
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter


@dataclass
class MultitaskCTCResult:
    """One hypothesis for the multitask CTC model.

    ``tokens`` and ``aux_labels`` are aligned 1:1 (one auxiliary label per
    recognized base token).
    """

    text: Optional[str]
    tokens: List[str]
    token_ids: List[int]
    aux_labels: List[str]
    score: float


class Speech2TextMultitaskCTC:
    """Speech2Text for the CTC-only multitask model (ASR + aux-info labels).

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
        """Construct a Speech2TextMultitaskCTC object."""
        super().__init__()

        asr_model, asr_train_args = ASRMultitaskCTCTask.build_model_from_file(
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
        converter = TokenIDConverter(token_list=token_list)

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.device = device
        self.dtype = dtype

        self.converter = converter
        self.tokenizer = tokenizer
        self.aux_token_list = list(asr_model.aux_token_list)
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

        # cross-chunk greedy-CTC collapse state
        self._prev_argmax = self.blank_id
        self._accum_token_ids: List[int] = []
        self._accum_aux_logits: List[torch.Tensor] = []
        self._cur_token_aux: Optional[torch.Tensor] = None
        self._score = 0.0

    @torch.no_grad()
    def streaming_decode(
        self, speech: Union[torch.Tensor, np.ndarray], is_final: bool = False
    ) -> List[MultitaskCTCResult]:
        """Decode a chunk of speech, accumulating greedy-CTC + aux state.

        Args:
            speech: Chunk of speech data. (S,)
            is_final: Whether this is the final chunk of the utterance.

        Returns:
            A length-1 list with the (partial or final) hypothesis so far.
        """
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

        if is_final:
            self._flush_current_token()

        results = self._build_results()

        if is_final:
            self.reset_streaming_cache()

        return results

    def _consume_frames(self, enc: torch.Tensor) -> None:
        """Incremental greedy CTC + aux over a chunk's encoder frames."""
        log_probs = self.asr_model.ctc.log_softmax(enc.unsqueeze(0))[0]  # (T, V)
        aux_logits = self.asr_model.aux_output_layer(enc)  # (T, A)
        best_lp, ids = log_probs.max(dim=-1)  # (T,)
        self._score += float(best_lp.sum().item())

        blank = self.blank_id
        for t in range(ids.size(0)):
            cur = int(ids[t].item())
            if cur == blank:
                self._flush_current_token()
                self._prev_argmax = blank
                continue
            if cur == self._prev_argmax:
                # same token continuing (incl. across a chunk boundary)
                self._cur_token_aux = self._cur_token_aux + aux_logits[t]
            else:
                # a new token starts here
                self._flush_current_token()
                self._accum_token_ids.append(cur)
                self._cur_token_aux = aux_logits[t].clone()
            self._prev_argmax = cur

    def _flush_current_token(self) -> None:
        """Finalize the aux logits accumulated for the just-completed token."""
        if self._cur_token_aux is not None:
            self._accum_aux_logits.append(self._cur_token_aux)
            self._cur_token_aux = None

    def _build_results(self) -> List[MultitaskCTCResult]:
        token_ids = list(self._accum_token_ids)
        aux_ids = [int(a.argmax().item()) for a in self._accum_aux_logits]
        if self._cur_token_aux is not None:
            # token still in progress (partial result): report its running argmax
            aux_ids.append(int(self._cur_token_aux.argmax().item()))

        tokens = self.converter.ids2tokens(token_ids)
        aux_labels = [self.aux_token_list[a] for a in aux_ids]
        text = self.tokenizer.tokens2text(tokens) if self.tokenizer is not None else None
        return [MultitaskCTCResult(text, tokens, token_ids, aux_labels, self._score)]

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
    ) -> List[MultitaskCTCResult]:
        """Offline (whole-utterance) decoding, for parity with the demo UI."""
        if isinstance(speech, np.ndarray):
            speech = torch.as_tensor(speech)
        speech = speech.to(getattr(torch, self.dtype)).unsqueeze(0).to(self.device)
        lengths = speech.new_full(
            [1], dtype=torch.long, fill_value=speech.size(1)
        )

        enc, enc_lens = self.asr_model.encode(speech, lengths)
        if isinstance(enc, tuple):
            enc = enc[0]
        (token_ids, aux_ids, score) = self.asr_model.ctc_greedy_with_aux(enc, enc_lens)[
            0
        ]
        tokens = self.converter.ids2tokens(token_ids)
        aux_labels = [self.aux_token_list[a] for a in aux_ids]
        text = self.tokenizer.tokens2text(tokens) if self.tokenizer is not None else None
        return [MultitaskCTCResult(text, tokens, token_ids, aux_labels, float(score))]

    def hypotheses_to_results(
        self, results: List[MultitaskCTCResult]
    ) -> List[MultitaskCTCResult]:
        """Identity pass-through (kept for API parity with the transducer class)."""
        return results

    @staticmethod
    def from_pretrained(
        model_tag: Optional[str] = None, **kwargs: Optional[Any]
    ) -> "Speech2TextMultitaskCTC":
        """Build a Speech2TextMultitaskCTC from a pretrained model tag.

        Args:
            model_tag: Model tag of the pretrained model (HuggingFace / zenodo).

        Returns:
            A Speech2TextMultitaskCTC instance.
        """
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

        return Speech2TextMultitaskCTC(**kwargs)
