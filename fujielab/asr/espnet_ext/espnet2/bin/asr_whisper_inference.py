"""Speech2Text for the Whisper large-v3 + LoRA models (utterance-level decoding).

The models output the same mora / auxiliary-marker token strings as the CBS
models (``token_type: whisper_mora``: one mora per token on top of the Whisper
vocabulary). This class decodes a whole utterance with attention beam search;
the block-causal (``chunk_size > 0``) encoders can be run this way too, which
gives exactly the look-ahead-limited result they were trained for. A
chunk-by-chunk API for those encoders is planned.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import torch
from espnet.nets.beam_search import BeamSearch, Hypothesis
from espnet.nets.e2e_asr_common import end_detect
from espnet.nets.scorers.length_bonus import LengthBonus

from fujielab.asr.espnet_ext.espnet2.tasks.asr_whisper import ASRWhisperTask
from fujielab.asr.espnet_ext.espnet2.text.whisper_mora_token_id_converter import (
    WhisperMoraTokenIDConverter,
)
from fujielab.asr.espnet_ext.espnet2.text.whisper_mora_tokenizer import (
    WhisperMoraTokenizer,
)

logger = logging.getLogger(__name__)

# Whisper's decoder has 448 positions; 4 are taken by <|startoftranscript|> and
# the task prefix, and the longest training target is 185 tokens.
MAX_OUTPUT_TOKENS = 440


class _BeamSearch(BeamSearch):
    """BeamSearch with end-detection also under an absolute max length.

    Stock ``BeamSearch`` applies ``end_detect`` only when ``maxlenratio == 0``,
    i.e. when the max length is the encoder length -- which for Whisper can
    exceed the 448 decoder positions. With a negative ``maxlenratio`` (absolute
    cap) stock search runs all ``maxlen`` steps per utterance; this subclass
    keeps the cap and still stops as soon as the ended hypotheses dominate.
    """

    def forward(self, x, maxlenratio: float = 0.0, minlenratio: float = 0.0, pre_x=None):
        inp = pre_x if pre_x is not None else x
        if maxlenratio == 0:
            maxlen = inp.shape[0]
        elif maxlenratio < 0:
            maxlen = -1 * int(maxlenratio)
        else:
            maxlen = max(1, int(maxlenratio * inp.size(0)))
        minlen = -1 * int(minlenratio) if minlenratio < 0 else int(minlenratio * inp.size(0))
        running_hyps = self.init_hyp(x if pre_x is None else pre_x)
        ended_hyps: List[Hypothesis] = []
        for i in range(maxlen):
            best = self.search(running_hyps, x, pre_x=pre_x)
            running_hyps = self.post_process(i, maxlen, minlen, maxlenratio, best, ended_hyps)
            if maxlenratio <= 0.0 and end_detect([h.asdict() for h in ended_hyps], i):
                break
            if len(running_hyps) == 0:
                break
        if self.normalize_length:
            nbest = sorted(ended_hyps, key=lambda h: h.score / (len(h.yseq) - 1), reverse=True)
        else:
            nbest = sorted(ended_hyps, key=lambda h: h.score, reverse=True)
        if len(nbest) == 0:
            return [] if minlenratio < 0.1 else self.forward(x, maxlenratio, max(0.0, minlenratio - 0.1))
        return nbest


@dataclass
class WhisperResult:
    text: str
    tokens: List[str]
    token_ids: List[int]
    score: float


class Speech2TextWhisper:
    """Utterance-level recognizer for the fujielab Whisper+LoRA models.

    Args:
        asr_train_config: training config (``config.yaml``).
        asr_model_file: adapter checkpoint (LoRA + added-token embeddings).
        device: ``cpu`` / ``cuda``.
        dtype: model dtype (``float32`` or ``float16``).
        beam_size: attention beam width (5 was used for the published results).
        penalty: length bonus weight.
        nbest: number of hypotheses to return.
        whisper_download_dir: where ``openai-whisper`` keeps the base weights
            (``None`` = its default cache, ``~/.cache/whisper``); overrides the
            training config so packed models work on any machine.
    """

    def __init__(
        self,
        asr_train_config: Union[Path, str] = None,
        asr_model_file: Union[Path, str] = None,
        device: str = "cpu",
        dtype: str = "float32",
        beam_size: int = 5,
        penalty: float = 0.0,
        nbest: int = 1,
        whisper_download_dir: Optional[str] = None,
    ) -> None:
        import yaml

        with Path(asr_train_config).open("r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        for key in ("encoder_conf", "decoder_conf"):
            cfg.setdefault(key, {})["download_dir"] = whisper_download_dir
        tmp_config = Path(asr_model_file).parent / ".fujielab_whisper_config.yaml"
        with tmp_config.open("w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, allow_unicode=True)

        asr_model, asr_train_args = ASRWhisperTask.build_model_from_file(
            tmp_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()
        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.device = device
        self.dtype = dtype
        self.nbest = nbest

        pconf = getattr(asr_train_args, "preprocessor_conf", {}) or {}
        self.tokenizer = WhisperMoraTokenizer()
        self.converter = WhisperMoraTokenIDConverter(
            token_list=asr_model.token_list,
            language=pconf.get("whisper_language", "ja") or "ja",
            task=pconf.get("whisper_task", "transcribe") or "transcribe",
        )
        scorers = dict(decoder=asr_model.decoder, length_bonus=LengthBonus(len(asr_model.token_list)))
        weights = dict(decoder=1.0, length_bonus=penalty)
        self.beam_search = _BeamSearch(
            beam_size=beam_size,
            weights=weights,
            scorers=scorers,
            sos=asr_model.sos,
            eos=asr_model.eos,
            vocab_size=len(asr_model.token_list),
            token_list=asr_model.token_list,
            pre_beam_score_key="full",
        )
        # start every hypothesis with <|startoftranscript|> + the task prefix, as in training
        self.beam_search.set_hyp_primer([asr_model.sos] + list(self.converter.prefix_ids))
        self.beam_search.to(device=device, dtype=getattr(torch, dtype)).eval()

    @property
    def is_streaming_encoder(self) -> bool:
        return getattr(self.asr_model.encoder, "chunk_size", 0) > 0

    @torch.no_grad()
    def __call__(self, speech: Union[torch.Tensor, np.ndarray]) -> List[WhisperResult]:
        """Recognize one utterance (16 kHz mono waveform)."""
        if isinstance(speech, np.ndarray):
            speech = torch.as_tensor(speech)
        speech = speech.to(getattr(torch, self.dtype)).unsqueeze(0)
        lengths = speech.new_full([1], dtype=torch.long, fill_value=speech.size(1))
        enc, _ = self.asr_model.encode(speech.to(self.device), lengths.to(self.device))
        if isinstance(enc, tuple):
            enc = enc[0]
        hyps = self.beam_search(x=enc[0], maxlenratio=-MAX_OUTPUT_TOKENS, minlenratio=0.0)
        results = []
        for hyp in hyps[: self.nbest]:
            ids = [int(t) for t in hyp.yseq[1:]]
            ids = [t for t in ids if t not in (0, self.asr_model.sos, self.asr_model.eos)]
            tokens = self.converter.ids2tokens(ids)
            results.append(WhisperResult(self.tokenizer.tokens2text(tokens), tokens, ids, float(hyp.score)))
        return results

    @staticmethod
    def from_pretrained(model_tag: Optional[str] = None, **kwargs: Any) -> "Speech2TextWhisper":
        """Build from a Hugging Face / espnet_model_zoo tag or a local zip."""
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader
            except ImportError:
                raise ImportError(
                    "`espnet_model_zoo` is not installed. Please install via `pip install -U espnet_model_zoo`."
                )
            kwargs.update(**ModelDownloader().download_and_unpack(model_tag))
        return Speech2TextWhisper(**kwargs)
