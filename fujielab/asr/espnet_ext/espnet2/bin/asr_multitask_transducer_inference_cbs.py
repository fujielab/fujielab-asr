"""Streaming inference for the multitask RNN-Transducer (ASR + aux-info head).

Transducer counterpart of :mod:`asr_multitask_ctc_inference_cbs`. Base tokens
are recognized by a streaming RNN-Transducer beam search over a Contextual
Block Streaming encoder; each recognized token additionally carries an
auxiliary-information label from {N, F, D} (normal / filler / repair).

How the auxiliary label is obtained
-----------------------------------
The auxiliary head sits on the joint network's pre-output hidden state,
``h = joint_activation(lin_enc(enc_t) + lin_dec(dec_u))``, so a label is only
defined at a specific (frame, predictor-state) node of the transducer lattice.
The offline recipe finds that node by re-encoding the whole utterance and
running a forced Viterbi alignment of the final hypothesis -- a second pass
over complete audio, which a streaming recognizer cannot do.

Here the node is instead taken from the beam search itself: when the search
emits a token it does so at a definite frame, which the search records in
``Hypothesis.emit_frames``. Reading the aux head at those nodes needs one
batched matmul over the L emitted tokens and no second pass, so partial
results are available on every chunk.

The two protocols therefore ask slightly different questions -- "where does the
search emit this token" versus "where would a forced alignment put it" -- and
can disagree on a token here and there. ``fujielab-asr`` ships the streaming
one because it is the one that can actually run online; see the model card for
the measured agreement against the offline numbers.
"""

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter
from espnet2.torch_utils.set_all_random_seed import set_all_random_seed
from espnet2.utils import config_argparse
from espnet2.utils.types import str2bool, str2triple_str, str_or_none

from fujielab.asr.espnet_ext.espnet2.asr.beam_search_transducer_online import (
    BeamSearchTransducer,
    Hypothesis,
)
from fujielab.asr.espnet_ext.espnet2.asr.frontend.online_audio_processor import (
    OnlineAudioProcessor,
)
from fujielab.asr.espnet_ext.espnet2.tasks.asr_multitask_transducer import (
    ASRMultitaskTransducerTask,
)


@dataclass
class MultitaskTransducerResult:
    """One hypothesis for the multitask Transducer model.

    ``tokens`` and ``aux_labels`` are aligned 1:1 (one auxiliary label per
    recognized base token). ``emit_frames`` gives the encoder frame each token
    was emitted at, counted from the start of the stream.
    """

    text: Optional[str]
    tokens: List[str]
    token_ids: List[int]
    aux_labels: List[str]
    emit_frames: List[int]
    score: float


class Speech2TextMultitaskTransducer:
    """Speech2Text for the multitask Transducer model (ASR + aux-info labels).

    Args:
        asr_train_config: ASR model training config path.
        asr_model_file: ASR model path.
        beam_search_config: Beam search config. Only ``search_type="maes"`` is
            exercised by the shipped models; the other search types in
            :class:`BeamSearchTransducer` predate this package's streaming
            decoder API and do not currently run.
        token_type: Type of token units.
        bpemodel: BPE model path.
        device: Device to use for inference.
        beam_size: Size of beam during search.
        dtype: Data type.
        nbest: Number of final hypotheses.
        streaming: Whether to perform chunk-by-chunk inference.
        decoding_window: Size of the decoding window (in milliseconds).
        left_context: Number of previous frames the attention module can see.
    """

    def __init__(
        self,
        asr_train_config: Union[Path, str] = None,
        asr_model_file: Union[Path, str] = None,
        beam_search_config: Dict[str, Any] = None,
        token_type: str = None,
        bpemodel: str = None,
        device: str = "cpu",
        beam_size: int = 20,
        dtype: str = "float32",
        nbest: int = 1,
        streaming: bool = True,
        decoding_window: int = 640,
        left_context: int = 32,
    ) -> None:
        super().__init__()

        asr_model, asr_train_args = ASRMultitaskTransducerTask.build_model_from_file(
            asr_train_config, asr_model_file, device
        )
        asr_model.to(dtype=getattr(torch, dtype)).eval()

        if not hasattr(asr_model, "aux_output_layer"):
            raise ValueError(
                "The loaded model has no auxiliary head. For a plain Transducer "
                "use asr_transducer_inference_cbs.Speech2Text instead."
            )

        if beam_search_config is None:
            beam_search_config = {"search_type": "maes"}
        beam_search_config.setdefault("search_type", "maes")

        beam_search = BeamSearchTransducer(
            asr_model.decoder,
            asr_model.joint_network,
            beam_size,
            lm=None,
            lm_weight=0.0,
            nbest=nbest,
            **beam_search_config,
        )

        token_list = asr_model.token_list
        if token_type is None:
            token_type = asr_train_args.token_type
        if bpemodel is None:
            bpemodel = asr_train_args.bpemodel

        if token_type == "bpe" and bpemodel is not None:
            tokenizer = build_tokenizer(token_type=token_type, bpemodel=bpemodel)
        elif token_type == "bpe":
            tokenizer = None
        else:
            tokenizer = build_tokenizer(token_type=token_type)

        self.asr_model = asr_model
        self.asr_train_args = asr_train_args
        self.device = device
        self.dtype = dtype
        self.nbest = nbest
        self.converter = TokenIDConverter(token_list=token_list)
        self.tokenizer = tokenizer
        self.beam_search = beam_search
        self.aux_token_list = list(asr_model.aux_token_list)

        self.streaming = streaming and decoding_window >= 0
        self.asr_model.encoder.dynamic_chunk_training = False
        self.left_context = max(left_context, 0)
        self.chunk_size = max(decoding_window, 0)

        if streaming:
            self.audio_processor = OnlineAudioProcessor(
                asr_model._extract_feats,
                asr_model.normalize,
                asr_train_args.frontend_conf,
                device,
            )
            self.reset_streaming_cache()

    def reset_streaming_cache(self) -> None:
        """Reset all streaming state, including the accumulated encoder output."""
        self.encoder_states = None
        self.beam_search.reset_cache()
        self.audio_processor.reset_cache()
        self.num_processed_frames = torch.tensor([[0]], device=self.device)
        # The aux head has to be read at frames that may lie in earlier chunks,
        # so the encoder output is kept for the whole utterance. One utterance of
        # 60 s is ~1.8k frames x 256 dims x 4 B ~= 1.9 MB, which is affordable;
        # callers that stream indefinitely should segment (e.g. by VAD) and let
        # this reset fire at each segment boundary.
        self.enc_out_history = None

    @torch.no_grad()
    def streaming_decode(
        self, speech: Union[torch.Tensor, np.ndarray], is_final: bool = False
    ) -> List[MultitaskTransducerResult]:
        """Decode one chunk of speech into tokens plus auxiliary labels.

        Args:
            speech: Chunk of speech data. (S,)
            is_final: Whether this is the final chunk of the utterance.

        Returns:
            Up to ``nbest`` (partial or final) hypotheses; empty while the
            recognizer has not emitted anything yet.
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

        enc_out_chunk = enc_out[0]
        if enc_out_chunk.size(0) > 0:
            self.enc_out_history = (
                enc_out_chunk
                if self.enc_out_history is None
                else torch.cat([self.enc_out_history, enc_out_chunk], dim=0)
            )

        nbest_hyps: List[Hypothesis] = []
        if enc_out_chunk.size(0) > 0:
            nbest_hyps = self.beam_search(enc_out_chunk, is_final=is_final)

        self.num_processed_frames += self.chunk_size

        # Read the aux head before the cache is dropped: reset_streaming_cache()
        # discards the encoder history the lookup needs.
        results = [self._to_result(h) for h in nbest_hyps[: self.nbest]]

        if is_final:
            self.reset_streaming_cache()

        return results

    def _to_result(self, hyp: Hypothesis) -> MultitaskTransducerResult:
        token_ids = list(hyp.yseq[1:])
        emit_frames = list(hyp.emit_frames)
        aux_labels = self.predict_aux(token_ids, emit_frames)

        tokens = self.converter.ids2tokens(token_ids)
        text = self.tokenizer.tokens2text(tokens) if self.tokenizer is not None else None
        return MultitaskTransducerResult(
            text=text,
            tokens=tokens,
            token_ids=token_ids,
            aux_labels=aux_labels,
            emit_frames=emit_frames,
            score=float(hyp.score),
        )

    @torch.no_grad()
    def predict_aux(
        self, token_ids: Sequence[int], emit_frames: Sequence[int]
    ) -> List[str]:
        """Read the auxiliary head at each token's emission node.

        Args:
            token_ids: Recognized base tokens y_1..y_L (without the leading blank).
            emit_frames: Stream-absolute encoder frame for each of those tokens.

        Returns:
            L auxiliary-label tokens, aligned 1:1 with ``token_ids``.
        """
        model = self.asr_model
        n = len(token_ids)
        if n == 0 or self.enc_out_history is None:
            return []
        if len(emit_frames) != n:
            # Without a frame per token there is no node to read, and guessing
            # one would silently fabricate labels.
            raise ValueError(
                f"emit_frames has {len(emit_frames)} entries for {n} tokens; "
                "the beam search did not record an emission frame per token."
            )

        enc = self.enc_out_history
        t_idx = torch.as_tensor(
            [min(max(t, 0), enc.size(0) - 1) for t in emit_frames],
            device=enc.device,
            dtype=torch.long,
        )

        # Predictor states u=0..L-1: state u is the one that emitted y_{u+1}.
        dec_in = torch.tensor(
            [[model.blank_id] + list(token_ids)], device=self.device, dtype=torch.long
        )
        model.decoder.set_device(self.device)
        decoder_out = model.decoder(dec_in)[0][:n]  # (L, D_dec)

        jn = model.joint_network
        h = jn.joint_activation(
            jn.lin_enc(enc.index_select(0, t_idx)) + jn.lin_dec(decoder_out)
        )  # (L, J)
        ids = model.aux_output_layer(h).argmax(dim=-1).tolist()
        return [self.aux_token_list[i] for i in ids]

    @torch.no_grad()
    def __call__(
        self, speech: Union[torch.Tensor, np.ndarray]
    ) -> List[MultitaskTransducerResult]:
        """Decode a whole utterance in one call (non-streaming convenience)."""
        if isinstance(speech, np.ndarray):
            speech = torch.as_tensor(speech)
        return self.streaming_decode(speech, is_final=True)

    @staticmethod
    def from_pretrained(model_tag: Optional[str] = None, **kwargs: Any):
        """Build from a Hugging Face / ESPnet model zoo tag."""
        if model_tag is not None:
            try:
                from espnet_model_zoo.downloader import ModelDownloader
            except ImportError:
                raise ImportError(
                    "`espnet_model_zoo` is not installed. "
                    "Please install via `pip install -U espnet_model_zoo`."
                )
            kwargs.update(**ModelDownloader().download_and_unpack(model_tag))
        return Speech2TextMultitaskTransducer(**kwargs)


def inference(
    output_dir: str,
    batch_size: int,
    dtype: str,
    ngpu: int,
    seed: int,
    num_workers: int,
    log_level: Union[int, str],
    data_path_and_name_and_type: Sequence[Tuple[str, str, str]],
    key_file: Optional[str],
    asr_train_config: Optional[str],
    asr_model_file: Optional[str],
    model_tag: Optional[str],
    token_type: Optional[str],
    bpemodel: Optional[str],
    beam_size: int,
    nbest: int,
    allow_variable_data_keys: bool,
    decoding_window: int,
    left_context: int,
) -> None:
    """Decode a dataset, writing text and a parallel aux_label file."""
    from espnet2.tasks.asr import ASRTask
    from espnet2.torch_utils.device_funcs import to_device
    from espnet2.utils.build_dataclass import build_dataclass  # noqa: F401
    from espnet.utils.cli_utils import get_commandline_args

    if batch_size > 1:
        raise NotImplementedError("batch decoding is not implemented")
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
    )
    logging.info("Decoding args:\n%s", get_commandline_args())
    if ngpu > 1:
        raise NotImplementedError("only single GPU decoding is supported")
    device = "cuda" if ngpu >= 1 else "cpu"
    set_all_random_seed(seed)

    kw = dict(
        asr_train_config=asr_train_config,
        asr_model_file=asr_model_file,
        token_type=token_type,
        bpemodel=bpemodel,
        device=device,
        dtype=dtype,
        beam_size=beam_size,
        nbest=nbest,
        streaming=True,
        decoding_window=decoding_window,
        left_context=left_context,
    )
    speech2text = (
        Speech2TextMultitaskTransducer.from_pretrained(model_tag, **kw)
        if model_tag is not None
        else Speech2TextMultitaskTransducer(**kw)
    )

    loader = ASRTask.build_streaming_iterator(
        data_path_and_name_and_type,
        dtype=dtype,
        batch_size=batch_size,
        key_file=key_file,
        num_workers=num_workers,
        preprocess_fn=None,
        collate_fn=ASRTask.build_collate_fn(argparse.Namespace(), False),
        allow_variable_data_keys=allow_variable_data_keys,
        inference=True,
    )

    out = Path(output_dir) / "1best_recog"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "text").open("w", encoding="utf-8") as f_text, (
        out / "aux_label"
    ).open("w", encoding="utf-8") as f_aux, (out / "token").open(
        "w", encoding="utf-8"
    ) as f_tok:
        for keys, batch in loader:
            assert len(keys) == 1, len(keys)
            batch = to_device(batch, device=device)
            results = speech2text(batch["speech"][0])
            key = keys[0]
            if not results:
                f_text.write(f"{key} \n")
                f_aux.write(f"{key} \n")
                f_tok.write(f"{key} \n")
                continue
            r = results[0]
            f_text.write(f"{key} {r.text if r.text is not None else ''}\n")
            f_aux.write(f"{key} {' '.join(r.aux_labels)}\n")
            f_tok.write(f"{key} {' '.join(r.tokens)}\n")


def get_parser():
    """Build the CLI parser."""
    parser = config_argparse.ArgumentParser(
        description="Multitask Transducer streaming decoding (base + aux labels)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--log_level", type=lambda x: x.upper(), default="INFO")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--ngpu", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", default="float32", choices=["float16", "float32"])
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument(
        "--data_path_and_name_and_type", type=str2triple_str, action="append"
    )
    parser.add_argument("--key_file", type=str_or_none)
    parser.add_argument("--allow_variable_data_keys", type=str2bool, default=False)
    parser.add_argument("--asr_train_config", type=str)
    parser.add_argument("--asr_model_file", type=str)
    parser.add_argument("--model_tag", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--nbest", type=int, default=1)
    parser.add_argument("--beam_size", type=int, default=20)
    parser.add_argument("--token_type", type=str_or_none, default=None)
    parser.add_argument("--bpemodel", type=str_or_none, default=None)
    parser.add_argument("--decoding_window", type=int, default=640)
    parser.add_argument("--left_context", type=int, default=32)
    return parser


def main(cmd=None):
    """CLI entry point."""
    parser = get_parser()
    args = parser.parse_args(cmd)
    kwargs = vars(args)
    kwargs.pop("config", None)
    inference(**kwargs)


if __name__ == "__main__":
    main()
