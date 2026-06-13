"""Multitask CTC-only ASR model with an auxiliary-information head.

CTC counterpart of :class:`ESPnetASRMultitaskModel` (Transformer decoder) and
:class:`ESPnetASRMultitaskTransducerModel` (RNN-T). The base tokens are
recognized by CTC alone (no decoder), and an auxiliary-information label
(e.g. {N, F, D}) is predicted per base token by a linear head on the encoder
output frames.

Per-token supervision is placed on the frames occupied by each token, found by
a forced Viterbi alignment of the reference token sequence over the CTC
lattice (``torchaudio.functional.forced_align``). Frames aligned to <blank>
are not supervised. At inference, the base tokens are decoded by greedy CTC
and the auxiliary label of each emitted token is read from the aux head at the
token's emission frames.
"""

from typing import Dict, List, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetASRMultitaskCTCModel(ESPnetASRModel):
    """CTC-only ASR model with a parallel auxiliary-information head."""

    @typechecked
    def __init__(
        self,
        *args,
        aux_token_list: Union[Tuple[str, ...], List[str]],
        aux_vocab_size: int,
        aux_weight: float = 0.3,
        aux_lsm_weight: float = 0.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        if self.ctc_weight != 1.0:
            raise ValueError(
                "ESPnetASRMultitaskCTCModel is CTC-only; set ctc_weight to 1.0 "
                f"(got {self.ctc_weight})."
            )
        if self.decoder is not None:
            raise ValueError(
                "ESPnetASRMultitaskCTCModel is CTC-only; do not give a decoder."
            )

        self.aux_token_list = list(aux_token_list)
        self.aux_vocab_size = aux_vocab_size
        self.aux_weight = aux_weight

        # Auxiliary head on the encoder output frames (purely acoustic input).
        encoder_output_size = self.encoder.output_size()
        self.aux_output_layer = torch.nn.Linear(encoder_output_size, aux_vocab_size)
        self.criterion_aux = LabelSmoothingLoss(
            size=aux_vocab_size,
            padding_idx=self.ignore_id,
            smoothing=aux_lsm_weight,
            normalize_length=False,
        )

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        aux_label: torch.Tensor = None,
        aux_label_lengths: torch.Tensor = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + CTC loss + per-token auxiliary loss."""
        assert text_lengths.dim() == 1, text_lengths.shape
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)

        if self.training:
            assert aux_label is not None and aux_label_lengths is not None, (
                "aux_label is required for training the multitask model."
            )
        batch_size = speech.shape[0]

        text[text == -1] = self.ignore_id
        text = text[:, : text_lengths.max()]
        if aux_label is not None:
            aux_label[aux_label == -1] = self.ignore_id
            aux_label = aux_label[:, : aux_label_lengths.max()]
            assert (aux_label_lengths == text_lengths).all(), (
                "aux_label length must equal text length per utterance: "
                f"{aux_label_lengths} vs {text_lengths}"
            )

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)
        intermediate_outs = None
        if isinstance(encoder_out, tuple):
            intermediate_outs = encoder_out[1]
            encoder_out = encoder_out[0]

        stats = dict()

        # 2. CTC branch (the base recognizer)
        loss_ctc, cer_ctc = self._calc_ctc_loss(
            encoder_out, encoder_out_lens, text, text_lengths
        )
        stats["loss_ctc"] = loss_ctc.detach()
        stats["cer_ctc"] = cer_ctc

        # Intermediate CTC (optional)
        if self.interctc_weight != 0.0 and intermediate_outs is not None:
            loss_interctc = 0.0
            for layer_idx, intermediate_out in intermediate_outs:
                loss_ic, cer_ic = self._calc_ctc_loss(
                    intermediate_out, encoder_out_lens, text, text_lengths
                )
                loss_interctc = loss_interctc + loss_ic
                stats["loss_interctc_layer{}".format(layer_idx)] = loss_ic.detach()
                stats["cer_interctc_layer{}".format(layer_idx)] = cer_ic
            loss_interctc = loss_interctc / len(intermediate_outs)
            loss_ctc = (
                1 - self.interctc_weight
            ) * loss_ctc + self.interctc_weight * loss_interctc

        # 3. Auxiliary-information branch
        if aux_label is not None:
            loss_aux, acc_aux = self._calc_aux_loss(
                encoder_out, encoder_out_lens, text, text_lengths, aux_label
            )
            loss = loss_ctc + self.aux_weight * loss_aux
            stats["loss_aux"] = loss_aux.detach()
            stats["acc_aux"] = acc_aux
        else:
            loss = loss_ctc

        stats["loss"] = loss.detach()

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _calc_aux_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        aux_label: torch.Tensor,
    ):
        """Per-token auxiliary CE on the CTC forced-aligned frames."""
        frame_targets = self._ctc_aligned_frame_targets(
            encoder_out, encoder_out_lens, text, text_lengths, aux_label
        )  # (B, T) aux ids; ignore_id on blank/pad frames

        aux_logits = self.aux_output_layer(encoder_out)  # (B, T, A)
        loss_aux = self.criterion_aux(aux_logits, frame_targets)
        acc_aux = th_accuracy(
            aux_logits.reshape(-1, self.aux_vocab_size),
            frame_targets,
            ignore_label=self.ignore_id,
        )
        return loss_aux, acc_aux

    @torch.no_grad()
    def _ctc_aligned_frame_targets(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        aux_label: torch.Tensor,
    ) -> torch.Tensor:
        """Map the per-token aux labels onto frames via CTC forced alignment.

        Returns:
            frame_targets: (B, T) aux label id per frame; frames aligned to
            <blank> (and padding / infeasible utterances) get ignore_id.
        """
        from torchaudio.functional import forced_align

        B, T, _ = encoder_out.shape
        device = encoder_out.device
        log_probs = self.ctc.log_softmax(encoder_out).float()  # (B, T, V)

        frame_targets = text.new_full((B, T), self.ignore_id)
        for b in range(B):
            T_b = int(encoder_out_lens[b].item())
            L_b = int(text_lengths[b].item())
            if L_b == 0 or T_b <= 0:
                continue
            target = text[b, :L_b].unsqueeze(0)  # (1, L)
            # CTC feasibility: T must cover all tokens + a blank between repeats
            n_repeats = int((target[0, 1:] == target[0, :-1]).sum().item())
            if T_b < L_b + n_repeats:
                continue
            try:
                aligned, _ = forced_align(
                    log_probs[b : b + 1, :T_b].contiguous(),
                    target.contiguous(),
                    blank=self.blank_id,
                )
            except Exception:
                continue
            ali = aligned[0]  # (T_b,) token id per frame (incl. blank)
            nonblank = ali != self.blank_id
            if not bool(nonblank.any()):
                continue
            prev = torch.full_like(ali, self.blank_id)
            prev[1:] = ali[:-1]
            # a new token starts on a non-blank frame after blank or a different token
            new_token = nonblank & ((prev == self.blank_id) | (ali != prev))
            token_idx = torch.cumsum(new_token.long(), dim=0) - 1  # (T_b,)
            aux_b = aux_label[b].to(device)
            frame_targets[b, :T_b] = torch.where(
                nonblank, aux_b[token_idx.clamp(min=0)], frame_targets[b, :T_b]
            )
        return frame_targets

    @torch.no_grad()
    def ctc_greedy_with_aux(
        self, encoder_out: torch.Tensor, encoder_out_lens: torch.Tensor
    ):
        """Greedy CTC decoding plus aux labels at the emission frames.

        Args:
            encoder_out: (B, T, D) encoder output.
            encoder_out_lens: (B,) valid lengths.
        Returns:
            List over the batch of (token_ids, aux_ids, score) where token_ids
            and aux_ids are aligned 1:1 and score is the summed frame-wise
            log-probability of the greedy path.
        """
        log_probs = self.ctc.log_softmax(encoder_out)  # (B, T, V)
        aux_logits = self.aux_output_layer(encoder_out)  # (B, T, A)
        results = []
        for b in range(log_probs.size(0)):
            T_b = int(encoder_out_lens[b].item())
            lp = log_probs[b, :T_b]
            best_lp, ali = lp.max(dim=-1)  # (T_b,)
            score = float(best_lp.sum().item())
            nonblank = ali != self.blank_id
            prev = torch.full_like(ali, self.blank_id)
            prev[1:] = ali[:-1]
            new_token = nonblank & ((prev == self.blank_id) | (ali != prev))
            token_ids = ali[new_token].tolist()
            # aux label per token: aggregate the aux logits over the token's frames
            token_idx = torch.cumsum(new_token.long(), dim=0) - 1
            aux_ids = []
            for k in range(len(token_ids)):
                frames = (token_idx == k) & nonblank
                aux_ids.append(int(aux_logits[b, :T_b][frames].sum(0).argmax().item()))
            results.append((token_ids, aux_ids, score))
        return results
