"""Multitask RNN-Transducer ASR model with an auxiliary-information head.

Transducer counterpart of :class:`ESPnetASRMultitaskModel`. The base tokens are
recognized by an RNN-Transducer (encoder + predictor + joint network), and an
auxiliary-information label (e.g. {N, F, D}) is predicted per base token by an
auxiliary head that **shares the joint network's pre-output hidden state**
(``h = act(lin_enc(enc) + lin_dec(dec))``). The aux head therefore takes acoustic
(encoder) information as input while being conditioned on the predicted tokens via
the predictor state — the Token Dependency Mechanism (TDM).

Per-token supervision is placed on the emission node of each token, found by a
forced Viterbi alignment of the reference token sequence over the main
transducer's (T, U) lattice.
"""

from typing import Dict, List, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.asr_transducer.utils import get_transducer_task_io
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,
)
from espnet2.torch_utils.device_funcs import force_gatherable


class ESPnetASRMultitaskTransducerModel(ESPnetASRModel):
    """RNN-Transducer ASR model with a parallel auxiliary-information head (TDM)."""

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

        if not self.use_transducer_decoder:
            raise ValueError(
                "ESPnetASRMultitaskTransducerModel requires a transducer decoder "
                "(decoder: transducer + joint_network)."
            )

        self.aux_token_list = list(aux_token_list)
        self.aux_vocab_size = aux_vocab_size
        self.aux_weight = aux_weight

        # Auxiliary head on the joint network's pre-output hidden (acoustic + token).
        joint_space_size = self.joint_network.lin_out.in_features
        self.aux_output_layer = torch.nn.Linear(joint_space_size, aux_vocab_size)
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
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        stats = dict()

        # 2. CTC branch (optional)
        loss_ctc = None
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        # 3. Transducer branch + auxiliary head
        (
            loss_transducer,
            cer_transducer,
            wer_transducer,
            loss_aux,
            acc_aux,
        ) = self._calc_transducer_aux_loss(
            encoder_out, encoder_out_lens, text, aux_label
        )

        loss = loss_transducer + self.aux_weight * loss_aux
        if loss_ctc is not None:
            loss = loss + self.ctc_weight * loss_ctc

        stats["loss_transducer"] = loss_transducer.detach()
        stats["cer_transducer"] = cer_transducer
        stats["wer_transducer"] = wer_transducer
        stats["loss_aux"] = loss_aux.detach()
        stats["acc_aux"] = acc_aux
        stats["loss"] = loss.detach()

        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def _joint_hidden_and_out(self, encoder_out, decoder_out):
        """Compute joint pre-output hidden h and token logits, sharing the joint net."""
        jn = self.joint_network
        h = jn.joint_activation(
            jn.lin_enc(encoder_out.unsqueeze(2)) + jn.lin_dec(decoder_out.unsqueeze(1))
        )  # (B, T, U, J)
        joint_out = jn.lin_out(h)  # (B, T, U, V)
        return h, joint_out

    def _calc_transducer_aux_loss(self, encoder_out, encoder_out_lens, labels, aux_label):
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )
        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)  # (B, U, D_dec), U = Lmax + 1

        h, joint_out = self._joint_hidden_and_out(encoder_out, decoder_out)

        loss_transducer = self.criterion_transducer(joint_out, target, t_len, u_len)

        # --- auxiliary loss on the per-token emission nodes ---
        # aux target aligned with the L base tokens (same shape as `target`)
        aux_target = self._build_aux_target(aux_label, target.size(1))
        # forced Viterbi alignment over the (T, U) lattice -> emission frame per token
        emit_frame = self._forced_align(
            joint_out.detach(), target, t_len, u_len, self.blank_id
        )  # (B, Lmax) frame index for token k (state u=k); -1 for padding
        # gather joint hidden at (emit_frame[k], u=k) for k=0..L-1 (state predicting y_{k+1})
        B, T, U, J = h.shape
        Lmax = target.size(1)
        valid = emit_frame >= 0  # (B, Lmax)
        t_idx = emit_frame.clamp(min=0)  # (B, Lmax)
        u_idx = torch.arange(Lmax, device=h.device).unsqueeze(0).expand(B, Lmax)
        b_idx = torch.arange(B, device=h.device).unsqueeze(1).expand(B, Lmax)
        h_sel = h[b_idx, t_idx, u_idx]  # (B, Lmax, J)
        aux_logits = self.aux_output_layer(h_sel)  # (B, Lmax, A)
        # mask out positions without a valid emission frame
        aux_target = aux_target.masked_fill(~valid, self.ignore_id)
        loss_aux = self.criterion_aux(aux_logits, aux_target)
        acc_aux = th_accuracy(
            aux_logits.reshape(-1, self.aux_vocab_size),
            aux_target,
            ignore_label=self.ignore_id,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer, loss_aux, acc_aux

    def _build_aux_target(self, aux_label, length):
        """Pad/truncate aux labels to (B, length) with ignore_id padding."""
        B = aux_label.size(0)
        out = aux_label.new_full((B, length), self.ignore_id)
        n = min(length, aux_label.size(1))
        out[:, :n] = aux_label[:, :n]
        # any remaining pad value already ignore_id
        out[out == -1] = self.ignore_id
        return out

    @staticmethod
    def _forced_align(joint_out, target, t_len, u_len, blank_id):
        """Forced Viterbi alignment of the reference tokens over the RNN-T lattice.

        Args:
            joint_out: (B, T, U, V) joint logits (U = Lmax + 1).
            target: (B, Lmax) reference token ids (y_1..y_L).
            t_len: (B,) encoder lengths.
            u_len: (B,) label lengths (L per utt).
            blank_id: blank index.
        Returns:
            emit_frame: (B, Lmax) frame index at which token k (state u=k) is emitted,
                        -1 for padding positions.
        """
        B, T, U, V = joint_out.shape
        Lmax = U - 1
        device = joint_out.device
        lp = torch.log_softmax(joint_out.float(), dim=-1)  # (B,T,U,V)
        blank_lp = lp[..., blank_id]  # (B,T,U)
        # emit_lp[b,t,u] = lp at node (t,u) of the token emitted from state u = target[b,u]
        tgt = target.long().clamp(min=0)  # (B, Lmax)
        emit_lp_full = torch.gather(
            lp[:, :, :Lmax, :], 3, tgt.view(B, 1, Lmax, 1).expand(B, T, Lmax, 1)
        ).squeeze(3)  # (B, T, Lmax) ; emit from state u in 0..Lmax-1

        NEG = float("-inf")
        alpha = torch.full((B, T, U), NEG, device=device)
        back = torch.zeros((B, T, U), dtype=torch.int8, device=device)  # 0=blank,1=emit
        alpha[:, 0, 0] = 0.0
        # fill u=0 column (only blank transitions along t)
        for t in range(1, T):
            alpha[:, t, 0] = alpha[:, t - 1, 0] + blank_lp[:, t - 1, 0]
            back[:, t, 0] = 0
        for u in range(1, U):
            for t in range(0, T):
                from_emit = alpha[:, t, u - 1] + emit_lp_full[:, t, u - 1]
                if t == 0:
                    alpha[:, t, u] = from_emit
                    back[:, t, u] = 1
                else:
                    from_blank = alpha[:, t - 1, u] + blank_lp[:, t - 1, u]
                    use_emit = from_emit >= from_blank
                    alpha[:, t, u] = torch.where(use_emit, from_emit, from_blank)
                    back[:, t, u] = use_emit.to(torch.int8)

        # backtrack per utterance from (t_len-1, u_len)
        emit_frame = torch.full((B, Lmax), -1, dtype=torch.long, device=device)
        back_cpu = back.cpu()
        for b in range(B):
            Tb = int(t_len[b].item())
            Lb = int(u_len[b].item())
            t, u = Tb - 1, Lb
            while u > 0 and t >= 0:
                if back_cpu[b, t, u] == 1:  # emit: token u emitted at frame t (state u-1)
                    emit_frame[b, u - 1] = t
                    u -= 1
                else:  # blank
                    t -= 1
        return emit_frame
