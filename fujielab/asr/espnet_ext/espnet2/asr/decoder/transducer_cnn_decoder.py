"""CNN-Transducer (CNN-T) decoder definition.

A convolutional prediction network for the (RNN-)Transducer: the label history
is encoded by a stack of *causal* 1-D convolutions over the token embeddings
instead of an LSTM/GRU. The receptive field is finite
(``context = 1 + num_layers * (kernel_size - 1)`` tokens including the current
one), which makes the decoder state a plain window of the last few label IDs:

* training-time ``forward`` and search-time ``score``/``batch_score`` are
  *exactly* consistent (the same valid convolution over a blank-padded window),
* hypothesis states are trivially serializable/mergeable (no hidden vectors),
* one-step scoring is cheap and batch-friendly.

This mirrors stateless/conv prediction-network transducers (e.g. Ghodsi et al.,
"RNN-Transducer with stateless prediction network", ICASSP 2020; ConvT), and is
exposed to configs as ``decoder: cnn_transducer``.

Interface-compatible with :class:`espnet2.asr.decoder.transducer_decoder
.TransducerDecoder` (init_state/score/batch_score/select_state/
create_batch_states), so it works with the existing transducer loss, the
multitask transducer model, and all beam-search implementations. The decoder
state is a ``(B, context - 1)`` LongTensor holding the most recent label IDs
(left-padded with the blank ID, whose embedding is fixed to zero via
``padding_idx``).
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from typeguard import typechecked

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.transducer.beam_search_transducer import ExtendedHypothesis, Hypothesis


class CNNTransducerDecoder(AbsDecoder):
    """CNN-T decoder module (convolutional prediction network).

    Args:
        vocab_size: Output dimension.
        num_layers: Number of causal convolution blocks.
        hidden_size: Embedding/channel dimension (also the decoder output dim).
        kernel_size: Convolution kernel size (per block; context over labels).
        dropout: Dropout rate applied inside each convolution block.
        dropout_embed: Dropout rate for the embedding layer.
        embed_pad: Embed/Blank symbol ID (its embedding is a fixed zero vector).

    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        num_layers: int = 2,
        hidden_size: int = 512,
        kernel_size: int = 4,
        dropout: float = 0.1,
        dropout_embed: float = 0.1,
        embed_pad: int = 0,
    ):
        if num_layers < 1:
            raise ValueError(f"num_layers must be >= 1: {num_layers}")
        if kernel_size < 2:
            raise ValueError(f"kernel_size must be >= 2: {kernel_size}")

        super().__init__()

        self.embed = torch.nn.Embedding(vocab_size, hidden_size, padding_idx=embed_pad)
        self.dropout_embed = torch.nn.Dropout(p=dropout_embed)

        self.convs = torch.nn.ModuleList(
            [
                torch.nn.Conv1d(hidden_size, hidden_size, kernel_size)
                for _ in range(num_layers)
            ]
        )
        self.norms = torch.nn.ModuleList(
            [torch.nn.LayerNorm(hidden_size) for _ in range(num_layers)]
        )
        self.dropout_conv = torch.nn.ModuleList(
            [torch.nn.Dropout(p=dropout) for _ in range(num_layers)]
        )
        self.activation = torch.nn.SiLU()

        self.dlayers = num_layers
        self.dunits = hidden_size
        self.kernel_size = kernel_size
        self.dtype = "cnn"
        self.odim = vocab_size

        # Receptive field in labels, including the current one.
        self.context = 1 + num_layers * (kernel_size - 1)

        self.ignore_id = -1
        self.blank_id = embed_pad

        self.device = next(self.parameters()).device

    def set_device(self, device: torch.device):
        """Set GPU device to use.

        Args:
            device: Device ID.

        """
        self.device = device

    def init_state(self, batch_size: int) -> torch.Tensor:
        """Initialize decoder states.

        Args:
            batch_size: Batch size.

        Returns:
            : Initial label-window state, all blanks. (B, context - 1)

        """
        return torch.full(
            (batch_size, self.context - 1),
            self.blank_id,
            dtype=torch.long,
            device=self.device,
        )

    def _conv_forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Run the embedding + causal conv stack over blank-padded labels.

        Args:
            labels: Label ID sequences, ALREADY left-padded with
                (context - 1) blanks. (B, U + context - 1)

        Returns:
            out: Decoder output sequences. (B, U, D_dec)

        """
        x = self.dropout_embed(self.embed(labels))  # (B, U + ctx - 1, D)

        for conv, norm, drop in zip(self.convs, self.norms, self.dropout_conv):
            # Valid (no-padding) conv: length shrinks by (kernel_size - 1);
            # the left context was provided up-front, so this stays causal.
            residual = x[:, self.kernel_size - 1 :, :]
            y = conv(x.transpose(1, 2)).transpose(1, 2)  # (B, U', D)
            x = residual + drop(self.activation(norm(y)))

        return x

    def forward(self, labels: torch.Tensor) -> torch.Tensor:
        """Encode source label sequences.

        Args:
            labels: Label ID sequences. (B, L)

        Returns:
            dec_out: Decoder output sequences. (B, L, D_dec)

        """
        pad = labels.new_full((labels.size(0), self.context - 1), self.blank_id)
        return self._conv_forward(torch.cat([pad, labels], dim=1))

    def _window_step(
        self, states: torch.Tensor, labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Consume one label per hypothesis.

        Args:
            states: Label-window states. (B, context - 1)
            labels: Labels to consume. (B, 1)

        Returns:
            dec_out: Decoder outputs. (B, D_dec)
            new_states: Updated label-window states. (B, context - 1)

        """
        window = torch.cat([states, labels], dim=1)  # (B, context)
        dec_out = self._conv_forward(window)[:, -1, :]  # (B, D_dec)
        return dec_out, window[:, 1:]

    def score(
        self, hyp: Hypothesis, cache: Dict[str, Any]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-step forward hypothesis.

        Args:
            hyp: Hypothesis.
            cache: Pairs of (dec_out, state) for each label sequence. (key)

        Returns:
            dec_out: Decoder output sequence. (D_dec,)
            new_state: Updated label-window state. (1, context - 1)
            label: Label ID for LM. (1,)

        """
        label = torch.full((1, 1), hyp.yseq[-1], dtype=torch.long, device=self.device)

        str_labels = "_".join(list(map(str, hyp.yseq)))

        if str_labels in cache:
            dec_out, dec_state = cache[str_labels]
        else:
            dec_out, dec_state = self._window_step(hyp.dec_state.to(self.device), label)
            cache[str_labels] = (dec_out, dec_state)

        return dec_out[0], dec_state, label[0]

    def batch_score(
        self,
        hyps: Union[List[Hypothesis], List[ExtendedHypothesis]],
        dec_states: torch.Tensor,
        cache: Dict[str, Any],
        use_lm: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """One-step forward hypotheses.

        Args:
            hyps: Hypotheses.
            dec_states: Label-window states. (B, context - 1)
            cache: Pairs of (dec_out, dec_state) for each label sequence. (keys)
            use_lm: Whether to compute label ID sequences for LM.

        Returns:
            dec_out: Decoder output sequences. (B, D_dec)
            dec_states: Updated label-window states. (B, context - 1)
            lm_labels: Label ID sequences for LM. (B, 1)

        """
        final_batch = len(hyps)

        process = []
        done = [None] * final_batch

        for i, hyp in enumerate(hyps):
            str_labels = "_".join(list(map(str, hyp.yseq)))

            if str_labels in cache:
                done[i] = cache[str_labels]
            else:
                process.append((str_labels, hyp.yseq[-1], hyp.dec_state))

        if process:
            labels = torch.tensor(
                [[p[1]] for p in process], device=self.device, dtype=torch.long
            )
            p_dec_states = torch.cat(
                [p[2].to(self.device) for p in process], dim=0
            )  # (B', context - 1)

            p_dec_out, p_new_states = self._window_step(p_dec_states, labels)

        j = 0
        for i in range(final_batch):
            if done[i] is None:
                state = p_new_states[j : j + 1]

                done[i] = (p_dec_out[j : j + 1], state)
                cache[process[j][0]] = (p_dec_out[j : j + 1], state)

                j += 1

        dec_out = torch.cat([d[0] for d in done], dim=0)
        dec_states = torch.cat([d[1] for d in done], dim=0)

        if use_lm:
            lm_labels = torch.tensor(
                [h.yseq[-1] for h in hyps], device=self.device, dtype=torch.long
            ).view(final_batch, 1)

            return dec_out, dec_states, lm_labels

        return dec_out, dec_states, None

    def select_state(self, states: torch.Tensor, idx: int) -> torch.Tensor:
        """Get specified ID state from decoder states.

        Args:
            states: Label-window states. (B, context - 1)
            idx: State ID to extract.

        Returns:
            : Label-window state for given ID. (1, context - 1)

        """
        return states[idx : idx + 1]

    def create_batch_states(
        self,
        states: torch.Tensor,
        new_states: List[torch.Tensor],
        check_list: Optional[List] = None,
    ) -> torch.Tensor:
        """Create batched decoder states.

        Args:
            states: Label-window states. (B, context - 1) (unused template)
            new_states: Label-window states. [B x (1, context - 1)]

        Returns:
            states: Batched label-window states. (B, context - 1)

        """
        return torch.cat([s.to(self.device) for s in new_states], dim=0)
