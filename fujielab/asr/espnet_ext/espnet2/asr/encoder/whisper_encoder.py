import copy
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from typeguard import typechecked

from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.specaug.specaug import SpecAug


class OpenAIWhisperEncoder(AbsEncoder):
    """Transformer-based Speech Encoder from OpenAI's Whisper Model:

    URL: https://github.com/openai/whisper
    """

    @typechecked
    def __init__(
        self,
        input_size: int = 1,
        dropout_rate: float = 0.0,
        whisper_model: str = "small",
        download_dir: Optional[str] = None,
        use_specaug: bool = False,
        specaug_conf: Union[dict, None] = None,
        do_pad_trim: bool = False,
        chunk_size: int = 0,
        right_context: int = 0,
        left_context: int = -1,
    ):
        """OpenAI Whisper encoder (optionally block-causal / streaming).

        chunk_size / right_context / left_context are in encoder frames
        (20 ms, i.e. after conv2's stride 2). chunk_size == 0 keeps the
        original bidirectional encoder. With chunk_size > 0 the self-attention
        is restricted Emformer-style: a frame in chunk k attends to
          * the real frames of chunks <= k (at most left_context frames back,
            -1 = unlimited), and
          * hard copies of the right_context frames that follow chunk k, which
            themselves attend only within chunk k's window.
        Past chunks' representations therefore never depend on later input,
        so at inference they can be cached and the look-ahead is exactly
        right_context frames regardless of depth. The copies are discarded
        from the output.
        """
        try:
            import whisper
            from whisper.audio import HOP_LENGTH, N_FFT, N_SAMPLES

            try:  # removed in openai-whisper >= 20231106 (n_mels is per model)
                from whisper.audio import N_MELS
            except ImportError:
                N_MELS = 80
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools &&",
                "./installers/install_whisper.sh",
            )
            raise e

        super().__init__()

        self.n_fft = N_FFT
        self.win_length = N_FFT
        self.hop_length = HOP_LENGTH
        self.n_mels = N_MELS

        self.mel_filters = whisper.audio.mel_filters

        # note that originally Whisper doesn't use dropouts
        self.dropout = torch.nn.Dropout(dropout_rate)

        assert whisper_model in whisper.available_models()
        _model = whisper.load_model(
            whisper_model, download_root=download_dir, device="cpu"
        )
        self.encoders = copy.deepcopy(_model.encoder)
        self.encoders.train()
        # large-v3 / turbo checkpoints take 128 mel bins, older ones 80
        self.n_mels = self.encoders.conv1.in_channels

        del _model

        if use_specaug:
            self.specaug = SpecAug(**specaug_conf)
        else:
            self.specaug = None

        self.do_pad_trim = do_pad_trim
        self.pad_samples = N_SAMPLES

        # ---- block-causal (streaming) self-attention ----
        self.chunk_size = int(chunk_size)
        self.right_context = int(right_context)
        self.left_context = int(left_context)
        self._stream_mask_cache = {}
        if self.chunk_size > 0:
            # whisper's qkv_attention turns ANY mask into is_causal=True when
            # SDPA is used, so route the encoder blocks through a version that
            # applies the boolean attention mask explicitly.
            import types

            for block in self.encoders.blocks:
                block.attn.qkv_attention = types.MethodType(
                    _masked_qkv_attention, block.attn
                )

    def output_size(self) -> int:
        return self.encoders.ln_post.normalized_shape[-1]

    def streaming_layout(self, n_frames: int, device):
        """Extended-sequence layout and attention mask for chunked attention.

        Returns (pos, real_idx, mask):
          pos      LongTensor (T',) original frame index of each extended slot
          real_idx LongTensor (T,)  slots holding the real (non-copy) frames
          mask     BoolTensor (T', T') True where query i may attend key j
        """
        key = (n_frames, str(device))
        if key in self._stream_mask_cache:
            return self._stream_mask_cache[key]
        c, r, L = self.chunk_size, self.right_context, self.left_context
        pos, chunk, is_copy = [], [], []
        n_chunks = (n_frames + c - 1) // c
        for k in range(n_chunks):
            s, e = k * c, min((k + 1) * c, n_frames)
            for t in range(s, e):
                pos.append(t); chunk.append(k); is_copy.append(False)
            for t in range(e, min(e + r, n_frames)):
                pos.append(t); chunk.append(k); is_copy.append(True)
        pos = torch.tensor(pos, dtype=torch.long)
        chunk = torch.tensor(chunk, dtype=torch.long)
        is_copy = torch.tensor(is_copy, dtype=torch.bool)
        qc, kc = chunk[:, None], chunk[None, :]
        real_key = ~is_copy[None, :]
        allowed = real_key & (kc <= qc)  # real frames of chunks <= own chunk
        if L >= 0:
            allowed &= pos[None, :] >= (qc * c - L)
        allowed |= (~real_key) & (kc == qc)  # own chunk's look-ahead copies
        real_idx = torch.nonzero(~is_copy).squeeze(1)
        out = (pos.to(device), real_idx.to(device), allowed.to(device))
        self._stream_mask_cache[key] = out
        return out

    def pad_or_trim(
        self,
        array: torch.Tensor,
        length: int,
        axis: int = -1,
    ) -> torch.Tensor:
        """Pad or trim the audio array to N_SAMPLES.

        Used in zero-shot inference cases.
        """
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length).to(array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])

        return array

    def log_mel_spectrogram(
        self,
        audio: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        """Use log-mel spectrogram computation native to Whisper training"""
        window = torch.hann_window(self.win_length).to(audio.device)
        stft = torch.stft(
            audio, self.n_fft, self.hop_length, window=window, return_complex=True
        )

        # whisper deletes the last frame by default (Shih-Lun)
        magnitudes = stft[..., :-1].abs() ** 2

        filters = self.mel_filters(audio.device, self.n_mels)
        mel_spec = filters @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()

        if ilens is not None:
            olens = ilens // self.hop_length
        else:
            olens = None

        log_spec = torch.maximum(
            log_spec,
            log_spec.view(audio.size(0), -1).max(dim=-1)[0][:, None, None] - 8.0,
        )
        log_spec = (log_spec + 4.0) / 4.0

        return log_spec, olens

    def whisper_encode(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor = None,
    ) -> torch.Tensor:
        x = F.gelu(self.encoders.conv1(input))
        x = F.gelu(self.encoders.conv2(x))
        x = x.permute(0, 2, 1)

        n_frames = x.size(1)
        max_pos = self.encoders.positional_embedding.size(0)
        if n_frames <= max_pos:
            x = (x + self.encoders.positional_embedding[: x.size(1), :]).to(x.dtype)
        else:
            # due to positional encoding, audios >30 sec won't be accepted
            x = x[:, :max_pos, :] + self.encoders.positional_embedding

        x = self.dropout(x)

        mask = None
        real_idx = None
        if self.chunk_size > 0:
            pos, real_idx, mask = self.streaming_layout(x.size(1), x.device)
            x = x.index_select(1, pos)  # real frames + look-ahead copies

        for layer, block in enumerate(self.encoders.blocks):
            x = block(x, mask=mask)
            if layer < len(self.encoders.blocks) - 1:
                x = self.dropout(x)

        if real_idx is not None:
            x = x.index_select(1, real_idx)

        x = self.encoders.ln_post(x)

        if ilens is not None:
            olens = (
                1
                + (
                    ilens
                    - self.encoders.conv2.kernel_size[0]
                    + 2 * self.encoders.conv2.padding[0]
                )
                // self.encoders.conv2.stride[0]
            )
            olens = torch.clamp(olens, max=max_pos)
        else:
            olens = None

        return x, olens

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        if self.do_pad_trim:
            xs_pad = self.pad_or_trim(xs_pad, self.pad_samples)

        feats, feats_lens = self.log_mel_spectrogram(xs_pad, ilens)

        if self.specaug is not None and self.encoders.training:
            feats = torch.transpose(feats, 1, 2)
            feats, feats_lens = self.specaug(feats, feats_lens)
            feats = torch.transpose(feats, 1, 2)

        xs_pad, olens = self.whisper_encode(feats, feats_lens)

        return xs_pad, olens, None


def _masked_qkv_attention(self, q, k, v, mask=None):
    """Replacement for whisper's MultiHeadAttention.qkv_attention that applies
    an explicit boolean attention mask (True = may attend) through SDPA.
    Bound to the encoder blocks' self-attention when chunk_size > 0."""
    n_batch, n_ctx, n_state = q.shape
    q = q.view(*q.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    k = k.view(*k.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    v = v.view(*v.shape[:2], self.n_head, -1).permute(0, 2, 1, 3)
    a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    out = a.permute(0, 2, 1, 3).flatten(start_dim=2)
    return out, None
