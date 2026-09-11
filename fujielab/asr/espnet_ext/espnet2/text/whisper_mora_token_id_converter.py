"""Token <-> id converter for the ``whisper_mora`` token type.

The token list is a full Whisper decoder vocabulary (one line per id, see
``espnet2.bin.whisper_mora_export_vocabulary``) followed by the mora tokens
that are not single Whisper tokens, so a plain :class:`TokenIDConverter`
lookup already yields the right ids. On top of that this class

* prepends the Whisper task prefix ``<|ja|> <|transcribe|> <|notimestamps|>``
  to every id sequence (the ``<|startoftranscript|>`` token itself is the
  model's ``sos`` and is added by ``add_sos_eos``), mirroring what
  :class:`OpenAIWhisperTokenIDConverter` does with
  ``sot_sequence_including_notimestamps[1:]``;
* drops every Whisper special token (``<|...|>``) when converting ids back to
  tokens, so the output is the plain mora sequence.
"""

from pathlib import Path
from typing import Iterable, List, Union

import numpy as np
from typeguard import typechecked

from espnet2.text.token_id_converter import TokenIDConverter

SPECIAL_PREFIX = "<|"


class WhisperMoraTokenIDConverter(TokenIDConverter):
    @typechecked
    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]],
        unk_symbol: str = "<unk>",
        language: str = "ja",
        task: str = "transcribe",
    ):
        super().__init__(token_list=token_list, unk_symbol=unk_symbol)
        self.language = language
        self.task = task
        self.prefix_tokens = [f"<|{language}|>", f"<|{task}|>", "<|notimestamps|>"]
        missing = [t for t in self.prefix_tokens if t not in self.token2id]
        if missing:
            raise RuntimeError(
                f"Whisper prefix tokens {missing} are not in the token list; "
                "was it produced by espnet2.bin.whisper_mora_export_vocabulary?"
            )
        self.prefix_ids = [self.token2id[t] for t in self.prefix_tokens]
        self.special_ids = {
            i for i, t in enumerate(self.token_list) if t.startswith(SPECIAL_PREFIX)
        }

    @staticmethod
    def is_special(token: str) -> bool:
        return token.startswith(SPECIAL_PREFIX)

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return list(self.prefix_ids) + super().tokens2ids(tokens)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        return [t for t in super().ids2tokens(integers) if not self.is_special(t)]
