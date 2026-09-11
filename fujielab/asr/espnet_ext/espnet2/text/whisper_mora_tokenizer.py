"""Whitespace tokenizer for the ``whisper_mora`` token type.

The text is already a space-separated sequence of mora-level tokens (the same
units as the ``word`` token type of the csj_cejc_aux recipes); this tokenizer
only splits / joins on whitespace. The mapping of those tokens onto the
Whisper decoder vocabulary is done by
:class:`espnet2.text.whisper_mora_token_id_converter.WhisperMoraTokenIDConverter`
together with a token list produced by
``espnet2.bin.whisper_mora_export_vocabulary``.
"""

from typing import Iterable, List

from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer


class WhisperMoraTokenizer(AbsTokenizer):
    @typechecked
    def __init__(self, delimiter: str = " "):
        self.delimiter = delimiter

    def __repr__(self):
        return f"{self.__class__.__name__}(delimiter={self.delimiter!r})"

    def text2tokens(self, line: str) -> List[str]:
        return line.split()

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.delimiter.join(tokens)
