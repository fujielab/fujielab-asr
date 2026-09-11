"""ASR task for the Whisper (+LoRA, optionally block-causal) models.

Same as stock ``espnet2.tasks.asr.ASRTask`` except that
  * ``encoder: whisper`` / ``decoder: whisper`` resolve to the extended classes
    in this package (streaming ``chunk_size`` / ``right_context`` options,
    ``return_hs``, and compatibility with current ``openai-whisper``), and
  * ``build_model_from_file`` inserts the LoRA layers with ``merge_weights=False``
    before loading the adapter checkpoint (see ``layers/lora.py``).
"""

import argparse
import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import yaml

from espnet2.asr.ctc import CTC
from espnet2.tasks.asr import (
    ASRTask,
    decoder_choices as _stock_decoder_choices,
    encoder_choices as _stock_encoder_choices,
    frontend_choices,
    model_choices,
    normalize_choices,
    postencoder_choices,
    preencoder_choices,
    specaug_choices,
)
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.espnet_model import ESPnetASRModel
from espnet2.torch_utils.initialize import initialize
from espnet2.train.class_choices import ClassChoices

from fujielab.asr.espnet_ext.espnet2.asr.decoder.whisper_decoder import (
    OpenAIWhisperDecoder,
)
from fujielab.asr.espnet_ext.espnet2.asr.encoder.whisper_encoder import (
    OpenAIWhisperEncoder,
)
from fujielab.asr.espnet_ext.espnet2.layers.lora import create_lora_adapter

encoder_choices = ClassChoices(
    "encoder",
    classes=dict(_stock_encoder_choices.classes, whisper=OpenAIWhisperEncoder),
    type_check=AbsEncoder,
    default="rnn",
)
decoder_choices = ClassChoices(
    "decoder",
    classes=dict(_stock_decoder_choices.classes, whisper=OpenAIWhisperDecoder),
    type_check=AbsDecoder,
    default="rnn",
    optional=True,
)


class ASRWhisperTask(ASRTask):
    """ASRTask whose ``whisper`` encoder/decoder are the fujielab extensions."""

    @classmethod
    def build_model(cls, args: argparse.Namespace) -> ESPnetASRModel:
        if isinstance(args.token_list, str):
            with open(args.token_list, encoding="utf-8") as f:
                token_list = [line.rstrip() for line in f]
            args.token_list = list(token_list)
        elif isinstance(args.token_list, (tuple, list)):
            token_list = list(args.token_list)
        else:
            raise RuntimeError("token_list must be str or list")
        vocab_size = len(token_list)
        logging.info(f"Vocabulary size: {vocab_size}")

        if args.input_size is None:
            frontend = frontend_choices.get_class(args.frontend)(**args.frontend_conf)
            input_size = frontend.output_size()
        else:
            args.frontend, args.frontend_conf, frontend = None, {}, None
            input_size = args.input_size
        specaug = (
            specaug_choices.get_class(args.specaug)(**args.specaug_conf)
            if args.specaug is not None
            else None
        )
        normalize = (
            normalize_choices.get_class(args.normalize)(**args.normalize_conf)
            if args.normalize is not None
            else None
        )
        if getattr(args, "preencoder", None) is not None:
            preencoder = preencoder_choices.get_class(args.preencoder)(**args.preencoder_conf)
            input_size = preencoder.output_size()
        else:
            preencoder = None
        encoder = encoder_choices.get_class(args.encoder)(input_size=input_size, **args.encoder_conf)
        encoder_output_size = encoder.output_size()
        if getattr(args, "postencoder", None) is not None:
            postencoder = postencoder_choices.get_class(args.postencoder)(
                input_size=encoder_output_size, **args.postencoder_conf
            )
            encoder_output_size = postencoder.output_size()
        else:
            postencoder = None
        if args.decoder == "transducer":
            raise ValueError("ASRWhisperTask does not support the transducer decoder")
        decoder = decoder_choices.get_class(args.decoder)(
            vocab_size=vocab_size, encoder_output_size=encoder_output_size, **args.decoder_conf
        )
        ctc = CTC(odim=vocab_size, encoder_output_size=encoder_output_size, **args.ctc_conf)
        model_class = model_choices.get_class(getattr(args, "model", "espnet"))
        model = model_class(
            vocab_size=vocab_size,
            frontend=frontend,
            specaug=specaug,
            normalize=normalize,
            preencoder=preencoder,
            encoder=encoder,
            postencoder=postencoder,
            decoder=decoder,
            ctc=ctc,
            joint_network=None,
            token_list=token_list,
            **args.model_conf,
        )
        if args.init is not None:
            initialize(model, args.init)
        return model

    @classmethod
    def build_model_from_file(
        cls,
        config_file: Optional[Union[Path, str]] = None,
        model_file: Optional[Union[Path, str]] = None,
        device: str = "cpu",
    ) -> Tuple[ESPnetASRModel, argparse.Namespace]:
        if config_file is None:
            assert model_file is not None
            config_file = Path(model_file).parent / "config.yaml"
        with Path(config_file).open("r", encoding="utf-8") as f:
            args = argparse.Namespace(**yaml.safe_load(f))
        model = cls.build_model(args)
        if not isinstance(model, ESPnetASRModel):
            raise RuntimeError(f"model must inherit ESPnetASRModel, but got {type(model)}")
        if device != "mps":
            model.to(device)
        use_adapter = bool(getattr(args, "use_adapter", False))
        if use_adapter:
            if args.adapter != "lora":
                raise ValueError(f"adapter {args.adapter} is not supported")
            create_lora_adapter(model, **args.adapter_conf)
            model.to(device)
        if model_file is not None:
            state_dict = torch.load(model_file, map_location=device, weights_only=False)
            if "module" in state_dict:
                state_dict = state_dict["module"]
            missing, unexpected = model.load_state_dict(state_dict, strict=not use_adapter)
            if unexpected:
                logging.warning(f"unexpected keys in {model_file}: {unexpected[:5]} ...")
        return model, args
