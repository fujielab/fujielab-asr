#!/usr/bin/env python3
"""Utterance-level recognition with a fujielab Whisper large-v3 + LoRA model.

    pip install "fujielab-asr[whisper]"
    python examples/run_whisper_asr.py utterance.wav [--device cuda]

The model outputs a space-separated mora sequence; in the composite encoding a
filler mora is written ``ア+F`` and a repair mora ``ン+D``.
"""
import argparse

import soundfile as sf

from fujielab.asr.espnet_ext.espnet2.bin.asr_whisper_inference import Speech2TextWhisper

TAG = "fujielab/espnet_asr_csj_cejc_pron_comp_whisper_large_v3_lora_stream_c100_la0"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("wav", help="16 kHz mono wav, at most 30 s")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--beam_size", type=int, default=5)
    args = ap.parse_args()

    s2t = Speech2TextWhisper.from_pretrained(args.tag, device=args.device, beam_size=args.beam_size)
    audio, fs = sf.read(args.wav)
    assert fs == 16000, "expected 16 kHz audio"
    for r in s2t(audio):
        print(f"{r.score:8.2f}  {r.text}")
        plain = " ".join(t.split("+")[0] for t in r.tokens if t not in ("<sp>",))
        print(f"          {plain}   (markers stripped)")


if __name__ == "__main__":
    main()
