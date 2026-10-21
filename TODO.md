# TODO

- **Cascade: event-stream API for the waseda-kyowa dialogue system.**
  `fujielab.asr.icefall_ext` has the recognizer of
  `pron-based-asr/demo/cascade_live/cascade_stream.py` (`CascadeStreamer` →
  `Speech2TextCascade`) and the live demo, but not the parts the dialogue system
  and the TTS data pipeline use:
  - `Segmenter`: webrtcvad utterance segmentation (same rules and defaults as
    waseda-kyowa's `_UtteranceSegmenter`), with the segmentation versions
    `PIPELINES` p1 / p2 / p3 (p3 is the default agreed with the TTS side).
  - `AsrSession`: 16-bit PCM of a whole conversation → time-stamped
    `utt` / `pron` / `text` / `vad` events (`t_audio`, `t_tok`, `t_wall`);
    spec in `pron-based-asr/docs/tts-interface.md`.
  - `asr_events.py`: offline JSONL event writer (`--realtime`, `--diff`,
    `--dump-enc` for VAP), and `keep_enc` / `enc_log` on the streamer.
  - Extra dependencies: `webrtcvad==2.0.10`, `setuptools<81` (webrtcvad imports
    `pkg_resources`), `soundfile`.

  Done when waseda-kyowa imports from `fujielab.asr.icefall_ext` instead of
  `cascade_stream`, and `asr_events.py --diff` shows the same events as before.
