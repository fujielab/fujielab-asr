# fujielab-asr

English | [日本語](https://github.com/fujielab/fujielab-asr/blob/main/README_ja.md)

Japanese speech recognition from Fujie Laboratory (Chiba Institute of Technology)
that transcribes **fillers and repairs** along with the words.

`fujielab-asr` puts several recognizer families behind one `from_pretrained`
interface: two-layer Zipformer models that give phonemes and the written form
(streaming or offline, icefall-trained, run as TorchScript), a streaming Zipformer
run by sherpa-onnx, streaming Transducer / CTC models with a Contextual Block
Streaming (CBS) encoder built on [ESPnet](https://github.com/espnet/espnet) (the
`fujielab.asr.espnet_ext` layer), and Whisper large-v3 + LoRA. The streaming
models take audio chunk by chunk (e.g. every 100 ms), and most models run in real
time on the CPU.

## Model availability

The pretrained models listed in this README are **private** on the Hugging Face
Hub. The code in this package is public under the Apache License 2.0. To use the
models, please contact Shinya Fujie at **shinya.fujie@chibatech.ac.jp**. Note that
the terms of the training corpora limit providing the models outside the
laboratory (see *License* below).

The examples here assume use within the laboratory. `from_pretrained` downloads a
model from the Hub when you are logged in (`hf auth login` or `HF_TOKEN`) with an
account that can read it. It also accepts a local copy (the packed `.zip` for the
ESPnet and Whisper models, the model directory for the sherpa-onnx and Cascade
models).

## Features

Recognizers, in the order of *Pretrained models* below:

- **Two-layer, two-rate streaming Zipformer ("Cascade")** (`Speech2TextCascade`)
  — one audio stream gives a **phoneme** sequence with filler / repair /
  laughter events every 80 ms (layer 1) and the **written form** (kana-kanji)
  every 160 ms (layer 2), with no look-ahead. icefall-trained, run as
  TorchScript on the CPU in real time; layer-1 P-TER 7.80 % at 209 ms word
  latency, layer-2 CER 11.53 % at 345 ms on the CSJ+CEJC evaluation sets, with
  the low-latency model that adds ReazonSpeech (TV) pseudo labels to its
  training data. Needs the `cascade` extra: `pip install "fujielab-asr[cascade]"`.
- **Two-layer offline Zipformer** (`Speech2TextCascadeOffline`) — the most
  accurate models of the package, for transcribing recordings: both layers see
  the whole utterance, and the **written form itself carries the filler /
  repair marks** (`(F えー)昨日(D こ)今度の話`, CSJ notation). Layer-2 CER
  9.37 %, filler / repair F1 84.6 / 59.7 on the written form, layer-1 P-TER
  6.30 %; 0.03 × real time on 4 CPU threads. A larger variant (layer 1 =
  Zipformer large, 190 M parameters) lowers the layer-2 CER to 9.05 % at
  0.04 × real time. Same `cascade` extra.
- **Streaming Zipformer transducer via sherpa-onnx** (`Speech2TextSherpaOnnx`) —
  an icefall-trained 66 M model exported to ONNX and run by
  [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): **CPU real time on one
  thread** (RTF 0.08 fp32 / 0.05 int8 at 320 ms chunks), partial hypotheses
  chunk by chunk, no PyTorch at inference. CER 12.24 % on the CSJ+CEJC
  evaluation sets (CBS RNN-T: 17.02 %). Needs the `sherpa` extra:
  `pip install "fujielab-asr[sherpa]"`.
- **Transducer multitask** (`Speech2TextMultitaskTransducer`) — streaming
  Transducer beam search on a CBS Conformer encoder (ESPnet) that, in addition
  to the transcript, predicts a per-token **auxiliary-information label** via a
  separate head:
  `N` (normal), `F` (filler / フィラー), `D` (repair / 言い直し).
- **RNN-Transducer** (`Speech2Text`) — the same streaming beam search without
  the auxiliary head.
- **CTC-only multitask** (`Speech2TextMultitaskCTC`) — the same auxiliary-label
  idea on a streaming greedy-CTC recognizer.
- **CTC-only** (`Speech2TextCTC`) — streaming greedy CTC for models whose
  auxiliary markers are part of the recognized tokens (composite `あ+F` / `チ+D`,
  or span `<F> あ の </F>`).
- **Whisper large-v3 + LoRA** (`Speech2TextWhisper`) — Whisper adapted with LoRA
  to the same mora + marker output (`token_type whisper_mora`), with an optional
  **block-causal encoder** (100 ms chunks, bounded look-ahead) for streaming use.
  Utterance-level decoding today; a chunk-by-chunk API is planned. Needs the
  `whisper` extra: `pip install "fujielab-asr[whisper]"`.

The Transducer searches are used with `search_type="maes"`, which is what every
example here does. The `default` / `tsd` / `alsd` branches of
`BeamSearchTransducer` predate this package's streaming decoder API and do not
currently run.

## Installation

### Requirements

- **Python 3.10 – 3.12** (tested on 3.11).
- **ESPnet 202412 – 202511** — pulled in automatically as a dependency.
  (Older `fujielab-asr` (<= 0.1.3) targeted ESPnet 202301–202503; from 0.1.4 the
  package follows the newer ESPnet line, which uses `typeguard` 4.x. ESPnet 202609
  and later no longer ship the `espnet` package this extension imports.)
- `torch` / `torchaudio` (install a build matching your CUDA / platform). torchaudio
  loads only with the torch it was built for, and torchaudio 2.11 does not declare
  that properly (2.11.1 fails to load with torch 2.11 and 2.12), so the package keeps
  both below 2.11, where every torchaudio pins its torch exactly (pip then installs
  torch 2.10.0 + torchaudio 2.10.0). To pick a version yourself, pin both, e.g.
  `pip install fujielab-asr "torch==2.7.1" "torchaudio==2.7.1"`. Importing the
  ESPnet recognizers with a pair that does not load raises an `ImportError` that
  says so.

ESPnet has a few dependencies that build from source (e.g. `pyworld`); a working
C/C++ toolchain is recommended when installing.

### Install from PyPI

```bash
pip install fujielab-asr
```

We recommend a fresh virtual environment, e.g.:

```bash
python3.11 -m venv .venv && . .venv/bin/activate
pip install -U pip
pip install fujielab-asr
```

### Install from source

```bash
git clone https://github.com/fujielab/fujielab-asr
cd fujielab-asr
pip install -e .
```

## Pretrained models

Loaded by tag with `from_pretrained`; all are private (see *Model availability*
above). Listed in the order of *Example Usage* below.

1. **Two-layer streaming Zipformer ("Cascade")** — `Speech2TextCascade`: phonemes
   with filler / repair / laughter marks + the written form, CPU real time
   (`cascade` extra; [usage](#1-two-layer-streaming-zipformer-cascade))
   - `fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_stream_c80_c160_la0`
     — **recommended**: low latency (80 / 160 ms chunks, no look-ahead), CSJ+CEJC+ReazonSpeech
   - `fujielab/icefall_asr_csj_cejc_reazon_medium_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0`
     — 160 / 640 ms chunks, about half the compute, CSJ+CEJC+ReazonSpeech
   - `fujielab/icefall_asr_csj_cejc_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0`
     — CSJ+CEJC only; `…_fold1` / `…_fold2`: trained without the speakers of fold 1 / 2
   - `fujielab/icefall_asr_csj_cejc_phn_writ_comp_cascade_zipformer_stream_c160_c640_la0` — earlier version, without events
2. **Two-layer offline Zipformer** — `Speech2TextCascadeOffline`: for recordings;
   the written form itself carries the marks, `(F えー)`, `(D こ)`
   (`cascade` extra; [usage](#2-two-layer-offline-zipformer))
   - `fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_offline_writaux_proncond` — **default**
   - `fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_l_offline_writaux_proncond`
     — larger (190 M): a little more accurate, about 1.4 × the time
3. **Streaming Zipformer via sherpa-onnx** — `Speech2TextSherpaOnnx`: kana with
   composite markers, CPU real time on one thread, no PyTorch
   (`sherpa` extra; [usage](#3-streaming-zipformer-via-sherpa-onnx-cpu-real-time))
   - `fujielab/sherpa_onnx_asr_csj_cejc_pron_comp_zipformer_stream_c320_la0`
4. **Streaming Transducer (ESPnet, CBS)** — kana, CSJ+CEJC ([usage](#4-streaming-transducer-multitask-recognition--auxiliary-information))
   - `fujielab/espnet_asr_csj_cejc_pron_aux_cbs_transducer_120300_hop132`
     — `Speech2TextMultitaskTransducer`: separate `N`/`F`/`D` head
   - `fujielab/espnet_asr_csj_cejc_pron_comp_cbs_transducer_120300_hop132`
     — `Speech2Text`: composite tokens `あ+F` / `チ+D`
   - `fujielab/espnet_asr_csj_cejc_pron_span_cbs_transducer_120300_hop132`
     — `Speech2Text`: span tags `<F> あ </F>`
5. **Streaming CTC (ESPnet, CBS)** — kana ([usage](#5-streaming-ctc-multitask-recognition--auxiliary-information))
   - `fujielab/espnet_asr_csj_cejc_pron_aux_cbs_ctc_120300_hop132` — `Speech2TextMultitaskCTC`
   - `fujielab/espnet_asr_csj_cejc_pron_comp_cbs_ctc_120300_hop132`,
     `fujielab/espnet_asr_csj_cejc_pron_span_cbs_ctc_120300_hop132` — `Speech2TextCTC`
   - the same three trained on CSJ only: `fujielab/espnet_asr_csj_pron_{aux,comp,span}_cbs_ctc_120300_hop132`
6. **Whisper large-v3 + LoRA** — `Speech2TextWhisper`: kana with composite
   markers, block-causal encoder (100 ms chunks), GPU (`whisper` extra; [usage](#6-whisper-models))
   - `fujielab/espnet_asr_csj_cejc_pron_comp_whisper_large_v3_lora_stream_c100_la0`
7. **Older streaming Transducer models (ESPnet, CBS)** — `Speech2Text`
   ([usage](#7-older-streaming-transducer-models))
   - `fujielab/espnet_asr_cejc_pron_aux_cbs_transducer_081616_hop132` — CEJC, kana
   - `fujielab/espnet_asr_csj_writ_aux_cbs_transducer_081616_hop132` — CSJ, kanji
   - `fujielab/espnet_asr_cbs_transducer_120303_hop132_cc0105` — CEJC+CSJ, kana

## Example Usage

One subsection per model family, in the order above: install, a minimal example,
the runnable script in `examples/`, and results. `examples/aps-smp.mp3` (35 s of
CSJ sample audio) can be used as input.

### 1. Two-layer streaming Zipformer ("Cascade")

`Speech2TextCascade` runs the two-layer, two-rate recognizers of the
pron-based-asr project, trained with [icefall](https://github.com/k2-fsa/icefall)
on CSJ + CEJC (the recommended models also on ReazonSpeech with pseudo labels)
and exported as one TorchScript module (84.5 M parameters, 345 MB).
From a single audio stream,

- **layer 1** (causal Zipformer transducer on fbank; 80 ms chunks on the
  low-latency model, 160 ms on the others) emits
  **phonemes** in an IPA-based notation (`ː` long vowel, `N` moraic nasal, `Q`
  geminate, `ʲ` palatalization, `|` word boundary, `<sp>` pause) with events:
  `a+F` phoneme in a filler, `a+D` in a repair / word fragment and, on the
  Event-DP model, `<laugh>` and `<L>` … `</L>` (speech while laughing). It is fast
  enough to drive turn-taking and backchannel decisions in spoken dialogue;
- **layer 2** (a smaller causal Zipformer on the layer-1 encoder output; 160 ms
  chunks on the low-latency model, 640 ms on the others) emits the **written
  form** (kana-kanji), about 0.15 s after layer 1 (0.3–0.6 s with 640 ms chunks).

Only `torch`, `kaldi-native-fbank`, `numpy` and `huggingface_hub` are needed;
ESPnet and icefall are not imported.

```bash
pip install "fujielab-asr[cascade]"
```

```python
import soundfile as sf
from fujielab.asr.icefall_ext import Speech2TextCascade

s2t = Speech2TextCascade.from_pretrained(
    "fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_stream_c80_c160_la0",
    num_threads=4,
)

audio, fs = sf.read("utterance.wav", dtype="float32")   # 16 kHz mono
step, n, result = int(fs * 0.1), len(audio), None        # 100 ms per call
for i in range(0, n, step):
    result = s2t.streaming_decode(audio[i:i + step], is_final=i + step >= n)[0]
    # result.kana / result.text are the hypotheses so far
print(result.kana)    # "エ、パラゲンゴジョーホート ユー コトナ ンデスガ ア カンタンニ ..."  (layer 1)
print(result.text)    # "えパラ言語情報ということなんですがあ簡単に ..."               (layer 2)
print(" ".join(f"{t}[{a}]" if a != "N" else t
               for t, a in zip(result.base_tokens, result.aux_labels)))   # "e[F] <sp> p a r a ..."
```

`result.tokens` / `aux_labels` / `base_tokens` / `timestamps` / `confidences` are
the layer-1 symbols aligned 1:1 (as in the other recognizers of this package);
`result.graphemes` / `grapheme_timestamps` hold layer 2. For incremental use,
`s2t.accept(chunk)` and `s2t.flush()` return only the new symbols of each layer as
`(frame, symbol, confidence)` with 40 ms frames; call `s2t.reset()` before the next
utterance. Loading turns off TorchScript's profiling executor for the process
(otherwise the first decoder call fails); eager-mode ESPnet models are unaffected.
All Cascade models have the same interface and output symbols (the earlier
model without events only lacks the laughter symbols), and the chunk sizes are
read from each model's `config.json`, so they are interchangeable. The
low-latency model needs about twice the compute of the 160 / 640 ms models.
`from_pretrained()` without a tag loads the CSJ+CEJC-only Cascade+Event-DP
model (`fujielab/icefall_asr_csj_cejc_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0`).

**Live demo.** `examples/run_cascade_live.py` shows layer 1 (as katakana) and
layer 2 side by side while you speak, on a laptop CPU:

```bash
pip install "fujielab-asr[cascade]" sounddevice soxr rich
python examples/run_cascade_live.py                         # microphone
python examples/run_cascade_live.py --wav examples/aps-smp.mp3   # a file, in real time (--fast: full speed)
python examples/run_cascade_live.py --bench                 # real-time factor, per layer
```

The demo loads the low-latency model by default. If the status line says
processing is falling behind, `--model fujielab/icefall_asr_csj_cejc_reazon_medium_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0`
gives the same accuracy at about half the compute; `--model` takes any other tag
or a local model directory.

Results on the 17 CSJ + CEJC evaluation sets (23,259 utterances; greedy search).
P-TER counts every layer-1 symbol; CER is on the written form. Word latency
(median) is the time at which a word's last symbol can be emitted in real
streaming minus the word end in the CSJ time annotation (8 CSJ core lectures):

| Model | Chunks (L1 / L2) | Layer-1 P-TER % | Layer-1 latency | Layer-2 CER % | Layer-2 latency | Laughter F1 | Filler F1 | Repair F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Low-latency Cascade+Event-DP + ReazonSpeech (`…_reazon_large_…_c80_c160_la0`) | 80 / 160 ms | **7.80** | **209 ms** | **11.53** | **345 ms** | 80.8 | 74.1 | 48.7 |
| Cascade+Event-DP + ReazonSpeech (`…_reazon_medium_…_c160_c640_la0`) | 160 / 640 ms | 7.81 | 279 ms | 11.55 | 583 ms | **81.6** | **74.4** | **49.5** |
| Cascade+Event-DP, CSJ+CEJC only (`…_csj_cejc_phn_…_ev_cascade_…`) | 160 / 640 ms | 8.05 | 275 ms | 12.30 | 562 ms | 80.5 | 72.5 | 46.3 |
| Cascade (`…_cascade_…`) | 160 / 640 ms | 7.97 | 356 ms | 12.31 | 589 ms | — | — | — |
| Single-layer grapheme Zipformer | 160 ms | — | — | 13.53 | | — | — | — |

The 160 / 640 ms models also read 130 ms of padding beyond each chunk; the
low-latency model reads none. It has the same accuracy as the 160 / 640 ms
ReazonSpeech model (layer-2 CER −0.03 [−0.16, +0.10], P-TER −0.01 [−0.10, +0.07],
paired bootstrap) with words arriving 70 ms (layer 1) and 238 ms (layer 2)
earlier. Adding ReazonSpeech lowers the layer-2 CER by 0.74 points on CSJ+CEJC
([−0.88, −0.63]) and by much more outside them (layer-2 CER %):

| Model | JSUT basic5000 (read) | TEDxJP-10K (lectures) | LaboroTV dev (TV) | ReazonSpeech test (TV) |
|---|---:|---:|---:|---:|
| Low-latency Cascade+Event-DP + ReazonSpeech (large) | **9.54** | 13.79 | **15.44** | **9.71** |
| Cascade+Event-DP + ReazonSpeech (medium) | 10.01 | **13.31** | 15.77 | 10.34 |
| Cascade+Event-DP, CSJ+CEJC only | 12.00 | 16.04 | 23.38 | 22.14 |

JSUT is scored with numbers in kanji (as in CSJ / CEJC); ReazonSpeech test is
2,000 held-out utterances scored against the captions, fillers removed from the
hypothesis. ReazonSpeech is distributed by Reazon Holdings under its own terms
(CDLA-Sharing-1.0, use limited to Article 30-4 of the Japanese Copyright Act);
the CSJ+CEJC-only model is the alternative where that matters.

The streaming implementation reproduces the evaluation decode (Cascade+Event-DP,
150 utterances: P-TER 6.36 vs 6.35, CER 7.66 vs 7.74; differences are mostly the
last 1–2 characters of an utterance) and runs at RTF 0.12 with 4 threads (0.16
with 1) on an i9-12900K; the low-latency model at 0.22 (0.27 with 1). See the
model cards for training details.

### 2. Two-layer offline Zipformer

`Speech2TextCascadeOffline` runs the offline model of the same project
(`fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_offline_writaux_proncond`,
84.1 M parameters, 344 MB), for transcribing recordings rather than live audio.
Both layers see the **whole utterance**, which matters most for repairs (whether a
word was abandoned is clear only from what follows). Compared with the streaming
models,

- **layer 2 marks fillers and repairs itself** (`え+F`, `こ+D`), so the written
  form comes out in CSJ notation, `(F えー)昨日(D こ)今度の話`, and laughter as
  `<笑>` / `(笑 …)`;
- **layer 2 also reads what layer 1 decided**: the phoneme / mark symbols that the
  layer-1 greedy search emitted at each frame are fed into the layer-2 encoder,
  next to the layer-1 encoder output.

```python
import soundfile as sf
from fujielab.asr.icefall_ext import Speech2TextCascadeOffline

s2t = Speech2TextCascadeOffline.from_pretrained(num_threads=4)   # the model above
audio, fs = sf.read("utterance.wav", dtype="float32")             # 16 kHz mono, one utterance
r = s2t(audio)[0]
print(r.tagged_text)  # "(F まあのー)こうやって(D あ)話しておりますとそれはもちろん(F あの)言語的情報を ..."
print(r.text)         # "まあのーこうやってあ話しておりますとそれはもちろんあの言語的情報を ..."
print(r.kana)         # "マアノー コー ヤッテ ハ ハナシテ オリマスト ..."   (layer 1)
```

The result is the same `CascadeResult` as for the streaming models, with
`base_graphemes` / `grapheme_aux_labels` splitting the marked layer-2 characters
like `base_tokens` / `aux_labels` do for layer 1; timestamps are seconds from the
start of the utterance. `s2t.decode_fbank(feats)` takes precomputed 80-dim fbank
(e.g. lhotse features) instead of audio. **Cut long recordings at pauses first**:
the model was trained on utterances of at most 30 s (99 % within 17.7 s). The
example script does this with a simple energy-based splitter:

```bash
pip install "fujielab-asr[cascade]"
python examples/run_cascade_offline.py examples/aps-smp.mp3              # per segment: layer 1 and layer 2
python examples/run_cascade_offline.py a.wav --out result.jsonl          # also every symbol with its time
```

**Larger variant.** `fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_l_offline_writaux_proncond`
has the same files, interface and search; its layer 1 is a Zipformer large
(encoder outputs 768 / 512 dims instead of 512 / 384; 190.0 M parameters, 768 MB).
It is a little more accurate everywhere (table below) and takes about 1.4 × the
time. Pass its tag to `from_pretrained` (or `--model` to the script):

```python
s2t = Speech2TextCascadeOffline.from_pretrained(
    "fujielab/icefall_asr_csj_cejc_reazon_large_phn_writ_comp_ev_cascade_zipformer_l_offline_writaux_proncond", num_threads=4)
```

Results on the 17 CSJ + CEJC evaluation sets (greedy search). Here filler / repair
F1 is scored on the **written form** (characters aligned by edit distance, marks
compared), unlike the layer-1 event F1 of the streaming table above; for models
whose layer 2 has no marks, they were moved over from layer 1 by time alignment:

| Model | Layer-2 CER % | CER with marks % | Filler F1 | Repair F1 | Layer-1 P-TER % | JSUT / TEDxJP / LaboroTV (CER %) |
|---|---:|---:|---:|---:|---:|---|
| **Offline Cascade, marked layer 2 + layer-1 input** (default model) | 9.37 | 10.02 | **84.6** | 59.7 | 6.30 | 7.90 / 10.37 / 13.71 |
| Offline Cascade large (`…_zipformer_l_offline_writaux_proncond`), marked layer 2 + layer-1 input | **9.05** | **9.70** | **84.6** | **59.8** | **6.13** | **7.50** / **10.32** / **13.34** |
| Offline Cascade, plain layer 2, same data and schedule (not released) | 9.33 | 10.49 | 81.8 | 54.0 | 6.36 | 8.04 / 10.42 / 13.61 |
| Streaming low-latency Cascade (`…_c80_c160_la0`) | 11.53 | 12.91 | 79.7 | 44.1 | 7.80 | 9.54 / 13.79 / 15.44 |

The marks in layer 2 cost nothing in CER (+0.04 against the plain layer 2, not
significant) and raise the filler / repair F1 by 2.8 / 5.7 points. The
implementation reproduces the evaluation decode: on 10 evaluation utterances its
symbols and frame times in both layers are identical to icefall's from the same
fbank, and from audio (kaldi-native-fbank) the symbols are identical too (checked
with torch 2.6, 2.7, 2.10 and 2.12; the large variant passes the same check with
torch 2.10). It runs
at RTF 0.028 with 4 threads (0.052 with 1) on an i9-12900K; the large variant at
0.041 (0.079 with 1). See the model card for training details.

### 3. Streaming Zipformer via sherpa-onnx (CPU real time)

`Speech2TextSherpaOnnx` runs the icefall-trained streaming Zipformer transducer
(`fujielab/sherpa_onnx_asr_csj_cejc_pron_comp_zipformer_stream_c320_la0`) through
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx). Only the ONNX files for the
requested chunk size / precision are downloaded (fp32 encoder 261 MB, int8 70 MB);
`torch` and ESPnet are not imported. The output tokens are the same composite
encoding as the CBS `comp` models (`ア+F`, `ン+D`, `|`, `<sp>`, `<mask>`), and the
result exposes `tokens` / `aux_labels` / `base_tokens` aligned 1:1 like the
multitask recognizers.

```bash
pip install "fujielab-asr[sherpa]"
```

```python
import soundfile as sf
from fujielab.asr.sherpa_onnx_ext import Speech2TextSherpaOnnx

s2t = Speech2TextSherpaOnnx.from_pretrained(
    "fujielab/sherpa_onnx_asr_csj_cejc_pron_comp_zipformer_stream_c320_la0",
    chunk_ms=320,          # 320 (default) or 640
    int8=False,            # True: int8 encoder, ~1.8x faster
    num_threads=1,
)

audio, fs = sf.read("utterance.wav", dtype="float32")   # 16 kHz mono
step, n, result = int(fs * 0.1), len(audio), None        # 100 ms per call
for i in range(0, n, step):
    result = s2t.streaming_decode(audio[i:i + step], is_final=i + step >= n)[0]
    # result.text is the partial hypothesis so far
print(result.text)        # "ア+F ノ+F ー+F | ス+D | イ ッ パ ツ メ カ ラ ー"
print(result.transcript)  # "アノースイッパツメカラー"
print(" ".join(f"{t}[{a}]" if a != "N" else t
               for t, a in zip(result.base_tokens, result.aux_labels)))
```

`is_final=True` appends 0.6 s of silence and flushes the last tokens (the model was
trained with trailing silence, so this is also what suppresses spurious tokens at
utterance ends); the next call starts a new utterance. For endpoint-driven use pass
`enable_endpoint_detection=True` and poll `s2t.is_endpoint()` / `s2t.finish_segment()`.

Results on the 17 CSJ+CEJC evaluation sets (CER counts every token, `+F`/`+D`
markers included, `<sp>`/`<mask>` as one symbol; RTF on one CPU thread with 100 ms
input):

| Model | Chunk | Params | CER % | F F1 | D F1 | RTF |
|---|---|---:|---:|---:|---:|---:|
| Zipformer, greedy | 320 ms | 66 M | **12.24** | 84.7 | 54.4 | 0.084 (int8 0.046) |
| Zipformer, modified beam search 4 | 320 ms | 66 M | 11.82 | 85.1 | 55.3 | |
| Zipformer, greedy | 640 ms | 66 M | 12.37 | 84.8 | 55.2 | 0.051 (int8 0.029) |
| CBS Conformer RNN-T `comp` (this package, beam 20) | 100 ms | 30 M | 17.02 | 82.0 | 45.5 | 0.275 |
| Whisper large-v3 + LoRA, block-causal, look-ahead 0 | 100 ms | 1.55 B | 12.25 | 85.3 | 57.6 | GPU |

Against the CBS RNN-T the Zipformer is 4.8 CER points better (paired bootstrap
[−4.94, −4.64]) and 3.3× faster; against the streaming Whisper it is on par in CER
(−0.01, not significant) at 1/23 of the parameters.

Script: `examples/run_streaming_asr_sherpa.py` (partial hypotheses every 100 ms).

### 4. Streaming Transducer multitask (recognition + auxiliary information)

```python
import numpy as np, soundfile as sf
from fujielab.asr.espnet_ext.espnet2.bin.asr_multitask_transducer_inference_cbs import (
    Speech2TextMultitaskTransducer,
)

s2t = Speech2TextMultitaskTransducer.from_pretrained(
    "fujielab/espnet_asr_csj_cejc_pron_aux_cbs_transducer_120300_hop132",
    beam_size=20, beam_search_config=dict(search_type="maes"),
)

audio, fs = sf.read("utterance.wav")   # 16 kHz mono
chunk = int(16000 * 0.1)               # 100 ms
n, result = len(audio), None
for i in range(0, n, chunk):
    c = audio[i:i + chunk]
    is_final = i + chunk >= n          # flag the last chunk even when it is full
    if len(c) < chunk:
        c = np.pad(c, (0, chunk - len(c)))
    r = s2t.streaming_decode(c, is_final=is_final)
    if r:
        result = r[0]

# result.tokens and result.aux_labels are aligned 1:1 (aux in {N, F, D})
print(" ".join(f"{t}[{a}]" if a != "N" else t
               for t, a in zip(result.tokens, result.aux_labels)))
```

The auxiliary head lives on the joint network's hidden state, so a label only
exists at a specific (frame, predictor-state) lattice node. The offline research
recipe locates that node by re-encoding the utterance and forced-aligning the
final hypothesis — a second pass over complete audio. This package instead reads
the head at the node the beam search emitted the token at, which needs no second
pass and so yields auxiliary labels on every chunk. On 297 held-out CEJC
utterances the two protocols agree on **99.5 %** of tokens (**90.2 %** of tokens
where either says `F` or `D`).

The `comp` / `span` Transducer models put the markers into the tokens and use the
plain `Speech2Text` (`fujielab.asr.espnet_ext.espnet2.bin.asr_transducer_inference_cbs`)
with the same 100 ms loop (see 7 for its usage). Scripts:
`examples/run_streaming_asr_multitask_transducer.py` (`aux`) and
`examples/bench_streaming_rtf.py` (CPU real-time factor).

### 5. Streaming CTC multitask (recognition + auxiliary information)

```python
import numpy as np, soundfile as sf
from fujielab.asr.espnet_ext.espnet2.bin.asr_multitask_ctc_inference_cbs import (
    Speech2TextMultitaskCTC,
)

s2t = Speech2TextMultitaskCTC.from_pretrained(
    "fujielab/espnet_asr_csj_pron_aux_cbs_ctc_120300_hop132", streaming=True
)

audio, fs = sf.read("utterance.wav")  # 16 kHz mono
chunk = int(16000 * 0.1)              # 100 ms
for i in range(0, len(audio), chunk):
    c = audio[i:i + chunk]
    is_final = len(c) < chunk
    if is_final:
        c = np.pad(c, (0, chunk - len(c)))
    r = s2t.streaming_decode(c, is_final=is_final)[0]
    # r.tokens and r.aux_labels are aligned 1:1 (aux in {N, F, D})
    print(" ".join(f"{t}[{a}]" if a != "N" else t
                   for t, a in zip(r.tokens, r.aux_labels)))
```

Scripts: `examples/run_streaming_asr_multitask.py` (`aux`, `Speech2TextMultitaskCTC`)
and `examples/run_streaming_asr_ctc.py` (`comp` / `span`, `Speech2TextCTC`).

**Choosing among the ESPnet models (4 and 5).** The six CSJ+CEJC ESPnet models
(three marker encodings × two heads) share the encoder, data and schedule. CER is measured with all markers stripped from both
sides:

| Encoding | RNN-T CER % | CTC CER % | RNN-T F/D F1 | CTC F/D F1 |
|---|---:|---:|---:|---:|
| `aux` | **16.01** | 16.90 | 81.9 / 44.4 | 81.6 / 43.1 |
| `comp` | 16.25 | 17.64 | **82.0** / **45.5** | 80.4 / 44.3 |
| `span` | 16.41 | **16.74** | 81.5 / 43.9 | 81.3 / 41.6 |

- **Head**: RNN-T beats CTC on every encoding (+0.33 to +1.39 points, all
  significant over 23,500 utterances). Pick CTC for a smaller, decoder-free model.
- **Encoding**: the gaps are within run-to-run variance; do not read a ranking
  into them. The filler / repair F1 hardly depends on either choice (`D` recall is
  34–38 % everywhere).
- Span tags may come out unbalanced (CTC 11.1 %, RNN-T 2.8 % of utterances):
  read `<F>`/`<D>` as *set the label* and `</F>`/`</D>` as *clear it*.
- The CSJ-only CTC trio: content CER ≈6.7–6.8 %, filler F1 ≈93.

### 6. Whisper models

`Speech2TextWhisper` loads the Whisper large-v3 + LoRA models. Only the adapter
(LoRA matrices and added-token embeddings, ~70 MB) is downloaded from the Hub;
the Whisper base weights (3 GB) are fetched by `openai-whisper` on first use.

```bash
pip install "fujielab-asr[whisper]"
python examples/run_whisper_asr.py utterance.wav --device cuda
```

```python
import soundfile as sf
from fujielab.asr.espnet_ext.espnet2.bin.asr_whisper_inference import Speech2TextWhisper

s2t = Speech2TextWhisper.from_pretrained(
    "fujielab/espnet_asr_csj_cejc_pron_comp_whisper_large_v3_lora_stream_c100_la0", device="cuda")
audio, fs = sf.read("utterance.wav")      # 16 kHz mono, <= 30 s
print(s2t(audio)[0].text)                 # "ア+F ノ+F ー+F | ス+D | イ ッ パ ツ メ ..."
```

On the 17 CSJ+CEJC evaluation sets the block-causal model (look-ahead 0) reaches
CER 11.82 % (CBS RNN-T comp: 16.25 %) with F/D F1 85.3 / 57.6; the full-context
version reaches 9.52 %. See the model card for details.

### 7. Older streaming Transducer models

The earlier models use the plain streaming `Speech2Text`; the `comp` / `span`
models of 4 load the same way.

```python
from fujielab.asr.espnet_ext.espnet2.bin.asr_transducer_inference_cbs import Speech2Text

s2t = Speech2Text.from_pretrained(
    "fujielab/espnet_asr_cejc_pron_aux_cbs_transducer_081616_hop132",
    streaming=True, lm_weight=0.0, beam_size=20, beam_search_config=dict(search_type="maes"),
)
# feed 100 ms chunks as in 4; the tokens of each result:
#   s2t.hypotheses_to_results(s2t.streaming_decode(chunk, is_final=is_final))[0][1]
```

Scripts: `examples/run_streaming_asr.py` (a file), `examples/run_streaming_asr_live.py`
(microphone, with Silero VAD) and `examples/demo.py` (Gradio, utterance-level).

## License

**Code.** The code of this package is under the Apache License 2.0; see
[LICENSE](LICENSE). The code license does not cover the pretrained models.

**Pretrained models and their training data.** The models were trained on the
corpora below, and their terms carry over to the models:

- **CSJ** (Corpus of Spontaneous Japanese, National Institute for Japanese Language
  and Linguistics, NINJAL) — models with `csj` in the tag, and `…_cc0105`.
  Licensed under NINJAL's non-commercial license agreement: commercial use is
  prohibited, and the current agreement treats AI models trained on the corpus as
  derivatives, which may be used only internally by the licensee and not provided
  to third parties.
- **CEJC** (Corpus of Everyday Japanese Conversation, NINJAL) — models with `cejc`
  in the tag, and `…_cc0105`. Also under NINJAL's non-commercial license agreement:
  commercial use is prohibited, and derivatives (data, databases and programs) may
  be used only internally and not provided to third parties. The corpus contains
  personal information of its speakers, which must not be disclosed.
- **ReazonSpeech** (Reazon Holdings) — models with `reazon` in the tag, where it
  supplies pseudo-labelled training data. Distributed under CDLA-Sharing-1.0, with
  use limited to the purposes of Article 30-4 of the Japanese Copyright Act.

Every model includes CSJ and / or CEJC in its training data, so the CSJ / CEJC
terms apply to all of them: the models are not for commercial use, and providing
them to anyone outside the laboratory, even for research, may not be possible
under these terms. When publishing results obtained with the models, please cite
this repository and the corpora used.
