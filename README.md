# fujielab-asr

Automatic Speech Recognition (ASR) modules for Fujie Laboratory, built on top of
[ESPnet](https://github.com/espnet/espnet).

`fujielab-asr` packages streaming-ASR extensions (the `fujielab.asr.espnet_ext`
layer) on top of stock ESPnet, together with pretrained models on the Hugging Face
Hub. It is designed for **online / chunk-by-chunk** recognition.

## Model availability

The pretrained models listed in this README are **private** on the Hugging Face
Hub. They were public up to `fujielab-asr` 0.4.0 under `fujie/...`; they now live,
private, under `fujielab/...`. The code in this package stays public under the
Apache License 2.0. To use the models, please contact Shinya Fujie at
**shinya.fujie@chibatech.ac.jp**.

Once access has been arranged, the examples here work unchanged: `from_pretrained`
downloads a model from the Hub when you are logged in (`hf auth login` or
`HF_TOKEN`) with an account that can read it, and it also accepts a local copy (the
packed `.zip` for the ESPnet and Whisper models, the model directory for the
sherpa-onnx and Cascade models).

学習済みモデルは Hugging Face Hub 上で非公開（private）にしました。モデルの利用を
希望される方は、藤江（shinya.fujie@chibatech.ac.jp）までご連絡ください。

## Features

- **Streaming ASR** with a Contextual Block Streaming (CBS) encoder.
- Recognizer families:
  - **RNN-Transducer** (`Speech2Text`) — streaming beam search.
  - **Transducer multitask** (`Speech2TextMultitaskTransducer`) — streaming
    Transducer beam search that, in addition to the transcript, predicts a
    per-token **auxiliary-information label** via a separate head:
    `N` (normal), `F` (filler / フィラー), `D` (repair / 言い直し).
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
  - **Streaming Zipformer transducer via sherpa-onnx** (`Speech2TextSherpaOnnx`) —
    an icefall-trained 66 M model exported to ONNX and run by
    [sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx): **CPU real time on one
    thread** (RTF 0.08 fp32 / 0.05 int8 at 320 ms chunks), partial hypotheses
    chunk by chunk, no PyTorch at inference. CER 12.24 % on the CSJ+CEJC
    evaluation sets (CBS RNN-T: 17.02 %). Needs the `sherpa` extra:
    `pip install "fujielab-asr[sherpa]"`.
  - **Two-layer, two-rate streaming Zipformer ("Cascade")** (`Speech2TextCascade`)
    — one audio stream gives a **phoneme** sequence with filler / repair /
    laughter events every 160 ms (layer 1) and the **written form** (kana-kanji)
    every 640 ms (layer 2). icefall-trained, run as TorchScript on the CPU in real
    time; layer-1 P-TER 7.81 % at 279 ms word latency, layer-2 CER 11.55 % at
    583 ms on the CSJ+CEJC evaluation sets, with the model that adds
    ReazonSpeech (TV) pseudo labels to its training data. Needs the `cascade`
    extra: `pip install "fujielab-asr[cascade]"`.
- Pretrained Japanese models hosted on the Hugging Face Hub (loaded with
  `from_pretrained`; private, available on request, see *Model availability*).

The Transducer searches are used with `search_type="maes"`, which is what every
example here does. The `default` / `tsd` / `alsd` branches of
`BeamSearchTransducer` predate this package's streaming decoder API and do not
currently run.

## Installation

### Requirements

- **Python 3.10 – 3.12** (tested on 3.11).
- **ESPnet >= 202412** — pulled in automatically as a dependency.
  (Older `fujielab-asr` (<= 0.1.3) targeted ESPnet 202301–202503; from 0.1.4 the
  package follows the newer ESPnet line, which uses `typeguard` 4.x.)
- `torch` / `torchaudio` (install a build matching your CUDA / platform).

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

Loaded by tag via `from_pretrained`. (The auxiliary-information models additionally
emit `F`/`D` markers for fillers and repairs.) All of these repositories are
private; see *Model availability* above for access (shinya.fujie@chibatech.ac.jp).

| Tag | Type | Tokens | Corpus |
|-----|------|--------|--------|
| `fujielab/espnet_asr_csj_cejc_pron_aux_cbs_transducer_120300_hop132` | **Transducer multitask (N/F/D head)** | kana | CSJ+CEJC |
| `fujielab/espnet_asr_csj_cejc_pron_comp_cbs_transducer_120300_hop132` | **Transducer composite (`あ+F`/`チ+D`)** | kana | CSJ+CEJC |
| `fujielab/espnet_asr_csj_cejc_pron_span_cbs_transducer_120300_hop132` | **Transducer span (`<F> あ </F>`)** | kana | CSJ+CEJC |
| `fujielab/espnet_asr_csj_cejc_pron_aux_cbs_ctc_120300_hop132` | **CTC multitask (N/F/D head)** | kana | CSJ+CEJC |
| `fujielab/espnet_asr_csj_cejc_pron_comp_cbs_ctc_120300_hop132` | **CTC composite (`あ+F`/`チ+D`)** | kana | CSJ+CEJC |
| `fujielab/espnet_asr_csj_cejc_pron_span_cbs_ctc_120300_hop132` | **CTC span (`<F> あ </F>`)** | kana | CSJ+CEJC |
| `fujielab/espnet_asr_csj_pron_aux_cbs_ctc_120300_hop132` | CTC multitask (N/F/D head) | kana | CSJ |
| `fujielab/espnet_asr_csj_cejc_pron_comp_whisper_large_v3_lora_stream_c100_la0` | **Whisper large-v3 + LoRA, block-causal encoder (100 ms chunks, look-ahead 0), composite (`あ+F`/`チ+D`)** | kana | CSJ+CEJC |
| `fujielab/sherpa_onnx_asr_csj_cejc_pron_comp_zipformer_stream_c320_la0` | **Streaming Zipformer transducer (icefall → ONNX, sherpa-onnx), 320 / 640 ms chunks, composite (`あ+F`/`チ+D`), CPU real time** | kana | CSJ+CEJC |
| `fujielab/icefall_asr_csj_cejc_reazon_medium_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0` | **Two-layer Cascade+Event-DP (icefall → TorchScript): phonemes with filler / repair / laughter events every 160 ms + written form every 640 ms, CPU real time; ReazonSpeech medium (TV, pseudo labels) added to the training data** | phoneme + kanji | CSJ+CEJC+ReazonSpeech |
| `fujielab/icefall_asr_csj_cejc_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0` | Two-layer Cascade+Event-DP, the same trained on CSJ+CEJC only | phoneme + kanji | CSJ+CEJC |
| `…_fold1`, `…_fold2` (suffixes of the tag above) | the same as the row above, trained without the speakers of fold 1 / 2 (for processing data of speakers the recognizer has not seen) | phoneme + kanji | CSJ+CEJC |
| `fujielab/icefall_asr_csj_cejc_phn_writ_comp_cascade_zipformer_stream_c160_c640_la0` | Two-layer Cascade, no events and no delay penalty (earlier version) | phoneme + kanji | CSJ+CEJC |
| `fujielab/espnet_asr_csj_pron_comp_cbs_ctc_120300_hop132` | CTC composite | kana | CSJ |
| `fujielab/espnet_asr_csj_pron_span_cbs_ctc_120300_hop132` | CTC span | kana | CSJ |
| `fujielab/espnet_asr_cejc_pron_aux_cbs_transducer_081616_hop132` | Transducer | kana | CEJC |
| `fujielab/espnet_asr_csj_writ_aux_cbs_transducer_081616_hop132` | Transducer | kanji | CSJ |
| `fujielab/espnet_asr_cbs_transducer_120303_hop132_cc0105` | Transducer | kana | CEJC+CSJ |

### The six CSJ+CEJC models

Three encodings of the filler/repair information × two heads. All six share the
encoder, data, effective batch and schedule (50 epochs × 1400 steps), so both
axes are controlled comparisons.

CER is measured **with all auxiliary information stripped from both sides**, so
every model is scored against one target; charging `comp`/`span` for the markup
they must emit and `aux` for none made an earlier comparison look like a 2–3
point gap when the real gap is well under half a point.

| Encoding | RNN-T CER % | CTC CER % | RNN-T F/D F1 | CTC F/D F1 |
|---|---:|---:|---:|---:|
| `aux` | **16.01** | 16.90 | 81.9 / 44.4 | 81.6 / 43.1 |
| `comp` | 16.25 | 17.64 | **82.0** / **45.5** | 80.4 / 44.3 |
| `span` | 16.41 | **16.74** | 81.5 / 43.9 | 81.3 / 41.6 |

**Head**: RNN-T beats CTC on every encoding (+0.33 to +1.39 points, each
significant under a paired bootstrap over 23,500 utterances), but the margin
varies fourfold by encoding, so there is no single figure for it. Pick CTC for a
smaller, decoder-free, greedy-decoding model; pick RNN-T for the best transcript.

**Encoding**: the ordering flips between heads (RNN-T `aux` < `comp` < `span`;
CTC `span` < `aux` < `comp`), and the `comp` vs `span` gap also flips sign
between the 14- and 50-epoch RNN-T checkpoints. Run-to-run variance exceeds
these gaps — do not read a ranking into them.

**Auxiliary labels barely depend on either choice**: `F` F1 spans 80.4–82.0 and
`D` F1 spans 41.6–45.5 across all six. `D` recall is 34–38 % everywhere, the
weak point of the whole set, and not something a different head or encoding
fixes.

One practical caveat: greedy CTC has nothing tying an opening span tag to its
closing one, so the CTC `span` model returns malformed markup on **11.1 %** of
utterances (RNN-T `span`: 2.8 %). Parse `<F>`/`<D>` as *set current label* and
`</F>`/`</D>` as *clear it* rather than assuming balanced brackets.

The older CSJ-only **CTC** trio is the same three-way encoding comparison on
CSJ alone: content CER ≈6.7–6.8 %, filler F1 ≈93.

## Example Usage

Runnable scripts are in the `examples/` directory:

- `examples/run_streaming_asr.py` — streaming Transducer ASR.
- `examples/run_streaming_asr_multitask_transducer.py` — streaming **Transducer
  multitask** ASR (transcript + filler/repair labels).
- `examples/run_streaming_asr_ctc.py` — streaming **CTC** ASR (composite / span markers in tokens).
- `examples/run_streaming_asr_multitask.py` — streaming **CTC multitask** ASR
  (transcript + filler/repair labels).
- `examples/run_streaming_asr_sherpa.py` — streaming **Zipformer via sherpa-onnx**
  (CPU real time; partial hypotheses every 100 ms, filler/repair markers in tokens).
- `examples/run_cascade_live.py` — live (microphone or file) demo of the
  two-layer **Cascade** models: phonemes as katakana every 160 ms next to the
  written form every 640 ms, fillers / repairs / laughter in colour, on the CPU.
- `examples/run_streaming_asr_live.py` — live (microphone) streaming ASR.
- `examples/demo.py` — Gradio demo.
- `examples/bench_streaming_rtf.py` — CPU real-time-factor benchmark of the streaming Transducer models.

### Streaming Transducer multitask (recognition + auxiliary information)

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

### Streaming CTC multitask (recognition + auxiliary information)

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

## License

Apache License 2.0. See [LICENSE](LICENSE).

## Whisper models

`Speech2TextWhisper` loads the Whisper large-v3 + LoRA models. Only the adapter
(LoRA matrices and added-token embeddings, ~70 MB) is downloaded from the Hub;
the Whisper base weights (3 GB) are fetched by `openai-whisper` on first use.

```bash
pip install "fujielab-asr[whisper]"
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

## Streaming Zipformer via sherpa-onnx (CPU real time)

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

## Two-layer streaming Zipformer ("Cascade"): phonemes + written form

`Speech2TextCascade` runs the two-layer, two-rate recognizers of the
pron-based-asr project, trained with [icefall](https://github.com/k2-fsa/icefall)
on CSJ + CEJC (the recommended model also on ReazonSpeech medium with pseudo
labels) and exported as one TorchScript module (84.5 M parameters, 345 MB).
From a single audio stream,

- **layer 1** (causal Zipformer transducer on fbank, 160 ms chunks) emits
  **phonemes** in an IPA-based notation (`ː` long vowel, `N` moraic nasal, `Q`
  geminate, `ʲ` palatalization, `|` word boundary, `<sp>` pause) with events:
  `a+F` phoneme in a filler, `a+D` in a repair / word fragment and, on the
  Event-DP model, `<laugh>` and `<L>` … `</L>` (speech while laughing). It is fast
  enough to drive turn-taking and backchannel decisions in spoken dialogue;
- **layer 2** (a smaller causal Zipformer on the layer-1 encoder output, 640 ms
  chunks) emits the **written form** (kana-kanji), 0.3–0.6 s after layer 1.

Only `torch`, `kaldi-native-fbank`, `numpy` and `huggingface_hub` are needed;
ESPnet and icefall are not imported.

```bash
pip install "fujielab-asr[cascade]"
```

```python
import soundfile as sf
from fujielab.asr.icefall_ext import Speech2TextCascade

s2t = Speech2TextCascade.from_pretrained(
    "fujielab/icefall_asr_csj_cejc_reazon_medium_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0",
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
All Cascade models have the same streaming settings and interface (the earlier
model without events only lacks the laughter symbols), so they are
interchangeable; `from_pretrained()` without a tag loads the CSJ+CEJC-only
Cascade+Event-DP model (`fujielab/icefall_asr_csj_cejc_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0`).

**Live demo.** `examples/run_cascade_live.py` shows layer 1 (as katakana) and
layer 2 side by side while you speak, on a laptop CPU:

```bash
pip install "fujielab-asr[cascade]" sounddevice soxr rich
python examples/run_cascade_live.py                         # microphone
python examples/run_cascade_live.py --wav examples/aps-smp.mp3   # a file, in real time (--fast: full speed)
python examples/run_cascade_live.py --bench                 # real-time factor, per layer
```

The demo loads the ReazonSpeech model by default; `--model` takes any other tag
(e.g. `--model fujielab/icefall_asr_csj_cejc_phn_writ_comp_ev_cascade_zipformer_stream_c160_c640_la0`) or a local model directory.

Results on the 17 CSJ + CEJC evaluation sets (23,259 utterances; greedy search).
P-TER counts every layer-1 symbol; CER is on the written form. Word latency
(median) is the time at which a word's last symbol can be emitted in real
streaming minus the word end in the CSJ time annotation (8 CSJ core lectures):

| Model | Layer-1 P-TER % | Layer-1 latency | Layer-2 CER % | Layer-2 latency | Laughter F1 | Filler F1 | Repair F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cascade+Event-DP + ReazonSpeech (`…_reazon_medium_…_ev_cascade_…`) | **7.81** | 279 ms | **11.55** | 583 ms | **81.6** | **74.4** | **49.5** |
| Cascade+Event-DP, CSJ+CEJC only (`…_csj_cejc_phn_…_ev_cascade_…`) | 8.05 | **275 ms** | 12.30 | **562 ms** | 80.5 | 72.5 | 46.3 |
| Cascade (`…_cascade_…`) | 7.97 | 356 ms | 12.31 | 589 ms | — | — | — |
| Single-layer grapheme Zipformer, 160 ms chunks | — | — | 13.53 | | — | — | — |

Adding ReazonSpeech lowers the layer-2 CER by 0.74 points on CSJ+CEJC (paired
bootstrap [−0.88, −0.63]) and by much more outside them (layer-2 CER %):

| Model | JSUT basic5000 (read) | TEDxJP-10K (lectures) | LaboroTV dev (TV) | ReazonSpeech test (TV) |
|---|---:|---:|---:|---:|
| Cascade+Event-DP + ReazonSpeech | **10.01** | **13.31** | **15.77** | **10.34** |
| Cascade+Event-DP, CSJ+CEJC only | 12.00 | 16.04 | 23.38 | 22.14 |

JSUT is scored with numbers in kanji (as in CSJ / CEJC); ReazonSpeech test is
2,000 held-out utterances scored against the captions, fillers removed from the
hypothesis. ReazonSpeech is distributed by Reazon Holdings under its own terms
(CDLA-Sharing-1.0, use limited to Article 30-4 of the Japanese Copyright Act);
the CSJ+CEJC-only model is the alternative where that matters.

The streaming implementation reproduces the evaluation decode (Cascade+Event-DP,
150 utterances: P-TER 6.36 vs 6.35, CER 7.66 vs 7.74; differences are mostly the
last 1–2 characters of an utterance) and runs at RTF 0.12 with 4 threads (0.16
with 1) on an i9-12900K. See the model cards for training details.
