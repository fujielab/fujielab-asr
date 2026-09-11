# fujielab-asr

Automatic Speech Recognition (ASR) modules for Fujie Laboratory, built on top of
[ESPnet](https://github.com/espnet/espnet).

`fujielab-asr` packages streaming-ASR extensions (the `fujielab.asr.espnet_ext`
layer) on top of stock ESPnet, together with ready-to-use pretrained models on the
Hugging Face Hub. It is designed for **online / chunk-by-chunk** recognition.

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
- Pretrained Japanese models distributed via the Hugging Face Hub
  (loaded with `from_pretrained`).

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
emit `F`/`D` markers for fillers and repairs.)

| Tag | Type | Tokens | Corpus |
|-----|------|--------|--------|
| `fujie/espnet_asr_csj_cejc_pron_aux_cbs_transducer_120300_hop132` | **Transducer multitask (N/F/D head)** | kana | CSJ+CEJC |
| `fujie/espnet_asr_csj_cejc_pron_comp_cbs_transducer_120300_hop132` | **Transducer composite (`あ+F`/`チ+D`)** | kana | CSJ+CEJC |
| `fujie/espnet_asr_csj_cejc_pron_span_cbs_transducer_120300_hop132` | **Transducer span (`<F> あ </F>`)** | kana | CSJ+CEJC |
| `fujie/espnet_asr_csj_cejc_pron_aux_cbs_ctc_120300_hop132` | **CTC multitask (N/F/D head)** | kana | CSJ+CEJC |
| `fujie/espnet_asr_csj_cejc_pron_comp_cbs_ctc_120300_hop132` | **CTC composite (`あ+F`/`チ+D`)** | kana | CSJ+CEJC |
| `fujie/espnet_asr_csj_cejc_pron_span_cbs_ctc_120300_hop132` | **CTC span (`<F> あ </F>`)** | kana | CSJ+CEJC |
| `fujie/espnet_asr_csj_pron_aux_cbs_ctc_120300_hop132` | CTC multitask (N/F/D head) | kana | CSJ |
| `fujie/espnet_asr_csj_cejc_pron_comp_whisper_large_v3_lora_stream_c100_la0` | **Whisper large-v3 + LoRA, block-causal encoder (100 ms chunks, look-ahead 0), composite (`あ+F`/`チ+D`)** | kana | CSJ+CEJC |
| `fujie/espnet_asr_csj_pron_comp_cbs_ctc_120300_hop132` | CTC composite | kana | CSJ |
| `fujie/espnet_asr_csj_pron_span_cbs_ctc_120300_hop132` | CTC span | kana | CSJ |
| `fujie/espnet_asr_cejc_pron_aux_cbs_transducer_081616_hop132` | Transducer | kana | CEJC |
| `fujie/espnet_asr_csj_writ_aux_cbs_transducer_081616_hop132` | Transducer | kanji | CSJ |
| `fujie/espnet_asr_cbs_transducer_120303_hop132_cc0105` | Transducer | kana | CEJC+CSJ |

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
- `examples/run_streaming_asr_live.py` — live (microphone) streaming ASR.
- `examples/demo.py` — Gradio demo.

### Streaming Transducer multitask (recognition + auxiliary information)

```python
import numpy as np, soundfile as sf
from fujielab.asr.espnet_ext.espnet2.bin.asr_multitask_transducer_inference_cbs import (
    Speech2TextMultitaskTransducer,
)

s2t = Speech2TextMultitaskTransducer.from_pretrained(
    "fujie/espnet_asr_csj_cejc_pron_aux_cbs_transducer_120300_hop132",
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
    "fujie/espnet_asr_csj_pron_aux_cbs_ctc_120300_hop132", streaming=True
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
    "fujie/espnet_asr_csj_cejc_pron_comp_whisper_large_v3_lora_stream_c100_la0", device="cuda")
audio, fs = sf.read("utterance.wav")      # 16 kHz mono, <= 30 s
print(s2t(audio)[0].text)                 # "ア+F ノ+F ー+F | ス+D | イ ッ パ ツ メ ..."
```

On the 17 CSJ+CEJC evaluation sets the block-causal model (look-ahead 0) reaches
CER 11.82 % (CBS RNN-T comp: 16.25 %) with F/D F1 85.3 / 57.6; the full-context
version reaches 9.52 %. See the model card for details.
