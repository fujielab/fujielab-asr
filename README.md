# fujielab-asr

Automatic Speech Recognition (ASR) modules for Fujie Laboratory, built on top of
[ESPnet](https://github.com/espnet/espnet).

`fujielab-asr` packages streaming-ASR extensions (the `fujielab.asr.espnet_ext`
layer) on top of stock ESPnet, together with ready-to-use pretrained models on the
Hugging Face Hub. It is designed for **online / chunk-by-chunk** recognition.

## Features

- **Streaming ASR** with a Contextual Block Streaming (CBS) encoder.
- Two recognizer families:
  - **RNN-Transducer** (`Speech2Text`) — streaming beam search.
  - **CTC-only multitask** (`Speech2TextMultitaskCTC`) — streaming greedy CTC that,
    in addition to the transcript, predicts a per-token **auxiliary-information
    label**: `N` (normal), `F` (filler / フィラー), `D` (repair / 言い直し).
- Pretrained Japanese models distributed via the Hugging Face Hub
  (loaded with `from_pretrained`).

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
| `fujie/espnet_asr_csj_pron_aux_cbs_ctc_120300_hop132` | **CTC multitask (N/F/D)** | kana | CSJ |
| `fujie/espnet_asr_cejc_pron_aux_cbs_transducer_081616_hop132` | Transducer | kana | CEJC |
| `fujie/espnet_asr_csj_writ_aux_cbs_transducer_081616_hop132` | Transducer | kanji | CSJ |
| `fujie/espnet_asr_cbs_transducer_120303_hop132_cc0105` | Transducer | kana | CEJC+CSJ |

## Example Usage

Runnable scripts are in the `examples/` directory:

- `examples/run_streaming_asr.py` — streaming Transducer ASR.
- `examples/run_streaming_asr_multitask.py` — streaming **CTC multitask** ASR
  (transcript + filler/repair labels).
- `examples/run_streaming_asr_live.py` — live (microphone) streaming ASR.
- `examples/demo.py` — Gradio demo.

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
