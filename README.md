# fujielab-asr

Automatic Speech Recognition Modules for Fujie Laboratory based on ESPnet.

## Features

- Supports streaming ASR using Contextual Block Transformer and RNN-T

## Installation

### Requirements

- Python 3.11 (later versions may not work, because of the dependency on ESPnet)
- ESPnet (202301 or later) required version is different for a model you want to use

### Installation with pip

You can install the latest version of fujielab-asr from PyPI using pip:
```bash
pip install fujielab-asr
```

Note that you need to use Python 3.11, so you may need to create a virtual environment first.

### Installation from source

In the case you want to use conda environment:
```
conda create -n fujielab-asr python=3.11
conda activate fujielab-asr
pip install -e .
```

## Example Usage

Check the example scripts in the `examples` directory.
- `examples/run_streaming_asr.py`: Example of streaming ASR.


