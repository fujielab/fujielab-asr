# fujielab-asr
Automatic Speech Recognition Modules for Fujie Laboratory


## Installation
```
conda create -n fujielab-asr python=3.11
conda activate fujielab-asr
pip install cmake==3.31
```

You shoud check the version of cmake
```bash
cmake --version
```
If the version is not 3.31, you should check your command search path,
and reconfigure it, or you will fail to install sentencepiece package,
which is required for espnet.

After that, you can install the required packages for espnet and transformers:
```
pip install espnet==202402 torchaudio
pip install espnet_model_zoo
```

