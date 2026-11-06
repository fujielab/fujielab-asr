from importlib.metadata import version


def _check_torchaudio():
    """Fail early, with the fix, when torchaudio cannot be loaded with the installed torch.

    torchaudio's compiled extension loads only with the torch it was built for, and ESPnet
    imports torchaudio, so a mismatched pair otherwise surfaces deep inside ESPnet as
    "OSError: Could not load this library: .../libtorchaudio.so".
    """
    try:
        import torchaudio  # noqa: F401
    except OSError as e:
        raise ImportError(
            f"torchaudio {version('torchaudio')} cannot be loaded with torch {version('torch')}. "
            'Install a pair built for each other, e.g. pip install "torch==2.10.0" "torchaudio==2.10.0"'
        ) from e


_check_torchaudio()
