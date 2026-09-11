"""LoRA insertion for the Whisper models (loralib), without weight merging.

espnet's own ``create_adapter`` inserts ``loralib`` layers with the library
default ``merge_weights=True`` and then calls ``model.eval()``. That marks the
layers as "merged" while the LoRA matrices are still zero, so the adapter
weights loaded afterwards are silently ignored at inference. The layers built
here never merge: the LoRA path is always added explicitly in ``forward``.
"""

from typing import List, Optional

import torch

from espnet2.layers.create_adapter_utils import (
    check_target_module_exists,
    get_submodules,
    replace_module,
)


def create_lora_adapter(
    model: torch.nn.Module,
    rank: int = 8,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
) -> None:
    """Replace the target Linear/Embedding modules of ``model`` by LoRA ones."""
    try:
        import loralib as lora
    except ImportError as e:
        raise ImportError(
            "loralib is required for the Whisper+LoRA models: "
            "pip install 'fujielab-asr[whisper]'"
        ) from e

    found = False
    for key in [k for k, _ in model.named_modules()]:
        if not check_target_module_exists(key, target_modules):
            continue
        found = True
        parent, name, target = get_submodules(model, key)
        if isinstance(target, lora.LoRALayer):
            continue
        bias = getattr(target, "bias", None) is not None
        if isinstance(target, torch.nn.Embedding):
            new = lora.Embedding(
                target.num_embeddings,
                target.embedding_dim,
                r=rank,
                lora_alpha=alpha,
                merge_weights=False,
            )
        elif isinstance(target, torch.nn.Linear):
            new = lora.Linear(
                target.in_features,
                target.out_features,
                bias=bias,
                r=rank,
                lora_alpha=alpha,
                lora_dropout=dropout_rate,
                merge_weights=False,
            )
        else:
            raise ValueError(f"LoRA target {key} ({type(target)}) is not supported")
        replace_module(parent, name, target, new)
    if not found:
        raise ValueError(f"Target modules {target_modules} not found in the model.")
