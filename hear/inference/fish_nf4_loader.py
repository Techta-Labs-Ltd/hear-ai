from pathlib import Path

import bitsandbytes as bnb
import torch
from fish_speech.models.text2semantic.llama import (
    BaseModelArgs,
    DualARTransformer,
    _convert_linear_layers_to_bnb4,
    precompute_freqs_cis,
)
from fish_speech.tokenizer import FishTokenizer


class FishNF4Loader:
    @staticmethod
    def load(path, *, load_weights=False, max_length=None, bnb4=False, bnb4_compute_dtype=None, **kwargs):
        if not load_weights or not bnb4 or kwargs:
            raise ValueError("fish_nf4_loader_requires_pinned_prequantized_checkpoint")
        checkpoint = Path(path)
        config = BaseModelArgs.from_pretrained(str(checkpoint))
        if config.model_type != "dual_ar":
            raise ValueError("fish_nf4_requires_dual_ar")
        if max_length is not None:
            config.max_seq_len = max_length
        tokenizer = FishTokenizer.from_pretrained(checkpoint)
        config.semantic_begin_id = tokenizer.semantic_begin_id
        config.semantic_end_id = tokenizer.semantic_end_id
        with torch.device("meta"):
            model = DualARTransformer(config)
            _convert_linear_layers_to_bnb4(model, compute_dtype=bnb4_compute_dtype)
        weights = torch.load(
            checkpoint / "model.pth", map_location="cpu", mmap=True, weights_only=True
        )
        if "state_dict" in weights:
            weights = weights["state_dict"]
        consumed = set()
        for name, module in model.named_modules():
            if not isinstance(module, bnb.nn.Linear4bit):
                continue
            key = name + ".weight"
            prefix = key + "."
            stats = {k[len(prefix):]: v for k, v in weights.items() if k.startswith(prefix)}
            if key not in weights or not stats:
                raise RuntimeError("fish_nf4_missing_quantized_linear_weight")
            module.weight = bnb.nn.Params4bit.from_prequantized(
                data=weights[key], quantized_stats=stats,
                requires_grad=False, device="cpu", module=module,
            )
            consumed.add(key)
            for stat in stats:
                consumed.add(prefix + stat)
            bias = name + ".bias"
            if module.bias is not None:
                module.bias = torch.nn.Parameter(weights[bias], requires_grad=False)
                consumed.add(bias)
        remaining = {k: v for k, v in weights.items() if k not in consumed}
        result = model.load_state_dict(remaining, strict=False, assign=True)
        if set(result.missing_keys) - consumed or result.unexpected_keys:
            raise RuntimeError(f"fish_nf4_checkpoint_model_mismatch:{set(result.missing_keys) - consumed}:{result.unexpected_keys}")
        model.freqs_cis = precompute_freqs_cis(config.max_seq_len, config.head_dim, config.rope_base)
        model.fast_freqs_cis = precompute_freqs_cis(
            config.num_codebooks, config.fast_head_dim, config.rope_base
        )
        model.causal_mask = torch.tril(torch.ones(config.max_seq_len, config.max_seq_len, dtype=torch.bool))
        if any(value.is_meta for value in (*model.parameters(), *model.buffers())):
            raise RuntimeError("fish_nf4_unmaterialized_model_tensor")
        model.tokenizer = tokenizer
        model._bnb4_prequantized = True
        return model

    @classmethod
    def install(cls):
        DualARTransformer.from_pretrained = staticmethod(cls.load)
