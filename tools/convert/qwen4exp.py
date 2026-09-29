"""Qwen4-Exp (Qwen3.8-Flash-Next) architecture adapter.

Text mathematics: hyper-connection residual streams, Gated DeltaNet with a sigmoid output gate,
QSA-indexed full attention, a PLE n-gram layer, and routed MoE with a gated shared expert. The
logical parameter names extend the Qwen3.5 adapter; see docs/maintainer/qwen4exp-model.md.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping

from .model import Model, Parameter
from .qwen3_5 import _Builder, _f32, _fixed, _positive, _rope_source
from .resources import load_resources
from .sources.logical import LogicalSource
from .sources.safetensors import SafetensorsSource, tensor_source

ARCHITECTURES = ("Qwen4ExpForConditionalGeneration", "Qwen4ExpForCausalLM")

_MASK64 = (1 << 64) - 1
_GAMMA = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
_PRIME_1 = 10007


def _splitmix64(value: int) -> int:
    value = (value + _GAMMA) & _MASK64
    value = ((value ^ (value >> 30)) * _M1) & _MASK64
    value = ((value ^ (value >> 27)) * _M2) & _MASK64
    return (value ^ (value >> 31)) & _MASK64


def _multipliers(vocab: int, ngram: int, ple_index: int, seed: int) -> list[int]:
    bound = max(1, ((1 << 63) - 1) // max(vocab, 1) // 2)
    base = seed + _PRIME_1 * ple_index
    return [
        2 * (_splitmix64((base + _GAMMA * (i + 1)) & _MASK64) % bound) + 1
        for i in range(ngram)
    ]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    for divisor in range(3, math.isqrt(value) + 1, 2):
        if value % divisor == 0:
            return False
    return True


def _nth_prime_after(start: int, count: int) -> int:
    prime = start
    for _ in range(count):
        prime += 1
        while not _is_prime(prime):
            prime += 1
    return prime


def is_qwen4exp(config: dict) -> bool:
    architectures = config.get("architectures")
    return isinstance(architectures, list) and bool(architectures) and architectures[0] in ARCHITECTURES


def text_config(source: dict) -> dict:
    if not is_qwen4exp(source) or len(source["architectures"]) != 1:
        raise ValueError(f"unsupported Qwen4-Exp architecture {source.get('architectures')!r}")
    raw = source.get("text_config", source)
    _fixed(raw, "hidden_act", "silu", "text")
    _fixed(raw, "attention_bias", False, "text")
    _fixed(raw, "mamba_ssm_dtype", "float32", "text")
    _fixed(raw, "norm_topk_prob", True, "text")
    _fixed(raw, "indexer_kv_heads", 1, "text")
    result = {"architectures": ["Qwen4ExpForCausalLM"], "model_type": "qwen4_exp_text"}
    for key in (
        "hidden_size", "vocab_size", "num_hidden_layers", "max_position_embeddings",
        "num_attention_heads", "num_key_value_heads", "head_dim",
        "linear_num_key_heads", "linear_key_head_dim", "linear_num_value_heads",
        "linear_value_head_dim", "linear_conv_kernel_dim",
        "num_experts", "num_experts_per_tok", "moe_intermediate_size",
        "shared_expert_intermediate_size", "hc_count", "hc_lowrank",
        "indexer_n_heads", "indexer_head_dim", "indexer_budget", "indexer_compress_ratio",
    ):
        result[key] = _positive(raw.get(key), "text." + key)
    result["tie_word_embeddings"] = bool(raw.get("tie_word_embeddings", False))
    if result["tie_word_embeddings"]:
        raise ValueError("Qwen4-Exp adapter expects an untied output head")
    result["rms_norm_eps"] = _f32(raw.get("rms_norm_eps", 1e-6), "text.rms_norm_eps")
    gate = raw.get("output_gate_type") or raw.get("hidden_act")
    if gate not in ("sigmoid", "silu"):
        raise ValueError(f"unsupported linear-attention output gate {gate!r}")
    result["output_gate_type"] = gate
    layers = [
        "full_attention" if kind in ("full_attention", "indexed_attention") else kind
        for kind in raw["layer_types"]
    ]
    if len(layers) != result["num_hidden_layers"] or set(layers) - {"full_attention", "linear_attention"}:
        raise ValueError("text.layer_types must describe every block")
    result["layer_types"] = layers
    if result["num_attention_heads"] % result["num_key_value_heads"]:
        raise ValueError("attention heads must be divisible by KV heads")
    if result["linear_num_value_heads"] % result["linear_num_key_heads"]:
        raise ValueError("linear value heads must be divisible by key heads")
    if result["indexer_budget"] % result["indexer_compress_ratio"]:
        raise ValueError("indexer_budget must be divisible by indexer_compress_ratio")
    rope = _rope_source(raw, "text")
    _fixed(rope, "mrope_interleaved", True, "text.rope_parameters")
    factor = _f32(rope.get("partial_rotary_factor", 0.25), "partial_rotary_factor")
    theta = _f32(rope.get("rope_theta", 10_000_000), "rope_theta")
    sections = rope.get("mrope_section")
    rotary = int(result["head_dim"] * factor)
    if not isinstance(sections, list) or len(sections) != 3 or sum(sections) != rotary // 2:
        raise ValueError("MRoPE sections and rotary width disagree")
    if rotary > result["indexer_head_dim"]:
        raise ValueError("rotary width must fit the QSA index head")
    result["rope_parameters"] = {
        "rope_theta": theta,
        "partial_rotary_factor": factor,
        "mrope_section": list(sections),
    }
    ple_ids = sorted(set(raw.get("ple_layer_ids") or []))
    if len(ple_ids) != 1:
        raise ValueError("Qwen4-Exp adapter implements exactly one PLE layer")
    layer = ple_ids[0] - 1  # one-indexed in the source config
    if not 0 <= layer < result["num_hidden_layers"] or layers[layer] != "linear_attention":
        raise ValueError("PLE layer must be a linear-attention layer")
    ngram = _positive(raw.get("ngram_size"), "ngram_size")
    per = _positive(raw.get("heads_per_ngram"), "heads_per_ngram")
    heads = (ngram - 1) * per
    embed = _positive(raw.get("ple_embed_dim", result["hidden_size"]), "ple_embed_dim")
    if embed % heads:
        raise ValueError("ple_embed_dim must be divisible by the n-gram head count")
    base_vocab = _positive(raw.get("ngram_vocab_size_base"), "ngram_vocab_size_base")
    divisor = _positive(raw.get("make_ngram_vocab_size_divisible_by"), "divisor")
    seed = raw.get("seed", 1234)
    eos = raw.get("eos_token_id", source.get("eos_token_id"))
    eos = eos[0] if isinstance(eos, list) else eos
    ple_index = 0
    sizes, offsets, total = [], [], 0
    for head in range(heads):
        size = _nth_prime_after(base_vocab - 1, ple_index * heads + head + 1)
        sizes.append(size)
        offsets.append(total)
        total += size
    rows = math.ceil(total / divisor) * divisor
    result["ple"] = {
        "layer": layer,
        "ngram_size": ngram,
        "heads_per_ngram": per,
        "head_dim": embed // heads,
        "embed_dim": embed,
        "conv_kernel": _positive(raw.get("ple_conv_kernel_size"), "ple_conv_kernel_size"),
        "eos_token_id": _positive(eos, "eos_token_id"),
        "multipliers": _multipliers(result["vocab_size"], ngram, ple_index, seed),
        "head_vocab_sizes": sizes,
        "head_offsets": offsets,
        "rows": rows,
        "table_packing": 8,
    }
    return result


def _concat_rows_source(store: SafetensorsSource, names: list[str], rows: int, width: int,
                        shape: tuple[int, ...]) -> LogicalSource:
    """C-order concatenation of row blocks stored as separate source tensors."""
    counts = [store.describe(name).shape[0] for name in names]
    if sum(counts) != rows or any(store.describe(n).shape[1:] != (width,) for n in names):
        raise ValueError("n-gram table shards do not cover the padded vocabulary")
    starts = [0]
    for count in counts:
        starts.append(starts[-1] + count * width)

    def read(begin: int, end: int):
        import torch

        parts = []
        for i, name in enumerate(names):
            low, high = max(begin, starts[i]), min(end, starts[i + 1])
            if low < high:
                parts.append(store.read_flat(name, low - starts[i], high - starts[i]))
        return torch.cat(parts) if len(parts) != 1 else parts[0]

    return LogicalSource(shape, "concat(" + ",".join(names) + ")", read)


class _Qwen4ExpBuilder(_Builder):
    def hyper(self, prefix, source_prefix, store, config, *, inject=True):
        h, hc, rank = config["hidden_size"], config["hc_count"], config["hc_lowrank"]
        wide = hc * h
        self.add(prefix + "norm", store, source_prefix + "hc_norm.weight", (wide,))
        self.add(prefix + "down", store, source_prefix + "input_mix_weight_down.weight",
                 (rank, wide), inputs=(prefix + "input",))
        self.add(prefix + "up", store, source_prefix + "input_mix_weight_up.weight",
                 (wide, rank), inputs=(prefix + "mix",))
        if inject:
            self.add(prefix + "inject", store, source_prefix + "block_inject_weight.weight",
                     (hc, wide), inputs=(prefix + "input",))

    def indexer(self, prefix, source_prefix, store, config):
        h, d = config["hidden_size"], config["indexer_head_dim"]
        heads = config["indexer_n_heads"]
        sp = source_prefix + "self_attn.indexer."
        self.add(prefix + "indexer/query_key", store, sp + "index_qk_proj.weight",
                 ((heads + 1) * d, h), inputs=(prefix + "mixer_input",))
        self.add(prefix + "indexer/query_norm", store, sp + "q_layernorm.weight", (d,))
        self.add(prefix + "indexer/key_norm", store, sp + "k_layernorm.weight", (d,))

    def ple(self, prefix, source_prefix, store, config):
        h, hc = config["hidden_size"], config["hc_count"]
        ple = config["ple"]
        wide, embed = hc * h, ple["embed_dim"]
        sp = source_prefix + "ple."
        self.add(prefix + "ple/key", store, sp + "key_proj.weight", (wide, embed),
                 inputs=(prefix + "ple/embedding",))
        self.add(prefix + "ple/value", store, sp + "value_proj.weight", (h, embed),
                 inputs=(prefix + "ple/embedding",))
        for role in ("norm_key", "norm_query", "norm_conv"):
            self.add(prefix + "ple/" + role, store, sp + role + ".weight", (wide,))
        taps = ple["conv_kernel"]
        self.add(prefix + "ple/convolution", store, sp + "conv1d.weight", (taps, wide),
                 source_shape=(wide, 1, taps), transpose=(2, 0, 1))
        names = [sp + f"ple_embedding.ngram_embedding.shard_{i}.weight" for i in range(len([
            n for n in store.weight_map if n.startswith(sp + "ple_embedding.ngram_embedding.shard_")
        ]))]
        packing = ple["table_packing"]
        shape = (ple["rows"] // packing, packing * ple["head_dim"])
        self.model.add(Parameter(prefix + "ple/table", shape,
                                 _concat_rows_source(store, names, ple["rows"], ple["head_dim"], shape),
                                 None, (), "bf16", residency="text"))

    def block(self, prefix, source_prefix, store, config, mixer):
        self.hyper(prefix + "attn_hc/", source_prefix + "attn_hyper_connection.", store, config)
        if prefix.endswith(f"/{config['ple']['layer']}/"):
            self.ple(prefix, source_prefix, store, config)
        if mixer == "full_attention":
            self.attention(prefix, source_prefix, store, config)
            self.indexer(prefix, source_prefix, store, config)
        else:
            self.gdn(prefix, source_prefix, store, config)
        self.hyper(prefix + "ffn_hc/", source_prefix + "mlp_hyper_connection.", store, config)
        self.moe(prefix, source_prefix, store, config)


def build_model(
    base: SafetensorsSource,
    *,
    components: tuple[str, ...] = ("text",),
    companions: Mapping[str, SafetensorsSource] | None = None,
    resource_overrides: Mapping[str, str | Path] | None = None,
) -> Model:
    if set(components) != {"text"}:
        raise ValueError("Qwen4-Exp adapter currently converts the text component only")
    config = text_config(base.config)
    records = {"text": {"config": config}}
    refs, resources, count, special = load_resources(
        base.root, vocab_size=config["vocab_size"], vision_config=None,
        overrides=resource_overrides,
    )
    for component, resource_refs in refs.items():
        records[component]["resources"] = resource_refs
    model = Model(records, resources=resources, token_count=count, special_token_ids=special)
    builder = _Qwen4ExpBuilder(model)
    h, r = config["hidden_size"], config["vocab_size"]
    prefix = "model.language_model."
    builder.add("text/token_embedding", base, prefix + "embed_tokens.weight", (r, h))
    builder.add("text/output_head", base, "lm_head.weight", (r, h), inputs=("text/final_hidden",))
    builder.hyper("text/head_hc/", prefix + "hyper_connection_mixer.", base, config, inject=False)
    for i, kind in enumerate(config["layer_types"]):
        builder.block(f"text/layers/{i}/", prefix + f"layers.{i}.", base, config, kind)
    return model
