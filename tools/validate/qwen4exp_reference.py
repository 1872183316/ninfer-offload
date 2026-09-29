"""Independent FP32 reference for a Qwen4-Exp (Qwen3.8-Flash-Next) v3 artifact.

The forward pass follows the reference transformers Qwen4ExpTextModel for one sequence without
cache: hyper-connection streams, the layer-1 PLE n-gram injection, GatedDeltaNet and gated
attention mixers (dense causal; QSA selects every token below 2052 visible tokens), and the
sparse MoE. Weights are decoded exactly from the artifact's stored codes and scales; routed
experts are read row range by row range, so only selected experts are touched.

Usage:
  python -m tools.validate.qwen4exp_reference ARTIFACT --tokenizer DIR --text FILE \
      [--tokens N] [--generate G]
Prints the mean causal NLL over the scored tokens, per-position top-1 ids, and optionally a
greedy continuation, for comparison with ninfer-perplexity / the ninfer CLI.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.artifact.codecs.direct import decode_direct
from tools.artifact.codecs.row_split import RowPlanes, dequantize_row_split
from tools.artifact.formats import DirectFormat, get_format
from tools.artifact.layouts import row_split_geometry
from tools.artifact.reader import Artifact


class Weights:
    """Logical parameter reads through the artifact bindings."""

    def __init__(self, artifact: Artifact) -> None:
        self.artifact = artifact
        self.bindings = artifact.directory.bindings
        self.cache: dict[str, torch.Tensor] = {}

    def _parts(self, name: str) -> list[tuple[str, int, int]]:
        binding = self.bindings[name]
        if "object" in binding:
            obj = self.artifact.object(binding["object"])
            return [(obj.id, 0, math.prod(obj.shape))]
        return [(p["object"], p["range"][0], p["range"][1]) for p in binding["parts"]]

    def _object_rows(self, object_id: str, row: int, count: int) -> torch.Tensor:
        obj = self.artifact.object(object_id)
        fmt = get_format(obj.format)
        if isinstance(fmt, DirectFormat):
            cols = math.prod(obj.shape[1:]) if len(obj.shape) > 1 else 1
            word = fmt.word_bytes
            raw = self.artifact.read_range(obj.offset + row * cols * word, count * cols * word)
            return decode_direct(raw, fmt, (count, cols)).float()
        n, k = obj.shape
        g = row_split_geometry(fmt, (n, k))

        def plane(offset: int, row_bytes: int) -> bytes:
            if row_bytes == 0:
                return b""
            return self.artifact.read_range(obj.offset + offset + row * row_bytes,
                                            count * row_bytes)

        planes = RowPlanes(plane(g.base_offset, g.base_row_bytes),
                           plane(g.high_offset, g.high_row_bytes),
                           plane(g.scale_offset, g.scale_row_bytes), count)
        return dequantize_row_split(planes, fmt, (count, k), dtype=torch.float32)

    def get(self, name: str, shape: tuple[int, ...], cache: bool = True) -> torch.Tensor:
        if cache and name in self.cache:
            return self.cache[name]
        pieces = []
        for object_id, begin, end in self._parts(name):
            obj = self.artifact.object(object_id)
            cols = obj.shape[-1] if len(obj.shape) > 1 else 1
            if begin % cols or end % cols:
                raise ValueError(f"{name}: binding is not row aligned")
            pieces.append(self._object_rows(object_id, begin // cols, (end - begin) // cols)
                          .reshape(-1))
        value = torch.cat(pieces).reshape(shape)
        if cache:
            self.cache[name] = value
        return value

    def rows(self, name: str, row: int, count: int, cols: int) -> torch.Tensor:
        """Rows [row, row+count) of a single-part row-aligned binding."""
        (object_id, begin, _end), = self._parts(name)
        obj = self.artifact.object(object_id)
        obj_cols = obj.shape[-1]
        return self._object_rows(object_id, begin // obj_cols + row, count)[:, :cols]


def rms(x: torch.Tensor, w: torch.Tensor | None, eps: float, offset: bool = True,
        group: int | None = None) -> torch.Tensor:
    shape = x.shape
    if group is not None:
        x = x.reshape(*shape[:-1], -1, group)
    y = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
    y = y.reshape(shape)
    if w is None:
        return y
    return y * ((1.0 + w) if offset else w)


class Model:
    def __init__(self, path: str) -> None:
        self.artifact = Artifact.open(path)
        self.w = Weights(self.artifact)
        components = self.artifact.directory.components
        text = json.loads(json.dumps(components))
        self.cfg = self._text_config(text)

    @staticmethod
    def _text_config(components: dict) -> dict:
        for value in components.values():
            if isinstance(value, dict):
                cfg = value.get("config", value)
                if isinstance(cfg, dict) and cfg.get("model_type") == "qwen4_exp_text":
                    return cfg
                if isinstance(cfg, dict) and "text_config" in cfg:
                    return cfg["text_config"]
        raise ValueError("no Qwen4-Exp text config in the artifact components")

    # --- blocks -------------------------------------------------------------------------------
    def hyper(self, prefix: str, x: torch.Tensor, inject: bool):
        c = self.cfg
        H, n, r = c["hidden_size"], c["hc_count"], c["hc_lowrank"]
        eps = c["rms_norm_eps"]
        xn = rms(x, self.w.get(prefix + "norm", (n * H,)), eps, group=H)
        mix = F.silu(xn @ self.w.get(prefix + "down", (r, n * H)).T / n)
        mix = torch.sigmoid(mix @ self.w.get(prefix + "up", (n * H, r)).T)
        mixed = (mix.unflatten(-1, (n, H)) * xn.unflatten(-1, (n, H))).mean(-2)
        if not inject:
            return mixed
        weights = 2 * torch.sigmoid(xn @ self.w.get(prefix + "inject", (n, n * H)).T / n)
        return mixed, weights

    def ple(self, prefix: str, x: torch.Tensor, ids: list[int]) -> torch.Tensor:
        p = self.cfg["ple"]
        H, n = self.cfg["hidden_size"], self.cfg["hc_count"]
        eps = self.cfg["rms_norm_eps"]
        ngram, heads_per, eos = p["ngram_size"], p["heads_per_ngram"], p["eos_token_id"]
        mult, vocab, offs = p["multipliers"], p["head_vocab_sizes"], p["head_offsets"]
        heads, hd, pack = (ngram - 1) * heads_per, p["head_dim"], p["table_packing"]
        # Reference _shift_right_ignore_eos over [eos]*(ngram-1) + ids.
        hist = [eos] * (ngram - 1) + list(ids)
        T = len(ids)
        mask = (1 << 64) - 1
        emb = torch.zeros(T, heads * hd)
        for t in range(T):
            pos = t + ngram - 1
            last_eos = max((i for i in range(pos) if hist[i] == eos), default=-1)
            shifted = []
            for shift in range(ngram):
                src = pos - shift
                valid = src >= 0 and pos - (last_eos + 1) >= shift
                shifted.append(hist[src] if valid else eos)
            for order in range(2, ngram + 1):
                mixed = (shifted[0] * mult[0]) & mask
                for k in range(1, order):
                    mixed ^= (shifted[k] * mult[k]) & mask
                # torch.long arithmetic: interpret as signed before the remainder
                signed = mixed - (1 << 64) if mixed >= 1 << 63 else mixed
                for j in range(heads_per):
                    head = (order - 2) * heads_per + j
                    row = signed % vocab[head] + offs[head]
                    r = self.w.rows(prefix + "table", row // pack, 1, pack * hd)
                    emb[t, head * hd:(head + 1) * hd] = r[0, (row % pack) * hd:(row % pack + 1) * hd]
        key = rms(emb @ self.w.get(prefix + "key", (n * H, heads * hd)).T,
                  self.w.get(prefix + "norm_key", (n * H,)), eps, group=H).unflatten(-1, (n, H))
        value = emb @ self.w.get(prefix + "value", (H, heads * hd)).T
        query = rms(x, self.w.get(prefix + "norm_query", (n * H,)), eps, group=H).unflatten(-1, (n, H))
        g = (key * query).sum(-1, keepdim=True) / math.sqrt(H)
        g = g.abs().clamp_min(1e-6).sqrt() * g.sign()
        gated = (torch.sigmoid(g) * value.unsqueeze(-2)).flatten(-2)
        normed = rms(gated, self.w.get(prefix + "norm_conv", (n * H,)), eps, group=H)
        kern = self.w.get(prefix + "convolution", (p["conv_kernel"], n * H))  # [K, C]
        K, dil = p["conv_kernel"], ngram
        conv = torch.zeros_like(normed)
        for t in range(T):
            acc = torch.zeros(n * H)
            for k in range(K):
                src = t - (K - 1 - k) * dil
                if src >= 0:
                    acc += kern[k] * normed[src]
            conv[t] = F.silu(acc)
        return x + gated + conv

    def attention(self, prefix: str, h: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        H, nh, kv, d = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        eps = c["rms_norm_eps"]
        T = h.shape[0]
        q = (h @ self.w.get(prefix + "attention/query", (nh * d, H)).T).view(T, nh, d)
        gate = h @ self.w.get(prefix + "attention/gate", (nh * d, H)).T
        k = (h @ self.w.get(prefix + "attention/key", (kv * d, H)).T).view(T, kv, d)
        v = (h @ self.w.get(prefix + "attention/value", (kv * d, H)).T).view(T, kv, d)
        q = rms(q, self.w.get(prefix + "attention/query_norm", (d,)), eps)
        k = rms(k, self.w.get(prefix + "attention/key_norm", (d,)), eps)
        rot = int(d * c["rope_parameters"]["partial_rotary_factor"])
        theta = c["rope_parameters"]["rope_theta"]
        inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))
        ang = torch.arange(T, dtype=torch.float64)[:, None] * inv[None, :]
        cos = torch.cat([ang.cos(), ang.cos()], -1).float()[:, None, :]
        sin = torch.cat([ang.sin(), ang.sin()], -1).float()[:, None, :]

        def rope(x):
            xr, xp = x[..., :rot], x[..., rot:]
            half = torch.cat([-xr[..., rot // 2:], xr[..., :rot // 2]], -1)
            return torch.cat([xr * cos + half * sin, xp], -1)

        q, k = rope(q), rope(k)
        k = k.repeat_interleave(nh // kv, dim=1)
        v = v.repeat_interleave(nh // kv, dim=1)
        scores = torch.einsum("thd,shd->hts", q, k) / math.sqrt(d)
        scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool), 1), -math.inf)
        o = torch.einsum("hts,shd->thd", scores.softmax(-1), v).reshape(T, nh * d)
        o = o * torch.sigmoid(gate)
        return o @ self.w.get(prefix + "attention/output", (H, nh * d)).T

    def gdn(self, prefix: str, h: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        H, eps = c["hidden_size"], c["rms_norm_eps"]
        nk, dk = c["linear_num_key_heads"], c["linear_key_head_dim"]
        nv, dv = c["linear_num_value_heads"], c["linear_value_head_dim"]
        K = c["linear_conv_kernel_dim"]
        T = h.shape[0]
        kw, vw = nk * dk, nv * dv
        q = h @ self.w.get(prefix + "gdn/query", (kw, H)).T
        k = h @ self.w.get(prefix + "gdn/key", (kw, H)).T
        v = h @ self.w.get(prefix + "gdn/value", (vw, H)).T
        z = h @ self.w.get(prefix + "gdn/z", (vw, H)).T
        a = h @ self.w.get(prefix + "gdn/a_projection", (nv, H)).T
        b = h @ self.w.get(prefix + "gdn/b_projection", (nv, H)).T
        mixed = torch.cat([q, k, v], -1)
        kern = self.w.get(prefix + "gdn/convolution", (K, 2 * kw + vw))
        padded = torch.cat([torch.zeros(K - 1, mixed.shape[1]), mixed])
        conv = sum(kern[j] * padded[j:j + T] for j in range(K))
        conv = F.silu(conv)
        q, k, v = conv[:, :kw], conv[:, kw:2 * kw], conv[:, 2 * kw:]
        q = F.normalize(q.view(T, nk, dk), dim=-1, eps=1e-6).repeat_interleave(nv // nk, 1)
        k = F.normalize(k.view(T, nk, dk), dim=-1, eps=1e-6).repeat_interleave(nv // nk, 1)
        q = q / math.sqrt(dk)
        v = v.view(T, nv, dv)
        beta = torch.sigmoid(b)
        g = -self.w.get(prefix + "gdn/a_log", (nv,)).exp() * F.softplus(
            a + self.w.get(prefix + "gdn/dt_bias", (nv,)))
        state = torch.zeros(nv, dk, dv)
        out = torch.zeros(T, nv, dv)
        for t in range(T):
            state = state * g[t].exp()[:, None, None]
            mem = (state * k[t][:, :, None]).sum(1)
            delta = (v[t] - mem) * beta[t][:, None]
            state = state + k[t][:, :, None] * delta[:, None, :]
            out[t] = (state * q[t][:, :, None]).sum(1)
        norm = self.w.get(prefix + "gdn/norm", (dv,))
        o = rms(out, norm, eps, offset=False)
        gate = torch.sigmoid(z.view(T, nv, dv)) if c["output_gate_type"] == "sigmoid" else F.silu(
            z.view(T, nv, dv))
        o = (o * gate).reshape(T, vw)
        return o @ self.w.get(prefix + "gdn/output", (H, vw)).T

    def moe(self, prefix: str, h: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        H, E, topk = c["hidden_size"], c["num_experts"], c["num_experts_per_tok"]
        I, Is = c["moe_intermediate_size"], c["shared_expert_intermediate_size"]
        p = prefix + "moe/"
        probs = (h @ self.w.get(p + "router", (E, H)).T).softmax(-1)
        top, ids = probs.topk(topk, -1)
        top = top / top.sum(-1, keepdim=True)
        out = torch.zeros_like(h)
        for e in sorted(set(ids.flatten().tolist())):
            ep = f"{p}experts/{e}/"
            gate = self.w.get(ep + "gate", (I, H), cache=False)
            up = self.w.get(ep + "up", (I, H), cache=False)
            down = self.w.get(ep + "down", (H, I), cache=False)
            for t in range(h.shape[0]):
                hit = (ids[t] == e).nonzero()
                if hit.numel():
                    y = (F.silu(h[t] @ gate.T) * (h[t] @ up.T)) @ down.T
                    out[t] += top[t, hit[0, 0]] * y
        shared = (F.silu(h @ self.w.get(p + "shared/gate", (Is, H)).T) *
                  (h @ self.w.get(p + "shared/up", (Is, H)).T)) @ self.w.get(p + "shared/down", (H, Is)).T
        score = torch.sigmoid(h @ self.w.get(p + "shared_score", (1, H)).T)
        return out + score * shared

    def forward(self, ids: list[int]) -> torch.Tensor:
        c = self.cfg
        H, n, V = c["hidden_size"], c["hc_count"], c["vocab_size"]
        emb = torch.stack([self.w.rows("text/token_embedding", i, 1, H)[0] for i in ids])
        x = emb.repeat(1, n)
        ple_layer = c["ple"]["layer"]
        for i, kind in enumerate(c["layer_types"]):
            prefix = f"text/layers/{i}/"
            if i == ple_layer:
                x = self.ple(prefix + "ple/", x, ids)
            h, wts = self.hyper(prefix + "attn_hc/", x, True)
            y = self.attention(prefix, h) if kind == "full_attention" else self.gdn(prefix, h)
            x = x + (y.unsqueeze(-2) * wts.unsqueeze(-1)).flatten(-2)
            h, wts = self.hyper(prefix + "ffn_hc/", x, True)
            y = self.moe(prefix, h)
            x = x + (y.unsqueeze(-2) * wts.unsqueeze(-1)).flatten(-2)
            print(f"layer {i} done", flush=True)
        final = self.hyper("text/head_hc/", x, False)
        head = self.w.get("text/output_head", (V, H), cache=False)
        return final @ head.T


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--text", required=True)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--generate", type=int, default=0)
    parser.add_argument("--continuation", default="",
                        help="space-separated token ids appended after the text (teacher forcing); "
                             "reports whether each is the reference argmax")
    args = parser.parse_args()
    from tokenizers import Tokenizer

    tokenizer = Tokenizer.from_file(str(Path(args.tokenizer) / "tokenizer.json"))
    ids = tokenizer.encode(Path(args.text).read_text(), add_special_tokens=False).ids[: args.tokens]
    prompt_tokens = len(ids)
    ids += [int(t) for t in args.continuation.split()]
    torch.set_num_threads(12)
    model = Model(args.artifact)
    with torch.no_grad():
        logits = model.forward(ids).double()
        logp = logits.log_softmax(-1)
        nll = [-logp[t, ids[t + 1]].item() for t in range(len(ids) - 1)]
        print(json.dumps({"tokens": ids, "nll": nll, "mean_nll": sum(nll) / len(nll),
                          "top1": logits.argmax(-1).tolist()}))
        if args.continuation:
            agree = []
            for t in range(prompt_tokens - 1, len(ids) - 1):
                top = logp[t].topk(2)
                agree.append({"pos": t + 1, "token": ids[t + 1], "ref_top1": int(top.indices[0]),
                              "margin": float(top.values[0] - top.values[1]),
                              "token_logp_gap": float(top.values[0] - logp[t, ids[t + 1]])})
            print(json.dumps({"continuation_check": agree,
                              "matches": sum(a["token"] == a["ref_top1"] for a in agree),
                              "total": len(agree)}))
        seq = list(ids)
        for _ in range(args.generate):
            seq.append(int(model.forward(seq)[-1].argmax()))
        if args.generate:
            print(json.dumps({"greedy": seq[len(ids):],
                              "text": tokenizer.decode(seq[len(ids):])}))


if __name__ == "__main__":
    main()
