"""FP32 reference of the Qwen3.8-Flash-Next MTP predictor and its draft acceptance.

The target pass is tools.validate.qwen4exp_reference over a .ninfer artifact. MTP weights are
read from raw BF16 tensors (`<dir>/manifest.json` + `<name>.bin`, Hugging Face names), because the
artifact does not carry the MTP component yet.

For target tokens x_0..x_N with final wide streams w_t (before head_hc), MTP position t combines
w_t and x_(t+1) at RoPE position t:

  e   = rmsnorm(embedding(x_(t+1)), 1 + pre_fc_norm_embedding)
  h   = rmsnorm(w_t, 1 + pre_fc_norm_hidden)                     (over the whole hc*H stream)
  u_s = fc_embedding e + fc_hidden h_s                           (every stream s)
  u   = one Qwen4-Exp block (attention + MoE, hyper-connection residual) of u
  logits = output_head(head mixer of u)                          (predicts x_(t+2))

A draft chain continues from position t with the block output u and its own predicted token,
attending to the teacher-forced MTP history [0, t] plus its own earlier chain positions, which is
how a speculative round runs after an accepted prefix. The script reports, for draft steps
1..K, the probability that step k matches the target token given steps 1..k-1 matched, and the
expected tokens per round for each draft width.

  python -m tools.validate.qwen4exp_mtp_reference ART --mtp-weights DIR \
      --sequence ids.json [--sequence ...] --prompt-tokens N [--steps 3]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from tools.validate.qwen4exp_reference import Model, rms


class MtpWeights:
    """Logical `mtp/...` names over raw BF16 Hugging Face tensors; other names use `base`."""

    HYPER = {"norm": "hc_norm.weight", "down": "input_mix_weight_down.weight",
             "up": "input_mix_weight_up.weight", "inject": "block_inject_weight.weight"}

    def __init__(self, directory: Path, base, config: dict) -> None:
        self.dir = directory
        self.manifest = json.loads((directory / "manifest.json").read_text())
        self.base = base
        self.cfg = config
        self.cache: dict[str, torch.Tensor] = {}

    def raw(self, name: str) -> torch.Tensor:
        if name not in self.cache:
            entry = self.manifest[name]
            data = (self.dir / entry["file"]).read_bytes()
            self.cache[name] = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).float().reshape(
                entry["shape"])
        return self.cache[name]

    def get(self, name: str, shape: tuple[int, ...], cache: bool = True) -> torch.Tensor:
        if not name.startswith("mtp/"):
            return self.base.get(name, shape, cache)
        return self.resolve(name[4:]).reshape(shape)

    def rows(self, name: str, row: int, count: int, cols: int) -> torch.Tensor:
        return self.base.rows(name, row, count, cols)

    def resolve(self, name: str) -> torch.Tensor:
        c = self.cfg
        layer = "mtp.layers.0."
        group, _, role = name.partition("/")
        if group in ("attn_hc", "ffn_hc", "head_hc"):
            source = {"attn_hc": layer + "attn_hyper_connection.",
                      "ffn_hc": layer + "mlp_hyper_connection.",
                      "head_hc": "mtp.hyper_connection_mixer."}[group]
            return self.raw(source + self.HYPER[role])
        if group == "attention":
            d, heads = c["head_dim"], c["num_attention_heads"]
            if role in ("query", "gate"):
                q = self.raw(layer + "self_attn.q_proj.weight").reshape(heads, 2, d, -1)
                return q[:, int(role == "gate")].reshape(heads * d, -1)
            source = {"key": "k_proj", "value": "v_proj", "output": "o_proj",
                      "query_norm": "q_norm", "key_norm": "k_norm"}[role]
            return self.raw(layer + "self_attn." + source + ".weight")
        if group == "indexer":
            source = {"query_key": "index_qk_proj", "query_norm": "q_layernorm",
                      "key_norm": "k_layernorm"}[role]
            return self.raw(layer + "self_attn.indexer." + source + ".weight")
        if group == "moe":
            mlp = layer + "mlp."
            if role == "router":
                return self.raw(mlp + "gate.weight")
            if role == "shared_score":
                return self.raw(mlp + "shared_expert_gate.weight")
            if role.startswith("shared/"):
                return self.raw(mlp + "shared_expert." + role.split("/")[1] + "_proj.weight")
            _, expert, kind = role.split("/")
            e, inter = int(expert), c["moe_intermediate_size"]
            if kind == "down":
                return self.raw(mlp + "experts.down_proj")[e]
            gate_up = self.raw(mlp + "experts.gate_up_proj")[e]
            return gate_up[:inter] if kind == "gate" else gate_up[inter:]
        raise KeyError(name)


class MtpReference:
    def __init__(self, artifact: str, mtp_dir: Path) -> None:
        self.model = Model(artifact, dense_attention=True)
        self.cfg = self.model.cfg
        self.model.w = MtpWeights(mtp_dir, self.model.w, self.cfg)
        self.w = self.model.w
        c = self.cfg
        rot = int(c["head_dim"] * c["rope_parameters"]["partial_rotary_factor"])
        theta = c["rope_parameters"]["rope_theta"]
        self.rot = rot
        self.inv = 1.0 / (theta ** (torch.arange(0, rot, 2, dtype=torch.float64) / rot))

    def embedding(self, ids: list[int]) -> torch.Tensor:
        H = self.cfg["hidden_size"]
        return torch.stack([self.w.rows("text/token_embedding", i, 1, H)[0] for i in ids])

    def stem(self, wide: torch.Tensor, next_ids: list[int]) -> torch.Tensor:
        c = self.cfg
        H, n, eps = c["hidden_size"], c["hc_count"], c["rms_norm_eps"]
        e = rms(self.embedding(next_ids), self.w.raw("mtp.pre_fc_norm_embedding.weight"), eps)
        h = rms(wide, self.w.raw("mtp.pre_fc_norm_hidden.weight"), eps)
        fe = e @ self.w.raw("mtp.fc_embedding.weight").T
        fh = h.unflatten(-1, (n, H)) @ self.w.raw("mtp.fc_hidden.weight").T
        return (fe.unsqueeze(-2) + fh).flatten(-2)

    def rope(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        ang = positions.double()[:, None] * self.inv[None, :]
        cos = torch.cat([ang.cos(), ang.cos()], -1).float()[:, None, :]
        sin = torch.cat([ang.sin(), ang.sin()], -1).float()[:, None, :]
        r = self.rot
        xr, xp = x[..., :r], x[..., r:]
        half = torch.cat([-xr[..., r // 2:], xr[..., :r // 2]], -1)
        return torch.cat([xr * cos + half * sin, xp], -1)

    def qkv(self, h: torch.Tensor, positions: torch.Tensor):
        c = self.cfg
        H, nh, kv, d = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        eps, T = c["rms_norm_eps"], h.shape[0]
        q = (h @ self.w.get("mtp/attention/query", (nh * d, H)).T).view(T, nh, d)
        gate = h @ self.w.get("mtp/attention/gate", (nh * d, H)).T
        k = (h @ self.w.get("mtp/attention/key", (kv * d, H)).T).view(T, kv, d)
        v = (h @ self.w.get("mtp/attention/value", (kv * d, H)).T).view(T, kv, d)
        q = self.rope(rms(q, self.w.get("mtp/attention/query_norm", (d,)), eps), positions)
        k = self.rope(rms(k, self.w.get("mtp/attention/key_norm", (d,)), eps), positions)
        return q, gate, k, v

    def attend(self, q, gate, keys, values, mask) -> torch.Tensor:
        """q [T,nh,d]; keys/values [T,S,kv,d] per query; mask [T,S] True = visible."""
        c = self.cfg
        H, nh, kv, d = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"], c["head_dim"]
        keys = keys.repeat_interleave(nh // kv, dim=2)
        values = values.repeat_interleave(nh // kv, dim=2)
        scores = torch.einsum("thd,tshd->ths", q, keys) / math.sqrt(d)
        scores = scores.masked_fill(~mask[:, None, :], -math.inf)
        o = torch.einsum("ths,tshd->thd", scores.softmax(-1), values).reshape(q.shape[0], nh * d)
        o = o * torch.sigmoid(gate)
        return o @ self.w.get("mtp/attention/output", (H, nh * d)).T

    def block(self, u: torch.Tensor, attention) -> torch.Tensor:
        h, wts = self.model.hyper("mtp/attn_hc/", u, True)
        y = attention(h)
        u = u + (y.unsqueeze(-2) * wts.unsqueeze(-1)).flatten(-2)
        h, wts = self.model.hyper("mtp/ffn_hc/", u, True)
        y = self.model.moe("mtp/", h)
        return u + (y.unsqueeze(-2) * wts.unsqueeze(-1)).flatten(-2)

    def logits(self, u: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        mixed = self.model.hyper("mtp/head_hc/", u, False)
        return mixed @ self.w.get("text/output_head", (c["vocab_size"], c["hidden_size"]),
                                  cache=False).T

    def chain(self, wide: torch.Tensor, ids: list[int], steps: int) -> list[torch.Tensor]:
        """Draft predictions of chain step 1..steps for every start position t (as [T] ids)."""
        T = len(ids) - 1  # positions with a known next token
        positions = torch.arange(T)
        u = self.stem(wide[:T], ids[1:T + 1])
        saved = {}

        def first(h):
            q, gate, k, v = self.qkv(h, positions)
            saved["k"], saved["v"] = k, v
            mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
            keys = k.unsqueeze(0).expand(T, -1, -1, -1)
            values = v.unsqueeze(0).expand(T, -1, -1, -1)
            return self.attend(q, gate, keys, values, mask)

        u = self.block(u, first)
        preds = [self.logits(u).argmax(-1)]
        chain_k, chain_v = [], []
        prefix_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
        for step in range(2, steps + 1):
            pos = positions + step - 1
            u_in = self.stem(u, preds[-1].tolist())
            def later(h):
                q, gate, k, v = self.qkv(h, pos)
                chain_k.append(k)
                chain_v.append(v)
                keys = torch.cat([saved["k"].unsqueeze(0).expand(T, -1, -1, -1),
                                  torch.stack(chain_k, 1)], 1)
                values = torch.cat([saved["v"].unsqueeze(0).expand(T, -1, -1, -1),
                                    torch.stack(chain_v, 1)], 1)
                mask = torch.cat([prefix_mask, torch.ones(T, len(chain_k), dtype=torch.bool)], 1)
                return self.attend(q, gate, keys, values, mask)
            u = self.block(u_in, later)
            preds.append(self.logits(u).argmax(-1))
        return preds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("artifact")
    parser.add_argument("--mtp-weights", type=Path, required=True)
    parser.add_argument("--sequence", type=Path, action="append", required=True,
                        help="JSON {name, ids, prompt_tokens}: prompt plus target continuation")
    parser.add_argument("--steps", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(12)
    ref = MtpReference(args.artifact, args.mtp_weights)
    report = []
    with torch.no_grad():
        for path in args.sequence:
            seq = json.loads(path.read_text())
            ids, prompt = seq["ids"], seq["prompt_tokens"]
            if len(ids) > 2051:
                raise SystemExit("sequences above 2051 tokens need QSA in the MTP layer")
            logits, wide = ref.model.forward(ids, return_wide=True)
            target_argmax = logits.argmax(-1)
            preds = ref.chain(wide, ids, args.steps)
            # Start positions t whose drafts predict generated tokens: t+2 > prompt.
            starts = [t for t in range(prompt - 1, len(ids) - 1 - args.steps)]
            accepted = [0] * (args.steps + 1)
            for t in starts:
                n = 0
                for k in range(args.steps):
                    if int(preds[k][t]) != ids[t + k + 2]:
                        break
                    n += 1
                for k in range(n + 1):
                    accepted[k] += 1
            ref_agree = sum(int(target_argmax[t]) == ids[t + 1] for t in range(prompt - 1, len(ids) - 1))
            entry = {"name": seq["name"], "positions": len(starts),
                     "target_argmax_agreement": ref_agree / (len(ids) - prompt),
                     "p_step": [accepted[k + 1] / accepted[k] if accepted[k] else 0.0
                                for k in range(args.steps)],
                     "tokens_per_round": [1 + sum(accepted[1:k + 1]) / accepted[0]
                                          for k in range(1, args.steps + 1)]}
            print(json.dumps(entry), flush=True)
            report.append(entry)
    print(json.dumps({"summary": report}))


if __name__ == "__main__":
    main()
