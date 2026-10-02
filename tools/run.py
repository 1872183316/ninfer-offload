"""One command from a model name to a chat: find or build a .ninfer model, size it, and run it.

  python -m tools.run Qwen/Qwen3.8-Flash-Next          # chat in this terminal
  python -m tools.run Qwen/Qwen3.6-35B-A3B --serve     # keep an OpenAI/Anthropic API running
  python -m tools.run ./models/x.ninfer --prompt "Hi"  # one answer with the CLI

MODEL is a .ninfer file, a local checkpoint directory (config.json + *.safetensors), or a
repository id. A repository id is resolved in this order: a model already under --models-dir; a
known prebuilt NInfer conversion; the repository itself when it holds .ninfer files; otherwise
the checkpoint is converted with its official recipe by streaming conversion (needs PyTorch with
CUDA and about the output size plus 30 GB of disk).

GPU placement is chosen from free GPU memory: models that do not fit run with routed experts in
host memory (--moe-offload) and as many experts per layer on the GPU as fit. Every chosen value
can be overridden; --dry-run prints the resolved model and command without running anything.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

from tools.artifact.reader import Artifact
from tools.artifact.schema import TensorObject
from tools.convert import download

ROOT = Path(__file__).resolve().parents[1]
GIB = 1 << 30

# Repository id (lower case) -> (hub, repository with a ready .ninfer conversion).
PREBUILT = {
    "qwen/qwen3.8-flash-next": ("modelscope", "mymodel3861/Qwen3.8-Flash-Next-NInfer-Offload"),
    "qwen/qwen3.6-35b-a3b": ("huggingface", "neroued/Qwen3.6-35B-A3B-NInfer"),
    "qwen/qwen3.6-27b": ("huggingface", "neroued/Qwen3.6-27B-NInfer"),
    "qwen/qwen3.8-27b": ("huggingface", "neroued/Qwen3.8-27B-NInfer"),
}
# Artifact metadata name -> routing statistics that rank GPU-resident experts.
EXPERT_STATS = {
    "qwen3.8-flash-next": "stats_flash_next.txt",
    "qwen3.6-35b-a3b": "stats_qwen3_6_35b_a3b.txt",
}
HF_ENDPOINTS = ("https://huggingface.co", "https://hf-mirror.com")

# MTP draft tokens per round. On the development machine one draft was faster than no MTP on
# every benchmark prompt; two and three drafts help code more but slow down free-form prose.
DEFAULT_DRAFT_TOKENS = 1

# Device memory kept free beyond the planned weights and runtime (CUDA context, graphs, slack).
HEADROOM = int(1.5 * GIB)
# Planned runtime besides weights and KV (workspace, state, tables).
RUNTIME = int(0.5 * GIB)


def log(message: str) -> None:
    print(f"[ninfer-run] {message}", file=sys.stderr, flush=True)


# --- locating and fetching models --------------------------------------------------------------

def entry_file(directory: Path) -> Path | None:
    """The entry .ninfer file of a model directory (continuation parts excluded)."""

    files = sorted(p for p in directory.glob("*.ninfer") if p.is_file())
    return files[0] if len(files) == 1 else None


def hf_endpoint() -> str:
    if os.environ.get("HF_ENDPOINT"):
        return os.environ["HF_ENDPOINT"].rstrip("/")
    for endpoint in HF_ENDPOINTS:
        try:
            urllib.request.urlopen(urllib.request.Request(endpoint, headers=download.USER_AGENT),
                                   timeout=10).close()
            return endpoint
        except (OSError, urllib.error.URLError):
            continue
    raise SystemExit("neither huggingface.co nor hf-mirror.com is reachable; set HF_ENDPOINT")


def list_files(hub: str, repo: str) -> dict[str, int]:
    """Every file of a repository with its size."""

    if hub == "modelscope":
        url = (f"https://modelscope.cn/api/v1/models/{repo}/repo/files"
               "?Recursive=true&Revision=master")
        data = json.loads(download.fetch(url))
        if data.get("Code") != 200:
            raise urllib.error.HTTPError(url, 404, "repository not found", None, None)
        return {f["Path"]: int(f["Size"]) for f in data["Data"]["Files"] if f["Type"] == "blob"}
    url = f"{hf_endpoint()}/api/models/{repo}/tree/main?recursive=true"
    return {f["path"]: int(f["size"]) for f in json.loads(download.fetch(url))
            if f["type"] == "file"}


def ninfer_files(files: dict[str, int]) -> dict[str, int]:
    return {n: s for n, s in files.items()
            if n.endswith(".ninfer") or ".ninfer.part-" in n or n == "SHA256SUMS"}


def fetch_prebuilt(hub: str, repo: str, files: dict[str, int], target: Path) -> Path:
    target.mkdir(parents=True, exist_ok=True)
    total = sum(files.values())
    need = sum(s for n, s in files.items()
               if not ((target / n).is_file() and (target / n).stat().st_size == s))
    if need > shutil.disk_usage(target).free - 2 * GIB:
        raise SystemExit(f"not enough disk in {target}: {need / 1e9:.1f} GB to download")
    base = download.repository_url(hub, repo, None if hub == "modelscope" else hf_endpoint())
    log(f"downloading {repo} ({total / 1e9:.1f} GB, {need / 1e9:.1f} GB missing) to {target}")
    done = 0
    lock = threading.Lock()
    started = time.monotonic()
    reported = [0.0]

    def progress(count: int) -> None:
        nonlocal done
        with lock:
            done += count
            now = time.monotonic()
            if now - reported[0] >= 10:
                reported[0] = now
                log(f"  {done / 1e9:.1f} / {need / 1e9:.1f} GB "
                    f"({done / max(now - started, 1e-9) / 1e6:.1f} MB/s)")

    threads = []
    errors = []
    for name, size in files.items():
        path = target / name
        if path.is_file() and path.stat().st_size == size:
            continue

        def run(name=name, size=size, path=path):
            try:
                download._download(base + name, path, size, progress)
            except BaseException as error:  # reported after join
                errors.append(error)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        threads.append(thread)
    for thread in threads:
        thread.join()
    if errors:
        raise errors[0]
    entry = entry_file(target)
    if entry is None:
        raise SystemExit(f"{repo} does not contain exactly one entry .ninfer file")
    return entry


def require_supported(config: dict, source: str) -> None:
    architectures = config.get("architectures") or []
    if not any(a.startswith(("Qwen3_5", "Qwen4Exp")) for a in architectures):
        raise SystemExit(f"{source}: {', '.join(architectures) or 'unknown architecture'} is not "
                         "supported; NInfer-Offload converts Qwen3.5/3.6/3.8 dense and MoE "
                         "checkpoints and Qwen3.8-Flash-Next")


def convert(checkpoint: Path, url: str | None, repo: str, target: Path, budget_gb: int) -> Path:
    """Convert with the official recipe; streaming when `url` is given."""

    try:
        from tools.convert.wizard import recipe_for
    except ModuleNotFoundError as error:
        if error.name and error.name.startswith("tools"):
            raise SystemExit("this model needs conversion, which runs from the NInfer-Offload "
                             "source tree: https://github.com/1872183316/ninfer-offload")
        raise SystemExit(f"conversion needs PyTorch and NumPy ({error}): "
                         "pip install torch numpy")
    import torch

    config = json.loads((checkpoint / "config.json").read_text())
    recipe, template = recipe_for(config, repo, interactive=False)
    name = repo.split("/")[-1].lower()
    out = target / f"{recipe}.ninfer"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    command = [sys.executable, "-u", "-m", "tools.convert", "--model", str(checkpoint),
               "--recipe", recipe, "--out", str(out), "--name", name, "--device", device]
    if template:
        command += ["--resource", f"chat_template.jinja={ROOT / template}"]
    if recipe in ("qwen3_6_27b", "qwen3_8_27b"):
        command.append("--proposal")
    if recipe == "qwen3_8_flash_next":
        command += ["--components", "text,mtp"]
    if url:
        command += ["--stream-url", url, "--stream-budget-gb", str(budget_gb)]
    log(f"converting with recipe {recipe} (this can take hours; rerun resumes downloads)")
    log("  " + shlex.join(command))
    for stale in target.glob(f".{recipe}.ninfer.*.tmp"):
        stale.unlink()
    subprocess.run(command, check=True, cwd=ROOT)
    return out


def resolve(model: str, models_dir: Path, budget_gb: int) -> Path:
    path = Path(model).expanduser()
    if path.is_file() and path.suffix == ".ninfer":
        return path
    if path.is_dir():
        if entry := entry_file(path):
            return entry
        if (path / "config.json").is_file():
            require_supported(json.loads((path / "config.json").read_text()), str(path))
            target = models_dir / path.resolve().name
            if entry := entry_file(target):
                return entry
            return convert(path, None, path.resolve().name, target, budget_gb)
        raise SystemExit(f"{path} holds neither a .ninfer model nor a checkpoint")
    if "/" not in model or model.startswith((".", "/", "~")):
        raise SystemExit(f"{model} is not a file, directory or repository id")

    target = models_dir / model.split("/")[-1]
    if entry := entry_file(target):
        log(f"using {entry}")
        return entry
    if model.lower() in PREBUILT:
        hub, repo = PREBUILT[model.lower()]
        log(f"found a prebuilt conversion: {hub} {repo}")
        return fetch_prebuilt(hub, repo, ninfer_files(list_files(hub, repo)), target)
    hub = "modelscope"
    try:
        files = list_files(hub, model)
    except (urllib.error.HTTPError, KeyError):
        hub = "huggingface"
        files = list_files(hub, model)
    if ninfer := ninfer_files(files):
        if any(n.endswith(".ninfer") for n in ninfer):
            return fetch_prebuilt(hub, model, ninfer, target)
    url = download.repository_url(hub, model, None if hub == "modelscope" else hf_endpoint())
    require_supported(json.loads(download.fetch(url + "config.json")), model)
    if not (ROOT / "tools" / "convert" / "wizard.py").is_file():
        raise SystemExit(f"{model} has no ready .ninfer conversion; converting it needs the "
                         "NInfer-Offload source tree: https://github.com/1872183316/ninfer-offload")
    checkpoint = target / "hf"
    log(f"no .ninfer conversion found; streaming conversion from {hub}")
    download.prepare(url, checkpoint)
    return convert(checkpoint, url, model, target, budget_gb)


# --- sizing ------------------------------------------------------------------------------------

def gpu_memory(device: int) -> tuple[int, int, str]:
    """(free, total) bytes and name of a GPU, from nvidia-smi."""

    try:
        line = subprocess.run(
            ["nvidia-smi", f"--id={device}", "--query-gpu=memory.free,memory.total,name",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, check=True
        ).stdout.strip().splitlines()[0]
    except (OSError, subprocess.CalledProcessError, IndexError):
        raise SystemExit("nvidia-smi failed: is the NVIDIA driver installed?")
    free, total, name = (part.strip() for part in line.split(","))
    return int(free) << 20, int(total) << 20, name


def available_memory() -> int:
    for line in Path("/proc/meminfo").read_text().splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return 0


def physical_cores() -> int:
    cores = set()
    physical = core = None
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        key, _, value = line.partition(":")
        key = key.strip()
        if key == "physical id":
            physical = value.strip()
        elif key == "core id":
            core = value.strip()
        elif not key and core is not None:
            cores.add((physical, core))
            physical = core = None
    if core is not None:
        cores.add((physical, core))
    return len(cores) or os.cpu_count() or 1


class ModelFacts:
    """Byte sizes of the Text (and MTP) weights by placement, from the artifact directory."""

    def __init__(self, path: Path) -> None:
        with Artifact(path) as artifact:
            directory = artifact.directory
            objects = {o.id: o for o in artifact.objects if isinstance(o, TensorObject)}
            self.name = directory.metadata.get("name", "")
            self.mtp = "mtp" in directory.components
            config = directory.components.get("text", {})
            config = config.get("config", config) if isinstance(config, dict) else {}
            config = config.get("text_config", config)
        self.experts = int(config.get("num_experts", 0))
        self.qwen4exp = "hc_count" in config
        self.indexer_limit = (int(config["indexer_budget"]) + int(config["indexer_compress_ratio"])
                              - 1) if config.get("indexer_budget") else 0
        layers = config.get("layer_types") or []
        attention = sum(1 for kind in layers if kind == "full_attention")
        if not layers and config.get("full_attention_interval"):
            attention = int(config["num_hidden_layers"]) // int(config["full_attention_interval"])
        kv_bytes = 2 * int(config.get("num_key_value_heads", 0)) * int(config.get("head_dim", 0)) * 2
        self.kv_bytes_per_token = attention * kv_bytes
        self.index_bytes_per_token = attention * int(config.get("indexer_head_dim", 0)) * 2
        self.mtp_kv_bytes_per_token = kv_bytes
        self.mtp_index_bytes_per_token = int(config.get("indexer_head_dim", 0)) * 2
        expert, host, device = set(), set(), set()
        mtp_device, mtp_host = set(), set()
        for name, binding in directory.bindings.items():
            ids = [binding["object"]] if "object" in binding else [
                part["object"] for part in binding.get("parts", [])]
            ids = [i for i in ids if i in objects]
            if name.startswith("text/"):
                bucket = expert if "/moe/experts/" in name else host if "/ple/table" in name else device
                bucket.update(ids)
            elif name.startswith("mtp/"):
                # Qwen4-Exp MTP experts stay in host memory; other MTP weights live on the GPU.
                (mtp_host if self.qwen4exp and "/moe/experts/" in name else mtp_device).update(ids)
        device -= expert | host
        mtp_device -= device | expert | host
        self.expert_bytes = sum(objects[i].bytes for i in expert)
        self.host_bytes = sum(objects[i].bytes for i in host)
        self.device_bytes = sum(objects[i].bytes for i in device)
        self.mtp_device_bytes = sum(objects[i].bytes for i in mtp_device)
        self.mtp_host_bytes = sum(objects[i].bytes for i in mtp_host)

    def kv(self, context: int) -> int:
        index = self.index_bytes_per_token if context > self.indexer_limit > 0 else 0
        return context * (self.kv_bytes_per_token + index)

    def mtp_kv(self, context: int) -> int:
        """The MTP layer's own attention KV (and QSA index) stream."""
        index = self.mtp_index_bytes_per_token if context > self.indexer_limit > 0 else 0
        return context * (self.mtp_kv_bytes_per_token + index)


def plan(facts: ModelFacts, args) -> list[str]:
    free, total, name = gpu_memory(args.device)
    context = args.max_context or (8192 if facts.qwen4exp else 32768)
    use_mtp = facts.mtp if args.mtp is None else args.mtp
    if use_mtp and not facts.mtp:
        raise SystemExit("--mtp: this model file has no MTP component")
    fixed = facts.device_bytes + RUNTIME + facts.kv(context) + HEADROOM
    if use_mtp:
        fixed += facts.mtp_device_bytes + facts.mtp_kv(context)
    full = fixed + facts.expert_bytes
    log(f"GPU {name}: {free / GIB:.1f} of {total / GIB:.1f} GiB free; model {facts.name or '?'}: "
        f"{facts.device_bytes / GIB:.1f} GiB dense weights, {facts.expert_bytes / GIB:.1f} GiB "
        f"routed experts, context {context}")
    options = ["--max-context", str(context), "--kv-capacity", str(context)]
    offload = args.offload
    if offload is None:
        offload = facts.qwen4exp or (facts.experts > 0 and full > free)
    if not offload:
        if full > free:
            raise SystemExit(f"the model needs about {full / GIB:.1f} GiB of free GPU memory; "
                             f"{free / GIB:.1f} GiB is free (dense models cannot offload)")
        if use_mtp:
            options += ["--spec", "mtp", "--draft-tokens", str(args.draft_tokens)]
        return options
    if facts.experts == 0:
        raise SystemExit("--moe-offload needs a MoE model")
    if fixed > free:
        raise SystemExit(f"even with every expert on the CPU the model needs about "
                         f"{fixed / GIB:.1f} GiB of free GPU memory; {free / GIB:.1f} GiB is free "
                         "(try a smaller --max-context)")
    # Routed experts are resident in host memory; host-mapped tables are paged in on demand.
    host_need = facts.expert_bytes + (facts.mtp_host_bytes if use_mtp else 0)
    if host_need + 4 * GIB > available_memory():
        log(f"warning: offload keeps {host_need / GIB:.0f} GiB in host memory but only "
            f"{available_memory() / GIB:.0f} GiB is available; loading may swap or fail")
    per_expert = facts.expert_bytes / facts.experts
    resident = args.gpu_experts
    if resident is None:
        resident = max(0, min(facts.experts, int((free - fixed) / per_expert)))
    threads = args.threads or physical_cores()
    options += ["--moe-offload", "--moe-gpu-experts", str(resident), "--moe-threads", str(threads)]
    stats = EXPERT_STATS.get(facts.name)
    for folder in (ROOT / "bench" / "offload", ROOT.parent / "stats"):
        if stats and (folder / stats).is_file():
            options += ["--moe-expert-stats", str((folder / stats).resolve())]
            break
    log(f"offload: {resident} of {facts.experts} experts per layer on the GPU, {threads} CPU "
        f"threads, {host_need / GIB:.0f} GiB of experts in host memory"
        + (f" (+{facts.host_bytes / GIB:.0f} GiB mapped table)" if facts.host_bytes else ""))
    if use_mtp:
        options += ["--spec", "mtp", "--draft-tokens", str(args.draft_tokens)]
        log(f"MTP speculative decoding with {args.draft_tokens} draft token(s) per round")
    return options


# --- running -----------------------------------------------------------------------------------

def binary(name: str, explicit: Path | None) -> Path:
    candidates = [explicit / name] if explicit else []
    candidates += [ROOT / name, ROOT.parent / name]  # wrappers of a prebuilt package
    candidates += sorted(ROOT.glob(f"build*/apps/{name}"))
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    found = shutil.which(name)
    if found:
        return Path(found)
    raise SystemExit(f"{name} not found: build NInfer-Offload or pass --ninfer-dir")


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def wait_ready(process: subprocess.Popen, port: int, log_path: Path) -> str:
    shown = 0
    started = last = time.monotonic()
    while True:
        if process.poll() is not None:
            tail = log_path.read_text(errors="replace").splitlines()[-15:]
            raise SystemExit("ninfer-serve stopped during startup:\n" + "\n".join(tail))
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=5) as r:
                return json.loads(r.read())["data"][0]["id"]
        except (OSError, urllib.error.URLError, ValueError):
            pass
        lines = log_path.read_text(errors="replace").splitlines()
        for line in lines[shown:]:
            if any(word in line for word in ("loading", "ready", "pinning", "error")):
                log("  " + line.split("  ", 1)[-1].strip())
                last = time.monotonic()
        shown = len(lines)
        if time.monotonic() - last >= 30:
            last = time.monotonic()
            log(f"  still loading ({int(last - started)} s; host experts load before the GPU "
                "weights) / 仍在加载")
        time.sleep(2)


def stream_chat(port: int, model: str, messages: list[dict], thinking: bool) -> str:
    body = {"model": model, "messages": messages, "stream": True,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    answer = []
    in_reasoning = False
    timings = {}
    with urllib.request.urlopen(request, timeout=3600) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            timings = chunk.get("timings", timings)
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if text := delta.get("reasoning_content"):
                    if not in_reasoning:
                        sys.stdout.write("\033[2m")
                        in_reasoning = True
                    sys.stdout.write(text)
                if text := delta.get("content"):
                    if in_reasoning:
                        sys.stdout.write("\033[0m\n")
                        in_reasoning = False
                    sys.stdout.write(text)
                    answer.append(text)
                sys.stdout.flush()
    if in_reasoning:
        sys.stdout.write("\033[0m")
    if timings:
        sys.stdout.write(f"\n\033[2m[{timings.get('predicted_n', 0)} tokens, "
                         f"{timings.get('predicted_per_second', 0):.1f} tok/s, prompt "
                         f"{timings.get('prompt_per_second', 0):.1f} tok/s]\033[0m")
    sys.stdout.write("\n")
    return "".join(answer)


def chat(port: int, model: str, thinking: bool) -> None:
    print(f"\nChat with {model}. Commands: /think (toggle thinking, now "
          f"{'on' if thinking else 'off'}), /reset, /exit.\n"
          f"与 {model} 对话。命令：/think 切换思考，/reset 清空对话，/exit 退出。")
    messages: list[dict] = []
    while True:
        try:
            text = input("\n> ").strip()
        except EOFError:
            return
        if not text:
            continue
        if text in ("/exit", "/quit"):
            return
        if text == "/reset":
            messages = []
            print("(conversation cleared / 对话已清空)")
            continue
        if text == "/think":
            thinking = not thinking
            print(f"(thinking {'on' if thinking else 'off'} / 思考已{'开启' if thinking else '关闭'})")
            continue
        messages.append({"role": "user", "content": text})
        try:
            answer = stream_chat(port, model, messages, thinking)
        except KeyboardInterrupt:
            print("\n(interrupted / 已中断)")
            messages.pop()
            continue
        messages.append({"role": "assistant", "content": answer})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", help=".ninfer file, checkpoint directory or repository id")
    parser.add_argument("--models-dir", type=Path,
                        default=Path(os.environ.get("NINFER_MODELS", "~/ninfer-models")),
                        help="where downloaded and converted models live (default ~/ninfer-models)")
    parser.add_argument("--serve", action="store_true",
                        help="run the OpenAI/Anthropic-compatible server instead of a chat")
    parser.add_argument("--port", type=int, default=8080, help="server port with --serve")
    parser.add_argument("--host", default="127.0.0.1", help="server address with --serve")
    parser.add_argument("--prompt", help="answer one prompt with the CLI and exit")
    parser.add_argument("--thinking", action="store_true", help="start with thinking enabled")
    parser.add_argument("--max-context", type=int,
                        help="context tokens (default 8192 for Flash-Next, else 32768)")
    parser.add_argument("--offload", action=argparse.BooleanOptionalAction, default=None,
                        help="force routed-expert offload on or off (default: automatic)")
    parser.add_argument("--gpu-experts", type=int, help="experts per layer on the GPU (offload)")
    parser.add_argument("--threads", type=int, help="CPU expert threads (default: physical cores)")
    parser.add_argument("--mtp", action=argparse.BooleanOptionalAction, default=None,
                        help="MTP speculative decoding (default: on when the model file has MTP)")
    parser.add_argument("--draft-tokens", type=int, default=DEFAULT_DRAFT_TOKENS,
                        help=f"MTP draft tokens per round (default {DEFAULT_DRAFT_TOKENS})")
    parser.add_argument("--device", type=int, default=0, help="CUDA device index")
    parser.add_argument("--stream-budget-gb", type=int, default=20,
                        help="download cache for streaming conversion")
    parser.add_argument("--ninfer-dir", type=Path, help="directory with ninfer and ninfer-serve")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the resolved model and command; download or run nothing")
    parser.epilog = "Arguments after -- are passed to ninfer/ninfer-serve unchanged."
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []
    args = parser.parse_args(argv[:argv.index("--")] if "--" in argv else argv)
    models_dir = args.models_dir.expanduser()

    if args.dry_run:
        path = Path(args.model).expanduser()
        if not (path.is_file() or entry_file(models_dir / args.model.split("/")[-1])):
            hub, repo = PREBUILT.get(args.model.lower(), ("?", "?"))
            log(f"would fetch or convert {args.model} (prebuilt: {hub} {repo}) into {models_dir}")
            return
    model = resolve(args.model, models_dir, args.stream_budget_gb)
    facts = ModelFacts(model)
    options = plan(facts, args)
    if not args.thinking:
        options.append("--no-thinking")
    if args.prompt is not None:
        command = [str(binary("ninfer", args.ninfer_dir)), str(model), "--prompt", args.prompt,
                   *options, *extra]
    else:
        serve_options = [o for o in options if o != "--no-thinking"]
        if "--moe-offload" in options:
            serve_options += ["--host-kv-mib", "1024"]
        port = args.port if args.serve else free_port()
        host = args.host if args.serve else "127.0.0.1"
        command = [str(binary("ninfer-serve", args.ninfer_dir)), str(model), "--host", host,
                   "--port", str(port), *serve_options, *extra]
    log("command: " + shlex.join(command))
    if args.dry_run:
        return
    if args.prompt is not None or args.serve:
        os.execv(command[0], command)

    log_path = models_dir / "logs" / f"{model.stem}.serve.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log(f"starting the server (log: {log_path}); large models take minutes to load")
    with log_path.open("w") as server_log:
        process = subprocess.Popen(command, stdout=server_log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
    try:
        name = wait_ready(process, port, log_path)
        chat(port, name, args.thinking)
    except KeyboardInterrupt:
        print()
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()


if __name__ == "__main__":
    main()
