"""Interactive conversion wizard: pick a model, a download mode and a precision, then convert.

  python -m tools.convert.wizard

The wizard only asks questions and composes an ordinary `tools.convert` command, which it prints
before running (or saves as a shell script), so every run can be reproduced without the wizard.
"""

from __future__ import annotations

from contextlib import ExitStack
import json
import os
from pathlib import Path
import shlex
import shutil
import sys
import subprocess

from . import download, precision, qwen4exp
from .official_recipes import RECIPES
from .qwen3_5 import build_model
from .recipe import Recipe
from .sources.safetensors import SafetensorsSource
from .sources.streaming import StreamingSafetensorsSource

ROOT = Path(__file__).resolve().parents[2]
GB = 1e9
MARGIN = 10 * GB

# Presets per model kind: (label, --precision choice). The first is the official recipe.
MOE_PRESETS = (
    ("Recommended: official recipe / 推荐：官方配方", {}),
    ("Smaller: all experts 4-bit / 更小：专家全 4 位", {"experts": 4, "expert-down": 4}),
    ("Higher quality: experts 5-bit, down 6-bit / 更高质量：专家 5 位、down 6 位",
     {"experts": 5, "expert-down": 6}),
    ("Near-lossless: all 8-bit / 接近无损：全部 8 位",
     {"experts": 8, "expert-down": 8, "linear": 8}),
)
DENSE_PRESETS = (
    ("Recommended: official recipe / 推荐：官方配方", {}),
    ("Smaller: all projections 4-bit / 更小：全部 4 位", {"linear": 4}),
    ("Higher quality: all projections 6-bit / 更高质量：全部 6 位", {"linear": 6}),
    ("Near-lossless: all projections 8-bit / 接近无损：全部 8 位", {"linear": 8}),
)
CLASS_LABELS = {
    "experts": "routed expert gate/up / 路由专家 gate/up",
    "expert-down": "routed expert down / 路由专家 down",
    "linear": "other layer projections (attention, GDN, shared expert, MLP) / 其他层投影",
}


def ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default else ""
    while True:
        answer = input(f"{prompt}{suffix}: ").strip()
        if answer:
            return answer
        if default is not None:
            return default


def yes(prompt: str, default: bool) -> bool:
    answer = ask(prompt + (" (Y/n)" if default else " (y/N)"), "y" if default else "n")
    return answer.lower().startswith("y")


def choose(prompt: str, options: list[str], default: int = 1) -> int:
    print(f"\n{prompt}")
    for number, option in enumerate(options, 1):
        print(f"  {number}. {option}")
    while True:
        answer = ask("Choose / 请选择", str(default))
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return int(answer)
        print("  Please enter one of the numbers above. / 请输入上面列出的编号。")


def free_bytes(path: Path) -> int:
    path = path.resolve()
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free


def same_disk(a: Path, b: Path) -> bool:
    def existing(p: Path) -> Path:
        p = p.resolve()
        while not p.exists():
            p = p.parent
        return p
    return os.stat(existing(a)).st_dev == os.stat(existing(b)).st_dev


def recipe_for(config: dict, repo: str, interactive: bool = True) -> tuple[str, str | None]:
    """Official recipe name and maintained chat template for a checkpoint config.

    A dense checkpoint is Qwen3.6 or Qwen3.8; without `interactive` the repository name decides.
    """

    if qwen4exp.is_qwen4exp(config):
        return "qwen3_8_flash_next", None
    text = config.get("text_config", config)
    if "num_experts" in text:
        return "qwen3_6_35b_a3b", "tools/chat_templates/qwen3_6.jinja"
    guess = 2 if "3.8" in repo or "3_8" in repo else 1
    options = ["qwen3_6_27b (Qwen3.6 Dense)", "qwen3_8_27b (Qwen3.8 Dense)"]
    if (choose("Dense model: which official recipe? / Dense 模型使用哪个官方配方？", options, guess)
            if interactive else guess) == 2:
        return "qwen3_8_27b", "tools/chat_templates/qwen3_8.jinja"
    return "qwen3_6_27b", "tools/chat_templates/qwen3_6.jinja"


def open_source(model_dir: Path, url: str | None, stack: ExitStack):
    if (model_dir / "shard_headers.json").is_file() and url:
        return stack.enter_context(StreamingSafetensorsSource(model_dir, url, 0))
    return stack.enter_context(SafetensorsSource(model_dir))


def estimates(base, components, recipe_name, presets):
    builder = qwen4exp.build_model if qwen4exp.is_qwen4exp(base.config) else build_model
    model = builder(base, components=components, companions={}, resource_overrides={})
    results = []
    for _, choice in presets:
        recipe = Recipe(model)
        RECIPES[recipe_name](model, recipe, {"base": base})
        if choice:
            precision.apply(recipe, choice)
        results.append(precision.estimate(recipe))
    present = precision.classes_present(Recipe(model))
    return results, present, model


def custom_choice(present: set[str]) -> dict[str, int]:
    print("\nBits per class (4, 5, 6 or 8; Enter keeps the official choice)."
          "\n每类权重的位数（4、5、6、8；直接回车保持官方配方）。")
    choice = {}
    for key in precision.CLASSES:
        if key not in present:
            continue
        while True:
            answer = input(f"  {CLASS_LABELS[key]}: ").strip()
            if not answer:
                break
            if answer.isdigit() and int(answer) in precision.FORMATS:
                choice[key] = int(answer)
                break
            print("  1-, 2- and 3-bit formats are not supported by NInfer kernels yet; use 4, 5, 6 "
                  "or 8.\n  NInfer 的计算内核目前不支持 1/2/3 位格式，请输入 4、5、6 或 8。")
    return choice


def main() -> None:
    os.chdir(ROOT)
    print("NInfer model conversion wizard / NInfer 模型转换向导")
    print("Press Ctrl-C at any time to quit. / 随时按 Ctrl-C 退出。")

    where = choose("Where are the original weights? / 原版权重在哪里？", [
        "ModelScope (download) / ModelScope（在线）",
        "HuggingFace or a mirror (download) / HuggingFace 或镜像站（在线）",
        "Already on this computer / 已经下载到本机",
    ])
    url = None
    stream = False
    delete_source = False
    if where == 3:
        model_dir = Path(ask("Checkpoint directory (config.json + *.safetensors) / 权重目录"))
        if not (model_dir / "config.json").is_file():
            raise SystemExit(f"{model_dir}/config.json not found / 找不到 config.json")
        repo = model_dir.resolve().name
    else:
        repo = ask("Repository id, e.g. Qwen/Qwen3.8-Flash-Next / 仓库名")
        if where == 1:
            url = download.repository_url("modelscope", repo)
        else:
            endpoint = ask("Endpoint (or a mirror such as https://hf-mirror.com) / 站点",
                           "https://huggingface.co")
            url = download.repository_url("huggingface", repo, endpoint)
        mode = choose("How should the weights be read? / 如何读取原版权重？", [
            "Streaming: download shards while converting, delete each when done (needs the least "
            "disk) / 流式：边下载边转换，用完即删（占用磁盘最少）",
            "Download everything first, then convert (source kept for reuse) / "
            "先完整下载再转换（原版权重保留，可重复使用）",
        ])
        stream = mode == 1
        model_dir = Path(ask("Working directory for index/shards / 索引与分片目录",
                             f"models/{repo.split('/')[-1]}-hf"))
        print(f"\nReading repository index from {url} ... / 正在读取仓库索引……")
        download.prepare(url, model_dir)

    config = json.loads((model_dir / "config.json").read_text())
    recipe_name, template = recipe_for(config, repo)
    print(f"\nDetected recipe / 识别到的配方: {recipe_name}")
    components = ["text"]
    if recipe_name != "qwen3_8_flash_next" and config.get("vision_config") and yes(
            "Include vision (image/video input)? / 包含视觉（图片视频输入）？", False):
        components.append("vision")
    if yes("Include MTP (speculative decoding, faster)? / 包含 MTP（投机解码，更快）？", True):
        components.append("mtp")
    components = tuple(components)

    print("\nEstimating sizes (reads headers only) ... / 正在估算大小（只读文件头）……")
    moe = recipe_name in ("qwen3_8_flash_next", "qwen3_6_35b_a3b")
    presets = MOE_PRESETS if moe else DENSE_PRESETS
    with ExitStack() as stack:
        base = open_source(model_dir, url, stack)
        sizes, present, model = estimates(base, components, recipe_name, presets)
        options = [f"{label}  ≈ {size['total'] / GB:.0f} GB" for (label, _), size in
                   zip(presets, sizes)]
        options.append("Custom bits per class / 自定义每类位数")
        picked = choose("Output precision / 输出精度（1/2/3 位暂不支持）", options)
        if picked <= len(presets):
            choice = presets[picked - 1][1]
            size = sizes[picked - 1]["total"]
        else:
            choice = custom_choice(present)
            recipe = Recipe(model)
            RECIPES[recipe_name](model, recipe, {"base": base})
            if choice:
                precision.apply(recipe, choice)
            size = precision.estimate(recipe)["total"]
            print(f"  estimated size / 估计大小 ≈ {size / GB:.0f} GB")

    suffix = "" if not choice else "-" + "-".join(f"{k}{v}" for k, v in choice.items())
    out = Path(ask("Output file / 输出文件", f"models/{recipe_name}{suffix}.ninfer"))
    if out.exists():
        raise SystemExit(f"{out} already exists / 输出文件已存在")

    source_bytes = sum(download.shard_sizes(model_dir).values()) if url else 0
    present_bytes = sum(p.stat().st_size for p in model_dir.glob("*.safetensors"))
    free_out = free_bytes(out.parent)
    shared = same_disk(out.parent, model_dir)
    budget = None
    if stream:
        free_work = free_out - size if shared else free_bytes(model_dir)
        budget = int(max(0, min(20 * GB, free_work - MARGIN)) // GB)
        need_work = 8 * GB + MARGIN
        if free_work < need_work:
            raise SystemExit(
                f"Not enough disk: output needs {size / GB:.0f} GB plus at least "
                f"{need_work / GB:.0f} GB for the download cache; {free_out / GB:.0f} GB free. "
                f"/ 磁盘空间不足。")
        if budget < 20:
            print(f"  Download cache limited to {budget} GB by free space; conversion may be "
                  f"slower.\n  受可用空间限制，下载缓存为 {budget} GB，转换可能变慢。")
    else:
        missing = max(0, source_bytes - present_bytes)
        need = size + MARGIN + (missing if shared else 0)
        if free_out < need or (not shared and free_bytes(model_dir) < missing + MARGIN):
            raise SystemExit(
                f"Not enough disk: {missing / GB:.0f} GB still to download and {size / GB:.0f} GB "
                f"output; {free_out / GB:.0f} GB free. / 磁盘空间不足。")
        if url:
            delete_source = yes("Delete downloaded original weights after a successful conversion? "
                                "/ 转换成功后删除下载的原版权重？", False)

    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        raise SystemExit("PyTorch is required: pip install torch numpy / 需要安装 PyTorch")

    command = [sys.executable, "-u", "-m", "tools.convert", "--model", str(model_dir),
               "--recipe", recipe_name, "--out", str(out), "--device", device,
               "--components", ",".join(components), "--max-file-bytes", "250000000000"]
    if choice:
        command += ["--precision", ",".join(f"{k}={v}" for k, v in choice.items())]
    if template:
        command += ["--resource", f"chat_template.jinja={template}"]
    if recipe_name in ("qwen3_6_27b", "qwen3_8_27b"):
        command.append("--proposal")
    if stream:
        command += ["--stream-url", url, "--stream-budget-gb", str(budget)]
    fetch = [sys.executable, "-u", "-m", "tools.convert.download", url, str(model_dir), "--all"] \
        if url and not stream else None

    lines = ([shlex.join(fetch)] if fetch else []) + [shlex.join(command)]
    if delete_source:
        lines.append(f"rm -f {shlex.quote(str(model_dir))}/*.safetensors")
    print("\nCommands / 将执行的命令:\n  " + "\n  ".join(lines))
    print(f"\nDevice / 计算设备: {device}.  Output / 输出: {out} (≈ {size / GB:.0f} GB)")
    if stream or fetch:
        print(f"Download / 需下载: ≈ {max(0, source_bytes - present_bytes) / GB:.0f} GB; "
              "time depends on your network. / 耗时取决于网速。")

    action = choose("What now? / 下一步？", [
        "Run now in this terminal / 现在在本终端运行",
        "Save as a shell script (to run under tmux/systemd) / 保存为脚本（用 tmux/systemd 运行）",
    ])
    if action == 2:
        script = out.with_suffix(".convert.sh")
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/sh\nset -e\ncd " + shlex.quote(str(ROOT)) + "\n"
                          + "\n".join(lines) + "\n")
        script.chmod(0o755)
        print(f"Saved / 已保存: {script}\nRun it inside tmux, e.g. / 建议在 tmux 中运行: "
              f"tmux new -s convert {shlex.quote(str(script.resolve()))}")
        return
    out.parent.mkdir(parents=True, exist_ok=True)
    if fetch:
        subprocess.run(fetch, check=True)
    subprocess.run(command, check=True)
    if delete_source:
        for shard in model_dir.glob("*.safetensors"):
            shard.unlink()
    print(f"\nDone / 完成: {out}")


if __name__ == "__main__":
    if any(arg in ("-h", "--help") for arg in sys.argv[1:]):
        print(__doc__)
        sys.exit(0)
    try:
        main()
    except KeyboardInterrupt:
        print("\ncancelled / 已取消")
        sys.exit(130)
    except EOFError:
        print("\ninput ended before all questions were answered / 输入提前结束，未完成全部问题")
        sys.exit(1)
