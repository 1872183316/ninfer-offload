# NInfer-Offload

> [NInfer](https://github.com/Neroued/ninfer) 的一个分支：把路由专家放在系统内存里，让消费级显卡也能
> 运行 1000 亿参数以上的 MoE 大模型。[English](README.md)

本分支在上游 NInfer（`e31bc99b`）的基础上新增了：

- **RTX 40 系显卡（sm_89）支持。** FP8 矩阵乘改用 sm_89 可用的指令；TMA、PDL 和 NVFP4 相关路径
  不参与编译，所以在 sm_89 上不能使用 NVFP4 权重和 NVFP4/K8V4 KV 缓存。
- **路由专家卸载**（`--moe-offload`）。路由专家留在主机内存里，由 CPU 线程计算（AVX2 内核）；
  路由器、共享专家、注意力、GDN、输出头和 KV 缓存仍在 GPU 上。每层可以把一定数量的常用专家复制到
  GPU 上计算。CPU 和 GPU 之间用 CUDA 流内存操作交接，可以整体录入 CUDA Graph。
- **Qwen3.8-Flash-Next**（`Qwen4ExpForCausalLM`，1770 亿参数，其中 510 亿是 n-gram 嵌入表）：
  超连接残差流、PLE n-gram 注入（嵌入表通过内存映射放在主机上按需查表）、24 查询头 / 2 KV 头的
  注意力，以及 sigmoid 门控的 GatedDeltaNet。
- **流式转换**：直接从 ModelScope / Hugging Face 边下载边转换，354 GB 的 BF16 原版模型除输出文件外
  只需约 20 GB 下载缓存；以及用于数值校验的**独立 FP32 参考实现**
  （`tools/validate/qwen4exp_reference.py`）。

分支相关的设计、限制、校验和测速细节见
[docs/maintainer/expert-offload.md](docs/maintainer/expert-offload.md)（英文）。

## 测试结果（一台开发机）

Xeon E5-2673 v3（12 核，实测内存读带宽约 21 GB/s）、94 GB 内存、RTX 4060 Ti 16 GB（PCIe 3.0
x8）、Ubuntu、CUDA 12.8。贪心解码、BF16 KV、4 个提示词各跑 2 遍：

| 模型（路由专家精度） | 每层放到 GPU 的专家数 | 解码 tok/s |
|---|---|---|
| Qwen3.8-Flash-Next，专家 4.58 bit/权重，模型文件 108 GB | 0（显存 5.2 GB） | 10.0～10.2 |
| | 64 | 12.5～14.1 |
| Qwen3.6-35B-A3B（groupwise-int） | 128 | 47.5～59.3 |
| | 176 | 60.6～68.7 |

同一台机器上的对照：ik_llama.cpp 在一个 Flash-Next GGUF（路由专家实际只有 2.72 bit/权重）上是
14.6 tok/s，在 35B-A3B 的 UD-Q4_K_M GGUF 上是 43.8～44.4 tok/s；测试条件和注意事项见上面的文档。
111 GB 的 UD-Q4_K_XL Flash-Next GGUF 超出这台机器的内存，llama.cpp 无法运行。

## 硬件和系统要求

- NVIDIA RTX 40 系（sm_89）或 RTX 5090（sm_120a）。RTX 30 系及更早的显卡不支持。
- x86-64 CPU，需支持 AVX2、FMA、F16C、BMI2（Intel Haswell、AMD Zen 及以后）。
- 运行 Flash-Next 需要约 80 GB 内存（路由专家 69 GB）和 110 GB 磁盘；解码速度主要取决于内存带宽。
- Linux、CUDA 12.8 或更新版本、GCC 13。

目前只在上面这一台机器上测试过。

## 快速开始：免编译、免转换

预编译的 Linux 程序（RTX 40 系）在 [Releases 页面](https://github.com/1872183316/ninfer-offload/releases)
（需 v0.1.1 或更新），转换好的 Flash-Next 模型（108 GB，3 个文件）在 ModelScope：
[mymodel3861/Qwen3.8-Flash-Next-NInfer-Offload](https://modelscope.cn/models/mymodel3861/Qwen3.8-Flash-Next-NInfer-Offload)。

```bash
modelscope download --model mymodel3861/Qwen3.8-Flash-Next-NInfer-Offload --local_dir flashnext
./ninfer flashnext/qwen3_8_flash_next.ninfer --prompt "你好" --no-thinking \
  --max-context 2048 --kv-capacity 2048 \
  --moe-offload --moe-threads 12 --moe-gpu-experts 64 --moe-expert-stats stats/stats_flash_next.txt
```

## 编译、转换和运行 Flash-Next

```bash
cmake -B build-sm89 -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_BUILD_TYPE=Release
cmake --build build-sm89 -j

# 交互式转换（选择来源、流式或完整下载、精度）：
python -m tools.convert.wizard
# 或手动：准备配置、分词器和分片索引（只下载小文件和每个分片的头部）
python -m tools.convert.download https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/ flashnext-hf
# 边下载边转换（按 15 MB/s 约需 8 小时）
python -m tools.convert --model flashnext-hf --recipe qwen3_8_flash_next --out qwen3_8_flash_next.ninfer \
  --device cuda --stream-url https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/ \
  --stream-budget-gb 20 --max-file-bytes 250000000000

./build-sm89/apps/ninfer qwen3_8_flash_next.ninfer --prompt "你好" --no-thinking \
  --max-context 2048 --kv-capacity 2048 \
  --moe-offload --moe-threads 12 --moe-gpu-experts 64 --moe-expert-stats bench/offload/stats_flash_next.txt
```

两种转换方式、精度选择、磁盘需求和排错见[转换器使用说明](tools/convert/README.zh-CN.md)。

上下文超过 2051 个 token 时使用模型自带的 QSA token 选择（每个查询只看得分最高的 512 个 4-token
块和最近的尾部 token），要求 `--kv-dtype bf16`（默认值）且只支持纯文本输入；2051 以内仍是原来的
稠密注意力。Flash-Next 还没有实现 MTP / 投机解码。`bench/offload/` 里有测速和校准脚本。

兼容 OpenAI / Anthropic 接口的服务端也支持同样的卸载参数（当前源码；v0.1.1 预编译包里还没有）：

```bash
./build-sm89/apps/ninfer-serve qwen3_8_flash_next.ninfer --port 8080 \
  --max-context 8192 --host-kv-mib 1024 \
  --moe-offload --moe-threads 12 --moe-gpu-experts 64 \
  --moe-expert-stats "$PWD/bench/offload/stats_flash_next.txt"
```

`--host-kv-mib 1024` 把前缀缓存用的锁页内存从默认 8 GiB 降下来，给常驻主机内存的专家权重和
n-gram 表（约 70 GB）留出空间。

## 许可证

代码沿用上游 NInfer 的 [Apache License 2.0](LICENSE)。Qwen3.8-Flash-Next 模型权重使用
Qwen Community License 1.0；分发转换后的权重时须附带该许可证，并遵守其中对商业用途的限制。
