# 权重转换器使用说明（tools/convert）

[English](README.md) | 中文

`tools.convert` 把 HuggingFace 格式的 safetensors 原版权重（BF16/FP16 等）量化，
写成 NInfer 能加载的 `.ninfer` 文件。

1. [环境准备](#1-环境准备)
2. [最简单的方法：转换向导](#2-最简单的方法转换向导)
3. [两种工作方式：流式 / 完整下载](#3-两种工作方式)
4. [手动转换实例：Qwen3.8-Flash-Next](#4-手动转换实例qwen38-flash-next)
5. [手动转换实例：已下载到本地的模型](#5-手动转换实例已下载到本地的模型)
6. [选择精度（位数）](#6-选择精度位数)
7. [全部命令行参数](#7-全部命令行参数)
8. [自定义配方](#8-自定义配方)
9. [转换后：检查与验证](#9-转换后检查与验证)
10. [常见问题与排错](#10-常见问题与排错)

更底层的配方 API 见 [docs/weight-conversion.md](../../docs/weight-conversion.md)。

---

## 1. 环境准备

| 项目 | 要求 |
|---|---|
| 系统 | Linux |
| Python | 3.11 |
| Python 包 | `torch`、`numpy` |
| GPU | 可选。有 CUDA GPU 时量化计算更快；没有也能用 CPU 转换 |
| 工作目录 | **在仓库根目录运行**（命令形如 `python -m tools.convert`） |

```bash
cd ninfer-offload                     # 仓库根目录
python3.11 -m venv ~/convert-venv && source ~/convert-venv/bin/activate
pip install torch numpy
```

转换器不需要编译 NInfer；转换出来的文件用 `ninfer` 运行。

---

## 2. 最简单的方法：转换向导

```bash
python -m tools.convert.wizard
```

向导会逐步提问，全部可以直接回车使用默认值：

1. **原版权重在哪里**：ModelScope、HuggingFace（或镜像站），还是已经下载到本机。
2. **如何读取**（在线模型）：流式（边下载边转换，占用磁盘最少）或先完整下载再转换。
3. 自动识别模型架构并选用对应的官方配方，再问是否包含 MTP（投机解码，Flash-Next 也支持）；带视觉部分的模型还会问是否包含视觉。
4. **输出精度**：列出几个预设及各自的预计文件大小，也可以自定义每类权重的位数（见[第 6 节](#6-选择精度位数)）。
5. 输出文件路径；向导检查磁盘空间，空间不够会直接说明缺多少。
6. 显示将要执行的完整命令，然后选择**立即运行**，或**保存为脚本**（适合放进 tmux/systemd 长时间运行）。

向导只是帮你拼出普通的转换命令，并在运行前把命令打印出来。以后想重复转换，直接复制这些命令即可。

---

## 3. 两种工作方式

| | 完整下载后转换 | 流式转换 |
|---|---|---|
| 适用 | 磁盘能同时放下原版权重和输出文件 | 原版权重比可用磁盘空间还大，或不想保留原版 |
| 磁盘占用 | 原版权重 + 输出文件 | 输出文件 + 有上限的下载缓存（`--stream-budget-gb`） |
| 原版权重 | 保留，可以用不同精度多次转换 | 用完即删，换精度要重新下载 |

**流式转换原理**：转换器先按配方排好任务顺序，算出每个任务要读哪些原版张量；
后台 6 个线程按这个顺序只下载这些张量的字节范围（HTTP Range，断点续传；同一任务里相邻的张量合并成一个请求），
存到 `<模型目录>/.stream/`，缓存总量不超过 `--stream-budget-gb`。每段数据在最后一个用到它的任务完成后立即删除；
超过缓存上限四分之一的大任务会边转换边释放已读完的部分。所以无论原版权重在分片里怎么分布，
磁盘占用都不会超过上限，每个字节通常只下载一次。模型目录里已经存在的完整分片直接读本地。
转换结束时会打印下载总量和缓存峰值。结果与本地转换完全相同。

---

## 4. 手动转换实例：Qwen3.8-Flash-Next

官方 BF16 原版约 360 GB（131 个分片），按官方配方转换后约 108 GB。
每个分片只下载一次，总耗时主要取决于网速（15 MB/s 时约 7～8 小时）。

### 4.1 磁盘空间

流式转换需要：输出约 108 GB + 下载缓存（默认 20 GB）+ 约 10 GB 余量，合计约 **140 GB**。
缓存里只放接下来几个任务要读的数据；只有下载明显快于转换时，调大缓存才有用。

### 4.2 第一步：下载索引（只有几 MB）

```bash
python -m tools.convert.download \
  https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/ \
  models/flashnext-hf
```

它下载 `config.json`、tokenizer、聊天模板等小文件（仓库里没有的文件会显示 `skipped ... (404)`，正常），
并读取每个分片的文件头，生成 `shard_headers.json`。最后应看到 `wrote shard_headers.json for 131 shards`。

加上 `--all` 会继续下载全部权重分片（断点续传），用于[完整下载后转换](#5-手动转换实例已下载到本地的模型)。

URL 必须是“可以直接下载单个文件的前缀”，以 `/` 结尾：
- ModelScope：`https://modelscope.cn/models/<组织>/<模型>/resolve/master/`
- HuggingFace：`https://huggingface.co/<组织>/<模型>/resolve/main/`（镜像站使用相同路径）

### 4.3 第二步：转换

```bash
python -m tools.convert \
  --model models/flashnext-hf \
  --recipe qwen3_8_flash_next \
  --components text,mtp \
  --out models/qwen3_8_flash_next.ninfer \
  --stream-url https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/ \
  --stream-budget-gb 20 \
  --max-file-bytes 250000000000
```

- `--model` 指向上一步的目录，下载的分片也暂存在这里。
- `--max-file-bytes 250000000000` 让输出保持单个文件；默认按 32 GB 切成
  `xxx.ninfer` + `xxx.ninfer.part-0001` … 多个文件（同样可用，运行时只传入口文件）。
  之后想改变已有模型的分片大小（例如为了满足上传限制），用
  `python -m tools.artifact.reshard xxx.ninfer out/xxx.ninfer --max-file-gb 50`，不要用 `split` 切。
- 加 `--components text,mtp` 会同时转换 MTP 预测层，用于投机解码（多 1.5 GB；ModelScope 上的转换已包含）。
  Flash-Next 没有 `vision` 组件，也不支持 `--proposal`。
- 想先看看结果多大：加 `--dry-run`，只打印每类权重的格式和预计大小，不下载、不转换。

### 4.4 长时间运行

转换持续数小时，远程连接断开会中止任务。用 tmux：

```bash
tmux new -s convert        # 在里面运行转换命令；Ctrl-b d 脱离；tmux attach -t convert 回来
```

或用 systemd 用户服务，还能限制内存用量，避免拖慢整台机器：

```bash
loginctl enable-linger $USER          # 允许注销后继续运行（只需一次）
systemd-run --user --unit ninfer-convert -p MemoryMax=40G -p MemorySwapMax=0 -p Nice=10 \
  --working-directory=$PWD \
  bash -c 'source ~/convert-venv/bin/activate; python -m tools.convert ...参数... > convert.log 2>&1'
systemctl --user status ninfer-convert     # 状态；stop 停止；失败后重启前先 reset-failed
```

`MemoryMax` 按自己的内存设置；Flash-Next 转换在 40G 限制内可以完成。

### 4.5 看懂进度

```
[73/972] text/layers/12/moe/experts/gate (+1): q4_g64_fp16 (...)
```

`73/972` 是任务序号/总数。各任务大小差别很大（最后的 PLE 表约 35 GB），序号不代表时间进度。
看实际写入量更准确：

```bash
du -h  models/.qwen3_8_flash_next.ninfer.*.tmp   # 已写入量，最终约 108 GB
du -sh models/flashnext-hf/.stream               # 下载缓存占用
```

成功后临时文件改名为 `qwen3_8_flash_next.ninfer`，并生成 `qwen3_8_flash_next.ninfer.conversion.json`
（记录来源、配方、格式和耗时）。

### 4.6 中断后怎么办

- `.stream/` 里已完整下载的数据段会被复用，未完成的 `.part` 文件从断点继续下载。
- 输出文件不能续写，转换从头开始；之前已用完删除的数据段要重新下载。
- 重启前删除残留的临时输出：`rm models/.qwen3_8_flash_next.ninfer.*.tmp`。
- 成功后可删除 `models/flashnext-hf/.stream`，保留 tokenizer 和 config。

### 4.7 给已有模型补上 MTP

Flash-Next 的 MTP 预测层（`--components text,mtp`，量化后约 1.5 GB）可以直接加到已有的纯文本
`.ninfer` 上，不用全部重新转换：`--reuse` 会把旧文件里格式完全相同的对象原样复制，只转换其余部分，
流式模式下只需下载 MTP 的张量（5.2 GB）。

```bash
python -m tools.convert --model models/flashnext-hf --recipe qwen3_8_flash_next \
  --components text,mtp --name qwen3.8-flash-next --out models/mtp/qwen3_8_flash_next.ninfer \
  --stream-url https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/ \
  --reuse models/qwen3_8_flash_next.ninfer --max-file-bytes 50000000000
```

旧文件必须来自同一份原版权重、同一个配方和同样的 `--precision`（转换器会检查旧文件里记录的配方和精度）。
新文件需要与旧文件同样大小的磁盘空间（开发机上用时 36 分钟，1003 个对象中复制了 972 个）。

---

## 5. 手动转换实例：已下载到本地的模型

原版权重已在本地（或用 `download --all` 下载完）时，不加 `--stream-*` 参数：

```bash
python -m tools.convert.download \
  https://modelscope.cn/models/Qwen/Qwen3.8-Flash-Next/resolve/master/ models/flashnext-hf --all
python -m tools.convert --model models/flashnext-hf --recipe qwen3_8_flash_next \
  --out models/qwen3_8_flash_next.ninfer --max-file-bytes 250000000000
```

Qwen3.6-35B-A3B（MoE）：

```bash
python -m tools.convert \
  --model /path/to/Qwen3.6-35B-A3B \
  --recipe qwen3_6_35b_a3b \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --out models/qwen3_6_35b_a3b.ninfer
```

Qwen3.6-27B（Dense）：

```bash
python -m tools.convert \
  --model /path/to/Qwen3.6-27B \
  --recipe qwen3_6_27b \
  --components text,vision,mtp \
  --resource chat_template.jinja=tools/chat_templates/qwen3_6.jinja \
  --proposal \
  --out models/qwen3_6_27b.ninfer
```

`--components` 只写需要的：只聊天用默认 `text`；图片输入加 `vision`；MTP 投机解码加 `mtp`。

---

## 6. 选择精度（位数）

### 6.1 官方配方

配方在 [official_recipes.py](official_recipes.py) 中定义：

| 配方 | 模型 | 精度 |
|---|---|---|
| `qwen3_8_flash_next` | Qwen3.8-Flash-Next | 专家 gate/up **4 位**、专家 down **5 位**、PLE 表 5 位、embedding 8 位、输出头 6 位、其他投影 8 位；router、shared_score、GDN a/b 保持 BF16 |
| `qwen3_6_35b_a3b` | Qwen3.6-35B-A3B | 专家 4 位，专家 down 5/6 位，其他投影 8 位 |
| `qwen3_6_27b` / `qwen3_8_27b` | Qwen3.6/3.8-27B | 投影 4/5 位，词表 6/8 位 |
| `qwen3_6_27b_nvfp4` / `qwen3_8_27b_nvfp4` | 需 `--source quantized=` 官方 NVFP4 权重 | NVFP4（只有 RTX 50 系能运行） |

### 6.2 用 `--precision` 改位数

在官方配方基础上，把三类权重统一改成指定位数：

| 类别 | 含义 |
|---|---|
| `experts` | 路由专家的 gate/up（MoE 模型的大头） |
| `expert-down` | 路由专家的 down |
| `linear` | 其他层内投影：注意力、GDN、共享专家、Dense MLP |

embedding、输出头、PLE 表以及配方保持 BF16 的小权重（router 等）不受影响。可选位数：**4、5、6、8**。

```bash
# 专家更高质量：gate/up 5 位、down 6 位
python -m tools.convert ... --recipe qwen3_8_flash_next --precision experts=5,expert-down=6
# 先看大小再决定
python -m tools.convert ... --precision experts=5,expert-down=6 --dry-run
```

Flash-Next 各选择的预计大小（`--dry-run` 输出）：

| 选择 | 预计大小 |
|---|---|
| 官方配方 | 108 GB |
| `experts=4,expert-down=4` | 103 GB |
| `experts=4,expert-down=4,linear=4` | 101 GB |
| `experts=5,expert-down=6` | 123 GB |
| `experts=8,expert-down=8,linear=8` | 167 GB |

位数越高质量越好，但文件更大；专家卸载到内存运行时，解码速度大致与每个 token 读取的专家字节数成反比。

注意：官方配方是测试过的组合。其他组合只经过格式与内核支持范围的检查，没有逐一实际运行，
转换后请先[试跑](#92-试跑)。

### 6.3 为什么没有 1、2、3 位

NInfer 的 GPU 和 CPU 计算内核目前只实现了 4/5/6/8 位分组整数（以及 FP8、NVFP4）格式，转换器
也只能写出这些格式。另外，直接按“组内最大值”取整到 1～3 位会严重损坏模型质量；可用的
低位量化需要重要性矩阵或码本等方法。支持 1～3 位需要新的存储格式、量化算法以及 GPU 和 CPU 内核，
是独立的开发工作，目前不在转换器中。

### 6.4 格式名对照

| 格式名 | 含义 | 平均每个权重 |
|---|---|---|
| `q4_g64_fp16` | 4 位整数，每 64 个一组、一个 FP16 缩放 | 4.25 bit |
| `q5_g64_fp16` | 5 位，64 一组 | 5.25 bit |
| `q6_g64_fp16` | 6 位，64 一组 | 6.25 bit |
| `q8_g32_fp16` | 8 位，32 一组 | 8.5 bit |
| `fp8_e4m3fn_row_bf16` | FP8，每行一个 BF16 缩放 | 约 8 bit |
| `bf16` | 不量化 | 16 bit |

量化方法 `grouped_absmax`：每组用绝对值最大值确定缩放，四舍五入为整数；不需要校准数据。

---

## 7. 全部命令行参数

`python -m tools.convert --help` 随时可查。

| 参数 | 默认 | 说明 |
|---|---|---|
| `--model DIR` | 必填 | 模型目录（config、tokenizer、权重；流式模式下是索引目录）。自动识别架构 |
| `--recipe 名字或文件` | 必填 | 官方配方名，或 `my_recipe.py`、`my_recipe.py:函数名` |
| `--out PATH` | 必填 | 输出路径；**已存在则报错，不会覆盖** |
| `--precision` | 无 | 按类别改位数，如 `experts=4,expert-down=5,linear=8` |
| `--dry-run` | 关 | 只打印格式和预计大小，然后退出 |
| `--override FILE` | 无 | 在配方和 `--precision` 之后执行的 Python 文件 |
| `--components` | `text` | `text,vision,mtp,dflash,dflash2` 中的若干项（Flash-Next：`text` 或 `text,mtp`） |
| `--source 名字=路径` | 无 | 额外数据源，可重复（NVFP4 配方的 `quantized=`、`dflash2=` 等） |
| `--resource 角色=路径` | 无 | 替换打包的资源，如 `chat_template.jinja=模板.jinja` |
| `--proposal` | 关 | 加入投机解码用的提议头（Dense 配方使用） |
| `--proposal-rows` | 131072 | 提议头行数 |
| `--ranking PATH` | 内置 | 提议头 token 排名文件 |
| `--name` | 无 | 写入元数据的模型名，不影响计算 |
| `--device` | `cuda` | 量化计算设备：`cuda`、`cuda:1`、`cpu` |
| `--rows-per-chunk` | 512 | 每次处理的矩阵行数；内存或显存不足时调小 |
| `--max-file-bytes` | 32000000000 | 单文件上限（字节），超过则切成 `.part-000N` |
| `--stream-url URL` | 无 | 开启流式模式（需要 `download` 生成的 `shard_headers.json`） |
| `--stream-budget-gb` | 20 | 流式下载缓存上限（GB）；当前任务正在等待的数据段可以超出 |
| `--reuse FILE` | 无 | 从同一原版权重、同一配方和精度的已有转换中复制格式相同的对象，只转换其余部分（见 4.7） |

---

## 8. 自定义配方

只改个别权重时写一个 override 文件，例如 `my_override.py`：

```python
def configure(model, recipe, sources):
    # 第 0 层 MLP down 改用 6 位
    recipe.assign("text/layers/0/mlp/down", format="q6_g64_fp16", method="grouped_absmax")
```

```bash
python -m tools.convert ... --recipe qwen3_6_27b --override my_override.py
```

- 名字支持 `*` 通配；一个都匹配不上会报错。
- 列出所有逻辑权重名：

  ```python
  def configure(model, recipe, sources):
      for name, p in model.parameters.items():
          print(name, p.shape)
      raise SystemExit
  ```

- 改 `format` 时要同时指定 `method`。
- 转换器只保证文件合法；某种格式组合能否运行由 NInfer 的算子决定，新组合请先试跑。

分组、共享权重、从其他 checkpoint 取权重、自写量化函数等见
[docs/weight-conversion.md](../../docs/weight-conversion.md)。

---

## 9. 转换后：检查与验证

### 9.1 查看文件内容（不跑推理）

```bash
python -m tools.artifact.inspect models/qwen3_8_flash_next.ninfer --objects --bindings
```

### 9.2 试跑

```bash
./build-sm89/apps/ninfer models/qwen3_8_flash_next.ninfer --prompt "The capital of France is" \
  --no-thinking --max-context 2048 --kv-capacity 2048 --max-new 32 \
  --moe-offload --moe-threads 12 --moe-gpu-experts 0
```

输出应是通顺的句子。`--moe-gpu-experts`、`--moe-expert-stats` 等提速参数见仓库根目录 README。

### 9.3 数值验证（可选）

用独立的 FP32 参考实现（直接解码文件里的量化权重，按官方数学逐层计算）与 NInfer 对比：

```bash
echo "A plain English paragraph of about sixty words ..." > sample.txt
python -m tools.validate.qwen4exp_reference models/qwen3_8_flash_next.ninfer \
  --tokenizer models/flashnext-hf --text sample.txt --tokens 64
./build-sm89/apps/ninfer-perplexity models/qwen3_8_flash_next.ninfer --text sample.txt \
  --moe-offload --moe-threads 12
```

官方配方的参考结果：同一段 59 token 英文段落，NInfer PPL 4.2346，FP32 参考 4.2407。
两者应只差千分之几。该参考实现只支持 Flash-Next，在 CPU 上运行，很慢，`--tokens` 取几十即可。

---

## 10. 常见问题与排错

| 现象 | 原因 / 处理 |
|---|---|
| `No module named tools` | 没在仓库根目录运行 |
| `conversion output already exists` | 输出文件已存在；换名字或删除旧文件 |
| 找不到 `shard_headers.json` | 流式模式要先运行 `tools.convert.download`，或 `--model` 指错了目录 |
| `model.safetensors.index.json not found` | URL 或仓库名错误；在浏览器打开 `<前缀>config.json` 应能直接下载 |
| `xxx.safetensors: download incomplete` | 多次重试仍失败：网络长时间中断或磁盘已满。`df -h` 查看后重新运行（已下载分片会复用） |
| 下载速度明显下降且不恢复 | 下载缓存已装满后续任务的数据，瓶颈在转换本身；调大 `--stream-budget-gb` 可以让下载跑得更靠前 |
| `... was released before a later read` | 分片被提前删除（不应发生）；删除该分片后重新运行 |
| `bits must be one of 4, 5, 6, 8` | 1～3 位暂不支持，见 [6.3](#63-为什么没有-123-位) |
| `CUDA out of memory` | 调小 `--rows-per-chunk`（如 128）或用 `--device cpu` |
| 转换时机器卡死 | 内存耗尽；用 systemd-run 的 `MemoryMax` 限制转换进程（见 4.4） |
| 系统盘写满 | `--model`、`--out` 放到空间足够的磁盘；用 `df -h` 检查 |
| 转换成功但 `ninfer` 报格式不支持 | 这个格式组合没有对应算子；改回官方配方或换位数 |
