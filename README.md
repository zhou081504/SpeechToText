# SenseVoice 音频转写与算力评测

使用 SenseVoiceSmall + FSMN-VAD，将文件夹及子目录中的音频转成文本。默认自动选择 CUDA/CPU，`batch_size_s` 默认为 **1**。原有已有结果跳过、单文件失败后继续、文本清理和 CSV 报告保留。

## 环境

```bash
conda create -n speech-text python=3.11 -y
conda activate speech-text
conda install -c conda-forge ffmpeg -y
python -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python -m pip install funasr==1.4.14 modelscope soundfile kaldi-native-fbank
```

上面的 PyTorch 命令安装 CPU 版。GPU 环境请改用与你的驱动兼容的 CUDA 版 PyTorch。确认所使用的 Python 环境可访问模型、FFmpeg 和 GPU：

```bash
python -c "import torch; print(torch.cuda.is_available())"
ffmpeg -version
ffprobe -version
```

首次使用模型可能需要下载。资源监控使用 Python 标准库和 Linux `/proc`，不额外依赖 psutil；整卡 GPU 监控需要 `nvidia-smi` 以及 PyTorch 返回的设备 UUID。

## 普通转写

```bash
python transcribe.py "data"
python transcribe.py "data" --batch-size 120
python transcribe.py "data" --device cpu
python transcribe.py "data" --device cuda:0 --output-dir "转写结果_GPU"
python transcribe.py "data" --no-benchmark
```

- `--batch-size`（别名 `--batch_size`）：正整数，默认 1，传给 FunASR 的 `batch_size_s`。代码默认值位于 `transcribe.py` 的 `DEFAULT_BATCH_SIZE`。
- 该值是动态组批的时长预算（秒），不是文件或片段数量。设为 1 不保证严格单片段。当前安装的 FunASR VAD 流程在 CPU 上逐片段推理，CPU 上增大该参数不保证提速。
- `--device`：默认 `auto`，也可指定 `cpu`、`cuda`、`cuda:0` 等；显式选择不可用设备时直接报错。
- `--output-dir`：默认项目下的 `转写结果`，保持输入目录层次。
- `--overwrite`：重新转写已有结果；普通运行默认跳过非空文本。失败时保留原有文本，CSV 标记本次失败。
- `--no-benchmark`：关闭应用自己的计时、资源采样及性能报告，FunASR 自身的日志不受此参数控制。CSV 保留原有列，测速项留空。之前已有的性能 JSON 不会删除，需注意其修改时间。

处理期间逐个删除转换后的临时 WAV，无需同时保存全量 PCM。

## 一次实测的性能报告

默认启用独立的 `benchmark.py` 和 `resource_monitor.py`。除原有成功文件处理统计外，输出 `性能统计.json`，包含：

- 文件数量、成功/失败/跳过数、成功音频总时长；
- 选定设备、CPU 型号/逻辑核数、GPU 型号/总显存；
- 模型加载时间、成功文件处理时间、端到端总时间、RTF、实时倍速、每小时音频预计耗时；
- CPU 使用率峰值、进程 RAM 峰值、系统内存总量和已用峰值；
- PyTorch 已分配/预留显存峰值、整卡显存已用峰值和利用率峰值；
- batch、VAD、模型、Python 和依赖版本。

普通转写端到端时间包含模型加载、解码、VAD/ASR、文件保存、失败任务和循环开销，排除初始 Python/torch 导入及最后的性能报告写入。原有“综合处理速度”仍按成功文件处理耗时计算，两者不要混用。

监控约每 0.5 秒采样，可能漏掉瞬时峰值；进程内存用 Linux VmHWM 高水位（不含 FFmpeg 子进程），系统 CPU/内存及整卡 GPU 指标包含其他进程。PyTorch 峰值只涵盖其分配器。不可用指标为 `null`，原因见 `resources.notes`。显存/内存峰值是当前数据的实测占用，不等于采购所需容量，需另留余量。

## 按方案对比 batch 与 CPU/GPU

独立评测入口 `evaluate.py` 会为每一组启动新进程，从同一输入集完整重跑，结果保存到唯一目录，不受普通转写已有结果影响。默认对比 60/120/300/600 秒，自动选择设备：

```bash
python evaluate.py "data"
```

分别测试 CPU 和 GPU，每组重复三次：

```bash
python evaluate.py "data" --devices cpu cuda:0 --batch-sizes 60 120 300 600 --repeats 3
```

建议分别准备累计约 1、5、10 小时的代表性音频目录，对每个目录运行相同命令。普通转写默认值仍为 1；如需对比基线可用 `--batch-sizes 1 60 120 300 600`。

评测结果位于 `评测结果/run_.../`：

- `评测对比.csv`：各设备、batch、重复轮次的性能与资源指标；
- `评测汇总.json`：完整结果、按中位倍速排序的配置及算力估算；
- `算力评估.md`：方便阅读的排名与配置容量报告；
- `case_.../`：各轮文本、逐文件 CSV、性能 JSON 和 `运行.log`。

评测的端到端耗时还包括子进程启动、依赖导入、报告和退出。没有额外预热，各轮均包含模型加载；请先准备好本地模型缓存，避免首轮模型下载污染比较。系统文件缓存和设备负载仍可能影响结果，应在空闲机器使用足够长的代表性数据重复测量。

只对全部文件成功、没有跳过且输入文件集未变化的配置进行排名。任何一次重复失败，整组不参加排名。设备不可用、模型加载失败或 OOM 的日志保留，其他配置继续测试；不会静默减小 batch。CPU 的 batch 对比主要作为验证，不应预期与 GPU 相同的动态组批收益。

## 根据业务量计算设备数量与存储

```bash
python evaluate.py "data" --devices cpu cuda:0 --repeats 3 \
  --history-hours 100000 --deadline-days 30 \
  --daily-audio-hours 500 --hours-per-day 20 --utilization 0.8
```

不提供业务量时仅输出实测速度和单节点每日容量，不编造设备数量。可只提供历史量，或只提供日增量。

公式：

```text
单节点日容量 = 实测实时倍速 × 每日运行小时 × 利用率
历史任务节点数 = ceil(历史小时数 / 完成天数 / 单节点日容量)
日增任务节点数 = ceil(日增音频小时数 / 单节点日容量)
同时处理节点数 = ceil((历史小时数 / 完成天数 + 日增小时数) / 单节点日容量)
全量 PCM TB = 音频小时数 × 16000 × 2 × 3600 / 10^12
```

例如倍速 30、每天 20 小时、利用率 0.8，单节点每天处理 480 小时；10 万小时在 30 天内完成需要 7 个同等节点。这是公式测试示例，不是本项目已经测得的速度。

GPU 结果的一个节点指当前测量的一张同型号 GPU 及其配套主机；CPU 结果指同等 CPU 主机。不能将测得的 3060 吞吐直接用于推算 4090/L4/A10，也未验证多节点线性扩展。全部历史 PCM 大小只是存储参考，流水临时盘不必达到该容量。

实现范围对应分享方案中的端到端 ASR 评测和算力计算；不包含文本脱敏或多 GPU 调度。论文最小环境和业务推荐硬件应结合真实测量及余量确定，报告不直接宣称某套配置已满足业务 SLA。

## 验证

```bash
python -m unittest discover -s tests -v
```

测试覆盖参数校验、资源监控、关闭测速、转写成功/失败/跳过、临时文件清理、评测报告以及设备数量计算。转写流程测试使用模拟模型，不代表真实识别精度或硬件性能。
