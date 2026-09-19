# 在自己的电脑上建立旧版基准

## 先准备数据和环境

本地入口使用 Notebook 中的原始模型、损失和训练器定义，跳过演示单元格。
它没有缩小网络，也没有引入新的训练算法。八条音频由固定 ID 选择。
Python 环境需要兼容的 PyTorch/TorchAudio、numpy、pandas、tqdm、IPython 和 matplotlib。

从仓库根目录运行：

```powershell
python tools/prepare_ljspeech_subset.py
python tools/run_local_eval.py --data-dir data/LJSpeech-1.1 --steps 20 --eval-every 20
```

准备脚本从公开镜像下载完整 metadata 和八条音频，固定本次下载的镜像版本，
校验镜像提供的 SHA256，并验证音频是 22,050 Hz、单声道、16-bit PCM。
来源和哈希保存在 `data/LJSpeech-1.1/download_manifest.json`。
这是对镜像文件的完整性检查，不声称已经独立核对官方压缩包内的每个文件。
数据来源：[LJSpeech 官方介绍](https://keithito.com/LJ-Speech-Dataset/)，
[逐文件镜像](https://huggingface.co/datasets/flexthink/ljspeech)。

正式预算：

```powershell
python tools/run_local_eval.py --data-dir data/LJSpeech-1.1 --steps 2000 --eval-every 500
```

每次都会创建新的 `output/local-eval/original_<UTC时间>/`，从头初始化模型。
不要用 20 步的 checkpoint 续训后，混称为独立的 2,000 步运行。

## 先看资源和执行状态，再看分数

查看本次目录中的 `status.json`、`local_measurement.json` 和 `run_config.json`：

- 是否正常完成，循环步数和实际 optimizer 更新次数是否相同。
- AMP 如果发现非有限梯度，可以跳过更新；脚本用优化器钩子统计实际更新次数。
- `peak_allocated_bytes` 是 PyTorch 记录的峰值张量显存，不是整个桌面的显存占用。
- 耗时包含训练、评估、checkpoint 保存，不包含前面的数据加载和模型构建。
- 这是八条训练语音上的重建实验。一次运行没有给出随机种子之间的波动范围。

## 让组员复现同一个 checkpoint 的评分

把原始运行目录及其协议、checkpoint 通过团队约定的文件渠道交接。权重和音频不进 Git。
对方准备同样的数据后，可以运行：

```powershell
python tools/run_local_eval.py --data-dir data/LJSpeech-1.1 --checkpoint output/local-eval/original_实际时间/checkpoints/vits_step_2000.pt
```

这条命令不训练。它会检查原始 manifest 中的数据哈希、采样设置、样本顺序和片段位置，
严格加载模型权重，重新评分，并把汇总指标的绝对差异写入新目录的 `status.json`。
保留 checkpoint 原本的 `运行目录/checkpoints/文件.pt` 层级和 `evaluation/` 文件夹。
跨机器得到多大差异需要实际测量；同机复评分不能证明跨机器逐位一致。

## 建议的阅读顺序

1. `prepare_ljspeech_subset.py`：理解数据来源、固定版本和哈希校验。
2. `run_local_eval.py` 的 `execute`：从 Notebook 读取原始定义，只替换运行预算和种子。
3. Notebook 的 `overfit_train`：理解 forward、loss、backward、optimizer 的顺序。
4. `sts_eval/reconstruction.py`：理解固定片段评分与训练模式、随机数状态恢复。
5. 最后对照 `evaluation_history.json` 与 `curves.png`，区分训练误差和固定片段重建误差。
