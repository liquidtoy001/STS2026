# 你的评估代码：按数据流学习

先打开 `notebooks/vits-light_v1.ipynb` 的第 14、15 节，再配合
`sts_eval/reconstruction.py` 阅读。原有模型定义不变；评估工作集中在训练器、
共享模块和最终运行入口。Kaggle 操作及交接规范见 [英文说明](EVALUATION.md)。

## 第一步：先问“我们在评估什么”

路径是：真实音频的频谱 → 后验编码器 → 潜变量 → 解码器 → 重建音频。
它没有用文本来生成声音，所以不能把结果叫完整 TTS 质量。
同一批音频既训练又评分，所以叫固定训练批次上的重建实验，不能声称泛化能力。

先看 `FIXED_IDS`：八个明确的语音 ID。旧版和新版必须使用相同顺序。
`fixed_batch` 是一次加载、重复训练的数据；`evaluator.batch` 保存它的独立副本。
manifest 中的哈希检查的是实际张量内容，不只是文件名。

## 第二步：理解帧与采样点的对齐

这里一帧对应 256 个采样点，32 帧对应 8192 点，约 0.37 秒。
在 `ReconstructionEvaluator.__init__` 中，`self.starts` 固定中央片段的起始帧。
到 `evaluate` 中，`start * hop_length` 将帧位置变成原音频的位置。

自己检查：如果 `start_frame = 4`，目标音频应从第 1024 个采样点开始。
如果用原音频开头和模型中央片段比较，即使模型很好，指标也会错误。
短音频补零是为了凑够解码器输入长度；`self.lengths` 保证评分不把补零当真数据。

## 第三步：为什么评估两种潜变量

编码器输出均值 μ 和 log 标准差。`mean` 模式直接解码 μ；
`sample` 模式解码 μ + ε × exp(log_scale)。前者方便观察稳定重建，后者
检查随机采样时的表现。两个模式分别汇总，不能混成一个分数。

注意 `model.eval()` 不会自动阻止 `torch.randn` 采样，也不会自动禁用梯度。
代码使用 `inference_mode` 禁用梯度，用独立 generator 固定评估噪声，
用 `fork_rng` 在退出时恢复训练随机数状态。`finally` 恢复模型模式，
即使评估报错，也不会把下一步训练悄悄留在 eval 模式。

## 第四步：读懂 score，而不是只看总分

- `log_mel_l1`：两段音频分别转 mel 频谱、取自然对数，再求平均绝对差。
- `stft_spectral_convergence`：频谱误差的范数除以目标频谱范数，再对三个分辨率平均。
- `stft_log_magnitude`：三个 STFT 分辨率下的对数幅度平均绝对差。

都越小越好，但它们与训练损失相关，不是独立的听感评分。
`score` 先逐条计算，`evaluate` 再对样本平均，避免长音频或 padding 支配结果。
相同音频应该得零误差；这也是测试中的一个有明确预期的数值检查。

## 第五步：跟踪一次训练更新

在 `overfit_train` 中按顺序找：forward → loss → backward → optimizer step。
`training_history` 记录这次更新所用的损失；它来自更新前、随机片段上的输出。
之后调用 `evaluator.evaluate`，重新计算更新后模型在固定片段上的分数。
这就是为什么损失曲线和评估曲线分开画。

第 0 步先评估未训练模型，之后每 500 步评估一次，最后保存真实匹配的试听片段。
2000 步是实验预算；是否收敛要看曲线，不能先写“已收敛”。

## 第六步：完成你的实验交付

运行 `python tools/build_kaggle_eval.py`，把生成的
`output/kaggle/vits-light_v1_eval.ipynb` 上传 Kaggle，添加 LJSpeech 数据和 GPU，
核对 `DATA_DIR` 后从上到下执行。最后训练单元会真的运行 2000 步。
共享模块已经自动打包，无需手工复制评估函数。源代码有修改时必须重新导出。

运行结束后下载 `/kaggle/working/sts2026-eval-runs/` 下的本次目录，检查：

1. `status.json` 是否 completed，step 是否为 2000。
2. 配置、样本哈希、逐样本指标、损失曲线和最终 checkpoint 是否齐全。
3. 对同一个编号依次听 target、mean、sample，观察改善和异常。
4. 把旧版实际结果与共享协议交给另一位组员；对方负责新版接入和对比分析。

本地测试使用人工正弦波和真实模型代码，验证评估会不会算错或干扰训练。
它不证明 LJSpeech 上的效果。缺少真实实验时，报告要明确写“待运行”，不填模拟分数。
