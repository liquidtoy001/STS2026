"""Summarize a completed local baseline from measured artifacts, without estimates."""

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run = args.run_dir.resolve()

    def read(relative):
        return json.loads((run / relative).read_text(encoding="utf-8"))

    status = read("status.json")
    if status["status"] != "completed":
        raise ValueError("Only completed runs can be summarized as baselines.")
    measurements = read("local_measurement.json")
    history = read("evaluation/evaluation_history.json")
    manifest = read("evaluation/manifest.json")
    config = read("run_config.json")
    step = status["step"]
    lines = ["# 本地旧版后验重建基准", "", f"运行：`{run.name}`。训练种子：{config['train_seed']}。",
             "", "这是固定八条训练音频上的单次过拟合实验；不是独立测试集评估，也没有估计种子间波动。",
             "", "## 执行与资源", "",
             f"- GPU：{measurements['device']}。",
             f"- 参数量：{measurements['parameters']:,}，使用原始 Notebook 网络配置。",
             f"- 循环步数：{step}；实际优化器更新：{measurements['actual_optimizer_updates']}；AMP 跳过：{measurements['amp_skipped_updates']}。",
             f"- 训练、评估和保存耗时：{measurements['wall_seconds_training_and_evaluation']:.2f} 秒。",
             f"- PyTorch 峰值已分配显存：{measurements['peak_allocated_bytes'] / 2**30:.3f} GiB；峰值保留显存：{measurements['peak_reserved_bytes'] / 2**30:.3f} GiB。",
             f"- 批次 SHA256：`{manifest['batch_sha256']}`。",
             "", "## 固定片段指标（越低越好）", "",
             "| 步数 | 模式 | log-mel L1 | STFT spectral convergence | STFT log magnitude |",
             "|---:|---|---:|---:|---:|"]
    for row in history:
        lines.append(f"| {row['step']} | {row['mode']} | {row['log_mel_l1']:.6f} | {row['stft_spectral_convergence']:.6f} | {row['stft_log_magnitude']:.6f} |")
    lines += ["", "指标不是百分比或准确率。均值解码与采样解码分别报告，不能混成一项结果。",
              "短片段与训练损失相关，不能仅凭这些数字判断完整语音自然度或泛化。",
              "", "![训练与评估曲线](curves.png)", "", "## 逐样本最终指标", "",
              "| 样本 | 模式 | 起始帧 | 有效采样点 | log-mel L1 | STFT spectral convergence |",
              "|---|---|---:|---:|---:|---:|"]
    for row in read(f"evaluation/step_{step:06d}/per_sample.json"):
        lines.append(f"| {row['sample_id']} | {row['mode']} | {row['start_frame']} | {row['valid_samples']} | {row['log_mel_l1']:.6f} | {row['stft_spectral_convergence']:.6f} |")
    lines += ["", "## 原始证据", "",
              "- [运行配置](run_config.json)", "- [硬件与耗时记录](local_measurement.json)",
              "- [评估协议与数据指纹](evaluation/manifest.json)",
              "- [完整评估曲线数据](evaluation/evaluation_history.json)",
              f"- [最终逐样本分数](evaluation/step_{step:06d}/per_sample.json)",
              f"- [最终 checkpoint](checkpoints/vits_step_{step}.pt)",
              "", "试听文件编号对应 manifest 中的 sample_ids；请依次比较相同编号的 target、mean、sample。",
              "本报告未执行人工听感打分，也没有对跨硬件一致性或多次训练波动给出结论。", ""]
    path = run / "REPORT.zh-CN.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print(path)


if __name__ == "__main__":
    main()
