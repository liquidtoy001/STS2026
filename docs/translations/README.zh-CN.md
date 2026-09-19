# STS2026

[English](../../README.md) | **简体中文** | [Deutsch](README.de.md) | [Français](README.fr.md)

团队课程项目，探索轻量级 VITS 风格语音模型及其评估方法。

**当前状态：** 已实现共享重建评估，并完成旧版八条真实语音的 2,000 次迭代实验（1,988 次有效更新）。同机重载 checkpoint 后评分一致。参见[实测记录](../LOCAL_BASELINE_RESULT.md)。独立测试集、完整 TTS 质量和新版对比仍未验证。

## 从这里开始

团队成员主要使用 **macOS 和 GitHub 网页**。接受邀请、编辑文档和审查 Pull Request（PR）不需要安装 Git。

1. 将 GitHub 用户名或个人主页链接发给维护者，使用同一账号接受仓库邀请。
2. 阅读 [GitHub 团队手册](../guides/STS2026_GitHub_Team_Handbook.pdf)：第 5 页讲解接受邀请，第 6–9 页介绍网页协作，第 19 页提供纯网页入门练习。
3. 每个任务从 `main` 创建分支，把修改提交到任务分支，再创建合并到 `main` 的 PR。
4. 请另一名队员审查；初期由维护者协调合并。

网页提交已经保存在 GitHub，不需要额外 push。运行代码需要配置[环境说明](../SETUP.md)中的运行环境。配套文档与 PDF 目前主要为英文。

## Notebook 当前实现

[`vits-light_v1.ipynb`](../../notebooks/vits-light_v1.ipynb) 包含文本编码器、后验编码器、流模型、对齐与时长组件、波形解码器、损失函数和训练器。其**当前启用的实验配置**比整体架构的范围更小：

- 使用 LJSpeech，数据路径硬编码为 Kaggle 路径，音频采样率为 22,050 Hz。
- 对八个固定 LJSpeech 样本运行 2,000 步过拟合实验。
- 将时长损失和 KL 损失权重设为零，启用以重建为目标的损失。
- GAN 损失仍为零，判别器仍是占位实现。
- 已接入共享后验重建评估，支持固定片段、均值/采样两种模式、指标和音频导出。参见[评估协议](../EVALUATION.md)。
- 有 CUDA 时使用 CUDA，否则使用 CPU；未选择 Apple 的 MPS 后端。

当前实测仅覆盖训练音频的后验重建，不代表完整 VITS 或语音转换已通过验证。Notebook 没有保存单元格输出。运行前请阅读[基线限制](../SETUP.md#current-limitations)。

## 仓库结构

```text
STS2026/
  README.md
  CONTRIBUTING.md
  .github/pull_request_template.md
  notebooks/
    vits-light_v1.ipynb
  requirements/
    notebook.txt
    docs.txt
  docs/
    README.md
    SETUP.md
    translations/
      README.zh-CN.md
      README.de.md
      README.fr.md
    guides/
      STS2026_GitHub_Team_Handbook.pdf
      ASR_VITS_Evaluation_Guide.pdf
  tools/handbook/
    build_github_handbook.py
    handbook_web_content.py
    check_github_handbook.py
  tmp/                            # Local only / Git-ignored
```

`tmp/`、运行产物 `output/`、数据、虚拟环境、缓存、检查点及生成的音频样本保留在本地，由 Git 忽略。正式 PDF 位于 `docs/guides/`；新增报告资料应先审查，再放入文档目录。

## 协作约定

- 每个任务使用一个短期分支，`main` 保存经过审查的基线。
- 修改 Notebook 前协调负责人，避免多人通过网页同时替换整个文件。
- 每个 PR 如实记录执行过的验证及尚未验证的内容。
- 数据、模型权重及凭据不得进入 Git，Notebook 输出也要检查。
- 不假定 GitHub 分支保护已启用；即使网页允许直接提交，也应遵守团队约定。

详见[贡献指南](../../CONTRIBUTING.md)与[文档索引](../README.md)。

## 验证与复用

11 项行为、报告和导出测试、Notebook 格式检查已通过，并已执行原始配置的本地真实数据实验。这些结果不证明泛化、跨机器一致性或听感质量。参见[验证记录](../EVALUATION_VALIDATION.md)和[本地复现说明](../LOCAL_EVALUATION.md)。

目前尚未选择软件许可证。公开发布或复用前，请确认课程规定、代码来源以及第三方内容和数据条款；许可证由团队共同决定。
