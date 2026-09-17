# STS2026

[English](README.md) | **简体中文** | [Deutsch](README.de.md) | [Français](README.fr.md)

团队课程项目，探索轻量级 VITS 风格语音模型及其评估方法。

**当前状态：** 初始 Notebook 基线及规划资料。仓库整理过程中尚未验证完整训练、端到端推理质量或可复现评估。

## 从这里开始

团队成员主要使用 **macOS 和 GitHub 网页**。接受邀请、编辑文档和审查 Pull Request（PR）不需要安装 Git。

1. 将 GitHub 用户名或个人主页链接发给维护者，使用同一账号接受仓库邀请。
2. 阅读 [GitHub 团队手册](output/pdf/STS2026_GitHub_Team_Handbook.pdf)：第 5 页讲解接受邀请，第 6–9 页介绍网页协作，第 19 页提供纯网页入门练习。
3. 每个任务从 `main` 创建分支，把修改提交到任务分支，再创建合并到 `main` 的 PR。
4. 请另一名队员审查；初期由维护者协调合并。

网页提交已经保存在 GitHub，不需要额外 push。运行代码需要配置[环境说明](docs/SETUP.md)中的运行环境。配套文档与 PDF 目前主要为英文。

## Notebook 当前实现

[`vits-light_v1.ipynb`](vits-light_v1.ipynb) 包含文本编码器、后验编码器、流模型、对齐与时长组件、波形解码器、损失函数和训练器。其**当前启用的实验配置**比整体架构的范围更小：

- 使用 LJSpeech，数据路径硬编码为 Kaggle 路径，音频采样率为 22,050 Hz。
- 对一个固定批次调用 `overfit_train(total_steps=1500)`。
- 将时长损失和 KL 损失权重设为零，启用以重建为目标的损失。
- GAN 损失仍为零，判别器仍是占位实现。
- `evaluate(outputs, batch)` 尚未实现。
- 有 CUDA 时使用 CUDA，否则使用 CPU；未选择 Apple 的 MPS 后端。

这些内容是开发起点，并不代表完整 VITS 已实现、评估流程已取得测量结果或语音转换已通过验证。Notebook 没有保存单元格输出。运行前请阅读[基线限制](docs/SETUP.md#current-limitations)。

## 仓库结构

```text
STS2026/
  vits-light_v1.ipynb              原始基线 Notebook
  README.md                      默认英文首页
  README.zh-CN.md                中文说明
  README.de.md                   德文说明
  README.fr.md                   法文说明
  CONTRIBUTING.md                团队协作与审查规则
  requirements-notebook.txt       Notebook 依赖（不含 PyTorch）
  requirements-docs.txt           PDF 生成与检查依赖
  .github/pull_request_template.md
  docs/
    SETUP.md                     环境、数据集及运行说明
    README.md                    文档索引与重建说明
    build_github_handbook.py      手册排版与生成脚本
    handbook_web_content.py       英文网页操作手册内容
    check_github_handbook.py      PDF 结构与排版检查
  output/pdf/
    STS2026_GitHub_Team_Handbook.pdf
    ASR_VITS_Evaluation_Guide.pdf
```

`tmp/`、数据、虚拟环境、缓存、检查点及生成的音频样本保留在本地，由 Git 忽略。两份正式 PDF 纳入仓库；新增输出文件须单独审查并调整忽略规则。

## 协作约定

- 每个任务使用一个短期分支，`main` 保存经过审查的基线。
- 修改 Notebook 前协调负责人，避免多人通过网页同时替换整个文件。
- 每个 PR 如实记录执行过的验证及尚未验证的内容。
- 数据、模型权重及凭据不得进入 Git，Notebook 输出也要检查。
- 不假定 GitHub 分支保护已启用；即使网页允许直接提交，也应遵守团队约定。

详见[贡献指南](CONTRIBUTING.md)与[文档索引](docs/README.md)。

## 验证与复用

仓库整理仅检查 Notebook 结构和 Python 语法，不执行训练，也不能证明模型正确。数据访问、兼容的 PyTorch/TorchAudio 版本、独立评估集的评估方案及实际实验结果仍需团队确认。

目前尚未选择软件许可证。公开发布或复用前，请确认课程规定、代码来源以及第三方内容和数据条款；许可证由团队共同决定。
