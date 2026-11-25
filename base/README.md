# 模型基础模块

这里是 FYP-120 的两个 PyTorch 模块起点。两个模块都接收 `[批量, 通道, 高, 宽]` 的特征图，并保持形状。后续接入 YOLO11n 时，在模型配置中单独登记模块和输入通道。

| 文件 | 来源和用途 |
| --- | --- |
| `REEM.py` | YOLO11-FR 作者公开版本。`strip_h` 和 `strip_v` 是横向、纵向条带分支。计划先接入 P3，比较细裂缝检出与接缝误报。第三方代码的 MIT 许可证见 `LICENSE-REEM`。[作者代码](https://github.com/xiaoming988/YOLO11-FR/blob/2325312af0fce5adf08b3e29640822daebe2986d/ultralytics/nn/modules/REEM.py) |
| `context.py` | 参考 Wang 和 Tang（2024）图 6 编写的 `ContextBlock`，作为后续 P4 实验的起点。当前实现用膨胀率 1、3、5 提取周边信息，拼接后用 SE 做通道加权，再用 1×1 卷积恢复通道数。这些实现选择用于适配我们的小组实验。[论文原文](https://doi.org/10.1038/s41598-024-60454-3) |

## 怎么开始

本地已有仓库时，从 `main` 拉取最新代码，再从最新的 `main` 建自己的分支。下面以 REEM 为例，做 Context 时把分支名换成 `experiment/context-p4`。

```sh
git switch main
git pull --ff-only
git switch -c experiment/reem-p3
```

修改完成并提交后，用 `git push -u origin HEAD` 上传自己的分支。把分支链接发到群里，方便大家一起看改动。

验收范围为两个模块的独立前向、输出形状和反向传播。接入 YOLO11n 及正式对照训练属于后续实验。
