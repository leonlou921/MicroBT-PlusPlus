# MicroBT++

MicroBT++，基于 nnU-Net v2，包含 **Lesion Mass Conservation (LMC)**、**Anatomy-aligned Pushforward Decoding (APD)** 和 **Support-to-Atom Continuation (SAC)**。

本仓库提供网络、损失、trainer、推理、评估及测试代码。

## 方法入口

| 方法 | 代码 |
| --- | --- |
| Object field `O=(S, rho, T)` | [Network](nnunetv2/nets/microbt_umctf/model.py) |
| Lesion Mass Conservation (LMC) | [Loss](nnunetv2/training/loss/microbt_umctf_losses.py) |
| Anatomy-aligned Pushforward Decoding (APD) | [Transport loss](nnunetv2/training/loss/microbt_ctf_losses.py) · [Decoder](nnunetv2/inference/microbt_umctf_decode.py) |
| Support-to-Atom Continuation (SAC) | [Continuation loss](nnunetv2/training/loss/microbt_sacl_losses.py) · [Trainer](nnunetv2/training/nnUNetTrainer/MicroBTSACLTrainer.py) |
| 300-epoch trainer | `MicroBTUnitMassSACLFinal300Trainer` |

## 安装与配置

使用 Python 3.10 或更高版本，并为目标设备安装兼容的 PyTorch / CUDA 环境。依赖清单根据源码 import 整理，未锁定版本；完整训练兼容性需要在目标环境验证。

```bash
git clone https://github.com/leonlou921/MicroBT-PlusPlus.git
cd MicroBT-PlusPlus
python -m pip install -r requirements.txt
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

请先按 [.env.example](.env.example) 配置自己的数据目录、数据集编号、缓存目录和解释器。文件内所有路径及数据标识均为示例。不要把填写了真实信息的配置提交到 GitHub。

```bash
cp .env.example .env
# 编辑 .env，将示例值替换成自己的配置
set -a
. ./.env
set +a
```

## 训练与评估模板

最终方法入口保留 `fold=all`、300 epochs、三区域标签合同及 continuation schedule。历史合同中固定的训练/测试数量和评估名称属于原代码约束，**不是通用数据集默认值**；在其他数据上运行前必须审核并适配相关校验和标签约定。

以下命令只打印训练与续训合同，无需数据或 GPU：

```bash
bash scripts/run_microbt_umctf_sacl_c2_foldall_scratch300.sh --print-contract
bash scripts/continue_microbt_umctf_sacl_c2_final300_foldall.sh --print-contract
```

配置好自己的环境、数据和预处理产物后，可使用最终方法训练模板：

```bash
python scripts/build_umctf_sacl_c2_source_manifest.py write
python scripts/build_umctf_sacl_c2_source_manifest.py verify
bash scripts/run_microbt_umctf_sacl_c2_foldall_scratch300.sh
```

训练脚本从当前仓库位置读取源码，要求显式提供本地数据配置，并拒绝覆盖已存在的输出目录。续训入口仍要求 epoch-200 且 optimizer state 符合原协议的检查点；不能用于任意检查点恢复。

原评估模板位于 [run_microbt_umctf_test251.sh](scripts/run_microbt_umctf_test251.sh)。它保留固定测试数量、模态顺序和目录合同；病例示例已改成合成名称。请按自己的数据格式适配解析器，不能直接据此声明原实验指标。其他历史脚本中的 `/path/to/example` 是需要手动配置的占位符，不能直接当成有效路径使用。

## 仓库结构

```text
nnunetv2/               框架、MicroBT++ 模块及原有测试
scripts/                最终方法入口与已脱敏的历史实验模板
tests/                  不依赖训练库的公开配置测试
artifacts/              本地生成的校验记录，不随仓库发布
.env.example            不含真实环境信息的配置示例
```

## 来源与授权状态

包含 nnU-Net v2 框架源码，原项目见 [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet)。
