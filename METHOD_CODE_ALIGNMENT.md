# MicroBT++ 方法与公开代码对应表

| 论文表述 | 公开代码位置 | 实现说明 |
| --- | --- | --- |
| Object field `O=(S, rho, T)` | `nnunetv2/nets/microbt_umctf/model.py` | 7-channel field：WT support、3-channel bounded transport、nested TC/ET probabilities、WT-gated object density。 |
| Lesion Mass Conservation (LMC) | `nnunetv2/training/loss/microbt_umctf_losses.py` | 有效 lesion 的积分质量约束到 1，并抑制背景质量。 |
| Anatomy-aligned Pushforward Decoding (APD) | `nnunetv2/training/loss/microbt_ctf_losses.py`; `nnunetv2/inference/microbt_umctf_decode.py` | 训练时执行 endpoint assignment consistency；推理时使用 trilinear splat、局部峰/NMS 与 endpoint assignment 解码。 |
| Support-to-Atom Continuation (SAC) | `nnunetv2/training/loss/microbt_sacl_losses.py`; `nnunetv2/training/nnUNetTrainer/MicroBTSACLTrainer.py` | 旧内部名称为 SACL。semantic support weight=1、measure weight=`0.25*lambda`、transport weight=`0.25*lambda^2`，endpoint epoch=29。 |
| 最终 trainer | `nnunetv2/training/nnUNetTrainer/MicroBTSACLTrainer.py` | `MicroBTUnitMassSACLFinal300Trainer`；300-epoch PolyLR horizon。 |
| 训练入口 | `scripts/run_microbt_umctf_sacl_c2_foldall_scratch300.sh` | 公开版使用显式本地配置，不包含原服务器或数据集定位信息。 |
| 续训入口 | `scripts/continue_microbt_umctf_sacl_c2_final300_foldall.sh` | 保留 epoch-200 / optimizer-state 校验。 |

APD 是受监督的确定性 pushforward/assignment 解码。SAC 和代码中的 SACL 是术语映射。公开版本经过脱敏；本地生成的校验清单只能检测之后的内容变化，不能替代原始实验来源记录；具体哈希记录不随公开仓库发布。
