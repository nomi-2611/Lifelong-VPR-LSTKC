# 4.4 长短期知识巩固消融实验记录

日期：2026-05-12

## 实验口径

- 训练顺序：`robotcar_place -> nordland_place -> pitts30k_place`
- 每阶段训练：20 epochs
- 评测：raw 单帧检索
- 未引入 VPRTempo 相似度融合
- 未引入序列一致性重排序
- 评价指标：mAP、R@1、R@5

## 脚本

- `run_thesis_4_4_place_ablation.ps1`

## 当前进度

已停止当前批跑进程。

### 已完成

1. `thesis_44_1_baseline_20260512`
   - 方法：普通顺序训练
   - `log_res.txt` 已有 mAP/R@1：
     - RobotCar：17.0 / 32.3
     - Nordland：33.1 / 42.7
     - Pittsburgh30k：79.7 / 79.1
   - 明天需要从 `log.txt` 抽取 R@5。

### 未完成

2. `thesis_44_2_short_term_20260512`
   - 方法：顺序训练 + 短期知识转移
   - 停止时已生成：
     - `robotcar_place_checkpoint.pth.tar`
     - `nordland_place_checkpoint.pth.tar`
   - 尚未完成 Pittsburgh30k 阶段和最终评测。
   - 建议明天重新完整跑该组，避免半成品续跑带来记录不清。

3. `thesis_44_3_long_term_20260512`
   - 方法：顺序训练 + 长期知识保持
   - 尚未开始。

4. `thesis_44_4_short_long_20260512`
   - 方法：顺序训练 + 长短期知识巩固
   - 尚未开始。

## 明天建议

为保证论文数据严谨，建议明天：

1. 保留 `thesis_44_1_baseline_20260512` 作为已完成结果。
2. 删除或另存 `thesis_44_2_short_term_20260512` 半成品目录。
3. 单独依次跑第 2、3、4 组实验。
4. 所有结果完成后，统一从各组 `log.txt` 抽取 mAP、R@1、R@5，填入表 4-5。
