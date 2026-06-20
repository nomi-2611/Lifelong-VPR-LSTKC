# Projected 口径下的遗忘与长短期知识巩固结果（2026-05-30）

## 评测说明

本次只做补评测，不重新训练。评测口径统一为 `projected` 图像描述子，未引入序列一致性重排序。

- RobotCar 初次学习后：`logs/projected_initial_eval_robotcar_20260530`
- Nordland 初次学习后：`logs/projected_initial_eval_nordland_20260530`
- 最终阶段无知识约束：`logs/thesis_44_projected_effect_1_baseline_20260513`
- 最终阶段长短期知识巩固：`logs/thesis_44_projected_effect_4_short_long_light_retry_20260515`

## 统一 projected 口径结果

| 数据集 | 阶段/方法 | mAP/% | R@1/% | R@5/% |
|---|---|---:|---:|---:|
| RobotCar | 初次学习后 | 16.8 | 27.2 | 47.6 |
| RobotCar | 最终阶段无知识约束 | 9.4 | 19.4 | 34.6 |
| RobotCar | 最终阶段长短期知识巩固 | 10.9 | 21.7 | 38.6 |
| Nordland | 初次学习后 | 32.6 | 39.9 | 53.2 |
| Nordland | 最终阶段无知识约束 | 20.6 | 27.1 | 38.7 |
| Nordland | 最终阶段长短期知识巩固 | 23.5 | 31.1 | 43.5 |

## 可计算结论

| 数据集 | 遗忘下降 mAP/% | 长短期巩固提升 mAP/% |
|---|---:|---:|
| RobotCar | 7.4 | 1.5 |
| Nordland | 12.0 | 2.9 |

遗忘下降 = 初次学习后 mAP - 最终阶段无知识约束 mAP。  
长短期巩固提升 = 最终阶段长短期知识巩固 mAP - 最终阶段无知识约束 mAP。

## ???????2 Nordland ??? RobotCar?2026-06-01?

????????4.3????2?Nordland ??? RobotCar?? `projected` ?? R@1/R@5?

- ?? checkpoint?`logs/thesis_44_projected_effect_1_baseline_20260513/nordland_place_checkpoint.pth.tar`
- ?????`logs/projected_stage2_nordland_eval_robotcar_20260601`
- ?????`projected`
- ???????????

| ???? | ????? | mAP/% | R@1/% | R@5/% |
|---|---|---:|---:|---:|
| ??2?Nordland | RobotCar | 16.0 | 26.7 | 47.0 |
