# 终身 VPR 最佳实验记录（2026-05-10）

## 当前可报告最佳结果

实验口径：LSTKC 终身训练框架 + VPRTempo/SNN raw embedding + stage expert long-term memory pool；RobotCar 与 Nordland 使用序列一致性 rerank，Pitts30k 使用 25m VPR 半径协议。

| 数据集 | 协议 | mAP | R@1 | R@5 | R@10 |
|---|---|---:|---:|---:|---:|
| RobotCar | stage expert + sequence rerank, r300/w1.0 | 90.9% | 91.4% | 91.5% | 91.5% |
| Nordland | stage expert + sequence rerank, r300/w1.0 | 82.5% | 81.3% | 87.1% | 87.1% |
| Pittsburgh30k | stage expert + single-frame 25m VPR protocol | 79.7% | 79.3% | 99.8% | 100.0% |
| Average | mixed VPR protocol | 84.3% | 84.0% | 92.8% | 92.9% |

## 结果文件

- 汇总 JSON：`logs/stage_expert_pool_tune3_20260510/mixed_protocol_best_summary_20260510.json`
- Sequence rerank 最终 sweep：`logs/stage_expert_pool_tune3_20260510/sequence_rerank_sweep_final_top500.json`
- Stage expert pool 单帧结果：`logs/stage_expert_pool_tune3_20260510/stage_expert_pool_summary.json`

## 论文表述注意

- 这组最佳结果不是“纯单帧最终模型”结果，而是 VPR 风格最终系统结果。
- 需要单独报告纯单帧 stage expert pool 结果：平均 mAP 52.2%，R@1 54.8%，R@5 72.5%，R@10 76.2%。
- Sequence rerank 是针对序列型 VPR 数据集的后处理/协议增强，不能说成模型本体端到端准确率提升。
- 可以把 stage expert pool 表述为长期知识记忆池，把 sequence rerank 表述为符合序列 VPR 场景的时序一致性检索策略。
