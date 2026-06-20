# Ablation queue for formal VPRTempo-SNN LSTKC experiments.
# Run these after the baseline in logs/formal_vprtempo_snn_3stage_raw05_stageadapt_bg4 finishes.
# All commands intentionally omit --place-train-limit.

$Root = "C:\Users\86189\Desktop\code"
$Project = "$Root\AAAI2024-LSTKC"
$Data = "$Root\DATA"
$BaseModel = "$Root\VPRTempo\vprtempo\models\springfall_VPRTempo_IN3136_FN6272_DB500.pth"
$Seq = "nordland_place,robotcar_place,tart_place"
Set-Location $Project

# 1) projected only: no raw VPRTempo affinity in AF/alpha.
conda run -n IRL python continual_train.py --data-dir $Data --logs-dir "$Project\logs\abl_projected_only" --task-type place --place-train-seq $Seq --MODEL vprtempo_snn --vprtempo-model-path $BaseModel --place-preprocess-mode auto --vprtempo-freeze-mode frozen --vprtempo-stage-adapt --vprtempo-stage-force-retrain --vprtempo-stage-output-dir "$Project\logs\abl_projected_only_stageadapt" --vprtempo-stage-train-layers output_only --vprtempo-raw-af-weight 0.0 --vprtempo-raw-alpha-weight 0.0 --batch-size 64 --eval_epoch 100

# 2) projected + raw AF only: raw affinity affects anti-forgetting, not adaptive alpha.
conda run -n IRL python continual_train.py --data-dir $Data --logs-dir "$Project\logs\abl_raw_af_only" --task-type place --place-train-seq $Seq --MODEL vprtempo_snn --vprtempo-model-path $BaseModel --place-preprocess-mode auto --vprtempo-freeze-mode frozen --vprtempo-stage-adapt --vprtempo-stage-force-retrain --vprtempo-stage-output-dir "$Project\logs\abl_raw_af_only_stageadapt" --vprtempo-stage-train-layers output_only --vprtempo-raw-af-weight 0.5 --vprtempo-raw-alpha-weight 0.0 --batch-size 64 --eval_epoch 100

# 3) stage-adapt off: same raw weights, frozen backbone, no VPRTempo per-stage adaptation.
conda run -n IRL python continual_train.py --data-dir $Data --logs-dir "$Project\logs\abl_stageadapt_off" --task-type place --place-train-seq $Seq --MODEL vprtempo_snn --vprtempo-model-path $BaseModel --place-preprocess-mode auto --vprtempo-freeze-mode frozen --vprtempo-stage-train-layers output_only --vprtempo-raw-af-weight 0.5 --vprtempo-raw-alpha-weight 0.5 --batch-size 64 --eval_epoch 100

# 4) trainable VPRTempo backbone: compare against frozen baseline.
conda run -n IRL python continual_train.py --data-dir $Data --logs-dir "$Project\logs\abl_trainable_backbone" --task-type place --place-train-seq $Seq --MODEL vprtempo_snn --vprtempo-model-path $BaseModel --place-preprocess-mode auto --vprtempo-freeze-mode trainable --vprtempo-stage-adapt --vprtempo-stage-force-retrain --vprtempo-stage-output-dir "$Project\logs\abl_trainable_backbone_stageadapt" --vprtempo-stage-train-layers output_only --vprtempo-raw-af-weight 0.5 --vprtempo-raw-alpha-weight 0.5 --batch-size 64 --eval_epoch 100

# 5) raw weight / alpha sample limit: examples; tune after baseline variance is known.
conda run -n IRL python continual_train.py --data-dir $Data --logs-dir "$Project\logs\abl_raw025_alpha512" --task-type place --place-train-seq $Seq --MODEL vprtempo_snn --vprtempo-model-path $BaseModel --place-preprocess-mode auto --vprtempo-freeze-mode frozen --vprtempo-stage-adapt --vprtempo-stage-force-retrain --vprtempo-stage-output-dir "$Project\logs\abl_raw025_alpha512_stageadapt" --vprtempo-stage-train-layers output_only --vprtempo-raw-af-weight 0.25 --vprtempo-raw-alpha-weight 0.25 --alpha-sample-limit 512 --batch-size 64 --eval_epoch 100
