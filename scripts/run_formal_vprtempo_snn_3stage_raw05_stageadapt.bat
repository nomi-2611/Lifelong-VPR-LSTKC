@echo off
call D:\ProgramData\anaconda3\condabin\conda.bat activate IRL
cd /d C:\Users\86189\Desktop\code\AAAI2024-LSTKC
python continual_train.py ^
  --data-dir "C:\Users\86189\Desktop\code\DATA" ^
  --logs-dir "C:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\formal_vprtempo_snn_3stage_raw05_stageadapt_bg4" ^
  --task-type place ^
  --place-train-seq "nordland_place,robotcar_place,tart_place" ^
  --MODEL vprtempo_snn ^
  --vprtempo-model-path "C:\Users\86189\Desktop\code\VPRTempo\vprtempo\models\springfall_VPRTempo_IN3136_FN6272_DB500.pth" ^
  --place-preprocess-mode auto ^
  --vprtempo-freeze-mode frozen ^
  --vprtempo-stage-adapt ^
  --vprtempo-stage-force-retrain ^
  --vprtempo-stage-output-dir "C:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\formal_vprtempo_stage_adapt_raw05_bg4" ^
  --vprtempo-stage-train-layers output_only ^
  --vprtempo-raw-af-weight 0.5 ^
  --vprtempo-raw-alpha-weight 0.5 ^
  --batch-size 64 ^
  --eval_epoch 100
