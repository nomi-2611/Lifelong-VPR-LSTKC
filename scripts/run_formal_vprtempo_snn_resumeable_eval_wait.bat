@echo off
set TRAIN_DIR=C:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\formal_vprtempo_snn_3stage_resumeable_v1
set EVAL_DIR=C:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\formal_vprtempo_snn_3stage_resumeable_v1_eval
set VPR_MODEL=C:\Users\86189\Desktop\code\VPRTempo\vprtempo\models\springfall_VPRTempo_IN3136_FN6272_DB500.pth
set DATA_DIR=C:\Users\86189\Desktop\code\DATA

:wait_loop
if not exist "%TRAIN_DIR%\nordland_place_checkpoint.pth.tar" goto wait_more
if not exist "%TRAIN_DIR%\robotcar_place_checkpoint.pth.tar" goto wait_more
if not exist "%TRAIN_DIR%\tart_place_checkpoint.pth.tar" goto run_eval

:wait_more
timeout /t 300 /nobreak > nul
goto wait_loop

:run_eval
call D:\ProgramData\anaconda3\condabin\conda.bat activate IRL
cd /d C:\Users\86189\Desktop\code\AAAI2024-LSTKC
python continual_train.py ^
  --evaluate ^
  --test_folder "%TRAIN_DIR%" ^
  --logs-dir "%EVAL_DIR%" ^
  --data-dir "%DATA_DIR%" ^
  --task-type place ^
  --place-train-seq "nordland_place,robotcar_place,tart_place" ^
  --MODEL vprtempo_snn ^
  --vprtempo-model-path "%VPR_MODEL%" ^
  --place-preprocess-mode auto ^
  --vprtempo-freeze-mode frozen ^
  --vprtempo-raw-af-weight 0.5 ^
  --vprtempo-raw-alpha-weight 0.5 ^
  --batch-size 64
