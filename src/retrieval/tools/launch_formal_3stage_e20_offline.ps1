$runDir = "c:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\formal_3stage_e20_bs128_vpr_offline_20260508"
$stageDir = "c:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\formal_3stage_e20_bs128_vpr_offline_20260508_stage_adapt"
$offlineEvalDir = Join-Path $runDir "offline_eval"
$stdout = Join-Path $runDir "train_stdout.txt"
$stderr = Join-Path $runDir "train_stderr.txt"
$pidFile = Join-Path $runDir "train_pid.txt"
$python = "D:\ProgramData\anaconda3\envs\IRL\python.exe"
$workdir = "c:\Users\86189\Desktop\code\AAAI2024-LSTKC"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
New-Item -ItemType Directory -Force -Path $stageDir | Out-Null
New-Item -ItemType Directory -Force -Path $offlineEvalDir | Out-Null

$args = @(
  "continual_train.py",
  "--data-dir", "c:\Users\86189\Desktop\code\DATA",
  "--logs-dir", $runDir,
  "--task-type", "place",
  "--place-train-seq", "nordland_place,robotcar_place,tart_place",
  "--MODEL", "vprtempo_snn",
  "--vprtempo-model-path", "c:\Users\86189\Desktop\code\VPRTempo\vprtempo\models\springfall_VPRTempo_IN3136_FN6272_DB500.pth",
  "--place-preprocess-mode", "auto",
  "--vprtempo-preprocess-cache",
  "--vprtempo-preprocess-cache-dir", "c:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\shared_vprtempo_preprocess_cache_20260508",
  "--vprtempo-freeze-mode", "frozen",
  "--vprtempo-stage-adapt",
  "--vprtempo-stage-force-retrain",
  "--vprtempo-stage-output-dir", $stageDir,
  "--vprtempo-stage-train-layers", "output_only",
  "--place-eval-backend", "vpr",
  "--place-offline-eval",
  "--place-offline-eval-dir", $offlineEvalDir,
  "--place-vpr-protocol", "dataset",
  "--tart-eval-pose-threshold", "3.0",
  "--batch-size", "128",
  "--workers", "4",
  "--persistent-workers",
  "--prefetch-factor", "8",
  "--amp",
  "--cudnn-benchmark",
  "--tf32",
  "--print-freq", "50",
  "--tb-log-freq", "200",
  "--epochs0", "20",
  "--epochs", "20",
  "--eval_epoch", "20"
)

$proc = Start-Process -FilePath $python `
  -ArgumentList $args `
  -WorkingDirectory $workdir `
  -RedirectStandardOutput $stdout `
  -RedirectStandardError $stderr `
  -WindowStyle Hidden `
  -PassThru

Set-Content -Path $pidFile -Value $proc.Id -Encoding ascii
Write-Output ("PID={0}" -f $proc.Id)
