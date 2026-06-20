$runDir = "c:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\abl2_reverse_neighborpos1_trainable_e40_offline_20260508"
$offlineEvalDir = Join-Path $runDir "offline_eval"
$stdout = Join-Path $runDir "train_stdout.txt"
$stderr = Join-Path $runDir "train_stderr.txt"
$pidFile = Join-Path $runDir "train_pid.txt"
$python = "D:\ProgramData\anaconda3\envs\IRL\python.exe"
$workdir = "c:\Users\86189\Desktop\code\AAAI2024-LSTKC"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null
New-Item -ItemType Directory -Force -Path $offlineEvalDir | Out-Null

$args = @(
  "continual_train.py",
  "--data-dir", "c:\Users\86189\Desktop\code\DATA",
  "--logs-dir", $runDir,
  "--task-type", "place",
  "--place-train-seq", "robotcar_place,nordland_place",
  "--MODEL", "vprtempo_snn",
  "--vprtempo-model-path", "c:\Users\86189\Desktop\code\VPRTempo\vprtempo\models\springfall_VPRTempo_IN3136_FN6272_DB500.pth",
  "--place-preprocess-mode", "auto",
  "--place-train-positive-tolerance", "1",
  "--vprtempo-preprocess-cache",
  "--vprtempo-preprocess-cache-dir", "c:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\shared_vprtempo_preprocess_cache_20260508",
  "--vprtempo-freeze-mode", "trainable",
  "--vprtempo-stage-train-layers", "output_only",
  "--place-eval-backend", "vpr",
  "--place-offline-eval",
  "--place-offline-eval-dir", $offlineEvalDir,
  "--place-vpr-protocol", "dataset",
  "--batch-size", "128",
  "--workers", "4",
  "--eval-workers", "2",
  "--persistent-workers",
  "--prefetch-factor", "8",
  "--amp",
  "--cudnn-benchmark",
  "--tf32",
  "--print-freq", "50",
  "--tb-log-freq", "200",
  "--epochs0", "40",
  "--epochs", "40",
  "--eval_epoch", "40"
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
