Set-Location "C:\Users\86189\Desktop\code\AAAI2024-LSTKC"
[Console]::OutputEncoding=[System.Text.Encoding]::UTF8

$Data = "C:\Users\86189\Desktop\code\DATA"
$BaseModel = "C:\Users\86189\Desktop\code\VPRTempo\vprtempo\models\springfall_VPRTempo_IN3136_FN6272_DB500.pth"
$CacheDir = "C:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\shared_vprtempo_preprocess_cache_20260508"
$Python = "D:\ProgramData\anaconda3\envs\IRL\python.exe"
$Seq = "robotcar_place,nordland_place,pitts30k_place"

function Invoke-PlaceRun {
    param(
        [string]$LogsDir,
        [string[]]$ExtraArgs
    )
    $ArgsList = @(
        "continual_train.py",
        "--data-dir", $Data,
        "--logs-dir", $LogsDir,
        "--task-type", "place",
        "--place-train-seq", $Seq,
        "--MODEL", "vprtempo_snn",
        "--vprtempo-model-path", $BaseModel,
        "--place-preprocess-mode", "auto",
        "--vprtempo-preprocess-cache",
        "--vprtempo-preprocess-cache-dir", $CacheDir,
        "--place-offline-eval",
        "--place-offline-eval-feature", "projected",
        "--place-vpr-protocol", "dataset",
        "--disable-stage-model-fusion",
        "--batch-size", "128",
        "--epochs", "20",
        "--epochs0", "20",
        "--eval_epoch", "20",
        "--eval-workers", "2",
        "--workers", "4",
        "--print-freq", "100"
    ) + $ExtraArgs
    Write-Host "==== Running $LogsDir ===="
    & $Python @ArgsList
}

Invoke-PlaceRun -LogsDir "C:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\thesis_44_projected_effect_3_long_20260514b" -ExtraArgs @(
    "--AF_weight", "0.0",
    "--place-teacher-sim-weight", "0.0",
    "--place-raw-distill-weight", "0.0",
    "--place-projected-distill-weight", "2.0",
    "--place-projected-distill-freq", "1",
    "--place-raw-distill-start-stage", "2",
    "--place-raw-memory-all-previous",
    "--place-raw-memory-size", "2048",
    "--place-raw-memory-batch-size", "64"
)

Invoke-PlaceRun -LogsDir "C:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\thesis_44_projected_effect_4_short_long_20260514b" -ExtraArgs @(
    "--AF_weight", "1.0",
    "--place-teacher-sim-weight", "0.5",
    "--place-teacher-sim-self",
    "--place-teacher-sim-limit", "0",
    "--place-teacher-sim-temp", "0.07",
    "--place-raw-distill-weight", "0.0",
    "--place-projected-distill-weight", "2.0",
    "--place-projected-distill-freq", "1",
    "--place-raw-distill-start-stage", "2",
    "--place-raw-memory-all-previous",
    "--place-raw-memory-size", "2048",
    "--place-raw-memory-batch-size", "64"
)
