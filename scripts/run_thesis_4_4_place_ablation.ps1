Set-Location "C:\Users\86189\Desktop\code\AAAI2024-LSTKC"

$Data = "C:\Users\86189\Desktop\code\DATA"
$BaseModel = "C:\Users\86189\Desktop\code\VPRTempo\vprtempo\models\springfall_VPRTempo_IN3136_FN6272_DB500.pth"
$Seq = "robotcar_place,nordland_place,pitts30k_place"
$Python = "D:\ProgramData\anaconda3\envs\IRL\python.exe"

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
        "--vprtempo-preprocess-cache-dir", "C:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\shared_vprtempo_preprocess_cache_20260508",
        "--place-offline-eval",
        "--place-offline-eval-feature", "raw",
        "--place-vpr-protocol", "dataset",
        "--batch-size", "128",
        "--epochs", "20",
        "--epochs0", "20",
        "--eval_epoch", "20",
        "--eval-workers", "2",
        "--workers", "4",
        "--print-freq", "100"
    ) + $ExtraArgs

    & $Python @ArgsList
}

# 1. Plain sequential training.
Invoke-PlaceRun -LogsDir "C:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\thesis_44_1_baseline_20260512" -ExtraArgs @(
    "--place-raw-distill-weight", "0.0",
    "--place-teacher-sim-weight", "0.0",
    "--place-raw-memory-size", "0"
)

# 2. Sequential training with short-term transfer.
Invoke-PlaceRun -LogsDir "C:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\thesis_44_2_short_term_20260512" -ExtraArgs @(
    "--place-teacher-sim-weight", "0.2",
    "--place-teacher-sim-self",
    "--place-teacher-sim-limit", "0",
    "--place-teacher-sim-temp", "0.07",
    "--place-raw-distill-weight", "0.0",
    "--place-raw-memory-size", "0"
)

# 3. Sequential training with long-term knowledge retention.
Invoke-PlaceRun -LogsDir "C:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\thesis_44_3_long_term_20260512" -ExtraArgs @(
    "--place-raw-distill-weight", "2.0",
    "--place-raw-distill-freq", "4",
    "--place-raw-distill-start-stage", "3",
    "--place-raw-memory-all-previous",
    "--place-raw-memory-size", "1024",
    "--place-raw-memory-batch-size", "64",
    "--place-teacher-sim-weight", "0.0"
)

# 4. Sequential training with short- and long-term consolidation.
Invoke-PlaceRun -LogsDir "C:\Users\86189\Desktop\code\AAAI2024-LSTKC\logs\thesis_44_4_short_long_20260512" -ExtraArgs @(
    "--place-teacher-sim-weight", "0.2",
    "--place-teacher-sim-self",
    "--place-teacher-sim-limit", "0",
    "--place-teacher-sim-temp", "0.07",
    "--place-raw-distill-weight", "2.0",
    "--place-raw-distill-freq", "4",
    "--place-raw-distill-start-stage", "3",
    "--place-raw-memory-all-previous",
    "--place-raw-memory-size", "1024",
    "--place-raw-memory-batch-size", "64"
)
