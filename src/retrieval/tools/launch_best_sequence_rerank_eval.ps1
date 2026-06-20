$ErrorActionPreference = "Stop"

$python = "D:\ProgramData\anaconda3\envs\IRL\python.exe"
$repo = Split-Path -Parent $PSScriptRoot
$dataDir = "C:\Users\86189\Desktop\code\DATA"
$manifest = Join-Path $repo "logs\abl2_reverse_rawtriplet_w0p5_tol1_e40_raw_eval_20260509\offline_eval\embedding_manifest_raw.json"
$outDir = Join-Path $repo "logs\best_sequence_rerank_eval_20260509"

New-Item -ItemType Directory -Force $outDir | Out-Null

& $python -u (Join-Path $repo "tools\eval_place_from_embeddings.py") `
  --data-dir $dataDir `
  --manifest $manifest `
  --dataset-names robotcar_place nordland_place `
  --sequence-rerank `
  --sequence-radius 120 `
  --sequence-weight 1.0 `
  --candidate-topk 1000 `
  --query-block-size 32 `
  --candidate-chunk-size 64 `
  --robotcar-eval-tolerance 1 `
  --nordland-eval-tolerance 0 `
  2>&1 | Tee-Object (Join-Path $outDir "log.txt")
