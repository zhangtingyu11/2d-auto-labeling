param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [string]$Output = ".\artifacts\v1_1_predictions\predictions.jsonl",
    [double]$ScoreThreshold = 0.2,
    [double]$CrossClassNmsIou = 0.5,
    [double]$WaterTruckMinScore = 0.6,
    [double]$WaterTruckAmbiguityMargin = 0.15,
    [int]$BatchSize = 32,
    [int]$Limit = 0,
    [string]$Checkpoint = ""
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv-train2d\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\models\rtmdet\rtmdet_tiny_mining6_deploy_all_12e.py"
$Manifest = Join-Path $ProjectRoot "artifacts\release_v1_hashed\manifest\manifest.jsonl"
$Annotations = Join-Path $ProjectRoot "artifacts\release_v1_hashed\coco\annotations_all.json"
$V1Checkpoint = Join-Path $ProjectRoot "artifacts\training\rtmdet_tiny_mining6_40e\best_coco_bbox_mAP_epoch_38.pth"

if (-not $Checkpoint) {
    $DeployDir = Join-Path $ProjectRoot "artifacts\training\rtmdet_tiny_mining6_deploy_all_12e"
    $Best = Get-ChildItem -LiteralPath $DeployDir -Filter "best_coco_bbox_mAP_epoch_*.pth" -File `
        | Sort-Object LastWriteTime -Descending `
        | Select-Object -First 1
    if (-not $Best) {
        throw "V1.1 deployment checkpoint not found in: $DeployDir"
    }
    $Checkpoint = $Best.FullName
}

foreach ($Required in @(
    $Python, $Config, $Checkpoint, $Manifest, $Annotations, $V1Checkpoint, $DatasetRoot
)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required path does not exist: $Required"
    }
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:CAR5_DATA_ROOT = (Resolve-Path -LiteralPath $DatasetRoot).Path.Replace("\", "/") + "/"
$env:CAR5_COCO_ALL = (Resolve-Path -LiteralPath $Annotations).Path.Replace("\", "/")
$env:CAR5_V1_CHECKPOINT = (Resolve-Path -LiteralPath $V1Checkpoint).Path.Replace("\", "/")

$InferArgs = @(
    ".\tools\infer_rtmdet.py",
    $Config,
    (Resolve-Path -LiteralPath $Checkpoint).Path,
    $Manifest,
    (Resolve-Path -LiteralPath $DatasetRoot).Path,
    $Output,
    "--batch-size", $BatchSize,
    "--score-threshold", $ScoreThreshold,
    "--cross-class-nms-iou", $CrossClassNmsIou,
    "--watertruck-min-score", $WaterTruckMinScore,
    "--watertruck-ambiguity-margin", $WaterTruckAmbiguityMargin,
    "--model-version", "car5-rtmdet-tiny-v1.1-deploy-all",
    "--resume"
)
if ($Limit -gt 0) {
    $InferArgs += @("--limit", $Limit)
}
& $Python @InferArgs
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

$OutputPath = Resolve-Path -LiteralPath $Output
$LabelStudioOutput = $OutputPath.Path + ".labelstudio.json"
& $Python ".\tools\export_label_studio_predictions.py" `
    $OutputPath.Path `
    $Manifest `
    $LabelStudioOutput
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

Write-Host "Predictions: $($OutputPath.Path)"
Write-Host "Label Studio: $LabelStudioOutput"
