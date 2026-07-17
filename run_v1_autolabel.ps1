param(
    [Parameter(Mandatory = $true)]
    [string]$DatasetRoot,
    [string]$Output = ".\artifacts\v1_predictions\predictions.jsonl",
    [double]$ScoreThreshold = 0.1,
    [int]$BatchSize = 32,
    [int]$Limit = 0
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv-train2d\Scripts\python.exe"
$Config = Join-Path $ProjectRoot "configs\models\rtmdet\rtmdet_tiny_mining6_40e.py"
$Checkpoint = Join-Path $ProjectRoot "artifacts\training\rtmdet_tiny_mining6_40e\best_coco_bbox_mAP_epoch_38.pth"
$Manifest = Join-Path $ProjectRoot "artifacts\release_v1_hashed\manifest\manifest.jsonl"
$FoldRoot = Join-Path $ProjectRoot "artifacts\release_v1_hashed\coco\fold_0"

foreach ($Required in @($Python, $Config, $Checkpoint, $Manifest, $FoldRoot, $DatasetRoot)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required path does not exist: $Required"
    }
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = Join-Path $ProjectRoot "src"
$env:CAR5_DATA_ROOT = (Resolve-Path -LiteralPath $DatasetRoot).Path.Replace("\", "/") + "/"
$env:CAR5_COCO_FOLD_ROOT = (Resolve-Path -LiteralPath $FoldRoot).Path.Replace("\", "/") + "/"

$InferArgs = @(
    ".\tools\infer_rtmdet.py",
    $Config,
    $Checkpoint,
    $Manifest,
    (Resolve-Path -LiteralPath $DatasetRoot).Path,
    $Output,
    "--batch-size", $BatchSize,
    "--score-threshold", $ScoreThreshold,
    "--model-version", "car5-rtmdet-tiny-v1",
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
