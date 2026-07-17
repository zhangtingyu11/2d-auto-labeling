param(
    [string]$DatasetRoot = "C:\Users\zhesh\Desktop\vla2\car5_20260611_day_02\car5_20260611_day_02",
    [string]$WorkDir = ".\artifacts\training\rtmdet_tiny_mining6_deploy_all_12e"
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ProjectRoot

$Python = Join-Path $ProjectRoot ".venv-train2d\Scripts\python.exe"
$TrainScript = Join-Path $ProjectRoot "vendor\mmdetection-3.2.0\tools\train.py"
$Config = Join-Path $ProjectRoot "configs\models\rtmdet\rtmdet_tiny_mining6_deploy_all_12e.py"
$Annotations = Join-Path $ProjectRoot "artifacts\release_v1_hashed\coco\annotations_all.json"
$V1Checkpoint = Join-Path $ProjectRoot "artifacts\training\rtmdet_tiny_mining6_40e\best_coco_bbox_mAP_epoch_38.pth"

foreach ($Required in @($Python, $TrainScript, $Config, $Annotations, $V1Checkpoint, $DatasetRoot)) {
    if (-not (Test-Path -LiteralPath $Required)) {
        throw "Required path does not exist: $Required"
    }
}

$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONPATH = (Join-Path $ProjectRoot "src") + ";" + (Join-Path $ProjectRoot "vendor\mmdetection-3.2.0")
$env:CAR5_DATA_ROOT = (Resolve-Path -LiteralPath $DatasetRoot).Path.Replace("\", "/") + "/"
$env:CAR5_COCO_ALL = (Resolve-Path -LiteralPath $Annotations).Path.Replace("\", "/")
$env:CAR5_V1_CHECKPOINT = (Resolve-Path -LiteralPath $V1Checkpoint).Path.Replace("\", "/")

& $Python $TrainScript $Config --work-dir $WorkDir
exit $LASTEXITCODE
