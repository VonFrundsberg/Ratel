param(
    [string]$VideoId = "OFT_11",
    [ValidateSet("cached", "video", "pose")]
    [string]$Mode = "cached"
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv/Scripts/python.exe"
$benchmarkRoot = Join-Path $projectRoot "output/open_field_benchmark"
$modelPath = Join-Path $projectRoot "output/native_benchmark/models/ratel_pose.joblib"
if (-not (Test-Path -LiteralPath $modelPath)) {
    $modelPath = Join-Path $projectRoot "output/native_benchmark/models/native_pose.joblib"
}
if (-not (Test-Path -LiteralPath $modelPath)) {
    throw "Run .venv/Scripts/python scripts/benchmark_ratel.py first to create the RATEL model."
}
$recordings = Get-Content -LiteralPath (Join-Path $benchmarkRoot "recordings.json") -Raw | ConvertFrom-Json
$recording = @($recordings | Where-Object { $_.video_id -eq $VideoId })
if ($recording.Count -ne 1) {
    throw "Unknown or duplicate video ID: $VideoId"
}
$recording = $recording[0]
$sourcePath = Join-Path $benchmarkRoot "tracks/opencv/$VideoId.csv"
$kind = "trajectory"
if ($Mode -eq "video") {
    $sourcePath = $recording.video_path
    $kind = "video"
}
if ($Mode -eq "pose") {
    $sourcePath = $recording.pose_path
    $kind = "pose"
}
$arguments = @("-m", "behaviour", "analyze", $sourcePath,
    "--input-kind", $kind, "--fps", "$($recording.fps)", "--video-id", $VideoId,
    "--config", (Join-Path $benchmarkRoot "configs/$VideoId.toml"),
    "--pose", $recording.pose_path, "--model", $modelPath,
    "--max-frames", "$($recording.video_header_frames)",
    "--output", (Join-Path $projectRoot "output/native_demo/$VideoId"))
& $pythonPath @arguments
if ($LASTEXITCODE -ne 0) {
    throw "RATEL analysis failed with exit code $LASTEXITCODE"
}
