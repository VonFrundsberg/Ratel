param(
    [string]$ReportDate = "2026-10-18",
    [switch]$SkipAssets
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv/Scripts/python.exe"
$sourceRoot = Join-Path $projectRoot "reports/native_analyzer"
$buildRoot = Join-Path $projectRoot "tmp/pdfs/native_tex"
$pdfRoot = Join-Path $projectRoot "output/pdf"
New-Item -ItemType Directory -Force -Path $buildRoot, $pdfRoot | Out-Null

if (-not $SkipAssets) {
    & $pythonPath (Join-Path $PSScriptRoot "ratel_tex.py") --report-date $ReportDate
    if ($LASTEXITCODE -ne 0) {
        throw "Generating RATEL TeX assets failed with exit code $LASTEXITCODE"
    }
}

$previousData = $env:MIKTEX_USERDATA
$previousConfig = $env:MIKTEX_USERCONFIG
$previousInstall = $env:MIKTEX_USERINSTALL
try {
    # Process-local user caches fit the restricted Windows build environment.
    $miktexProfileRoot = [Environment]::GetFolderPath("UserProfile")
    $env:MIKTEX_USERDATA = Join-Path $miktexProfileRoot "AppData/Local/MiKTeX"
    $env:MIKTEX_USERCONFIG = Join-Path $miktexProfileRoot "AppData/Local/MiKTeX"
    $env:MIKTEX_USERINSTALL = Join-Path $miktexProfileRoot "AppData/Roaming/MiKTeX"
    Push-Location $sourceRoot
    try {
        foreach ($pass in 1..2) {
            $lines = & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "-output-directory=$buildRoot" report.tex 2>&1
            $lines | Set-Content -LiteralPath (Join-Path $buildRoot "pass_$pass.txt")
            if ($LASTEXITCODE -ne 0) {
                $lines | Select-Object -Last 35 | Write-Output
                throw "LaTeX pass $pass failed; inspect $buildRoot/pass_$pass.txt"
            }
        }
    }
    finally {
        Pop-Location
    }
}
finally {
    $env:MIKTEX_USERDATA = $previousData
    $env:MIKTEX_USERCONFIG = $previousConfig
    $env:MIKTEX_USERINSTALL = $previousInstall
}

$targetPdf = Join-Path $pdfRoot "ratel_report.pdf"
Copy-Item -LiteralPath (Join-Path $buildRoot "report.pdf") -Destination $targetPdf -Force
Copy-Item -LiteralPath $targetPdf -Destination (Join-Path $pdfRoot "native_analyzer_report.pdf") -Force
$bundleFiles = @("report.tex", "generated", "BUILD.md", "workflow.md", "tex_manifest.json", "branding_manifest.json") |
    ForEach-Object { Join-Path $sourceRoot $_ }
Compress-Archive -LiteralPath $bundleFiles -DestinationPath (Join-Path $pdfRoot "ratel_tex.zip") -Force
Copy-Item -LiteralPath (Join-Path $pdfRoot "ratel_tex.zip") -Destination (Join-Path $pdfRoot "native_analyzer_tex.zip") -Force
Write-Output "PDF: $targetPdf"
Write-Output "TeX bundle: $(Join-Path $pdfRoot 'ratel_tex.zip')"
