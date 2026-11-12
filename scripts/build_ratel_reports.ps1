param(
    [string]$ReportDate = "2026-10-18",
    [switch]$SkipAssets
)

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv/Scripts/python.exe"
$pdfRoot = Join-Path $projectRoot "output/pdf"
& (Join-Path $PSScriptRoot "build_ratel_tex.ps1") @PSBoundParameters
if (-not $SkipAssets) {
    & $pythonPath (Join-Path $PSScriptRoot "open_field_tex.py")
    if ($LASTEXITCODE -ne 0) {
        throw "Generating RATEL open-field report assets failed"
    }
}

$previousData = $env:MIKTEX_USERDATA
$previousConfig = $env:MIKTEX_USERCONFIG
$previousInstall = $env:MIKTEX_USERINSTALL
try {
    $miktexProfileRoot = [Environment]::GetFolderPath("UserProfile")
    $env:MIKTEX_USERDATA = Join-Path $miktexProfileRoot "AppData/Local/MiKTeX"
    $env:MIKTEX_USERCONFIG = Join-Path $miktexProfileRoot "AppData/Local/MiKTeX"
    $env:MIKTEX_USERINSTALL = Join-Path $miktexProfileRoot "AppData/Roaming/MiKTeX"
    foreach ($reportName in @("open_field_expansion", "zenodo_pilot")) {
        $sourceRoot = Join-Path $projectRoot "reports/$reportName"
        $buildRoot = Join-Path $projectRoot "tmp/pdfs/ratel_$reportName"
        New-Item -ItemType Directory -Force -Path $buildRoot | Out-Null
        Push-Location $sourceRoot
        try {
            foreach ($pass in 1..2) {
                $lines = & pdflatex -interaction=nonstopmode -halt-on-error -file-line-error "-output-directory=$buildRoot" report.tex 2>&1
                $lines | Set-Content -LiteralPath (Join-Path $buildRoot "pass_$pass.txt")
                if ($LASTEXITCODE -ne 0) {
                    $lines | Select-Object -Last 30 | Write-Output
                    throw "RATEL $reportName LaTeX pass $pass failed"
                }
            }
        }
        finally {
            Pop-Location
        }
        $stem = if ($reportName -eq "open_field_expansion") { "ratel_open_field" } else { "ratel_pilot" }
        $targetPdf = Join-Path $pdfRoot "$($stem)_report.pdf"
        $targetZip = Join-Path $pdfRoot "$($stem)_tex.zip"
        Copy-Item -LiteralPath (Join-Path $buildRoot "report.pdf") -Destination $targetPdf -Force
        $bundleFiles = @("report.tex", "generated", "BUILD.md", "branding_manifest.json") |
            ForEach-Object { Join-Path $sourceRoot $_ } | Where-Object { Test-Path -LiteralPath $_ }
        Compress-Archive -LiteralPath $bundleFiles -DestinationPath $targetZip -Force
        if ($reportName -eq "open_field_expansion") {
            Copy-Item -LiteralPath $targetPdf -Destination (Join-Path $pdfRoot "open_field_benchmark_report.pdf") -Force
            Copy-Item -LiteralPath $targetZip -Destination (Join-Path $pdfRoot "open_field_benchmark_tex.zip") -Force
        }
        else {
            Copy-Item -LiteralPath $targetPdf -Destination (Join-Path $sourceRoot "report.pdf") -Force
        }
        Write-Output "PDF: $targetPdf"
        Write-Output "TeX bundle: $targetZip"
    }
}
finally {
    $env:MIKTEX_USERDATA = $previousData
    $env:MIKTEX_USERCONFIG = $previousConfig
    $env:MIKTEX_USERINSTALL = $previousInstall
}
