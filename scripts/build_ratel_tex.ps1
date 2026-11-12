param(
    [string]$ReportDate = "2026-10-18",
    [switch]$SkipAssets
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "build_native_tex.ps1") @PSBoundParameters
