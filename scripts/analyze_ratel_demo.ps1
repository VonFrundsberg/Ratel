param(
    [string]$VideoId = "OFT_11",
    [ValidateSet("cached", "video", "pose")]
    [string]$Mode = "cached"
)

$ErrorActionPreference = "Stop"
& (Join-Path $PSScriptRoot "analyze_native_demo.ps1") @PSBoundParameters
