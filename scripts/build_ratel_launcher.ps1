$ErrorActionPreference = 'Stop'
$compiler = Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'
$source = Join-Path $PSScriptRoot 'RatelLauncher.cs'
$destination = Join-Path (Split-Path -Parent $PSScriptRoot) 'RATEL.exe'
& $compiler /nologo /target:winexe /platform:anycpu /optimize+ /reference:System.Windows.Forms.dll /reference:System.Drawing.dll "/out:$destination" $source
if ($LASTEXITCODE -ne 0) { throw 'RATEL launcher compilation failed.' }
Write-Output $destination
