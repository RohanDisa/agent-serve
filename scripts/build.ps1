# Build tinferno with MSVC 2022 x64.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not (Get-Command cmake -ErrorAction SilentlyContinue)) {
    python -m pip install cmake --quiet
}
cmake -B build -G "Visual Studio 17 2022" -A x64
if ($LASTEXITCODE -ne 0) { throw "cmake configure failed" }
cmake --build build --config Release -j
if ($LASTEXITCODE -ne 0) { throw "build failed" }
Write-Host "binaries in $root\build\Release"
