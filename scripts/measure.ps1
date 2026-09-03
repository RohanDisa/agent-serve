# Ceilings first, then e2e if TINFERNO_MODEL is set.
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
$out = Join-Path $root "bench\results"
New-Item -ItemType Directory -Force -Path $out | Out-Null
$bin = Join-Path $root "build\Release"
if (-not (Test-Path (Join-Path $bin "stream.exe"))) { $bin = Join-Path $root "build" }
if (-not (Test-Path (Join-Path $bin "stream.exe"))) { throw "build first (scripts/build.ps1)" }

function Invoke-Utf8($exe, $jsonName, $argList) {
    $path = Join-Path $out $jsonName
    $argStr = ($argList | ForEach-Object { $_ }) -join " "
    cmd /c "`"$exe`" $argStr > `"$path`""
    Write-Host "wrote $path"
}

# STREAM ceiling is gcc/WSL2 read-only + MSVC comparison; do not overwrite with the MSVC-only binary.
python (Join-Path $root "scripts\run_stream.py")
Invoke-Utf8 (Join-Path $bin "bench_matmul.exe") "gemm.json" @("--threads", "6")

$model = $env:TINFERNO_MODEL
$e2e = Join-Path $bin "bench_e2e.exe"
if ($model -and (Test-Path $model) -and (Test-Path $e2e)) {
    foreach ($st in @("v0","v1","v2","v3","v4","v5")) {
        Invoke-Utf8 $e2e "e2e_$st.json" @("--model", "`"$model`"", "--stage", $st, "--n-predict", "64", "--runs", "10", "--warmup", "3")
    }
} else {
    Write-Host "set TINFERNO_MODEL to a GGUF path to run e2e stages"
}
python (Join-Path $root "bench\roofline.py") $out
