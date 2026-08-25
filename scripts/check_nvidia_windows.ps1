param(
    [version]$MinimumDriverVersion = [version]"528.33"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
if ($null -eq $nvidiaSmi) {
    throw "nvidia-smi.exe was not found. Install/update the NVIDIA Windows driver first."
}

$rows = & $nvidiaSmi.Source `
    --query-gpu=name,memory.total,driver_version `
    --format=csv,noheader,nounits
if ($LASTEXITCODE -ne 0 -or -not $rows) {
    throw "nvidia-smi could not query an NVIDIA GPU."
}

$first = [string](@($rows)[0])
$parts = $first.Split(",") | ForEach-Object { $_.Trim() }
if ($parts.Count -ne 3) {
    throw "Unexpected nvidia-smi output: $first"
}

$gpuName = $parts[0]
$memoryMiB = [int]$parts[1]
$driverVersion = [version]$parts[2]
if ($driverVersion -lt $MinimumDriverVersion) {
    throw (
        "NVIDIA driver $driverVersion is too old for the CUDA 12.x worker profile. " +
        "Required >= $MinimumDriverVersion; update the driver and reboot first."
    )
}

Write-Host "[PASS] NVIDIA GPU: $gpuName"
Write-Host "[PASS] VRAM: $memoryMiB MiB"
Write-Host "[PASS] Driver: $driverVersion (required >= $MinimumDriverVersion)"
