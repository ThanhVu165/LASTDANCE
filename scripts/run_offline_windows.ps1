param(
    [Parameter(Mandatory = $true)]
    [string]$Module,
    [string[]]$PythonArguments = @(),
    [string]$EnvironmentPath = ".venv-offline"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$environmentPrefix = [IO.Path]::GetFullPath((Join-Path $repoRoot $EnvironmentPath))
$python = Join-Path $environmentPrefix "python.exe"
$binaryDirectory = Join-Path $environmentPrefix "Library\bin"
$ffmpeg = Join-Path $binaryDirectory "ffmpeg.exe"
$ffprobe = Join-Path $binaryDirectory "ffprobe.exe"

if (-not (Test-Path -LiteralPath $python -PathType Leaf)) {
    throw "Offline Python environment not found: $python"
}
if (-not (Test-Path -LiteralPath $ffmpeg -PathType Leaf)) {
    throw "FFmpeg not found in the offline environment: $ffmpeg"
}
if (-not (Test-Path -LiteralPath $ffprobe -PathType Leaf)) {
    throw "ffprobe not found in the offline environment: $ffprobe"
}

# ffmpeg-python invokes the executable name, while our own pipeline accepts explicit
# paths. Set both forms for this child process without changing the user's system PATH.
$env:PATH = "$environmentPrefix;$environmentPrefix\Scripts;$binaryDirectory;$env:PATH"
if ([string]::IsNullOrWhiteSpace($env:AIC_FFMPEG)) {
    $env:AIC_FFMPEG = $ffmpeg
}
if ([string]::IsNullOrWhiteSpace($env:AIC_FFPROBE)) {
    $env:AIC_FFPROBE = $ffprobe
}

Push-Location $repoRoot
try {
    & $python -m $Module @PythonArguments
    $moduleExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($moduleExitCode -ne 0) {
    throw "Python module exited with code ${moduleExitCode}: $Module"
}
