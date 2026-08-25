param(
    [string]$EnvironmentPath = ".venv-offline",
    [string]$ToolchainRoot = "",
    [ValidateSet("dev", "offline-local", "shot-windows-gpu")]
    [string]$Profile = "offline-local"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($Profile -eq "shot-windows-gpu" -and -not $PSBoundParameters.ContainsKey("EnvironmentPath")) {
    $EnvironmentPath = ".venv-shot-gpu"
}

# Pin both the installer and its digest so a new machine cannot silently receive a
# different bootstrap runtime. Miniforge itself must use a path without spaces because
# its upstream Windows installer does not reliably support paths such as this repo path.
$miniforgeVersion = "26.3.2-3"
$installerName = "Miniforge3-$miniforgeVersion-Windows-x86_64.exe"
$installerUri = "https://github.com/conda-forge/miniforge/releases/download/$miniforgeVersion/$installerName"
$expectedInstallerSha256 = "14a8635465b5190537ddad6286746ffebbc55a1ed2a7bb14a506595fe3191e1e"

$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$repoPrefix = $repoRoot + [IO.Path]::DirectorySeparatorChar
$toolsRoot = [IO.Path]::GetFullPath((Join-Path $repoRoot ".tools"))
$downloadRoot = [IO.Path]::GetFullPath((Join-Path $toolsRoot "downloads"))
$environmentPrefix = [IO.Path]::GetFullPath((Join-Path $repoRoot $EnvironmentPath))
$environmentFile = switch ($Profile) {
    "dev" { Join-Path $repoRoot "environment.dev.yml" }
    "shot-windows-gpu" { Join-Path $repoRoot "environment.shot-windows-gpu.yml" }
    default { Join-Path $repoRoot "environment.yml" }
}
$installerPath = Join-Path $downloadRoot $installerName

if ($Profile -eq "shot-windows-gpu") {
    & (Join-Path $PSScriptRoot "check_nvidia_windows.ps1")
}

if ([string]::IsNullOrWhiteSpace($ToolchainRoot)) {
    $localAppData = [Environment]::GetFolderPath([Environment+SpecialFolder]::LocalApplicationData)
    if ([string]::IsNullOrWhiteSpace($localAppData)) {
        throw "Cannot resolve LocalApplicationData; pass -ToolchainRoot with a short ASCII path."
    }
    $ToolchainRoot = Join-Path $localAppData "LASTDANCE\toolchains"
}
$miniforgePrefix = [IO.Path]::GetFullPath((Join-Path $ToolchainRoot "miniforge-$miniforgeVersion"))

foreach ($path in @($toolsRoot, $downloadRoot, $environmentPrefix)) {
    if (-not $path.StartsWith($repoPrefix, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Bootstrap target must stay inside the repository: $path"
    }
}
if (($miniforgePrefix -match "\s") -or ($miniforgePrefix -notmatch "^[\x20-\x7E]+$")) {
    throw "Miniforge requires a short ASCII path without spaces: $miniforgePrefix"
}

New-Item -ItemType Directory -Force -Path $downloadRoot | Out-Null
New-Item -ItemType Directory -Force -Path (Split-Path $miniforgePrefix -Parent) | Out-Null
$condaCommand = Join-Path $miniforgePrefix "condabin\conda.bat"

if (-not (Test-Path -LiteralPath $condaCommand -PathType Leaf)) {
    if ((Test-Path -LiteralPath $miniforgePrefix) -and
        (Get-ChildItem -LiteralPath $miniforgePrefix -Force | Select-Object -First 1)) {
        throw "Partial/non-Miniforge directory already exists: $miniforgePrefix"
    }

    if (-not (Test-Path -LiteralPath $installerPath -PathType Leaf)) {
        Write-Host "Downloading pinned Miniforge $miniforgeVersion..."
        Invoke-WebRequest -Uri $installerUri -OutFile $installerPath -UseBasicParsing
    }

    $actualSha256 = (Get-FileHash -LiteralPath $installerPath -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actualSha256 -ne $expectedInstallerSha256) {
        throw "Miniforge checksum mismatch. Expected $expectedInstallerSha256, got $actualSha256"
    }

    Write-Host "Installing the pinned Miniforge base toolchain..."
    $installerArguments = @(
        "/S",
        "/InstallationType=JustMe",
        "/RegisterPython=0",
        "/AddToPath=0",
        "/D=$miniforgePrefix"
    )
    $process = Start-Process -FilePath $installerPath -ArgumentList $installerArguments `
        -Wait -PassThru -WindowStyle Hidden
    if ($process.ExitCode -ne 0) {
        throw "Miniforge installer exited with code $($process.ExitCode)"
    }
}

if (-not (Test-Path -LiteralPath $condaCommand -PathType Leaf)) {
    throw "Conda was not found after bootstrap: $condaCommand"
}

function Invoke-CondaChecked {
    param([Parameter(Mandatory = $true)][string[]]$Arguments)

    & $condaCommand @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Conda command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

Push-Location $repoRoot
try {
    if (Test-Path -LiteralPath (Join-Path $environmentPrefix "python.exe") -PathType Leaf) {
        Write-Host "Updating the existing repo-local environment from $environmentFile..."
        Invoke-CondaChecked @(
            "env", "update", "--prefix", $environmentPrefix,
            "--file", $environmentFile
        )
    } else {
        Write-Host "Creating the repo-local Python 3.11 + FFmpeg environment from $environmentFile..."
        Invoke-CondaChecked @(
            "env", "create", "--prefix", $environmentPrefix,
            "--file", $environmentFile, "--yes"
        )
    }

    $doctorArguments = @(
        "run", "--no-capture-output", "--prefix", $environmentPrefix,
        "python", "-m", "scripts.environment_doctor", "--profile", $Profile
    )
    if ($Profile -in @("offline-local", "shot-windows-gpu")) {
        $doctorArguments += "--skip-data"
    }
    Invoke-CondaChecked $doctorArguments
    Invoke-CondaChecked @(
        "run", "--no-capture-output", "--prefix", $environmentPrefix,
        "python", "-m", "compileall", "-q", "offline", "shared", "scripts", "tests"
    )
    Invoke-CondaChecked @(
        "run", "--no-capture-output", "--prefix", $environmentPrefix,
        "python", "-m", "unittest", "discover", "-s", "tests", "-q"
    )
} finally {
    Pop-Location
}

Write-Host "Bootstrap complete. Activate with:"
Write-Host "  & '$miniforgePrefix\shell\condabin\conda-hook.ps1'"
Write-Host "  conda activate '$environmentPrefix'"
Write-Host "Or run a module without activation:"
Write-Host "  .\scripts\run_offline_windows.ps1 -EnvironmentPath '$EnvironmentPath' -Module scripts.environment_doctor -PythonArguments @('--profile','$Profile')"
