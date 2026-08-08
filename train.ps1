param(
    [string]$Device = "cuda",
    [int]$SelfPlayBatchSize = 8,
    [int]$SelfPlayWorkers = 1,
    [int]$StartStage = 1,
    [int]$EndStage = 5,
    [string]$InitialCheckpoint = "",
    [string]$OutLog = "logs\train_multistage.out",
    [string]$ErrLog = "logs\train_multistage.err",
    [switch]$IncludeSmoke,
    [switch]$Overwrite,
    [switch]$NoResume,
    [switch]$NoPromoteLatest
)

$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    New-Item -ItemType Directory -Force -Path "logs" | Out-Null

    foreach ($logPath in @($OutLog, $ErrLog)) {
        $parent = Split-Path -Parent $logPath
        if ($parent) {
            New-Item -ItemType Directory -Force -Path $parent | Out-Null
        }
    }

    $python = ".\venv\Scripts\python.exe"
    if (-not (Test-Path -LiteralPath $python)) {
        $python = "python"
    }

    $arguments = @(
        "-u",
        "-m",
        "src.train.train_multistage",
        "--device",
        $Device,
        "--self-play-batch-size",
        [string]$SelfPlayBatchSize,
        "--self-play-workers",
        [string]$SelfPlayWorkers,
        "--start-stage",
        [string]$StartStage,
        "--end-stage",
        [string]$EndStage
    )

    if ($InitialCheckpoint) {
        $arguments += @("--initial-checkpoint", $InitialCheckpoint)
    }
    if ($IncludeSmoke) {
        $arguments += "--include-smoke"
    }
    if ($Overwrite) {
        $arguments += "--overwrite"
    }
    if (-not $NoResume -and -not $Overwrite) {
        $arguments += "--resume"
    }
    if (-not $NoPromoteLatest) {
        $arguments += "--promote-latest"
    }

    $process = Start-Process `
        -FilePath $python `
        -ArgumentList $arguments `
        -WorkingDirectory "." `
        -RedirectStandardOutput $OutLog `
        -RedirectStandardError $ErrLog `
        -WindowStyle Hidden `
        -PassThru

    Write-Host "Started multi-stage training. PID: $($process.Id)"
    Write-Host "stdout: $OutLog"
    Write-Host "stderr: $ErrLog"
}
finally {
    Pop-Location
}
