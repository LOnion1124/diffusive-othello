param(
    [string]$Device = "cuda",
    [int]$SelfPlayBatchSize = 8,
    [int]$SelfPlayWorkers = 1,
    [ValidateSet("full", "continue")]
    [string]$Schedule = "full",
    [int]$StartStage = -1,
    [int]$EndStage = -1,
    [string]$InitialCheckpoint = "",
    [int]$ArenaGames = 40,
    [int]$ArenaSimulations = 0,
    [double]$ArenaMinimumScore = 0.5,
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
        "--schedule",
        $Schedule
    )

    if ($StartStage -ge 0) {
        $arguments += @("--start-stage", [string]$StartStage)
    }
    if ($EndStage -ge 0) {
        $arguments += @("--end-stage", [string]$EndStage)
    }
    if ($InitialCheckpoint) {
        $arguments += @("--initial-checkpoint", $InitialCheckpoint)
    }
    if ($Schedule -eq "continue") {
        $arguments += @("--arena-games", [string]$ArenaGames)
        if ($ArenaSimulations -gt 0) {
            $arguments += @("--arena-simulations", [string]$ArenaSimulations)
        }
        $arguments += @("--arena-minimum-score", [string]$ArenaMinimumScore)
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
    if ($Schedule -ne "continue" -and -not $NoPromoteLatest) {
        $arguments += "--promote-latest"
    }
    if ($Schedule -eq "continue") {
        Write-Host "Continuation stages use the accepted product checkpoint; an arena rejection stops this run."
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
