#!/usr/bin/env bash
# Linux training entry point for the multi-stage AlphaZero pipeline.

set -Eeuo pipefail

usage() {
    cat <<'EOF'
Usage: bash ./train.sh [options]

Runs the multi-stage trainer in the background by default and writes logs under
logs/. Use --foreground to keep the trainer attached to the current terminal.

Options:
  --device DEVICE                 Training device (default: cuda)
  --self-play-batch-size N        Concurrent in-process self-play games (default: 8)
  --self-play-workers N           Self-play worker processes (default: 1)
  --schedule {full,continue}      Training schedule (default: full)
  --start-stage N                 First curriculum stage or continuation round
  --end-stage N                   Last curriculum stage or continuation round
  --initial-checkpoint PATH       Explicit initial/incumbent checkpoint
  --arena-games N                 Even arena game count for continue (default: 40)
  --arena-simulations N           Arena MCTS simulations per move
  --arena-minimum-score SCORE     Strict promotion threshold (default: 0.5)
  --out-log PATH                  Standard-output log (default: logs/train_multistage.out)
  --err-log PATH                  Standard-error log (default: logs/train_multistage.err)
  --include-smoke                 Include the small smoke stage (full only)
  --overwrite                     Replace existing stage outputs
  --no-resume                     Do not skip already-completed stage outputs
  --no-promote-latest             Do not promote the final full-schedule model
  --foreground                    Run in the current terminal instead of backgrounding
  -h, --help                      Show this help message
EOF
}

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

device="cuda"
self_play_batch_size=8
self_play_workers=1
schedule="full"
start_stage=""
end_stage=""
initial_checkpoint=""
arena_games=40
arena_simulations=""
arena_minimum_score=0.5
out_log="logs/train_multistage.out"
err_log="logs/train_multistage.err"
include_smoke=0
overwrite=0
resume=1
promote_latest=1
foreground=0

require_value() {
    if [[ $# -lt 2 || -z ${2:-} ]]; then
        printf 'Missing value for %s\n' "$1" >&2
        exit 2
    fi
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --device)
            require_value "$@"; device="$2"; shift 2 ;;
        --self-play-batch-size)
            require_value "$@"; self_play_batch_size="$2"; shift 2 ;;
        --self-play-workers)
            require_value "$@"; self_play_workers="$2"; shift 2 ;;
        --schedule)
            require_value "$@"; schedule="$2"; shift 2 ;;
        --start-stage)
            require_value "$@"; start_stage="$2"; shift 2 ;;
        --end-stage)
            require_value "$@"; end_stage="$2"; shift 2 ;;
        --initial-checkpoint)
            require_value "$@"; initial_checkpoint="$2"; shift 2 ;;
        --arena-games)
            require_value "$@"; arena_games="$2"; shift 2 ;;
        --arena-simulations)
            require_value "$@"; arena_simulations="$2"; shift 2 ;;
        --arena-minimum-score)
            require_value "$@"; arena_minimum_score="$2"; shift 2 ;;
        --out-log)
            require_value "$@"; out_log="$2"; shift 2 ;;
        --err-log)
            require_value "$@"; err_log="$2"; shift 2 ;;
        --include-smoke)
            include_smoke=1; shift ;;
        --overwrite)
            overwrite=1; shift ;;
        --no-resume)
            resume=0; shift ;;
        --no-promote-latest)
            promote_latest=0; shift ;;
        --foreground)
            foreground=1; shift ;;
        -h|--help)
            usage; exit 0 ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2 ;;
    esac
done

if [[ "$schedule" != "full" && "$schedule" != "continue" ]]; then
    printf 'Invalid --schedule value: %s (expected full or continue)\n' "$schedule" >&2
    exit 2
fi

if [[ -x "venv/bin/python" ]]; then
    python_bin="venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
    python_bin="python3"
elif command -v python >/dev/null 2>&1; then
    python_bin="python"
else
    printf 'Python was not found. Create venv/ or install python3.\n' >&2
    exit 127
fi

mkdir -p "$(dirname -- "$out_log")" "$(dirname -- "$err_log")"

arguments=(
    -u
    -m
    src.train.train_multistage
    --device "$device"
    --self-play-batch-size "$self_play_batch_size"
    --self-play-workers "$self_play_workers"
    --schedule "$schedule"
)

if [[ -n "$start_stage" ]]; then
    arguments+=(--start-stage "$start_stage")
fi
if [[ -n "$end_stage" ]]; then
    arguments+=(--end-stage "$end_stage")
fi
if [[ -n "$initial_checkpoint" ]]; then
    arguments+=(--initial-checkpoint "$initial_checkpoint")
fi
if [[ "$schedule" == "continue" ]]; then
    arguments+=(--arena-games "$arena_games" --arena-minimum-score "$arena_minimum_score")
    if [[ -n "$arena_simulations" ]]; then
        arguments+=(--arena-simulations "$arena_simulations")
    fi
fi
if [[ $include_smoke -eq 1 ]]; then
    arguments+=(--include-smoke)
fi
if [[ $overwrite -eq 1 ]]; then
    arguments+=(--overwrite)
fi
if [[ $resume -eq 1 && $overwrite -eq 0 ]]; then
    arguments+=(--resume)
fi
if [[ "$schedule" != "continue" && $promote_latest -eq 1 ]]; then
    arguments+=(--promote-latest)
fi

if [[ $foreground -eq 1 ]]; then
    exec "$python_bin" "${arguments[@]}"
fi

nohup "$python_bin" "${arguments[@]}" >"$out_log" 2>"$err_log" < /dev/null &
pid=$!
printf 'Started multi-stage training. PID: %s\n' "$pid"
printf 'stdout: %s\n' "$out_log"
printf 'stderr: %s\n' "$err_log"
