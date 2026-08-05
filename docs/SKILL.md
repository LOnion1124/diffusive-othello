# Agent Development Guide

This document defines how coding agents should work in this repository.

## Entry Point

Start every development session from `AGENT.md`.

Before making changes:

1. Read `AGENT.md`.
2. Read the relevant docs in `docs/`.
3. Inspect the current code before planning edits.
4. Check `git status --short`.

## Development Principles

Use the existing project direction unless the task explicitly changes it.

Core priorities:

- Make game rules reliable before expanding UI or AI.
- Keep one authoritative game engine.
- Do not duplicate rule logic in UI, MCTS, self-play, or inference code.
- Keep PVP playable without AI dependencies.
- Make AI experiments reproducible through scripts and documented commands.
- Prefer small, testable changes over broad rewrites.

## Documentation Requirements

Every completed phase or meaningful behavior change must update documentation.

Update docs when changing:

- game rules;
- public game-engine APIs;
- setup or dependencies;
- training commands;
- dataset formats;
- model checkpoint formats;
- UI launch commands;
- web app architecture;
- repository layout;
- agent workflow.

Expected documentation targets:

- `docs/TASKS.md` for roadmap and task status.
- `docs/ENVIRONMENT.md` for setup, dependency, and runtime-profile changes.
- `docs/SKILL.md` for agent workflow updates.
- `README.md` for user-facing play and install instructions.
- `AGENT.md` for high-level handoff changes.

Do not finish a phase with code only if the phase changes how future work should be done.

## Testing Requirements

Add or update tests when changing game behavior, data formats, training code, or inference behavior.

Minimum expectations:

- Game-engine changes require rule tests.
- Dataset changes require schema and validation tests.
- Training changes require at least a CPU smoke test when practical.
- UI refactors should keep a manual launch command documented.

If tests cannot be run, explain why in the final report.

## Git Workflow

Protect user work.

- Check `git status --short` before editing.
- Do not revert changes you did not make unless explicitly asked.
- Keep changes scoped to the requested task.
- Avoid committing generated datasets, temporary files, caches, or accidental checkpoints.

Commit conventions:

- Use clear, imperative commit messages.
- Prefer this format:

```text
<type>: <short summary>
```

Recommended types:

- `docs`: documentation-only changes;
- `fix`: bug fix;
- `feat`: new feature;
- `refactor`: code restructuring without intended behavior change;
- `test`: tests only;
- `train`: training pipeline, dataset, or model workflow changes;
- `chore`: tooling or maintenance.

Examples:

```text
docs: add agent development roadmap
fix: count player scores explicitly
refactor: centralize legal move generation
train: add MCTS visit policy dataset
```

When a commit includes multiple areas, choose the type that best describes the user-visible intent.

## Code Style

- Prefer simple, explicit Python.
- Use type hints for public APIs.
- Keep game logic independent from pygame, torch, and web frameworks.
- Use pure functions where practical for legal moves, move application, and scoring.
- Keep device-specific torch logic out of generic data structures.
- Do not introduce new global state unless it is truly configuration.

## Handoff Notes

At the end of a task, report:

- what changed;
- what was verified;
- what was not verified and why;
- the next recommended task if it is obvious.

For long work, update relevant docs before the final report.
