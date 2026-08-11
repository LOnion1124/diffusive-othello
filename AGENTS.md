# Agent Entry Point

This is the starting point for coding agents working on Diffusive Othello.

## Read First

1. `docs/SKILL.md`: development workflow and agent rules.
2. `docs/TASKS.md`: project assessment and refactoring roadmap.
3. `docs/ENVIRONMENT.md`: dependency and runtime environment notes.
4. `README.md`: current user-facing overview.

## Project Goals

The project has two primary engineering goals:

1. Build an excellent, reliable implementation of Diffusive Othello.
2. Build an AI training method that can learn effective play.

Near-term direction:

- Centralize game rules into one authoritative engine.
- Keep pygame as the desktop client.
- Add a web game client as a separate playable target.
- Replace the current AI training path with a consistent self-play and MCTS pipeline.

## Before Editing

Run:

```sh
git status --short
```

Then inspect the relevant files. Do not assume the working tree is clean, and do not revert user changes without explicit instruction.

## Important Architecture Rule

Game rules must not be duplicated.

The same authoritative game engine should serve:

- CLI play;
- pygame play;
- future web play;
- legal move masks;
- MCTS;
- self-play;
- dataset generation;
- inference validation.

## Documentation Rule

When a development phase changes behavior, commands, architecture, dependencies, data formats, or agent workflow, update the relevant docs before considering the task complete.

Use:

- `docs/TASKS.md` for roadmap progress;
- `docs/ENVIRONMENT.md` for setup and dependency changes;
- `docs/SKILL.md` for agent workflow changes;
- `README.md` for user-facing instructions.
