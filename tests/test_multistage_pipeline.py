import argparse
import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.train.train_multistage import (
    CONTINUATION_STAGE,
    DEFAULT_STAGES,
    continuation_next_checkpoint,
    select_stages,
    stage_bounds_for_schedule,
    resolve_initial_checkpoint,
    stage_paths,
)
from src.train.arena import ArenaGameResult, ArenaResult
import src.train.train_multistage as multistage


def test_select_stages_can_include_smoke():
    selected = select_stages(include_smoke=True, start_stage=0, end_stage=2)

    assert [stage.index for stage in selected] == [0, 1, 2]


def test_stage_paths_use_stage_prefix():
    dataset_path, output_path = stage_paths(
        DEFAULT_STAGES[0],
        data_dir=Path("data"),
        model_dir=Path("model"),
        prefix="az",
    )

    assert dataset_path == Path("data/az1_bootstrap.pt")
    assert output_path == Path("model/az1_bootstrap.pth")


def test_resolve_initial_checkpoint_infers_previous_stage(tmp_path):
    previous = tmp_path / "stage1_bootstrap.pth"
    previous.write_bytes(b"checkpoint")

    checkpoint = resolve_initial_checkpoint(
        None,
        DEFAULT_STAGES[1],
        include_smoke=False,
        model_dir=tmp_path,
        prefix="stage",
    )

    assert checkpoint == str(previous)


def test_continuation_schedule_repeats_stage_five_workload_by_round():
    start_stage, end_stage = stage_bounds_for_schedule(
        "continue",
        start_stage=None,
        end_stage=2,
    )
    selected = select_stages(
        include_smoke=False,
        start_stage=start_stage,
        end_stage=end_stage,
        schedule="continue",
    )

    assert [stage.index for stage in selected] == [1, 2]
    assert all(stage.name == "stage5" for stage in selected)
    assert all(stage.games == CONTINUATION_STAGE.games for stage in selected)
    assert all(stage.simulations == CONTINUATION_STAGE.simulations for stage in selected)
    assert all(stage.epochs == CONTINUATION_STAGE.epochs for stage in selected)

    assert stage_bounds_for_schedule("continue", start_stage=3, end_stage=None) == (3, 3)


def test_continuation_advances_only_the_promoted_product_checkpoint(tmp_path):
    product_checkpoint = tmp_path / "latest.pth"

    next_checkpoint, stopped_early = continuation_next_checkpoint(
        stage_start_checkpoint="models/seed.pth",
        product_checkpoint=product_checkpoint,
        promoted=True,
    )

    assert next_checkpoint == str(product_checkpoint)
    assert not stopped_early

    next_checkpoint, stopped_early = continuation_next_checkpoint(
        stage_start_checkpoint="models/seed.pth",
        product_checkpoint=product_checkpoint,
        promoted=False,
    )

    assert next_checkpoint == "models/seed.pth"
    assert stopped_early


def test_continuation_uses_product_checkpoint_and_stops_after_arena_failure(
    tmp_path,
    monkeypatch,
):
    seed_checkpoint = tmp_path / "seed.pth"
    latest_checkpoint = tmp_path / "latest.pth"
    manifest_path = tmp_path / "manifest.json"
    seed_checkpoint.write_bytes(b"seed")
    stages = (
        multistage.StageSpec(1, "one", 1, 1, 1, 1, 1e-3),
        multistage.StageSpec(2, "two", 1, 1, 1, 1, 1e-3),
    )
    args = argparse.Namespace(
        schedule="continue",
        device="cpu",
        board_size=4,
        data_dir=str(tmp_path / "data"),
        model_dir=str(tmp_path / "models"),
        prefix="candidate",
        initial_checkpoint=str(seed_checkpoint),
        start_stage=1,
        end_stage=2,
        include_smoke=False,
        resume=False,
        overwrite=False,
        promote_latest=False,
        latest_path=str(latest_checkpoint),
        manifest=str(manifest_path),
        weight_decay=0.0,
        self_play_batch_size=1,
        self_play_workers=1,
        temperature=1.0,
        temperature_drop_move=1,
        seed_base=1,
        no_root_noise=True,
        arena_games=2,
        arena_simulations=1,
        arena_c_puct=1.5,
        arena_temperature=1.0,
        arena_seed_base=1,
        arena_minimum_score=0.5,
    )
    checkpoint_inputs: list[str] = []
    arena_inputs: list[str] = []
    arena_results = iter((_arena_result(candidate_wins=2), _arena_result(candidate_wins=0)))

    monkeypatch.setattr(multistage, "parse_args", lambda: args)
    monkeypatch.setattr(multistage, "select_stages", lambda **_: list(stages))
    monkeypatch.setattr(multistage, "apply_alphanet_arg_overrides", lambda *_: {})
    monkeypatch.setattr(
        multistage,
        "load_model_for_self_play",
        lambda checkpoint, **_: checkpoint_inputs.append(str(checkpoint)) or object(),
    )
    monkeypatch.setattr(multistage, "generate_self_play_dataset", lambda **_: [object()])

    def fake_train_from_dataset(*, output_path, **_):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(b"candidate")
        return {
            "steps": 1,
            "last_loss": 1.0,
            "last_policy_loss": 1.0,
            "last_value_loss": 0.0,
        }

    def fake_run_checkpoint_arena(*, incumbent_checkpoint, **_):
        arena_inputs.append(str(incumbent_checkpoint))
        return next(arena_results)

    monkeypatch.setattr(multistage, "train_from_dataset", fake_train_from_dataset)
    monkeypatch.setattr(multistage, "run_checkpoint_arena", fake_run_checkpoint_arena)

    assert multistage.main() == 0
    assert checkpoint_inputs == [str(seed_checkpoint), str(latest_checkpoint)]
    assert arena_inputs == [str(seed_checkpoint), str(latest_checkpoint)]
    assert latest_checkpoint.read_bytes() == b"candidate"

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["stopped_early"] is True
    assert len(manifest["stages"]) == 2
    assert manifest["stages"][1]["arena"]["promoted"] is False


def _arena_result(*, candidate_wins: int) -> ArenaResult:
    games = [
        ArenaGameResult(
            game_index=index,
            candidate_player=1 if index == 0 else -1,
            winner=(1 if index == 0 else -1) if candidate_wins == 2 else (-1 if index == 0 else 1),
            candidate_margin=1 if candidate_wins == 2 else -1,
            move_count=1,
            pass_count=0,
        )
        for index in range(2)
    ]
    return ArenaResult(
        candidate_wins=candidate_wins,
        incumbent_wins=2 - candidate_wins,
        draws=0,
        games=games,
    )
