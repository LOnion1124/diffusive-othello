from pathlib import Path

import pytest

pytest.importorskip("torch")

from src.train.train_multistage import (
    CONTINUATION_STAGE,
    DEFAULT_STAGES,
    select_stages,
    stage_bounds_for_schedule,
    resolve_initial_checkpoint,
    stage_paths,
)


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
