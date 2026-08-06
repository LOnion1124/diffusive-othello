from pathlib import Path

import pytest

pytest.importorskip("torch")

from train_multistage import (
    DEFAULT_STAGES,
    select_stages,
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
