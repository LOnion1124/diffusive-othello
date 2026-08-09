import json

import pytest

torch = pytest.importorskip("torch")

from src.model.alphanet.network import AlphaNet
from src.model.mcts.mymcts import UniformEvaluator
from src.train.arena import (
    ArenaConfig,
    ArenaGameResult,
    ArenaResult,
    promote_if_stronger,
    run_arena,
    run_checkpoint_arena,
    should_promote,
)


def test_arena_balances_colors_and_records_all_games():
    result = run_arena(
        candidate_evaluator=UniformEvaluator(),
        incumbent_evaluator=UniformEvaluator(),
        config=ArenaConfig(board_size=4, games=2, num_simulations=1, seed=3),
    )

    assert len(result.games) == 2
    assert [game.candidate_player for game in result.games] == [1, -1]
    assert result.candidate_wins + result.incumbent_wins + result.draws == 2
    assert 0.0 <= result.candidate_score <= 1.0


def test_promotion_requires_a_strictly_better_score_and_copies_metadata(tmp_path):
    games = [
        ArenaGameResult(0, 1, 1, 2, 3, 0),
        ArenaGameResult(1, -1, -1, 4, 4, 0),
    ]
    winning_result = ArenaResult(2, 0, 0, games)
    tied_result = ArenaResult(1, 1, 0, games)
    candidate = tmp_path / "candidate.pth"
    incumbent = tmp_path / "latest.pth"
    candidate.write_bytes(b"candidate")
    incumbent.write_bytes(b"incumbent")
    candidate_metadata = candidate.with_suffix(".pth.json")
    candidate_metadata.write_text(json.dumps({"source": "candidate"}), encoding="utf-8")

    assert should_promote(winning_result)
    assert not should_promote(tied_result)
    assert not promote_if_stronger(
        candidate_checkpoint=candidate,
        incumbent_checkpoint=incumbent,
        result=tied_result,
    )
    assert incumbent.read_bytes() == b"incumbent"

    assert promote_if_stronger(
        candidate_checkpoint=candidate,
        incumbent_checkpoint=incumbent,
        result=winning_result,
    )
    assert incumbent.read_bytes() == b"candidate"
    assert json.loads(incumbent.with_suffix(".pth.json").read_text(encoding="utf-8")) == {
        "source": "candidate"
    }


def test_checkpoint_arena_loads_the_requested_model_shape(tmp_path):
    candidate = tmp_path / "candidate.pth"
    incumbent = tmp_path / "incumbent.pth"
    torch.save(AlphaNet(board_size=4, num_filters=8, num_res_blocks=1).state_dict(), candidate)
    torch.save(AlphaNet(board_size=4, num_filters=8, num_res_blocks=1).state_dict(), incumbent)

    result = run_checkpoint_arena(
        candidate_checkpoint=candidate,
        incumbent_checkpoint=incumbent,
        device="cpu",
        config=ArenaConfig(board_size=4, games=2, num_simulations=1, seed=4),
        model_kwargs={"num_filters": 8, "num_res_blocks": 1},
    )

    assert len(result.games) == 2
    assert result.candidate_wins + result.incumbent_wins + result.draws == 2
