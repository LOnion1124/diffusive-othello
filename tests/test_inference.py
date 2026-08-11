import pytest


pytest.importorskip("torch")

from src.model.inference import GameAI


def test_suggestions_from_prediction_uses_only_legal_top_probabilities():
    prediction = {
        "scores": [0.99, 0.20, 0.70, 0.40],
        "mask": [False, True, True, True],
    }

    suggestions = GameAI.suggestions_from_prediction(
        prediction,
        board_size=2,
        limit=3,
    )

    assert suggestions == (((1, 0), 0.70), ((1, 1), 0.40), ((0, 1), 0.20))
