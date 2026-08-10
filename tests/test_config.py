from pathlib import Path

from src.config.config import get_alphanet_kwargs, load_config


def test_load_config_exposes_new_ai_sections():
    config = load_config()

    assert config["game"]["board_size"] == 9
    assert config["ai"]["runtime"]["model_path"].endswith(".pth")
    assert config["ai"]["model"]["architecture"] == "alphanet"
    assert config["ai"]["mcts"]["num_simulations"] > 0
    assert config["ai"]["arena"]["games"] % 2 == 0
    assert config["ai"]["arena"]["move_temperature"] >= 0
    assert config["ai"]["self_play"]["output_path"].endswith(".pt")
    assert config["ai"]["train"]["output_path"] == "models/latest.pth"


def test_alphanet_kwargs_match_config_schema():
    config = load_config()

    assert get_alphanet_kwargs(config) == {
        "board_size": 9,
        "in_channels": 3,
        "num_filters": 96,
        "num_res_blocks": 6,
        "value_hidden_dim": 128,
    }


def test_legacy_ai_config_fields_are_normalized(tmp_path):
    path = tmp_path / "legacy.yaml"
    path.write_text(
        "\n".join(
            [
                "model_path: models/old.pth",
                "use_cuda: false",
                "board_size: 4",
                "mcts:",
                "  num_simulations: 8",
                "train:",
                "  data_path: data/old.pt",
            ]
        ),
        encoding="utf-8",
    )

    config = load_config(Path(path))

    assert config["game"]["board_size"] == 4
    assert config["ai"]["runtime"]["model_path"] == "models/old.pth"
    assert config["ai"]["runtime"]["device"] == "cpu"
    assert config["ai"]["model"]["board_size"] == 4
    assert config["ai"]["mcts"]["num_simulations"] == 8
    assert config["ai"]["train"]["dataset_path"] == "data/old.pt"
