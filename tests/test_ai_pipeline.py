import pytest

torch = pytest.importorskip("torch")

from src.train.dataset import load_dataset, make_dataset, save_dataset, validate_dataset
from src.game.state import encode_state, legal_mask, new_game
from src.model.alphanet.network import AlphaNet
from src.model.mcts.mymcts import AlphaZeroMCTS, MCTSConfig
from src.train.selfplay import SelfPlayConfig, generate_self_play_dataset
from src.train.train import train_from_dataset
from src.train.train_utils import train_step


def _initial_sample_tensors(board_size=4):
    state = new_game(board_size)
    encoded = torch.tensor(encode_state(state, state.current_player), dtype=torch.float32)
    mask = torch.tensor(legal_mask(state, state.current_player), dtype=torch.bool)
    policy = torch.zeros(board_size * board_size, dtype=torch.float32)
    legal_indices = mask.nonzero().view(-1)
    policy[legal_indices] = 1.0 / len(legal_indices)
    return encoded, mask, policy


def test_dataset_round_trip_and_validation(tmp_path):
    encoded, mask, policy = _initial_sample_tensors()
    dataset = make_dataset(
        encoded.unsqueeze(0),
        mask.unsqueeze(0),
        policy.unsqueeze(0),
        torch.tensor([1.0]),
        board_size=4,
    )

    path = tmp_path / "selfplay.pt"
    save_dataset(dataset, path)
    loaded = load_dataset(path)

    validate_dataset(loaded)
    assert loaded.metadata.format_version == "az-do-dataset-v1"
    assert loaded.metadata.board_size == 4
    assert len(loaded) == 1


def test_dataset_rejects_policy_mass_on_illegal_moves():
    encoded, mask, policy = _initial_sample_tensors()
    illegal_index = (~mask).nonzero().view(-1)[0]
    policy[illegal_index] = 0.25

    with pytest.raises(ValueError, match="Illegal moves"):
        make_dataset(
            encoded.unsqueeze(0),
            mask.unsqueeze(0),
            policy.unsqueeze(0),
            torch.tensor([0.0]),
            board_size=4,
        )


def test_local_mcts_returns_legal_visit_distribution():
    state = new_game(4)
    mcts = AlphaZeroMCTS(config=MCTSConfig(num_simulations=8))
    root = mcts.search(state)
    distribution = torch.tensor(mcts.visit_distribution(root))
    mask = torch.tensor(legal_mask(state, state.current_player))

    assert torch.isclose(distribution.sum(), torch.tensor(1.0))
    assert torch.all(distribution[~mask] == 0)


def test_self_play_generates_valid_dataset():
    dataset = generate_self_play_dataset(
        config=SelfPlayConfig(
            board_size=4,
            games=1,
            num_simulations=2,
            seed=1,
            add_root_noise=False,
        )
    )

    validate_dataset(dataset)
    assert len(dataset) > 0


def test_train_step_accepts_soft_policy_targets():
    encoded, mask, policy = _initial_sample_tensors()
    batch = {
        "state": torch.stack([encoded, encoded]),
        "legal_mask": torch.stack([mask, mask]),
        "policy": torch.stack([policy, policy]),
        "value": torch.tensor([1.0, -1.0]),
    }
    model = AlphaNet(board_size=4, num_filters=8, num_res_blocks=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    loss, policy_loss, value_loss = train_step(model, optimizer, batch, device="cpu")

    assert loss > 0
    assert policy_loss > 0
    assert value_loss >= 0


def test_train_from_dataset_can_initialize_from_checkpoint(tmp_path):
    encoded, mask, policy = _initial_sample_tensors()
    dataset = make_dataset(
        encoded.unsqueeze(0),
        mask.unsqueeze(0),
        policy.unsqueeze(0),
        torch.tensor([0.0]),
        board_size=4,
    )
    dataset_path = tmp_path / "dataset.pt"
    save_dataset(dataset, dataset_path)

    init_model = AlphaNet(board_size=4, num_filters=8, num_res_blocks=1)
    init_path = tmp_path / "init.pth"
    output_path = tmp_path / "continued.pth"
    torch.save(init_model.state_dict(), init_path)

    metadata = train_from_dataset(
        dataset_path=dataset_path,
        output_path=output_path,
        init_checkpoint=init_path,
        board_size=4,
        epochs=0,
        batch_size=1,
        device="cpu",
        model_kwargs={"num_filters": 8, "num_res_blocks": 1},
    )

    saved_state = torch.load(output_path, map_location="cpu", weights_only=False)
    for name, value in init_model.state_dict().items():
        assert torch.equal(saved_state[name], value)
    assert metadata["init_checkpoint"] == str(init_path)
