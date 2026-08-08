"""Project-local AlphaZero-style MCTS for Diffusive Othello."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol

import torch

from src.game.state import (
    GameState,
    Move,
    apply_move,
    encode_state,
    legal_mask,
    legal_moves,
    pass_turn,
    score,
    winner,
)

PASS_ACTION = -1


class Evaluator(Protocol):
    def evaluate(
        self,
        state: GameState,
        player: int,
        legal_mask_: list[bool] | None = None,
    ) -> tuple[list[float], float]:
        """Return flattened policy priors and value from player's perspective."""

    def evaluate_batch(
        self,
        requests: Sequence[tuple[GameState, int, list[bool]]],
    ) -> list[tuple[list[float], float]]:
        """Return priors and values for a batch of states."""


class UniformEvaluator:
    """Legal-uniform priors and a neutral value, useful before a model exists."""

    def evaluate(
        self,
        state: GameState,
        player: int,
        legal_mask_: list[bool] | None = None,
    ) -> tuple[list[float], float]:
        mask = legal_mask_ if legal_mask_ is not None else legal_mask(state, player)
        legal_count = sum(1 for allowed in mask if allowed)
        if legal_count == 0:
            return [0.0 for _ in mask], 0.0
        prob = 1.0 / legal_count
        return [prob if allowed else 0.0 for allowed in mask], 0.0

    def evaluate_batch(
        self,
        requests: Sequence[tuple[GameState, int, list[bool]]],
    ) -> list[tuple[list[float], float]]:
        return [
            self.evaluate(state, player, legal_mask_)
            for state, player, legal_mask_ in requests
        ]


class NeuralEvaluator:
    """Adapter from AlphaNet-style models to MCTS priors and values."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device
        self.model.eval()

    def evaluate(
        self,
        state: GameState,
        player: int,
        legal_mask_: list[bool] | None = None,
    ) -> tuple[list[float], float]:
        mask = legal_mask_ if legal_mask_ is not None else legal_mask(state, player)
        return self.evaluate_batch(((state, player, mask),))[0]

    def evaluate_batch(
        self,
        requests: Sequence[tuple[GameState, int, list[bool]]],
    ) -> list[tuple[list[float], float]]:
        if not requests:
            return []

        with torch.inference_mode():
            encoded = torch.tensor(
                [encode_state(state, player) for state, player, _ in requests],
                dtype=torch.get_default_dtype(),
                device=self.device,
            )
            masks = torch.tensor(
                [legal_mask_ for _, _, legal_mask_ in requests],
                dtype=torch.bool,
                device=self.device,
            )
            log_policy, value = self.model(encoded, legal_mask=masks)
            probs_batch = log_policy.exp().detach().cpu().tolist()
            masks_batch = masks.detach().cpu().tolist()
            values = value.detach().cpu().tolist()

        return [
            (_renormalize_legal_probs(probs, mask), float(value_))
            for probs, mask, value_ in zip(probs_batch, masks_batch, values)
        ]


@dataclass
class SearchNode:
    state: GameState
    player: int
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)
    expanded: bool = False
    legal_moves_cache: list[Move] | None = None
    legal_mask_cache: list[bool] | None = None
    terminal_cache: bool | None = None

    @property
    def value_mean(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


@dataclass(frozen=True)
class MCTSConfig:
    num_simulations: int = 64
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25


class AlphaZeroMCTS:
    def __init__(
        self,
        evaluator: Evaluator | None = None,
        config: MCTSConfig | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self.evaluator = evaluator or UniformEvaluator()
        self.config = config or MCTSConfig()
        self.rng = rng or random.Random()

    def search(
        self,
        state: GameState,
        *,
        add_root_noise: bool = False,
    ) -> SearchNode:
        root = SearchNode(state=state, player=state.current_player)
        self._expand_and_evaluate(root)
        if add_root_noise:
            self._add_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            self._simulate(root)
        return root

    def search_batch(
        self,
        states: Sequence[GameState],
        *,
        add_root_noise: bool = False,
    ) -> list[SearchNode]:
        roots = [
            SearchNode(state=state, player=state.current_player)
            for state in states
        ]
        self._expand_and_evaluate_batch(roots)
        if add_root_noise:
            for root in roots:
                self._add_dirichlet_noise(root)

        for _ in range(self.config.num_simulations):
            pending: list[tuple[SearchNode, list[SearchNode]]] = []
            for root in roots:
                path = self._select_path(root)
                leaf = path[-1]
                if self._is_terminal_node(leaf):
                    self._backup(path, _terminal_value(leaf.state, leaf.player))
                elif not leaf.expanded:
                    pending.append((leaf, path))

            if pending:
                values = self._expand_and_evaluate_batch([leaf for leaf, _ in pending])
                for (_, path), value in zip(pending, values):
                    self._backup(path, value)

        return roots

    def visit_distribution(self, root: SearchNode, temperature: float = 1.0) -> list[float]:
        size = root.state.size
        distribution = [0.0 for _ in range(size * size)]
        action_visits = {
            action: child.visit_count
            for action, child in root.children.items()
            if action != PASS_ACTION
        }
        if not action_visits:
            return distribution

        if temperature <= 0:
            best_action = max(action_visits.items(), key=lambda item: item[1])[0]
            distribution[best_action] = 1.0
            return distribution

        scaled = {
            action: visit_count ** (1.0 / temperature)
            for action, visit_count in action_visits.items()
        }
        total = sum(scaled.values())
        if total <= 0:
            prob = 1.0 / len(scaled)
            for action in scaled:
                distribution[action] = prob
            return distribution

        for action, value in scaled.items():
            distribution[action] = value / total
        return distribution

    def _simulate(self, node: SearchNode) -> float:
        if self._is_terminal_node(node):
            value = _terminal_value(node.state, node.player)
            node.visit_count += 1
            node.value_sum += value
            return value

        if not node.expanded:
            value = self._expand_and_evaluate(node)
            node.visit_count += 1
            node.value_sum += value
            return value

        child = self._select_child(node)
        child_value = self._simulate(child)
        value = -child_value
        node.visit_count += 1
        node.value_sum += value
        return value

    def _expand_and_evaluate(self, node: SearchNode) -> float:
        if node.expanded:
            return node.value_mean
        if self._is_terminal_node(node):
            node.expanded = True
            return _terminal_value(node.state, node.player)

        moves = self._legal_moves(node)
        if not moves:
            passed_state = pass_turn(node.state).state
            node.children[PASS_ACTION] = SearchNode(
                state=passed_state,
                player=passed_state.current_player,
                prior=1.0,
            )
            node.expanded = True
            return 0.0

        mask = self._legal_mask(node)
        priors, value = self.evaluator.evaluate(node.state, node.player, mask)
        priors = _renormalize_legal_probs(priors, mask)
        size = node.state.size
        for x, y in moves:
            action = x * size + y
            child_state = apply_move(node.state, node.player, (x, y), validate=False).state
            node.children[action] = SearchNode(
                state=child_state,
                player=child_state.current_player,
                prior=priors[action],
            )
        node.expanded = True
        return value

    def _expand_and_evaluate_batch(self, nodes: Sequence[SearchNode]) -> list[float]:
        values: list[float | None] = [None for _ in nodes]
        evaluable: list[SearchNode] = []
        evaluable_indices: list[int] = []
        requests: list[tuple[GameState, int, list[bool]]] = []

        for index, node in enumerate(nodes):
            if node.expanded:
                values[index] = node.value_mean
                continue
            if self._is_terminal_node(node):
                node.expanded = True
                values[index] = _terminal_value(node.state, node.player)
                continue

            moves = self._legal_moves(node)
            if not moves:
                passed_state = pass_turn(node.state).state
                node.children[PASS_ACTION] = SearchNode(
                    state=passed_state,
                    player=passed_state.current_player,
                    prior=1.0,
                )
                node.expanded = True
                values[index] = 0.0
                continue

            evaluable.append(node)
            evaluable_indices.append(index)
            requests.append((node.state, node.player, self._legal_mask(node)))

        if requests:
            results = self.evaluator.evaluate_batch(requests)
            for node, index, (priors, value) in zip(evaluable, evaluable_indices, results):
                mask = self._legal_mask(node)
                priors = _renormalize_legal_probs(priors, mask)
                size = node.state.size
                for x, y in self._legal_moves(node):
                    action = x * size + y
                    child_state = apply_move(
                        node.state,
                        node.player,
                        (x, y),
                        validate=False,
                    ).state
                    node.children[action] = SearchNode(
                        state=child_state,
                        player=child_state.current_player,
                        prior=priors[action],
                    )
                node.expanded = True
                values[index] = value

        return [float(value) for value in values]

    def _select_path(self, root: SearchNode) -> list[SearchNode]:
        path = [root]
        node = root
        while node.expanded and not self._is_terminal_node(node):
            node = self._select_child(node)
            path.append(node)
        return path

    def _backup(self, path: Sequence[SearchNode], leaf_value: float) -> None:
        value = leaf_value
        for node in reversed(path):
            node.visit_count += 1
            node.value_sum += value
            value = -value

    def _select_child(self, node: SearchNode) -> SearchNode:
        if not node.children:
            raise ValueError("Cannot select from a leaf node.")

        sqrt_parent_visits = math.sqrt(max(node.visit_count, 1))
        best_score = -float("inf")
        best_children: list[SearchNode] = []
        for child in node.children.values():
            q_value = -child.value_mean
            exploration = (
                self.config.c_puct
                * child.prior
                * sqrt_parent_visits
                / (1 + child.visit_count)
            )
            score_value = q_value + exploration
            if score_value > best_score + 1e-12:
                best_score = score_value
                best_children = [child]
            elif abs(score_value - best_score) <= 1e-12:
                best_children.append(child)
        return self.rng.choice(best_children)

    def _legal_moves(self, node: SearchNode) -> list[Move]:
        if node.legal_moves_cache is None:
            node.legal_moves_cache = legal_moves(node.state, node.player)
        return node.legal_moves_cache

    def _legal_mask(self, node: SearchNode) -> list[bool]:
        if node.legal_mask_cache is None:
            mask = [False for _ in range(node.state.size * node.state.size)]
            for x, y in self._legal_moves(node):
                mask[x * node.state.size + y] = True
            node.legal_mask_cache = mask
        return node.legal_mask_cache

    def _is_terminal_node(self, node: SearchNode) -> bool:
        if node.terminal_cache is None:
            if self._legal_moves(node):
                node.terminal_cache = False
            else:
                node.terminal_cache = not legal_moves(node.state, -node.player)
        return node.terminal_cache

    def _add_dirichlet_noise(self, root: SearchNode) -> None:
        actions = [action for action in root.children if action != PASS_ACTION]
        if not actions:
            return

        concentration = torch.full((len(actions),), self.config.dirichlet_alpha)
        noise = torch.distributions.Dirichlet(concentration).sample().tolist()
        for action, noise_value in zip(actions, noise):
            child = root.children[action]
            child.prior = (
                (1.0 - self.config.dirichlet_epsilon) * child.prior
                + self.config.dirichlet_epsilon * float(noise_value)
            )


def choose_action_from_distribution(
    distribution: list[float],
    *,
    temperature: float = 1.0,
    rng: random.Random | None = None,
) -> int:
    rng = rng or random.Random()
    if temperature <= 0:
        return max(range(len(distribution)), key=lambda idx: distribution[idx])

    scaled = [prob ** (1.0 / temperature) for prob in distribution]
    total = sum(scaled)
    if total <= 0:
        legal_actions = [idx for idx, prob in enumerate(distribution) if prob > 0]
        if not legal_actions:
            raise ValueError("Cannot choose an action from an empty distribution.")
        return rng.choice(legal_actions)

    threshold = rng.random() * total
    cumulative = 0.0
    for action, probability in enumerate(scaled):
        cumulative += probability
        if cumulative >= threshold:
            return action
    return len(distribution) - 1


def _renormalize_legal_probs(probs: list[float], mask: list[bool]) -> list[float]:
    if len(probs) != len(mask):
        raise ValueError("Policy prior length must match the legal mask length.")

    cleaned = [
        max(float(prob), 0.0) if allowed and math.isfinite(float(prob)) else 0.0
        for prob, allowed in zip(probs, mask)
    ]
    total = sum(cleaned)
    legal_count = sum(1 for allowed in mask if allowed)
    if legal_count == 0:
        return cleaned
    if total <= 0:
        prob = 1.0 / legal_count
        return [prob if allowed else 0.0 for allowed in mask]
    return [prob / total for prob in cleaned]


def _terminal_value(state: GameState, player: int) -> float:
    game_winner = winner(state)
    if game_winner == 0:
        return 0.0
    return 1.0 if game_winner == player else -1.0


def score_margin_value(state: GameState, player: int) -> float:
    """Bounded deterministic value heuristic for diagnostics and baselines."""

    counts = score(state)
    occupied = counts[1] + counts[-1]
    if occupied == 0:
        return 0.0
    return (counts[player] - counts[-player]) / occupied
