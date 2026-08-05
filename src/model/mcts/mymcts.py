"""Project-local AlphaZero-style MCTS for Diffusive Othello."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Protocol

import torch

from src.game.state import (
    GameState,
    Move,
    apply_move,
    encode_state,
    is_terminal,
    legal_mask,
    legal_moves,
    pass_turn,
    score,
    winner,
)

PASS_ACTION = -1


class Evaluator(Protocol):
    def evaluate(self, state: GameState, player: int) -> tuple[list[float], float]:
        """Return flattened policy priors and value from player's perspective."""


class UniformEvaluator:
    """Legal-uniform priors and a neutral value, useful before a model exists."""

    def evaluate(self, state: GameState, player: int) -> tuple[list[float], float]:
        mask = legal_mask(state, player)
        legal_count = sum(1 for allowed in mask if allowed)
        if legal_count == 0:
            return [0.0 for _ in mask], 0.0
        prob = 1.0 / legal_count
        return [prob if allowed else 0.0 for allowed in mask], 0.0


class NeuralEvaluator:
    """Adapter from AlphaNet-style models to MCTS priors and values."""

    def __init__(self, model: torch.nn.Module, device: str = "cpu") -> None:
        self.model = model.to(device)
        self.device = device

    def evaluate(self, state: GameState, player: int) -> tuple[list[float], float]:
        self.model.eval()
        with torch.no_grad():
            encoded = torch.tensor(
                encode_state(state, player),
                dtype=torch.get_default_dtype(),
                device=self.device,
            ).unsqueeze(0)
            mask = torch.tensor(
                legal_mask(state, player),
                dtype=torch.bool,
                device=self.device,
            ).unsqueeze(0)
            log_policy, value = self.model(encoded, legal_mask=mask)
            probs = log_policy.exp().squeeze(0).detach().cpu().tolist()
        return _renormalize_legal_probs(probs, mask.squeeze(0).detach().cpu().tolist()), float(value.item())


@dataclass
class SearchNode:
    state: GameState
    player: int
    prior: float = 1.0
    visit_count: int = 0
    value_sum: float = 0.0
    children: dict[int, "SearchNode"] = field(default_factory=dict)
    expanded: bool = False

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
        if is_terminal(node.state):
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

        moves = legal_moves(node.state, node.player)
        if not moves:
            passed_state = pass_turn(node.state).state
            node.children[PASS_ACTION] = SearchNode(
                state=passed_state,
                player=passed_state.current_player,
                prior=1.0,
            )
            node.expanded = True
            return 0.0

        priors, value = self.evaluator.evaluate(node.state, node.player)
        mask = legal_mask(node.state, node.player)
        priors = _renormalize_legal_probs(priors, mask)
        size = node.state.size
        for x, y in moves:
            action = x * size + y
            child_state = apply_move(node.state, node.player, (x, y)).state
            node.children[action] = SearchNode(
                state=child_state,
                player=child_state.current_player,
                prior=priors[action],
            )
        node.expanded = True
        return value

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
