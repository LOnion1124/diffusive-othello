"""Local web dashboard for Diffusive Othello self-play datasets.

Run from the repository root:
    python webtools/data_visualizer/server.py
"""

from __future__ import annotations

import argparse
import json
import math
import mimetypes
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse


REPO_ROOT = Path(__file__).resolve().parents[2]
WEB_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = REPO_ROOT / "data"
INITIAL_STONES = 4


@dataclass(frozen=True)
class CacheKey:
    path: Path
    size: int
    mtime_ns: int


_SUMMARY_CACHE: dict[CacheKey, dict[str, Any]] = {}


def _load_torch():
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - depends on local environment
        raise RuntimeError(
            "torch is required to read self-play .pt datasets. "
            "Install requirements-ai.txt or run inside the project venv."
        ) from exc
    return torch


def _data_dir_from_server(server: ThreadingHTTPServer) -> Path:
    return Path(getattr(server, "data_dir", DEFAULT_DATA_DIR))


def _dataset_files(data_dir: Path) -> list[Path]:
    if not data_dir.exists():
        return []
    return sorted(
        (p for p in data_dir.glob("*.pt") if p.is_file()),
        key=lambda p: (p.stat().st_mtime_ns, p.name),
    )


def list_datasets(data_dir: Path) -> list[dict[str, Any]]:
    datasets = []
    for path in _dataset_files(data_dir):
        stat = path.stat()
        datasets.append(
            {
                "name": path.name,
                "sizeBytes": stat.st_size,
                "modified": stat.st_mtime,
            }
        )
    return datasets


def resolve_requested_files(data_dir: Path, query: dict[str, list[str]]) -> list[Path]:
    requested: list[str] = []
    for value in query.get("file", []):
        requested.extend(part for part in value.split(",") if part)

    all_files = {path.name: path for path in _dataset_files(data_dir)}
    if not requested:
        return list(all_files.values())

    resolved = []
    for name in requested:
        clean_name = Path(unquote(name)).name
        if clean_name not in all_files:
            raise FileNotFoundError(f"Dataset not found: {clean_name}")
        resolved.append(all_files[clean_name])
    return resolved


def summarize_dataset(path: Path) -> dict[str, Any]:
    stat = path.stat()
    key = CacheKey(path.resolve(), stat.st_size, stat.st_mtime_ns)
    if key in _SUMMARY_CACHE:
        return _SUMMARY_CACHE[key]

    torch = _load_torch()
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if hasattr(payload, "to_payload"):
        payload = payload.to_payload()
    if not isinstance(payload, dict):
        raise ValueError(f"{path.name} does not contain a dataset dict payload.")

    metadata = dict(payload.get("metadata") or {})
    if metadata.get("format_version") != "az-do-dataset-v2":
        raise ValueError(f"{path.name} is not an az-do-dataset-v2 file.")
    states = payload["states"].detach().cpu().float()
    legal_masks = payload["legal_masks"].detach().cpu().bool()
    policies = payload["policies"].detach().cpu().float()
    values = payload["values"].detach().cpu().float().view(-1)
    sample_metadata = {
        key: value.detach().cpu()
        for key, value in payload["sample_metadata"].items()
    }
    game_metadata = {
        key: value.detach().cpu()
        for key, value in payload["game_metadata"].items()
    }

    sample_count = int(states.shape[0])
    board_size = int(metadata.get("board_size") or states.shape[-1])
    game_count = int(game_metadata["game_id"].numel())
    if sample_count <= 0:
        raise ValueError(f"{path.name} contains no samples.")

    empties = sample_metadata["empty_count"].float()
    own = sample_metadata["own_count"].float()
    opponent = sample_metadata["opponent_count"].float()
    occupied = own + opponent
    ply = sample_metadata["ply"].float()
    piece_diff = sample_metadata["current_margin"].float()
    legal_counts = legal_masks.sum(dim=1).float()
    policy_entropy = sample_metadata["policy_entropy"].float()
    top_policy = sample_metadata["top_policy"].float()
    root_value = sample_metadata["root_value"].float()
    chosen_q = sample_metadata["chosen_q"].float()
    root_visits = sample_metadata["root_visit_count"].clamp_min(1).float()
    chosen_visit_share = sample_metadata["chosen_visit_count"].float() / root_visits
    flipped_counts = sample_metadata["flipped_count"].float()

    winners = game_metadata["winner"].long()
    first_players = game_metadata["first_player"].long()
    first_mover_values = torch.where(
        winners == 0,
        torch.zeros_like(winners, dtype=torch.float32),
        torch.where(winners == first_players, 1.0, -1.0),
    )
    game_lengths = game_metadata["move_count"].float()
    pass_counts = game_metadata["pass_count"].float()
    final_margins = game_metadata["final_margin_p1"].float()

    phase_labels = _phase_labels(occupied, board_size)
    phase_summary = _phase_outcomes(phase_labels, values)

    policy_heatmap = policies.mean(dim=0).view(board_size, board_size)
    legal_heatmap = legal_masks.float().mean(dim=0).view(board_size, board_size)
    own_heatmap = states[:, 1].mean(dim=0)
    opponent_heatmap = states[:, 2].mean(dim=0)

    summary = {
        "name": path.name,
        "sizeBytes": stat.st_size,
        "modified": stat.st_mtime,
        "metadata": metadata,
        "sampleCount": sample_count,
        "gameCount": game_count,
        "boardSize": board_size,
        "metrics": {
            "outcomes": _outcome_counts(values),
            "firstMoverOutcomes": _outcome_counts(first_mover_values),
            "value": _series_stats(values),
            "legalMoves": _series_stats(legal_counts),
            "policyEntropy": _series_stats(policy_entropy),
            "topPolicy": _series_stats(top_policy),
            "currentPieceDiff": _series_stats(piece_diff),
            "occupiedCells": _series_stats(occupied),
            "ply": _series_stats(ply),
            "flippedCount": _series_stats(flipped_counts),
            "rootValue": _series_stats(root_value),
            "chosenQ": _series_stats(chosen_q),
            "chosenVisitShare": _series_stats(chosen_visit_share),
            "gameLength": _series_stats(game_lengths),
            "passCount": _series_stats(pass_counts),
            "finalMarginP1": _series_stats(final_margins),
        },
        "distributions": {
            "legalMoves": _int_histogram(legal_counts),
            "ply": _int_histogram(ply),
            "currentPieceDiff": _int_histogram(piece_diff),
            "topPolicy": _range_histogram(top_policy, 0.0, 1.0, 10),
            "policyEntropy": _range_histogram(policy_entropy, 0.0, math.log(board_size * board_size), 12),
            "flippedCount": _int_histogram(flipped_counts),
            "chosenVisitShare": _range_histogram(chosen_visit_share, 0.0, 1.0, 10),
            "rootValue": _range_histogram(root_value, -1.0, 1.0, 12),
            "gameLength": _int_histogram(game_lengths),
            "passCount": _int_histogram(pass_counts),
            "finalMarginP1": _int_histogram(final_margins),
        },
        "phaseSummary": phase_summary,
        "heatmaps": {
            "policy": _matrix(policy_heatmap),
            "legal": _matrix(legal_heatmap),
            "own": _matrix(own_heatmap),
            "opponent": _matrix(opponent_heatmap),
        },
        "limitations": [],
    }
    _SUMMARY_CACHE[key] = summary
    return summary


def combine_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"sampleCount": 0}

    sample_count = sum(s["sampleCount"] for s in summaries)
    game_count = sum(s["gameCount"] for s in summaries)
    board_sizes = sorted({s["boardSize"] for s in summaries})

    combined = {
        "name": "selected",
        "datasetCount": len(summaries),
        "sampleCount": sample_count,
        "gameCount": game_count,
        "boardSize": board_sizes[0] if len(board_sizes) == 1 else None,
        "boardSizes": board_sizes,
        "metrics": {
            "outcomes": _combine_counts([s["metrics"]["outcomes"] for s in summaries]),
            "firstMoverOutcomes": _combine_counts(
                [s["metrics"]["firstMoverOutcomes"] for s in summaries]
            ),
            "value": _combine_series_stats(summaries, "value"),
            "legalMoves": _combine_series_stats(summaries, "legalMoves"),
            "policyEntropy": _combine_series_stats(summaries, "policyEntropy"),
            "topPolicy": _combine_series_stats(summaries, "topPolicy"),
            "currentPieceDiff": _combine_series_stats(summaries, "currentPieceDiff"),
            "occupiedCells": _combine_series_stats(summaries, "occupiedCells"),
            "ply": _combine_series_stats(summaries, "ply"),
            "flippedCount": _combine_series_stats(summaries, "flippedCount"),
            "rootValue": _combine_series_stats(summaries, "rootValue"),
            "chosenQ": _combine_series_stats(summaries, "chosenQ"),
            "chosenVisitShare": _combine_series_stats(summaries, "chosenVisitShare"),
            "gameLength": _combine_series_stats(summaries, "gameLength", count_key="gameCount"),
            "passCount": _combine_series_stats(summaries, "passCount", count_key="gameCount"),
            "finalMarginP1": _combine_series_stats(summaries, "finalMarginP1", count_key="gameCount"),
        },
        "distributions": {
            "legalMoves": _combine_distribution(summaries, "legalMoves"),
            "ply": _combine_distribution(summaries, "ply"),
            "currentPieceDiff": _combine_distribution(summaries, "currentPieceDiff"),
            "topPolicy": _combine_distribution(summaries, "topPolicy"),
            "policyEntropy": _combine_distribution(summaries, "policyEntropy"),
            "flippedCount": _combine_distribution(summaries, "flippedCount"),
            "chosenVisitShare": _combine_distribution(summaries, "chosenVisitShare"),
            "rootValue": _combine_distribution(summaries, "rootValue"),
            "gameLength": _combine_distribution(summaries, "gameLength"),
            "passCount": _combine_distribution(summaries, "passCount"),
            "finalMarginP1": _combine_distribution(summaries, "finalMarginP1"),
        },
        "phaseSummary": _combine_phase_summary(summaries),
        "heatmaps": _combine_heatmaps(summaries),
        "limitations": summaries[0].get("limitations", []),
    }
    return combined


def _series_stats(values: Any) -> dict[str, float]:
    torch = _load_torch()
    flat = values.float().view(-1)
    quantiles = torch.quantile(flat, torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9]))
    std = flat.std(unbiased=False) if flat.numel() > 1 else torch.tensor(0.0)
    return {
        "count": int(flat.numel()),
        "mean": _float(flat.mean()),
        "std": _float(std),
        "min": _float(flat.min()),
        "p10": _float(quantiles[0]),
        "p25": _float(quantiles[1]),
        "median": _float(quantiles[2]),
        "p75": _float(quantiles[3]),
        "p90": _float(quantiles[4]),
        "max": _float(flat.max()),
    }


def _combine_series_stats(
    summaries: list[dict[str, Any]],
    key: str,
    *,
    count_key: str = "sampleCount",
) -> dict[str, float]:
    total = sum(s[count_key] for s in summaries)
    if total == 0:
        return {}
    values = [s["metrics"][key] for s in summaries]
    mean = sum(v["mean"] * v["count"] for v in values) / total
    # Quantiles cannot be exactly reconstructed from per-file summaries.
    return {
        "count": total,
        "mean": mean,
        "std": math.sqrt(
            sum((v["std"] ** 2 + (v["mean"] - mean) ** 2) * v["count"] for v in values) / total
        ),
        "min": min(v["min"] for v in values),
        "p10": _weighted_average(values, "p10"),
        "p25": _weighted_average(values, "p25"),
        "median": _weighted_average(values, "median"),
        "p75": _weighted_average(values, "p75"),
        "p90": _weighted_average(values, "p90"),
        "max": max(v["max"] for v in values),
    }


def _weighted_average(values: list[dict[str, float]], field: str) -> float:
    total = sum(v["count"] for v in values)
    return sum(v[field] * v["count"] for v in values) / total if total else 0.0


def _outcome_counts(values: Any) -> dict[str, Any]:
    wins = int((values > 1e-6).sum())
    losses = int((values < -1e-6).sum())
    draws = int((values.abs() <= 1e-6).sum())
    total = wins + losses + draws
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total": total,
        "winRate": wins / total if total else 0.0,
        "lossRate": losses / total if total else 0.0,
        "drawRate": draws / total if total else 0.0,
    }


def _combine_counts(items: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(int(item["wins"]) for item in items)
    losses = sum(int(item["losses"]) for item in items)
    draws = sum(int(item["draws"]) for item in items)
    total = wins + losses + draws
    return {
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "total": total,
        "winRate": wins / total if total else 0.0,
        "lossRate": losses / total if total else 0.0,
        "drawRate": draws / total if total else 0.0,
    }


def _int_histogram(values: Any) -> list[dict[str, Any]]:
    rounded = values.round().long().view(-1)
    unique, counts = rounded.unique(sorted=True, return_counts=True)
    return [
        {"label": str(int(label)), "value": int(count)}
        for label, count in zip(unique.tolist(), counts.tolist())
    ]


def _range_histogram(values: Any, start: float, end: float, bins: int) -> list[dict[str, Any]]:
    torch = _load_torch()
    flat = values.float().view(-1)
    edges = torch.linspace(start, end, bins + 1)
    hist = torch.histc(flat, bins=bins, min=start, max=end)
    result = []
    for idx, count in enumerate(hist.tolist()):
        left = _float(edges[idx])
        right = _float(edges[idx + 1])
        result.append({"label": f"{left:.2f}-{right:.2f}", "value": int(count)})
    return result


def _combine_distribution(summaries: list[dict[str, Any]], key: str) -> list[dict[str, Any]]:
    counts: dict[str, int] = {}
    for summary in summaries:
        for item in summary["distributions"][key]:
            counts[item["label"]] = counts.get(item["label"], 0) + int(item["value"])

    def sort_key(label: str) -> tuple[float, str]:
        try:
            return (float(label), label)
        except ValueError:
            try:
                return (float(label.split("-", 1)[0]), label)
            except ValueError:
                return (0.0, label)

    return [{"label": label, "value": counts[label]} for label in sorted(counts, key=sort_key)]


def _phase_labels(occupied: Any, board_size: int) -> list[str]:
    capacity = board_size * board_size
    labels = []
    for value in occupied.tolist():
        ratio = float(value) / capacity
        if ratio < 0.35:
            labels.append("opening")
        elif ratio < 0.7:
            labels.append("middle")
        else:
            labels.append("late")
    return labels


def _phase_outcomes(labels: list[str], values: Any) -> dict[str, Any]:
    result = {
        "opening": {"label": "前期", "samples": 0, "outcomes": _empty_outcomes()},
        "middle": {"label": "中期", "samples": 0, "outcomes": _empty_outcomes()},
        "late": {"label": "后期", "samples": 0, "outcomes": _empty_outcomes()},
    }
    for phase in result:
        mask = [idx for idx, label in enumerate(labels) if label == phase]
        if not mask:
            continue
        phase_values = values[mask]
        result[phase]["samples"] = int(len(mask))
        result[phase]["outcomes"] = _outcome_counts(phase_values)
    return result


def _combine_phase_summary(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    result = {
        "opening": {"label": "前期", "samples": 0, "outcomes": _empty_outcomes()},
        "middle": {"label": "中期", "samples": 0, "outcomes": _empty_outcomes()},
        "late": {"label": "后期", "samples": 0, "outcomes": _empty_outcomes()},
    }
    for phase in result:
        outcome_parts = []
        samples = 0
        for summary in summaries:
            entry = summary["phaseSummary"][phase]
            samples += int(entry["samples"])
            outcome_parts.append(entry["outcomes"])
        result[phase]["samples"] = samples
        result[phase]["outcomes"] = _combine_counts(outcome_parts)
    return result


def _empty_outcomes() -> dict[str, Any]:
    return {
        "wins": 0,
        "losses": 0,
        "draws": 0,
        "total": 0,
        "winRate": 0.0,
        "lossRate": 0.0,
        "drawRate": 0.0,
    }


def _combine_heatmaps(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    board_sizes = {s["boardSize"] for s in summaries}
    if len(board_sizes) != 1:
        return None
    total = sum(s["sampleCount"] for s in summaries)
    if total == 0:
        return None
    keys = ("policy", "legal", "own", "opponent")
    combined = {}
    for key in keys:
        size = summaries[0]["boardSize"]
        matrix = [[0.0 for _ in range(size)] for _ in range(size)]
        for summary in summaries:
            weight = summary["sampleCount"] / total
            source = summary["heatmaps"][key]
            for row_idx in range(size):
                for col_idx in range(size):
                    matrix[row_idx][col_idx] += source[row_idx][col_idx] * weight
        combined[key] = matrix
    return combined


def _matrix(values: Any) -> list[list[float]]:
    return [[_float(cell) for cell in row] for row in values.tolist()]


def _float(value: Any) -> float:
    if hasattr(value, "item"):
        value = value.item()
    result = float(value)
    if math.isfinite(result):
        return result
    return 0.0


class DataVisualizerHandler(BaseHTTPRequestHandler):
    server_version = "DiffusiveOthelloDataVisualizer/1.0"

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        try:
            if parsed.path == "/":
                self._serve_file(WEB_ROOT / "index.html")
            elif parsed.path == "/api/datasets":
                self._send_json({"datasets": list_datasets(_data_dir_from_server(self.server))})
            elif parsed.path == "/api/summary":
                query = parse_qs(parsed.query)
                paths = resolve_requested_files(_data_dir_from_server(self.server), query)
                summaries = [summarize_dataset(path) for path in paths]
                self._send_json(
                    {
                        "dataDir": str(_data_dir_from_server(self.server)),
                        "selected": [path.name for path in paths],
                        "combined": combine_summaries(summaries),
                        "datasets": summaries,
                    }
                )
            else:
                self._serve_static(parsed.path)
        except FileNotFoundError as exc:
            self._send_error(HTTPStatus.NOT_FOUND, str(exc))
        except Exception as exc:  # pragma: no cover - exercised manually
            self._send_error(HTTPStatus.INTERNAL_SERVER_ERROR, str(exc))

    def log_message(self, format: str, *args: Any) -> None:
        print(f"{self.address_string()} - {format % args}")

    def _serve_static(self, path: str) -> None:
        requested = (WEB_ROOT / path.lstrip("/")).resolve()
        if WEB_ROOT.resolve() not in requested.parents and requested != WEB_ROOT.resolve():
            self._send_error(HTTPStatus.FORBIDDEN, "Forbidden")
            return
        self._serve_file(requested)

    def _serve_file(self, path: Path) -> None:
        if not path.exists() or not path.is_file():
            self._send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        content = path.read_bytes()
        mime_type = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", f"{mime_type}; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_json(self, payload: dict[str, Any]) -> None:
        content = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def _send_error(self, status: HTTPStatus, message: str) -> None:
        content = json.dumps({"error": message}, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the self-play data visualizer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    server = ThreadingHTTPServer((args.host, args.port), DataVisualizerHandler)
    server.data_dir = data_dir  # type: ignore[attr-defined]

    print(f"Diffusive Othello data visualizer")
    print(f"Data directory: {data_dir}")
    print(f"Open: http://{args.host}:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping server.")


if __name__ == "__main__":
    main()
