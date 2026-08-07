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
    states = payload["states"].detach().cpu().float()
    legal_masks = payload["legal_masks"].detach().cpu().bool()
    policies = payload["policies"].detach().cpu().float()
    values = payload["values"].detach().cpu().float().view(-1)

    sample_count = int(states.shape[0])
    board_size = int(metadata.get("board_size") or states.shape[-1])
    if sample_count <= 0:
        raise ValueError(f"{path.name} contains no samples.")

    empties = states[:, 0].sum(dim=(1, 2))
    own = states[:, 1].sum(dim=(1, 2))
    opponent = states[:, 2].sum(dim=(1, 2))
    occupied = own + opponent
    move_index_est = torch.clamp(occupied - INITIAL_STONES, min=0)
    piece_diff = own - opponent
    legal_counts = legal_masks.sum(dim=1).float()
    policy_floor = policies.clamp_min(1e-12)
    policy_entropy = -(policy_floor * policy_floor.log()).sum(dim=1)
    top_policy = policies.amax(dim=1)

    estimated_first_turn = (move_index_est.long() % 2) == 0
    estimated_first_value = torch.where(estimated_first_turn, values, -values)

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
        "boardSize": board_size,
        "metrics": {
            "outcomes": _outcome_counts(values),
            "estimatedFirstMoverOutcomes": _outcome_counts(estimated_first_value),
            "value": _series_stats(values),
            "legalMoves": _series_stats(legal_counts),
            "policyEntropy": _series_stats(policy_entropy),
            "topPolicy": _series_stats(top_policy),
            "currentPieceDiff": _series_stats(piece_diff),
            "occupiedCells": _series_stats(occupied),
            "estimatedMoveIndex": _series_stats(move_index_est),
        },
        "distributions": {
            "legalMoves": _int_histogram(legal_counts),
            "estimatedMoveIndex": _int_histogram(move_index_est),
            "currentPieceDiff": _int_histogram(piece_diff),
            "topPolicy": _range_histogram(top_policy, 0.0, 1.0, 10),
            "policyEntropy": _range_histogram(policy_entropy, 0.0, math.log(board_size * board_size), 12),
        },
        "phaseSummary": phase_summary,
        "heatmaps": {
            "policy": _matrix(policy_heatmap),
            "legal": _matrix(legal_heatmap),
            "own": _matrix(own_heatmap),
            "opponent": _matrix(opponent_heatmap),
        },
        "limitations": [
            {
                "metric": "totalGames",
                "label": "总对局数",
                "reason": "当前 .pt 文件只保存训练样本，没有逐局边界。",
            },
            {
                "metric": "gameMoveCounts",
                "label": "单局总步数",
                "reason": "缺少逐局边界；页面展示的是样本所在局面的估算手数分布。",
            },
            {
                "metric": "passMoves",
                "label": "无效/跳过步数",
                "reason": "自对弈生成器在无合法落子时 pass，但不会把 pass 写入训练样本。",
            },
            {
                "metric": "finalMargin",
                "label": "终局目差",
                "reason": "训练样本只有胜负值，没有终局棋盘；页面展示的是当前局面目差。",
            },
            {
                "metric": "firstSecondExact",
                "label": "精确先后手胜率",
                "reason": "样本使用当前玩家视角编码，没有保存绝对玩家；先后手统计按估算手数奇偶样本加权。",
            },
        ],
    }
    _SUMMARY_CACHE[key] = summary
    return summary


def combine_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"sampleCount": 0}

    sample_count = sum(s["sampleCount"] for s in summaries)
    board_sizes = sorted({s["boardSize"] for s in summaries})

    combined = {
        "name": "selected",
        "datasetCount": len(summaries),
        "sampleCount": sample_count,
        "boardSize": board_sizes[0] if len(board_sizes) == 1 else None,
        "boardSizes": board_sizes,
        "metrics": {
            "outcomes": _combine_counts([s["metrics"]["outcomes"] for s in summaries]),
            "estimatedFirstMoverOutcomes": _combine_counts(
                [s["metrics"]["estimatedFirstMoverOutcomes"] for s in summaries]
            ),
            "value": _combine_series_stats(summaries, "value"),
            "legalMoves": _combine_series_stats(summaries, "legalMoves"),
            "policyEntropy": _combine_series_stats(summaries, "policyEntropy"),
            "topPolicy": _combine_series_stats(summaries, "topPolicy"),
            "currentPieceDiff": _combine_series_stats(summaries, "currentPieceDiff"),
            "occupiedCells": _combine_series_stats(summaries, "occupiedCells"),
            "estimatedMoveIndex": _combine_series_stats(summaries, "estimatedMoveIndex"),
        },
        "distributions": {
            "legalMoves": _combine_distribution(summaries, "legalMoves"),
            "estimatedMoveIndex": _combine_distribution(summaries, "estimatedMoveIndex"),
            "currentPieceDiff": _combine_distribution(summaries, "currentPieceDiff"),
            "topPolicy": _combine_distribution(summaries, "topPolicy"),
            "policyEntropy": _combine_distribution(summaries, "policyEntropy"),
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


def _combine_series_stats(summaries: list[dict[str, Any]], key: str) -> dict[str, float]:
    total = sum(s["sampleCount"] for s in summaries)
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
