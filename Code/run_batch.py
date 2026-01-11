#!/usr/bin/env python3

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Sequence


METRIC_KEYS = [
    "RMSE",
    "MAE",
    "MAPE",
    "Wasserstein",
    "Wasserstein_per_dim",
    "Wasserstein_tail_q90",
]

# 默认跑五个种子，可以通过 --seeds 覆盖
DEFAULT_SEEDS = [3407, 3408, 3409, 3410, 3411]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="批量跑多个种子并汇总六个指标，适配 Pixi 任务使用"
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["kdd", "guangzhou", "physio"],
        help="数据集名称",
    )
    parser.add_argument(
        "--model",
        required=True,
        choices=["timesnet", "fgti", "brits", "csdi", "mtsci", "grin", "spin", "imputeformer", "pristi"],
        help="模型名称",
    )
    parser.add_argument(
        "--missing_rate",
        required=True,
        type=float,
        help="缺失率，例如 0.2 / 0.4 / 0.6",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=DEFAULT_SEEDS,
        help="需要跑的随机种子列表，默认 5 个",
    )
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        default=[],
        help="追加传给 Code/train.py 的额外参数",
    )
    args, unknown = parser.parse_known_args()
    # 兼容直接在命令末尾追加超参（无需显式写 --extra）
    extra = list(args.extra) if args.extra else []
    if unknown:
        extra.extend(unknown)
    args.extra = extra
    return args


def extract_metric_from_line(line: str, key: str) -> float:
    """从单行输出中抽取数字，兼容 tensor(0.123, device=...) 形式。"""
    match = re.search(r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", line)
    if not match:
        raise ValueError(f"无法在行中解析 {key}: {line}")
    return float(match.group(1))


def parse_metrics(text: str) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for line in text.splitlines():
        stripped = line.strip()
        for key in METRIC_KEYS:
            if stripped.startswith(f"{key}:"):
                values[key] = extract_metric_from_line(stripped, key)
    return values


def run_one_seed(
    repo_root: Path, model: str, dataset: str, missing_rate: float, seed: int, extra: Sequence[str]
) -> Dict[str, float]:
    cmd = [
        sys.executable,
        str(repo_root / "Code" / "train.py"),
        "--model",
        model,
        "--dataset",
        dataset,
        "--missing_rate",
        str(missing_rate),
        "--seed",
        str(seed),
    ]
    if extra:
        cmd.extend(extra)

    print(f"\n=== 开始 seed={seed} ===")
    env = os.environ.copy()
    env.setdefault("PYTHONUNBUFFERED", "1")  # 确保 tqdm/print 及时刷新

    print(f"\n=== 开始 seed={seed} ===")
    output_lines: List[str] = []
    with subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        text=True,
        bufsize=1,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ) as proc:
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")  # 逐行透传，保留进度条/日志
            output_lines.append(line)
        ret = proc.wait()
        if ret != 0:
            raise RuntimeError(f"seed={seed} 运行失败，返回码 {ret}")

    combined_out = "".join(output_lines)
    metrics = parse_metrics(combined_out)
    missing_keys = [k for k in METRIC_KEYS if k not in metrics]
    if missing_keys:
        raise RuntimeError(
            f"seed={seed} 解析指标失败，缺少: {', '.join(missing_keys)}"
        )
    return metrics


def mean_metrics(metrics_list: List[Dict[str, float]]) -> Dict[str, float]:
    averages: Dict[str, float] = {}
    for key in METRIC_KEYS:
        averages[key] = sum(m[key] for m in metrics_list) / len(metrics_list)
    return averages


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]

    results: List[Dict[str, float]] = []
    for seed in args.seeds:
        metrics = run_one_seed(
            repo_root=repo_root,
            model=args.model,
            dataset=args.dataset,
            missing_rate=args.missing_rate,
            seed=seed,
            extra=args.extra,
        )
        results.append(metrics)

    averages = mean_metrics(results)

    # 构造日志目录与文件名：log/<model>/<dataset>/pXX.txt（日志放在 Code 之外的 log 目录）
    log_dir = repo_root / "log" / args.model / args.dataset
    log_dir.mkdir(parents=True, exist_ok=True)
    miss_tag = f"p{int(args.missing_rate * 100)}"
    log_path = log_dir / f"{miss_tag}.txt"

    header = f"model={args.model}, dataset={args.dataset}, missing_rate={args.missing_rate}, seeds={args.seeds}"
    lines = [header]

    print("\n=== 每个种子结果（保留 3 位小数） ===")
    for seed, m in zip(args.seeds, results):
        line = ", ".join(f"{k}={m[k]:.3f}" for k in METRIC_KEYS)
        print(f"seed={seed}: {line}")
        lines.append(f"seed={seed}: {line}")

    print("\n=== 5 次均值（保留 3 位小数） ===")
    avg_line = ", ".join(f"{k}={averages[k]:.3f}" for k in METRIC_KEYS)
    print(avg_line)
    lines.append(f"avg: {avg_line}")

    with open(log_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n日志已保存到: {log_path}")


if __name__ == "__main__":
    main()
