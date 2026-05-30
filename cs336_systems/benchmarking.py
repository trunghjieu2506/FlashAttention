from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import sys
import timeit
from dataclasses import asdict, dataclass
from typing import Any

import torch
import torch.nn.functional as F

try:
    from cs336_basics.model import BasicsTransformerLM
    from cs336_basics.optimizer import AdamW
except ModuleNotFoundError:
    bundled_basics_path = pathlib.Path(__file__).resolve().parents[1] / "cs336-basics"
    if bundled_basics_path.exists():
        sys.path.insert(0, str(bundled_basics_path))
        from cs336_basics.model import BasicsTransformerLM
        from cs336_basics.optimizer import AdamW
    else:
        raise

MODEL_SPECS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}
MODEL_SIZE_ALIASES = {"10B": "10b"}

DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class BenchmarkConfig:
    vocab_size: int
    batch_size: int
    context_length: int
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int
    warmup_steps: int
    measure_steps: int
    lr: float
    device: str
    dtype: str
    mode: str
    seed: int
    emit_json: bool
    size: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark the basics Transformer forward/backward/optimizer paths.")
    parser.add_argument("--size", choices=[*MODEL_SPECS, *MODEL_SIZE_ALIASES], help="Named model size from the assignment handout.")
    parser.add_argument("--d-model", type=int, help="Model hidden size.")
    parser.add_argument("--d-ff", type=int, help="Feed-forward hidden size.")
    parser.add_argument("--num-layers", type=int, help="Number of Transformer blocks.")
    parser.add_argument("--num-heads", type=int, help="Number of attention heads.")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measure-steps", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=list(DTYPE_MAP), default="float32")
    parser.add_argument(
        "--mode",
        choices=("forward", "forward-backward", "train"),
        default="train",
        help="Benchmark only forward, forward+backward, or the full train step including optimizer.step().",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON instead of a text summary.")
    return parser.parse_args()


def resolve_config(args: argparse.Namespace) -> BenchmarkConfig:
    size = MODEL_SIZE_ALIASES.get(args.size, args.size)
    spec = MODEL_SPECS.get(size, {})
    d_model = args.d_model if args.d_model is not None else spec.get("d_model")
    d_ff = args.d_ff if args.d_ff is not None else spec.get("d_ff")
    num_layers = args.num_layers if args.num_layers is not None else spec.get("num_layers")
    num_heads = args.num_heads if args.num_heads is not None else spec.get("num_heads")

    missing = [
        flag
        for flag, value in (
            ("--d-model", d_model),
            ("--d-ff", d_ff),
            ("--num-layers", num_layers),
            ("--num-heads", num_heads),
        )
        if value is None
    ]
    if missing:
        raise SystemExit(f"Missing model hyperparameters: {', '.join(missing)}. Pass --size or specify all four values.")

    if args.dtype == "float16" and not args.device.startswith("cuda"):
        raise SystemExit("float16 benchmarking requires a CUDA device.")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(f"CUDA device requested ({args.device}) but CUDA is not available in this environment.")
    if args.warmup_steps < 0:
        raise SystemExit("--warmup-steps must be non-negative.")
    if args.measure_steps <= 0:
        raise SystemExit("--measure-steps must be positive.")

    return BenchmarkConfig(
        size=size,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        d_model=d_model,
        d_ff=d_ff,
        num_layers=num_layers,
        num_heads=num_heads,
        warmup_steps=args.warmup_steps,
        measure_steps=args.measure_steps,
        lr=args.lr,
        device=args.device,
        dtype=args.dtype,
        mode=args.mode,
        seed=args.seed,
        emit_json=args.json,
    )


def maybe_sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def build_model(config: BenchmarkConfig, device: torch.device, dtype: torch.dtype) -> BasicsTransformerLM:
    model = BasicsTransformerLM(
        vocab_size=config.vocab_size,
        context_length=config.context_length,
        d_model=config.d_model,
        num_layers=config.num_layers,
        num_heads=config.num_heads,
        d_ff=config.d_ff,
    )
    return model.to(device=device, dtype=dtype)


def make_batch(config: BenchmarkConfig, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (config.batch_size, config.context_length)
    input_ids = torch.randint(config.vocab_size, shape, device=device)
    targets = torch.randint(config.vocab_size, shape, device=device)
    return input_ids, targets


def compute_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    vocab_size = logits.shape[-1]
    return F.cross_entropy(logits.reshape(-1, vocab_size), targets.reshape(-1))


def run_step(
    model: BasicsTransformerLM,
    optimizer: AdamW,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    device: torch.device,
) -> dict[str, float]:
    timings: dict[str, float] = {}

    if mode == "forward":
        maybe_sync(device)
        start = timeit.default_timer()
        with torch.no_grad():
            _ = model(input_ids)
        maybe_sync(device)
        elapsed = timeit.default_timer() - start
        timings["forward"] = elapsed
        timings["step"] = elapsed
        return timings

    optimizer.zero_grad(set_to_none=True)

    maybe_sync(device)
    start = timeit.default_timer()
    logits = model(input_ids)
    loss = compute_loss(logits, targets)
    maybe_sync(device)
    forward_elapsed = timeit.default_timer() - start
    timings["forward"] = forward_elapsed

    maybe_sync(device)
    start = timeit.default_timer()
    loss.backward()
    maybe_sync(device)
    backward_elapsed = timeit.default_timer() - start
    timings["backward"] = backward_elapsed

    if mode == "forward-backward":
        timings["step"] = forward_elapsed + backward_elapsed
        return timings

    maybe_sync(device)
    start = timeit.default_timer()
    optimizer.step()
    maybe_sync(device)
    optimizer_elapsed = timeit.default_timer() - start
    timings["optimizer"] = optimizer_elapsed
    timings["step"] = forward_elapsed + backward_elapsed + optimizer_elapsed
    return timings


def summarize(samples: list[float]) -> dict[str, float]:
    mean = statistics.mean(samples)
    stddev = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return {"mean_s": mean, "std_s": stddev, "mean_ms": mean * 1_000, "std_ms": stddev * 1_000}


def benchmark(config: BenchmarkConfig) -> dict[str, Any]:
    torch.manual_seed(config.seed)
    if config.device.startswith("cuda"):
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device(config.device)
    dtype = DTYPE_MAP[config.dtype]

    try:
        model = build_model(config, device=device, dtype=dtype)
        optimizer = AdamW(model.parameters(), lr=config.lr)
        input_ids, targets = make_batch(config, device=device)

        for _ in range(config.warmup_steps):
            run_step(model, optimizer, input_ids, targets, config.mode, device)

        raw_timings: dict[str, list[float]] = {}
        for _ in range(config.measure_steps):
            step_timings = run_step(model, optimizer, input_ids, targets, config.mode, device)
            for key, value in step_timings.items():
                raw_timings.setdefault(key, []).append(value)
    except RuntimeError as exc:
        if "out of memory" in str(exc).lower():
            raise SystemExit("Benchmark failed with CUDA out-of-memory. Try a smaller model, shorter context, or lower precision.") from exc
        raise

    return {
        "config": asdict(config),
        "num_parameters": model.get_num_params(),
        "results": {name: summarize(samples) for name, samples in raw_timings.items()},
    }


def format_text_report(report: dict[str, Any]) -> str:
    config = report["config"]
    lines = [
        "Benchmark configuration:",
        f"  size={config['size'] or 'custom'} mode={config['mode']} device={config['device']} dtype={config['dtype']}",
        (
            "  "
            f"batch_size={config['batch_size']} context_length={config['context_length']} vocab_size={config['vocab_size']}"
        ),
        (
            "  "
            f"d_model={config['d_model']} d_ff={config['d_ff']} num_layers={config['num_layers']} num_heads={config['num_heads']}"
        ),
        f"  warmup_steps={config['warmup_steps']} measure_steps={config['measure_steps']}",
        f"  num_parameters={report['num_parameters']:,}",
        "",
        "Timings:",
    ]

    ordered_keys = ("forward", "backward", "optimizer", "step")
    for key in ordered_keys:
        summary = report["results"].get(key)
        if summary is None:
            continue
        lines.append(f"  {key:>9}: mean={summary['mean_ms']:.3f} ms std={summary['std_ms']:.3f} ms")
    return "\n".join(lines)


def main() -> int:
    config = resolve_config(parse_args())
    report = benchmark(config)
    if config.emit_json:
        print(json.dumps(report, indent=2))
    else:
        print(format_text_report(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
