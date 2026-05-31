import argparse
import importlib.util
from pathlib import Path


def _load_module_from_file(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _get_algorithm_file(algorithm: str) -> Path:
    alg_dir = Path(__file__).resolve().parent / "alg"
    mapping = {
        "one-shot": alg_dir / "one-shot.py",
        "interation": alg_dir / "interation.py",
    }

    if algorithm not in mapping:
        raise ValueError(f"Unsupported algorithm: {algorithm}")

    target = mapping[algorithm]
    if not target.exists():
        raise FileNotFoundError(f"Algorithm file not found: {target}")
    return target


def run(algorithm: str):
    file_path = _get_algorithm_file(algorithm)
    module_name = f"alg_{algorithm.replace('-', '_')}"
    module = _load_module_from_file(module_name, file_path)

    if not hasattr(module, "Server"):
        raise AttributeError(f"`Server` class not found in {file_path}")

    server = module.Server()
    server.train()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run FedSALT algorithm by parameter."
    )
    parser.add_argument(
        "--alg",
        choices=["one-shot", "interation"],
        default="one-shot",
        help="Algorithm to run: one-shot or interation",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args.alg)
    except KeyboardInterrupt:
        print("训练被手动中断")
