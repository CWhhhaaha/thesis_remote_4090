import argparse
from pathlib import Path
import sys

EXPERIMENTS_ROOT = Path(__file__).resolve().parents[1]
if str(EXPERIMENTS_ROOT) not in sys.path:
    sys.path.insert(0, str(EXPERIMENTS_ROOT))

from src.data import build_cifar10_datasets


def main():
    parser = argparse.ArgumentParser(description="Download CIFAR-10 to the local experiments data directory.")
    parser.add_argument("--data-dir", type=str, default="data/cifar10")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.is_absolute():
        data_dir = EXPERIMENTS_ROOT / data_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    train_set, val_set = build_cifar10_datasets(str(data_dir))
    print(f"Prepared CIFAR-10 in {data_dir.resolve()}")
    print(f"Train examples: {len(train_set)}")
    print(f"Test examples: {len(val_set)}")


if __name__ == "__main__":
    main()
