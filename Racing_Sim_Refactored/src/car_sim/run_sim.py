from pathlib import Path
from car_sim.config import load_config
import argparse


def main():
    # Navigate to the root to allow easier path selection
    root = Path(__file__).resolve().parents[2]

    # Declare the parser
    parser = argparse.ArgumentParser(description="Load and display configuration files")

    # Add the config argument to the parser
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to the config file",
        default=root / "config.json",
    )

    # Parse and use the path to gather the correct config
    args = parser.parse_args()
    config_path = args.config if args.config.is_absolute() else (root / args.config)
    cfg = load_config(config_path)

    # Run the simulation with the correct path
    from car_sim.engine import race_start

    race_start(cfg)


if __name__ == "__main__":
    main()
