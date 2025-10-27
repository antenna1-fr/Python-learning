from pathlib import Path
import argparse
from src.car_sim.config import Config, load_config

root = Path(__file__).resolve().parents[2]
parser= argparse.ArgumentParser(description="Load and test configurations")
parser.add_argument('--config', type=Path, help='path to the config file', default=root / 'config.json')

# Choose the correct path for the configurations
config_path = parser.parse_args().config
try:
    cfg: Config = load_config(config_path)
    print('config loaded succesfully!')
except Exception as e:
    print(f'Bad config: \n {e}')