from config import Config, load_config
from pathlib import Path

# Navigate to the root to allow easier path selection
current_path = Path(__file__)
root: Path = current_path.parent.parent.parent
# Choose the correct path for the configurations
config_path: Path = root / "config.json"
bad_config_path: Path = root / "bad_config.json"

# Get the config from config.json
cfg: Config = load_config(Path(config_path))
# def start_race(Car1, Car2, Racetrack):

print(cfg)

# attempt to get the config from bad_config.json
bad_cfg: Config = load_config(Path(bad_config_path))
