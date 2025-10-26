from config import Config, load_config
from pathlib import Path
from garage import Car, Racetrack

# Navigate to the root to allow easier path selection
current_path = Path(__file__)
root: Path = current_path.parent.parent.parent
# Choose the correct path for the configurations
config_path: Path = root / "config.json"


# Get the config from config.json
cfg: Config = load_config(Path(config_path))

# Get the classes from the configs and assign them to variables
car1: Car = cfg.car1
car2: Car = cfg.car2
track: Racetrack= cfg.racetrack

# Define one race step
def race_step(car, racetrack):
    if car.current_speed < car.top_speed:
        # Speed in m/s, accelration in m/s^2
        car.current_speed += car.acceleration * ((1/(1+car.current_speed/car.top_speed))-0.5)
    
    # Update position in meters
    car.current_position += car.current_speed
    # Track completion
    car.track_completion = racetrack.length/car.current_position


# def start_race(Car1, Car2, Racetrack):
def race_start(cfg):
    
    
