from config import Config, load_config
from pathlib import Path
from garage import Car, Racetrack
import time

# Navigate to the root to allow easier path selection
current_path = Path(__file__)
root: Path = current_path.parent.parent.parent
# Choose the correct path for the configurations
config_path: Path = root / "config.json"


# Get the config from config.json
cfg: Config = load_config(Path(config_path))

def main():
    race_start(cfg)

# Define one race step  
def race_step(car, racetrack):
    if car.current_speed < car.top_speed:
        # Speed in m/s, accelration in m/s^2
        car.current_speed += car.acceleration * ((1/(1+car.current_speed/car.top_speed))-0.5)
    
    # Update position in meters
    car.current_position += car.current_speed
    # Track completion
    car.track_completion = racetrack.length/car.current_position

def check_winner(car1, car2):
    if car1.track_completion*cfg.laps > cfg.laps & car2.track_completion * cfg.laps > cfg.laps:
        return (car1, car2)
    elif car1.track_completion*cfg.laps > cfg.laps:
        return car1
    else:
        return car2

def display_results(*args):
    if len(args) == 2:
        print(f'The {args[0].make} {args[0].model} tied with the {args[1].make} {args[1].model}')
    else:
        print(f'The winner is the {args[0].make} {args[0].model}')






# def start_race(Car1, Car2, Racetrack):
def race_start(cfg):
    car1: Car = cfg.car1
    car2: Car = cfg.car2
    track: Racetrack= cfg.racetrack

    while car1.track_completion*cfg.laps < cfg.laps & car2.track_completion * cfg.laps < cfg.laps:
        race_step(car1, track)
        race_step(car2, track)
        time.sleep(0.1)
    winners = check_winner(car1, car2)
    display_results(winners)



    
if __name__ == "__main__":
    main()

