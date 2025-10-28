from car_sim.config import Config, load_config
from pathlib import Path
from car_sim.garage import Car, Racetrack
import time

#* Code for running as a script, not CLI
# Navigate to the root to allow easier path selection
current_path = Path(__file__)
root: Path = current_path.parent.parent.parent
# Choose the correct path for the configurations
config_path: Path = root / "config.json"

# Get the config from config.json
cfg: Config = load_config(Path(config_path))

def main():
    race_start(cfg)


#* Main function definitions

# Define one race step  
def race_step(car: Car, racetrack: Racetrack) -> None:
    if car.current_speed < car.top_speed:
        # Speed in m/s, acceleration in m/s^2
        car.current_speed += car.acceleration_start * ((1/(1+car.current_speed/car.top_speed))-0.5)
    
    # Update position in meters
    car.current_distance += car.current_speed
    # Track completion
    car.track_completion = (car.current_distance / (racetrack.length*1000))
    print(f"{car.make} {car.model} {car.year} is at {car.track_completion*100:.2f} % track completion")

def check_winner(car1, car2) -> Car | tuple[Car, Car] | None:
    if (car1.track_completion > cfg.laps) and (car2.track_completion > cfg.laps):
        return car1, car2
    elif car1.track_completion*cfg.laps > cfg.laps:
        return car1
    else:
        return car2

def display_results(*args) -> None:
    if len(args) == 2:
        print(f'The {args[0].year} {args[0].make} {args[0].model} tied with the {args[0].year} {args[1].make} {args[1].model}')
    else:
        print(f'The winner is the {args[0].year} {args[0].make} {args[0].model}')

# def start_race(Car1, Car2, Racetrack):
def race_start(config):
    car1: Car = config.car1
    car2: Car = config.car2
    track: Racetrack= config.racetrack
    print('Race starting!')

    while (car1.track_completion < config.laps) and (car2.track_completion < config.laps):
        race_step(car1, track)
        race_step(car2, track)
        time.sleep(0.01)
    print('Race ending!')
    winners = check_winner(car1, car2)
    display_results(winners)

    
if __name__ == "__main__":
    main()

