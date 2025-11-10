from car_sim.config import Config, load_config
from pathlib import Path
from car_sim.garage import Car, Racetrack
import time

import logging
log = logging.getLogger(__name__)


#* Main function definitions

# Define one race step  
def race_step(car: Car, racetrack: Racetrack) -> None:
    if car.current_speed < car.top_speed:
        # Speed in m/s, acceleration in m/s^2
        car.current_speed += car.acceleration_start * ((1 / (1 + car.current_speed / car.top_speed)) - 0.5)

    # Update position in meters
    car.current_distance += car.current_speed
    # Track completion
    car.track_completion = (car.current_distance / (racetrack.length * 1000))
    log.debug("step", extra={"car": f"{car.make} {car.model}", "completion": car.track_completion, "speed": car.current_speed})


def check_winner(car1, car2, cfg) -> tuple[Car,Car] | Car:
    if (car1.track_completion > cfg.laps) and (car2.track_completion > cfg.laps):
        return car1, car2
    elif car1.track_completion * cfg.laps > cfg.laps:
        return car1
    else:
        return car2


def display_results(*args) -> None:
    if len(args) == 2:
        print(
            log.info("tie"),
            f'The {args[0].year} {args[0].make} {args[0].model} tied with the {args[1].year} {args[1].make} {args[1].model}')
    else:
        log.info("winner", extra={"winner": f"{args[0].year} {args[0].make} {args[0].model}"})


# def start_race(Car1, Car2, Racetrack):
def race_start(cfg: Config):
    car1: Car = cfg.car1
    car2: Car = cfg.car2
    track: Racetrack = cfg.racetrack
    log.info("race_start", extra={"track": track.name, "laps": cfg.laps})

    while (car1.track_completion < cfg.laps) and (car2.track_completion < cfg.laps):
        race_step(car1, track)
        race_step(car2, track)
        time.sleep(0.01)
    log.info("race_end", extra={"track": track.name, "laps": cfg.laps})
    winners = check_winner(car1, car2, cfg)
    display_results(winners)
    return winners

