from dataclasses import dataclass
import time
@dataclass
class Car:
    make: str = "Default Make"
    model: str = "Default Model"
    year: int = 1886
    color: str = "Black"
    acceleration: float = 10
    top_speed: float = 180
    current_speed: float = 0
    track_completion: float = 0

    def test_drive(self):
        print(f'This {self.year} {self.color} {self.make} {self.model} is test driving!')
    def stop(self):
        print(f'This {self.year} {self.color} {self.make} {self.model} has stopped!')

class Racetrack:
    def __init__(self, name, length, car_1, car_2):
        self.name = name
        self.length = length
        self.car_1 = car_1
        self.car_2 = car_2
        self.winner = None
    def race_start(self):
        print(f'Racing {self.car_1.make} {self.car_1.model} and {self.car_2.make} {self.car_2.model} on {self.name} track!')
        # Simulate
        while self.car_1.track_completion < self.length and self.car_2.track_completion < self.length:
            self.car_1.current_speed += self.car_1.acceleration
            self.car_2.current_speed += self.car_2.acceleration
            if self.car_1.current_speed > self.car_1.top_speed:
                self.car_1.current_speed = self.car_1.top_speed
            if self.car_2.current_speed > self.car_2.top_speed:
                self.car_2.current_speed = self.car_2.top_speed
            self.car_1.track_completion += self.car_1.current_speed * 0.1
            self.car_2.track_completion += self.car_2.current_speed * 0.1
            print(f'{self.car_1.make} {self.car_1.model} is at {self.car_1.track_completion} meters with speed {self.car_1.current_speed} km/h')
            print(f'{self.car_2.make} {self.car_2.model} is at {self.car_2.track_completion} meters with speed {self.car_2.current_speed} km/h')
            time.sleep(.1)
    