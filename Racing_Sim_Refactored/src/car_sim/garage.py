from dataclasses import dataclass
import time

# Datalasses
@dataclass
class Car:
    make: str = "Default Make"
    model: str = "Default Model"
    year: int = 1886
    color: str = "Black"
    acceleration_start: float = 7.5
    top_speed: float = 180
    current_speed: float = 0
    track_completion: float = 0

    def __post_init__(self) -> None:
        if self.year < 1886:
            raise ValueError("Year cannot be before cars were invented")
        if self.top_speed < 0 or self.acceleration_start < 0:
            raise ValueError("Speed and acceleration must be positive")
        if self.top_speed > 500:
            raise ValueError("Speed cannot be above 500 kph")
        if self.acceleration_start > 30:
            raise ValueError("Acceleration cannot exceed 3 forward Gs")
        
    



@dataclass
class Racetrack:
    name: str = "Nurburgring"
    length: float = 12

    def __post_init__(self) -> None:
        if self.length > 25:
            raise ValueError("Track length cannot exceed 25 km")
    