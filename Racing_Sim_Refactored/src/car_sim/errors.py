# src/car_sim/errors.py
class SimError(Exception):
    """Base simulation error."""
    exit_code: int = 1


class ConfigError(SimError):
    """Bad path, unreadable file, or invalid config."""
    exit_code: int = 2


class DataError(SimError):
    """Bad runtime data (e.g., physics/state errors)."""
    exit_code: int = 3
