# src/car_sim/cli.py
from __future__ import annotations
import uuid
from pathlib import Path
import logging
import typer
from pydantic import ValidationError

from car_sim.logging_setup import configure_logging
from car_sim.config import load_config, Config
from car_sim.engine import race_start
from car_sim.errors import ConfigError, SimError

app = typer.Typer(add_completion=False, no_args_is_help=True)
__version__ = "0.1.0"

def _resolve(path: Path) -> Path:
    return path if path.is_absolute() else (Path(__file__).resolve().parents[2] / path)

@app.command(help="Run the racing simulation.")
def run(
    config: Path = typer.Option(Path("config.json"), "--config", "-c", exists=False, dir_okay=False, readable=True, help="Path to config JSON."),
    seed: int = typer.Option(None, "--seed", help="Override random seed in config."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable DEBUG console logging."),
    log_file: Path = typer.Option(Path("reports/run.log"), "--log-file", help="Log file path (created if missing)."),
):
    run_id = uuid.uuid4().hex[:8]
    log_path = _resolve(log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Console level per --verbose
    console_level = "DEBUG" if verbose else "INFO"
    configure_logging(str(log_path), console_level, run_id=run_id)
    log = logging.getLogger(__name__)
    log.debug("cli_start", extra={"run_id": run_id})

    try:
        cfg_path = _resolve(config)
        if not cfg_path.exists():
            raise ConfigError(f"Config path does not exist: {cfg_path}")

        cfg: Config = load_config(cfg_path)

        if seed is not None:
            cfg.seed = int(seed)

        race_start(cfg)

    except ValidationError as e:
        logging.getLogger(__name__).error("invalid_config", extra={"detail": str(e).splitlines()[0]})
        raise typer.Exit(code=ConfigError.exit_code)
    except ConfigError as e:
        logging.getLogger(__name__).error("config_error", extra={"detail": str(e)})
        raise typer.Exit(code=e.exit_code)
    except SimError as e:
        logging.getLogger(__name__).error("sim_error", extra={"detail": str(e)})
        raise typer.Exit(code=e.exit_code)
    except Exception as e:
        # last-resort guardrail
        logging.getLogger(__name__).exception("unhandled_exception")
        raise typer.Exit(code=1)

@app.command(help="Show version and exit.")
def version():
    typer.echo(__version__)

