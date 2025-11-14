# This refactored version of a simple racecar simulator I made to teach myself object-oriented programming 
# expands on the original concept with dataclasses, typechecking, and a project structure that lends itself to
# easy expansion. It includes config verification, cli usage, and a more realistic simulation. 

# The code has been modularized, so the relevant classes live in separate files from one another and from the execution code
# use run-sim and config_tester to run and test config files. Both support the --config flag. The older cli commands have been
# deprecated. They've been replaced by car-sim, which is below

# To install: 
pip install -e .


# CLI help
car-sim --help
car-sim run --help
car-sim version

# Example
car-sim run --config config.json
Config flag optional, defaults to config.json. Allows user to select config file.
# Logs: reports/run.log (key=value lines with run_id)
Highly granular logging, with important messages printed to the console.
