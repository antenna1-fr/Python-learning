from pathlib import Path

from context_management.batcher import Batcher
from context_management.filebundle import FileBundle

csv_paths = [Path("large_data_A.csv"), Path("large_data_B.csv")]

with FileBundle(csv_paths) as handles:
    for handle in handles:
        print(f"Processing {handle.name}")
        batcher = Batcher(iterable=handle, batch_size=1000, logging_period=50)
        for batch in batcher:

            pass