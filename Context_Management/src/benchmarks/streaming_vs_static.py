from pathlib import Path

from context_management.batcher import Batcher
from context_management.filebundle import FileBundle

csv_paths = [Path("large_data_A.csv"), Path("large_data_B.csv")]

with FileBundle(csv_paths) as handles:
    print(handles[1].readline())
    pass