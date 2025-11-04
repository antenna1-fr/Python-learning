from pathlib import Path
import csv

from context_management.batcher import Batcher
from context_management.filebundle import FileBundle

csv_paths = [Path("large_data_A.csv"), Path("large_data_B.csv")]

temperature_sum = 0.0
precipitation_sum = 0.0
total_rows = 0

def process_row(row):
    # A sample row processing function that does quick operations on the row
    try:
        row[0] = float(row[0])+1
        row[1] = float(row[1])*2.5
        row[2] = float(row[2])/15
    except (ValueError, TypeError) as e:
        print(e)
        pass

with FileBundle(csv_paths) as handles:
    for handle in handles:
        print(f"Processing {handle.name}")
        batcher = Batcher(iterable=csv.reader(handle), batch_size=1000, logging_period=50)
        for batch in batcher:
            for row in batch:
                try:
                    temperature_sum = temperature_sum + float(row[1])
                    precipitation_sum = precipitation_sum + float(row[2])
                    total_rows += 1
                except (TypeError, ValueError):
                    pass
                process_row(row)
print(f"Average temperature: {temperature_sum/total_rows}")
print(f"Average precipitation: {precipitation_sum/total_rows}")


