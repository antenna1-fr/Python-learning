from pathlib import Path
import csv
import time
import tracemalloc
import io

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
    except (ValueError, TypeError):
        pass

def run_batcher(csv_paths):
    temperature_sum, precipitation_sum, total_rows = 0.0, 0.0, 0
    start = time.perf_counter()
    tracemalloc.reset_peak()
    with FileBundle(csv_paths) as handles:
        for handle in handles:
            print(f"Processing {handle.name}")
            batcher = Batcher(iterable=csv.reader(handle), batch_size=100, logging_period=500)
            for batch in batcher:
                for row in batch:
                    try:
                        temperature_sum = temperature_sum + float(row[1])
                        precipitation_sum = precipitation_sum + float(row[2])
                        total_rows += 1
                    except (TypeError, ValueError):
                        pass
                    process_row(row)
    elapsed = time.perf_counter() - start  # NEW
    peak_bytes = tracemalloc.get_traced_memory()[1]  # NEW
    return {
        "mode": "batched",
        "rows": total_rows,
        "temp_sum": temperature_sum,
        "precip_sum": precipitation_sum,
        "time_s": elapsed,
        "peak_mb": peak_bytes / (1024 * 1024),
    }

def run_static(csv_paths):
    temperature_sum, precipitation_sum, total_rows = 0.0, 0.0, 0
    start = time.perf_counter()
    tracemalloc.reset_peak()
    with FileBundle(csv_paths) as handles:
        for handle in handles:
            print(f"Processing {handle.name} all at once")
            data = handle.read()
            reader = csv.reader(io.StringIO(data))
            rows = list(reader)

            for row in rows:
                try:
                    temperature_sum = temperature_sum + float(row[1])
                    precipitation_sum = precipitation_sum + float(row[2])
                    total_rows += 1
                except (TypeError, ValueError):
                    pass
                process_row(row)
    elapsed = time.perf_counter() - start  # NEW
    peak_bytes = tracemalloc.get_traced_memory()[1]  # NEW
    return {
        "mode": "all_at_once",
        "rows": total_rows,
        "temp_sum": temperature_sum,
        "precip_sum": precipitation_sum,
        "time_s": elapsed,
        "peak_mb": peak_bytes / (1024 * 1024),
    }

def main ():
    tracemalloc.start()

    results = []
    results.append(run_batcher(csv_paths))
    results.append(run_static(csv_paths))
    if (results[0]["rows"] != results[1]["rows"]
            or results[0]["temp_sum"] != results[1]["temp_sum"]
            or results[0]["precip_sum"] != results[1]["precip_sum"]):
        raise (
            Exception("Batcher and static results do not match"))
    for r in results:
        print(
            f"{r['mode']:>12}: rows={r['rows']:,} "
            f"time={r['time_s']:.3f}s peak_mem={r['peak_mb']:.2f} MiB "
        )
    print(f"Memory advantage: {results[1]['peak_mb']/results[0]['peak_mb']:.1%}")
if __name__ == "__main__":
    main()
