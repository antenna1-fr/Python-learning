from math import floor, ceil

def progress(count: int, total: int | None, batch_size: int):
    batch_num = floor(count / batch_size)
    if total is None:
        # Unknown total: just show what we know
        print(f"Batch {batch_num} completed ({count} items processed so far)")
    else:
        # Known total: show full progress info
        total_batches = ceil(total / batch_size)
        print(
            f"Batch {batch_num}/{total_batches} completed "
            f"(item {min(count, total)}/{total})"
        )
