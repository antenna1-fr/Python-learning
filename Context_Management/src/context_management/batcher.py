from context_management.progress import progress
import itertools
class Batcher:
    def __init__(self, iterable=None, batch_size=1, logging_period=2):
        self.iterable = iterable or []
        self.batch_size = batch_size
        self.logging_period = logging_period
        self.stream = None
        self.batch_index = 0
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")

    def __iter__(self):
        self.stream = iter(self.iterable)
        self.batch_index = 0
        return self

    def __next__(self):
        batch = list(itertools.islice(self.stream, self.batch_size))
        if not batch:
            raise StopIteration
        self.batch_index += 1
        if self.batch_index % self.logging_period == 0:
            progress(self.batch_index*self.batch_size, None, self.batch_size)
        return batch