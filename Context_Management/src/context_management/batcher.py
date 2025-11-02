from math import floor, ceil
class Batcher:
    def __init__(self, iterable=None, batch_size=1):
        self.iterable = iterable or []
        self.batch_size = batch_size
        if batch_size < 1:
            raise ValueError("Batch size must be at least 1")

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index >= len(self.iterable):
            raise StopIteration
        batch = self.iterable[self.index:self.index + self.batch_size]
        self.index += self.batch_size
        print(f"Batch {floor(self.index / self.batch_size)}/{ceil(len(self.iterable) / self.batch_size)}:")
        return batch

batcher = Batcher()
batcher.iterable = [1, 2, 3, 4, 5, 6, "hi", 10, True]
batcher.batch_size = 3

# Iterate through it and print each batch
for sfa in batcher:
    print(sfa)