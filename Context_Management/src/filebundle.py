from pathlib import Path
from typing import Union, List, TextIO


class FileBundle:
    """
    FileBundle (Paths)

    Usage:
    with FileBundle(["a.txt", "b.txt"]) as files:
        data_a = files[0].read()
        data_b = files[1].read()

    Ensures opened files are closed when the with block ends or when an exception is encountered. Leak-proof IO.
    """
    def __init__(self, paths: List[Union[str, Path]]):
        self.paths = [Path(p) for p in paths]
        self.handles: List[TextIO] = []

    @property
    def __enter__(self) -> list[TextIO]:
        try:
            for path in self.paths:
                f = path.open("r")
                self.handles.append(f)
        except Exception as e:
            for handle in self.handles:
                try:
                    handle.close()
                except Exception:
                    print(f"error closing {handle}")
                    pass
            raise e
        return self.handles


    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for handle in self.handles:
            try:
                handle.close()
            except Exception:
                pass




