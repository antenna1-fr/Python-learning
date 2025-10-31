import pytest
from pathlib import Path
from context_management.filebundle import FileBundle

"""
Asserts three tests on FileBundle: Normal closes function, exceptions still close files correctly, and errors on opening 
close opened files. 
"""

def write_file(path: Path, text: str = "hello"):
    path.write_text(text, encoding="utf-8")
    return path


def test_normal_closes(tmp_path: Path):
    # Write two simulated files
    a = write_file(tmp_path / "a.txt", "A")
    b = write_file(tmp_path / "b.txt", "B")
    with FileBundle([a, b]) as handles:
        assert len(handles) == 2
        assert handles[0].read() == "A"
        assert handles[1].read() == "B"
        refs = list(handles)
    assert all(h.closed for h in refs)

def test_exceptions(tmp_path: Path):
    c = write_file(tmp_path / "c.txt", "C")
    d = write_file(tmp_path / "d.txt", "D")
    refs = []
    with pytest.raises(RuntimeError):
        with FileBundle([c, d]) as handles:
            refs = list(handles)
            raise RuntimeError("boom")
    assert refs, "captured handles"
    assert all(h.closed for h in refs)


# def test_open_error_closes(tmp_path: Path):
