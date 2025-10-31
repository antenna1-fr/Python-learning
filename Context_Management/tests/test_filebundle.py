import io
import os
import pytest
from pathlib import Path
from src.filebundle import FileBundle

"""
Asserts three tests on FileBundle: Normal closes function, exceptions still close files correctly, and errors on opening 
close opened files. 
"""

def write_file(path: Path, text: str = "hello"):
    path.write_text(text, encoding="utf-8")
    return path


def test_normal_closes(tmp_path: Path):

def test_exceptions(tmp_path: Path):

def test_open_error_closes(tmp_path: Path):
