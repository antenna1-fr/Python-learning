# This repository contains a context manager that opens and reads multiple files at once.
    # Filebundle is a way to open multiple files at once with the guarantee 
    # It is designed to safely close all files even when it encounters issues.
    #  Filebundle tests probe normal closing, error closing, and closing while opening

# This repository also contains a Batcher
    # The point of batcher is to stream parts of a large object, like a csv to save memory
    # Tests ensure it can operate with all sorts of objects, handle errors, and reject nonsense inputs
    # It is also benchmarked in memory and speed versus naive full-object loading
# The goal of this repository is to be a proving ground for my own implementations of maintainable, scalable IO
    # It will be usable entirely from the CL. Commands and documentation will be added here when the project is completed
