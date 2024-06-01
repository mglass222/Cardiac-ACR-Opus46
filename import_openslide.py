import os
import cardiac_globals as cg 

OPENSLIDE_PATH = cg.OPENSLIDE_BIN_PATH

if hasattr(os, 'add_dll_directory'):
    # Windows
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    # Linux
    import openslide
    