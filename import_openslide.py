import os
import cardiac_globals as cg

OPENSLIDE_PATH = cg.OPENSLIDE_BIN_PATH

if hasattr(os, 'add_dll_directory') and OPENSLIDE_PATH:
    # Windows — requires DLL directory for openslide
    with os.add_dll_directory(OPENSLIDE_PATH):
        import openslide
else:
    # macOS / Linux
    import openslide
