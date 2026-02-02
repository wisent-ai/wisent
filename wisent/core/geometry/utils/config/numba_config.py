"""Configure numba threading to prevent hangs on macOS."""
import numba
numba.set_num_threads(1)
