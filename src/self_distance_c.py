"""
self_distance_c.py

Python wrapper for libself_distance_c.so.
Exposes the same function signature as self_distance.py.

Build the shared library with:
    clang++ -O3 -Wall -dynamiclib self_distance_c.cpp -o libself_distance_c.so

Modelled on morphometry.py.
"""

import ctypes
import os
import numpy as np

# ---------------------------------------------------------------------------
# Load the shared library from the same directory as this file.
# ---------------------------------------------------------------------------
_lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         'libself_distance_c.so')
_lib = ctypes.CDLL(_lib_path)

# ---------------------------------------------------------------------------
# Declare the C function signature.
#
# int check_new_positions_do_not_cause_overlaps(
#     int     num_strands,
#     int    *strand_lengths,
#     double *all_data_x,
#     double *all_data_y,
#     double *all_data_z,
#     int     num_A,
#     int    *A_strand,
#     int    *A_j,
#     double *new_positions_x,
#     double *new_positions_y,
#     double *new_positions_z,
#     int     config_closed,
#     int     skippedInteger,
#     double  upper_bound
# );
# ---------------------------------------------------------------------------
_lib.check_new_positions_do_not_cause_overlaps.argtypes = [
    ctypes.c_int,                      # num_strands
    ctypes.POINTER(ctypes.c_int),      # strand_lengths
    ctypes.POINTER(ctypes.c_double),   # all_data_x
    ctypes.POINTER(ctypes.c_double),   # all_data_y
    ctypes.POINTER(ctypes.c_double),   # all_data_z
    ctypes.c_int,                      # num_A
    ctypes.POINTER(ctypes.c_int),      # A_strand
    ctypes.POINTER(ctypes.c_int),      # A_j
    ctypes.POINTER(ctypes.c_double),   # new_positions_x
    ctypes.POINTER(ctypes.c_double),   # new_positions_y
    ctypes.POINTER(ctypes.c_double),   # new_positions_z
    ctypes.c_int,                      # config_closed  (1=closed, 0=open)
    ctypes.c_int,                      # skippedInteger
    ctypes.c_double,                   # upper_bound
]
_lib.check_new_positions_do_not_cause_overlaps.restype = ctypes.c_int


def _ptr_int(arr):
    """Return a ctypes int pointer to a numpy int32 array."""
    return ctypes.cast(arr.ctypes.data, ctypes.POINTER(ctypes.c_int))


def _ptr_dbl(arr):
    """Return a ctypes double pointer to a numpy float64 array."""
    return ctypes.cast(arr.ctypes.data, ctypes.POINTER(ctypes.c_double))


def check_new_positions_do_not_cause_overlaps(
        data, configType, skippedInteger, newPositions, indices,
        upper_bound_closest_self_distance):
    """
    Python wrapper matching the signature of self_distance.check_new_positions_do_not_cause_overlaps.

    Returns 1 if no overlap detected, 0 if overlap detected.

    data           : nested list; data[i][j] is a numpy array [x, y, z]
    configType     : 'open' or 'closed'
    skippedInteger : int
    newPositions   : list of numpy arrays — new positions for indices[1:-1:]
    indices        : full interval list including fixed endpoints
                     Note: indices[0] or indices[-1] may be dummy tuples (i,'x',n)
                     for open curves; these are NOT passed to C++.
    upper_bound_closest_self_distance : float threshold
    """
    num_strands = len(data)

    # Strand lengths
    strand_lengths_np = np.array([len(data[i]) for i in range(num_strands)],
                                  dtype=np.int32)

    # Flatten all data positions into x/y/z arrays (strand-major order)
    total_pts = int(strand_lengths_np.sum())
    all_data_x = np.empty(total_pts, dtype=np.float64)
    all_data_y = np.empty(total_pts, dtype=np.float64)
    all_data_z = np.empty(total_pts, dtype=np.float64)
    idx = 0
    for i in range(num_strands):
        for j in range(len(data[i])):
            all_data_x[idx] = data[i][j][0]
            all_data_y[idx] = data[i][j][1]
            all_data_z[idx] = data[i][j][2]
            idx += 1

    # Interior moved points: A = indices[1:-1:]
    # These are always real (strand, vertex_index) pairs — never dummy.
    A = indices[1:-1:]
    num_A = len(A)

    A_strand_np = np.array([p[0] for p in A], dtype=np.int32)
    A_j_np      = np.array([p[1] for p in A], dtype=np.int32)

    new_pos_x = np.array([newPositions[s][0] for s in range(num_A)], dtype=np.float64)
    new_pos_y = np.array([newPositions[s][1] for s in range(num_A)], dtype=np.float64)
    new_pos_z = np.array([newPositions[s][2] for s in range(num_A)], dtype=np.float64)

    config_closed = 1 if configType == 'closed' else 0

    result = _lib.check_new_positions_do_not_cause_overlaps(
        ctypes.c_int(num_strands),
        _ptr_int(strand_lengths_np),
        _ptr_dbl(all_data_x),
        _ptr_dbl(all_data_y),
        _ptr_dbl(all_data_z),
        ctypes.c_int(num_A),
        _ptr_int(A_strand_np),
        _ptr_int(A_j_np),
        _ptr_dbl(new_pos_x),
        _ptr_dbl(new_pos_y),
        _ptr_dbl(new_pos_z),
        ctypes.c_int(config_closed),
        ctypes.c_int(int(skippedInteger)),
        ctypes.c_double(float(upper_bound_closest_self_distance)),
    )
    return int(result)
