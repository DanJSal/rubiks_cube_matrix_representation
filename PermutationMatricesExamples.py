"""
This module provides example lines of code to demonstrate how to work with permutation matrices
represented as 1D numpy arrays. It includes examples of converting between permutation arrays and
dense numpy matrices, multiplying permutation matrices, transposing permutation matrices, and
multiplying permutation matrices onto column vectors. Additionally, it shows how to represent
the identity matrix in this format.
"""

import numpy as np


# Example of converting a permutation array to a dense numpy matrix representation

# Given permutation array P
P = np.array([2, 0, 1])

# Convert permutation array to a dense matrix
n = len(P)
dense = np.zeros((n, n), dtype=int)
dense[np.arange(n), P] = 1

print("Permutation array to dense matrix:")
print(dense)

# Example of converting a dense numpy matrix back to a permutation array

# Given dense permutation matrix
dense = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 0, 0]
])

# Convert dense matrix to permutation array
P_back = np.argmax(dense, axis=1)

print("Dense matrix back to permutation array:")
print(P_back)

# Example of multiplying two permutation matrices

# Given permutation arrays P1 and P2
P1 = np.array([2, 0, 1])
P2 = np.array([1, 2, 0])

# Multiplying P1 from the left by P2 (P2 @ P1)
P_mult = P1[P2]

print("Multiplying permutation matrices P2 @ P1:")
print(P_mult)

# Example of transposing a permutation matrix

# Given permutation array P
P = np.array([2, 0, 1])

# Transposing permutation matrix (finding inverse)
P_transpose = np.argsort(P)

print("Transposing permutation matrix:")
print(P_transpose)

# Example of multiplying a permutation matrix onto a column vector

# Given permutation array P
P = np.array([2, 0, 1])

# Given column vector v
v = np.array([10, 20, 30])

# Permutation matrix multiplying column vector (P @ v)
v_permuted = v[P]

print("Multiplying permutation matrix onto column vector:")
print(v_permuted)

# Example of representing the identity matrix

# Identity matrix of size n x n
n = 3
identity = np.arange(n)

print("Identity matrix representation:")
print(identity)
