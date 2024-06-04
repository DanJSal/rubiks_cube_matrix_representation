"""
Rubik's Cube Simulator

This module implements a Rubik's Cube simulator using a group theory approach to represent the cube and its operations
using matrix algebra. The cube is represented as a 1D numpy array, and operations on the cube are represented as permutation
arrays. This allows for efficient and mathematically rigorous manipulation of the cube's state.

Classes:
    RubiksCube: A class to represent and manipulate a Rubik's Cube, including methods for rotating, scrambling, undoing,
                redoing moves, and checking if the cube is solved. It also supports saving and loading the cube's state.

Functions:
    _site_pairs(S: int) -> Tuple[List[np.ndarray], List[np.ndarray]]: Generates pairs of site indices for rotating planes
                                                                      about the x or z axes.
    _create_rotation_operators(S: int) -> np.ndarray: Constructs rotation operators along the x, y, and z axes.
    _create_win_permutations(rotation_operators: np.ndarray) -> np.ndarray: Generates permutation operators for solved states.
"""

import numpy as np
from copy import deepcopy
from typing import Optional, List, Tuple, Dict


class RubiksCube:
    def __init__(self, S: int, num_scrambles: int = 0, seed: Optional[int] = None):
        """
        Initializes the Rubik's Cube.

        Parameters:
        S (int): The size of the cube (e.g., 3 for a standard 3x3x3 cube).
        num_scrambles (int): Number of random rotations to scramble the cube.
        seed (Optional[int]): Random seed for reproducibility.
        """
        self.size: int = S
        self.base_seed: Optional[int] = seed

        self._rotation_operators: np.ndarray = _create_rotation_operators(self.size)
        self.solved_permutations: np.ndarray = _create_win_permutations(self._rotation_operators)

        self.current_state: np.ndarray = np.arange(6 * self.size ** 2)
        self.all_operations: List[Tuple[str, Tuple[int, int, int]]] = []
        self.undone_operations: List[Tuple[str, Tuple[int, int, int]]] = []
        self.undo_count: int = 0
        self.check_count: int = 0
        self.scramble_count: int = 0

        self.saved_state: Optional[Dict[str, np.ndarray, List[List[Tuple[str, Tuple[int, int, int]]]], int]] = None

        if num_scrambles > 0:
            self.scramble(num_scrambles)

    def scramble(self, num_scrambles: int) -> None:
        """
        Scrambles the cube with a specified number of random rotations.

        Parameters:
        num_scrambles (int): Number of random rotations to scramble the cube.
        """
        if self.base_seed is not None:
            scramble_seed = self.base_seed + self.scramble_count
            np.random.seed(scramble_seed)
        self.scramble_count += 1

        for _ in range(num_scrambles):
            axis = np.random.choice([0, 1, 2])
            plane = np.random.randint(0, self.size)
            direction = np.random.choice([0, 1])
            self._operate(axis, plane, direction)
            self.all_operations.append(('s', (axis, plane, direction)))

    def rotate(self, axis: int, plane: int, direction: int) -> None:
        """
        Performs a single rotation operation on the cube.

        Parameters:
        axis (int): Axis of rotation (0 for x, 1 for y, 2 for z).
        plane (int): Index of the plane to rotate.
        direction (int): Direction of rotation (0 for counter-clockwise, 1 for clockwise).

        Raises:
        ValueError: If the axis, plane, or direction is invalid.
        """
        if axis not in [0, 1, 2]:
            raise ValueError(f"Invalid axis index: {axis}. Options are 0, 1, 2")
        if plane not in range(self.size):
            raise ValueError(
                f"Invalid plane index: {plane}. Options are in closed integer interval [0, {self.size - 1}]")
        if direction not in [0, 1]:
            raise ValueError(f"Invalid direction: {direction}. Options are 0, 1")

        self._operate(axis, plane, direction)
        self.all_operations.append(('r', (axis, plane, direction)))
        self.undone_operations.clear()

    def undo(self, num_moves: int = 1) -> None:
        """
        Undoes the specified number of moves.

        Parameters:
        num_moves (int): Number of moves to undo. Defaults to 1.
        """
        if len(self.all_operations) == 0:
            return
        num_moves = min(num_moves, len(self.all_operations))
        for _ in range(num_moves):
            last_move = self.all_operations.pop()
            move_type, (axis, plane, direction) = last_move
            self._operate(axis, plane, 1 - direction)
            self.undone_operations.append(last_move)
            self.undo_count += 1

    def redo(self, num_moves: int = 1) -> None:
        """
        Redoes the specified number of undone moves.

        Parameters:
        num_moves (int): Number of moves to redo. Defaults to 1.
        """
        if len(self.undone_operations) == 0:
            return
        num_moves = min(num_moves, len(self.undone_operations))
        for _ in range(num_moves):
            last_undone_move = self.undone_operations.pop()
            move_type, (axis, plane, direction) = last_undone_move
            self._operate(axis, plane, direction)
            self.all_operations.append(last_undone_move)

    def check(self) -> bool:
        """
        Checks if the cube is in a solved state.

        Returns:
        bool: True if the cube is solved, False otherwise.
        """
        self.check_count += 1
        return np.any(np.all(self.solved_permutations == self.current_state, axis=1))

    def reset(self) -> None:
        """
        Resets the cube to the solved state and clears all operation logs.
        """
        self.current_state = np.arange(6 * self.size ** 2)
        self.all_operations = []
        self.undone_operations = []
        self.check_count = 0
        self.undo_count = 0

    def save_state(self) -> None:
        """
        Saves the current state of the cube, including all logs and count variables.
        """
        self.saved_state = {
            'current_state': deepcopy(self.current_state),
            'all_operations': deepcopy(self.all_operations),
            'undone_operations': deepcopy(self.undone_operations),
            'undo_count': self.undo_count,
            'check_count': self.check_count,
            'scramble_count': self.scramble_count
        }

    def load_state(self) -> None:
        """
        Loads the saved state of the cube, including all logs and count variables.

        Raises:
        ValueError: If there is no saved state to load.
        """
        if self.saved_state is None:
            raise ValueError("No saved state to load.")

        self.current_state = deepcopy(self.saved_state['current_state'])
        self.all_operations = deepcopy(self.saved_state['all_operations'])
        self.undone_operations = deepcopy(self.saved_state['undone_operations'])
        self.undo_count = self.saved_state['undo_count']
        self.check_count = self.saved_state['check_count']
        self.scramble_count = self.saved_state['scramble_count']

    def _operate(self, axis: int, plane: int, direction: int) -> None:
        """
        Performs a rotation operation on the cube without validation.

        Parameters:
        axis (int): Axis of rotation (0 for x, 1 for y, 2 for z).
        plane (int): Index of the plane to rotate.
        direction (int): Direction of rotation (0 for counter-clockwise, 1 for clockwise).
        """
        rotation_operator = self._rotation_operators[axis, direction, plane]
        self.current_state = self.current_state[rotation_operator]


def _site_pairs(S: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Generate pairs of sites for z-face and x-face swaps for a Rubik's Cube of size S.

    Parameters:
    S (int): The size of the Rubik's Cube (number of squares along one edge).

    Returns:
    Tuple[List[np.ndarray], List[np.ndarray]]: Two lists containing arrays of site pairs for z-face and x-face swaps.
    """
    z_face_swaps = np.array([[2, 0], [0, 3], [3, 1], [1, 2]])
    x_face_swaps = np.array([[2, 5], [5, 3], [3, 4], [4, 2]])
    site_pairs_z = []
    site_pairs_x = []

    for s in range(S):
        these_z_swaps = []
        these_x_swaps = []

        for i in range(S):
            for j in range(4):
                pair_z = z_face_swaps[j]
                pair_x = x_face_swaps[j]

                these_z_swaps.append([[pair_z[0], s, i],
                                      [pair_z[1], s, i]])
                these_x_swaps.append([[pair_x[0], i, s],
                                      [pair_x[1], i, s]])

        if s == 0:
            for p in range(S):
                for q in range(S):
                    these_z_swaps.append([[4, p, q],
                                          [4, q, p]])
                    these_x_swaps.append([[0, p, q],
                                          [0, q, p]])

        if s == S - 1:
            for p in range(S):
                for q in range(S):
                    these_z_swaps.append([[5, p, q],
                                          [5, q, p]])
                    these_x_swaps.append([[1, p, q],
                                          [1, q, p]])

        site_pairs_z.append(np.array(these_z_swaps))
        site_pairs_x.append(np.array(these_x_swaps))

    return site_pairs_z, site_pairs_x


def _create_rotation_operators(S: int) -> np.ndarray:
    """
    Create rotation operators for a Rubik's Cube of size S.

    Parameters:
    S (int): The size of the Rubik's Cube (number of squares along one edge).

    Returns:
    np.ndarray: A 3D array containing rotation operators and their inverses.
    """
    def get_idx(f: int, r: int, c: int) -> int:
        return f * S**2 + r * S + c

    num_squares = 6 * S ** 2
    site_pairs_z, site_pairs_x = _site_pairs(S)

    Z = []
    X = []

    for n in range(S):
        site_pairs_z_n = site_pairs_z[n]
        site_pairs_x_n = site_pairs_x[n]

        idx_pairs_z = []
        idx_pairs_x = []
        used_idxs_z = []
        used_idxs_x = []

        for this_pair in site_pairs_z_n:
            f1, r1, c1 = this_pair[0]
            f2, r2, c2 = this_pair[1]

            idx1 = get_idx(f1, r1, c1)
            idx2 = get_idx(f2, r2, c2)

            idx_pairs_z.append([idx1, idx2])

            if idx1 not in used_idxs_z:
                used_idxs_z.append(idx1)
            if idx2 not in used_idxs_z:
                used_idxs_z.append(idx2)

        for this_pair in site_pairs_x_n:
            f1, r1, c1 = this_pair[0]
            f2, r2, c2 = this_pair[1]

            idx1 = get_idx(f1, r1, c1)
            idx2 = get_idx(f2, r2, c2)

            idx_pairs_x.append([idx1, idx2])

            if idx1 not in used_idxs_x:
                used_idxs_x.append(idx1)
            if idx2 not in used_idxs_x:
                used_idxs_x.append(idx2)

        used_idxs_z = np.sort(np.array(used_idxs_z))
        used_idxs_x = np.sort(np.array(used_idxs_x))

        self_inds_z = np.delete(np.arange(num_squares), used_idxs_z)
        self_inds_x = np.delete(np.arange(num_squares), used_idxs_x)

        idx_pairs_z.extend(np.column_stack((self_inds_z, self_inds_z)).tolist())
        idx_pairs_x.extend(np.column_stack((self_inds_x, self_inds_x)).tolist())

        idx_pairs_z = np.array(idx_pairs_z).T
        idx_pairs_x = np.array(idx_pairs_x).T

        permuted_columns_z = idx_pairs_z[1][np.argsort(idx_pairs_z[0])]
        permuted_columns_x = idx_pairs_x[1][np.argsort(idx_pairs_x[0])]

        Z.append(permuted_columns_z)
        X.append(permuted_columns_x)

    Z, X = np.array(Z), np.array(X)

    Y = []

    Rx = np.arange(6 * S ** 2)
    for i in range(S):
        Rx = Rx[X[i]]
    for i in range(S):
        Y.append(Rx[Z[i][np.argsort(Rx)]])

    X_inv = [np.argsort(X_i) for X_i in X]
    Y_inv = [np.argsort(Y_i) for Y_i in Y]
    Z_inv = [np.argsort(Z_i) for Z_i in Z]

    rotation_operators = np.array([[X, X_inv], [Y, Y_inv], [Z, Z_inv]])

    return rotation_operators


def _create_win_permutations(rotation_operators: np.ndarray) -> np.ndarray:
    """
    Create winning permutations for a Rubik's Cube using the rotation operators.

    Parameters:
    rotation_operators (np.ndarray): A 3D array containing rotation operators and their inverses.

    Returns:
    np.ndarray: A 2D array containing winning permutations.
    """
    X, Y, Z = rotation_operators[0][0], rotation_operators[1][0], rotation_operators[2][0]
    S = len(X)
    num_squares = rotation_operators.shape[-1]

    I = np.arange(num_squares)

    Rx = np.arange(num_squares)
    Ry = np.arange(num_squares)
    Rz = np.arange(num_squares)

    for i in range(S):
        Rx = Rx[X[i]]
        Ry = Ry[Y[i]]
        Rz = Rz[Z[i]]

    permutation_operators = []

    axis_rotations = [I, Rx[Rx], Rx, np.argsort(Rx), Ry, np.argsort(Ry)]

    for permutation_operator in axis_rotations:
        permutation_operators.append(permutation_operator)
        permutation_operators.append(permutation_operator[Rz])
        permutation_operators.append(permutation_operator[Rz[Rz]])
        permutation_operators.append(permutation_operator[Rz[Rz[Rz]]])

    return np.array(permutation_operators)


def main():
    pass


if __name__ == '__main__':
    main()
