"""
Rubik's Cube Simulator

This module implements a Rubik's Cube simulator using a group theory approach to represent the cube and its operations
using matrix algebra. The cube is represented as a set of permutation matrices, and operations on the cube are represented
as matrix multiplications. This allows for efficient and mathematically rigorous manipulation of the cube's state.

Classes:
    RubiksCube: A class to represent and manipulate a Rubik's Cube, including methods for rotating, scrambling, undoing,
                redoing moves, and checking if the cube is solved. It also supports saving and loading the cube's state.

Functions:
    _site_pairs(S: int) -> Dict[str, np.ndarray]: Finds pairs of site indices for rotating planes about the x or z axes.
    _operator_data(S: int) -> Dict[str, np.ndarray]: Constructs data for creating rotation operators for each plane.
    _create_rotation_operators(S: int) -> Any: Constructs rotation operators along the x, y, and z axes.
    _create_win_permutations(rotation_operators: Any) -> Any: Generates permutation operators for solved states.
"""

import numpy as np
import scipy.sparse as sparse
import copy
from typing import List, Optional, Dict, Any


class RubiksCube:
    def __init__(self, S: int, num_scrambles: int = 50, seed: Optional[int] = None):
        """
        Initializes the Rubik's Cube.

        Parameters:
        S (int): The size of the cube (e.g., 3 for a standard 3x3x3 cube).
        num_scrambles (int): Number of random rotations to scramble the cube.
        seed (Optional[int]): Random seed for reproducibility.
        """
        self.size: int = S
        self.base_seed: Optional[int] = seed

        self._rotation_operators: Any = _create_rotation_operators(self.size)
        self.solved_permutations: Any = _create_win_permutations(self._rotation_operators)

        self.current_state: sparse.csr_matrix = sparse.eye(6 * self.size ** 2, dtype=int, format='csr')
        self.scramble_operations: List[List[int]] = []
        self.user_operations: List[List[int]] = []
        self.undone_operations: List[List[int]] = []
        self.undo_count: int = 0
        self.check_count: int = 0
        self.scramble_count: int = 0

        self.saved_state: Optional[Dict[str, Any]] = None

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

        self.current_state = sparse.eye(6 * self.size ** 2, dtype=int, format='csr')
        self.scramble_operations = []
        self.user_operations = []
        self.undone_operations = []
        self.check_count = 0
        self.undo_count = 0
        for _ in range(num_scrambles):
            axis = np.random.choice([0, 1, 2])
            plane = np.random.randint(0, self.size)
            direction = np.random.choice([-1, 1])
            self._operate(axis, plane, direction)
            self.scramble_operations.append([axis, plane, direction])

    def rotate(self, axis: int, plane: int, direction: int) -> None:
        """
        Performs a single rotation operation on the cube.

        Parameters:
        axis (int): Axis of rotation (0 for x, 1 for y, 2 for z).
        plane (int): Index of the plane to rotate.
        direction (int): Direction of rotation (1 for clockwise, -1 for counter-clockwise).

        Raises:
        ValueError: If the axis, plane, or direction is invalid.
        """
        if axis not in [0, 1, 2]:
            raise ValueError(f"Invalid axis index: {axis}. Options are 0, 1, 2")
        if plane not in range(self.size):
            raise ValueError(
                f"Invalid plane index: {plane}. Options are in closed integer interval [0, {self.size - 1}]")
        if direction not in [-1, 1]:
            raise ValueError(f"Invalid direction: {direction}. Options are -1, 1")

        self._operate(axis, plane, direction)
        self.user_operations.append([axis, plane, direction])
        self.undone_operations.clear()

    def undo(self, num_moves: int = 1) -> None:
        """
        Undoes the specified number of moves.

        Parameters:
        num_moves (int): Number of moves to undo. Defaults to 1.
        """
        if len(self.user_operations) == 0:
            return
        num_moves = min(num_moves, len(self.user_operations))
        for _ in range(num_moves):
            last_move = self.user_operations.pop()
            axis, plane, direction = last_move
            self._operate(axis, plane, -direction)
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
            axis, plane, direction = last_undone_move
            self._operate(axis, plane, direction)
            self.user_operations.append(last_undone_move)

    def check(self) -> bool:
        """
        Checks if the cube is in a solved state.

        Returns:
        bool: True if the cube is solved, False otherwise.
        """
        self.check_count += 1
        for permutation in self.solved_permutations:
            difference = permutation - self.current_state
            delta = np.sum(np.abs(difference.data))
            if delta == 0:
                return True
        return False

    def reset(self) -> None:
        """
        Resets the cube to the solved state and clears all operation logs.
        """
        self.current_state = sparse.eye(6 * self.size ** 2, dtype=int, format='csr')
        self.user_operations = []
        self.undone_operations = []
        self.check_count = 0
        self.undo_count = 0

    def save_state(self) -> None:
        """
        Saves the current state of the cube, including all logs and count variables.
        """
        self.saved_state = {
            'current_state': copy.deepcopy(self.current_state),
            'scramble_operations': copy.deepcopy(self.scramble_operations),
            'user_operations': copy.deepcopy(self.user_operations),
            'undone_operations': copy.deepcopy(self.undone_operations),
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

        self.current_state = copy.deepcopy(self.saved_state['current_state'])
        self.scramble_operations = copy.deepcopy(self.saved_state['scramble_operations'])
        self.user_operations = copy.deepcopy(self.saved_state['user_operations'])
        self.undone_operations = copy.deepcopy(self.saved_state['undone_operations'])
        self.undo_count = self.saved_state['undo_count']
        self.check_count = self.saved_state['check_count']
        self.scramble_count = self.saved_state['scramble_count']

    def _operate(self, axis: int, plane: int, direction: int) -> None:
        """
        Performs a rotation operation on the cube without validation.

        Parameters:
        axis (int): Axis of rotation (0 for x, 1 for y, 2 for z).
        plane (int): Index of the plane to rotate.
        direction (int): Direction of rotation (1 for clockwise, -1 for counter-clockwise).
        """
        rotation_operator = self._rotation_operators[axis][plane] if direction == 1 else self._rotation_operators[axis][
            plane].transpose()
        self.current_state = rotation_operator @ self.current_state


def _site_pairs(S: int) -> Dict[str, np.ndarray]:
    """
    Finds the pairs of site indices for rotating discrete planes about the x or z axes for a cube of dimensions SxSxS.
    These axes are referred to respectively as the horizontal (h) and vertical (v) axes as the legacy code has not been
    updated to reflect the current use of X, Y, and Z for labeling the three axes.

    Parameters:
    S (int): The size of the cube.

    Returns:
    Dict[str, np.ndarray]: A dictionary of site pairs for vertical and horizontal swaps.
    """
    vert_strip_swaps = np.array([[2, 0], [0, 3], [3, 1], [1, 2]])
    hor_strip_swaps = np.array([[2, 5], [5, 3], [3, 4], [4, 2]])
    site_pairs = {}

    for s in range(S):
        these_vert_swaps = []
        these_hor_swaps = []

        for i in range(S):
            for j in range(4):
                pair_v = vert_strip_swaps[j]
                pair_h = hor_strip_swaps[j]

                these_vert_swaps.append([[pair_v[0], s, i],
                                         [pair_v[1], s, i]])
                these_hor_swaps.append([[pair_h[0], i, s],
                                        [pair_h[1], i, s]])

        if s == 0:
            for p in range(S):
                for q in range(S):
                    these_vert_swaps.append([[4, p, q],
                                             [4, q, p]])
                    these_hor_swaps.append([[0, p, q],
                                            [0, q, p]])

        if s == S - 1:
            for p in range(S):
                for q in range(S):
                    these_vert_swaps.append([[5, p, q],
                                             [5, q, p]])
                    these_hor_swaps.append([[1, p, q],
                                            [1, q, p]])

        site_pairs[f"v{s}_swaps"] = np.array(these_vert_swaps)
        site_pairs[f"h{s}_swaps"] = np.array(these_hor_swaps)

    return site_pairs


def _operator_data(S: int) -> Dict[str, np.ndarray]:
    """
    Calls '_site_pairs' to construct the data used in creating the rotation operators for each discrete plane along the
    x and z axes. This data is formatted for scipy sparse csr matrix representations of these operators. Operators
    permute the sites involved in the rotation of a particular plane, leaving all other sites on the cube unchanged.

    Parameters:
    S (int): The size of the cube.

    Returns:
    Dict[str, np.ndarray]: A dictionary of operator pairs for vertical and horizontal planes.
    """
    num_squares = 6 * S ** 2
    indexing_array = np.arange(num_squares).reshape((6, S, S))
    site_pairs_dict = _site_pairs(S)

    operator_pairs_dict = {}

    for n in range(S):
        site_pairs_v = site_pairs_dict[f"v{n}_swaps"]
        site_pairs_h = site_pairs_dict[f"h{n}_swaps"]

        idx_pairs_v = []
        idx_pairs_h = []
        used_idxs_v = []
        used_idxs_h = []

        for this_pair in site_pairs_v:
            f1, r1, c1 = this_pair[0]
            f2, r2, c2 = this_pair[1]

            idx1 = indexing_array[f1, r1, c1]
            idx2 = indexing_array[f2, r2, c2]

            idx_pairs_v.append([idx1, idx2])

            if idx1 not in used_idxs_v:
                used_idxs_v.append(idx1)
            if idx2 not in used_idxs_v:
                used_idxs_v.append(idx2)

        for this_pair in site_pairs_h:
            f1, r1, c1 = this_pair[0]
            f2, r2, c2 = this_pair[1]

            idx1 = indexing_array[f1, r1, c1]
            idx2 = indexing_array[f2, r2, c2]

            idx_pairs_h.append([idx1, idx2])

            if idx1 not in used_idxs_h:
                used_idxs_h.append(idx1)
            if idx2 not in used_idxs_h:
                used_idxs_h.append(idx2)

        used_idxs_v = np.sort(np.array(used_idxs_v))
        used_idxs_h = np.sort(np.array(used_idxs_h))

        self_inds_v = np.delete(np.arange(num_squares), used_idxs_v)
        self_inds_h = np.delete(np.arange(num_squares), used_idxs_h)

        for j in range(self_inds_v.size):
            idx_pairs_v.append([self_inds_v[j], self_inds_v[j]])
        for j in range(self_inds_h.size):
            idx_pairs_h.append([self_inds_h[j], self_inds_h[j]])

        idx_pairs_v_arr = np.empty((3, num_squares), dtype=int)
        idx_pairs_v_arr[0, :] = np.repeat(1, num_squares)
        idx_pairs_v_arr[1:, :] = np.array(idx_pairs_v).T

        idx_pairs_h_arr = np.empty((3, num_squares), dtype=int)
        idx_pairs_h_arr[0, :] = np.repeat(1, num_squares)
        idx_pairs_h_arr[1:, :] = np.array(idx_pairs_h).T

        operator_pairs_dict[f"v_plane_{n}"] = idx_pairs_v_arr
        operator_pairs_dict[f"h_plane_{n}"] = idx_pairs_h_arr

    return operator_pairs_dict


def _create_rotation_operators(S: int) -> Any:
    """
    Calls '_operator_data' and constructs the rotation operators along the x, y, and z axes. Each axis is associated
    with S independent operators as each axis has S discrete planes for an SxSxS cube. The X and Z operator sets are
    constructed directly from the data given by '_operator_data'. The Y operators are then constructed by rotating the
    entire cube about the x-axis so that the y-axis is now aligned with the z-axis, performing rotations with the
    Z operators, and performing an inverse X rotation of the entire cube to realign the principle axes.

    Parameters:
    S (int): The size of the cube.

    Returns:
    Any: A tuple of tuples containing the rotation operators for each axis.
    """
    num_squares = 6 * S ** 2

    pairs_dict = _operator_data(S)

    X, Y, Z = [], [], []
    for n in range(S):
        move_pairs_v = pairs_dict[f"v_plane_{n}"]
        move_pairs_h = pairs_dict[f"h_plane_{n}"]

        X_n = sparse.csr_matrix((move_pairs_h[0], (move_pairs_h[1], move_pairs_h[2])), shape=(num_squares, num_squares))
        Z_n = sparse.csr_matrix((move_pairs_v[0], (move_pairs_v[1], move_pairs_v[2])), shape=(num_squares, num_squares))

        X.append(X_n)
        Z.append(Z_n)

    operator_size = X[0].shape[0]
    Rx = sparse.identity(operator_size, dtype=int, format='csr')
    for i in range(S):
        Rx = X[i] @ Rx
    for i in range(S):
        Y.append(Rx.transpose() @ Z[i] @ Rx)

    return tuple([tuple(X), tuple(Y), tuple(Z)])


def _create_win_permutations(rotation_operators: Any) -> Any:
    """
    Generates the matrices which represent 'win' states of the cube. These include the identity matrix (representing an
    unscrambled cube) and 23 other matrices which simply represent all other possible orientations of the solved cube
    when the principle axes are rotated by some integer multiples of 90 degrees.

    Parameters:
    rotation_operators (Any): The rotation operators of the cube.

    Returns:
    Any: A tuple containing the permutation operators for all solved states of the cube.
    """
    X, Y, Z = rotation_operators
    S = len(X)
    num_squares = X[0].shape[0]

    I = sparse.identity(num_squares, dtype=int, format='csr')

    Rx = sparse.identity(num_squares, dtype=int, format='csr')
    Ry = sparse.identity(num_squares, dtype=int, format='csr')
    Rz = sparse.identity(num_squares, dtype=int, format='csr')

    for i in range(S):
        Rx = X[i] @ Rx
        Ry = Y[i] @ Ry
        Rz = Z[i] @ Rz

    permutation_operators = []

    axis_rotations = [I, Rx @ Rx, Rx, Rx.transpose(), Ry, Ry.transpose()]

    for permutation_operator in axis_rotations:
        permutation_operators.append(permutation_operator)
        permutation_operators.append(Rz @ permutation_operator)
        permutation_operators.append(Rz @ Rz @ permutation_operator)
        permutation_operators.append(Rz @ Rz @ Rz @ permutation_operator)

    return tuple(permutation_operators)


def main():
    pass


if __name__ == '__main__':
    main()
