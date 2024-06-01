# Rubik's Cube Simulator and Visualizer

This repository contains three modules:

1. `RubiksCube.py`:
    - A Rubik's Cube simulator using a group theory approach to represent the cube and its operations using matrix algebra.
    - The cube is represented as a 1D numpy array, and operations on the cube are represented as permutation arrays.
    - This allows for efficient and mathematically rigorous manipulation of the cube's state.
    - Includes functionality to create, scramble, rotate, undo, redo moves, and save or load the state of the cube.

2. `VisualizeCube.py`:
    - A Rubik's Cube visualizer using matplotlib.
    - Provides a visual representation of the cube's state and interaction through terminal input.
    - Allows users to rotate the cube, scramble it, save the state, and load a saved state through terminal commands.

3. `PermutationMatricesExamples.py`:
    - Provides example lines of code to demonstrate how to work with permutation matrices represented as 1D numpy arrays.
    - Includes examples of converting between permutation arrays and dense numpy matrices, multiplying permutation matrices, transposing permutation matrices, and multiplying permutation matrices onto column vectors.
    - Shows how to represent the identity matrix in this format.

## How to Use

To use the simulator and visualizer, follow these steps:

1. Install the required packages:
    ```bash
    pip install numpy matplotlib
    ```

2. Run the visualizer:
    ```bash
    python VisualizeCube.py
    ```

## Classes and Functions

### `RubiksCube.py`

- **Classes**:
    - `RubiksCube`: A class to represent and manipulate a Rubik's Cube, including methods for rotating, scrambling, undoing, redoing moves, and saving and loading the cube's state.

- **Functions**:
    - `_site_pairs(S: int) -> Tuple[List[np.ndarray], List[np.ndarray]]`: Generates pairs of site indices for rotating planes about the x or z axes.
    - `_create_rotation_operators(S: int) -> np.ndarray`: Constructs rotation operators along the x, y, and z axes.
    - `_create_win_permutations(rotation_operators: np.ndarray) -> np.ndarray`: Generates permutation operators for solved states.

### `VisualizeCube.py`

- **Functions**:
    - `display_cube(cube, ax) -> None`: Displays the current state of the Rubik's Cube using matplotlib.
    - `play() -> None`: Initiates the Rubik's Cube game, prompting the user for the cube size and allowing interaction through terminal input.
    - `main() -> None`: The main function to start the game by prompting the user for the cube size.

### `PermutationMatricesExamples.py`

- **Examples**:
    - Converting a permutation array to a dense numpy matrix representation.
    - Converting a dense numpy matrix back to a permutation array.
    - Multiplying two permutation matrices.
    - Transposing a permutation matrix.
    - Multiplying a permutation matrix onto a column vector.
    - Representing the identity matrix in permutation array format.
