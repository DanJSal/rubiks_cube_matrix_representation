# Rubik's Cube Simulator and Visualizer

This repository contains two modules:

1. `RubiksCube.py`:
    - A Rubik's Cube simulator using a group theory approach to represent the cube and its operations using matrix algebra.
    - The cube is represented as a set of permutation matrices, and operations on the cube are represented as matrix multiplications.
    - This allows for efficient and mathematically rigorous manipulation of the cube's state.

2. `VisualizeCube.py`:
    - A Rubik's Cube visualizer using matplotlib.
    - Provides a visual representation of the cube's state and interaction through terminal input.

## How to Use

To use the simulator and visualizer, follow these steps:

1. Install the required packages:
    ```bash
    pip install numpy scipy matplotlib
    ```

2. Run the visualizer:
    ```bash
    python VisualizeCube.py
    ```

## Classes and Functions

### `RubiksCube.py`

- **Classes**:
    - `RubiksCube`: A class to represent and manipulate a Rubik's Cube, including methods for rotating, scrambling, undoing, redoing moves, and checking if the cube is solved. It also supports saving and loading the cube's state.

- **Functions**:
    - `_site_pairs(S: int) -> Dict[str, np.ndarray]`: Finds pairs of site indices for rotating planes about the x or z axes.
    - `_operator_data(S: int) -> Dict[str, np.ndarray]`: Constructs data for creating rotation operators for each plane.
    - `_create_rotation_operators(S: int) -> Any`: Constructs rotation operators along the x, y, and z axes.
    - `_create_win_permutations(rotation_operators: Any) -> Any`: Generates permutation operators for solved states.

### `VisualizeCube.py`

- **Functions**:
    - `display_cube(cube, ax) -> None`: Displays the current state of the Rubik's Cube using matplotlib.
    - `play(S: int, num_scrambles: int = 50) -> None`: Initiates the Rubik's Cube game, allowing the user to interact with the cube through terminal input.
    - `main() -> None`: The main function to start the game with a default cube size of 3x3x3 and no initial scrambles.
