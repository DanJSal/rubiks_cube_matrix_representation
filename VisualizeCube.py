"""
Rubik's Cube Visualizer

This module provides a visualization for a Rubik's Cube simulator using matplotlib. It allows for visual representation of the cube's state and interaction through terminal input.

Functions:
    display_cube(cube: rc) -> None:
        Displays the current state of the Rubik's Cube using matplotlib.

    play(S: int, num_scrambles: int = 50) -> None:
        Initiates the Rubik's Cube game, allowing the user to interact with the cube through terminal input.

    main() -> None:
        The main function to start the game with a default cube size of 3x3x3 and no initial scrambles.
"""

import numpy as np
import matplotlib.pyplot as plt
from RubiksCube import RubiksCube as rc


def display_cube(cube: rc) -> None:
    """
    Displays the current state of the Rubik's Cube using matplotlib.

    Parameters:
    cube (rc): An instance of the RubiksCube class representing the current state of the cube.
    """
    side_length = cube.size

    # Define the color vectors for each face
    face_vecs = np.eye(6, dtype=int)
    row_vecs = np.eye(side_length, dtype=int)
    column_vecs = np.eye(side_length, dtype=int)

    initial_state = np.zeros(6 * (side_length ** 2), dtype=int)

    for color, face in enumerate(face_vecs):
        for row in row_vecs:
            for column in column_vecs:
                initial_state += color * np.kron(face, np.kron(row, column))

    cube_state = cube.current_state.toarray() @ initial_state

    plt.close('all')
    fig, axes = plt.subplots(3, 2)

    colors_list = [
        [1., 0.5, 0., 1.],  # Orange
        [1., 0., 0., 1.],  # Red
        [0., 1., 0., 1.],  # Green
        [0., 0., 1., 1.],  # Blue
        [1., 1., 0., 1.],  # Yellow
        [1., 1., 1., 1.]  # White
    ]

    ordered_colors = [colors_list[site_value] for site_value in cube_state]

    x1 = np.array(ordered_colors[:side_length ** 2]).reshape((side_length, side_length, 4))
    x2 = np.array(ordered_colors[side_length ** 2:2 * side_length ** 2]).reshape((side_length, side_length, 4))
    y1 = np.array(ordered_colors[2 * side_length ** 2:3 * side_length ** 2]).reshape((side_length, side_length, 4))
    y2 = np.array(ordered_colors[3 * side_length ** 2:4 * side_length ** 2]).reshape((side_length, side_length, 4))
    z1 = np.array(ordered_colors[4 * side_length ** 2:5 * side_length ** 2]).reshape((side_length, side_length, 4))
    z2 = np.array(ordered_colors[5 * side_length ** 2:6 * side_length ** 2]).reshape((side_length, side_length, 4))

    faces = [(y1, 'front'), (y2, 'back'), (x1, 'left'), (x2, 'right'), (z1, 'bottom'), (z2, 'top')]
    for i, (face, title) in enumerate(faces):
        row, col = divmod(i, 2)
        ax = axes[row, col]
        ax.clear()
        cax = ax.imshow(face, origin='lower')

        ax.set_xticks(np.arange(-0.5, side_length, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, side_length, 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=2)

        ax.tick_params(axis='both', which='both', length=0)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

        for edge, spine in ax.spines.items():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(2)

        ax.set_title(title)

    plt.draw()
    plt.pause(0.001)


def play(S: int, num_scrambles: int = 50) -> None:
    """
    Initiates the Rubik's Cube game, allowing the user to interact with the cube through terminal input.

    Parameters:
    S (int): The size of the cube (e.g., 3 for a standard 3x3x3 cube).
    num_scrambles (int): Number of random rotations to scramble the cube initially. Defaults to 50.
    """
    cube = rc(S, num_scrambles=num_scrambles)
    display_cube(cube)

    while True:
        try:
            user_input = input("Enter move: ")
            if user_input.lower() == 'exit':
                break
            elif user_input.lower() == 'check':
                print(cube.check())
            elif len(user_input.split()) != 3:
                raise ValueError("Invalid input")
            else:
                axis, plane, direction = user_input.split()
                axis = int(axis)
                plane = int(plane)
                direction = int(direction)

                cube.rotate(axis, plane, direction)
                display_cube(cube)

        except Exception as e:
            print(e)
            print(f"An error occurred. Please try again.")


def main() -> None:
    """
    The main function to start the game with a default cube size of 3x3x3 and no initial scrambles.
    """
    S = 3
    play(S, num_scrambles=0)


if __name__ == '__main__':
    main()
