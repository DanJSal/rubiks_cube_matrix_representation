"""
Rubik's Cube Visualizer

This module provides a visualization for a Rubik's Cube simulator using matplotlib. It allows for visual representation
of the cube's state and interaction through terminal input.

Functions:
    display_cube(cube, ax) -> None:
        Displays the current state of the Rubik's Cube using matplotlib.

    play(S: int, num_scrambles: int = 50) -> None:
        Initiates the Rubik's Cube game, allowing the user to interact with the cube through terminal input.

    main() -> None:
        The main function to start the game with a default cube size of 3x3x3 and no initial scrambles.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from RubiksCube import RubiksCube as rc


def display_cube(cube, ax) -> None:
    """
    Displays the current state of the Rubik's Cube using matplotlib in a conventional 2D map.

    Parameters:
    cube (rc): An instance of the RubiksCube class representing the current state of the cube.
    ax (matplotlib.axes.Axes): The axes on which to plot the cube.
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

    # Define the colors for each face
    colors_list = [
        [1., 0.5, 0., 1.],  # Orange
        [1., 0., 0., 1.],   # Red
        [0., 1., 0., 1.],   # Green
        [0., 0., 1., 1.],   # Blue
        [1., 1., 0., 1.],   # Yellow
        [1., 1., 1., 1.]    # White
    ]

    ordered_colors = [colors_list[site_value] for site_value in cube_state]

    # Define each face's colors
    faces = {
        'left': np.array(ordered_colors[:side_length ** 2]).reshape((side_length, side_length, 4)),
        'right': np.array(ordered_colors[side_length ** 2:2 * side_length ** 2]).reshape((side_length, side_length, 4)),
        'front': np.array(ordered_colors[2 * side_length ** 2:3 * side_length ** 2]).reshape((side_length, side_length, 4)),
        'back': np.array(ordered_colors[3 * side_length ** 2:4 * side_length ** 2]).reshape((side_length, side_length, 4)),
        'bottom': np.array(ordered_colors[4 * side_length ** 2:5 * side_length ** 2]).reshape((side_length, side_length, 4)),
        'top': np.array(ordered_colors[5 * side_length ** 2:6 * side_length ** 2]).reshape((side_length, side_length, 4))
    }

    ax.clear()  # Clear the axes before replotting

    # Mapping of faces to their positions on the canvas
    face_positions = {
        'top': (0, side_length),
        'left': (side_length, 0),
        'front': (side_length, side_length),
        'right': (side_length, 2 * side_length),
        'back': (side_length, 3 * side_length),
        'bottom': (2 * side_length, side_length)
    }

    # Plot each face
    for face, (row_start, col_start) in face_positions.items():
        face_colors = faces[face]
        for i in range(side_length):
            for j in range(side_length):
                y = row_start + i
                x = col_start + j
                rect = Rectangle((x, y), 1, 1, facecolor=face_colors[i, j])
                ax.add_patch(rect)

    ax.set_xlim(0, 4 * side_length)
    ax.set_ylim(3 * side_length, 0)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add outlines
    for face, (row_start, col_start) in face_positions.items():
        for i in range(side_length):
            for j in range(side_length):
                y = row_start + i
                x = col_start + j
                rect = Rectangle((x, y), 1, 1, edgecolor='black', facecolor='none', linewidth=1)
                ax.add_patch(rect)

    plt.draw()
    plt.pause(0.01)  # Pause to allow the plot to update


def play(S: int, num_scrambles: int = 50) -> None:
    """
    Initiates the Rubik's Cube game, allowing the user to interact with the cube through terminal input.

    Parameters:
    S (int): The size of the cube (e.g., 3 for a standard 3x3x3 cube).
    num_scrambles (int): Number of random rotations to scramble the cube initially. Defaults to 50.
    """
    cube = rc(S, num_scrambles=num_scrambles)

    plt.ion()  # Enable interactive mode
    fig, ax = plt.subplots(figsize=(8, 8))

    while True:
        display_cube(cube, ax)  # Display the cube's state after each move

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
                display_cube(cube, ax)  # Update the display after the move

        except Exception as e:
            print(e)
            print(f"An error occurred. Please try again.")

    plt.ioff()  # Disable interactive mode
    plt.show()  # Ensure the plot is shown when the game ends


def main() -> None:
    """
    The main function to start the game with a default cube size of 3x3x3 and no initial scrambles.
    """
    S = 3
    play(S, num_scrambles=0)


if __name__ == '__main__':
    main()
