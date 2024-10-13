# This file will be deleted later.
# Only for ease of use.
import numpy as np
from matplotlib import pyplot as plt
from collections import Counter

from main import pmcgs

RED_PLAYER = 'R'
YELLOW_PLAYER = 'Y'
OPEN_SPACE = 'O'

ROWS = 6
COLUMNS = 7

# Example board
# Should be saved in a file!

example_board_string = """
OOOOOOO
OOOOOOO
OOYOOOY
OOROOOY
OYRYOYR
YRRYORR
"""

full_board_string = """
RRRRRRR
YYYYYYY
RRRRRRR
RRRRRRR
RRRRRRR
YRRYRRR
"""
won_board_by_red = """
OOOOOOO
OOOOOOO
OOYOROY
OYRRRYY
RYRYYYR
YRRYYRR
"""
about_to_win_for_red = """
OOOOOOO
OOOOOOO
OOYOROO
YYRORYY
RYRYYYR
YRRYYRR
"""
detection_kernels = [
    np.array([[1, 1, 1, 1]]),  # Horizontal kernel
    np.array([[1], [1], [1], [1]]),  # Vertical kernel
    np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 0]]),  # Diagonal (left-to-right)
    np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 0, 0]])   # Diagonal (right-to-left)
]
def board_init(string_board):
    rows = string_board.strip().split("\n")
    board = [list(row) for row in rows]
    return board

def print_board(board):
    print(end='  ')
    for i in range(COLUMNS): print(i, end=' ')
    print()
    i = 0
    for row in board:
        print(i, end=' ')
        print(" ".join(row))
        i += 1
    print("\n")


def run_multiple_pmcgs(board, num_runs=100, simulations=1000):
    """Runs the PMCGS algorithm multiple times and records the best moves."""
    move_counter = Counter()

    for _ in range(num_runs):
        best_move = pmcgs(board, simulations=simulations, verbose=False)
        move_counter[best_move] += 1

    return move_counter


def plot_best_moves(move_counter):
    """Plots the frequency of the best moves chosen during multiple PMCGS runs."""
    moves = list(move_counter.keys())
    counts = list(move_counter.values())

    plt.bar(moves, counts)
    plt.xlabel('Move (Column Number)')
    plt.ylabel('Frequency Chosen')
    plt.title('Frequency of Moves Chosen by PMCGS')
    plt.xticks(range(len(moves)), [f'Column {i}' for i in moves])
    plt.show()
