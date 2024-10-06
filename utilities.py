# This file will be deleted later.
# Only for ease of use.
import numpy as np

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


