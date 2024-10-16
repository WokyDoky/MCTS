# This file will be deleted later.
# Only for ease of use.
import numpy as np
from scipy.signal import convolve2d

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
OOYOROO
YYRRRYY
RYRYYYR
YRRYYRR
"""

test_board_yellow = """
OOOOOOO
OOOOOOO
OOOOOOO
OOOOOOO
OOOOOOO
YYYYYOR
"""
about_to_win_for_red = """
OOOOOOO
OOOOOOO
OOYOROO
YYRORYY
RYRYYYR
YRRYYRR
"""
draw = """
RRRYRRO
YYYRYYY
RRRYRRR
YYYRYYY
RRRYRRR
YRYRYYY
"""
horizontal_kernel = np.array([[ 1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
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

def did_player_win (board, player):
    """Returns +1 if 'X' wins, -1 if 'O' wins, and 0 for draw."""
    board_array = np.array(board)
    for kernel in detection_kernels:
        if (convolve2d(board_array == player, kernel, mode="valid") == 4).any():
            return 1
    return -1

def board_is_full(board):
    return not (OPEN_SPACE in (item for sublist in board for item in sublist))
"""
    Inserts a piece into the Connect 4 board in the given column.

    :param board: 2D list representing the Connect 4 board (rows x columns).
    :param column: The column index where the player wants to drop their piece.
    :param piece: The player's piece ('X' or 'O', for example).
    :return: True if the piece was successfully inserted, False if the column is full.
    """
def place_a_piece(board, column, player):
    rows = len(board)
    # Iterate from the bottom of the column upwards
    for row in reversed(range(rows)):
        if board[row][column] == OPEN_SPACE:  # Check if the cell is empty
            board[row][column] = player
            return True  # Successfully inserted piece

    return False  # Column is full

def print_pretty_board(board):
  print("\n     A    B    C    D    E    F    G  ", end="")
  for x in range(len(board)):
    print("\n   +----+----+----+----+----+----+----+")
    print(x, " |", end="")
    for y in range(len(board[x])):
      if(board[x][y] == "Y"):
        print("", "🔵" , end=" |")
      elif(board[x][y] == "R"):
        print("", "🔴", end=" |")
      else:
        print("", "⚪", end=" |")
  print("\n   +----+----+----+----+----+----+----+")

def deep_copy(board):
    return [[i for i in row] for row in board]