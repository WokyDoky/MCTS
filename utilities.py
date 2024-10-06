# This file will be deleted later.
# Only for ease of use.

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
def board_init(string_board):
    rows = string_board.strip().split("\n")
    board = [list(row) for row in rows]
    return board

def print_board(board):
    for row in board:
        print(" ".join(row))
    print("\n")


