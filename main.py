import random
import utilities
from utilities import board_init, print_board, OPEN_SPACE


def alg1(board):
    move = random.choice([col for col in range(len(board[0])) if board[0][col] == OPEN_SPACE])
    print(move)

def alg2(board):

def main():
    print("!Hola!")

    board = board_init(utilities.example_board_string)
    print_board(board)

    alg1(board)


if __name__ == "__main__":
    main()
