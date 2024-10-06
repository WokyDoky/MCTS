import utilities
from utilities import board_init, print_board


# The main method
def main():
    print("!Hola!")

    board = board_init(utilities.example_board_string)
    print_board(board)


# The entry point for running the script
if __name__ == "__main__":
    main()
