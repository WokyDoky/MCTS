import random
import numpy as np
from scipy.signal import convolve2d

import utilities
from utilities import board_init, print_board, OPEN_SPACE, RED_PLAYER, full_board_string, detection_kernels, \
    YELLOW_PLAYER, about_to_win_for_red


class Node:
    def __init__(self, board, move=None, parent=None):
        self.board = board  # Game state (board at this point)
        self.parent = parent  # Reference to the parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times node has been visited
        self.wins = 0  # Number of wins resulting from this node
        self.move = move  # Move that led to this node

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, result):
        self.visits += 1
        self.wins += result


def select_random_node(node):
    """Selects a child node at random."""
    return random.choice(node.children)


def expand_node(node):
    """Expands the current node by generating all possible child nodes."""
    available_moves = [col for col in range(len(node.board[0])) if node.board[0][col] == OPEN_SPACE]
    for move in available_moves:
        new_board = simulate_move(node.board, move)
        child_node = Node(new_board, move=move, parent=node)  # Pass move to the new child node
        node.add_child(child_node)
    return node.children


def simulate_move(board, move):
    """Simulate a move in the given board."""
    new_board = [row[:] for row in board]  # Create a deep copy of the board
    for row in range(len(board) - 1, -1, -1):
        if new_board[row][move] == OPEN_SPACE:
            new_board[row][move] = RED_PLAYER  # Player. Will need to alternate based on turn.
            break
    return new_board


def simulate_random_game(node):
    """Simulates a random game from the given node."""
    current_board = node.board
    while not is_terminal(current_board):
        move = alg1(current_board)
        current_board = simulate_move(current_board, move)
    return get_game_result(current_board)


def is_terminal(board):
    """Checks if the game has ended (win/draw)."""
    if board_is_full(board):
        return True

    board_array = np.array(board)
    # Check for each kernel
    for kernel in detection_kernels:
        if (convolve2d(board_array == RED_PLAYER, kernel, mode="valid") == 4).any():
            return True
    for kernel in detection_kernels:
        if (convolve2d(board_array == YELLOW_PLAYER, kernel, mode="valid") == 4).any():
            return True

    return False


def get_game_result(board):
    """Returns +1 if 'RED_PLAYER' wins, -1 if 'YELLOW_PLAYER' wins, and 0 for draw."""
    board_array = np.array(board)
    for kernel in detection_kernels:
        if (convolve2d(board_array == RED_PLAYER, kernel, mode="valid") == 4).any():
            return 1
        if (convolve2d(board_array == YELLOW_PLAYER, kernel, mode="valid") == 4).any():
            return -1
    if board_is_full(board):
        return 0
    return None


def backpropagate(node, result):
    """Propagates the result of a simulation back through the tree."""
    while node is not None:
        node.update(result)
        result = -result  # Alternate the result for the opponent's perspective
        node = node.parent


def board_is_full(board):
    return not (OPEN_SPACE in (item for sublist in board for item in sublist))


def alg1(board):
    if board_is_full(board): return -1
    return random.choice([col for col in range(len(board[0])) if board[0][col] == OPEN_SPACE])


def pmcgs(board, simulations=1000, verbose=False):
    """Pure Monte Carlo Game Search implementation."""
    root_node = Node(board)

    for _ in range(simulations):
        node = root_node

        # Selection: Traverse the tree randomly
        while node.children:
            node = select_random_node(node)
            if verbose:
                print(f"wi: {node.wins} ni: {node.visits} Move selected: {node.move}")

        # Expansion: Generate new child nodes if the node is not terminal
        if not is_terminal(node.board):
            expand_node(node)
            if verbose:
                print("NODE ADDED")

        # Simulation: Simulate a random game from the node
        result = simulate_random_game(node)

        # Backpropagation: Propagate the result of the simulation back through the tree
        backpropagate(node, result)
        if verbose:
            print(f"TERMINAL NODE VALUE: {result}")
            print(f"Updated values: wi: {node.wins} ni: {node.visits}")

    # Select the best move based on the number of visits
    best_move_node = max(root_node.children, key=lambda child: child.visits)
    print(f"FINAL Move selected: {best_move_node.move}")  # Print the move attribute instead of the node object

    # Output the values of each immediate next move
    for i, child in enumerate(root_node.children):
        if child.visits > 0:
            value = child.wins / child.visits
        else:
            value = "Null"
        print(f"Column {i + 1}: {value}")

    return best_move_node.move


def alg1(board):
    if board_is_full(board): return -1
    return random.choice([col for col in range(len(board[0])) if board[0][col] == OPEN_SPACE])

def main():
    board = board_init(utilities.example_board_string)
    print_board(board)
    best_move = pmcgs(board, simulations=1000, verbose=True)
    print(f"Best move: {best_move}")

    board = board_init(about_to_win_for_red)
    move_counter = utilities.run_multiple_pmcgs(board, num_runs=100, simulations=1000)

    # Plot the frequency of each move
    utilities.plot_best_moves(move_counter)

if __name__ == "__main__":
    main()
