import random

import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import convolve2d

import utilities
from utilities import board_init, print_board, OPEN_SPACE, RED_PLAYER, full_board_string, detection_kernels, \
    YELLOW_PLAYER, won_board_by_red, board_is_full, did_player_win, test_board_yellow, print_pretty_board, deep_copy, \
    print_pretty_board, place_a_piece, example_board_string, COLUMNS, ROWS, about_to_win_for_red


class MonteCarloNode:
    def __init__(self, board, move=None, parent=None):
        self.board = board
        self.move = move
        self.parent = parent
        self.children = []
        self.wi = 0  # Total value of this node
        self.ni = 0  # Total visits to this node
        self.untried_moves = self.get_legal_moves()

    def get_legal_moves(self):
        """Return a list of columns with open spaces."""
        return [col for col in range(COLUMNS) if self.board[0][col] == OPEN_SPACE]

    def add_child(self, move, board_copy):
        child_node = MonteCarloNode(board_copy, move, self)
        self.children.append(child_node)
        self.untried_moves.remove(move)
        return child_node

    def is_fully_expanded(self):
        return len(self.untried_moves) == 0

    def update(self, value):
        self.ni += 1
        self.wi += value


def pmcgs(board, player, num_simulations, verbose=False):
    root = MonteCarloNode(board)
    for _ in range(num_simulations):
        node = root
        current_board = deep_copy(board)

        # Selection: traverse the tree until reaching a node with untried moves or no children
        while node.is_fully_expanded() and node.children:
            node = random.choice(node.children)
            place_a_piece(current_board, node.move, player)
            player = RED_PLAYER if player == YELLOW_PLAYER else YELLOW_PLAYER

        # Expansion: if there are untried moves, expand the node
        if not node.is_fully_expanded():
            move = random.choice(node.untried_moves)
            board_copy = deep_copy(current_board)
            place_a_piece(board_copy, move, player)
            node = node.add_child(move, board_copy)
            if verbose: print(f"NODE ADDED with move {move}")

        # Simulation: play out a random game from the current state
        rollout_board = deep_copy(current_board)
        current_player = player
        # >>> Replace the existing simulation code with the following part <<<
        while not board_is_full(rollout_board) and not did_player_win(rollout_board, RED_PLAYER) and not did_player_win(
                rollout_board, YELLOW_PLAYER):
            available_moves = [col for col in range(COLUMNS) if rollout_board[0][col] == OPEN_SPACE]
            if not available_moves:
                break
            move = random.choice(available_moves)
            place_a_piece(rollout_board, move, current_player)
            current_player = RED_PLAYER if current_player == YELLOW_PLAYER else YELLOW_PLAYER
            if verbose: print(f"Rollout move: {move}")

        # Check the outcome of the game
        winner = 0  # Default to draw
        if did_player_win(rollout_board, RED_PLAYER):
            winner = 1 if player == RED_PLAYER else -1
        elif did_player_win(rollout_board, YELLOW_PLAYER):
            winner = 1 if player == YELLOW_PLAYER else -1

        if verbose: print(f"TERMINAL NODE VALUE: {winner}")

        # Backpropagation: update all nodes with the result
        while node is not None:
            node.update(winner)
            if verbose: print(f"Updated node values wi={node.wi}, ni={node.ni}")
            node = node.parent

    # After simulations, select the move with the best average win rate
    best_move = None
    best_value = -float('inf')
    for child in root.children:
        avg_value = child.wi / child.ni
        if avg_value > best_value:
            best_value = avg_value
            best_move = child.move
        if verbose:
            print(f"Move: {child.move}, wi: {child.wi}, ni: {child.ni}, avg: {avg_value}")
    return best_move

def alg1(board):
    if board_is_full(board): return -1
    return random.choice([col for col in range(len(board[0])) if board[0][col] == OPEN_SPACE])


def run_multiple_simulations(board, num_runs, num_simulations_per_run):
    move_counts = {}
    player = RED_PLAYER

    for _ in range(num_runs):
        best_move = pmcgs(board, player, num_simulations_per_run, verbose=False)
        if best_move in move_counts:
            move_counts[best_move] += 1
        else:
            move_counts[best_move] = 1

    # Plotting the results
    moves = list(move_counts.keys())
    counts = list(move_counts.values())

    plt.figure(figsize=(10, 6))
    plt.bar(moves, counts, color='blue')
    plt.xlabel('Move (Column Index)')
    plt.ylabel('Frequency of Selection')
    plt.title(f'Frequency of Best Move Selection in {num_runs} Simulations')
    plt.xticks(moves)
    plt.show()
def main():
    board = board_init(example_board_string)
    test_board = board_init(about_to_win_for_red)

    """player = RED_PLAYER
    best_move = pmcgs(board, player, 20, verbose=True)
    print(f"Best move selected: {best_move}")
    print_pretty_board(board)
    place_a_piece(board, best_move, player)
    print_pretty_board(board)"""

    run_multiple_simulations(test_board, num_runs=100, num_simulations_per_run=2000)

if __name__ == "__main__":
    main()
