import random
import math
from copyreg import constructor
from dataclasses import dataclass

import numpy as np
import time
from matplotlib import pyplot as plt
from collections import defaultdict
from scipy.signal import convolve2d
from copy import deepcopy

import utilities
from utilities import board_init, print_board, OPEN_SPACE, RED_PLAYER, full_board_string, detection_kernels, \
    YELLOW_PLAYER, won_board_by_red, board_is_full, did_player_win, test_board_yellow, print_pretty_board, deep_copy, \
    print_pretty_board, place_a_piece, example_board_string, COLUMNS, ROWS, about_to_win_for_red

@dataclass
class BoardState:
    def __init__(self, board, player, other_player):
        """
        Class holds the state of the current board.
        Args:
            board: 2d array
            player: Character "Y" or "R"
            other_player: Character "Y" or "R"
        """
        self.board = board
        self.player = player
        self.other_player = other_player
        self.last_played = []

    @staticmethod
    def default_board():
        """Generate the default empty board for a new game."""
        return [[OPEN_SPACE for _ in range(COLUMNS)] for _ in range(ROWS)]

    def move(self, col):
        """Given a column, drops a piece into the game"""
        for row in range(len(self.board) - 1, -1, -1):
            if self.board[row][col] == OPEN_SPACE:
                self.board[row][col] = self.player
                temp = self.player
                self.player = self.other_player
                self.other_player = temp

                break

    def get_legal_moves(self):
        """Return a list of columns with open spaces."""
        return [col for col in range(COLUMNS) if self.board[0][col] == OPEN_SPACE]

    """Returns 1 for wins, -1 for lose and 0 for draws."""
    def check_win(self):
        """Checks if there is 4 in a row of a given piece"""
        board_array = np.array(self.board)
        for kernel in detection_kernels:
            if (convolve2d(board_array == self.player, kernel, mode="valid") == 4).any():
                return 1
            if (convolve2d(board_array == self.other_player, kernel, mode="valid") == 4).any():
                return -1
        if board_is_full(self.board):
            return 0
        return None

    #Returns None for games not over
        #Problematic.
    def game_over(self):
        """Returns true if someone won of if no more available moves."""
        return len(self.get_legal_moves()) == 0 or self.check_win()

    def print(self):
        print_pretty_board(self.board)

class Node:
    def __init__(self, move, parent):
        self.move = move
        self.parent = parent
        self.N = 0
        self.Q = 0
        self.children = {}
        self.outcome = None

    def add_children(self, children: dict) -> None:
        for child in children:
            self.children[child.move] = child

    #https://en.wikipedia.org/wiki/Monte_Carlo_tree_search#:~:text=c%20is%20the%20exploration%20parameter,in%20practice%20usually%20chosen%20empirically
    def value(self, explore: math.sqrt(2)):
        if self.N == 0:
            return 0 if explore == 0 else float('inf')
        else:
            return self.Q / self.N + explore * math.sqrt(math.log(self.parent.N) / self.N)

class MCTS:
    def __init__(self, state=None, board=None, player=RED_PLAYER, other_player=YELLOW_PLAYER):
        """
        Initialize MCTS with a state, or create a new state with the given board and players.
        Args:
            state (BoardState): An optional game state to start from.
            board (2D array): An optional custom board. If not provided, the default board will be used.
            player (str): The player who starts, defaults to RED_PLAYER.
            other_player (str): The other player, defaults to YELLOW_PLAYER.
        """
        if state is None:
            if board is None:
                # If no board is provided, generate a default board
                board = BoardState.default_board()
            state = BoardState(board, player, other_player)
        self.root_state = deepcopy(state)
        self.root = Node(None, None)
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0

    def select_node(self) -> tuple:
        node = self.root
        state = deepcopy(self.root_state)

        while len(node.children) != 0:
            children = node.children.values()
            max_value = max(children, key=lambda n: n.value()).value()
            max_nodes = [n for n in children if n.value() == max_value]

            node = random.choice(max_nodes)
            state.move(node.move)

            if node.N == 0:
                return node, state

        if self.expand(node, state):
            node = random.choice(list(node.children.values()))
            state.move(node.move)

        return node, state

    def expand(self, parent: Node, state: BoardState) -> bool:
        if state.game_over():
            return False

        children = [Node(move, parent) for move in state.get_legal_moves()]
        parent.add_children(children)

        return True

    def roll_out(self, state: BoardState) -> int:
        while not state.game_over():
            """
            TODO: 
                This should be replaced with alg1 method. 
            """
            state.move(random.choice(state.get_legal_moves()))

        return state.check_win()

    def back_propagate(self, node: Node, turn: int, outcome: int) -> None:

        # For the current player, not the next player
        reward = 0 if outcome == turn else 1

        while node is not None:
            node.N += 1
            node.Q += reward
            node = node.parent
            if outcome == 0:
                reward = 0
            else:
                reward = 1 - reward

    def search(self, time_limit: int):
        start_time = time.process_time()

        num_rollouts = 0
        while time.process_time() - start_time < time_limit:
            node, state = self.select_node()
            outcome = self.roll_out(state)
            self.back_propagate(node, state.player, outcome)
            num_rollouts += 1

        run_time = time.process_time() - start_time
        self.run_time = run_time
        self.num_rollouts = num_rollouts

    def best_move(self):
        if self.root_state.game_over():
            return -1

        max_value = max(self.root.children.values(), key=lambda n: n.N).N
        max_nodes = [n for n in self.root.children.values() if n.N == max_value]
        best_child = random.choice(max_nodes)

        return best_child.move

    def move(self, move):
        if move in self.root.children:
            self.root_state.move(move)
            self.root = self.root.children[move]
            return

        self.root_state.move(move)
        self.root = Node(None, None)

    def statistics(self) -> tuple:
        return self.num_rollouts, self.run_time


def run_multiple_simulations(board, num_runs, time_per_simulation):
    """Outputs a graph from most chosen move. """
    move_counts = {}

    for _ in range(num_runs):
        best_move = alg2(board, time_per_simulation)
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

def alg1(board):
    if board_is_full(board): return -1
    return random.choice([col for col in range(len(board[0])) if board[0][col] == OPEN_SPACE])

def alg2(board, time_per_simulation, verbose = False):
    """Returns the best move for a given board. """
    # TODO:
    #  Add verbose.

    state = BoardState(board, RED_PLAYER, YELLOW_PLAYER)
    mcts = MCTS(state=state)

    print("thinking")
    mcts.search(time_per_simulation)
    move = mcts.best_move()

    return move
def test(board):
    state = BoardState(board, RED_PLAYER, YELLOW_PLAYER)
    mcts = MCTS(state=state)

    state.print()

    print("thinking")
    mcts.search(8)
    num_rollouts, run_time = mcts.statistics()
    print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
    move = mcts.best_move()

    print("MCTS chose move: ", move)

    state.move(move)
    mcts.move(move)
    state.print()

def test2(board, board_won):
    state = BoardState(board, RED_PLAYER, YELLOW_PLAYER)
    state2 = BoardState(board_won, YELLOW_PLAYER, RED_PLAYER)
    state3 = BoardState(board_won, RED_PLAYER, YELLOW_PLAYER)
    mcts = MCTS(state=state)
    mcts2 = MCTS(state=state2)

    state2.print()
    print(True if state.game_over() else False)
    print(True if state2.game_over() else False)

    print(state2.check_win())
    print(state3.check_win())

def main():

    #board = board_init(example_board_string)
    #test(board)

    board = board_init(about_to_win_for_red)
    test(board)

    # Do not run method unless you want to generate graph of most picked move.
    # It takes a long time.
    #run_multiple_simulations(board, 100, 10)


if __name__ == "__main__":
    main()
