import os
import random
import math
import sys
from dataclasses import dataclass
import time

from matplotlib import pyplot as plt
from copy import deepcopy

import numpy as np
from scipy.signal import convolve2d
RED_PLAYER = 'R'
YELLOW_PLAYER = 'Y'
OPEN_SPACE = 'O'

ROWS = 6
COLUMNS = 7
horizontal_kernel = np.array([[1, 1, 1, 1]])
vertical_kernel = np.transpose(horizontal_kernel)
diag1_kernel = np.eye(4, dtype=np.uint8)
diag2_kernel = np.fliplr(diag1_kernel)
detection_kernels = [horizontal_kernel, vertical_kernel, diag1_kernel, diag2_kernel]
class Utilities:

    @staticmethod
    def board_init(string_board):
        rows = string_board.strip().split("\n")
        board = [list(row) for row in rows]
        return board

    @staticmethod
    def print_board(board):
        print(end='  ')
        for i in range(COLUMNS):  # Access class variables using `Utilities.`
            print(i, end=' ')
        print()
        i = 0
        for row in board:
            print(i, end=' ')
            print(" ".join(row))
            i += 1
        print("\n")

    @staticmethod
    def did_player_win(board, player):
        """Returns +1 if player wins, -1 otherwise."""
        board_array = np.array(board)
        for kernel in detection_kernels:  # Access class variables via `Utilities.`
            if (convolve2d(board_array == player, kernel, mode="valid") == 4).any():
                return 1
        return -1

    @staticmethod
    def board_is_full(board):
        return not (OPEN_SPACE in (item for sublist in board for item in sublist))

    """
        Inserts a piece into the Connect 4 board in the given column.

        :param board: 2D list representing the Connect 4 board (rows x columns).
        :param column: The column index where the player wants to drop their piece.
        :param piece: The player's piece ('X' or 'O', for example).
        :return: True if the piece was successfully inserted, False if the column is full.
    """
    @staticmethod
    def place_a_piece(board, column, player):
        rows = len(board)
        for row in reversed(range(rows)):
            if board[row][column] == OPEN_SPACE:
                board[row][column] = player
                return True
        return False

    @staticmethod
    def print_pretty_board(board):
        print("\n     A    B    C    D    E    F    G  ", end="")
        for x in range(len(board)):
            print("\n   +----+----+----+----+----+----+----+")
            print(x, " |", end="")
            for y in range(len(board[x])):
                if (board[x][y] == "Y"):
                    print("", "🔵", end=" |")
                elif (board[x][y] == "R"):
                    print("", "🔴", end=" |")
                else:
                    print("", "⚪", end="  |")
        print("\n   +----+----+----+----+----+----+----+")

    @staticmethod
    def deep_copy(board):
        return [[i for i in row] for row in board]

@dataclass
class ConnectState:
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


    def check_win(self):
        """Return 1 for wins, -1 for lose and 0 for draw."""
        board_array = np.array(self.board)
        for kernel in detection_kernels:
            if (convolve2d(board_array == self.player, kernel, mode="valid") == 4).any():
                return 1
            if (convolve2d(board_array == self.other_player, kernel, mode="valid") == 4).any():
                return -1
        if Utilities.board_is_full(self.board):
            return 0
        return None

    #Returns None for games not over
        #Problematic.
    def game_over(self):
        return len(self.get_legal_moves()) == 0 or self.check_win()

    def print(self):
        Utilities.print_pretty_board(self.board)

    def ugly_print(self):
        Utilities.print_board(self.board)
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
    def value(self, explore: float = math.sqrt(2)):
        if self.N == 0:
            return 0 if explore == 0 else float('inf')
        else:
            return self.Q / self.N + explore * math.sqrt(math.log(self.parent.N) / self.N)

    """
    Exploitation: 
       a strategy of using the accumulated knowledge to make decisions that maximize 
       the expected reward based on the present information.
       In other words, if you now it works, keep doing it. 
    
    Exploration: 
        Exploration is used to increase knowledge about an environment or model.
        Explore new ideas. 
        
    C constant, in this case sqrt(2), decides the ratio between exploration and exploitation. 
    """

class MCTS:
    def __init__(self, state=None, board=None, player=RED_PLAYER, other_player=YELLOW_PLAYER, verbose=False):
        """
        Initialize MCTS with a state, or create a new state with the given board and players.
        Args:
            state (ConnectState): An optional game state to start from.
            board (2D array): An optional custom board. If not provided, the default board will be used.
            player (str): The player who starts, defaults to RED_PLAYER.
            other_player (str): The other player, defaults to YELLOW_PLAYER.
            verbose (bool): Prints extra information.
        """
        if state is None:
            if board is None:
                # If no board is provided, generate a default board
                board = ConnectState.default_board()
            state = ConnectState(board, player, other_player)
        self.root_state = deepcopy(state)
        self.root = Node(None, None)
        self.run_time = 0
        self.node_count = 0
        self.num_rollouts = 0
        self.verbose = verbose

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
            if self.verbose: print("NODE ADDED")

        return node, state

    def expand(self, parent: Node, state: ConnectState) -> bool:
        if state.game_over():
            return False

        children = [Node(move, parent) for move in state.get_legal_moves()]
        parent.add_children(children)

        return True

    def roll_out(self, state: ConnectState) -> int:
        while not state.game_over():
            move = random.choice(state.get_legal_moves())
            state.move(move)
            if self.verbose: print(f"Move selected: {move}")
        if self.verbose: print(f"TERMINAL NODE VALUE: {state.check_win()}")
        return state.check_win()

    def back_propagate(self, node: Node, turn: int, outcome: int) -> None:

        # For the current player, not the next player
        reward = 0 if outcome == turn else 1

        while node is not None:
            node.N += 1
            node.Q += reward
            if self.verbose: print(f"Updated values:\nwi: {node.Q}\nni: {node.N}")
            node = node.parent
            if outcome == 0:
                reward = 0
            else:
                reward = 1 - reward

    def search(self, num_simulations: int):
        start_time = time.process_time()

        num_rollouts = 0
        while num_rollouts < num_simulations:
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

def run_multiple_simulations(board, num_runs, num_simulations):
    """
        Simulates multiple games of Connect Four to determine the frequency of the best move.

        Args:
            board (2D list): The current state of the game board, represented as a 2D list where each element is either OPEN_SPACE, RED_PLAYER, or a YELLOW_PLAYER.
            num_runs (int): The number of simulations to run.
            num_simulations (int): The number of simulations to run.

        Returns:
            None: The function displays a bar chart of the frequency of the best move selection for each column over the course of the simulations.

        Behavior:
            - For each simulation, the best move is calculated using the `alg2` function.
            - The frequency of each move being selected as the best is tracked in the `move_counts` dictionary.
            - A bar chart is generated and displayed showing how often each move (column index) was selected as the best move across all simulations.
    """
    move_counts = {}

    for _ in range(num_runs):
        best_move = alg2(board, num_simulations, 0)
        if best_move in move_counts:
            move_counts[best_move] += 1
        else:
            move_counts[best_move] = 1

    # Plotting the results
    moves = list(move_counts.keys())
    counts = list(move_counts.values())

    letter_moves = [chr(ord('A') + move) for move in moves]

    plt.figure(figsize=(10, 6))
    plt.bar(letter_moves, counts, color='blue')
    plt.xlabel('Move (Column)')  # Update label to reflect letters
    plt.ylabel('Frequency of Selection')
    plt.title(f'Frequency of Best Move Selection in {num_runs} Simulations')
    plt.xticks(letter_moves)  # Use letter_moves for x-axis ticks
    plt.show()

def alg1(board):
    if Utilities.board_is_full(board): raise Exception ("Board is full.")
    return random.choice([col for col in range(len(board[0])) if board[0][col] == OPEN_SPACE])

def alg2(board, num_simulations, option_to_print):
    """
    Heuristic driven search algorithm.
    For each position, all feasible moves are determined:
        k random games are played out to the very end, and the scores are recorded.
        The move leading to the best score is chosen.

    board(2D Array): Representation of the board in a 2D string array
                        if no board given, it is generated with all 0's.
    option_to_print(int 0..2):
        0 -> No printing other than the best move
        1 -> Brief printing.
        2 -> Verbose.
    """
    verbose = True if option_to_print > 1 else False
    state = ConnectState(board, RED_PLAYER, YELLOW_PLAYER)
    mcts = MCTS(state=state, verbose=verbose)

    if option_to_print == 0:
        print("...")
        mcts.search(num_simulations)
        move = mcts.best_move()
        print("MCTS chose move: ", move)
    elif option_to_print == 1:
        print("Thinking...")
        mcts.search(num_simulations)
        move = mcts.best_move()
        num_rollouts, run_time = mcts.statistics()
        print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
        state.move(move)
        state.ugly_print()
        print("MCTS chose move: ", move)
    else:
        state.print()
        print("Thinking...")
        mcts.search(num_simulations)
        move = mcts.best_move()
        num_rollouts, run_time = mcts.statistics()
        print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
        state.move(move)
        state.print()
        print("MCTS chose move: ", move)

    return move
def test(board, verbose):
    state = ConnectState(board, RED_PLAYER, YELLOW_PLAYER)
    mcts = MCTS(state=state, verbose=verbose)

    if verbose: state.print()

    print("thinking")
    mcts.search(1)
    num_rollouts, run_time = mcts.statistics()
    if verbose: print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
    move = mcts.best_move()

    print("MCTS chose move: ", move)


"""
========================================================
==========================MAIN==========================
========================================================
"""

def main(input_file, verbosity, simulations):
    if not os.path.exists(input_file):
        raise Exception(f"Error: File '{input_file}' does not exist.")

    with open(input_file, 'r') as f:
        algorithm = f.readline().strip()
        player = f.readline().strip()
        board = [list(f.readline().strip()) for _ in range(ROWS)]

    if verbosity == "None":
        option_to_print = 0
    elif verbosity == "Brief":
        option_to_print = 1
    elif verbosity == "Verbose":
        option_to_print = 2
    else:
        raise Exception("typo")

    if algorithm == "UR":
        print(f"FINAL	Move	selected:{alg1(board)}")
        if option_to_print > 1: Utilities.print_board(board)
    elif algorithm == "PMCGS":
        alg2(board, simulations, option_to_print)
    elif algorithm == "UCT":
        alg2(board, simulations, option_to_print)



if __name__ == "__main__":
    input_file = sys.argv[1]
    verbosity = sys.argv[2]
    simulations = int(sys.argv[3])
    main(input_file, verbosity, simulations)