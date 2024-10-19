import random
import math
from copyreg import constructor
from dataclasses import dataclass

import numpy as np
import time

from PyQt5.pyrcc_main import verbose
from matplotlib import pyplot as plt
from collections import defaultdict
from scipy.signal import convolve2d
from copy import deepcopy

import utilities
from utilities import board_init, print_board, OPEN_SPACE, RED_PLAYER, full_board_string, detection_kernels, \
    YELLOW_PLAYER, won_board_by_red, board_is_full, did_player_win, test_board_yellow, print_pretty_board, deep_copy, \
    print_pretty_board, place_a_piece, example_board_string, COLUMNS, ROWS, about_to_win_for_red

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

    #Return 1 for wins, -1 for lose and 0 for draw.
    def check_win(self):
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
        return len(self.get_legal_moves()) == 0 or self.check_win()

    def print(self):
        print_pretty_board(self.board)

    def ugly_print(self):
        print_board(self.board)
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
    if board_is_full(board): return -1
    return random.choice([col for col in range(len(board[0])) if board[0][col] == OPEN_SPACE])

def alg2(board, time_per_simulation, option_to_print):
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
        mcts.search(time_per_simulation)
        move = mcts.best_move()
        print("MCTS chose move: ", move)
    elif option_to_print == 1:
        print("Thinking...")
        mcts.search(time_per_simulation)
        move = mcts.best_move()
        num_rollouts, run_time = mcts.statistics()
        print("Statistics: ", num_rollouts, "rollouts in", run_time, "seconds")
        state.move(move)
        state.ugly_print()
        print("MCTS chose move: ", move)
    else:
        state.print()
        print("Thinking...")
        mcts.search(time_per_simulation)
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



def main():
    #TODO:
    # Add parameters to the command line after file call in.
    # Example: python	test1.txt	Verbose 0

    """
    El profe quiere que uno de los parametros sea numero de iteraciones cuando lo corres
       Yo lo tengo por tiempo por que no sabia cual seria un buen numero de iteraciones pero lo pueden cambiar.

        Para referencia, ~10 segundos de este algorimo son como ~80k iteraciones en mi laptop.
    """

    board = board_init(example_board_string)
    #alg2(board, 1, 2)

    #Will output a graph.
    run_multiple_simulations(board, 5, 1)


if __name__ == "__main__":
    main()