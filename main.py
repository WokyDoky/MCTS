import random
import utilities
from utilities import board_init, print_board, OPEN_SPACE

class Node:
    def __init__(self, board, parent=None):
        self.board = board  # Game state (board at this point)
        self.parent = parent  # Reference to the parent node
        self.children = []  # List of child nodes
        self.visits = 0  # Number of times node has been visited
        self.wins = 0  # Number of wins resulting from this node

    def add_child(self, child_node):
        self.children.append(child_node)

    def update(self, result):
        """Update node statistics after a simulation result."""
        self.visits += 1
        self.wins += result

def select_node(node):
    """Selects the node with the highest UCT value."""
    best_node = max(node.children, key=lambda child: uct_value(child))
    return best_node

def uct_value(node, exploration_param=1.41):
    """Calculates UCT (Upper Confidence Bound applied to Trees) value for node."""
    if node.visits == 0:
        return float('inf')  # Prioritize unvisited nodes
    return node.wins / node.visits + exploration_param * (2 * (node.parent.visits ** 0.5) / node.visits)

def expand_node(node):
    """Expands the current node by generating all possible child nodes."""
    available_moves = [col for col in range(len(node.board[0])) if node.board[0][col] == OPEN_SPACE]
    for move in available_moves:
        new_board = simulate_move(node.board, move)
        child_node = Node(new_board, parent=node)
        node.add_child(child_node)
    return node.children

def simulate_move(board, move):
    """Simulate a move in the given board."""
    new_board = [row[:] for row in board]  # Create a deep copy of the board
    for row in range(len(board)-1, -1, -1):
        if new_board[row][move] == OPEN_SPACE:
            new_board[row][move] = "X"  # Assuming 'X' is the current player
            break
    return new_board

def simulate_random_game(node):
    """Simulates a random game from the given node."""
    current_board = node.board
    while not is_terminal(current_board):
        move = random.choice([col for col in range(len(current_board[0])) if current_board[0][col] == OPEN_SPACE])
        current_board = simulate_move(current_board, move)
    return get_game_result(current_board)

def is_terminal(board):
    """Checks if the game has ended (win/draw)."""
    # Implement logic to check if the board has a winning state or if it's a draw
    return False  # Placeholder, should return True if game over

def get_game_result(board):
    """Returns +1 if 'X' wins, -1 if 'O' wins, and 0 for draw."""
    # Implement logic to evaluate the final board state
    return 0  # Placeholder

def backpropagate(node, result):
    """Propagates the result of a simulation back through the tree."""
    while node is not None:
        node.update(result)
        result = -result  # Alternate the result for the opponent's perspective
        node = node.parent

def alg1(board):
    move = random.choice([col for col in range(len(board[0])) if board[0][col] == OPEN_SPACE])
    print(move)

def alg2(board):
    # Initialize root node
    root_node = Node(board)

    # Number of simulations
    for _ in range(1000):  # Typically, you'd do many simulations
        node = root_node

        # Selection: traverse the tree to find an unexpanded node
        while node.children:
            node = select_node(node)

        # Expansion: generate new child nodes if the node is not terminal
        if not is_terminal(node.board):
            expand_node(node)

        # Simulation: simulate a random game from the node
        result = simulate_random_game(node)

        # Backpropagation: propagate the result of the simulation back through the tree
        backpropagate(node, result)

    # Select the best move based on the number of visits
    best_move = max(root_node.children, key=lambda child: child.visits)
    print("Best move:", best_move)

    print(board)
def main():
    print("!Hola!")

    board = board_init(utilities.example_board_string)
    print_board(board)

    alg2(board)


if __name__ == "__main__":
    main()
