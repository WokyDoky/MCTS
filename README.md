# MCTS
## Monte Carlo Tree Search

In computer science, Monte Carlo tree search (MCTS) is a heuristic search algorithm for some kinds of decision processes, most notably those employed in software that plays board games. 
In that context MCTS is used to solve the game tree.

## Assignment 

Developing AI agents to	play the game of Connect Four. 

> Board: 7 x 6. 

### Part 1

Implement three algorithms that build on each other for selecting moves for given board configurations. Your code must read a game board from a file given in a specific format and run an algorithm with given parameters. The expected output of your algorithms is described below. The board is specified in a standard text file; you do not need to check for the validity of the board (your code will only be tested on valid game states). An example of a game file is shown below. The first line specifies an algorithm to run. The second line specifies the player who will make the next move (R or Y). The next six lines represent the current configuration of the game board. We will use the colors Red and Yellow for the two players; their pieces are represented by the characters ‘R’ and ‘Y’ respectively. The character ‘O’ represents an open space. Moves are made by specifying a valid column from 1-7 to add a new piece to (columns that are already full are illegal moves). We will consider the Red player the “Min” player and represent a win for -1. The Yellow player is the “Max” player and a win for yellow is represented by a 1. A draw has a value of 0.

UR

R

| O   | O   | O   | O   | O   | O   | O   |
| --- | --- | --- | --- | --- | --- | --- |
| O   | O   | O   | O   | O   | O   | O   |
| O   | O   | Y   | O   | O   | O   | Y   |
| O   | O   | R   | O   | O   | O   | Y   |
| O   | Y   | R   | Y   | O   | Y   | R   |
| Y   | R   | R   | Y   | O   | R   | R   |

### Algorithms

1. Algorithm 1: Uniform Random (UR)
   - Choose a random legal move. 
3. Algorithm 2: Pure Monte Carlo Game Search (PMCGS)
   - For each position, all feasible moves are determined: k random games are played out to the very end, and the scores are recorded. The move leading to the best score is chosen. Ties are broken by fair coin flips.
5. Algorithm 3: Upper Confidence bound for Trees (UCT)	
   - The final algorithm builds on PMCGS and uses most of the same structure. The only difference is in how nodes are selected within the existing search tree; instead of selecting randomly the nodes are selected using the Upper Confidence Bounds (UCB) algorithm.