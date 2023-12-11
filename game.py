import itertools
from typing import List

class Game:
    @property
    def states(self):
        return self._states

    def rewards(self, state, action):
        return self._rewards[state][action]

    @property
    def actions(self):
        return self._actions

    def transitions(self, state, action):
        return self._transitions[state][action]

    def gamma(self):
        return self.gamma

    def values(self):
        return self._values

class SimpleGame(Game):
    def __init__(self):
        self._actions = [[0, 1, 2, 3, 4] for _ in range(5)]
        self._states = (0, 1, 2, 3, 4)
        self._rewards = [-1, -1, 10, -1, -1]
        self.gamma = 1
        self._transitions = [
                [0.9, 0.1, 0, 0, 0], # state 0
                [0.9, 0, 0.1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0.9, 0, 0.1],
                [0, 0, 0, 0.9, 0.1],
            ]
        self._values = [0, 0, 0, 0, 0]


class GridWorld:
    def __init__(self, n):
        """ Make a grid of size n"""
        self.n = n
        self.states = list(itertools.product(list(range(self.n)), list(range(self.n))))
        self._actions = {}
        self.gamma = 0.9
        self.V = { state: 0 for state in self.states}

    def actions(self, state) -> List:
        if state in self._actions:
            return self._actions[state]

        x, y = state
        action_set = []

        if x - 1 >= 0:
            action_set.append("LEFT")
        if x + 1 < self.n:
            action_set.append("RIGHT")
        if y - 1 >= 0:
            action_set.append("UP")
        if y + 1 < self.n:
            action_set.append("DOWN")
            

        self._actions[state] = action_set
        
        return action_set

    def transitions(self, state, action):
        """
        List of tuples with states and probability of transitioning to 
        given state
        """
        x, y = state
        if action == "LEFT":
            return [((x - 1, y), 1.0)]
        if action == "RIGHT":
            return [((x + 1, y), 1.0)]
        if action == "UP":
            return [((x, y - 1), 1.0)]
        if action == "DOWN":
            return [((x, y + 1), 1.0)]

    def rewards(self, state):
        x, y = state
        if x == 0 and y == self.n - 1:
            return 1
        else:
            return 0


class TicTacToe(Game):
    def __init__(self):
        # init the board
        self.board = [['-'] * 3 for _ in range(3)]
        self.is_over = False
        self.remaining_pos = {(i, j) for i in range(3) for j in range(3)}

    def __str__(self):
        s = ""
        size = len(self.board)
        for row in range(size):
            for col in range(size):
                s += self.board[row][col]
                if col == 2:
                    s += "\n"
        return s

    def mark(self, player, i, j):
        if self.is_over:
            raise Exception("Game is over.")
        if i < 0 or j < 0 or i > 2 or j > 2:
            raise Exception(f"({i}, {j}) is out of boundary.")
        if self.board[i][j] != '-':
            raise Exception(f"({i}, {j}) is taken.")

        self.board[i][j] = player.mark
        self.remaining_pos.remove((i, j))
        if self._is_over(player, i, j):
            return f"Game over. Winer is {player}."

        if len(self.remaining_pos) <= 0:
            return "Tie."

        return "Game is not over. Continue."

    def _is_over(self, player, i, j):
        # horizontally
        is_over = True
        for k in range(3):
            is_over &= self.board[k][j] == player.mark        

        # vertically
        is_over = True
        for k in range(3):
            is_over &= self.board[i][k] == player.mark
        
        # diagonally
        if i == j:
            is_over = True
            for k in range(3):
                is_over &= self.board[k][k] == player.mark

        diagonal = ((0, 2), (1, 1), (2, 0))
        if (i, j) in diagonal:
            is_over = True
            for k in diagonal:
                is_over &= self.board[k[0]][k[1]] == player.mark

        self.is_over = is_over
        return is_over        

class Player:
    def __init__(self, name, mark):
        self.name = name
        self.mark = mark

    def __str__(self):
        return f"Player {self.name}"
