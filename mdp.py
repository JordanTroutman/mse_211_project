import itertools
from abc import ABC, abstractmethod, abstractproperty
from enum import Enum

class MDP(ABC):

    @abstractmethod
    def actions(self,state):
        pass

    @abstractmethod
    def transition(self,state, action):
        pass

    @abstractmethod
    def reward(self, state, action):
        pass

    @abstractmethod
    def prob(self, state, action):
        pass

def find_indices(list_to_check, item_to_find):
    indices = []
    for idx, value in enumerate(list_to_check):
        if value == item_to_find:
            indices.append(idx)
    return indices

def replace_char(s, char, index):
    return s[:index] + char + s[index + 1:]
    

class TicTacToe(MDP):
    class StateCondition(Enum):
        CONTINUE = "Continue"
        WIN = "Player win"
        LOSE = "Opponent win"
        TIE = "Tie"
        INVALID = "Invalid"
    

    def __init__(self, player="X"):
        self.states = self.generate_states(for_player=player)
        self._actions = {}
        self.player = player
        self.opponent = "O" if player == "X" else "O"

    
    def actions(self, state):
        action_set = []
        # If we're in a goal state, there are no more actions to do
        if self.condition(state, "actions") != TicTacToe.StateCondition.CONTINUE:
            return [state]
        
        # Actions are filling in any of the empty spaces with a "pawn"
        blank_positions = find_indices(list(state), "-")

        player_count = state.count(self.player)
        opponent_count = state.count(self.opponent)

        # The player goes first, so if the player has more pieces on the board, then
        # It's the opponent's turn to place something
        
        pawn = self.opponent if player_count > opponent_count else self.player
            
        for pos in blank_positions:
            action = replace_char(state, pawn, pos)
            action_set.append(action)
            
        
        return action_set

    
    def transition(self, state, action):
        # Being in the win condition just returns you to the same
        # state. same with a tie, or end of game generally
        condition = self.condition(state, "transition")
        if (condition != TicTacToe.StateCondition.CONTINUE):
            return [(state, 1.0)]
        
        
        # We can place more pieces on the board
        # The action is the board when the spot is placed
        # With the given action, the next state is placing the opponent
        blank_positions = find_indices(list(action), "-")
        next_states = []

        # Fill in each available space one at a time
        for pos in blank_positions:
            next_state = replace_char(action, self.opponent, pos)
            next_states.append(next_state)
        # print("State:", state, "Next States", next_states)
        # Equal probability for each state to happen
        # e.g. this action has an equal probability to transition to any state
        return [(next_state, 1.0 / len(next_states)) for next_state in next_states]

    
    def reward(self, state, action):
        # According to the lecture notes, a win is
        # Noted by a reward of -1
        # Losing is noted by 1
        # Ties are 0
        #print("Value of state:", state)
        condition = self.condition(state, "reward")
        
        if condition == TicTacToe.StateCondition.WIN:
            #print("Win Condition: {}".format(state))
            return 1
        elif condition == TicTacToe.StateCondition.LOSE:
            return -1
        else:
            #print("Rewards")
            return 0

    
    def prob(self, state, action):
        actions = self.actions(state)
        return { action: 1 / len(actions) for action in actions }

    def generate_states(self, for_player):
        # Fill out the grid for the board
        # Prune unreachable variants
        # States here are only for the player, not opponent

        def valid_board(board):
            # X and O count are at most 1 different from each other
            x_count = board.count("X")
            o_count = board.count("O")

            one_off_count = abs(x_count - o_count) <= 1
            empty_board = board == "-" * 9
            majority_player_board = x_count >= o_count if for_player == "X" else o_count >= x_count
            
            return  (one_off_count and majority_player_board) or empty_board
        
        all_boards = list(map(lambda board: "".join(board), itertools.product(["X", "O", "-"], repeat=9)))
        #boards = list(filter(valid_board, all_boards))
        return all_boards


    def winner(self, board, player):
        """
        Determine the winner for a game of Tic Tac Toe
        """
        player_match = player * 3
        # Column matches
        for i in range(3):
            if board[i::3] == player_match:
                return True
        
        # Row matches
        for i in range(0,7,3):
            if board[i: i + 3] == player_match:
                return True
    
        # Diagonals
        left_diagonal = board[0] + board[4] + board[8]
        right_diagonal = board[2] + board[4] + board[6]
    
        return left_diagonal == player_match or right_diagonal == player_match

    def tie(self, board):
        """
        Determine if a tie has occured
        """
        # Tie happens when there are 8 elements (4 of each) and no winner
        x_count = board.count("X")
        o_count = board.count("O")

        four_each = x_count == 4 and o_count == 4
        # No win when both the player and opponent have not won
        no_win = not (self.winner(board, self.player) or self.winner(board, self.opponent))

        return four_each and no_win


    def condition(self, board, func):
        # if func == "actions":
        #     print("From", func)
        #     print(self.winner(board, self.player))
        #print(TicTacToe.StateCondition.WIN == TicTacToe.StateCondition.WIN)
        
        if self.winner(board, self.player):
            # print("Returning", TicTacToe.StateCondition.WIN)
            return TicTacToe.StateCondition.WIN
        elif self.winner(board, self.opponent):
            return TicTacToe.StateCondition.LOSE
        elif self.tie(board):
            return TicTacToe.StateCondition.TIE
        else:
            return TicTacToe.StateCondition.CONTINUE

        
            

class GridWorld(MDP):

    def __init__(self, n):
        """ Make a grid of size n"""
        self.n = n
        self.states = list(itertools.product(list(range(self.n)), list(range(self.n))))
        self._actions = {}
        self.V = { state: 0 for state in self.states}

    def actions(self, state):
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

    def transition(self, state, action):
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

    def reward(self, state, action):
        x, y = state
        if x == 0 and y == self.n - 1:
            return 1
        else:
            return 0
        

    def prob(self, state, action):
        actions = self.actions(state)
        return { action: 1 / len(actions) for action in actions }

