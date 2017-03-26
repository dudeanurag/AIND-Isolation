"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import sys
import math

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score1(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # If the board position is such that our agent (Student)
    # does not have any move to play, return -INF
    if game.is_loser(player):
        return float("-inf")

    # If the board position is such that the opponent
    # does not have any move to play, return INF
    if game.is_winner(player):
        return float("inf")

    opponent = game.get_opponent(player)

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opponent)

    try:
        return float(len(own_moves)) / len(opp_moves)
    except ZeroDivisionError:
        return float(len(own_moves)) / (len(opp_moves) + 1)

def custom_score2(game, player):
    # If the board position is such that our agent (Student)
    # does not have any move to play, return -INF
    if game.is_loser(player):
        return float("-inf")

    # If the board position is such that the opponent
    # does not have any move to play, return INF
    if game.is_winner(player):
        return float("inf")

    opponent = game.get_opponent(player)

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opponent)

    # In this custom heuristic function we look one move ahead and 
    # find the number of legal moves in each subsequent position.
    # We sum them up to find a kind of weighted sum of legal moves
    # for the given board.
    weighted_own_moves = 0.0
    weighted_opp_moves = 0.0
    for m in own_moves:
        new_board = game.forecast_move(m)
        weighted_own_moves += len(new_board.get_legal_moves(player))

    for m in opp_moves:
        new_board = game.forecast_move(m)
        weighted_opp_moves += len(new_board.get_legal_moves(opponent))

    # ratio of weighted_own_moves and weighted_opp_moves is the heuristic score
    # When there is ZeroDivisionError, add 1 in denominator to handle it
    try:
        return weighted_own_moves / weighted_opp_moves
    except ZeroDivisionError:
        return weighted_own_moves / (weighted_opp_moves + 1)
    
def custom_score3(game, player):
    # If the board position is such that our agent (Student)
    # does not have any move to play, return -INF
    if game.is_loser(player):
        return float("-inf")

    # If the board position is such that the opponent
    # does not have any move to play, return INF
    if game.is_winner(player):
        return float("inf")

    opponent = game.get_opponent(player)

    own_moves = game.get_legal_moves(player)
    opp_moves = game.get_legal_moves(opponent)

    # In this custom heuristic function we look one move ahead and 
    # find the number of legal moves in each subsequent position.
    # We sum them up to find a kind of weighted sum of legal moves
    # for the given board.
    weighted_own_moves = 0.0
    weighted_opp_moves = 0.0
    for m in own_moves:
        new_board = game.forecast_move(m)
        weighted_own_moves += len(new_board.get_legal_moves(player))

    for m in opp_moves:
        new_board = game.forecast_move(m)
        weighted_opp_moves += len(new_board.get_legal_moves(opponent))

    # ratio of weighted_own_moves and weighted_opp_moves is the heuristic score
    # 1 is added in denominator to avoid division by zero exception
    return (weighted_own_moves - weighted_opp_moves)

# set custom_score to the function custom_score2()
custom_score = custom_score2

class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=15.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout

    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves
        nextmove = None
        if not legal_moves:
            nextmove = (-1, -1)

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            if self.iterative == False:
                if self.method == 'minimax': 
                    hscore, nextmove = self.minimax(game, self.search_depth)
                else:
                    hscore, nextmove = self.alphabeta(game, self.search_depth)
            else:
                for d in range(1, sys.maxsize):
                    if self.method == 'minimax':
                        hscore, nextmove = self.minimax(game, d)
                    else:
                        hscore, nextmove = self.alphabeta(game, d)

        except Timeout:
            # Handle any actions required at timeout, if necessary
            return nextmove

        # Return the best move from the last completed search iteration
        return nextmove 


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()

        # my_player refers to our agent (Student), which is also a maximizing player
        if maximizing_player == True:
            my_player = game.__active_player__
        else: 
            my_player = game.__inactive_player__

        # If there are no legal moves left, we reach the end node
        if not legal_moves:
            return self.score(game, my_player), (-1, -1)

        # When a pre-specified depth in the game tree is reached,
        # the corresponding heuristic value of that node is returned
        if depth == 0:
            return self.score(game, my_player), game.__last_player_move__[game.__inactive_player__]

        # Recursively iterate over all possible legal moves, and return 
        # the move with largest (or smallest) backed-up heuristic value
        if maximizing_player:
            bestValue = -math.inf
            for m in legal_moves:
                v, t = self.minimax(game.forecast_move(m), depth - 1, False)
                if v >= bestValue:
                    bestValue = v
                    bestTuple = m
            return bestValue, bestTuple

        else:
            bestValue = math.inf
            for m in legal_moves:
                v, t = self.minimax(game.forecast_move(m), depth - 1, True)
                if v <= bestValue:
                    bestValue = v
                    bestTuple = m
            return bestValue, bestTuple


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        legal_moves = game.get_legal_moves()

        if maximizing_player == True:
            my_player = game.__active_player__
        else: 
            my_player = game.__inactive_player__

        if not legal_moves:
            return self.score(game, my_player), (-1, -1)

        if depth == 0:
            return self.score(game, my_player), game.__last_player_move__[game.__inactive_player__]

        # Update the value of alpha at maximimizing node
        # Prune a branch when alpha exceeds beta
        if maximizing_player:
            bestValue = -math.inf
            for m in legal_moves:
                v, t = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, False)
                if v >= bestValue:
                    bestValue = v
                    bestTuple = m
                if bestValue >= beta:
                    return bestValue, bestTuple
                alpha = max(alpha, bestValue)
            return bestValue, bestTuple

        # Update the value of beta at minimizing node
        # Prune a branch when beta falls below alpha
        else:
            bestValue = math.inf
            for m in legal_moves:
                v, t = self.alphabeta(game.forecast_move(m), depth - 1, alpha, beta, True)
                if v <= bestValue:
                    bestValue = v
                    bestTuple = m
                if bestValue <= alpha:
                    return bestValue, bestTuple
                beta = min(beta, bestValue)
            return bestValue, bestTuple