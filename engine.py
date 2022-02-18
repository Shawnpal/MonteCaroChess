"""
Starter Code for Assignment 1 - COMP 8085
Please do not redistribute this code our your solutions
The game engine to keep track of the game and provider of a generic AI implementation
 You need to extend the GenericAI class to perform a better job in searching for the next move!
"""
# pip install Chessnut
import json
import os
from math import sqrt, log
import sys
from tqdm import trange
from csv import writer
from Chessnut import Game
import csv
from pathlib import Path
from enum import Enum
import random
import time
from collections import deque

import copy
sys.setrecursionlimit(1500)

def create_generator(g_list):
    for num in g_list:
        yield num
    yield "done"


class GameResult(Enum):
    WON = 1
    LOST = 2
    STALEMATE = 3
    DRAWBY50M = 4


class AIEngine:
    def __init__(self, board_state, reasoning_depth=3):
        self.game = Game(board_state)
        self.reasoning_depth = reasoning_depth
        self.computer = MCTSAIAI(self.game, reasoning_depth)
        self.leaderboard = {"White": {"Wins": 0, "Loses": 0, "Draws": 0}, "Black": {"Wins": 0, "Loses": 0, "Draws": 0}}

    def prompt_user(self,userselection):

        if userselection == "1":
            self.computer == GenericAI(self.game, self.reasoning_depth)
        elif userselection == "2":
            self.computer == GainMaxAI(self.game, self.reasoning_depth)
        elif userselection == "3":
            self.computer == MinimaxAI(self.game, self.reasoning_depth)
        elif userselection == "4":
            self.computer == IterativeDeepeningAI(self.game, self.reasoning_depth)

        """
        Use this function to play with the ai bot created in the constructor
        """
        self.computer.print_board(str(self.game))
        fifty_move_draw = False
        playing_side = "White"

        try:
            while self.game.status < 2 and not fifty_move_draw:
                playing_side = "White"
                user_move = input("\nMake a move: \033[95m")
                print("\033[0m")
                while user_move not in self.game.get_moves() and user_move != "ff":
                    user_move = input("Please enter a valid move: ")
                if user_move == "ff":
                    print("Execution Stopped!")
                    break
                self.game.apply_move(user_move)
                fifty_move_draw = self.check_fifty_move_draw()
                captured = self.captured_pieces(str(self.game))
                start_time = time.time()
                self.computer.print_board(str(self.game), captured)
                print("\nComputer Playing...\n")
                if self.game.status < 2 and not fifty_move_draw:
                    playing_side = "Black"
                    current_state = str(self.game)
                    computer_move = self.computer.make_move(current_state)
                    piece_name = {'p': 'pawn', 'b': 'bishop', 'n': 'knight', 'r': 'rook', 'q': 'queen', 'k': 'king'}
                    start = computer_move[:2]
                    end = computer_move[2:4]
                    piece = piece_name[self.game.board.get_piece(self.game.xy2i(computer_move[:2]))]
                    captured_piece = self.game.board.get_piece(self.game.xy2i(computer_move[2:4]))
                    if captured_piece != " ":
                        captured_piece = piece_name[captured_piece.lower()]
                        print("---------------------------------")
                        print("Computer's \033[92m{piece}\033[0m at \033[92m{start}\033[0m captured \033[91m{captured_piece}\033[0m "
                              "at \033[91m{end}\033[0m.".format(piece=piece, start=start, captured_piece=captured_piece, end=end))
                        print("---------------------------------")
                    else:
                        print("---------------------------------")
                        print("Computer moved \033[92m{piece}\033[0m at \033[92m{start}\033[0m to \033[92m{end}\033[0m.".format(
                            piece=piece, start=start, end=end))
                        print("---------------------------------")
                    print("\033[1mNodes visited:\033[0m        \033[93m{}\033[0m".format(self.computer.node_count))
                    print("\033[1mElapsed time in sec:\033[0m  \033[93m{time}\033[0m".format(time=time.time() - start_time))
                    self.game.apply_move(computer_move)
                    fifty_move_draw = self.check_fifty_move_draw()
                captured = self.captured_pieces(str(self.game))
                self.computer.print_board(str(self.game), captured)
            self.record_winner(self.computer, "Black", playing_side, fifty_move_draw)
            print("Game Ended!")
        except KeyboardInterrupt:
            print("Execution Stopped!")

    def record_winner(self, ai_bot, bot_side, losing_side, fifty_move_draw):
        print("\nGame result for {}:".format(bot_side))
        winscore = 0
        if fifty_move_draw:
            ai_bot.record_winner(GameResult.DRAWBY50M)
            self.leaderboard[bot_side]["Draws"] += 1
            winscore += 0
            return winscore
        elif self.game.status == 3:
            ai_bot.record_winner(GameResult.STALEMATE)
            self.leaderboard[bot_side]["Draws"] += 1
            winscore += 1
            return winscore
        elif self.game.status == 2 and losing_side == bot_side:
            ai_bot.record_winner(GameResult.LOST)
            self.leaderboard[bot_side]["Loses"] += 1
            winscore += 2
            return winscore
        elif self.game.status == 2 and losing_side != bot_side:
            ai_bot.record_winner(GameResult.WON)
            self.leaderboard[bot_side]["Wins"] += 1
            winscore += 2
            return winscore
        else:
            raise ValueError("Should not happen!")

    def check_fifty_move_draw(self):

        return int(str(self.game).split()[4]) > 100

    def play_with_self(self):
        """
        Use this function to have two different AI bots play with each other and see their game
        """
        self.game = Game('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        self.computer = {"White": GenericAI(self.game, 2), "Black": MinimaxAI(self.game, 2)}
        self.computer["White"].print_board(str(self.game))
        bot_side = "White"
        fifty_move_draw = False
        while self.game.status < 2 and not fifty_move_draw:
            start_time = time.time()
            computer_move = self.computer[bot_side].make_move(str(self.game))
            piece_name = {'p': 'pawn', 'b': 'bishop', 'n': 'knight', 'r': 'rook', 'q': 'queen', 'k': 'king'}
            start = computer_move[:2]
            end = computer_move[2:4]
            piece = piece_name[self.game.board.get_piece(self.game.xy2i(computer_move[:2])).lower()]
            captured_piece = self.game.board.get_piece(self.game.xy2i(computer_move[2:4]))
            if captured_piece != " ":
                captured_piece = piece_name[captured_piece.lower()]
                print("---------------------------------")
                print("{bot_side}'s \033[92m{piece}\033[0m at \033[92m{start}\033[0m captured \033[91m{captured_piece}\033[0m "
                      "at \033[91m{end}\033[0m.".format(bot_side=bot_side, piece=piece, start=start, captured_piece=captured_piece, end=end))
                print("---------------------------------")
            else:
                print("---------------------------------")
                print("{bot_side} moved \033[92m{piece}\033[0m at \033[92m{start}\033[0m to \033[92m{end}\033[0m.".format(
                    bot_side=bot_side, piece=piece, start=start, end=end))
                print("---------------------------------")
            print("\033[1mNodes visited:\033[0m        \033[93m{}\033[0m".format(self.computer[bot_side].node_count))
            print("\033[1mElapsed time in sec:\033[0m  \033[93m{time}\033[0m".format(time=time.time() - start_time))
            print(str(self.game))
            self.game.apply_move(computer_move)
            fifty_move_draw = self.check_fifty_move_draw()
            captured = self.captured_pieces(str(self.game))
            self.computer[bot_side].print_board(str(self.game), captured)
            bot_side = "Black" if bot_side == "White" else "White"
        self.record_winner(self.computer["White"], "White", bot_side, fifty_move_draw)
        self.record_winner(self.computer["Black"], "Black", bot_side, fifty_move_draw)
        print("Game Ended!")

    def play_with_self_non_verbose(self,userselection):
        """
        Use this function to have two different AI bots play with each other without printing the game board or the decisions they make in the console.
        """

        self.game = Game('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        self.computer = {"White": GenericAI(self.game, 3), "Black": GenericAI(self.game, 3)}
        if userselection == "6":
            self.computer = {"White": GenericAI(self.game, 3), "Black": GenericAI(self.game, 3)}
        elif userselection == "7":
            self.computer = {"White": GenericAI(self.game, 3), "Black": MinimaxAI(self.game, 3)}
        elif userselection == "8":
            self.computer = {"White": GenericAI(self.game, 4), "Black": IterativeDeepeningAI(self.game, 3)}
        elif userselection == "9":
            self.computer = {"White": MCTSAIAI(self.game, 3), "Black": GenericAI(self.game, 3)}
            self.computer["White"].Populate_tree()
        bot_side = "White"
        fifty_move_draw = False
        while self.game.status < 2 and not fifty_move_draw:

            computer_move = self.computer[bot_side].make_move(str(self.game))
            self.game.apply_move(computer_move)
            fifty_move_draw = self.check_fifty_move_draw()
            bot_side = "Black" if bot_side == "White" else "White"
        self.record_winner(self.computer["White"], "White", bot_side, fifty_move_draw)
        self.record_winner(self.computer["Black"], "Black", bot_side, fifty_move_draw)

    @staticmethod
    def captured_pieces(board_state):
        piece_tracker = {'P': 8, 'B': 2, 'N': 2, 'R': 2, 'Q': 1, 'K': 1, 'p': 8, 'b': 2, 'n': 2, 'r': 2, 'q': 1, 'k': 1}
        captured = {"w": [], "b": []}
        for char in board_state.split()[0]:
            if char in piece_tracker:
                piece_tracker[char] -= 1
        for piece in piece_tracker:
            if piece_tracker[piece] > 0:
                if piece.isupper():
                    captured['w'] += piece_tracker[piece] * piece
                else:
                    captured['b'] += piece_tracker[piece] * piece
            piece_tracker[piece] = 0
        return captured


class BoardNode:
    def __init__(self, board_state=None, algebraic_move=None, value=None):
        self.board_state = board_state
        self.algebraic_move = algebraic_move
        self.value = value

class GenericAI:
    def __init__(self, game, max_depth=4, leaf_nodes=None, node_count=0):
        if leaf_nodes is None:
            leaf_nodes = []
        self.max_depth = max_depth
        self.leaf_nodes = create_generator(leaf_nodes)
        self.game = game
        self.node_count = node_count
        self.PiecesValue = {
            "p": 100,
            "P": 100,
            "n": 300,
            "N": 300,
            "b": 300,
            "B": 300,
            "r": 500,
            "R": 500,
            "q": 800,
            "Q": 800,
            "k": 1000,
            "K": 1000,
            " ": 0
        }

    @property
    def name(self):
        return "Dumb AI"

    def get_moves(self, board_state=None):
        if board_state is None:
            board_state = str(self.game)
        possible_moves = []
        for move in Game(board_state).get_moves():
            if len(move) < 5 or move[4] == "q":
                clone = Game(board_state)
                clone.apply_move(move)
                node = BoardNode(str(clone))
                node.algebraic_move = move
                possible_moves.append(node)
        return possible_moves

    def make_move(self, board_state):
        possible_moves = self.get_moves(board_state)
        # TODO use search algorithms to find the best move in here
        best_move = random.choice(possible_moves)
        return best_move.algebraic_move

    def record_winner(self, result):
        print("The game result: {}".format(result))

    def print_board(self, board_state, captured=None):
        if captured is None:
            captured = {"w": [], "b": []}
        piece_symbols = {'p': '♟', 'b': '♝', 'n': '♞', 'r': '♜', 'q': '♛', 'k': '♚', 'P': '\033[36m\033[1m♙\033[0m',
                         'B': '\033[36m\033[1m♗\033[0m', 'N': '\033[36m\033[1m♘\033[0m', 'R': '\033[36m\033[1m♖\033[0m',
                         'Q': '\033[36m\033[1m♕\033[0m', 'K': '\033[36m\033[1m♔\033[0m'}
        board_state = board_state.split()[0].split("/")
        board_state_str = "\n"
        white_captured = " ".join(piece_symbols[piece] for piece in captured['w'])
        black_captured = " ".join(piece_symbols[piece] for piece in captured['b'])
        for i, row in enumerate(board_state):
            board_state_str += str(8 - i)
            for char in row:
                if char.isdigit():
                    board_state_str += " ♢" * int(char)
                else:
                    board_state_str += " " + piece_symbols[char]
            if i == 0:
                board_state_str += "   Captured:" if len(white_captured) > 0 else ""
            if i == 1:
                board_state_str += "   " + white_captured
            if i == 6:
                board_state_str += "   Captured:" if len(black_captured) > 0 else ""
            if i == 7:
                board_state_str += "   " + black_captured
            board_state_str += "\n"
        board_state_str += "  A B C D E F G H"
        self.node_count = 0
        print(board_state_str)

class GainMaxAI(GenericAI):
    def __init__(self, game, max_depth=5, leaf_nodes=None, node_count=0):
        super(GainMaxAI, self).__init__(game, max_depth, leaf_nodes, node_count)
        self.cache = {}
        self.found_in_cache = 0

    @property
    def name(self):
        return "Dumb AI"

    def get_moves(self, board_state=None):
        if board_state is None:
            board_state = str(self.game)
        possible_moves = []
        for move in Game(board_state).get_moves():
            if len(move) < 5 or move[4] == "q":
                clone = Game(board_state)
                clone.apply_move(move)
                node = BoardNode(str(clone))
                node.algebraic_move = move
                possible_moves.append(node)
        return possible_moves

    def make_move(self, board_state):
        possible_moves = self.get_moves(board_state)
        # TODO use search algorithms to find the best move in here



        def evaluation(movestate):
            pseduboard_state = movestate.split()[0]
            WPawncount = pseduboard_state.count('p')
            WKnightCount = pseduboard_state.count('n')
            WKingCount = pseduboard_state.count('k')
            WBishopCount = pseduboard_state.count('b')
            WRookCount = pseduboard_state.count('r')
            WQueenCount = pseduboard_state.count('q')
            BPawncount = pseduboard_state.count('P')
            BKnightCount = pseduboard_state.count('N')
            BKingCount = pseduboard_state.count('K')
            BBishopCount = pseduboard_state.count('B')
            BRookCount = pseduboard_state.count('R')
            BQueenCount = pseduboard_state.count('Q')
            whiteEval = WPawncount * self.PiecesValue['p'] + WKnightCount * self.PiecesValue['n'] + WBishopCount * self.PiecesValue[
                'b'] + WRookCount * self.PiecesValue['r'] + WQueenCount * self.PiecesValue['q'] + WKingCount * self.PiecesValue['k']
            blackEval = BPawncount * self.PiecesValue['P'] + BKnightCount * self.PiecesValue['N'] + BBishopCount * self.PiecesValue[
                'B'] + BRookCount * self.PiecesValue['R'] + BQueenCount * self.PiecesValue['Q'] + BKingCount * self.PiecesValue['K']
            return (whiteEval - blackEval)


        def Max_search():
            bestEvaluation = -9999
            bestmove = "ff"
            for move in possible_moves:
                currentevaluation = evaluation(move.board_state)
                if currentevaluation >= bestEvaluation:
                    bestEvaluation = currentevaluation
                    bestmove = move
            return bestmove

        def Min_search():
            bestEvaluation = 9999
            bestmove = "ff"
            for move in possible_moves:
                currentevaluation = evaluation(move.board_state)
                if currentevaluation <= bestEvaluation:
                    bestEvaluation = currentevaluation
                    bestmove = move
            return bestmove



        if self.game.state.player=="b":
            move  = Max_search()
        elif self.game.state.player=="w":
            move = Min_search()

        return move.algebraic_move





class MinimaxAI(GenericAI):
    def __init__(self, game, max_depth=5, leaf_nodes=None, node_count=0):
        super(MinimaxAI, self).__init__(game, max_depth, leaf_nodes, node_count)
        self.cache = {}
        self.found_in_cache = 0

    @property
    def name(self):
        return "Minimax AI"

    def make_move(self, board_state):
        # TODO re-write this code to use minimax function to pick the best move
        pseudoGame = self
        value,best_move = self.minimax(pseudoGame.game,-99999,99999)
        return best_move


    def minimax(self, node,alpha,beta,current_depth=0):




        # TODO implement this function
        def eveluate(pseudoGame):

            pseduboard_state = str(pseudoGame).split()[0]
            WPawncount = pseduboard_state.count('p')
            WKnightCount = pseduboard_state.count('n')
            WKingCount = pseduboard_state.count('k')
            WBishopCount = pseduboard_state.count('b')
            WRookCount = pseduboard_state.count('r')
            WQueenCount = pseduboard_state.count('q')
            BPawncount = pseduboard_state.count('P')
            BKnightCount = pseduboard_state.count('N')
            BKingCount = pseduboard_state.count('K')
            BBishopCount = pseduboard_state.count('B')
            BRookCount = pseduboard_state.count('R')
            BQueenCount = pseduboard_state.count('Q')
            whiteEval = WPawncount * self.PiecesValue['p'] + WKnightCount * self.PiecesValue['n']  + WBishopCount * self.PiecesValue['b']  + WRookCount * self.PiecesValue['r']  + WQueenCount * self.PiecesValue['q'] + WKingCount*self.PiecesValue['k']
            blackEval = BPawncount * self.PiecesValue['P'] + BKnightCount * self.PiecesValue['N'] + BBishopCount * self.PiecesValue['B'] + BRookCount * self.PiecesValue['R'] + BQueenCount * self.PiecesValue['Q'] + BKingCount*self.PiecesValue['K']
            return (whiteEval - blackEval)


        def MaxValue(node, current_depth,alpha,beta):
            if (current_depth == self.max_depth) or node.status >= 2:
                return eveluate(node),None
            v = -9999

            for move in node.get_moves():
                Testmove = move.lower()
                start = Game.xy2i(Testmove[:2])
                end = Game.xy2i(Testmove[2:4])
                piece = node.board.get_piece(start)
                target = node.board.get_piece(end)
                piecevalue = self.PiecesValue[piece]
                targetvalue = self.PiecesValue[target]

                self.node_count = self.node_count+1

                nodeclone = Game(str(node))
                nodeclone.apply_move(move)
                evel,tmpmove= MinValue(nodeclone,current_depth+1,alpha,beta)

                if (evel > v):
                    v,bestmove = evel,move
                    alpha = max(alpha,v)
                if v >= beta:
                    return v,bestmove
                if targetvalue >= piecevalue:
                    # prioritize if we can take a piece with an equal value or more
                    bestmove = move
                    return v, bestmove
            return v,bestmove

        def MinValue(node, current_depth,alpha,beta):
            if (current_depth == self.max_depth) or node.status >= 2:
                return eveluate(node), None
            v = 9999
            for move in  node.get_moves():
                Testmove = move.lower()
                start = Game.xy2i(Testmove[:2])
                end = Game.xy2i(Testmove[2:4])
                piece = node.board.get_piece(start)
                target = node.board.get_piece(end)
                piecevalue = self.PiecesValue[piece]
                targetvalue = self.PiecesValue[target]

                self.node_count = self.node_count + 1
                nodeclone= Game(str(node))
                nodeclone.apply_move(move)
                evel,tmpmove = MaxValue(nodeclone,current_depth+1,alpha,beta)
                if (evel < v):
                    v, bestmove = evel, move
                    beta = min(beta,v)
                if v <= alpha:
                    return v, bestmove
                if targetvalue >= piecevalue:
                    #prioritize if we can take a piece with an equal value or more
                    bestmove = move
                    return v,bestmove
            return v, bestmove


        if node.state.player=="b":
            value,move  = MaxValue(node, current_depth,alpha,beta)
        elif node.state.player=="w":
            value, move = MinValue(node, current_depth, alpha, beta)
        return value,move


# TODO use the example of MinimaxAI class definition to prepare other AI bot search algorithms
class IterativeDeepeningAI(GenericAI):
    def __init__(self, game, max_depth=4, leaf_nodes=None, node_count=0):
        super(IterativeDeepeningAI, self).__init__(game, max_depth, leaf_nodes, node_count)
        self.cache = {}
        self.found_in_cache = 0


    @property
    def name(self):
        return "ITrMinimax AI"

    def make_move(self, board_state):
        # TODO re-write this code to use MCTSAI function to pick the best move
        pseudoGame = self
        cutoff = 9999999
        for x in range(1, 99):
            value,best_move = self.Itrminimax(pseudoGame.game,x,-99999,99999)
            if value == cutoff:
                return best_move

        return best_move


    def Itrminimax(self, node,depth_limit,alpha,beta,current_depth=0, timelimit=5):

        # TODO implement this function
        def eveluate(pseudoGame):

            pseduboard_state = str(pseudoGame).split()[0]
            WPawncount = pseduboard_state.count('p')
            WKnightCount = pseduboard_state.count('n')
            WKingCount = pseduboard_state.count('k')
            WBishopCount = pseduboard_state.count('b')
            WRookCount = pseduboard_state.count('r')
            WQueenCount = pseduboard_state.count('q')
            BPawncount = pseduboard_state.count('P')
            BKnightCount = pseduboard_state.count('N')
            BKingCount = pseduboard_state.count('K')
            BBishopCount = pseduboard_state.count('B')
            BRookCount = pseduboard_state.count('R')
            BQueenCount = pseduboard_state.count('Q')
            whiteEval = WPawncount * self.PiecesValue['p'] + WKnightCount * self.PiecesValue['n']  + WBishopCount * self.PiecesValue['b']  + WRookCount * self.PiecesValue['r']  + WQueenCount * self.PiecesValue['q'] + WKingCount*self.PiecesValue['k']
            blackEval = BPawncount * self.PiecesValue['P'] + BKnightCount * self.PiecesValue['N'] + BBishopCount * self.PiecesValue['B'] + BRookCount * self.PiecesValue['R'] + BQueenCount * self.PiecesValue['Q'] + BKingCount*self.PiecesValue['K']
            return (whiteEval - blackEval)

        start_time = time.time()

        def MaxValue(node, current_depth, alpha, beta):
            if (current_depth == depth_limit) or node.status >= 2:
                return eveluate(node), None
            v = -9999

            for move in node.get_moves():
                Testmove = move.lower()
                start = Game.xy2i(Testmove[:2])
                end = Game.xy2i(Testmove[2:4])
                piece = node.board.get_piece(start)
                target = node.board.get_piece(end)
                piecevalue = self.PiecesValue[piece]
                targetvalue = self.PiecesValue[target]

                self.node_count = self.node_count + 1

                nodeclone = Game(str(node))
                nodeclone.apply_move(move)
                evel, tmpmove = MinValue(nodeclone, current_depth + 1, alpha, beta)

                if (evel > v):
                    v, bestmove = evel, move
                    alpha = max(alpha, v)
                if v >= beta:
                    return v, bestmove
                if targetvalue >= piecevalue:
                    # prioritize if we can take a piece with an equal value or more
                    bestmove = move
                    return v, bestmove
                if ((time.time() - start_time) > timelimit):#Cut off time limit
                    v = 9999999
                    bestmove = move
                    return v, bestmove
            return v, bestmove


        def MinValue(node, current_depth, alpha, beta):
            if (current_depth == depth_limit) or node.status >= 2:
                return eveluate(node), None
            v = 9999
            for move in node.get_moves():
                Testmove = move.lower()
                start = Game.xy2i(Testmove[:2])
                end = Game.xy2i(Testmove[2:4])
                piece = node.board.get_piece(start)
                target = node.board.get_piece(end)
                piecevalue = self.PiecesValue[piece]
                targetvalue = self.PiecesValue[target]

                self.node_count = self.node_count + 1

                nodeclone = Game(str(node))
                nodeclone.apply_move(move)
                evel, tmpmove = MaxValue(nodeclone, current_depth + 1, alpha, beta)

                if (evel < v):
                    v, bestmove = evel, move
                    beta = min(beta, v)
                if v <= alpha:
                    return v, bestmove
                if targetvalue >= piecevalue:
                    # prioritize if we can take a piece with an equal value or more
                    bestmove = move
                    return v, bestmove
                if ((time.time() - start_time) > timelimit):#Cut off time limit
                    v = 9999999
                    bestmove = move
                    return v, bestmove
            return v, bestmove

        value, move = MaxValue(node, current_depth, alpha, beta)

        if node.state.player == "b":
            value, move = MaxValue(node, current_depth, alpha, beta)
        elif node.state.player == "w":
            value, move = MinValue(node, current_depth, alpha, beta)
        return value, move



# TODO use the example of MinimaxAI class definition to prepare other AI bot search algorithms
class MCTSAIAI(GenericAI):
    def __init__(self, game, max_depth=4, leaf_nodes=None, node_count=0):
        super(MCTSAIAI, self).__init__(game, max_depth, leaf_nodes, node_count)
        self.cache = {}
        self.found_in_cache = 0
        self.leaderboard = {"White": {"Wins": 0, "Loses": 0, "Draws": 0}, "Black": {"Wins": 0, "Loses": 0, "Draws": 0}}
        self.Root_node = Class_Node(Game('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'))
        self.Current_node = self.Root_node


    @property
    def name(self):
        return "MCTSAI AI"

    #Expand current Tree and check to see if we need to add addtional Nodes
    def make_move(self, board_state):
        # TODO re-write this code to use MCTSAI function to pick the best move
        possible_moves = self.get_moves(board_state)
        expanded_children = []
        best_score = float('-inf')
        #if the other player is not a MCTSAI and thus doesn't traverse the node

        if (str(self.Current_node.gamenode) != board_state):
            founded = False
            # we first check if other player have already made a move as part of the Tree
            for child in self.Current_node.children:
                # If we found a match, we change our Current node to child node.
                if (str(child.gamenode) == board_state):
                    self.Current_node = child
                    break
            #if we didn't find a child node then we start a new branch
            if (founded) is not True:
                newchild = Class_Node(Game(board_state))
                self.Current_node.add_child(newchild)
                self.Current_node = newchild

        #check for who is the player
        current_player = self.Current_node.gamenode.state.player

        # we  check if there possible moves have already been been made
        for move in possible_moves:
            founded = False
            for child in self.Current_node.children:
                """
                #we found that there are nodes that have been visit so we add them
                #to our expanded group
                # we also attach the corresponding algebraic move to the child
                """
                if (str(child.gamenode) == move.board_state):
                    child.algebraic_move = move.algebraic_move
                    expanded_children.append(child)
                    founded = True
                    break
            """
            #For every move that we didn't found a matching child. We create an unvisit Node Object
            #an unvisited Node Object is basically a new branch
            #we set the parent to current node but it is not added as a child, merely for ucb1 scoring
            """
            if (founded) is not True:
                newchild = Class_Node(Game(move.board_state))
                newchild.parent = self.Current_node
                newchild.algebraic_move = move.algebraic_move
                expanded_children.append(newchild)
        best_children=[]
        """
        #Now for every child, we feed it into our ucb scoring algorthim which
        #picks the highest child depending on the player
        #If the child is scored same as the highest, then a random among them will be picked
        """
        for child in expanded_children:
            #here we use our selection policy (UCT) to Look for the best child to play
            score = child.ucb1(current_player)
            if score > best_score:
                best_score = score
                best_children = [child]
            elif score == best_score:
                best_children.append(child)
        bestChild = random.choice(best_children)
        #we move our Current_node to the selected Child
        self.Current_node = bestChild
        return bestChild.algebraic_move

    #straightForward function, we simply backprogogate from our current node and increase black and white values
    #The black and white win values determines how likely each player will take the path again
    def BackProgogate(self ,Node,Black_win_value,White_win_value):
        Node.visit += 1

        Node.Black_win_value += Black_win_value
        Node.White_win_value += White_win_value
        while (Node.parent != None):
            Node.parent.visits += 1
            Node= Node.parent
            Node.Black_win_value += Black_win_value
            Node.White_win_value += White_win_value
    def Populate_tree(self):


        print("\nPopulating Tree...\n")
        with open("starters.csv", 'r') as data:
              for line in csv.reader(data):
                if str(self.Current_node.gamenode==line[0]):
                    Child_node = Class_Node(Game(line[1]))
                    self.Current_node.add_child(Child_node)
                    self.Current_node = Child_node

        self.BackProgogate(self.Current_node,120000,120000)


        with open("mcts.model", 'r') as data:
            for line in csv.reader(data):
                if(line):
                    if line[0] == "WHITEWIN":
                        self.BackProgogate(self.Current_node,0, 120000)
                    elif line[0] ==  "BLACKWIN":
                        self.BackProgogate(self.Current_node,120000,0)
                          #  Parent_node.White_loss_value += 2
                    elif line[0] == "STALEMATE":
                        self.BackProgogate(self.Current_node,60000,60000)
                    elif line[0] == "50TurnDraw":
                        self.BackProgogate(self.Current_node,0,0)
                    elif str(self.Current_node.gamenode == line[0]):
                        Child_node = Class_Node(Game(line[1]))
                        found = False
                        for child in self.Current_node.children:
                            if str(child.gamenode) == line[1]:
                                Child_node = child
                                found = True
                                break
                        if found is not True :
                            self.Current_node.add_child(Child_node)
                        self.Current_node = Child_node
        print("\nFinished Populating Tree...\n")







    def record_winner_and_score(self,Game, ai_bot, bot_side, losing_side, fifty_move_draw):

        whitescore = 0
        blackscore = 0

        if fifty_move_draw:
            with open('mcts.model', 'a') as f_object:
                Result = ["50TurnDraw"]
                writer_object = writer(f_object)
                writer_object.writerow(Result)
            ai_bot.record_winner("DRAWBY50M")
            self.leaderboard[bot_side]["Draws"] += 0
            whitescore = 0
            blackscore = 0
            return whitescore,blackscore
        elif Game.status == 3:
            with open('mcts.model', 'a') as f_object:
                Result = ["STALEMATE"]
                writer_object = writer(f_object)
                writer_object.writerow(Result)
            ai_bot.record_winner("STALEMATE")
            self.leaderboard[bot_side]["Draws"] += 1
            whitescore = 60000
            blackscore = 60000
            return whitescore,blackscore
        elif Game.status == 2 and losing_side == bot_side:
            with open('mcts.model', 'a') as f_object:
                Result = ["BLACKWIN"]
                writer_object = writer(f_object)
                writer_object.writerow(Result)
            ai_bot.record_winner("BLACKWIN")
            self.leaderboard[bot_side]["Loses"] += 1
            blackscore = 120000
            return whitescore, blackscore
        elif Game.status == 2 and losing_side != bot_side:
            with open('mcts.model', 'a') as f_object:
                Result = ["WHITEWIN"]
                writer_object = writer(f_object)
                writer_object.writerow(Result)
            ai_bot.record_winner("WHITEWIN")
            self.leaderboard[bot_side]["Wins"] += 1
            whitescore = 120000
            return whitescore, blackscore
        else:
            raise ValueError("Should not happen!")

    #  simulating random MCTSAI moves with 2 MCTSAI AI to populate the search Tree
    def play_with_self_simulate(self):
        """
        Use this function to simulate two MCTSAI AI bots play with each other without printing the game board or the decisions they make in the console.
        """

        Clone = AIEngine('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
        Clone.computer = {"White": MCTSAIAI(Clone.game, 2), "Black": MCTSAIAI(Clone.game, 2)}
        bot_side = "White"
        fifty_move_draw = False


        while Clone.game.status < 2 and not fifty_move_draw:
            move = Clone.computer[bot_side].make_move(str(Clone.game))
            Previous_state = str(Clone.game)
            Clone.game.apply_move(move)
            List = [Previous_state,str(self.Current_node.gamenode)]
            fifty_move_draw = Clone.check_fifty_move_draw()
            bot_side = "Black" if bot_side == "White" else "White"
            with open('mcts.model', 'a') as f_object:
                writer_object = writer(f_object)
                writer_object.writerow(List)
                f_object.close()
                # Close the file object




        Wwinscore,Bwinscore = self.record_winner_and_score(Clone.game,Clone.computer["White"], "White", bot_side, fifty_move_draw)

        #back Propogation and incrementing visit and wins depending on who
        self.Current_node.visits += 1
        self.Current_node.White_win_value += Wwinscore
        self.Current_node.Black_win_value += Bwinscore
        while (self.Current_node.parent != None):
            self.Current_node.parent.visits += 1
            self.Current_node = self.Current_node.parent
            self.Current_node.White_win_value += Wwinscore
            self.Current_node.Black_win_value += Bwinscore






class Class_Node:

    def __init__(self, gamenode):
        self.gamenode = gamenode
        self.White_win_value = 0
        self.White_loss_value = 0
        self.Black_win_value = 0
        self.Black_loss_value = 0
        self.Draw = 0
        self.visits = 0
        self.parent = None
        self.HasParent = False
        self.children = []
        self.algebraic_move =""



    def __repr__(self):
        return self.name

    def add_child(self, child):
        self.children.append(child)
        child.hasParent = True
        child.parent = self

    def add_children(self, children):
        for child in children:
            self.add_child(child)

    """
      IF THERE IS A WINNING PATH FOR WHITE, ITS WIN VALUE IS HIGHER THAN AN UNEXPLORED NODE AND 
      IT WILL TRY TO KEEP GOING DOWN THAT PATH 
    # WHEN ITS BLACK'S TURN, BLACK WILL TRY TO EXPLORE AN ALT LEAF NODE SO IT WILL EXPLORE UNVISITED NODES, 
    IT CAN END IN BLACK WIN, STALE MATE OR 50 TURN #DRAW OR ANOTHER WHITE WIN
    # WHITE WILL ONLY START EXPLORING UNEXPLORED NODES, 
    WHEN BLACK FOUND A LEAF NODE THAT END IN BLACK WIN, AT WHICH IT WILL TRY TO FIND  UNEXPLORED  INSTEAD
    # BOTH PLAYERS WILL TRY TO AVOID 50 TURN DRAW UNLESS ALL NODES ARE EXPLORED AND HAS NO WINNING PATHS, 
    AT WHICH IT'S JUST RANDOM AGAIN
        """
    def ucb1(self,current_player):
        if current_player == "b":
            exploit = (self.Black_win_value-self.White_win_value)
            explore = 1 * (sqrt(log(self.parent.visits +1+ (10 ** -6)) / (self.visits + (10 ** -10))))
            ans = exploit + explore
        if current_player == "w":
            exploit = self.White_win_value-self.Black_win_value
            explore = 1 * (sqrt(log(self.parent.visits +1+ (10 ** -6)) / (self.visits + (10 ** -10))))

            ans = exploit + explore
        return ans






if __name__ == '__main__':
    test_engine = AIEngine('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')
  #  test_engine = AIEngine('8/8/8/6R1/1k5N/1P6/2PPPPPP/2BQKBNR b K - 20 35')
    print("Input Value to Play against \n1: Generic Ai \n2: Gain Max Ai \n3: MinMax Ai \n4: "
          "IterativeDeepeningAI Ai \n5: Simulate Monte  vs self X 100 \n6: Generic(w) vs Generic(b) x50"
          "\n7: Generic(w) vs MinMax(b) x50 \n8: Generic(w) vs Iterative Deepening AI(b) x50 \
          \n9: Monte Carlo(w) vs Generic(b) x50")
    user_move = input("\nInput : \033[95m")
    if user_move == "1" or user_move == "2" or user_move == "3" or user_move == "4":
        test_engine.prompt_user(user_move)
    elif user_move == "5":
        test_engine.computer.Populate_tree()
        for x in range(50):
            test_engine.computer.play_with_self_simulate()
    elif user_move == "6" or user_move == "7" or user_move == "8" or user_move == "9":
        for x in range(50):
            test_engine.play_with_self_non_verbose(user_move)
    print (test_engine.leaderboard)










