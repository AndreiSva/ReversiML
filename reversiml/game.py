import copy
import random
import time

class Board:
    # this represents the depth of the board in the posibility tree
    # this is going to be zero for the first (original) board and increase with each
    # minimax iteration
    depth = 0
    def __init__(self):
        self._board = [
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", "a", "b", " ", " ", " "],
            [" ", " ", " ", "b", "a", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "],
            [" ", " ", " ", " ", " ", " ", " ", " "]
        ]
    def is_valid(self, p):
        """check if move p is valid"""
        return p[1] >= 0 and p[1] < len(self._board[0]) and p[0] >= 0 and p[0] < len(self._board[0])
    def get_enemy(self, player):
        """returns the enemy of the player"""
        if player == "a":
            return "b"
        else:
            return "a"
    def get(self, p):
        """returns the value of a cell on the board"""
        if self.is_valid(p):
            return self._board[p[1]][p[0]]
        else:
            return False
    def set(self, p, v):
        """sets the value of a cell on the board"""
        if self.is_valid(p):
            self._board[p[1]][p[0]] = v
    def calc_move(self, player: str, pos: tuple):
        """returns a list of all the cells that have to be changed if player 'player' moves to position 'pos'"""
        moves = set()
        directions = [
            (-1, 0), (1, 0), (0, 1), (0, -1), (-1, -1), (-1, 1), (1, 1), (1, -1)
        ]
        moves.add(pos)

        if self.get(pos) not in ["a", "b"]:
            for direction in directions:
                direction_moves = set()
                current_pos = (pos[0] + direction[0], pos[1] + direction[1])

                while self.is_valid(current_pos):
                    cell_value = self.get(current_pos)

                    if cell_value == player:
                        moves.update(direction_moves)
                        break
                    elif cell_value == self.get_enemy(player):
                        direction_moves.add(current_pos)
                    else:
                        break

                    current_pos = (current_pos[0] + direction[0], current_pos[1] + direction[1])

                    if self.get(current_pos) not in ["a", "b"]:
                        break
        return list(moves)
    def get_moves(self, player):
        """returns a list of all valid moves for player 'player'"""
        return [(j, i) for i, row in enumerate(self._board) for j, cell in enumerate(row) if len(self.calc_move(player, (j, i))) > 1]

    @staticmethod
    def get_score(board):
        """return an evaluation of the board"""
        score = 0
        for row in board._board:
            for cell in row:
                if cell == "a":
                    score += 1
                elif cell == "b":
                    score -= 1
        return score
    
    def move(self, player, pos):
        """make a move on the board
           returns False if the move is not valid"""
        if len(moves := self.calc_move(player, pos)) > 1:
            for move in moves:
                self.set(move, player)
        else:
            return False

    def max_moves(self, player, feval):
        fcmp = (max if player == "a" else min)
        moves = self.get_moves(player)
        move_results = []
        
        for move in moves:
            b = Board()
            b._board = copy.deepcopy(self._board)
            b.move(player, move)
            move_results.append((move, feval(b)))

        return [move[0] for move in move_results if move[1] == fcmp(move_results, key=lambda k: k[1])[1]]
        
    def n_minimax(self, n, player, feval):
        fcmp = (max if player == "a" else min)
        if n > 0:
            moves = self.get_moves(player)
            move_results = []
            for move in moves:
                b = Board()
                # we increment the depth for the new board
                b.depth = self.depth + 1
                b._board = copy.deepcopy(self._board)
                b.move(player, move)

                enemy_moves = b.max_moves(self.get_enemy(player), feval)

                # the enemy doesn't have any more moves so we can just report the current board evaluation
                if enemy_moves == []:
                    move_results.append((move, feval(self)))
                    continue
                
                b.move(self.get_enemy(player), random.choice(enemy_moves))
                # this is a branch in the possibility tree of the game
                branch = b.n_minimax(n - 1, player, feval)

                # we store the maximum evaluation of the branch
                move_results.append((move, fcmp(branch, key = lambda k : k[1])[1]))

            if move_results != []:
                # if we're finished with the recursion (depth 0) we can exclude the evaluations from the return
                # otherwise we return the move / evaluation pair
                return [move[0] if self.depth == 0 else move for move in move_results if move[1] == fcmp(move_results, key=lambda k: k[1])[1]]
            else:
                # end of branch
                return [(None, feval(self))]
        else:
            return [(None, feval(self))]
        
        
    def __str__(self):
        return "\n".join([str(row) for row in self._board])
    


def cliplay():
    b = Board()

    a_moves = b.get_moves("a")
    b_moves = b.get_moves("b")
    
    while a_moves != [] and b_moves != []:
        a_moves = b.get_moves("a")
        b_moves = b.get_moves("b")
        a_minimax = b.n_minimax(2, "a", b.get_score)
        a_move = random.choice(list(filter(lambda x: x[1] == max(a_minimax, key=lambda k: k[1])[1], a_minimax)))[0]
        if a_move != None:
            b.move("a", a_move)
        print("\n" + str(b))
        print(list(filter(lambda x: x[1] == max(a_minimax, key=lambda k: k[1])[1], a_minimax)))
        print(f"Score: {b.get_score(b)}")
        time.sleep(0.5)
        b_moves = b.max_moves("b", b.get_score)
        if b_moves != []:
            b_move = random.choice(b_moves)
            b.move("b", b_move)
        print("\n" + str(b))
        print(f"Score: {b.get_score(b)}")
        time.sleep(0.5)
       

if __name__ == "__main__":
    cliplay()
