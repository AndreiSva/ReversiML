import logging
import sys
import os
import pygame
import argparse
import random
import time
import ml

import game

sys.path.append(os.path.dirname(os.path.realpath(__file__)))

SCREEN_SIZE = 512

parser = argparse.ArgumentParser()
parser.add_argument('--autoplay', action='store_true')
args = parser.parse_args()

BOT_MOVE_EVENT = pygame.USEREVENT

def main():
    logging.info("Starting Engine")

    pygame.init()    
    screen = pygame.display.set_mode((SCREEN_SIZE, SCREEN_SIZE))
    clock = pygame.time.Clock()
    pygame.display.set_caption("ReversiML")

    players = ("a", "b")
    current_player = players[0]
    
    board = game.Board()
    player_moves = board.get_moves(current_player)

    # load robin
    # robin was trained on 1000 + 670 games
    romeo = ml.ReversiAI("model2/")

    print(romeo.ml_eval(board))
    
    running = True
    while running:
        clock.tick(60)
        
        # event loop
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONUP and args.autoplay == False:
                x, y = pygame.mouse.get_pos()
                mouse_cell = (x // (SCREEN_SIZE // 8), y // (SCREEN_SIZE // 8))
                if board.get(mouse_cell) not in players and mouse_cell in player_moves:
                    board.move(current_player, mouse_cell)
                    current_player = board.get_enemy(current_player)
                    player_moves = board.get_moves(current_player)

        screen.fill("black")

        # render board and pieces
        for row in range(8):
            for cell in range(8):
                cell_rect = pygame.Rect((cell * SCREEN_SIZE / 8, row * SCREEN_SIZE / 8), (SCREEN_SIZE / 8, SCREEN_SIZE / 8))
                if (cell, row) not in player_moves:
                    cell_color = "#33cc33" if (cell + row) % 2 == 0 else "#00cc00"
                else:
                    cell_color = "blue"
                pygame.draw.rect(screen, cell_color, cell_rect)

                if (piece := board.get((cell, row))) in players:
                    piece_color = "white" if piece == "a" else "black"
                    pygame.draw.circle(screen, piece_color, (cell * SCREEN_SIZE / 8 + SCREEN_SIZE / 16, row * SCREEN_SIZE / 8 + SCREEN_SIZE / 16), SCREEN_SIZE / 16)
        
        pygame.display.flip()

        if board.get_moves(current_player) == []:
            current_player = board.get_enemy(current_player)

        if current_player == "b":
            if len(board.get_moves(current_player)) < 1:
                continue
            minimax_moves = board.n_minimax(1, current_player, game.Board.get_score if current_player == "a" else romeo.ml_eval)
            board.move(current_player, random.choice(minimax_moves))
            current_player = board.get_enemy(current_player)
            player_moves = board.get_moves(current_player)
            print(f"score: {game.Board.get_score(board)}")
            print(f"eval: {romeo.ml_eval(board)}")
        
    pygame.quit()

if __name__ == "__main__":
    main()
