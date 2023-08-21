import tensorflow as tf
import random
import json
import datetime
from tensorflow import keras
import numpy as np
import copy

import game


players = ("a", "b")

print(f"Running with tensorflow version {tf.version.VERSION}")

def numerize_board(board):
    numerical_board = copy.deepcopy(board._board)
    for i, row in enumerate(numerical_board):
        for j, item in enumerate(row):
            if item == " ":
                numerical_board[i][j] = 0
            elif item == "a":
                numerical_board[i][j] = 1
            elif item == "b":
                numerical_board[i][j] = 2
    return numerical_board

# this is how we generate the training data for the neural network
def gen_data(n, n_ignore):
    running = True
    
    # generation loop
    for game_index in range(n):
        b = game.Board()
        current_player = players[0]

        boards = []

        # we keep track of how many moves we're into each game so we don't overfit the beginning moves
        epoch = 0
        
        # we run a game
        while True:
            player_moves = (b.get_moves(current_player), b.get_moves(b.get_enemy(current_player)))
            if player_moves == ([], []):
                break
            
            if b.get_moves(current_player) == []:
                current_player = b.get_enemy(current_player)

            moves = b.n_minimax(3, current_player, game.Board.get_score)
            b.move(current_player, random.choice(moves))

            if epoch > n_ignore:
                # we want all the pieces in a numerical representation
                numerical_board = numerize_board(b)
                # numpy has a neat flatten function for matrixes
                board_array = np.array(numerical_board).flatten()
                boards.append(board_array)
                

            current_player = b.get_enemy(current_player)
            epoch += 1

        # we keep track if the player a won or not
        # TODO: move this to numpy's native savetxt functionality
        print(f"{game_index + 1}/{n} games played")
        a_won = game.Board.get_score(b) > 0
        for board in boards:
            with open("remed.csv", "a") as f:
                board_str = np.array2string(board, separator=',')[1:-1].replace('\n', '').replace(' ', '')
                data = f"{int(a_won)},{board_str}\n"
                f.write(data)

def tensorize_board(board_list):
    """takes in a flattened board and returns a tensor
       this function is to be used for inference
    """
    board_numpy = np.array(board_list).reshape((8, 8))
    player_boards = { 1 : np.copy(board_numpy), 2 : np.copy(board_numpy) }

    for p in player_boards.keys():
        for i, row in enumerate(player_boards[p]):
            for j, cell in enumerate(row):
                if cell == p:
                    player_boards[p][i][j] = 1
                else:
                    player_boards[p][i][j] = 0

    return np.array(list(player_boards.values())).reshape((2, 8, 8))
                
def load_dataset(path):
    """load the csv dataset from path"""
    outputs = []
    inputs = []
    print("loading dataset...")
    with open(path, "r") as f:
        boards = f.readlines()

        for board in boards:
            board = [ int(cell) for cell in board.split(",") ]
            # we have to split the board into a tensor where black / white pieces
            # are separated into their own boards
            board_tensor = tensorize_board(board[1:])

            outputs.append(board[0])
            inputs.append(board_tensor)
            #print(outputs)
            #board_data[int(board[0])] = np.array(list(player_boards.values())).reshape((1, 2, 8, 8))
    print("dataset loaded")
    return (np.array(inputs), np.array(outputs))

def create_model():
    # this defines the topology of our neural network
    # we take in a tensor containing the board with white pieces
    # and the board with black pieces
    return tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(2, 8, 8)),
        tf.keras.layers.Dense(512, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(256, activation="relu"),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

def train(max_epoch, output_dir):
    # define the model architecture
    # we have a model with a total of 128 + 256 + 128 + 256 + 64 + 32 + 16 + 1 = 881 neurons
    model = create_model()

    inputs, outputs = load_dataset("remed.csv")

    loss_fn = tf.keras.losses.BinaryCrossentropy()

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=output_dir, 
        verbose=1, 
        save_freq="epoch")
    
    model.compile(optimizer="adam", loss=loss_fn, metrics=['accuracy'])

    # this is where we actually train the model
    model.fit(inputs, outputs, epochs=max_epoch, batch_size=1, verbose=1, callbacks=[cp_callback])
    
    print(inputs[1].shape)
    predictions = model.predict(np.array([inputs[1]]), batch_size=1)
    print(predictions)

def load_model(model_path):
    model = create_model()
    model.load_weights(model_path)
    return model

class ReversiAI:
    def __init__(self, model_path):
        self.model = load_model(model_path)
    def ml_eval(self, board):
        """smart evalutation function"""
        numerized_board = numerize_board(board)
        flattened_board = np.array(numerized_board).flatten()
        board_tensor = tensorize_board(flattened_board)
        evaluation = self.model.predict(np.array([board_tensor]), batch_size=1)
        return evaluation[0][0]

def benchmark(n, model):
    results = {"romeo": 0, "max_moves": 0}

    romeo = ReversiAI(model)
    for i in range(n):
        print(f"game {i}/{n}")
        b = game.Board()
        current_player = players[0]
        while True:
            player_moves = (b.get_moves(current_player), b.get_moves(b.get_enemy(current_player)))
            if player_moves == ([], []):
                break
            
            if b.get_moves(current_player) == []:
                current_player = b.get_enemy(current_player)

            moves = b.n_minimax(1, current_player, romeo.ml_eval if current_player == "b" else game.Board.get_score)
            b.move(current_player, random.choice(moves))

            current_player = b.get_enemy(current_player)
        results["romeo" if game.Board.get_score(b) < 0 else "max_moves"] += 1
    with open("benchmark.txt", "w") as f:
        f.write(str(results))

if __name__ == "__main__":
    # compare model 1 and model 2
    romeo = ReversiAI("model/")
    robin = ReversiAI("model2/")
    b = game.Board()
    print(romeo.ml_eval(b))
    print(robin.ml_eval(b))
    
