# ReversiML
A reversi game that uses machine learning for the evaluation function.

[![License: MIT](https://img.shields.io/bower/l/mi?style=for-the-badge)](https://opensource.org/license/mit/)

![sponsors](https://img.shields.io/github/sponsors/0?style=for-the-badge)

# Status

- [X] advanced minimax 
- [X] user interface
- [X] supervized learning

# Running
Run `pip install tensorflow numpy pygame` to install the necessary dependencies. The project relies on tensorflow for the machine learning and on pygame for the user interface.

Afterwards, you can run the gui version of the game by running `python3 reversiml/main.py`, which will launch a game between a user and Model 2.

# Models

- Model
  - original model trained on 1000 games excluding the first five moves of each game
  - is pretty bad against human players
  - might offer more statistically accurate evaluations for beginning moves
- Model 2
  - improved model trained on the original 1000 games plus an extra 670 games featuring all moves
  - is better against human players
