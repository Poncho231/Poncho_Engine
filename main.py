import cupy as np
import chess
import os
import numpy as np_cpu
import cProfile
import pstats
import random

def save_model_parameters(weights_to_save, biases_to_save, filename="poncho_engine_model.npz"):
    """Saves model weights and biases to a .npz file."""
    print(f"Saving model parameters to {filename}...")
    # Convert CuPy arrays to NumPy arrays for saving
    weights_cpu = np.asnumpy(weights_to_save)
    biases_cpu = np.asnumpy(biases_to_save)
    np_cpu.savez(filename, weights=weights_cpu, biases=biases_cpu)
    print("Model parameters saved.")

def load_model_parameters(filename="poncho_engine_model.npz"):
    """Loads model weights and biases from a .npz file."""
    if os.path.exists(filename):
        print(f"Loading model parameters from {filename}...")
        data = np_cpu.load(filename)
        # Convert loaded NumPy arrays back to CuPy arrays
        weights_loaded = np.asarray(data['weights'])
        biases_loaded = np.asarray(data['biases'])
        print("Model parameters loaded.")
        return weights_loaded, biases_loaded
    else:
        print(f"No saved model found at {filename}. Initializing new parameters.")
        return None, None

convert = {
    "p": 1,    "b": 2,  "n": 3,  "r": 4,  "q": 5,  "k": 6,
    "P": 7,    "B": 8,  "N": 9,  "R": 10, "Q": 11, "K": 12,
    ".": 0
}
game = []

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sig_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

def material_count(board, color):
    piece_values = {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9
        # King is not included (it's never captured)
    }
    material = 0
    for piece_type, value in piece_values.items():
        material += len(board.pieces(piece_type, color)) * value
    return material

def choose_move(values, moves, turn):
    if random.uniform(0, 1) < 0.35:
        return np_cpu.random.choice(moves)
    return moves[np.argmax(values).item()] if turn == -1 else moves[np.argmin(values).item()]

def convert_board(board: chess.Board):
    board_representation = []
    white_mat = material_count(board, chess.WHITE)
    black_mat = material_count(board, chess.BLACK)
    mat_diff = (white_mat - black_mat) / 39.0  # 39 = max possible non-king material
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            board_representation.append(float(convert[piece.symbol()]))
        else:
            board_representation.append(0.0)
    board_representation.append(1.0 if board.turn == chess.WHITE else -1.0)
    board_representation.append((white_mat-black_mat)/39)
    return board_representation


# tanh, ReLU, tanh
alpha_lrelu = 2 # Or 0.1
non_linears = [
    lambda x: np.where(x > 0, x, x * alpha_lrelu),
    lambda x: np.tanh(x)
]
derivatives  = [
    lambda x: np.where(x > 0, 1.0, alpha_lrelu).astype(np.float32),
    lambda x: 1-np.tanh(x)**2
]
loaded_weights, loaded_biases = load_model_parameters()

if loaded_weights is not None and loaded_biases is not None:
    weights = loaded_weights
    biases = loaded_biases
else:
    rs = np.random.default_rng()
    weights = rs.uniform(-1.7,1.7,(6,66,66)).astype(np.float32)
    biases = np.random.uniform(-1.7, 1.7, (6, 66)).astype(np.float32)

head_w = np.random.uniform(-1.5, 1.5, size=(66,1)).astype(np.float32)
head_b = np.zeros((1,), np.float32)
def eval(weights, biases, current_input_activation):
    """Return z_list, a_list, and final scalar output."""
    x = current_input_activation # x will be updated through layers
    z_list, a_list = [], [x]

    for i in range(6):
        z = x @ weights[i] + biases[i]
        z_list.append(z)
        x = non_linears[i % 2](z) # x is updated here
        a_list.append(x)

    # x now holds the activations of the last layer
    before_sig = (x @ head_w).ravel() + head_b
    y_hat = sigmoid(before_sig)  # Use the final 'x'
    return z_list, a_list, before_sig, y_hat


def gradient():
    global weights
    global biases
    global result
    result = np.array(result, dtype=np.float32)
    result_vec = np.ones(len(game))*result
    positions = []

    weight_grad_l = np.zeros_like(weights)
    bias_grad_l = np.zeros_like(biases)

    for i in game:
        positions.append(convert_board(i))
    z_list, s_list, before_sig, input_v = eval(weights, biases, np.array(positions))
    dLossdSum = input_v - result_vec
    dSumdSig = sig_derivative(before_sig)
    dL_du = dLossdSum * dSumdSig
    final_feats = s_list[-1]
    head_w_grad = final_feats.T @ dL_du.reshape(-1, 1)
    head_b_grad = np.sum(dL_du)

    delta_all = dL_du.reshape(-1, 1) * head_w.ravel()
    for i in range(len(input_v)):
        delta_cur = delta_all[i]

        for layer in reversed(range(6)):
            dz = delta_cur * derivatives[layer % 2](z_list[layer][i])  # scalar multiplication
            weight_grad_l[layer] += np.outer(s_list[layer][i], dz)
            bias_grad_l[layer] += dz
            delta_cur = weights[layer].T @ dz

    return head_w_grad, head_b_grad, weight_grad_l, bias_grad_l  # same shapes as weights / biases


profiler = cProfile.Profile()
profiler.enable()

board = chess.Board()
turn = 1
for j in range(200):

    turn = 1
    while not board.is_game_over():
        best_move = None

        possible_moves = list(board.legal_moves)
        possible_moves_full_cpu = []
        for i in possible_moves:
            board.push(i)
            possible_moves_full_cpu.append(convert_board(board))
            board.pop()
        possible_moves_full_gpu = np.array(possible_moves_full_cpu)
        value_vector = eval(weights, biases, possible_moves_full_gpu)[-1]
        """
        for idx, move in enumerate(possible_moves):
            original_score_for_this_move = value_vector[idx]
            board.push(move)
            if board.is_repetition(3):
                if turn == 1:
                    if original_score_for_this_move > 0.5:
                        value_vector[idx] = 0.5
                else:
                    if original_score_for_this_move < 0.5:
                        value_vector[idx] = 0.5
            board.pop()
        """

        best_move = choose_move(value_vector, possible_moves, turn)
        board.push(best_move)
        attempts = 0
        max_attempts = 20  # avoid infinite loops in pathological cases

        while board.is_repetition(2) and attempts < max_attempts:
            board.pop()  # remove the last move
            # Recompute legal moves and their values for the new board
            possible_moves = list(board.legal_moves)
            if not possible_moves:
                break  # No legal moves: game is over

            possible_moves_full_cpu = []
            for i in possible_moves:
                board.push(i)
                possible_moves_full_cpu.append(convert_board(board))
                board.pop()
            possible_moves_full_gpu = np.array(possible_moves_full_cpu)
            value_vector = eval(weights, biases, possible_moves_full_gpu)[-1]
            best_move = choose_move(value_vector, possible_moves, turn)
            board.push(best_move)
            attempts += 1

        game.append(board.copy())
        turn = -turn  # Flip turn for the next player
    game.append(board.copy())
    game_outcome_string = board.result()
    if game_outcome_string == "1-0":  # White wins
        result = 1.0
    elif game_outcome_string == "0-1":  # Black wins
        result = 0.0
    else:  # Draw "1/2-1/2"
        result = 0.5
    lr = 0.1
    head_w_grad, head_b_grad, weight_grad, bias_grad = gradient()
    weights -= weight_grad * lr / len(game)
    biases -= bias_grad * lr / len(game)
    head_w -= head_w_grad * lr / len(game)
    head_b -= head_b_grad * lr / len(game)
    print(result)
    if (j + 1) % 10 == 0:  # Save after every game (or change 1 to 10 to save every 10 games, etc.)
        save_model_parameters(weights, biases)
    board = chess.Board()
    game.clear()
turn = 1
while not board.is_game_over():
    best_move = None

    possible_moves = list(board.legal_moves)
    possible_moves_full_cpu = []
    for i in possible_moves:
        board.push(i)
        possible_moves_full_cpu.append(convert_board(board))
        board.pop()
    possible_moves_full_gpu = np.array(possible_moves_full_cpu)
    value_vector = eval(weights, biases, possible_moves_full_gpu)[-1]
    """
    for idx, move in enumerate(possible_moves):
        original_score_for_this_move = value_vector[idx]
        board.push(move)
        if board.is_repetition(3):
            if turn == 1:
                if original_score_for_this_move > 0.5:
                    value_vector[idx] = 0.5
            else:
                if original_score_for_this_move < 0.5:
                    value_vector[idx] = 0.5
        board.pop()
    """

    best_move = choose_move(value_vector, possible_moves, turn)
    board.push(best_move)
    attempts = 0
    max_attempts = 20  # avoid infinite loops in pathological cases

    while board.is_repetition(2) and attempts < max_attempts:
        board.pop()  # remove the last move
        # Recompute legal moves and their values for the new board
        possible_moves = list(board.legal_moves)
        if not possible_moves:
            break  # No legal moves: game is over

        possible_moves_full_cpu = []
        for i in possible_moves:
            board.push(i)
            possible_moves_full_cpu.append(convert_board(board))
            board.pop()
        possible_moves_full_gpu = np.array(possible_moves_full_cpu)
        value_vector = eval(weights, biases, possible_moves_full_gpu)[-1]
        best_move = choose_move(value_vector, possible_moves, turn)
        board.push(best_move)
        attempts += 1

    game.append(board.copy())
    turn = -turn  # Flip turn for the next player
uci_string = " ".join(move.uci() for move in board.move_stack)
print(uci_string)
game.append(board.copy())
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime') # 'cumtime' is cumulative time
stats.print_stats(20) # Show top 20 time consumers