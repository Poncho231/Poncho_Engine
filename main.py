import cupy as np
import chess
import os
import numpy as np_cpu
import cProfile
import pstats
import random

np.cuda.set_allocator(np.cuda.MemoryPool().malloc)
np.cuda.set_pinned_memory_allocator()

cache = {} # Add caching for positions, flush after each grad descent
convert_cache = {} # Add caching for boards->positions. Key is the fen() + castling rights + en passant
def save_model_parameters(weights_to_save, biases_to_save, head_w_to_save, head_b_to_save,
                          filename="poncho_engine_model.npz"):
    """Saves model weights, biases, and head layer parameters to a .npz file."""
    print(f"Saving model parameters to {filename}...")
    # Convert CuPy arrays to NumPy arrays for saving
    weights_cpu = np.asnumpy(weights_to_save)
    biases_cpu  = np.asnumpy(biases_to_save)
    head_w_cpu  = np.asnumpy(head_w_to_save)
    head_b_cpu  = np.asnumpy(head_b_to_save)

    np_cpu.savez(
        filename,
        weights=weights_cpu,
        biases=biases_cpu,
        head_w=head_w_cpu,
        head_b=head_b_cpu
    )
    print("Model parameters saved.")

def load_model_parameters(filename="poncho_engine_model.npz"):
    """Loads model weights, biases, and head parameters from a .npz file."""
    if os.path.exists(filename):
        print(f"Loading model parameters from {filename}...")
        data = np_cpu.load(filename)
        # Convert loaded NumPy arrays back to CuPy arrays
        weights_loaded = np.asarray(data['weights'])
        biases_loaded  = np.asarray(data['biases'])
        head_w_loaded  = np.asarray(data['head_w'])
        head_b_loaded  = np.asarray(data['head_b'])
        print("Model parameters loaded.")
        return weights_loaded, biases_loaded, head_w_loaded, head_b_loaded
    else:
        print(f"No saved model found at {filename}. Initializing new parameters.")
        return None, None, None, None
convert = {
    "p": 1,    "b": 2,  "n": 3,  "r": 4,  "q": 5,  "k": 6,
    "P": 7,    "B": 8,  "N": 9,  "R": 10, "Q": 11, "K": 12,
    ".": 0
}
game = []

def sigmoid(z):
    return 1/(1 + np.exp(-z))


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
    # turn ==  +1  → White  (maximise value)
    # turn == –1  → Black  (minimise value)
    if random.random() < 0.15:
        return np_cpu.random.choice(moves)
    return moves[np.argmax(values).item()] if turn == +1 else moves[np.argmin(values).item()]

PIECE_VALUES_FOR_MATERIAL = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
    # King's material value is not typically counted
}

def convert_board(board: chess.Board):
    features = np.zeros(66, dtype=np.float32)
    piece_map = board.piece_map()
    white_material = 0
    black_material = 0
    for square, piece in piece_map.items():
        features[square] = convert[piece.symbol()]
        if piece.piece_type in PIECE_VALUES_FOR_MATERIAL:
            if piece.color == chess.WHITE:
                white_material += PIECE_VALUES_FOR_MATERIAL[piece.piece_type]
            else:
                black_material += PIECE_VALUES_FOR_MATERIAL[piece.piece_type]

    material_difference = (white_material - black_material) / 39.0
    features[65] = material_difference

    features[64] = 1 if board.turn == chess.WHITE else -1
    return features


# tanh, ReLU, tanh
alpha_lrelu = 0.01
non_linears = [
    lambda x: np.where(x > 0, x, x * alpha_lrelu),
    lambda x: np.tanh(x)
]
derivatives  = [
    lambda x: np.where(x > 0, 1.0, alpha_lrelu).astype(np.float32),
    lambda x: 1-np.tanh(x)**2
]
loaded_weights, loaded_biases, head_w, head_b = load_model_parameters()


if loaded_weights is not None and loaded_biases is not None:
    weights = loaded_weights
    biases = loaded_biases
else:
    # Glorot (Xavier) normal init for 6 hidden layers of size 66→66
    weights = np.zeros((6, 66, 66), dtype=np.float32)
    biases  = np.zeros((6, 66), dtype=np.float32)
    fan_in, fan_out = 66, 1
    std = np.sqrt(2.0 / (fan_in + fan_out))
    head_w = np.random.randn(fan_in, fan_out).astype(np.float32) * std
    head_b = np.zeros((fan_out,), dtype=np.float32)
    for i in range(6):
        fan_in, fan_out = 66, 66
        std = np.sqrt(2.0 / (fan_in + fan_out))
        weights[i] = np.random.randn(fan_in, fan_out).astype(np.float32) * std
        # biases stay zero for symmetry
        biases[i]  = np.zeros((fan_out,), dtype=np.float32)

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
    y_hat = sigmoid(before_sig.copy())  # Use the final 'x'
    return z_list, a_list, before_sig, y_hat

boards = []
def minimax(depth, turn, board: chess.Board):
    global boards

    b = (board.board_fen(), board.turn, board.castling_rights, board.ep_square)
    key = (b, depth, turn)
    if key in cache:
        return cache[key]

    possible_moves = list(board.legal_moves)

    if depth == 0:
        best_value = None
        best_move = None
        boards.append(board.copy())

        # Process in batches
        if len(boards) >= 15:  # Process when we have 15 boards
            # Create big matrix with all possible moves from all boards
            all_positions = []
            move_mappings = []  # (board_idx, move_idx, move, turn)

            for board_idx, curr_board in enumerate(boards):
                curr_moves = list(curr_board.legal_moves)
                for move_idx, move in enumerate(curr_moves):
                    curr_board.push(move)
                    if curr_board.is_checkmate():
                        curr_board.pop()
                        best_value = 1 if turn == 1 else 0
                        if best_value == 1 and turn == 1 or best_value == 0 and turn == -1:
                            best_move = move
                        else:
                            best_value = None
                    cache_key = (curr_board.board_fen(), curr_board.turn, curr_board.castling_rights, curr_board.ep_square)
                    if cache_key not in convert_cache:
                        convert_cache[cache_key] = convert_board(curr_board)
                    all_positions.append(convert_cache[cache_key])
                    move_mappings.append((board_idx, move_idx, move, turn))
                    curr_board.pop()
            # Run eval on the big matrix
            big_matrix = np.array(all_positions)
            fat_value_vector = eval(weights, biases, big_matrix)[-1]
            if best_value is None:
                # Find max/min and map back to original moves
                if turn == 1:  # White maximizes
                    best_idx = np.argmax(fat_value_vector).item()
                else:  # Black minimizes
                    best_idx = np.argmin(fat_value_vector).item()

                board_idx, move_idx, best_move, _ = move_mappings[best_idx]

                best_value = fat_value_vector[best_idx]
            # If it's not None, we found mate

            # Clear the batch
            boards.clear()

            cache[key] = (best_value, best_move)
            return best_value, best_move
        else:
            # If batch not full, go back to original method
            positions = []
            for i in possible_moves:
                board.push(i)
                cache_key = (board.board_fen(), board.turn, board.castling_rights, board.ep_square)
                if cache_key not in convert_cache:
                    convert_cache[cache_key] = convert_board(board)
                positions.append(convert_cache[cache_key])
                board.pop()
            value_vector = eval(weights, biases, np.array(positions))[-1]

            for idx, move in enumerate(possible_moves):
                board.push(move)
                if board.is_repetition(3):
                    if turn == 1:
                        if value_vector[idx] > 0.5:
                            value_vector[idx] = 0.5
                    else:
                        if value_vector[idx] < 0.5:
                            value_vector[idx] = 0.5
                board.pop()

            best_move = choose_move(value_vector, possible_moves, turn)
            cache[key] = (np.max(value_vector) if turn == 1 else np.min(value_vector), best_move)
            return np.max(value_vector) if turn == 1 else np.min(value_vector), best_move

    best_value = 2 if board.turn == chess.BLACK else -2
    absolute_best_move = possible_moves[0]

    # Process 5-10 moves at a time instead of all at once
    batch_size = 8
    for i in range(0, len(possible_moves), batch_size):
        batch_moves = possible_moves[i:i + batch_size]

        for move in batch_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                best_value = 1 if turn == 1 else 0
                return best_value, move
            elif board.is_repetition(
                    3) or board.is_stalemate() or board.is_fifty_moves() or board.is_insufficient_material():
                value, best_move = 0.5, move
            else:
                value = minimax(depth - 1, -turn, board)[0]
            if value * turn > best_value * turn:
                absolute_best_move = move
                best_value = value
            board.pop()

    cache[key] = (best_value, absolute_best_move)
    return best_value, absolute_best_move


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




board = chess.Board()
turn = 1
for j in range(10):
    turn = 1
    while not board.is_game_over():
        profiler = cProfile.Profile()
        profiler.enable()
        value, best_move = minimax(2, turn, board)
        board.push(best_move)
        print("Values: ", value)
        turn = -turn  # Flip turn for the next player
        game.append(board.copy())
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats('cumtime')  # 'cumtime' is cumulative time
        stats.print_stats(20)  # Show top 20 time consumers
    game_outcome_string = board.result()
    if game_outcome_string == "1-0":  # White wins
        result = 1.0
    elif game_outcome_string == "0-1":  # Black wins
        result = 0.0
    else:  # Draw "1/2-1/2"
        result = 0.5
    lr = 0.001
    head_w_grad, head_b_grad, weight_grad, bias_grad = gradient()
    weights -= weight_grad * lr / len(game)
    biases -= bias_grad * lr / len(game)
    head_w -= head_w_grad * lr / len(game)
    head_b -= head_b_grad * lr / len(game)
    print(result)
    uci_string = " ".join(move.uci() for move in board.move_stack)
    print(uci_string)
    del cache
    cache = {}
    if (j + 1) % 10 == 0:  # Save after every game (or change 1 to 10 to save every 10 games, etc.)
        save_model_parameters(weights, biases, head_w, head_b)
    board = chess.Board()
    game.clear()
turn = 1


while not board.is_game_over():
    value, best_move = minimax(1, turn, board)
    board.push(best_move)
    print("Values: ", value)
    print(f"Cache size: {len(convert_cache)}, Cache hits vs misses ratio")
    turn = -turn  # Flip turn for the next player
uci_string = " ".join(move.uci() for move in board.move_stack)
print(uci_string)
game.append(board.copy())
