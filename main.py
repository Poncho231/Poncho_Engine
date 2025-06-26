import numpy as np
import chess
import os
import numpy as np_cpu
import cProfile
import pstats

import random
from multiprocessing import Pool, cpu_count

cores_to_use = cpu_count()
cache = {} # Add caching for positions, flush after each grad descent
convert_cache = {} # Add caching for boards->positions. Key is the fen() + castling rights + en passant
def save_model_parameters(weights_to_save, biases_to_save, head_w_to_save, head_b_to_save,
                          filename="poncho_engine_model.npz"):
    """Saves model weights, biases, and head layer parameters to a .npz file."""
    print(f"Saving model parameters to {filename}...")
    # Convert CuPy arrays to NumPy arrays for saving
    weights_cpu = weights_to_save
    biases_cpu = biases_to_save
    head_w_cpu = head_w_to_save
    head_b_cpu = head_b_to_save

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


def save_games(games_to_save, results_to_save, filename="poncho_engine_all_games.npz"):
    """
    Saves games and results to a .npz file.

    Args:
        games_to_save: List of games (each game is a list of board positions)
        results_to_save: List of game results
        filename: Name of file to save to
    """
    print(f"Saving {len(games_to_save)} games to {filename}...")

    # Convert CuPy arrays to NumPy if needed and prepare data
    games_cpu = []
    for game in games_to_save:
        game_positions = []
        for pos in game:
            if hasattr(pos, 'get'):  # Check if it's a CuPy array
                game_positions.append(np.asnumpy(pos))
            else:
                game_positions.append(np.array(pos) if not isinstance(pos, np.ndarray) else pos)
        games_cpu.append(game_positions)

    results_cpu = np.asnumpy(results_to_save) if hasattr(results_to_save, 'get') else np.array(results_to_save)

    # Save as object arrays to handle variable-length games
    np_cpu.savez(
        filename,
        games=np.array(games_cpu, dtype=object),
        results=results_cpu
    )
    print("Games saved successfully.")


def load_games(filename="poncho_engine_all_games.npz"):
    """
    Loads games data from a .npz file.

    Returns:
        games: List of games, where each game is a list of board positions
        results: List of game results (1.0 for white win, 0.0 for black win, 0.5 for draw)
    """
    if os.path.exists(filename):
        print(f"Loading games from {filename}...")
        try:
            data = np_cpu.load(filename, allow_pickle=True)

            # Handle different possible formats
            if 'games' in data and 'results' in data:
                games_loaded = data['games']
                results_loaded = data['results']
            elif 'arr_0' in data and 'arr_1' in data:
                # If saved with np.savez without named parameters
                games_loaded = data['arr_0']
                results_loaded = data['arr_1']
            else:
                # Try to find the data by examining all keys
                keys = list(data.keys())
                print(f"Available keys in file: {keys}")
                if len(keys) >= 2:
                    games_loaded = data[keys[0]]
                    results_loaded = data[keys[1]]
                else:
                    raise ValueError("Unexpected file format")

            # Convert games from object array back to list of lists
            if isinstance(games_loaded, np.ndarray):
                if games_loaded.dtype == object:
                    # Handle object array (variable-length games)
                    games = [list(game) for game in games_loaded]
                else:
                    # Handle regular array
                    games = games_loaded.tolist()
            else:
                games = list(games_loaded)

            # Convert results
            if isinstance(results_loaded, np.ndarray):
                results = results_loaded.tolist()
            else:
                results = list(results_loaded)

            print(f"Loaded {len(games)} games with {len(results)} results")
            return games, results

        except Exception as e:
            print(f"Error loading games from {filename}: {e}")
            return [], []
    else:
        print(f"No games file found at {filename}")
        return [], []
game = []
games = []
results = []

def sigmoid(z):
    return 1/(1 + np.exp(-z))


def sig_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))

piece_values = {
    # black pieces (color=False)
    (chess.PAWN,   False): -1,
    (chess.BISHOP, False): -2,
    (chess.KNIGHT, False): -3,
    (chess.ROOK,   False): -4,
    (chess.QUEEN,  False): -5,
    (chess.KING,   False): -6,
    # white pieces (color=True)
    (chess.PAWN, True): 1,
    (chess.BISHOP, True): 2,
    (chess.KNIGHT, True): 3,
    (chess.ROOK, True): 4,
    (chess.QUEEN, True): 5,
    (chess.KING, True): 6,
}

piece_vals_single = {
    chess.PAWN: 1,
    chess.KNIGHT: 3,
    chess.BISHOP: 3,
    chess.ROOK: 5,
    chess.QUEEN: 9
}


def move_score(board, m):
    score = 0
    # Captures are good candidates for early evaluation
    if board.is_capture(m):
        victim_piece = board.piece_at(m.to_square)
        if victim_piece:
            score += 10 * piece_vals_single.get(victim_piece.piece_type, 0)
    # Prioritize checks
    board.push(m)
    if board.is_check():
        score += 5
    if board.is_checkmate():
        score += 1000
    board.pop()
    return score
def choose_move(values, moves, turn):
    # turn ==  +1  → White  (maximise value)
    # turn == –1  → Black  (minimise value)
    if random.randint(1, 10) == 3:
        return random.choice(moves)
    return moves[np.argmax(values).item()] if turn == +1 else moves[np.argmin(values).item()]
features = np.zeros(66, dtype=np.float32)
def convert_board(board: chess.Board):
    global features
    features.fill(0.0)
    for (piece, color), encoded in piece_values.items():
        bitboard = board.pieces(piece, color)
        b = int(bitboard)
        while b:
            LSB = b & -b
            features[LSB.bit_length() - 1] = encoded
            b ^= LSB
    white_mat = sum(piece_vals_single[i]*int(board.pieces(i, True)).bit_count() for i in piece_vals_single)
    black_mat = sum(piece_vals_single[i] * int(board.pieces(i, False)).bit_count() for i in piece_vals_single)
    features[65] = (white_mat - black_mat) / 9
    features[64] = 1 if board.turn == chess.WHITE else -1
    return features.copy()



# tanh, ReLU, tanh
alpha_lrelu = 0.01
non_linears = [
    lambda x: np.where(x > 0, x, x * alpha_lrelu),
    lambda x: np.tanh(x)
]
derivatives = [
    lambda x: np.where(x > 0, 1.0, alpha_lrelu).astype(np.float32),
    lambda x: 1-np.tanh(x)**2
]



def eval(weights, biases, head_w, head_b, current_input_activation):
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
    y_hat = before_sig.copy()
    return z_list, a_list, y_hat

boards = []
# Alpha - best value for white, Beta - Best value for black
def minimax(depth, turn, board: chess.Board, alpha=-float('inf'), beta=float('inf')):
    global boards

    key = (board._transposition_key(), depth, turn)
    if key in cache:
        return cache[key]
    possible_moves = list(board.legal_moves)
    if depth == 0:
        positions = []
        for i in possible_moves:
            board.push(i)
            cache_key = board._transposition_key()
            if cache_key not in convert_cache:
                convert_cache[cache_key] = convert_board(board)
            positions.append(convert_cache[cache_key])
            board.pop()
        value_vector = eval(weights, biases, head_w, head_b, np.array(positions))[-1]
        for idx, move in enumerate(possible_moves):
            board.push(move)
            if board.is_repetition(3):
                value_vector[idx] = 0.5
            if board.is_checkmate():
                value_vector[idx] = float('inf') if board.turn == chess.BLACK else -float('inf')
            board.pop()
        best_move = choose_move(value_vector, possible_moves, turn)
        cache[key] = (np.max(value_vector) if turn == 1 else np.min(value_vector), best_move)
        return np.max(value_vector) if turn == 1 else np.min(value_vector), best_move

    best_value = float('inf') if board.turn == chess.BLACK else -float('inf')
    absolute_best_move = possible_moves[0]
    best_move = absolute_best_move

    batch_size = 1
    possible_moves.sort(key= lambda m: move_score(board, m), reverse=True)
    for i in range(0, len(possible_moves), batch_size):
        batch_moves = possible_moves[i:i + batch_size]
        for move in batch_moves:
            board.push(move)
            if board.is_checkmate():
                board.pop()
                best_value = float('inf') if turn == 1 else -float('inf')
                return best_value, move
            elif board.is_repetition(
                    3) or board.is_stalemate() or board.is_fifty_moves() or board.is_insufficient_material():
                value = 0.5
            else:
                value = minimax(depth - 1, -turn, board, alpha, beta)[0]
            if turn == 1:
                alpha = max(alpha, value)
                if value > best_value:
                    best_value = value
                    best_move = move
                if beta <= alpha: # Meaning, white can now assure a move worse for black
                    board.pop()
                    return best_value, best_move
            else: # Meaning, black can now assure a move worse for white
                beta = min(beta, value)
                if value < best_value:
                    best_value = value
                    best_move = move
                if beta <= alpha:
                    board.pop()
                    return best_value, best_move
            board.pop()

    cache[key] = (best_value, best_move)
    return best_value, best_move


def initialize_network_properly():
    """Initialize a 6-layer network with proper He and Glorot initialization based on activation functions"""
    # 6 layers with 66→66 dimensions
    weights = np.zeros((6, 66, 66), dtype=np.float32)
    biases = np.zeros((6, 66), dtype=np.float32)

    fan_in, fan_out = 66, 66

    for i in range(6):
        if i % 2 == 0:  # Layers 0, 2, 4 use LeakyReLU
            # He initialization for LeakyReLU layers
            std = np.sqrt(2.0 / fan_in)
            weights[i] = np.random.randn(fan_in, fan_out).astype(np.float32) * std
            # Small non-zero bias for better training dynamics
            biases[i] = np.random.randn(fan_out).astype(np.float32) * 0.01
        else:  # Layers 1, 3, 5 use tanh
            # Glorot/Xavier initialization for tanh layers
            std = np.sqrt(2.0 / (fan_in + fan_out))
            weights[i] = np.random.randn(fan_in, fan_out).astype(np.float32) * std
            biases[i] = np.random.randn(fan_out).astype(np.float32) * 0.01

    # Head layer initialization
    std = np.sqrt(2.0 / fan_in)  # He initialization for output
    head_w = np.random.randn(fan_in, 1).astype(np.float32) * std
    head_b = np.zeros((1,), dtype=np.float32)

    return weights, biases, head_w, head_b

def gradient(games, results, weights, biases, head_w, head_b):
    # global move_rewards
    results_arr = np.array(results, dtype=np.float32)
    result_vec = np.ones((len(results_arr))) * results_arr
    positions = []
    positions_arr = []
    all_evals = []
    all_z_lists = []
    all_s_lists = []
    head_w_tot = np.zeros_like(head_w)
    head_b_tot = np.zeros_like(head_b)
    w_grad_tot = np.zeros_like(weights)
    b_grad_tot = np.zeros_like(biases)


    weight_grad_l = np.zeros_like(weights)
    bias_grad_l = np.zeros_like(biases)

    for index, cur_game in enumerate(games):
        for i in cur_game:
            positions.append(i)
        positions_arr.append(np.array(positions))
        positions = []
    for i in range(len(positions_arr)):
        z_list, s_list, game_eval = eval(weights, biases, head_w, head_b, positions_arr[i])
        all_evals.append(game_eval)
        all_s_lists.append(s_list)
        all_z_lists.append(z_list)


    # Here swapped the MSE (minimal squares) with BCE.

    for i in range(len(positions_arr)):

        weight_grad_l = np.zeros_like(weights)
        bias_grad_l = np.zeros_like(biases)
        cur_evals = all_evals[i]
        results_for_current_iter = result_vec[i] * np.ones_like(cur_evals)

        # For the TD we assume we have one game. Therefore, the only element in result_vec is the first one.
        # So we edit it according to TD.
        for k in range(len(results_for_current_iter) - 1):
            results_for_current_iter[k] = sigmoid(all_evals[0][k + 1] * 0.99 + move_rewards[k])
        results_for_current_iter[-1] = result


        temporal_weights = np.power(0.8, np.arange(len(cur_evals) - 1, -1, -1))
        dL_du = (sigmoid(cur_evals) - results_for_current_iter) * temporal_weights.reshape(-1)# was - DlossDsum
        s_list = all_s_lists[i]
        z_list = all_z_lists[i]
        final_feats = s_list[-1]
        head_w_grad = final_feats.T @ dL_du.reshape(-1, 1)
        head_b_grad = np.sum(dL_du)

        delta_all = dL_du.reshape(-1, 1) * head_w.ravel()
        for j in range(len(cur_evals)):
            delta_cur = delta_all[j]

            for layer in reversed(range(6)):
                dz = delta_cur * derivatives[layer % 2](z_list[layer][j])  # scalar multiplication
                weight_grad_l[layer] += np.outer(s_list[layer][j], dz)
                bias_grad_l[layer] += dz
                delta_cur = weights[layer].T @ dz
        head_w_tot += head_w_grad
        head_b_tot += head_b_grad
        b_grad_tot += bias_grad_l
        w_grad_tot += weight_grad_l

    loss = -np.mean([np.mean(r * np.log(sigmoid(e) + 1e-8) +
                                 (1 - r) * np.log(1 - sigmoid(e) + 1e-8))
                         for r, e in zip(results_arr, all_evals)])
    return loss, head_w_tot, head_b_tot, w_grad_tot, b_grad_tot

def gradient_worker(args):
    games, results, weights, biases, head_w, head_b = args
    weights_copy = weights.copy()
    biases_copy = biases.copy()
    head_w_copy = head_w.copy()
    head_b_copy = head_b.copy()

    return gradient(games, results, weights_copy, biases_copy, head_w_copy, head_b_copy)



move_rewards = []
board = chess.Board()
turn = 1
"""
for j in range(2000):
    turn = 1
    while not board.is_game_over():
        value, best_move = minimax(2, turn, board)
        white_mat = sum(piece_vals_single[i] * int(board.pieces(i, True)).bit_count() for i in piece_vals_single)
        black_mat = sum(piece_vals_single[i] * int(board.pieces(i, False)).bit_count() for i in piece_vals_single)
        if random.randint(1, 10) == 3: best_move = list(board.legal_moves)[0]
        board.push(best_move)
        print(best_move.uci())
        turn = -turn  # Flip turn for the next player
        white_mat_after = sum(piece_vals_single[i] * int(board.pieces(i, True)).bit_count() for i in piece_vals_single)
        black_mat_after = sum(piece_vals_single[i] * int(board.pieces(i, False)).bit_count() for i in piece_vals_single)
        if board.turn == chess.WHITE:
            mat_diff = white_mat_after - white_mat
        else:
            mat_diff = black_mat_after - black_mat
        game.append(board.copy())
        move_rewards.append(mat_diff / 9)
    game_outcome_string = board.result()
    if game_outcome_string == "1-0":  # White wins
        result = 1.0
    elif game_outcome_string == "0-1":  # Black wins
        result = 0.0
    else:  # Draw "1/2-1/2"
        result = 0.2 # Set it to 0.2 in order to discourage white from drawing.
    lr = 0.01
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
    if j % 100 == 0:
        convert_cache.clear()
        del convert_cache
        convert_cache = {}
    if (j + 1) % 3 == 0:  # Save after every game (or change 1 to 10 to save every 10 games, etc.)
        save_model_parameters(weights, biases, head_w, head_b)
    board = chess.Board()
    game.clear()
    move_rewards.clear()
turn = 1
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')  # 'cumtime' is cumulative time
stats.print_stats(20)  # Show top 20 time consumers

while not board.is_game_over():
    value, best_move = minimax(1, turn, board)
    board.push(best_move)
    print("Values: ", value)
    print(f"Cache size: {len(convert_cache)}, Cache hits vs misses ratio")
    turn = -turn  # Flip turn for the next player
uci_string = " ".join(move.uci() for move in board.move_stack)
print(uci_string)
game.append(board.copy())
profiler.disable()
stats = pstats.Stats(profiler).sort_stats('cumtime')  # 'cumtime' is cumulative time
stats.print_stats(20)  # Show top 20 time consumers
"""

# Now train on auto-generated data. We will generate games where one side won. We would make the bold estimation that
# the last few moves on those games were probably bad for the losing side and good for the winning.
def basic_engine(board):
    moves = list(board.legal_moves)
    random.shuffle(moves)
    return random.choice(moves) if random.randint(1, 10) == 1 else max(moves, key=lambda m: move_score(board, m))

def make_game(n):
    game = []
    board = chess.Board()
    while not board.is_checkmate():
        board.push(basic_engine(board))
        game.append(convert_board(board))
        if len(board.move_stack) > 20:
            board = chess.Board()
            game = []
    return (game, 1 if board.turn == chess.BLACK else 0)
# Generate 10,000 games.


if __name__ == "__main__":
    loaded_weights, loaded_biases, head_w, head_b = load_model_parameters()
    if loaded_weights is not None and loaded_biases is not None:
        weights = loaded_weights
        biases = loaded_biases
    else:
        weights, biases, head_w, head_b = initialize_network_properly()
    """cores = cpu_count()
    with Pool(cores) as pool:
        gameList = pool.map(make_game, range(10000))
    save_games([game[0] for game in gameList], [game[1] for game in gameList])"""
    """games, results = load_games()
    gameList = [(games[i], results[i]) for i in range(len(games))]
    profiler = cProfile.Profile()
    profiler.enable()
    lr = 0.01
    for j in range(3000):
        for_now = random.sample(gameList, len(gameList) // 10)
        games, results = zip(*for_now)

        args_list = [(games[len(games)//cores_to_use * i: len(games)//cores_to_use * (i+1)],
                      results[len(games)//cores_to_use * i: len(games)//cores_to_use * (i+1)], weights, biases,
                      head_w, head_b) for i in range(cores_to_use)]
        with Pool(cores_to_use) as pool:
            gradients = pool.map(gradient_worker, args_list)
        head_w_tot = sum([hw for _, hw, _, _, _ in gradients])
        head_b_tot = sum([hb for _, _, hb, _, _ in gradients])
        w_grad_tot = sum([w for _, _, _, w, _ in gradients])
        b_grad_tot = sum([hb for _, _, _, _, hb in gradients])
        total_loss = np.mean([l for l, _, _, _, _ in gradients])
        # Clip head_w gradients
        clip_threshold = 1.0
        head_w_norm = np.sqrt(np.sum(head_w_tot ** 2))
        if head_w_norm > clip_threshold:
            head_w_tot = head_w_tot * (clip_threshold / head_w_norm)

        # Clip head_b gradients (scalar, so just simple clipping)
        head_b_tot = np.clip(head_b_tot, -clip_threshold, clip_threshold)

        # Clip weight gradients for each layer separately
        for i in range(len(w_grad_tot)):
            layer_norm = np.sqrt(np.sum(w_grad_tot[i] ** 2))
            if layer_norm > clip_threshold:
                w_grad_tot[i] = w_grad_tot[i] * (clip_threshold / layer_norm)

        # Clip bias gradients for each layer separately
        for i in range(len(b_grad_tot)):
            layer_norm = np.sqrt(np.sum(b_grad_tot[i] ** 2))
            if layer_norm > clip_threshold:
                b_grad_tot[i] = b_grad_tot[i] * (clip_threshold / layer_norm)
        print(f"total loss: {total_loss:.4f}")
        weights -= w_grad_tot * lr
        biases -= head_b_tot * lr
        head_w -= head_w_tot * lr
        head_b -= head_b_tot * lr
        print(f"{j} descent completed.")
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')  # 'cumtime' is cumulative time
    stats.print_stats(20)  # Show top 20 time consumers
    save_model_parameters(weights, biases, head_w, head_b)
    print(convert_board(chess.Board()))"""
    # Now play a game
    for j in range(3000):
        turn = 1
        while not board.is_game_over():
            value, best_move = minimax(2, turn, board)
            white_mat = sum(piece_vals_single[i] * int(board.pieces(i, True)).bit_count() for i in piece_vals_single)
            black_mat = sum(piece_vals_single[i] * int(board.pieces(i, False)).bit_count() for i in piece_vals_single)
            if random.randint(1, 10) == 3: best_move = list(board.legal_moves)[0]
            board.push(best_move)
            turn = -turn  # Flip turn for the next player
            white_mat_after = sum(
                piece_vals_single[i] * int(board.pieces(i, True)).bit_count() for i in piece_vals_single)
            black_mat_after = sum(
                piece_vals_single[i] * int(board.pieces(i, False)).bit_count() for i in piece_vals_single)
            if board.turn == chess.WHITE:
                mat_diff = white_mat_after - white_mat
            else:
                mat_diff = black_mat_after - black_mat
            game.append(convert_board(board.copy()))
            move_rewards.append(mat_diff / 9)
        game_outcome_string = board.result()
        if game_outcome_string == "1-0":  # White wins
            result = 1.0
        elif game_outcome_string == "0-1":  # Black wins
            result = 0.0
        else:  # Draw "1/2-1/2"
            result = 0.5
        lr = 0.01
        loss, head_w_grad, head_b_grad, weight_grad, bias_grad = gradient([np.array(game)], [result], weights, biases, head_w, head_b)
        print(f"loss: {loss:.4f}")
        weights -= weight_grad * lr / len(game)
        biases -= bias_grad * lr / len(game)
        head_w -= head_w_grad * lr / len(game)
        head_b -= head_b_grad * lr / len(game)
        print(result)
        uci_string = " ".join(move.uci() for move in board.move_stack)
        print(uci_string)
        del cache
        cache = {}
        if j % 100 == 0:
            convert_cache.clear()
            del convert_cache
            convert_cache = {}
        if (j + 1) % 3 == 0:  # Save after every game (or change 1 to 10 to save every 10 games, etc.)
            save_model_parameters(weights, biases, head_w, head_b)
        board = chess.Board()
        game.clear()
        move_rewards.clear()
    save_model_parameters(weights, biases, head_w, head_b)
    turn = 1
    while not board.is_game_over():
        value, best_move = minimax(2, turn, board)
        board.push(best_move)
        turn = -turn  # Flip turn for the next player
    uci_string = " ".join(move.uci() for move in board.move_stack)
    print(uci_string)
    game.append(board.copy())